use core::any::TypeId;
extern crate libm;

use super::{alloc, schema_generated, traits::TensorOp};
use alloc::boxed::Box;
use micromath::F32Ext;

pub mod conv2d;
pub mod fully_connect;
pub mod max_pooling;
pub mod softmax;

pub enum OperatorOptions<'a> {
    Conv2D(conv2d::Conv2DParams<'a>),
    MaxPool(max_pooling::MaxPoolParams),
    FullyConnected(fully_connect::FullyConnectedParams<'a>),
    Softmax(softmax::SoftMaxParams),
}

fn calculate_activation_range_quantized(
    output: &Box<dyn TensorOp>,
    activation_typ: i8,
) -> (i32, i32) {
    let qmin;
    let qmax;
    if output.data_typ() == TypeId::of::<i8>() {
        qmin = i8::MIN as i32;
        qmax = i8::MAX as i32;
    } else if output.data_typ() == TypeId::of::<i32>() {
        qmin = i32::MIN;
        qmax = i32::MAX;
    } else {
        unreachable!()
    }

    let scale = output.scales()[0];
    let zero_point = output.zero_points()[0];

    let act_min;
    let act_max;

    if activation_typ == schema_generated::tflite::ActivationFunctionType::RELU.0 {
        let tmp_q = quantize(scale, zero_point, 0.0);

        act_min = qmin.max(tmp_q);
        act_max = qmax;
    } else {
        act_min = qmin;
        act_max = qmax;
    }

    (act_min, act_max)
}

fn quantize(scale: f32, zero_point: i32, f: f32) -> i32 {
    let tmp = (f / scale).round();
    let no_integer_overflow_from_quantization = tmp >= i32::MIN as f32 && tmp <= i32::MAX as f32;
    assert!(no_integer_overflow_from_quantization);
    zero_point + tmp as i32
}

fn compute_padding(
    stride: i32,
    dilation_rate: i32,
    in_size: i32,
    filter_size: i32,
    out_size: i32,
) -> i32 {
    let effective_filter_size = (filter_size - 1) * dilation_rate + 1;
    let total_padding = (out_size - 1) * stride + effective_filter_size - in_size;
    let total_padding = if total_padding > 0 { total_padding } else { 0 };
    // *offset = total_padding % 2;
    total_padding / 2
}

fn quantize_multiplier(multiplier: f64) -> (i32, i32) {
    if multiplier == 0.0 {
        return (0, 0);
    }

    let (q, mut shift) = libm::frexp(multiplier);
    let mut q_fixed = libm::round(q * ((1u64 << 31) as f64)) as i64;

    assert!(q_fixed <= ((1u64) << 31) as i64);
    if q_fixed == ((1u64) << 31) as i64 {
        q_fixed /= 2;
        shift += 1;
    }

    assert!(q_fixed < i32::MAX as i64);

    if shift < -31 {
        shift = 0;
        q_fixed = 0;
    }

    let q_fixed = q_fixed as i32;
    assert!(q_fixed < i32::MAX);
    (q_fixed, shift)
}

fn flat_size_skip_dim(shape: &[i32], skip_dim: usize) -> i32 {
    shape
        .iter()
        .enumerate()
        .fold(1, |acc, (i, &x)| if i != skip_dim { acc * x } else { acc })
}
