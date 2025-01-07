// Language
use alloc::vec;
use alloc::vec::Vec;
use burn_ndarray::NdArrayTensor;
use burn_tensor::ops::IntTensorOps;
use burn_tensor::Distribution;

use burn_tensor::ElementConversion;
use core::ops::Range;
use ndarray::IntoDimension;
use ndarray::Zip;

// Current crate

// Workspace crates
use burn_tensor::{backend::Backend, Shape, TensorData};

use crate::backend::CmsisNN;
use crate::backend::CmsisNNDevice;
use crate::tensor::CmsisNNTensor;

impl IntTensorOps<Self> for CmsisNN {
    fn int_from_data(data: TensorData, _device: &CmsisNNDevice) -> CmsisNNTensor<i64> {
        CmsisNNTensor::<i64> {
            inner: NdArrayTensor::from_data(data)
        }
    }

    fn int_shape(tensor: &CmsisNNTensor<i64>) -> Shape {
        todo!()
    }

    async fn int_into_data(tensor: CmsisNNTensor<i64>) -> TensorData {
        todo!()
    }

    fn int_to_device(tensor: CmsisNNTensor<i64>, _device: &CmsisNNDevice) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_reshape(tensor: CmsisNNTensor<i64>, shape: Shape) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_slice(tensor: CmsisNNTensor<i64>, ranges: &[Range<usize>]) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_device(_tensor: &CmsisNNTensor<i64>) -> CmsisNNDevice {
        todo!()
    }

    fn int_empty(shape: Shape, _device: &CmsisNNDevice) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_mask_where(
        tensor: CmsisNNTensor<i64>,
        mask: CmsisNNTensor<bool>,
        source: CmsisNNTensor<i64>,
    ) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_mask_fill(
        tensor: CmsisNNTensor<i64>,
        mask: CmsisNNTensor<bool>,
        value: i64,
    ) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_slice_assign(
        tensor: CmsisNNTensor<i64>,
        ranges: &[Range<usize>],
        value: CmsisNNTensor<i64>,
    ) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_cat(tensors: Vec<CmsisNNTensor<i64>>, dim: usize) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_equal(lhs: CmsisNNTensor<i64>, rhs: CmsisNNTensor<i64>) -> CmsisNNTensor<bool> {
        todo!()
    }

    fn int_equal_elem(lhs: CmsisNNTensor<i64>, rhs: i64) -> CmsisNNTensor<bool> {
        todo!()
    }

    fn int_greater(lhs: CmsisNNTensor<i64>, rhs: CmsisNNTensor<i64>) -> CmsisNNTensor<bool> {
        todo!()
    }

    fn int_greater_elem(lhs: CmsisNNTensor<i64>, rhs: i64) -> CmsisNNTensor<bool> {
        todo!()
    }

    fn int_greater_equal(lhs: CmsisNNTensor<i64>, rhs: CmsisNNTensor<i64>) -> CmsisNNTensor<bool> {
        todo!()
    }

    fn int_greater_equal_elem(lhs: CmsisNNTensor<i64>, rhs: i64) -> CmsisNNTensor<bool> {
        todo!()
    }

    fn int_lower(lhs: CmsisNNTensor<i64>, rhs: CmsisNNTensor<i64>) -> CmsisNNTensor<bool> {
        todo!()
    }

    fn int_lower_elem(lhs: CmsisNNTensor<i64>, rhs: i64) -> CmsisNNTensor<bool> {
        todo!()
    }

    fn int_lower_equal(lhs: CmsisNNTensor<i64>, rhs: CmsisNNTensor<i64>) -> CmsisNNTensor<bool> {
        todo!()
    }

    fn int_lower_equal_elem(lhs: CmsisNNTensor<i64>, rhs: i64) -> CmsisNNTensor<bool> {
        todo!()
    }

    fn int_add(lhs: CmsisNNTensor<i64>, rhs: CmsisNNTensor<i64>) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_add_scalar(lhs: CmsisNNTensor<i64>, rhs: i64) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_sub(lhs: CmsisNNTensor<i64>, rhs: CmsisNNTensor<i64>) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_sub_scalar(lhs: CmsisNNTensor<i64>, rhs: i64) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_mul(lhs: CmsisNNTensor<i64>, rhs: CmsisNNTensor<i64>) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_mul_scalar(lhs: CmsisNNTensor<i64>, rhs: i64) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_div(lhs: CmsisNNTensor<i64>, rhs: CmsisNNTensor<i64>) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_div_scalar(lhs: CmsisNNTensor<i64>, rhs: i64) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_remainder_scalar(lhs: CmsisNNTensor<i64>, rhs: i64) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_neg(tensor: CmsisNNTensor<i64>) -> CmsisNNTensor<i64> {
        Self::int_mul_scalar(tensor, -1)
    }

    fn int_zeros(shape: Shape, device: &CmsisNNDevice) -> CmsisNNTensor<i64> {
        Self::int_from_data(TensorData::zeros::<i64, _>(shape), device)
    }

    fn int_ones(shape: Shape, device: &CmsisNNDevice) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_full(
        shape: Shape,
        fill_value: i64,
        device: &CmsisNNDevice,
    ) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_sum(tensor: CmsisNNTensor<i64>) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_sum_dim(tensor: CmsisNNTensor<i64>, dim: usize) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_prod(tensor: CmsisNNTensor<i64>) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_prod_dim(tensor: CmsisNNTensor<i64>, dim: usize) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_mean(tensor: CmsisNNTensor<i64>) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_mean_dim(tensor: CmsisNNTensor<i64>, dim: usize) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_gather(
        dim: usize,
        tensor: CmsisNNTensor<i64>,
        indices: CmsisNNTensor<i64>,
    ) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_scatter(
        dim: usize,
        tensor: CmsisNNTensor<i64>,
        indices: CmsisNNTensor<i64>,
        value: CmsisNNTensor<i64>,
    ) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_select(
        tensor: CmsisNNTensor<i64>,
        dim: usize,
        indices: CmsisNNTensor<i64>,
    ) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_select_assign(
        tensor: CmsisNNTensor<i64>,
        dim: usize,
        indices: CmsisNNTensor<i64>,
        value: CmsisNNTensor<i64>,
    ) -> CmsisNNTensor<i64> {
        todo!()
    }
    fn int_argmax(tensor: CmsisNNTensor<i64>, dim: usize) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_argmin(tensor: CmsisNNTensor<i64>, dim: usize) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_clamp_min(tensor: CmsisNNTensor<i64>, min: i64) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_clamp_max(tensor: CmsisNNTensor<i64>, max: i64) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_clamp(tensor: CmsisNNTensor<i64>, min: i64, max: i64) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_abs(tensor: CmsisNNTensor<i64>) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_into_float(tensor: CmsisNNTensor<i64>) -> CmsisNNTensor<i8> {
        todo!()
    }

    fn int_swap_dims(tensor: CmsisNNTensor<i64>, dim1: usize, dim2: usize) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_random(
        shape: Shape,
        distribution: Distribution,
        device: &CmsisNNDevice,
    ) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_powi(lhs: CmsisNNTensor<i64>, rhs: CmsisNNTensor<i64>) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_powf(lhs: CmsisNNTensor<i64>, rhs: CmsisNNTensor<i8>) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_powf_scalar(lhs: CmsisNNTensor<i64>, rhs: f32) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_permute(tensor: CmsisNNTensor<i64>, axes: &[usize]) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_flip(tensor: CmsisNNTensor<i64>, axes: &[usize]) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_sign(tensor: CmsisNNTensor<i64>) -> CmsisNNTensor<i64> {
        todo!()
    }

    fn int_expand(tensor: CmsisNNTensor<i64>, shape: Shape) -> CmsisNNTensor<i64> {
        todo!()
    }
}
