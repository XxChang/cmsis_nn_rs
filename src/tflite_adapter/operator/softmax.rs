use core::any::TypeId;

use super::alloc::vec::Vec;
use crate::{softmax::softmax_s8, tflite_adapter::schema_generated};

pub struct SoftMaxParams {
    num_rows: i32,
    row_size: i32,
    input_multipliter: i32,
    input_left_shift: i32,
    diff_min: i32,
}

impl defmt::Format for SoftMaxParams {
    fn format(&self, fmt: defmt::Formatter) {
        defmt::write!(fmt, "SoftMaxParams {{ num_rows: {}, ", self.num_rows);
        defmt::write!(fmt, "row_size: {}, ", self.row_size);
        defmt::write!(fmt, "input_multipliter: {}, ", self.input_multipliter);
        defmt::write!(fmt, "input_left_shift: {}, ", self.input_left_shift);
        defmt::write!(fmt, "diff_min: {} }}", self.diff_min);
    }
}

impl SoftMaxParams {
    const ScaledDiffIntegerBits: i32 = 5;

    pub fn new(
        op: &schema_generated::tflite::Operator<'_>,
        subgraph: &crate::tflite_adapter::subgraph::SubGraph,
    ) -> Option<Self> {
        if let Some(schema_params) = op.builtin_options_as_softmax_options() {
            let beta = schema_params.beta();

            let inputs_array: Vec<i32> = op.inputs().iter().flatten().map(|x| x).collect();
            let outputs_array: Vec<i32> = op.outputs().iter().flatten().map(|x| x).collect();

            let input_index = inputs_array[0];
            let output_index = outputs_array[0];

            let tensors = subgraph.tensors();
            let input = &tensors[input_index as usize];
            let output = &tensors[output_index as usize];

            assert!(
                input.data_typ() == TypeId::of::<i8>(),
                "Only support i8 at now"
            );

            assert_eq!(
                input.data_typ(),
                output.data_typ(),
                "input and output data type must be the same",
            );

            let output_zero_point = output.zero_points()[0];
            let output_scale = output.scales()[0];

            assert_eq!(output_zero_point, -128);
            assert_eq!(output_scale, 1.0f32 / 256.0);

            let trailing_dim = input.dims().len() - 1;
            for (index, (i_s, o_s)) in input.dims().iter().zip(output.dims()).enumerate() {
                if index == trailing_dim {
                    continue;
                }
                assert_eq!(i_s, o_s);
            }
            let outer_size = super::flat_size_skip_dim(input.dims(), trailing_dim);

            assert_eq!(input.dims()[trailing_dim], output.dims()[trailing_dim]);
            let depth = input.dims()[trailing_dim];
            let (quanized_multiplier, left_shift) = Self::preprocess_softmax_scaling(
                beta as f64,
                input.scales()[0] as f64,
                Self::ScaledDiffIntegerBits,
            );

            let diff_min =
                -1 * Self::calculate_input_radius(Self::ScaledDiffIntegerBits, left_shift);

            Some(SoftMaxParams {
                num_rows: outer_size,
                row_size: depth,

                input_left_shift: left_shift,
                input_multipliter: quanized_multiplier,
                diff_min,
            })
        } else {
            None
        }
    }

    pub fn eval(self, input_data: &[i8], output: &mut [i8]) -> Result<(), crate::Error> {
        softmax_s8(
            input_data,
            self.num_rows,
            self.row_size,
            self.input_multipliter,
            self.input_left_shift,
            self.diff_min,
            output,
        )
    }

    fn preprocess_softmax_scaling(
        beta: f64,
        input_scale: f64,
        input_integer_bits: i32,
    ) -> (i32, i32) {
        let max_real_multiplier = (1u64 << 31) as f64 - 1.0;

        let input_beta_real_multiplier =
            (beta * input_scale * ((1u64 << (31 - input_integer_bits)) as f64))
                .min(max_real_multiplier);

        assert!(input_beta_real_multiplier > 1.0);
        let (quantized_multiplier, left_shift) =
            super::quantize_multiplier(input_beta_real_multiplier);

        (quantized_multiplier, left_shift)
    }

    fn calculate_input_radius(
        input_integer_bits: i32,
        input_left_shift: i32,
        // total_signed_bits: i32,
    ) -> i32 {
        defmt::debug!("input_left_shift: {}", input_left_shift);
        let total_signed_bits = 31;
        let max_input_rescaled = 1.0
            * (((1 << input_integer_bits) - 1) as f64)
            * ((1u64 << (total_signed_bits - input_integer_bits)) as f64)
            / ((1u64 << input_left_shift) as f64);

        super::libm::floor(max_input_rescaled) as i32
    }
}
