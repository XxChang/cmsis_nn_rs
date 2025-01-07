use burn_ndarray::NdArrayTensor;
use burn_tensor::{ops::FloatTensorOps, TensorData};

use crate::{backend::{CmsisNN, CmsisNNDevice}, tensor::CmsisNNTensor};

impl FloatTensorOps<Self> for CmsisNN {
    fn float_from_data(data: burn_tensor::TensorData, device: &burn_tensor::Device<Self>) -> CmsisNNTensor<i8> {
        CmsisNNTensor::<i8> {
            inner: NdArrayTensor::<i8>::from_data(data)
        }
    }

    fn float_random(shape: burn_tensor::Shape, distribution: burn_tensor::Distribution, device: &burn_tensor::Device<Self>)
            -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_shape(tensor: &burn_tensor::ops::FloatTensor<Self>) -> burn_tensor::Shape {
        todo!()
    }

    async fn float_into_data(tensor: burn_tensor::ops::FloatTensor<Self>) -> burn_tensor::TensorData {
        let shape = tensor.shape();
        let values = tensor.inner.array.into_iter().collect();
        TensorData::new(values, shape)
    }

    fn float_device(tensor: &burn_tensor::ops::FloatTensor<Self>) -> burn_tensor::Device<Self> {
        CmsisNNDevice::Mcu
    }

    fn float_to_device(tensor: burn_tensor::ops::FloatTensor<Self>, device: &burn_tensor::Device<Self>) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_empty(shape: burn_tensor::Shape, device: &burn_tensor::Device<Self>) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_add(lhs: burn_tensor::ops::FloatTensor<Self>, rhs: burn_tensor::ops::FloatTensor<Self>) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_add_scalar(lhs: burn_tensor::ops::FloatTensor<Self>, rhs: burn_tensor::ops::FloatElem<Self>) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_sub(lhs: burn_tensor::ops::FloatTensor<Self>, rhs: burn_tensor::ops::FloatTensor<Self>) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_sub_scalar(lhs: burn_tensor::ops::FloatTensor<Self>, rhs: burn_tensor::ops::FloatElem<Self>) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_mul(lhs: burn_tensor::ops::FloatTensor<Self>, rhs: burn_tensor::ops::FloatTensor<Self>) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_mul_scalar(lhs: burn_tensor::ops::FloatTensor<Self>, rhs: burn_tensor::ops::FloatElem<Self>) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_div(lhs: burn_tensor::ops::FloatTensor<Self>, rhs: burn_tensor::ops::FloatTensor<Self>) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_div_scalar(lhs: burn_tensor::ops::FloatTensor<Self>, rhs: burn_tensor::ops::FloatElem<Self>) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_remainder_scalar(lhs: burn_tensor::ops::FloatTensor<Self>, rhs: burn_tensor::ops::FloatElem<Self>) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_matmul(lhs: burn_tensor::ops::FloatTensor<Self>, rhs: burn_tensor::ops::FloatTensor<Self>) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_neg(tensor: burn_tensor::ops::FloatTensor<Self>) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_recip(tensor: burn_tensor::ops::FloatTensor<Self>) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_swap_dims(tensor: burn_tensor::ops::FloatTensor<Self>, dim1: usize, dim2: usize) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_reshape(tensor: burn_tensor::ops::FloatTensor<Self>, shape: burn_tensor::Shape) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_gather(dim: usize, tensor: burn_tensor::ops::FloatTensor<Self>, indices: burn_tensor::ops::IntTensor<Self>) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_scatter(
            dim: usize,
            tensor: burn_tensor::ops::FloatTensor<Self>,
            indices: burn_tensor::ops::IntTensor<Self>,
            value: burn_tensor::ops::FloatTensor<Self>,
        ) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_select(tensor: burn_tensor::ops::FloatTensor<Self>, dim: usize, indices: burn_tensor::ops::IntTensor<Self>) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_select_assign(
            tensor: burn_tensor::ops::FloatTensor<Self>,
            dim: usize,
            indices: burn_tensor::ops::IntTensor<Self>,
            value: burn_tensor::ops::FloatTensor<Self>,
        ) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_slice(tensor: burn_tensor::ops::FloatTensor<Self>, ranges: &[core::ops::Range<usize>]) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_slice_assign(
            tensor: burn_tensor::ops::FloatTensor<Self>,
            ranges: &[core::ops::Range<usize>],
            value: burn_tensor::ops::FloatTensor<Self>,
        ) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_mask_where(
            tensor: burn_tensor::ops::FloatTensor<Self>,
            mask: burn_tensor::ops::BoolTensor<Self>,
            value: burn_tensor::ops::FloatTensor<Self>,
        ) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_mask_fill(
            tensor: burn_tensor::ops::FloatTensor<Self>,
            mask: burn_tensor::ops::BoolTensor<Self>,
            value: burn_tensor::ops::FloatElem<Self>,
        ) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_equal(lhs: burn_tensor::ops::FloatTensor<Self>, rhs: burn_tensor::ops::FloatTensor<Self>) -> burn_tensor::ops::BoolTensor<Self> {
        todo!()
    }

    fn float_equal_elem(lhs: burn_tensor::ops::FloatTensor<Self>, rhs: burn_tensor::ops::FloatElem<Self>) -> burn_tensor::ops::BoolTensor<Self> {
        todo!()
    }

    fn float_greater(lhs: burn_tensor::ops::FloatTensor<Self>, rhs: burn_tensor::ops::FloatTensor<Self>) -> burn_tensor::ops::BoolTensor<Self> {
        todo!()
    }

    fn float_greater_elem(lhs: burn_tensor::ops::FloatTensor<Self>, rhs: burn_tensor::ops::FloatElem<Self>) -> burn_tensor::ops::BoolTensor<Self> {
        todo!()
    }

    fn float_greater_equal(lhs: burn_tensor::ops::FloatTensor<Self>, rhs: burn_tensor::ops::FloatTensor<Self>) -> burn_tensor::ops::BoolTensor<Self> {
        todo!()
    }

    fn float_greater_equal_elem(lhs: burn_tensor::ops::FloatTensor<Self>, rhs: burn_tensor::ops::FloatElem<Self>) -> burn_tensor::ops::BoolTensor<Self> {
        todo!()
    }

    fn float_lower(lhs: burn_tensor::ops::FloatTensor<Self>, rhs: burn_tensor::ops::FloatTensor<Self>) -> burn_tensor::ops::BoolTensor<Self> {
        todo!()
    }

    fn float_lower_elem(lhs: burn_tensor::ops::FloatTensor<Self>, rhs: burn_tensor::ops::FloatElem<Self>) -> burn_tensor::ops::BoolTensor<Self> {
        todo!()
    }

    fn float_lower_equal(lhs: burn_tensor::ops::FloatTensor<Self>, rhs: burn_tensor::ops::FloatTensor<Self>) -> burn_tensor::ops::BoolTensor<Self> {
        todo!()
    }

    fn float_lower_equal_elem(lhs: burn_tensor::ops::FloatTensor<Self>, rhs: burn_tensor::ops::FloatElem<Self>) -> burn_tensor::ops::BoolTensor<Self> {
        todo!()
    }

    fn float_detach(tensor: burn_tensor::ops::FloatTensor<Self>) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_mean(tensor: burn_tensor::ops::FloatTensor<Self>) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_sum(tensor: burn_tensor::ops::FloatTensor<Self>) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_mean_dim(tensor: burn_tensor::ops::FloatTensor<Self>, dim: usize) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_sum_dim(tensor: burn_tensor::ops::FloatTensor<Self>, dim: usize) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_argmax(tensor: burn_tensor::ops::FloatTensor<Self>, dim: usize) -> burn_tensor::ops::IntTensor<Self> {
        todo!()
    }

    fn float_argmin(tensor: burn_tensor::ops::FloatTensor<Self>, dim: usize) -> burn_tensor::ops::IntTensor<Self> {
        todo!()
    }

    fn float_exp(tensor: burn_tensor::ops::FloatTensor<Self>) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_log(tensor: burn_tensor::ops::FloatTensor<Self>) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_log1p(tensor: burn_tensor::ops::FloatTensor<Self>) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_powf_scalar(tensor: burn_tensor::ops::FloatTensor<Self>, value: f32) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_sqrt(tensor: burn_tensor::ops::FloatTensor<Self>) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_abs(tensor: burn_tensor::ops::FloatTensor<Self>) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_cos(tensor: burn_tensor::ops::FloatTensor<Self>) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_sin(tensor: burn_tensor::ops::FloatTensor<Self>) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_tanh(tensor: burn_tensor::ops::FloatTensor<Self>) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_erf(tensor: burn_tensor::ops::FloatTensor<Self>) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_cat(tensors: alloc::vec::Vec<burn_tensor::ops::FloatTensor<Self>>, dim: usize) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_clamp_min(tensor: burn_tensor::ops::FloatTensor<Self>, min: burn_tensor::ops::FloatElem<Self>) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_clamp_max(tensor: burn_tensor::ops::FloatTensor<Self>, max: burn_tensor::ops::FloatElem<Self>) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_clamp(tensor: burn_tensor::ops::FloatTensor<Self>, min: burn_tensor::ops::FloatElem<Self>, max: burn_tensor::ops::FloatElem<Self>) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_into_int(tensor: burn_tensor::ops::FloatTensor<Self>) -> burn_tensor::ops::IntTensor<Self> {
        todo!()
    }

    fn float_powf(lhs: burn_tensor::ops::FloatTensor<Self>, rhs: burn_tensor::ops::FloatTensor<Self>) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_permute(tensor: burn_tensor::ops::FloatTensor<Self>, axes: &[usize]) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_flip(tensor: burn_tensor::ops::FloatTensor<Self>, axes: &[usize]) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_sign(tensor: burn_tensor::ops::FloatTensor<Self>) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn float_expand(tensor: burn_tensor::ops::FloatTensor<Self>, shape: burn_tensor::Shape) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }
}