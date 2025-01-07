use burn_ndarray::NdArrayTensor;
use burn_tensor::ops::BoolTensorOps;

use crate::{backend::CmsisNN, tensor::CmsisNNTensor};

impl BoolTensorOps<Self> for CmsisNN {
    fn bool_from_data(data: burn_tensor::TensorData, device: &burn_tensor::Device<Self>) -> burn_tensor::ops::BoolTensor<Self> {
        CmsisNNTensor {
            inner: NdArrayTensor::from_data(data)
        }
    }
    
    fn bool_shape(tensor: &burn_tensor::ops::BoolTensor<Self>) -> burn_tensor::Shape {
        todo!()
    }

    async fn bool_into_data(tensor: burn_tensor::ops::BoolTensor<Self>) -> burn_tensor::TensorData {
        todo!()
    }

    fn bool_to_device(tensor: burn_tensor::ops::BoolTensor<Self>, device: &burn_tensor::Device<Self>) -> burn_tensor::ops::BoolTensor<Self> {
        todo!()
    }

    fn bool_reshape(tensor: burn_tensor::ops::BoolTensor<Self>, shape: burn_tensor::Shape) -> burn_tensor::ops::BoolTensor<Self> {
        todo!()
    }

    fn bool_slice(tensor: burn_tensor::ops::BoolTensor<Self>, ranges: &[core::ops::Range<usize>]) -> burn_tensor::ops::BoolTensor<Self> {
        todo!()
    }

    fn bool_into_int(tensor: burn_tensor::ops::BoolTensor<Self>) -> burn_tensor::ops::IntTensor<Self> {
        todo!()
    }

    fn bool_device(tensor: &burn_tensor::ops::BoolTensor<Self>) -> burn_tensor::Device<Self> {
        todo!()
    }

    fn bool_empty(shape: burn_tensor::Shape, device: &burn_tensor::Device<Self>) -> burn_tensor::ops::BoolTensor<Self> {
        todo!()
    }

    fn bool_slice_assign(
            tensor: burn_tensor::ops::BoolTensor<Self>,
            ranges: &[core::ops::Range<usize>],
            value: burn_tensor::ops::BoolTensor<Self>,
        ) -> burn_tensor::ops::BoolTensor<Self> {
        todo!()
    }

    fn bool_cat(tensors: alloc::vec::Vec<burn_tensor::ops::BoolTensor<Self>>, dim: usize) -> burn_tensor::ops::BoolTensor<Self> {
        todo!()
    }

    fn bool_equal(lhs: burn_tensor::ops::BoolTensor<Self>, rhs: burn_tensor::ops::BoolTensor<Self>) -> burn_tensor::ops::BoolTensor<Self> {
        todo!()
    }

    fn bool_not(tensor: burn_tensor::ops::BoolTensor<Self>) -> burn_tensor::ops::BoolTensor<Self> {
        todo!()
    }

    fn bool_into_float(tensor: burn_tensor::ops::BoolTensor<Self>) -> burn_tensor::ops::FloatTensor<Self> {
        todo!()
    }

    fn bool_swap_dims(tensor: burn_tensor::ops::BoolTensor<Self>, dim1: usize, dim2: usize) -> burn_tensor::ops::BoolTensor<Self> {
        todo!()
    }

    fn bool_permute(tensor: burn_tensor::ops::BoolTensor<Self>, axes: &[usize]) -> burn_tensor::ops::BoolTensor<Self> {
        todo!()
    }

    fn bool_expand(tensor: burn_tensor::ops::BoolTensor<Self>, shape: burn_tensor::Shape) -> burn_tensor::ops::BoolTensor<Self> {
        todo!()
    }

    fn bool_flip(tensor: burn_tensor::ops::BoolTensor<Self>, axes: &[usize]) -> burn_tensor::ops::BoolTensor<Self> {
        todo!()
    }
}