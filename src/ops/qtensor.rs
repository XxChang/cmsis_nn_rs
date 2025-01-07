use burn_ndarray::{NdArrayQTensor, NdArrayTensor};
use burn_tensor::{ops::QTensorOps, quantization::QuantizationStrategy, DType, TensorData};

use crate::{backend::{CmsisNN, CmsisNNDevice}, tensor::{CmsisNNQTensor, CmsisNNTensor}};

impl QTensorOps<Self> for CmsisNN {
    fn q_from_data(data: TensorData, _device: &CmsisNNDevice) -> burn_tensor::ops::QuantizedTensor<Self> {
        match data.dtype {
            DType::QFloat(strategy) => match strategy {
                QuantizationStrategy::PerTensorAffineInt8(_) => {
                    let data = data.convert::<i8>();
                    CmsisNNQTensor {
                        inner: NdArrayQTensor::<i8> {
                            qtensor: NdArrayTensor::<i8>::from_data(data),
                            scheme: strategy.scheme(),
                            strategy,
                        }
                    }
                }
                QuantizationStrategy::PerTensorSymmetricInt8(_) => {
                    unimplemented!()
                }
            },
            _ => panic!(
                "Invalid dtype (expected DType::QFloat, got {:?})",
                data.dtype
            )
        }
    }

    fn quantize(
            tensor: CmsisNNTensor<i8>,
            scheme: &burn_tensor::quantization::QuantizationScheme,
            qparams: burn_tensor::quantization::QuantizationParametersPrimitive<Self>,
        ) -> CmsisNNQTensor {
        todo!()
    }

    fn dequantize(tensor: CmsisNNQTensor) -> CmsisNNTensor<i8> {
        todo!()
    }

    fn q_shape(tensor: &CmsisNNQTensor) -> burn_tensor::Shape {
        tensor.shape()
    }

    fn q_device(_tensor: &CmsisNNQTensor) -> CmsisNNDevice {
        CmsisNNDevice::Mcu
    }

    fn q_to_device(tensor: burn_tensor::ops::QuantizedTensor<Self>, device: &burn_tensor::Device<Self>) -> burn_tensor::ops::QuantizedTensor<Self> {
        todo!()
    }

    fn q_reshape(tensor: CmsisNNQTensor, shape: burn_tensor::Shape) -> burn_tensor::ops::QuantizedTensor<Self> {
        todo!()
    }

    async fn q_into_data(tensor: burn_tensor::ops::QuantizedTensor<Self>) -> TensorData {
        todo!()
    }

    fn q_swap_dims(tensor: burn_tensor::ops::QuantizedTensor<Self>, dim1: usize, dim2: usize) -> burn_tensor::ops::QuantizedTensor<Self> {
        todo!()
    }

    fn q_permute(tensor: burn_tensor::ops::QuantizedTensor<Self>, axes: &[usize]) -> burn_tensor::ops::QuantizedTensor<Self> {
        todo!()
    }

    fn q_flip(tensor: burn_tensor::ops::QuantizedTensor<Self>, axes: &[usize]) -> burn_tensor::ops::QuantizedTensor<Self> {
        todo!()
    }

    fn q_gather(
            dim: usize,
            tensor: burn_tensor::ops::QuantizedTensor<Self>,
            indices: burn_tensor::ops::IntTensor<Self>,
        ) -> burn_tensor::ops::QuantizedTensor<Self> {
        todo!()
    }

    fn q_select(
            tensor: burn_tensor::ops::QuantizedTensor<Self>,
            dim: usize,
            indices: burn_tensor::ops::IntTensor<Self>,
        ) -> burn_tensor::ops::QuantizedTensor<Self> {
        todo!()
    }

    fn q_slice(tensor: burn_tensor::ops::QuantizedTensor<Self>, ranges: &[core::ops::Range<usize>]) -> burn_tensor::ops::QuantizedTensor<Self> {
        todo!()
    }

    fn q_argmax(tensor: burn_tensor::ops::QuantizedTensor<Self>, dim: usize) -> burn_tensor::ops::IntTensor<Self> {
        todo!()
    }

    fn q_argmin(tensor: burn_tensor::ops::QuantizedTensor<Self>, dim: usize) -> burn_tensor::ops::IntTensor<Self> {
        todo!()
    }

    fn q_expand(tensor: burn_tensor::ops::QuantizedTensor<Self>, shape: burn_tensor::Shape) -> burn_tensor::ops::QuantizedTensor<Self> {
        todo!()
    }
}
 