use burn_ndarray::{NdArrayQTensor, NdArrayTensor};
use burn_tensor::{
    quantization::{QTensorPrimitive, QuantizationScheme, QuantizationStrategy},
    Element, Shape
};

/// Tensor primitive used by the [ndarray backend](crate::NdArray).
#[derive(Debug, Clone)]
pub struct CmsisNNTensor<E: Element> {
    /// Dynamic array that contains the data of type E.
    pub inner: NdArrayTensor<E>,
}

#[derive(Clone, Debug)]
pub struct CmsisNNQTensor {
    pub inner: NdArrayQTensor<i8>,
}

impl QTensorPrimitive for CmsisNNQTensor {
    fn scheme(&self) -> &QuantizationScheme {
        &self.inner.scheme()
    }

    fn strategy(&self) -> QuantizationStrategy {
        self.inner.strategy()
    }
}

impl CmsisNNQTensor {
    pub(crate) fn shape(&self) -> Shape {
        Shape::from(self.inner.qtensor.array.shape().to_vec())
    }
}

impl<E: Element> CmsisNNTensor<E> {
    pub(crate) fn shape(&self) -> Shape {
        Shape::from(self.inner.array.shape().to_vec())
    }
}




