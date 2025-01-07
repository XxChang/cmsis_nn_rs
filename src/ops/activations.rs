use burn_tensor::{ops::ActivationOps, ElementConversion};

use crate::{backend::CmsisNN, tensor::CmsisNNTensor};

impl ActivationOps<Self> for CmsisNN {
    fn relu(tensor: CmsisNNTensor<i8>) -> CmsisNNTensor<i8> {
        todo!()
    }
}
