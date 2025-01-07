use burn_tensor::backend::BackendBridge;

use crate::backend::CmsisNN;

#[derive(Debug)]
pub struct PrecisionBridge;

impl BackendBridge<CmsisNN> for PrecisionBridge {
    type Target = CmsisNN;
    
    fn from_target(
            tensor: burn_tensor::ops::FloatTensor<Self::Target>,
            device: Option<burn_tensor::Device<CmsisNN>>,
        ) -> burn_tensor::ops::FloatTensor<CmsisNN> {
        todo!()
    }

    fn into_target(
            tensor: burn_tensor::ops::FloatTensor<CmsisNN>,
            device: Option<burn_tensor::Device<Self::Target>>,
        ) -> burn_tensor::ops::FloatTensor<Self::Target> {
        todo!()
    }
}