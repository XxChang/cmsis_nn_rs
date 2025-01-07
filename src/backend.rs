
use alloc::string::String;
use burn_tensor::backend::{Backend, DeviceId, DeviceOps};

use crate::{bridge::PrecisionBridge, tensor::{CmsisNNQTensor, CmsisNNTensor}};

/// The device type for the ndarray backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmsisNNDevice {
    /// The CPU device.
    Mcu,
    Dsp,
    Mve,
}

impl DeviceOps for CmsisNNDevice {
    fn id(&self) -> burn_tensor::backend::DeviceId {
        match self {
            CmsisNNDevice::Mcu => DeviceId::new(0, 0),
            CmsisNNDevice::Dsp => DeviceId::new(0, 1),
            CmsisNNDevice::Mve => DeviceId::new(0, 2),
        }
    }
}

impl Default for CmsisNNDevice {
    fn default() -> Self {
        Self::Mcu
    }
}

/// Tensor backend that uses the [CMSIS-NN](cmsis-nn) crate for executing tensor operations.
///
#[derive(Clone, Copy, Default, Debug)]
pub struct CmsisNN;

// impl Backend for CmsisNN {
//     type Device = CmsisNNDevice;
//     type FullPrecisionBridge = PrecisionBridge;

//     type FloatTensorPrimitive = CmsisNNTensor<i8>;
//     type FloatElem = i8;

//     type IntTensorPrimitive = CmsisNNTensor<i64>;
//     type IntElem = i64;

//     type BoolTensorPrimitive = CmsisNNTensor<bool>;

//     type QuantizedTensorPrimitive = CmsisNNQTensor;
//     type QuantizedEncoding = i8;

//     fn name() -> alloc::string::String {
//         String::from("CMSIS-NN")
//     }

//     fn ad_enabled() -> bool {
//         false
//     }

//     fn seed(_seed: u64) {
//         todo!()
//     }
// }

