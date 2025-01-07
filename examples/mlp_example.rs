#![no_main]
#![no_std]

#[path = "minst/mod.rs"]
mod minst;

extern crate alloc;

use defmt_rtt as _;
use nrf52833_hal as _;
use panic_probe as _;

use burn::tensor::backend::Backend;
use embedded_alloc::LlffHeap as Heap;
use minst::{
    mlp::MlpConfig,
    model::{MnistConfig, Model},
};

#[global_allocator]
static HEAP: Heap = Heap::empty();

#[cortex_m_rt::entry]
fn main() -> ! {
    use core::mem::MaybeUninit;
    const HEAP_SIZE: usize = 100 * 1024;
    static mut HEAP_MEM: [MaybeUninit<u8>; HEAP_SIZE] = [MaybeUninit::uninit(); HEAP_SIZE];
    unsafe { HEAP.init(HEAP_MEM.as_ptr() as usize, HEAP_SIZE) }

    type Backend = cmsis_nn_rs::backend::CmsisNN;

    let device = Default::default();
    let mlp_config = MlpConfig::new();
    let minst_config = MnistConfig::new(mlp_config);
    let minst_model: Model<Backend> = Model::new(&minst_config, &device);

    loop {}
}
