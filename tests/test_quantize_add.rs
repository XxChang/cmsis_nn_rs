#![no_std]
#![no_main]

#[cfg(test)]
#[defmt_test::tests]
mod tests {
    use burn_tensor::TensorData;
    use cmsis_nn_rs::*;
    use core::{
        alloc::{GlobalAlloc, Layout},
        ffi::c_void,
    };
    use defmt_rtt as _;
    use embedded_alloc::TlsfHeap as Heap;
    use nrf52833_hal as _;
    use panic_probe as _;
    extern crate alloc;
    use alloc::vec::Vec;

    #[global_allocator]
    static HEAP: Heap = Heap::empty();

    #[init]
    fn init() {
        use core::mem::MaybeUninit;
        const HEAP_SIZE: usize = 100 * 1024;
        static mut HEAP_MEM: [MaybeUninit<u8>; HEAP_SIZE] = [MaybeUninit::uninit(); HEAP_SIZE];
        unsafe { HEAP.init(HEAP_MEM.as_ptr() as usize, HEAP_SIZE) }
    }

    #[test]
    fn test_quantize_add_d2() {}
}
