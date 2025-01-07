#![no_std]
#![no_main]

#[path = "data/avgpooling.rs"]
mod avgpooling;

#[path = "utils/utils.rs"]
mod utils;

#[cfg(test)]
#[defmt_test::tests]
mod tests {
    use burn_tensor::{module::avg_pool2d, Shape, Tensor, TensorData};
    use cmsis_nn_rs::{
        backend::{CmsisNN, CmsisNNDevice},
        *,
    };
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

    use crate::{avgpooling::*, utils};

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
    fn test_avgpool_i8() {
        let input_data: Vec<i8> = Vec::from(AVGPOOLING_INPUT_TENSOR.as_slice());
        let shape_x = Shape::new([
            AVGPOOLING_BATCH_SIZE,
            AVGPOOLING_INPUT_C,
            AVGPOOLING_INPUT_H,
            AVGPOOLING_INPUT_W,
        ]);

        let data = TensorData::new(input_data, shape_x);

        let x = Tensor::<CmsisNN, 4>::from_data(data, &CmsisNNDevice::Mcu);

        let output = avg_pool2d(
            x,
            [AVGPOOLING_FILTER_H, AVGPOOLING_FILTER_W],
            [AVGPOOLING_STRIDE_H, AVGPOOLING_STRIDE_W],
            [AVGPOOLING_PADDING_H, AVGPOOLING_PADDING_W],
            true,
        );

        let _output = output.into_data();
    }

    #[test]
    fn test_avgpool_s8() {
        let mut output: [i8; AVGPOOLING_OUTPUT_W * AVGPOOLING_OUTPUT_H * AVGPOOLING_OUTPUT_C] =
            [0; AVGPOOLING_OUTPUT_W * AVGPOOLING_OUTPUT_H * AVGPOOLING_OUTPUT_C];

        let input_data: *const i8 = AVGPOOLING_INPUT_TENSOR.as_slice().as_ptr();

        let input_dims = cmsis_nn_dims {
            n: AVGPOOLING_BATCH_SIZE as i32,
            w: AVGPOOLING_INPUT_W as i32,
            h: AVGPOOLING_INPUT_H as i32,
            c: AVGPOOLING_INPUT_C as i32,
        };

        let filter_dims = cmsis_nn_dims {
            n: 0,
            w: AVGPOOLING_FILTER_W as i32,
            h: AVGPOOLING_FILTER_H as i32,
            c: 0,
        };

        let output_dims = cmsis_nn_dims {
            n: 0,
            w: AVGPOOLING_OUTPUT_W as i32,
            h: AVGPOOLING_OUTPUT_H as i32,
            c: AVGPOOLING_OUTPUT_C as i32,
        };

        let pool_params = cmsis_nn_pool_params {
            stride: cmsis_nn_tile {
                w: AVGPOOLING_STRIDE_W as i32,
                h: AVGPOOLING_STRIDE_H as i32,
            },
            padding: cmsis_nn_tile {
                w: AVGPOOLING_PADDING_W as i32,
                h: AVGPOOLING_PADDING_H as i32,
            },
            activation: cmsis_nn_activation {
                min: AVGPOOLING_ACTIVATION_MIN as i32,
                max: AVGPOOLING_ACTIVATION_MAX as i32,
            },
        };

        let ctx_size = unsafe {
            arm_avgpool_s8_get_buffer_size(AVGPOOLING_OUTPUT_W as i32, AVGPOOLING_INPUT_C as i32)
        };
        let mut ctx = cmsis_nn_context {
            size: ctx_size,
            buf: unsafe {
                HEAP.alloc(Layout::from_size_align(ctx_size as usize, 1).unwrap()) as *mut c_void
            },
        };

        let result = unsafe {
            arm_avgpool_s8(
                core::ptr::addr_of!(ctx),
                core::ptr::addr_of!(pool_params),
                core::ptr::addr_of!(input_dims),
                input_data,
                core::ptr::addr_of!(filter_dims),
                core::ptr::addr_of!(output_dims),
                output.as_mut_slice().as_mut_ptr(),
            )
            .check_status()
        };

        if !ctx.buf.is_null() {
            ctx.size = 0;
            unsafe {
                HEAP.dealloc(
                    ctx.buf as *mut u8,
                    Layout::from_size_align(ctx_size as usize, 1).unwrap(),
                );
            }
        };

        assert!(result.is_ok());
        assert!(utils::validate(
            output.as_mut_slice().as_mut_ptr(),
            AVGPOOLING_OUTPUT.as_slice().as_ptr(),
            AVGPOOLING_OUTPUT_W * AVGPOOLING_OUTPUT_H * AVGPOOLING_BATCH_SIZE * AVGPOOLING_OUTPUT_C
        ));
    }
}
