use core::{alloc::Layout, ffi::c_void};

use alloc::vec::Vec;
use burn_ndarray::NdArrayTensor;
use burn_tensor::{ops::*, ElementConversion};
use ndarray::Array4;

use crate::{arm_avgpool_s8, arm_avgpool_s8_get_buffer_size, backend::CmsisNN, cmsis_nn_activation, cmsis_nn_context, cmsis_nn_dims, cmsis_nn_pool_params, cmsis_nn_tile, tensor::CmsisNNTensor, StatusCode};

impl ModuleOps<Self> for CmsisNN {
    fn conv2d(
        x: CmsisNNTensor<i8>,
        weight: CmsisNNTensor<i8>,
        bias: Option<CmsisNNTensor<i8>>,
        options: ConvOptions<2>,
    ) -> CmsisNNTensor<i8> {
        todo!()
    }

    fn deform_conv2d(
        x: CmsisNNTensor<i8>,
        offset: CmsisNNTensor<i8>,
        weight: CmsisNNTensor<i8>,
        mask: Option<CmsisNNTensor<i8>>,
        bias: Option<CmsisNNTensor<i8>>,
        options: DeformConvOptions<2>,
    ) -> CmsisNNTensor<i8> {
        todo!()
    }

    fn deform_conv2d_backward(
        x: CmsisNNTensor<i8>,
        offset: CmsisNNTensor<i8>,
        weight: CmsisNNTensor<i8>,
        mask: Option<CmsisNNTensor<i8>>,
        bias: Option<CmsisNNTensor<i8>>,
        output_grad: CmsisNNTensor<i8>,
        options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Self> {
        todo!()
    }

    fn conv_transpose2d(
        x: CmsisNNTensor<i8>,
        weight: CmsisNNTensor<i8>,
        bias: Option<CmsisNNTensor<i8>>,
        options: ConvTransposeOptions<2>,
    ) -> CmsisNNTensor<i8> {
        todo!()
    }

    fn avg_pool2d(
        x: CmsisNNTensor<i8>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        _count_include_pad: bool,
    ) -> CmsisNNTensor<i8> {
        let [kernel_height, kernel_width] = kernel_size;
        let [padding_height, padding_width] = padding;
        let [stride_height, stride_width] = stride;

        let input_data = x.inner.array.as_slice().unwrap().as_ptr();

        let [batch_size, channels, x_height, x_width] = x.shape().dims();

        let out_height = (x_height + 2 * padding_height - kernel_height) / stride_height + 1;
        let out_width = (x_width + 2 * padding_width - kernel_width) / stride_width + 1;

        let input_dims = cmsis_nn_dims {
            n: batch_size as i32,
            w: x_width as i32,
            h: x_height as i32,
            c: channels as i32,
        };

        let filter_dims = cmsis_nn_dims {
            n: 0,
            w: kernel_width as i32,
            h: kernel_height as i32,
            c: 0,
        };

        let output_dims = cmsis_nn_dims {
            n: 0,
            w: out_width as i32,
            h: out_height as i32,
            c: channels as i32,
        };

        let pool_params = cmsis_nn_pool_params {
            stride: cmsis_nn_tile {
                w: stride_width as i32,
                h: stride_height as i32,
            },
            padding: cmsis_nn_tile {
                w: padding_width as i32,
                h: padding_height as i32,
            },
            activation: cmsis_nn_activation {
                min: i8::MIN as i32,
                max: i8::MAX as i32,
            },
        };

        let ctx_size = unsafe { 
            arm_avgpool_s8_get_buffer_size(
                out_width as i32,
                channels as i32) 
        };
        let mut ctx = cmsis_nn_context {
            size: ctx_size,
            buf: unsafe { alloc::alloc::alloc(
                Layout::from_size_align(ctx_size as usize, 1).unwrap()
            ) as *mut c_void }
        };

        let mut output_data = Array4::from_elem((batch_size, channels, out_height, out_width), 0.elem());

        unsafe {
            arm_avgpool_s8(
                core::ptr::addr_of!(ctx), 
                core::ptr::addr_of!(pool_params), 
                core::ptr::addr_of!(input_dims), 
                input_data, 
                core::ptr::addr_of!(filter_dims), 
                core::ptr::addr_of!(output_dims), 
                output_data.as_slice_mut().unwrap().as_mut_ptr(),
            ).check_status()
        }.unwrap();

        if !ctx.buf.is_null() {
            ctx.size = 0;
            unsafe {
                alloc::alloc::dealloc(ctx.buf as *mut u8, 
                    Layout::from_size_align(ctx_size as usize, 1).unwrap()
                );
            }
        };

        CmsisNNTensor {
            inner: NdArrayTensor::new(output_data.into_dyn().into_shared())
        }
    }

    fn avg_pool2d_backward(
        x: CmsisNNTensor<i8>,
        grad: CmsisNNTensor<i8>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> CmsisNNTensor<i8> {
        todo!()
    }

    fn max_pool2d(
        x: CmsisNNTensor<i8>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> CmsisNNTensor<i8> {
        todo!()
    }

    fn max_pool2d_with_indices(
        x: CmsisNNTensor<i8>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> MaxPool2dWithIndices<CmsisNN> {
        todo!() 
    }

    fn max_pool2d_with_indices_backward(
        x: CmsisNNTensor<i8>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        output_grad: CmsisNNTensor<i8>,
        indices: CmsisNNTensor<i64>,
    ) -> MaxPool2dBackward<CmsisNN> {
       todo!()
    }

    fn adaptive_avg_pool2d(x: CmsisNNTensor<i8>, output_size: [usize; 2]) -> CmsisNNTensor<i8> {
        todo!()
    }

    fn adaptive_avg_pool2d_backward(
        x: CmsisNNTensor<i8>,
        grad: CmsisNNTensor<i8>,
    ) -> CmsisNNTensor<i8> {
        todo!()
    }

    fn interpolate(
        x: CmsisNNTensor<i8>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> CmsisNNTensor<i8> {
        todo!()
    }

    fn interpolate_backward(
        x: CmsisNNTensor<i8>,
        grad: CmsisNNTensor<i8>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> CmsisNNTensor<i8> {
        todo!()
    }

    fn conv3d(
        x: CmsisNNTensor<i8>,
        weight: CmsisNNTensor<i8>,
        bias: Option<CmsisNNTensor<i8>>,
        options: ConvOptions<3>,
    ) -> CmsisNNTensor<i8> {
        todo!()
    }

    fn conv_transpose3d(
        x: CmsisNNTensor<i8>,
        weight: CmsisNNTensor<i8>,
        bias: Option<CmsisNNTensor<i8>>,
        options: ConvTransposeOptions<3>,
    ) -> CmsisNNTensor<i8> {
        todo!()
    }
}
