#![no_main]
#![no_std]

#[path = "./data.rs"]
mod data;

use cmsis_nn_rs::tflite_adapter::operator::OperatorOptions
;
use defmt_rtt as _;
use embedded_alloc::LlffHeap as Heap;
use nrf52833_hal as _;
use panic_probe as _;

#[global_allocator]
static HEAP: Heap = Heap::empty();

const MODEL: &'static [u8] = include_bytes!("mnist_forecast_model.tflite");

#[cortex_m_rt::entry]
fn main() -> ! {
    {
        use core::mem::MaybeUninit;
        const HEAP_SIZE: usize = 50 * 1024;
        static mut HEAP_MEM: [MaybeUninit<u8>; HEAP_SIZE] = [MaybeUninit::uninit(); HEAP_SIZE];
        unsafe {
            HEAP.init(HEAP_MEM.as_ptr() as usize, HEAP_SIZE);
        }
    }

    let image_data_9 = &data::IMG99_9.map(|x| (x as i32 - 128) as i8);
    let res = eval_model(image_data_9);
    defmt::info!("image_data_9 was recognized as: {}", res);

    let image_data_7 = &data::IMG0_7.map(|x| (x as i32 - 128) as i8);
    let res = eval_model(image_data_7);
    defmt::info!("image_data_7 was recognized as: {}", res);
    
    loop {}
}

fn eval_model(input: &[i8]) -> u8 {
    let mut output_data = [0; 4 * 26 * 26];
    let mut tmp_output = [0; 13 * 13 * 4];

    let model = cmsis_nn_rs::tflite_adapter::Model::get_model(MODEL).unwrap();

    let mut subgraph = model.get_subgraph(0).unwrap();
    subgraph.allocate_tensor();

    let conv1_params = if let Some(OperatorOptions::Conv2D(params)) = subgraph.get_operator(0) {
        params
    } else {
        unreachable!()
    };

    conv1_params
        .eval(input, &mut output_data)
        .unwrap();

    let pool1_params = if let Some(OperatorOptions::MaxPool(params)) = subgraph.get_operator(1)
    {
        params
    } else {
        unreachable!()
    };

    pool1_params
        .eval(
            &output_data[0..4 * 26 * 26],
            &mut tmp_output[0..13 * 13 * 4],
        )
        .unwrap();

    let conv2_params = if let Some(OperatorOptions::Conv2D(params)) = subgraph.get_operator(2) {
        params
    } else {
        unreachable!()
    };

    conv2_params
        .eval(
            &tmp_output[0..13 * 13 * 4],
            &mut output_data[0..11 * 11 * 8],
        )
        .unwrap();

        let pool2_params = if let Some(OperatorOptions::MaxPool(params)) = subgraph.get_operator(3)
        {
            params
        } else {
            unreachable!()
        };

        pool2_params
            .eval(&output_data[0..8 * 11 * 11], &mut tmp_output[0..5 * 5 * 8])
            .unwrap();

        let fully_connected =
            if let Some(OperatorOptions::FullyConnected(params)) = subgraph.get_operator(5) {
                params
            } else {
                unreachable!()
            };

        fully_connected
            .eval(&tmp_output[0..5 * 5 * 8], &mut output_data[0..10])
            .unwrap();

    if let Some((index, _v)) = output_data[0..10].iter().enumerate().max_by_key(|&(_, value)| value) {
        return index as u8;
    } else {
        panic!("No max value found");
    }
}