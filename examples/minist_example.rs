#![no_main]
#![no_std]

extern crate alloc;

use defmt_rtt as _;
use embedded_graphics::pixelcolor::Gray8;
use nrf52833_hal as _;
use panic_probe as _;
use tinybmp::Bmp;

const CONV1_IM_DIM: i32 = 28;
const CONV1_IM_CH: i32 = 1;
const CONV1_KERNEL_DIM: i32 = 3;
const CONV1_PADDING: i32 = 0;
const CONV1_STRIDE: i32 = 1;
const CONV1_OUT_CH: i32 = 4;
const CONV1_OUT_DIM: i32 = 26;

#[cortex_m_rt::entry]
fn main() -> ! {
    let bmp_data = include_bytes!("../data/1.bmp");

    let bmp = Bmp::<Gray8>::from_slice(bmp_data).unwrap();

    loop {}
}
