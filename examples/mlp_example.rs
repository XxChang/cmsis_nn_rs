#![no_main]
#![no_std]

extern crate alloc;

use defmt_rtt as _;
use nrf52833_hal as _;
use panic_probe as _;

#[cortex_m_rt::entry]
fn main() -> ! {
    loop {}
}
