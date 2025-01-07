#![no_std]
#![no_main]
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

extern crate alloc;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// pub mod backend;
// pub mod tensor;
// mod bridge;
// mod ops;

#[derive(Debug)]
pub enum Error {
    Argument,
    NoImpl,
    Failure,
}

pub trait StatusCode {
    fn check_status(self) -> Result<(), Error>;
}

impl StatusCode for arm_cmsis_nn_status {
    fn check_status(self) -> Result<(), Error> {
        match self {
            arm_cmsis_nn_status_ARM_CMSIS_NN_SUCCESS => Ok(()),
            arm_cmsis_nn_status_ARM_CMSIS_NN_ARG_ERROR => Err(Error::Argument),
            arm_cmsis_nn_status_ARM_CMSIS_NN_NO_IMPL_ERROR => Err(Error::NoImpl),
            _ => Err(Error::Failure),
        }
    }
}
