#![no_std]
#![no_main]
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

pub mod activation;
pub mod basic;
pub mod softmax;

#[allow(unused)]
mod private {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

#[derive(Debug)]
pub enum Error {
    Argument,
    NoImpl,
    Failure,
}

type Result<T> = core::result::Result<T, Error>;

pub trait StatusCode {
    fn check_status(self) -> Result<()>;
}

impl StatusCode for private::arm_cmsis_nn_status {
    fn check_status(self) -> Result<()> {
        match self {
            private::arm_cmsis_nn_status_ARM_CMSIS_NN_SUCCESS => Ok(()),
            private::arm_cmsis_nn_status_ARM_CMSIS_NN_ARG_ERROR => Err(Error::Argument),
            private::arm_cmsis_nn_status_ARM_CMSIS_NN_NO_IMPL_ERROR => Err(Error::NoImpl),
            _ => Err(Error::Failure),
        }
    }
}

#[macro_export]
macro_rules! test_length {
    ($input1:ident, $input2:ident) => {
        if $input1.len() == $input2.len() {
            Ok(())
        } else {
            Err(Error::Argument)
        }
    };
    ($input:ident, $len:expr) => {
        if $input.len() == $len as usize {
            Ok(())
        } else {
            Err(Error::Argument)
        }
    };
    ($input1:ident, $input2:ident, $output:ident) => {
        if $input1.len() == $input2.len() && $input1.len() == $output.len() {
            Ok(())
        } else {
            Err(Error::Argument)
        }
    };
    () => {};
}

#[cfg(test)]
mod test_utils {
    pub fn validate(act: *const i8, ref_data: *const i8, size: usize) -> bool {
        let mut test_passed = true;
        let mut count = 0;
        let mut total = 0;

        for i in 0..size {
            total += 1;
            let act_data = unsafe { *act.offset(i as _) };
            let ref_data = unsafe { *ref_data.offset(i as _) };
            if act_data != ref_data {
                defmt::error!("ERROR at pos {}: Act: {} Ref: {}", i, act_data, ref_data);
                count += 1;
                test_passed = false;
            }
        }

        if !test_passed {
            defmt::error!("{} of {} failed", count, total);
        }

        test_passed
    }
}

#[cfg(test)]
#[defmt_test::tests]
mod tests {
    use defmt_rtt as _;
    use nrf52833_hal as _;
    use panic_probe as _;

    #[test]
    fn test_elementwise_add_s8() {
        crate::basic::tests::test_elementwise_add_s8();
    }

    #[test]
    fn test_softmax_s8() {
        crate::softmax::tests::test_softmax_s8();
    }
}
