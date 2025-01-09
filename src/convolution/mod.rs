use crate::private::{cmsis_nn_activation, cmsis_nn_conv_params, cmsis_nn_tile};

pub struct Config(cmsis_nn_conv_params);

impl Config {
    pub fn new(
        input_offset: i32,
        output_offset: i32,
        stride: (i32, i32),
        padding: (i32, i32),
        dilation: (i32, i32),
        range: (i32, i32),
    ) -> Config {
        Config(cmsis_nn_conv_params {
            input_offset,
            output_offset,
            stride: cmsis_nn_tile {
                w: stride.0,
                h: stride.1,
            },
            padding: cmsis_nn_tile {
                w: padding.0,
                h: padding.1,
            },
            dilation: cmsis_nn_tile {
                w: dilation.0,
                h: dilation.1,
            },
            activation: cmsis_nn_activation {
                min: range.0,
                max: range.1,
            },
        })
    }
}

impl AsRef<cmsis_nn_conv_params> for Config {
    fn as_ref(&self) -> &cmsis_nn_conv_params {
        &self.0
    }
}
