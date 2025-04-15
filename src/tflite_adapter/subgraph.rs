use super::alloc;

use super::operator::OperatorOptions;
use super::schema_generated;
use super::tensor::TensorInt32;
use super::tensor::TensorInt8;
use super::traits::TensorOp;
use alloc::boxed::Box;
use alloc::vec::Vec;

pub struct SubGraph<'a> {
    pub(super) internal: schema_generated::tflite::SubGraph<'a>,
    internal_model: &'a schema_generated::tflite::Model<'a>,
    tensors: Vec<Box<dyn TensorOp>>,
}

impl<'b: 'a, 'a> SubGraph<'a> {
    pub(crate) fn new(
        subgraph: schema_generated::tflite::SubGraph<'a>,
        model: &'b super::Model,
    ) -> Self {
        SubGraph {
            internal: subgraph,
            internal_model: &model.internal,
            tensors: Vec::new(),
        }
    }

    pub fn name(&self) -> Option<&str> {
        self.internal.name()
    }

    pub fn allocate_tensor(&mut self) {
        let buffer = self.internal_model.buffers();

        for tensor in self.internal.tensors().iter().flatten() {
            let mut data = Vec::new();
            if let Some(ref buffer_inner) = buffer {
                if let Some(data_tmp) = buffer_inner.get(tensor.buffer() as usize).data() {
                    data.extend_from_slice(data_tmp.bytes());
                }
            }

            match tensor.type_().0 {
                x if x == schema_generated::tflite::TensorType::INT8.0 => {
                    self.tensors
                        .push(Box::new(TensorInt8::create_frome_tflite(&tensor, &data)));
                }
                x if x == schema_generated::tflite::TensorType::INT32.0 => {
                    self.tensors
                        .push(Box::new(TensorInt32::create_frome_tflite(&tensor, &data)));
                }
                x => {
                    panic!("unsupport tensor type {x}")
                }
            }
        }
    }

    pub fn tensors(&self) -> &[Box<dyn TensorOp>] {
        &self.tensors
    }

    pub fn operator_size(&self) -> usize {
        if let Some(op) = self.internal.operators() {
            op.len()
        } else {
            0
        }
    }

    pub fn get_operator(&self, op_index: usize) -> Option<OperatorOptions> {
        if let Some(ops) = self.internal.operators() {
            let operator_codes = self.internal_model.operator_codes().unwrap();

            let op = ops.get(op_index);
            let opcode_index = op.opcode_index();

            let opcode = operator_codes.get(opcode_index as usize);

            match opcode.builtin_code().0 {
                x if x == schema_generated::tflite::BuiltinOperator::CONV_2D.0 => {
                    let params = super::operator::conv2d::Conv2DParams::new(&op, self).unwrap();
                    defmt::info!("Conv2D");
                    Some(OperatorOptions::Conv2D(params))
                }
                x if x == schema_generated::tflite::BuiltinOperator::MAX_POOL_2D.0 => {
                    let params =
                        super::operator::max_pooling::MaxPoolParams::new(&op, self).unwrap();
                    defmt::info!("MaxPool2D");
                    Some(OperatorOptions::MaxPool(params))
                }
                x if x == schema_generated::tflite::BuiltinOperator::RESHAPE.0 => {
                    defmt::info!("Reshape");
                    None
                }
                x if x == schema_generated::tflite::BuiltinOperator::RELU.0 => {
                    defmt::info!("Relu");
                    None
                }
                x if x == schema_generated::tflite::BuiltinOperator::FULLY_CONNECTED.0 => {
                    defmt::info!("FullyConnected");
                    let params =
                        super::operator::fully_connect::FullyConnectedParams::new(&op, self)
                            .unwrap();
                    Some(OperatorOptions::FullyConnected(params))
                }
                x if x == schema_generated::tflite::BuiltinOperator::SOFTMAX.0 => {
                    defmt::info!("Softmax");
                    let params = super::operator::softmax::SoftMaxParams::new(&op, self).unwrap();
                    Some(OperatorOptions::Softmax(params))
                }
                x => {
                    panic!("Unsupport opcode {x}")
                }
            }
        } else {
            None
        }
    }
}
