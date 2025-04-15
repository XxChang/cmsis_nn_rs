#[allow(non_snake_case)]
#[allow(unused_imports)]
#[path = "../../target/flatbuffers/schema_generated.rs"]
pub mod schema_generated;

extern crate alloc;

// pub mod conv2d_options;
pub mod operator;

pub mod subgraph;
use subgraph::SubGraph;

mod tensor;
pub mod traits;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum AdapterError {
    #[error("miss required field")]
    MissingRequiredField,
    #[error("invalud flatbuffer")]
    InvalidFlatbuffer,
}

pub struct Model<'a> {
    internal: schema_generated::tflite::Model<'a>,
}

impl<'a: 'b, 'b> Model<'a> {
    pub fn version(&self) -> u32 {
        self.internal.version()
    }

    pub fn description(&self) -> Option<&str> {
        self.internal.description()
    }

    pub fn get_subgraph(&'a self, subgraph_index: usize) -> Option<SubGraph<'b>> {
        if let Some(subgraph) = self.internal.subgraphs() {
            let subgraph = subgraph.get(subgraph_index);
            Some(SubGraph::new(subgraph, self))
        } else {
            None
        }
    }

    pub fn get_model(model: &'static [u8]) -> Result<Model<'a>, AdapterError> {
        let model = schema_generated::tflite::root_as_model(model)
            .map_err(|_| AdapterError::InvalidFlatbuffer)?;
        Ok(Model { internal: model })
    }
}

// pub struct Operator<'a> {
//     internal: schema_generated::tflite::Operator<'a>,
// }

// impl Operator<'_> {
//     pub fn opcode_index(&self) -> u32 {
//         self.internal.opcode_index()
//     }

//     pub fn input_size(&self) -> usize {
//         if let Some(inputs) = self.internal.inputs() {
//             inputs.len()
//         } else {
//             0
//         }
//     }

//     pub fn input_tensor_index(&self) -> Option<i32> {
//         self.internal.inputs().map(|x| x.get(0))
//     }

//     pub fn weight_tensor_index(&self) -> Option<i32> {
//         self.internal.inputs().map(|x| x.get(1))
//     }

//     pub fn bias_tensor_index(&self) -> Option<i32> {
//         self.internal.inputs().map(|x| x.get(2))
//     }

//     pub fn output_tensor_index(&self) -> Option<i32> {
//         self.internal.outputs().map(|x| x.get(0))
//     }
// }

// pub struct OperatorSlice<'a> {
//     internal: Option<flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<schema_generated::tflite::Operator<'a>>>>,
//     len: usize,
// }

// impl OperatorSlice<'_> {
//     pub fn len(&self) -> usize {
//         self.len
//     }

//     pub fn get(&self, index: usize) -> Operator {
//         Operator { internal: self.internal.unwrap().get(index) }
//     }
// }

// pub struct Subgraph<'a> {
//     internal: schema_generated::tflite::SubGraph<'a>,
// }

// pub struct Tensor<'a> {
//     internal: schema_generated::tflite::Tensor<'a>,
// }

// impl Tensor<'_> {
//     pub fn shape(&self) -> Vec<i32> {
//         let mut shape = Vec::new();
//         for i in self.internal.shape().iter().flat_map(|x| x.iter()) {
//             shape.push(i);
//         }
//         shape
//     }

//     pub fn typ(&self) {
//         self.internal.type_();
//     }
// }

// pub struct TensorSlice<'a> {
//     internal: Option<flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<schema_generated::tflite::Tensor<'a>>>>,
//     len: usize,
// }

// impl TensorSlice<'_> {
//     pub fn len(&self) -> usize {
//         self.len
//     }

//     pub fn get(&self, index: usize) -> Tensor {
//         Tensor { internal: self.internal.unwrap().get(index) }
//     }
// }

// impl<'a> Subgraph<'a> {
//     pub fn operators(&self) -> OperatorSlice<'a> {
//         let operators = self.internal.operators();
//         if operators.is_none() {
//             return OperatorSlice { internal: None, len: 0 };
//         } else {
//             let len = operators.unwrap().len();
//             OperatorSlice { internal: operators, len }
//         }
//     }

//     pub fn tensors(&self) -> TensorSlice<'a> {
//         let tensors = self.internal.tensors();
//         if tensors.is_none() {
//             return TensorSlice { internal: None, len: 0 };
//         } else {
//             let len = tensors.unwrap().len();
//             TensorSlice { internal: tensors, len }
//         }
//     }
// }

// pub struct SubgraphSlice<'a> {
//     internal: Option<flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<schema_generated::tflite::SubGraph<'a>>>>,
//     len: usize,
// }

// impl SubgraphSlice<'_> {
//     pub fn len(&self) -> usize {
//         self.len
//     }

//     pub fn get(&self, index: usize) -> Subgraph {
//         Subgraph { internal: self.internal.unwrap().get(index) }
//     }
// }

// pub struct OperatorCode<'a> {
//     internal: schema_generated::tflite::OperatorCode<'a>,
// }

// impl OperatorCode<'_> {
//     pub fn variant_name(&self) -> Option<&'static str> {
//         self.internal.builtin_code().variant_name()
//     }
// }

// pub struct OperatorCodeSlice<'a> {
//     internal: Option<flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<schema_generated::tflite::OperatorCode<'a>>>>,
//     len: usize,
// }

// impl OperatorCodeSlice<'_> {
//     pub fn len(&self) -> usize {
//         self.len
//     }

//     pub fn get(&self, index: usize) -> OperatorCode {
//         OperatorCode { internal: self.internal.unwrap().get(index) }
//     }
// }

// impl<'a> ModelAdapter<'a> {
//     pub fn version(&self) -> u32 {
//         self.internal.version()
//     }

//     pub fn getModel(model: &'static [u8]) -> Result<ModelAdapter<'a>, AdapterError> {
//         let model = schema_generated::tflite::root_as_model(model).map_err(|_| AdapterError::InvalidFlatbuffer)?;
//         Ok(ModelAdapter { internal: model })
//     }

//     pub fn subgraphs(&self) -> SubgraphSlice<'a> {
//         let subgraph = self.internal.subgraphs();
//         if subgraph.is_none() {
//             return SubgraphSlice { internal: None, len: 0 };
//         } else {
//             let len = subgraph.unwrap().len();
//             SubgraphSlice { internal: subgraph, len }
//         }
//     }

//     pub fn input_size(&self) -> Result<usize, AdapterError> {
//         Ok(self.internal.subgraphs()
//             .ok_or(AdapterError::MissingRequiredField)?.get(0)
//             .inputs().ok_or(AdapterError::MissingRequiredField)?.len())
//     }

//     pub fn operator_codes(&self) -> OperatorCodeSlice<'a> {
//         let operator_codes = self.internal.operator_codes();
//         if operator_codes.is_none() {
//             return OperatorCodeSlice { internal: None, len: 0 };
//         } else {
//             let len = operator_codes.unwrap().len();
//             OperatorCodeSlice { internal: operator_codes, len }
//         }
//     }
// }
