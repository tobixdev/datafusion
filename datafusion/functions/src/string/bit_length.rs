// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use arrow::compute::kernels::length::bit_length;
use arrow::datatypes::DataType;
use std::any::Any;

use crate::utils::utf8_to_int_type;
use datafusion_common::{exec_err, Result, ScalarValue};
use datafusion_expr::{ColumnarValue, Documentation, ScalarFunctionArgs, Volatility};
use datafusion_expr::{ScalarUDFImpl, Signature};
use datafusion_macros::user_doc;

#[user_doc(
    doc_section(label = "String Functions"),
    description = "Returns the bit length of a string.",
    syntax_example = "bit_length(str)",
    sql_example = r#"```sql
> select bit_length('datafusion');
+--------------------------------+
| bit_length(Utf8("datafusion")) |
+--------------------------------+
| 80                             |
+--------------------------------+
```"#,
    standard_argument(name = "str", prefix = "String"),
    related_udf(name = "length"),
    related_udf(name = "octet_length")
)]
#[derive(Debug)]
pub struct BitLengthFunc {
    signature: Signature,
}

impl Default for BitLengthFunc {
    fn default() -> Self {
        Self::new()
    }
}

impl BitLengthFunc {
    pub fn new() -> Self {
        Self {
            signature: Signature::string(1, Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for BitLengthFunc {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "bit_length"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        utf8_to_int_type(&arg_types[0], "bit_length")
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        if args.args.len() != 1 {
            return exec_err!(
                "bit_length function requires 1 argument, got {}",
                args.args.len()
            );
        }

        match &args.args[0] {
            ColumnarValue::Array(v) => Ok(ColumnarValue::Array(bit_length(v.as_ref())?)),
            ColumnarValue::Scalar(v) => {
                let string = v.try_as_str().flatten();
                match args.args_data_types[0] {
                    DataType::Utf8 | DataType::Utf8View => Ok(ColumnarValue::Scalar(
                        ScalarValue::Int32(string.map(|x| (x.len() * 8) as i32)),
                    )),
                    DataType::LargeUtf8 => Ok(ColumnarValue::Scalar(ScalarValue::Int64(
                        string.map(|x| (x.len() * 8) as i64),
                    ))),
                    _ => unreachable!("Unexpected argument type for bit_length"),
                }
            }
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}
