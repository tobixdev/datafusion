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

use crate::error::_internal_err;
use crate::types::extension::DFExtensionType;
use arrow::array::{Array, Int8Array};
use arrow::datatypes::DataType;
use arrow::util::display::{ArrayFormatter, DisplayIndex, FormatOptions, FormatResult};
use std::fmt::Write;

/// Defines the extension type logic for the canonical `arrow.uuid` extension type.
///
/// See [`DFExtensionType`] for information on DataFusion's extension type mechanism.
impl DFExtensionType for arrow_schema::extension::Bool8 {
    fn storage_type(&self) -> DataType {
        DataType::Int8
    }

    fn serialize_metadata(&self) -> Option<String> {
        None
    }

    fn create_array_formatter<'fmt>(
        &self,
        array: &'fmt dyn Array,
        options: &FormatOptions<'fmt>,
    ) -> crate::Result<Option<ArrayFormatter<'fmt>>> {
        if array.data_type() != &DataType::Int8 {
            return _internal_err!("Wrong array type for Bool8");
        }

        let display_index = Bool8ValueDisplayIndex {
            array: array.as_any().downcast_ref().unwrap(),
            null_str: options.null(),
        };
        Ok(Some(ArrayFormatter::new(
            Box::new(display_index),
            options.safe(),
        )))
    }
}

/// Pretty printer for binary UUID values.
#[derive(Debug, Clone, Copy)]
struct Bool8ValueDisplayIndex<'a> {
    array: &'a Int8Array,
    null_str: &'a str,
}

impl DisplayIndex for Bool8ValueDisplayIndex<'_> {
    fn write(&self, idx: usize, f: &mut dyn Write) -> FormatResult {
        if self.array.is_null(idx) {
            write!(f, "{}", self.null_str)?;
            return Ok(());
        }

        let bytes = self.array.value(idx);
        write!(f, "{}", bytes != 0)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_pretty_bool8() {
        let values = Int8Array::from_iter([Some(0), Some(1), Some(-20), None]);

        let extension_type = arrow_schema::extension::Bool8 {};
        let formatter = extension_type
            .create_array_formatter(&values, &FormatOptions::default().with_null("NULL"))
            .unwrap()
            .unwrap();

        assert_eq!(formatter.value(0).to_string(), "false");
        assert_eq!(formatter.value(1).to_string(), "true");
        assert_eq!(formatter.value(2).to_string(), "true");
        assert_eq!(formatter.value(3).to_string(), "NULL");
    }
}
