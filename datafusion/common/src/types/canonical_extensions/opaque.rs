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

use crate::types::extension::DFExtensionType;
use arrow::datatypes::DataType;
use arrow_schema::ArrowError;
use arrow_schema::extension::{ExtensionType, Opaque};

/// Defines the extension type logic for the canonical `arrow.opaque` extension type.
///
/// See [`DFExtensionType`] for information on DataFusion's extension type mechanism.
#[derive(Debug, Clone)]
pub struct DFOpaque {
    inner: Opaque,
    storage_type: DataType,
}

impl ExtensionType for DFOpaque {
    const NAME: &'static str = Opaque::NAME;
    type Metadata = <Opaque as ExtensionType>::Metadata;

    fn metadata(&self) -> &Self::Metadata {
        self.inner.metadata()
    }

    fn serialize_metadata(&self) -> Option<String> {
        self.inner.serialize_metadata()
    }

    fn deserialize_metadata(
        metadata: Option<&str>,
    ) -> Result<Self::Metadata, ArrowError> {
        Opaque::deserialize_metadata(metadata)
    }

    fn supports_data_type(&self, data_type: &DataType) -> Result<(), ArrowError> {
        self.inner.supports_data_type(data_type)
    }

    fn try_new(
        data_type: &DataType,
        metadata: Self::Metadata,
    ) -> Result<Self, ArrowError> {
        Ok(Self {
            inner: <Opaque as ExtensionType>::try_new(data_type, metadata)?,
            storage_type: data_type.clone(),
        })
    }
}

impl DFExtensionType for DFOpaque {
    fn storage_type(&self) -> DataType {
        self.storage_type.clone()
    }

    fn serialize_metadata(&self) -> Option<String> {
        self.inner.serialize_metadata()
    }
}
