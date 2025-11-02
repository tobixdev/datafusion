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

use crate::types::{LogicalType, NativeType, TypeSignature, ValuePrettyPrinter};

/// Represents the canonical [UUID extension type](https://arrow.apache.org/docs/format/CanonicalExtensions.html#uuid).
pub struct UuidType;

impl UuidType {
    /// Creates a new [UuidType].
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for UuidType {
    fn default() -> Self {
        Self::new()
    }
}

impl LogicalType for UuidType {
    fn native(&self) -> &NativeType {
        &NativeType::FixedSizeBinary(16)
    }

    fn signature(&self) -> TypeSignature<'_> {
        TypeSignature::Extension {
            name: "arrow.uuid",
            parameters: &[],
        }
    }

    fn pretty_printer(&self) -> &dyn ValuePrettyPrinter {
        todo!("UUID printer not implemented")
    }
}

/// Represents an unknown extension type with a given native type and name.
///
/// TODO
pub struct UnknownExtensionType {
    /// The underlying native type.
    native_type: NativeType,
    /// The name of the underlying extension type.
    name: String,
}

impl UnknownExtensionType {
    /// Creates a new [UnknownExtensionType].
    pub fn new(name: String, native_type: NativeType) -> Self {
        Self { name, native_type }
    }
}

impl LogicalType for UnknownExtensionType {
    fn native(&self) -> &NativeType {
        &self.native_type
    }

    fn signature(&self) -> TypeSignature<'_> {
        TypeSignature::Extension {
            name: &self.name,
            parameters: &[],
        }
    }

    fn pretty_printer(&self) -> &dyn ValuePrettyPrinter {
        todo!()
    }
}
