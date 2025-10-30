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

//! FunctionRegistry trait

use crate::expr_rewriter::FunctionRewrite;
use crate::planner::ExprPlanner;
use crate::{AggregateUDF, ScalarUDF, UserDefinedLogicalNode, WindowUDF};
use datafusion_common::types::{LogicalTypeRef, TypeSignature};
use datafusion_common::{
    internal_err, not_impl_err, plan_datafusion_err, HashMap, Result,
};
use std::collections::HashSet;
use std::fmt::Debug;
use std::sync::Arc;

/// A registry knows how to build logical expressions out of user-defined function' names
pub trait FunctionRegistry {
    /// Returns names of all available scalar user defined functions.
    fn udfs(&self) -> HashSet<String>;

    /// Returns names of all available aggregate user defined functions.
    fn udafs(&self) -> HashSet<String>;

    /// Returns names of all available window user defined functions.
    fn udwfs(&self) -> HashSet<String>;

    /// Returns a reference to the user defined scalar function (udf) named
    /// `name`.
    fn udf(&self, name: &str) -> Result<Arc<ScalarUDF>>;

    /// Returns a reference to the user defined aggregate function (udaf) named
    /// `name`.
    fn udaf(&self, name: &str) -> Result<Arc<AggregateUDF>>;

    /// Returns a reference to the user defined window function (udwf) named
    /// `name`.
    fn udwf(&self, name: &str) -> Result<Arc<WindowUDF>>;

    /// Registers a new [`ScalarUDF`], returning any previously registered
    /// implementation.
    ///
    /// Returns an error (the default) if the function can not be registered,
    /// for example if the registry is read only.
    fn register_udf(&mut self, _udf: Arc<ScalarUDF>) -> Result<Option<Arc<ScalarUDF>>> {
        not_impl_err!("Registering ScalarUDF")
    }
    /// Registers a new [`AggregateUDF`], returning any previously registered
    /// implementation.
    ///
    /// Returns an error (the default) if the function can not be registered,
    /// for example if the registry is read only.
    fn register_udaf(
        &mut self,
        _udaf: Arc<AggregateUDF>,
    ) -> Result<Option<Arc<AggregateUDF>>> {
        not_impl_err!("Registering AggregateUDF")
    }
    /// Registers a new [`WindowUDF`], returning any previously registered
    /// implementation.
    ///
    /// Returns an error (the default) if the function can not be registered,
    /// for example if the registry is read only.
    fn register_udwf(&mut self, _udaf: Arc<WindowUDF>) -> Result<Option<Arc<WindowUDF>>> {
        not_impl_err!("Registering WindowUDF")
    }

    /// Deregisters a [`ScalarUDF`], returning the implementation that was
    /// deregistered.
    ///
    /// Returns an error (the default) if the function can not be deregistered,
    /// for example if the registry is read only.
    fn deregister_udf(&mut self, _name: &str) -> Result<Option<Arc<ScalarUDF>>> {
        not_impl_err!("Deregistering ScalarUDF")
    }

    /// Deregisters a [`AggregateUDF`], returning the implementation that was
    /// deregistered.
    ///
    /// Returns an error (the default) if the function can not be deregistered,
    /// for example if the registry is read only.
    fn deregister_udaf(&mut self, _name: &str) -> Result<Option<Arc<AggregateUDF>>> {
        not_impl_err!("Deregistering AggregateUDF")
    }

    /// Deregisters a [`WindowUDF`], returning the implementation that was
    /// deregistered.
    ///
    /// Returns an error (the default) if the function can not be deregistered,
    /// for example if the registry is read only.
    fn deregister_udwf(&mut self, _name: &str) -> Result<Option<Arc<WindowUDF>>> {
        not_impl_err!("Deregistering WindowUDF")
    }

    /// Registers a new [`FunctionRewrite`] with the registry.
    ///
    /// `FunctionRewrite` rules are used to rewrite certain / operators in the
    /// logical plan to function calls.  For example `a || b` might be written to
    /// `array_concat(a, b)`.
    ///
    /// This allows the behavior of operators to be customized by the user.
    fn register_function_rewrite(
        &mut self,
        _rewrite: Arc<dyn FunctionRewrite + Send + Sync>,
    ) -> Result<()> {
        not_impl_err!("Registering FunctionRewrite")
    }

    /// Set of all registered [`ExprPlanner`]s
    fn expr_planners(&self) -> Vec<Arc<dyn ExprPlanner>>;

    /// Registers a new [`ExprPlanner`] with the registry.
    fn register_expr_planner(
        &mut self,
        _expr_planner: Arc<dyn ExprPlanner>,
    ) -> Result<()> {
        not_impl_err!("Registering ExprPlanner")
    }
}

/// Serializer and deserializer registry for extensions like [UserDefinedLogicalNode].
pub trait SerializerRegistry: Debug + Send + Sync {
    /// Serialize this node to a byte array. This serialization should not include
    /// input plans.
    fn serialize_logical_plan(
        &self,
        node: &dyn UserDefinedLogicalNode,
    ) -> Result<Vec<u8>>;

    /// Deserialize user defined logical plan node ([UserDefinedLogicalNode]) from
    /// bytes.
    fn deserialize_logical_plan(
        &self,
        name: &str,
        bytes: &[u8],
    ) -> Result<Arc<dyn UserDefinedLogicalNode>>;
}

/// A  [`FunctionRegistry`] that uses in memory [`HashMap`]s
#[derive(Default, Debug)]
pub struct MemoryFunctionRegistry {
    /// Scalar Functions
    udfs: HashMap<String, Arc<ScalarUDF>>,
    /// Aggregate Functions
    udafs: HashMap<String, Arc<AggregateUDF>>,
    /// Window Functions
    udwfs: HashMap<String, Arc<WindowUDF>>,
}

impl MemoryFunctionRegistry {
    pub fn new() -> Self {
        Self::default()
    }
}

impl FunctionRegistry for MemoryFunctionRegistry {
    fn udfs(&self) -> HashSet<String> {
        self.udfs.keys().cloned().collect()
    }

    fn udf(&self, name: &str) -> Result<Arc<ScalarUDF>> {
        self.udfs
            .get(name)
            .cloned()
            .ok_or_else(|| plan_datafusion_err!("Function {name} not found"))
    }

    fn udaf(&self, name: &str) -> Result<Arc<AggregateUDF>> {
        self.udafs
            .get(name)
            .cloned()
            .ok_or_else(|| plan_datafusion_err!("Aggregate Function {name} not found"))
    }

    fn udwf(&self, name: &str) -> Result<Arc<WindowUDF>> {
        self.udwfs
            .get(name)
            .cloned()
            .ok_or_else(|| plan_datafusion_err!("Window Function {name} not found"))
    }

    fn register_udf(&mut self, udf: Arc<ScalarUDF>) -> Result<Option<Arc<ScalarUDF>>> {
        Ok(self.udfs.insert(udf.name().to_string(), udf))
    }
    fn register_udaf(
        &mut self,
        udaf: Arc<AggregateUDF>,
    ) -> Result<Option<Arc<AggregateUDF>>> {
        Ok(self.udafs.insert(udaf.name().into(), udaf))
    }
    fn register_udwf(&mut self, udaf: Arc<WindowUDF>) -> Result<Option<Arc<WindowUDF>>> {
        Ok(self.udwfs.insert(udaf.name().into(), udaf))
    }

    fn expr_planners(&self) -> Vec<Arc<dyn ExprPlanner>> {
        vec![]
    }

    fn udafs(&self) -> HashSet<String> {
        self.udafs.keys().cloned().collect()
    }

    fn udwfs(&self) -> HashSet<String> {
        self.udwfs.keys().cloned().collect()
    }
}

/// Supports registering custom [LogicalType]s, including native types.
pub trait LogicalTypeRegistry {
    /// Returns a reference to the logical type named `name`.
    fn get_logical_type(&self, name: &str) -> Result<LogicalTypeRef>;

    /// Registers a new [LogicalTypeRef], returning any previously registered implementation.
    ///
    /// Returns an error if the type cannot be registered, for example, if the registry is
    /// read-only.
    fn register_logical_type(
        &mut self,
        logical_type: LogicalTypeRef,
    ) -> Result<Option<LogicalTypeRef>>;

    /// Deregisters a logical type with the name `name`, returning the implementation that was
    /// deregistered.
    ///
    /// Returns an error if the type cannot be deregistered, for example, if the registry is
    /// read-only.
    fn deregister_logical_type(&mut self, name: &str) -> Result<Option<LogicalTypeRef>>;
}

/// An [`LogicalTypeRegistry`] that uses in memory [`HashMap`]s.
#[derive(Clone, Debug)]
pub struct MemoryLogicalTypeRegistry {
    /// Holds a mapping between the name of an extension type and its logical type.
    ///
    /// TODO: Use TypeSignature to support arguments
    logical_types: HashMap<String, LogicalTypeRef>,
}

impl Default for MemoryLogicalTypeRegistry {
    fn default() -> Self {
        MemoryLogicalTypeRegistry {
            logical_types: HashMap::new(),
        }
    }
}

impl MemoryLogicalTypeRegistry {
    /// Creates an empty [MemoryLogicalTypeRegistry].
    pub fn new() -> Self {
        Self {
            logical_types: HashMap::new(),
        }
    }

    /// Creates a new [MemoryLogicalTypeRegistry] with the provided `types`.
    ///
    /// # Errors
    ///
    /// Returns an error if one of the `types` is a native type.
    pub fn new_with_types(
        types: impl IntoIterator<Item = LogicalTypeRef>,
    ) -> Result<Self> {
        let extension_types = types
            .into_iter()
            .map(|t| match t.signature() {
                TypeSignature::Native(_) => todo!("TODO"),
                TypeSignature::Extension { name, .. } => (name.to_owned(), t),
            })
            .collect();
        Ok(Self {
            logical_types: extension_types,
        })
    }

    /// Returns a list of all registered types.
    pub fn all_types(&self) -> Vec<LogicalTypeRef> {
        self.logical_types.values().cloned().collect()
    }
}

impl LogicalTypeRegistry for MemoryLogicalTypeRegistry {
    fn get_logical_type(&self, name: &str) -> Result<LogicalTypeRef> {
        self.logical_types
            .get(name)
            .ok_or_else(|| plan_datafusion_err!("Logical type not found."))
            .cloned()
    }

    fn register_logical_type(
        &mut self,
        logical_type: LogicalTypeRef,
    ) -> Result<Option<LogicalTypeRef>> {
        let signature = match logical_type.signature() {
            TypeSignature::Native(_) => {
                return internal_err!("Cannot register a native type")
            }
            TypeSignature::Extension { name, .. } => name,
        };
        Ok(self.logical_types.insert(signature.into(), logical_type))
    }

    fn deregister_logical_type(&mut self, name: &str) -> Result<Option<LogicalTypeRef>> {
        Ok(self.logical_types.remove(name))
    }
}

impl From<HashMap<String, LogicalTypeRef>> for MemoryLogicalTypeRegistry {
    fn from(value: HashMap<String, LogicalTypeRef>) -> Self {
        Self {
            logical_types: value,
        }
    }
}
