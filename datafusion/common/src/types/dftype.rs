use crate::types::{LogicalType, LogicalTypeRef, NativeType, UnknownExtensionType};
use crate::ScalarValue;
use crate::{Result, _plan_datafusion_err};
use arrow::datatypes::DataType;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// TODO
#[derive(Debug, Clone, PartialOrd, Ord, Eq)]
pub struct DFType {
    /// The underlying storage type.
    storage_type: DataType,
    /// The logical type.
    ///
    /// Metadata information that is relevant for the type system should be captured in the
    /// [LogicalType] instance.
    logical_type: Arc<dyn LogicalType>,
}

impl DFType {
    /// Tries to create a new [DFType].
    ///
    /// Returns an error if `storage_type` is not compatible with `logical_type`.
    pub fn try_new(storage_type: DataType, logical_type: LogicalTypeRef) -> Result<Self> {
        // TODO check

        Ok(Self {
            storage_type,
            logical_type,
        })
    }

    /// Creates a new [DFType], falling back to the logical type [UnknownExtensionType] if extension
    /// type information is available.
    ///
    /// In general, this function should only be called if no extension type registry is available.
    /// With increasing penetration of extension types in DataFusion, more and more validations
    /// will reject the existence of an [UnknownExtensionType], thus leading to validation errors.
    pub fn new_with_fallback(
        storage_type: &DataType,
        metadata: &HashMap<String, String>,
    ) -> Self {
        if let Some(extension_type_name) =
            metadata.get(arrow_schema::extension::EXTENSION_TYPE_NAME_KEY)
        {
            return Self {
                storage_type: storage_type.clone(),
                logical_type: Arc::new(UnknownExtensionType::new(
                    extension_type_name.to_owned(),
                    storage_type.clone().into(),
                )),
            };
        }

        Self {
            storage_type: storage_type.clone(),
            logical_type: Arc::new(NativeType::from(storage_type.clone())),
        }
    }

    /// Creates a new [DFType] from the given `storage_type` using the native type as logical type.
    pub fn new_from_storage_type(storage_type: DataType) -> Self {
        Self {
            logical_type: Arc::new(NativeType::from(&storage_type)),
            storage_type,
        }
    }
}

impl PartialEq for DFType {
    fn eq(&self, other: &Self) -> bool {
        self.storage_type == other.storage_type
            && &self.logical_type == &other.logical_type
    }
}

impl Hash for DFType {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.storage_type.hash(state);
        self.logical_type.hash(state);
    }
}

/// TODO
pub struct NullableDFType {
    /// The inner [DFType].
    inner: DFType,
    /// Whether or not the value is nullable.
    nullable: bool,
}

/// TODO
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct DFTypedScalarValue {
    /// TODO
    value: ScalarValue,
    /// TODO
    df_type: DFType,
}

impl DFTypedScalarValue {
    /// Creates a new [DFTypedScalarValue] ensuring that the data type of the `value` matches the
    /// storage type of `df_type`.
    ///
    /// Will return an error if the data types do not match.
    pub fn try_new(value: ScalarValue, df_type: DFType) -> Result<Self> {
        if value.data_type() != df_type.storage_type {
            return Err(_plan_datafusion_err!(
                "The DataType of the scalar value does not match its DataFusion Type."
            ));
        }

        Ok(Self { value, df_type })
    }
}

impl PartialOrd for DFTypedScalarValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.df_type != other.df_type {
            return None;
        }

        self.value.partial_cmp(&other.value)
    }
}
