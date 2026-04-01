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

use crate::ScalarValue;
use crate::error::_internal_err;
use crate::types::extension::DFExtensionType;
use arrow::array::{Array, AsArray, Int16Array};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{
    DataType, Int16Type, TimeUnit, TimestampMicrosecondType, TimestampMillisecondType,
    TimestampNanosecondType, TimestampSecondType,
};
use arrow::util::display::{ArrayFormatter, DisplayIndex, FormatOptions, FormatResult};
use arrow_schema::ArrowError;
use arrow_schema::extension::{ExtensionType, TimestampWithOffset};
use std::fmt::Write;

/// Defines the extension type logic for the canonical `arrow.timestamp_with_offset` extension type.
///
/// See [`DFExtensionType`] for information on DataFusion's extension type mechanism.
#[derive(Debug, Clone)]
pub struct DFTimestampWithOffset {
    inner: TimestampWithOffset,
    storage_type: DataType,
}

impl ExtensionType for DFTimestampWithOffset {
    const NAME: &'static str = TimestampWithOffset::NAME;
    type Metadata = <TimestampWithOffset as ExtensionType>::Metadata;

    fn metadata(&self) -> &Self::Metadata {
        self.inner.metadata()
    }

    fn serialize_metadata(&self) -> Option<String> {
        self.inner.serialize_metadata()
    }

    fn deserialize_metadata(
        metadata: Option<&str>,
    ) -> Result<Self::Metadata, ArrowError> {
        TimestampWithOffset::deserialize_metadata(metadata)
    }

    fn supports_data_type(&self, data_type: &DataType) -> Result<(), ArrowError> {
        self.inner.supports_data_type(data_type)
    }

    fn try_new(
        data_type: &DataType,
        metadata: Self::Metadata,
    ) -> Result<Self, ArrowError> {
        Ok(Self {
            inner: <TimestampWithOffset as ExtensionType>::try_new(data_type, metadata)?,
            storage_type: data_type.clone(),
        })
    }
}

impl DFExtensionType for DFTimestampWithOffset {
    fn storage_type(&self) -> DataType {
        self.storage_type.clone()
    }

    fn serialize_metadata(&self) -> Option<String> {
        self.inner.serialize_metadata()
    }

    fn create_array_formatter<'fmt>(
        &self,
        array: &'fmt dyn Array,
        options: &FormatOptions<'fmt>,
    ) -> crate::Result<Option<ArrayFormatter<'fmt>>> {
        if array.data_type() != &self.storage_type {
            return _internal_err!(
                "Unexpected data type for TimestampWithOffset: {}",
                array.data_type()
            );
        }

        let struct_array = array.as_struct();
        let timestamp_array = struct_array
            .column_by_name("timestamp")
            .expect("Type checked above")
            .as_ref();
        let offset_array = struct_array
            .column_by_name("offset_minutes")
            .expect("Type checked above")
            .as_primitive::<Int16Type>();

        let display_index = TimestampWithOffsetDisplayIndex {
            null_buffer: struct_array.nulls(),
            timestamp_array,
            offset_array,
            options: options.clone(),
        };

        Ok(Some(ArrayFormatter::new(
            Box::new(display_index),
            options.safe(),
        )))
    }
}

struct TimestampWithOffsetDisplayIndex<'a> {
    /// The inner arrays are always non-null. Use the null buffer of the struct array to check
    /// whether an element is null.
    null_buffer: Option<&'a NullBuffer>,
    timestamp_array: &'a dyn Array,
    offset_array: &'a Int16Array,
    options: FormatOptions<'a>,
}

impl DisplayIndex for TimestampWithOffsetDisplayIndex<'_> {
    fn write(&self, idx: usize, f: &mut dyn Write) -> FormatResult {
        if self.null_buffer.map(|nb| nb.is_null(idx)).unwrap_or(false) {
            write!(f, "{}", self.options.null())?;
            return Ok(());
        }

        let offset_minutes = self.offset_array.value(idx);
        let offset = format_offset(offset_minutes);

        // The timestamp array must be UTC, so we can ignore the timezone.
        let scalar = match self.timestamp_array.data_type() {
            DataType::Timestamp(TimeUnit::Second, _) => {
                let ts = self
                    .timestamp_array
                    .as_primitive::<TimestampSecondType>()
                    .value(idx);
                ScalarValue::TimestampSecond(Some(ts), Some(offset.into()))
            }
            DataType::Timestamp(TimeUnit::Millisecond, _) => {
                let ts = self
                    .timestamp_array
                    .as_primitive::<TimestampMillisecondType>()
                    .value(idx);
                ScalarValue::TimestampMillisecond(Some(ts), Some(offset.into()))
            }
            DataType::Timestamp(TimeUnit::Microsecond, _) => {
                let ts = self
                    .timestamp_array
                    .as_primitive::<TimestampMicrosecondType>()
                    .value(idx);
                ScalarValue::TimestampMicrosecond(Some(ts), Some(offset.into()))
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                let ts = self
                    .timestamp_array
                    .as_primitive::<TimestampNanosecondType>()
                    .value(idx);
                ScalarValue::TimestampNanosecond(Some(ts), Some(offset.into()))
            }
            _ => unreachable!("TimestampWithOffset storage must be a Timestamp array"),
        };

        let array = scalar.to_array().map_err(|_| {
            ArrowError::ComputeError("Failed to convert scalar to array".to_owned())
        })?;
        let formatter = ArrayFormatter::try_new(&array, &self.options)?;
        formatter.value(0).write(f)?;

        Ok(())
    }
}

/// Formats the offset in the format `+/-HH:MM`, which can be used as an offset in the regular
/// timestamp types.
fn format_offset(minutes: i16) -> String {
    let sign = if minutes >= 0 { '+' } else { '-' };
    let minutes = minutes.abs();
    let hours = minutes / 60;
    let minutes = minutes % 60;
    format!("{sign}{hours:02}:{minutes:02}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{StructArray, TimestampSecondArray};
    use arrow::datatypes::{Field, Fields};
    use chrono::{TimeZone, Utc};
    use std::sync::Arc;

    #[test]
    fn test_pretty_print_timestamp_with_offset() -> Result<(), ArrowError> {
        let fields = create_fields(TimeUnit::Second);

        let ts = Utc
            .with_ymd_and_hms(2024, 4, 1, 0, 0, 0)
            .unwrap()
            .timestamp();

        let timestamp_array =
            Arc::new(TimestampSecondArray::from(vec![ts, ts, ts]).with_timezone("UTC"));
        let offset_array = Arc::new(Int16Array::from(vec![60, -105, 0]));

        let struct_array = StructArray::try_new(
            fields,
            vec![timestamp_array, offset_array],
            Some(NullBuffer::from(vec![true, true, false])),
        )?;

        let formatter = DFTimestampWithOffset::try_new(struct_array.data_type(), ())?
            .create_array_formatter(
                &struct_array,
                &FormatOptions::default().with_null("NULL"),
            )?
            .unwrap();

        assert_eq!(formatter.value(0).to_string(), "2024-04-01T01:00:00+01:00");
        assert_eq!(formatter.value(1).to_string(), "2024-03-31T22:15:00-01:45");
        assert_eq!(formatter.value(2).to_string(), "NULL");

        Ok(())
    }

    fn create_fields(time_unit: TimeUnit) -> Fields {
        let ts_field = Field::new(
            "timestamp",
            DataType::Timestamp(time_unit, Some("UTC".into())),
            false,
        );
        let offset_field = Field::new("offset_minutes", DataType::Int16, false);
        Fields::from(vec![ts_field, offset_field])
    }
}
