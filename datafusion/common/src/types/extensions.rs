use crate::Result;
use crate::ScalarValue;
use arrow::array::Array;

/// Implements pretty printing for a set of types.
///
/// For example, the default pretty-printer for a byte array might not be adequate for a UUID type,
/// which is physically stored as a fixed-length byte array. This extension allows the user to
/// override the default pretty-printer for a given type.
pub trait ValuePrettyPrinter {
    /// Pretty print a scalar value.
    ///
    /// # Error
    ///
    /// Will return an error if the given `df_type` is not supported by this pretty printer.
    fn pretty_print_scalar(&self, value: &ScalarValue) -> Result<String>;

    /// Pretty print a specific value of a given array.
    ///
    /// # Error
    ///
    /// Will return an error if the given `df_type` is not supported by this pretty printer.
    fn pretty_print_array(&self, array: &dyn Array, index: usize) -> Result<String> {
        let value = ScalarValue::try_from_array(array, index)?;
        self.pretty_print_scalar(&value)
    }
}

/// The default pretty printer.
///
/// Uses the arrow implementation of printing values.
pub struct DefaultValuePrettyPrinter;

impl ValuePrettyPrinter for DefaultValuePrettyPrinter {
    fn pretty_print_scalar(&self, value: &ScalarValue) -> Result<String> {
        Ok(value.to_string())
    }
}
