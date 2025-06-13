/// Represents the null-handling behavior when evaluating an equality expression.
///
/// # Order
///
/// The order on this type represents the "restrictiveness" of the behavior. The more restrictive
/// a behavior is, the fewer elements can be matched with `null` while evaluating equalities.
/// [EqualityNullBehavior::NullEqualsNothing] represents the most restrictive behavior.
///
/// This mirrors the old order with booleans, as `false` indicated that `null != null`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Hash)]
pub enum EqualityNullBehavior {
    /// Null is *not* equal to null while joining (`null != null`)
    NullEqualsNothing,
    /// Null is equal to null while joining (`null == null`)
    NullEqualsNull,
}
