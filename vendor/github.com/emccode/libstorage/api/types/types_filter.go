package types

// FilterOperator is a filter operator.
type FilterOperator int

const (
	// FilterAnd is the & operator.
	FilterAnd FilterOperator = iota

	// FilterOr is the | operator.
	FilterOr

	// FilterNot is the ! operator.
	FilterNot

	// FilterPresent is the =* operator.
	FilterPresent

	// FilterEqualityMatch is the = operator.
	FilterEqualityMatch

	// FilterSubstrings is the = operator in conjunction with a string that
	// has leading and trailing * characters.
	FilterSubstrings

	// FilterSubstringsPrefix is the = operator in conjunction with a string
	// that has a leading * character.
	FilterSubstringsPrefix

	// FilterSubstringsPostfix is the = operator in conjunction with a string
	// that has a trailing * character.
	FilterSubstringsPostfix

	// FilterGreaterOrEqual is the >= operator.
	FilterGreaterOrEqual

	// FilterLessOrEqual is the <= operator.
	FilterLessOrEqual

	// FilterApproxMatch is the ~= operator.
	FilterApproxMatch
)

// Filter is an LDAP-style filter string.
type Filter struct {

	// Op is the operation.
	Op FilterOperator

	// Children is a list of any sub-filters if this filter is a compound
	// filter.
	Children []*Filter

	// Left is the left operand.
	Left string

	// Right is the right operand.
	Right string
}
