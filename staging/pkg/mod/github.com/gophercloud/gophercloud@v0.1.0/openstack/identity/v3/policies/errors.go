package policies

import "fmt"

// InvalidListFilter is returned by the ToPolicyListQuery method when
// validation of a filter does not pass
type InvalidListFilter struct {
	FilterName string
}

func (e InvalidListFilter) Error() string {
	s := fmt.Sprintf(
		"Invalid filter name [%s]: it must be in format of TYPE__COMPARATOR",
		e.FilterName,
	)
	return s
}

// StringFieldLengthExceedsLimit is returned by the
// ToPolicyCreateMap/ToPolicyUpdateMap methods when validation of
// a type does not pass
type StringFieldLengthExceedsLimit struct {
	Field string
	Limit int
}

func (e StringFieldLengthExceedsLimit) Error() string {
	return fmt.Sprintf("String length of field [%s] exceeds limit (%d)",
		e.Field, e.Limit,
	)
}
