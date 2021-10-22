package users

import "fmt"

// InvalidListFilter is returned by the ToUserListQuery method when validation of
// a filter does not pass
type InvalidListFilter struct {
	FilterName string
}

func (e InvalidListFilter) Error() string {
	s := fmt.Sprintf(
		"Invalid filter name [%s]: it must be in format of NAME__COMPARATOR",
		e.FilterName,
	)
	return s
}
