package graphite

import "fmt"

// An UnsupportedValueError is returned when a parsed value is not
// supported.
type UnsupportedValueError struct {
	Field string
	Value float64
}

func (err *UnsupportedValueError) Error() string {
	return fmt.Sprintf(`field "%s" value: "%v" is unsupported`, err.Field, err.Value)
}
