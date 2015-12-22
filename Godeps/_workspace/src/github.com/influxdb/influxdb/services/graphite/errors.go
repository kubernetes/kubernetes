package graphite

import "fmt"

// An UnsupposedValueError is returned when a parsed value is not
// supposed.
type UnsupposedValueError struct {
	Field string
	Value float64
}

func (err *UnsupposedValueError) Error() string {
	return fmt.Sprintf(`field "%s" value: "%v" is unsupported`, err.Field, err.Value)
}
