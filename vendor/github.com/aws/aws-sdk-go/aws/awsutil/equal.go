package awsutil

import (
	"reflect"
)

// DeepEqual returns if the two values are deeply equal like reflect.DeepEqual.
// In addition to this, this method will also dereference the input values if
// possible so the DeepEqual performed will not fail if one parameter is a
// pointer and the other is not.
//
// DeepEqual will not perform indirection of nested values of the input parameters.
func DeepEqual(a, b interface{}) bool {
	ra := reflect.Indirect(reflect.ValueOf(a))
	rb := reflect.Indirect(reflect.ValueOf(b))

	if raValid, rbValid := ra.IsValid(), rb.IsValid(); !raValid && !rbValid {
		// If the elements are both nil, and of the same type the are equal
		// If they are of different types they are not equal
		return reflect.TypeOf(a) == reflect.TypeOf(b)
	} else if raValid != rbValid {
		// Both values must be valid to be equal
		return false
	}

	return reflect.DeepEqual(ra.Interface(), rb.Interface())
}
