//go:build !go1.17
// +build !go1.17

// TODO: once support for Go 1.16 is dropped, this file can be
//       merged/removed with assertion_compare_go1.17_test.go and
//       assertion_compare_can_convert.go

package assert

import "reflect"

// Older versions of Go does not have the reflect.Value.CanConvert
// method.
func canConvert(value reflect.Value, to reflect.Type) bool {
	return false
}
