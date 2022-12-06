//go:build go1.17
// +build go1.17

// TODO: once support for Go 1.16 is dropped, this file can be
//       merged/removed with assertion_compare_go1.17_test.go and
//       assertion_compare_legacy.go

package assert

import "reflect"

// Wrapper around reflect.Value.CanConvert, for compatibility
// reasons.
func canConvert(value reflect.Value, to reflect.Type) bool {
	return value.CanConvert(to)
}
