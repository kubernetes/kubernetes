//This package is copied from Go library text/template.
//The original private functions indirect and printableValue
//are exported as public functions.
package template

import (
	"fmt"
	"reflect"
)

var (
	errorType       = reflect.TypeOf((*error)(nil)).Elem()
	fmtStringerType = reflect.TypeOf((*fmt.Stringer)(nil)).Elem()
)

// Indirect returns the item at the end of indirection, and a bool to indicate if it's nil.
// We indirect through pointers and empty interfaces (only) because
// non-empty interfaces have methods we might need.
func Indirect(v reflect.Value) (rv reflect.Value, isNil bool) {
	for ; v.Kind() == reflect.Pointer || v.Kind() == reflect.Interface; v = v.Elem() {
		if v.IsNil() {
			return v, true
		}
		if v.Kind() == reflect.Interface && v.NumMethod() > 0 {
			break
		}
	}
	return v, false
}

// PrintableValue returns the, possibly indirected, interface value inside v that
// is best for a call to formatted printer.
func PrintableValue(v reflect.Value) (interface{}, bool) {
	if v.Kind() == reflect.Pointer {
		v, _ = Indirect(v) // fmt.Fprint handles nil.
	}
	if !v.IsValid() {
		return "<no value>", true
	}

	if !v.Type().Implements(errorType) && !v.Type().Implements(fmtStringerType) {
		if v.CanAddr() && (reflect.PointerTo(v.Type()).Implements(errorType) || reflect.PointerTo(v.Type()).Implements(fmtStringerType)) {
			v = v.Addr()
		} else {
			switch v.Kind() {
			case reflect.Chan, reflect.Func:
				return nil, false
			}
		}
	}
	return v.Interface(), true
}
