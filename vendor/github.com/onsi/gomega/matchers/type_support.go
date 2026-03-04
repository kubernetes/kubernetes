/*
Gomega matchers

This package implements the Gomega matchers and does not typically need to be imported.
See the docs for Gomega for documentation on the matchers

http://onsi.github.io/gomega/
*/

// untested sections: 11

package matchers

import (
	"encoding/json"
	"fmt"
	"reflect"

	"github.com/onsi/gomega/matchers/internal/miter"
)

type omegaMatcher interface {
	Match(actual any) (success bool, err error)
	FailureMessage(actual any) (message string)
	NegatedFailureMessage(actual any) (message string)
}

func isBool(a any) bool {
	return reflect.TypeOf(a).Kind() == reflect.Bool
}

func isNumber(a any) bool {
	if a == nil {
		return false
	}
	kind := reflect.TypeOf(a).Kind()
	return reflect.Int <= kind && kind <= reflect.Float64
}

func isInteger(a any) bool {
	kind := reflect.TypeOf(a).Kind()
	return reflect.Int <= kind && kind <= reflect.Int64
}

func isUnsignedInteger(a any) bool {
	kind := reflect.TypeOf(a).Kind()
	return reflect.Uint <= kind && kind <= reflect.Uint64
}

func isFloat(a any) bool {
	kind := reflect.TypeOf(a).Kind()
	return reflect.Float32 <= kind && kind <= reflect.Float64
}

func toInteger(a any) int64 {
	if isInteger(a) {
		return reflect.ValueOf(a).Int()
	} else if isUnsignedInteger(a) {
		return int64(reflect.ValueOf(a).Uint())
	} else if isFloat(a) {
		return int64(reflect.ValueOf(a).Float())
	}
	panic(fmt.Sprintf("Expected a number!  Got <%T> %#v", a, a))
}

func toUnsignedInteger(a any) uint64 {
	if isInteger(a) {
		return uint64(reflect.ValueOf(a).Int())
	} else if isUnsignedInteger(a) {
		return reflect.ValueOf(a).Uint()
	} else if isFloat(a) {
		return uint64(reflect.ValueOf(a).Float())
	}
	panic(fmt.Sprintf("Expected a number!  Got <%T> %#v", a, a))
}

func toFloat(a any) float64 {
	if isInteger(a) {
		return float64(reflect.ValueOf(a).Int())
	} else if isUnsignedInteger(a) {
		return float64(reflect.ValueOf(a).Uint())
	} else if isFloat(a) {
		return reflect.ValueOf(a).Float()
	}
	panic(fmt.Sprintf("Expected a number!  Got <%T> %#v", a, a))
}

func isError(a any) bool {
	_, ok := a.(error)
	return ok
}

func isChan(a any) bool {
	if isNil(a) {
		return false
	}
	return reflect.TypeOf(a).Kind() == reflect.Chan
}

func isMap(a any) bool {
	if a == nil {
		return false
	}
	return reflect.TypeOf(a).Kind() == reflect.Map
}

func isArrayOrSlice(a any) bool {
	if a == nil {
		return false
	}
	switch reflect.TypeOf(a).Kind() {
	case reflect.Array, reflect.Slice:
		return true
	default:
		return false
	}
}

func isString(a any) bool {
	if a == nil {
		return false
	}
	return reflect.TypeOf(a).Kind() == reflect.String
}

func toString(a any) (string, bool) {
	aString, isString := a.(string)
	if isString {
		return aString, true
	}

	aBytes, isBytes := a.([]byte)
	if isBytes {
		return string(aBytes), true
	}

	aStringer, isStringer := a.(fmt.Stringer)
	if isStringer {
		return aStringer.String(), true
	}

	aJSONRawMessage, isJSONRawMessage := a.(json.RawMessage)
	if isJSONRawMessage {
		return string(aJSONRawMessage), true
	}

	return "", false
}

func lengthOf(a any) (int, bool) {
	if a == nil {
		return 0, false
	}
	switch reflect.TypeOf(a).Kind() {
	case reflect.Map, reflect.Array, reflect.String, reflect.Chan, reflect.Slice:
		return reflect.ValueOf(a).Len(), true
	case reflect.Func:
		if !miter.IsIter(a) {
			return 0, false
		}
		var l int
		if miter.IsSeq2(a) {
			miter.IterateKV(a, func(k, v reflect.Value) bool { l++; return true })
		} else {
			miter.IterateV(a, func(v reflect.Value) bool { l++; return true })
		}
		return l, true
	default:
		return 0, false
	}
}
func capOf(a any) (int, bool) {
	if a == nil {
		return 0, false
	}
	switch reflect.TypeOf(a).Kind() {
	case reflect.Array, reflect.Chan, reflect.Slice:
		return reflect.ValueOf(a).Cap(), true
	default:
		return 0, false
	}
}

func isNil(a any) bool {
	if a == nil {
		return true
	}

	switch reflect.TypeOf(a).Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Ptr, reflect.Slice:
		return reflect.ValueOf(a).IsNil()
	}

	return false
}
