package lumberjack

import (
	"fmt"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"
)

// assert will log the given message if condition is false.
func assert(condition bool, t testing.TB, msg string, v ...interface{}) {
	assertUp(condition, t, 1, msg, v...)
}

// assertUp is like assert, but used inside helper functions, to ensure that
// the file and line number reported by failures corresponds to one or more
// levels up the stack.
func assertUp(condition bool, t testing.TB, caller int, msg string, v ...interface{}) {
	if !condition {
		_, file, line, _ := runtime.Caller(caller + 1)
		v = append([]interface{}{filepath.Base(file), line}, v...)
		fmt.Printf("%s:%d: "+msg+"\n", v...)
		t.FailNow()
	}
}

// equals tests that the two values are equal according to reflect.DeepEqual.
func equals(exp, act interface{}, t testing.TB) {
	equalsUp(exp, act, t, 1)
}

// equalsUp is like equals, but used inside helper functions, to ensure that the
// file and line number reported by failures corresponds to one or more levels
// up the stack.
func equalsUp(exp, act interface{}, t testing.TB, caller int) {
	if !reflect.DeepEqual(exp, act) {
		_, file, line, _ := runtime.Caller(caller + 1)
		fmt.Printf("%s:%d: exp: %v (%T), got: %v (%T)\n",
			filepath.Base(file), line, exp, exp, act, act)
		t.FailNow()
	}
}

// isNil reports a failure if the given value is not nil.  Note that values
// which cannot be nil will always fail this check.
func isNil(obtained interface{}, t testing.TB) {
	isNilUp(obtained, t, 1)
}

// isNilUp is like isNil, but used inside helper functions, to ensure that the
// file and line number reported by failures corresponds to one or more levels
// up the stack.
func isNilUp(obtained interface{}, t testing.TB, caller int) {
	if !_isNil(obtained) {
		_, file, line, _ := runtime.Caller(caller + 1)
		fmt.Printf("%s:%d: expected nil, got: %v\n", filepath.Base(file), line, obtained)
		t.FailNow()
	}
}

// notNil reports a failure if the given value is nil.
func notNil(obtained interface{}, t testing.TB) {
	notNilUp(obtained, t, 1)
}

// notNilUp is like notNil, but used inside helper functions, to ensure that the
// file and line number reported by failures corresponds to one or more levels
// up the stack.
func notNilUp(obtained interface{}, t testing.TB, caller int) {
	if _isNil(obtained) {
		_, file, line, _ := runtime.Caller(caller + 1)
		fmt.Printf("%s:%d: expected non-nil, got: %v\n", filepath.Base(file), line, obtained)
		t.FailNow()
	}
}

// _isNil is a helper function for isNil and notNil, and should not be used
// directly.
func _isNil(obtained interface{}) bool {
	if obtained == nil {
		return true
	}

	switch v := reflect.ValueOf(obtained); v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Ptr, reflect.Slice:
		return v.IsNil()
	}

	return false
}
