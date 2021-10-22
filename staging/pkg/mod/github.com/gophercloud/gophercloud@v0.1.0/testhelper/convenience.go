package testhelper

import (
	"bytes"
	"encoding/json"
	"fmt"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

const (
	logBodyFmt = "\033[1;31m%s %s\033[0m"
	greenCode  = "\033[0m\033[1;32m"
	yellowCode = "\033[0m\033[1;33m"
	resetCode  = "\033[0m\033[1;31m"
)

func prefix(depth int) string {
	_, file, line, _ := runtime.Caller(depth)
	return fmt.Sprintf("Failure in %s, line %d:", filepath.Base(file), line)
}

func green(str interface{}) string {
	return fmt.Sprintf("%s%#v%s", greenCode, str, resetCode)
}

func yellow(str interface{}) string {
	return fmt.Sprintf("%s%#v%s", yellowCode, str, resetCode)
}

func logFatal(t *testing.T, str string) {
	t.Fatalf(logBodyFmt, prefix(3), str)
}

func logError(t *testing.T, str string) {
	t.Errorf(logBodyFmt, prefix(3), str)
}

type diffLogger func([]string, interface{}, interface{})

type visit struct {
	a1  uintptr
	a2  uintptr
	typ reflect.Type
}

// Recursively visits the structures of "expected" and "actual". The diffLogger function will be
// invoked with each different value encountered, including the reference path that was followed
// to get there.
func deepDiffEqual(expected, actual reflect.Value, visited map[visit]bool, path []string, logDifference diffLogger) {
	defer func() {
		// Fall back to the regular reflect.DeepEquals function.
		if r := recover(); r != nil {
			var e, a interface{}
			if expected.IsValid() {
				e = expected.Interface()
			}
			if actual.IsValid() {
				a = actual.Interface()
			}

			if !reflect.DeepEqual(e, a) {
				logDifference(path, e, a)
			}
		}
	}()

	if !expected.IsValid() && actual.IsValid() {
		logDifference(path, nil, actual.Interface())
		return
	}
	if expected.IsValid() && !actual.IsValid() {
		logDifference(path, expected.Interface(), nil)
		return
	}
	if !expected.IsValid() && !actual.IsValid() {
		return
	}

	hard := func(k reflect.Kind) bool {
		switch k {
		case reflect.Array, reflect.Map, reflect.Slice, reflect.Struct:
			return true
		}
		return false
	}

	if expected.CanAddr() && actual.CanAddr() && hard(expected.Kind()) {
		addr1 := expected.UnsafeAddr()
		addr2 := actual.UnsafeAddr()

		if addr1 > addr2 {
			addr1, addr2 = addr2, addr1
		}

		if addr1 == addr2 {
			// References are identical. We can short-circuit
			return
		}

		typ := expected.Type()
		v := visit{addr1, addr2, typ}
		if visited[v] {
			// Already visited.
			return
		}

		// Remember this visit for later.
		visited[v] = true
	}

	switch expected.Kind() {
	case reflect.Array:
		for i := 0; i < expected.Len(); i++ {
			hop := append(path, fmt.Sprintf("[%d]", i))
			deepDiffEqual(expected.Index(i), actual.Index(i), visited, hop, logDifference)
		}
		return
	case reflect.Slice:
		if expected.IsNil() != actual.IsNil() {
			logDifference(path, expected.Interface(), actual.Interface())
			return
		}
		if expected.Len() == actual.Len() && expected.Pointer() == actual.Pointer() {
			return
		}
		for i := 0; i < expected.Len(); i++ {
			hop := append(path, fmt.Sprintf("[%d]", i))
			deepDiffEqual(expected.Index(i), actual.Index(i), visited, hop, logDifference)
		}
		return
	case reflect.Interface:
		if expected.IsNil() != actual.IsNil() {
			logDifference(path, expected.Interface(), actual.Interface())
			return
		}
		deepDiffEqual(expected.Elem(), actual.Elem(), visited, path, logDifference)
		return
	case reflect.Ptr:
		deepDiffEqual(expected.Elem(), actual.Elem(), visited, path, logDifference)
		return
	case reflect.Struct:
		for i, n := 0, expected.NumField(); i < n; i++ {
			field := expected.Type().Field(i)
			hop := append(path, "."+field.Name)
			deepDiffEqual(expected.Field(i), actual.Field(i), visited, hop, logDifference)
		}
		return
	case reflect.Map:
		if expected.IsNil() != actual.IsNil() {
			logDifference(path, expected.Interface(), actual.Interface())
			return
		}
		if expected.Len() == actual.Len() && expected.Pointer() == actual.Pointer() {
			return
		}

		var keys []reflect.Value
		if expected.Len() >= actual.Len() {
			keys = expected.MapKeys()
		} else {
			keys = actual.MapKeys()
		}

		for _, k := range keys {
			expectedValue := expected.MapIndex(k)
			actualValue := actual.MapIndex(k)

			if !expectedValue.IsValid() {
				logDifference(path, nil, actual.Interface())
				return
			}
			if !actualValue.IsValid() {
				logDifference(path, expected.Interface(), nil)
				return
			}

			hop := append(path, fmt.Sprintf("[%v]", k))
			deepDiffEqual(expectedValue, actualValue, visited, hop, logDifference)
		}
		return
	case reflect.Func:
		if expected.IsNil() != actual.IsNil() {
			logDifference(path, expected.Interface(), actual.Interface())
		}
		return
	default:
		if expected.Interface() != actual.Interface() {
			logDifference(path, expected.Interface(), actual.Interface())
		}
	}
}

func deepDiff(expected, actual interface{}, logDifference diffLogger) {
	if expected == nil || actual == nil {
		logDifference([]string{}, expected, actual)
		return
	}

	expectedValue := reflect.ValueOf(expected)
	actualValue := reflect.ValueOf(actual)

	if expectedValue.Type() != actualValue.Type() {
		logDifference([]string{}, expected, actual)
		return
	}
	deepDiffEqual(expectedValue, actualValue, map[visit]bool{}, []string{}, logDifference)
}

// AssertEquals compares two arbitrary values and performs a comparison. If the
// comparison fails, a fatal error is raised that will fail the test
func AssertEquals(t *testing.T, expected, actual interface{}) {
	if expected != actual {
		logFatal(t, fmt.Sprintf("expected %s but got %s", green(expected), yellow(actual)))
	}
}

// CheckEquals is similar to AssertEquals, except with a non-fatal error
func CheckEquals(t *testing.T, expected, actual interface{}) {
	if expected != actual {
		logError(t, fmt.Sprintf("expected %s but got %s", green(expected), yellow(actual)))
	}
}

// AssertDeepEquals - like Equals - performs a comparison - but on more complex
// structures that requires deeper inspection
func AssertDeepEquals(t *testing.T, expected, actual interface{}) {
	pre := prefix(2)

	differed := false
	deepDiff(expected, actual, func(path []string, expected, actual interface{}) {
		differed = true
		t.Errorf("\033[1;31m%sat %s expected %s, but got %s\033[0m",
			pre,
			strings.Join(path, ""),
			green(expected),
			yellow(actual))
	})
	if differed {
		logFatal(t, "The structures were different.")
	}
}

// CheckDeepEquals is similar to AssertDeepEquals, except with a non-fatal error
func CheckDeepEquals(t *testing.T, expected, actual interface{}) {
	pre := prefix(2)

	deepDiff(expected, actual, func(path []string, expected, actual interface{}) {
		t.Errorf("\033[1;31m%s at %s expected %s, but got %s\033[0m",
			pre,
			strings.Join(path, ""),
			green(expected),
			yellow(actual))
	})
}

func isByteArrayEquals(t *testing.T, expectedBytes []byte, actualBytes []byte) bool {
	return bytes.Equal(expectedBytes, actualBytes)
}

// AssertByteArrayEquals a convenience function for checking whether two byte arrays are equal
func AssertByteArrayEquals(t *testing.T, expectedBytes []byte, actualBytes []byte) {
	if !isByteArrayEquals(t, expectedBytes, actualBytes) {
		logFatal(t, "The bytes differed.")
	}
}

// CheckByteArrayEquals a convenience function for silent checking whether two byte arrays are equal
func CheckByteArrayEquals(t *testing.T, expectedBytes []byte, actualBytes []byte) {
	if !isByteArrayEquals(t, expectedBytes, actualBytes) {
		logError(t, "The bytes differed.")
	}
}

// isJSONEquals is a utility function that implements JSON comparison for AssertJSONEquals and
// CheckJSONEquals.
func isJSONEquals(t *testing.T, expectedJSON string, actual interface{}) bool {
	var parsedExpected, parsedActual interface{}
	err := json.Unmarshal([]byte(expectedJSON), &parsedExpected)
	if err != nil {
		t.Errorf("Unable to parse expected value as JSON: %v", err)
		return false
	}

	jsonActual, err := json.Marshal(actual)
	AssertNoErr(t, err)
	err = json.Unmarshal(jsonActual, &parsedActual)
	AssertNoErr(t, err)

	if !reflect.DeepEqual(parsedExpected, parsedActual) {
		prettyExpected, err := json.MarshalIndent(parsedExpected, "", "  ")
		if err != nil {
			t.Logf("Unable to pretty-print expected JSON: %v\n%s", err, expectedJSON)
		} else {
			// We can't use green() here because %#v prints prettyExpected as a byte array literal, which
			// is... unhelpful. Converting it to a string first leaves "\n" uninterpreted for some reason.
			t.Logf("Expected JSON:\n%s%s%s", greenCode, prettyExpected, resetCode)
		}

		prettyActual, err := json.MarshalIndent(actual, "", "  ")
		if err != nil {
			t.Logf("Unable to pretty-print actual JSON: %v\n%#v", err, actual)
		} else {
			// We can't use yellow() for the same reason.
			t.Logf("Actual JSON:\n%s%s%s", yellowCode, prettyActual, resetCode)
		}

		return false
	}
	return true
}

// AssertJSONEquals serializes a value as JSON, parses an expected string as JSON, and ensures that
// both are consistent. If they aren't, the expected and actual structures are pretty-printed and
// shown for comparison.
//
// This is useful for comparing structures that are built as nested map[string]interface{} values,
// which are a pain to construct as literals.
func AssertJSONEquals(t *testing.T, expectedJSON string, actual interface{}) {
	if !isJSONEquals(t, expectedJSON, actual) {
		logFatal(t, "The generated JSON structure differed.")
	}
}

// CheckJSONEquals is similar to AssertJSONEquals, but nonfatal.
func CheckJSONEquals(t *testing.T, expectedJSON string, actual interface{}) {
	if !isJSONEquals(t, expectedJSON, actual) {
		logError(t, "The generated JSON structure differed.")
	}
}

// AssertNoErr is a convenience function for checking whether an error value is
// an actual error
func AssertNoErr(t *testing.T, e error) {
	if e != nil {
		logFatal(t, fmt.Sprintf("unexpected error %s", yellow(e.Error())))
	}
}

// CheckNoErr is similar to AssertNoErr, except with a non-fatal error
func CheckNoErr(t *testing.T, e error) {
	if e != nil {
		logError(t, fmt.Sprintf("unexpected error %s", yellow(e.Error())))
	}
}
