/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package unstructured

import (
	"fmt"
	"io/ioutil"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestCodecOfUnstructuredList tests that there are no data races in Encode().
// i.e. that it does not mutate the object being encoded.
func TestCodecOfUnstructuredList(t *testing.T) {
	var wg sync.WaitGroup
	concurrency := 10
	list := UnstructuredList{
		Object: map[string]interface{}{},
	}
	wg.Add(concurrency)
	for i := 0; i < concurrency; i++ {
		go func() {
			defer wg.Done()
			assert.NoError(t, UnstructuredJSONScheme.Encode(&list, ioutil.Discard))
		}()
	}
	wg.Wait()
}

func TestRemoveNestedField(t *testing.T) {
	obj := map[string]interface{}{
		"x": map[string]interface{}{
			"y": 1,
			"a": "foo",
		},
	}
	RemoveNestedField(obj, "x", "a")
	assert.Len(t, obj["x"], 1)
	RemoveNestedField(obj, "x", "y")
	assert.Empty(t, obj["x"])
	RemoveNestedField(obj, "x")
	assert.Empty(t, obj)
	RemoveNestedField(obj, "x") // Remove of a non-existent field
	assert.Empty(t, obj)
}

func TestPanicsOnInvalidTypes(t *testing.T) {
	if !invalidTypeDetectionEnabled {
		t.Skipf("set %s=true to run the test", invalidTypeDetectionEnvVarName)
	}
	tests := map[string]struct {
		f   func(map[string]interface{})
		err string
	}{
		"nestedFieldNoCopy": {
			f: func(m map[string]interface{}) {
				nestedFieldNoCopy(m, "a", "b")
			},
			err: `unexpected nested field type - cannot get ("a" in ["a" "b"]). Expected map[string]interface{}, got string`,
		},
		"NestedString": {
			f: func(m map[string]interface{}) {
				NestedString(m, "b")
			},
			err: `unexpected nested field type while getting ["b"]. Expected string, got int`,
		},
		"NestedBool": {
			f: func(m map[string]interface{}) {
				NestedBool(m, "c")
			},
			err: `unexpected nested field type while getting ["c"]. Expected bool, got string`,
		},
		"NestedFloat64": {
			f: func(m map[string]interface{}) {
				NestedFloat64(m, "d")
			},
			err: `unexpected nested field type while getting ["d"]. Expected float64, got string`,
		},
		"NestedInt64": {
			f: func(m map[string]interface{}) {
				NestedInt64(m, "e")
			},
			err: `unexpected nested field type while getting ["e"]. Expected int64, got string`,
		},
		"NestedStringSlice invalid element": {
			f: func(m map[string]interface{}) {
				NestedStringSlice(m, "f")
			},
			err: `unexpected slice element type while getting ["f"]. Expected string, got int`,
		},
		"NestedStringSlice": {
			f: func(m map[string]interface{}) {
				NestedStringSlice(m, "g")
			},
			err: `unexpected nested field type while getting ["g"]. Expected []interface{}, got string`,
		},
		"NestedSlice": {
			f: func(m map[string]interface{}) {
				NestedSlice(m, "h")
			},
			err: `unexpected nested field type while getting ["h"]. Expected []interface{}, got string`,
		},
		"NestedStringMap invalid element": {
			f: func(m map[string]interface{}) {
				NestedStringMap(m, "i")
			},
			err: `unexpected map value type while getting ["i"]. Expected string, got int`,
		},
		"NestedStringMap": {
			f: func(m map[string]interface{}) {
				NestedStringMap(m, "a")
			},
			err: `unexpected nested field type while getting ["a"]. Expected map[string]interface{}, got string`,
		},
		"NestedMap": {
			f: func(m map[string]interface{}) {
				NestedMap(m, "a")
			},
			err: `unexpected nested field type while getting ["a"]. Expected map[string]interface{}, got string`,
		},
		"setNestedFieldNoCopy": {
			f: func(m map[string]interface{}) {
				setNestedFieldNoCopy(m, 44, "a", "x")
			},
			err: `unexpected nested field type - cannot set ("a" in ["a" "x"]). Expected map[string]interface{}, got string`,
		},
		"RemoveNestedField": {
			f: func(m map[string]interface{}) {
				RemoveNestedField(m, "a", "x")
			},
			err: `unexpected nested field type - cannot remove ("a" in ["a" "x"]). Expected map[string]interface{}, got string`,
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			m := map[string]interface{}{
				"a": "should be a map[string]interface{}",
				"b": 5, // should be a string
				"c": "should be a bool",
				"d": "should be a float64",
				"e": "should be an int64",
				"f": []interface{}{42}, // should be a []interface{} with strings elements
				"g": "should be a []interface{} with strings elements",
				"h": "should be a []interface{}",
				"i": map[string]interface{}{
					"a": 43, // should be a string
				},
			}
			panicsWithErrorMessage(t, test.err, func() {
				test.f(m)
			})
		})
	}
}

// didPanic returns true if the function passed to it panics. Otherwise, it returns false.
func didPanic(f func()) (bool, interface{}) {
	didPanic := false
	var message interface{}
	func() {
		defer func() {
			if message = recover(); message != nil {
				didPanic = true
			}
		}()
		// call the target function
		f()
	}()
	return didPanic, message
}

func panicsWithErrorMessage(t *testing.T, expected string, f func()) {
	funcDidPanic, panicValue := didPanic(f)
	if !funcDidPanic {
		assert.Fail(t, fmt.Sprintf("func %#v should panic\n\r\tPanic value:\t%v", f, panicValue))
		return
	}
	e, ok := panicValue.(error)
	if !ok || e.Error() != expected {
		assert.Fail(t, fmt.Sprintf("func %#v should panic with value:\t%v\n\r\tPanic value:\t%v", f, expected, panicValue))
		return
	}
}
