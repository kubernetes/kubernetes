/*
Copyright 2015 The Kubernetes Authors.

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

package compat

import (
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

// Based on: https://github.com/openshift/origin/blob/master/pkg/api/compatibility_test.go
//
// TestCompatibility reencodes the input using the codec for the given
// version and checks for the presence of the expected keys and absent
// keys in the resulting JSON.
func TestCompatibility(
	t *testing.T,
	version unversioned.GroupVersion,
	input []byte,
	validator func(obj runtime.Object) field.ErrorList,
	expectedKeys map[string]string,
	absentKeys []string,
) {

	// Decode
	codec := api.Codecs.LegacyCodec(version)
	obj, err := runtime.Decode(codec, input)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Validate
	errs := validator(obj)
	if len(errs) != 0 {
		t.Fatalf("Unexpected validation errors: %v", errs)
	}

	// Encode
	output, err := runtime.Encode(codec, obj)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Validate old and new fields are encoded
	generic := map[string]interface{}{}
	if err := json.Unmarshal(output, &generic); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	hasError := false
	for k, expectedValue := range expectedKeys {
		keys := strings.Split(k, ".")
		if actualValue, ok, err := getJSONValue(generic, keys...); err != nil || !ok {
			t.Errorf("Unexpected error for %s: %v", k, err)
			hasError = true
		} else if !reflect.DeepEqual(expectedValue, fmt.Sprintf("%v", actualValue)) {
			hasError = true
			t.Errorf("Unexpected value for %v: expected %v, got %v", k, expectedValue, actualValue)
		}
	}

	for _, absentKey := range absentKeys {
		keys := strings.Split(absentKey, ".")
		actualValue, ok, err := getJSONValue(generic, keys...)
		if err == nil || ok {
			t.Errorf("Unexpected value found for key %s: %v", absentKey, actualValue)
			hasError = true
		}
	}

	if hasError {
		printer := new(kubectl.JSONPrinter)
		printer.PrintObj(obj, os.Stdout)
		t.Logf("2: Encoded value: %#v", string(output))
	}
}

func getJSONValue(data map[string]interface{}, keys ...string) (interface{}, bool, error) {
	// No keys, current value is it
	if len(keys) == 0 {
		return data, true, nil
	}

	// Get the key (and optional index)
	key := keys[0]
	index := -1
	if matches := regexp.MustCompile(`^(.*)\[(\d+)\]$`).FindStringSubmatch(key); len(matches) > 0 {
		key = matches[1]
		index, _ = strconv.Atoi(matches[2])
	}

	// Look up the value
	value, ok := data[key]
	if !ok {
		return nil, false, fmt.Errorf("No key %s found", key)
	}

	// Get the indexed value if an index is specified
	if index >= 0 {
		valueSlice, ok := value.([]interface{})
		if !ok {
			return nil, false, fmt.Errorf("Key %s did not hold a slice", key)
		}
		if index >= len(valueSlice) {
			return nil, false, fmt.Errorf("Index %d out of bounds for slice at key: %v", index, key)
		}
		value = valueSlice[index]
	}

	if len(keys) == 1 {
		return value, true, nil
	}

	childData, ok := value.(map[string]interface{})
	if !ok {
		return nil, false, fmt.Errorf("Key %s did not hold a map", keys[0])
	}
	return getJSONValue(childData, keys[1:]...)
}
