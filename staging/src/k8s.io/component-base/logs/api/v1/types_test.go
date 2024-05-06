/*
Copyright 2022 The Kubernetes Authors.

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

package v1

import (
	enjson "encoding/json"
	"fmt"
	"math"
	"reflect"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"sigs.k8s.io/json"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestVModule(t *testing.T) {
	testcases := []struct {
		arg         string
		expectError string
		expectValue VModuleConfiguration
		expectParam string
	}{
		{
			arg: "gopher*=1",
			expectValue: VModuleConfiguration{
				{
					FilePattern: "gopher*",
					Verbosity:   1,
				},
			},
		},
		{
			arg: "foo=1,bar=2",
			expectValue: VModuleConfiguration{
				{
					FilePattern: "foo",
					Verbosity:   1,
				},
				{
					FilePattern: "bar",
					Verbosity:   2,
				},
			},
		},
		{
			arg: "foo=1,bar=2,",
			expectValue: VModuleConfiguration{
				{
					FilePattern: "foo",
					Verbosity:   1,
				},
				{
					FilePattern: "bar",
					Verbosity:   2,
				},
			},
			expectParam: "foo=1,bar=2",
		},
		{
			arg:         "gopher*",
			expectError: `"gopher*" does not have the pattern=N format`,
		},
		{
			arg:         "=1",
			expectError: `"=1" does not have the pattern=N format`,
		},
		{
			arg:         "foo=-1",
			expectError: `parsing verbosity in "foo=-1": strconv.ParseUint: parsing "-1": invalid syntax`,
		},
		{
			arg: fmt.Sprintf("validint32=%d", math.MaxInt32),
			expectValue: VModuleConfiguration{
				{
					FilePattern: "validint32",
					Verbosity:   math.MaxInt32,
				},
			},
		},
		{
			arg:         fmt.Sprintf("invalidint32=%d", math.MaxInt32+1),
			expectError: `parsing verbosity in "invalidint32=2147483648": strconv.ParseUint: parsing "2147483648": value out of range`,
		},
	}

	for _, test := range testcases {
		t.Run(test.arg, func(t *testing.T) {
			var actual VModuleConfiguration
			value := VModuleConfigurationPflag(&actual)
			err := value.Set(test.arg)
			if test.expectError != "" {
				if err == nil {
					t.Fatal("parsing should have failed")
				}
				assert.Equal(t, test.expectError, err.Error(), "parse error")
			} else {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				param := value.String()
				expectParam := test.expectParam
				if expectParam == "" {
					expectParam = test.arg
				}
				assert.Equal(t, expectParam, param, "encoded parameter value not identical")
			}
		})
	}
}

// TestCompatibility ensures that a) valid JSON remains valid and has the same
// effect and b) that new fields are covered by the test data.
func TestCompatibility(t *testing.T) {
	testcases := map[string]struct {
		// fixture holds a representation of a LoggingConfiguration struct in JSON format.
		fixture string
		// baseConfig is the struct that Unmarshal writes into.
		baseConfig LoggingConfiguration
		// expectAllFields enables a reflection check to ensure that the
		// result has all fields set.
		expectAllFields bool
		// expectConfig is the intended result.
		expectConfig LoggingConfiguration
	}{
		"defaults": {
			// No changes when nothing is specified.
			fixture:      "{}",
			baseConfig:   *NewLoggingConfiguration(),
			expectConfig: *NewLoggingConfiguration(),
		},
		"all-fields": {
			// The JSON fixture includes all fields. The result
			// must have all fields as non-empty when starting with
			// an empty base, otherwise the fixture is incomplete
			// and must be updated for the test case to pass.
			fixture: `{
	"format": "json",
	"flushFrequency": 1,
	"verbosity": 5,
	"vmodule": [
		{"filePattern": "someFile", "verbosity": 10},
		{"filePattern": "anotherFile", "verbosity": 1}
	],
	"options": {
		"text": {
			"splitStream": true,
			"infoBufferSize": "2048"
		},
		"json": {
			"splitStream": true,
			"infoBufferSize": "1024"
		}
	}
}
`,
			baseConfig:      LoggingConfiguration{},
			expectAllFields: true,
			expectConfig: LoggingConfiguration{
				Format:         JSONLogFormat,
				FlushFrequency: TimeOrMetaDuration{Duration: metav1.Duration{Duration: time.Nanosecond}},
				Verbosity:      VerbosityLevel(5),
				VModule: VModuleConfiguration{
					{
						FilePattern: "someFile",
						Verbosity:   VerbosityLevel(10),
					},
					{
						FilePattern: "anotherFile",
						Verbosity:   VerbosityLevel(1),
					},
				},
				Options: FormatOptions{
					Text: TextOptions{
						OutputRoutingOptions: OutputRoutingOptions{
							SplitStream: true,
							InfoBufferSize: resource.QuantityValue{
								Quantity: *resource.NewQuantity(2048, resource.DecimalSI),
							},
						},
					},
					JSON: JSONOptions{
						OutputRoutingOptions: OutputRoutingOptions{
							SplitStream: true,
							InfoBufferSize: resource.QuantityValue{
								Quantity: *resource.NewQuantity(1024, resource.DecimalSI),
							},
						},
					},
				},
			},
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			// Beware, not a deep copy. Different test cases must
			// not share anything.
			config := tc.baseConfig
			if strictErr, err := json.UnmarshalStrict([]byte(tc.fixture), &config); err != nil {
				t.Fatalf("unexpected unmarshal error: %v", err)
			} else if strictErr != nil {
				t.Fatalf("unexpected strict unmarshal error: %v", strictErr)
			}
			// This sets the internal "s" field just like unmarshaling does.
			// Required for assert.Equal to pass.
			_ = tc.expectConfig.Options.Text.InfoBufferSize.String()
			_ = tc.expectConfig.Options.JSON.InfoBufferSize.String()
			assert.Equal(t, tc.expectConfig, config)
			if tc.expectAllFields {
				notZeroRecursive(t, config, "LoggingConfiguration")
			}
		})
	}
}

// notZero asserts that i is not the zero value for its type
// and repeats that check recursively for all pointers,
// structs, maps, arrays, and slices.
func notZeroRecursive(t *testing.T, i interface{}, path string) bool {
	typeOfI := reflect.TypeOf(i)

	if i == nil || reflect.DeepEqual(i, reflect.Zero(typeOfI).Interface()) {
		t.Errorf("%s: should not have been zero, but was %v", path, i)
		return false
	}

	valid := true
	kind := typeOfI.Kind()
	value := reflect.ValueOf(i)
	switch kind {
	case reflect.Pointer:
		if !notZeroRecursive(t, value.Elem().Interface(), path) {
			valid = false
		}
	case reflect.Struct:
		for i := 0; i < typeOfI.NumField(); i++ {
			if !typeOfI.Field(i).IsExported() {
				// Cannot access value.
				continue
			}
			if typeOfI.Field(i).Tag.Get("json") == "-" {
				// unserialized field
				continue
			}
			if !notZeroRecursive(t, value.Field(i).Interface(), path+"."+typeOfI.Field(i).Name) {
				valid = false
			}
		}
	case reflect.Map:
		iter := value.MapRange()
		for iter.Next() {
			k := iter.Key()
			v := iter.Value()
			if !notZeroRecursive(t, k.Interface(), path+"."+"<key>") {
				valid = false
			}
			if !notZeroRecursive(t, v.Interface(), path+"["+fmt.Sprintf("%v", k.Interface())+"]") {
				valid = false
			}
		}
	case reflect.Slice, reflect.Array:
		for i := 0; i < value.Len(); i++ {
			if !notZeroRecursive(t, value.Index(i).Interface(), path+"["+fmt.Sprintf("%d", i)+"]") {
				valid = false
			}
		}
	}

	return valid
}

func TestTimeOrMetaDuration_UnmarshalJSON(t *testing.T) {
	tests := []struct {
		name   string
		tomd   *TimeOrMetaDuration
		arg    any
		wanted string
	}{
		{
			name:   "string values unmarshal as metav1.Duration",
			tomd:   &TimeOrMetaDuration{},
			arg:    "1s",
			wanted: `"1s"`,
		}, {
			name:   "int values unmarshal as metav1.Duration",
			tomd:   &TimeOrMetaDuration{},
			arg:    1000000000,
			wanted: `1000000000`,
		}, {
			name:   "invalid value return error",
			tomd:   &TimeOrMetaDuration{},
			arg:    "invalid",
			wanted: "time: invalid duration \"invalid\"",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b, err := enjson.Marshal(tt.arg)
			if err != nil {
				t.Errorf("unexpect error: %v", err)
			}

			if err := tt.tomd.UnmarshalJSON(b); err == nil {
				data, err := tt.tomd.MarshalJSON()
				if err != nil {
					t.Fatal(err)
				}
				if tt.wanted != string(data) {
					t.Errorf("unexpected wanted for %s, wanted: %v, got: %v", tt.name, tt.wanted, string(data))
				}
			} else {
				if err.Error() != tt.wanted {
					t.Errorf("UnmarshalJSON() error = %v", err)
				}
			}
		})
	}

}
