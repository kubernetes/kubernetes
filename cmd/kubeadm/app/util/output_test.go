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

package util

import (
	"encoding/json"
	"fmt"
	"reflect"
	"testing"

	"sigs.k8s.io/yaml"
)

type testData struct {
	Name string
	List []string
}

func (data *testData) Short() string {
	return data.Name
}

func (data *testData) Text() string {
	return fmt.Sprintf("Name: %s, List: %v", data.Name, data.List)
}

func TestConvertToOutputFormat(t *testing.T) {
	tstData := testData{
		Name: "test",
		List: []string{"item 1", "item 2", "item 3"},
	}

	var tests = []struct {
		name           string
		data           interface{}
		outputFormat   string
		expectedOutput string
		expectedErr    bool
	}{
		{
			name:           "to short",
			data:           &tstData,
			outputFormat:   "short",
			expectedOutput: "test",
		},
		{
			name:           "to text",
			data:           &tstData,
			outputFormat:   "text",
			expectedOutput: "Name: test, List: [item 1 item 2 item 3]",
		},
		{
			name:         "to JSON",
			data:         &tstData,
			outputFormat: "json",
		},
		{
			name:         "to YAML",
			data:         &tstData,
			outputFormat: "yaml",
		},
		{
			name:         "invalid data",
			data:         make(chan int),
			outputFormat: "json",
			expectedErr:  true,
		},
		{
			name:         "invalid format",
			data:         &tstData,
			outputFormat: "invalid",
			expectedErr:  true,
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t2 *testing.T) {
			output, err := ConvertToOutputFormat(rt.data, rt.outputFormat)
			if err != nil {
				if !rt.expectedErr {
					t2.Fatalf("unexpected error: %v", err)
				}
				return
			}

			if rt.outputFormat == "json" || rt.outputFormat == "yaml" {
				// unmarshal result back to the struct for validation
				var unmarshaled testData
				if rt.outputFormat == "json" {
					err = json.Unmarshal([]byte(output), &unmarshaled)
				} else if rt.outputFormat == "yaml" {
					err = yaml.Unmarshal([]byte(output), &unmarshaled)
				}
				if err != nil {
					t2.Errorf("unexpected unmarshall error: %v", err)
					return
				}

				if !reflect.DeepEqual(&unmarshaled, rt.data) {
					t2.Errorf("unexpected result: %+v, expected: %+v", &unmarshaled, rt.data)
				}
			} else if output != rt.expectedOutput {
				t2.Errorf("unexpected result: |%s|, expected: |%s|", output, rt.expectedOutput)
			}
		})
	}
}
