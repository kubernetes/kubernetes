/*
Copyright 2021 The Kubernetes Authors.

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
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
)

// Test_ServiceSpecRemovedFieldProtobufNumberReservation tests that the reserved protobuf field numbers
// for removed fields are not re-used. DO NOT remove this test for any reason, this ensures that tombstoned
// protobuf field numbers are not accidentally reused by other fields.
func Test_ServiceSpecRemovedFieldProtobufNumberReservation(t *testing.T) {
	obj := reflect.ValueOf(ServiceSpec{}).Type()
	for i := 0; i < obj.NumField(); i++ {
		f := obj.Field(i)

		protobufNum := strings.Split(f.Tag.Get("protobuf"), ",")[1]
		if protobufNum == "15" {
			t.Errorf("protobuf 15 in ServiceSpec is reserved for removed ipFamily field")
		}
		if protobufNum == "16" {
			t.Errorf("protobuf 16 in ServiceSpec is reserved for removed topologyKeys field")
		}
	}
}

// TestEphemeralContainer ensures that the tags of Container and EphemeralContainerCommon are kept in sync.
func TestEphemeralContainer(t *testing.T) {
	ephemeralType := reflect.TypeOf(EphemeralContainerCommon{})
	containerType := reflect.TypeOf(Container{})

	ephemeralFields := ephemeralType.NumField()
	containerFields := containerType.NumField()
	if containerFields != ephemeralFields {
		t.Fatalf("%v has %d fields, %v has %d fields", ephemeralType, ephemeralFields, containerType, containerFields)
	}
	for i := 0; i < ephemeralFields; i++ {
		ephemeralField := ephemeralType.Field(i)
		containerField := containerType.Field(i)
		if !reflect.DeepEqual(ephemeralField, containerField) {
			t.Errorf("field %v differs:\n\t%#v\n\t%#v", ephemeralField.Name, ephemeralField, containerField)
		}
	}
}

func TestResourceListUnmarshalJSON(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected ResourceList
	}{
		{
			name:  "null value is skipped",
			input: `{"cpu": null, "memory": "128Mi"}`,
			expected: ResourceList{
				ResourceMemory: resource.MustParse("128Mi"),
			},
		},
		{
			name:  "all null values",
			input: `{"cpu": null, "memory": null}`,
			expected: ResourceList{},
		},
		{
			name:  "no null values",
			input: `{"cpu": "100m", "memory": "128Mi"}`,
			expected: ResourceList{
				ResourceCPU:    resource.MustParse("100m"),
				ResourceMemory: resource.MustParse("128Mi"),
			},
		},
		{
			name:     "null object",
			input:    `null`,
			expected: nil,
		},
		{
			name:     "empty object",
			input:    `{}`,
			expected: ResourceList{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var rl ResourceList
			if err := json.Unmarshal([]byte(tt.input), &rl); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if tt.expected == nil {
				if rl != nil {
					t.Errorf("expected nil, got %v", rl)
				}
				return
			}
			if len(rl) != len(tt.expected) {
				t.Errorf("expected %d entries, got %d: %v", len(tt.expected), len(rl), rl)
				return
			}
			for k, v := range tt.expected {
				got, ok := rl[k]
				if !ok {
					t.Errorf("missing key %q", k)
					continue
				}
				if !got.Equal(v) {
					t.Errorf("key %q: expected %v, got %v", k, v, got)
				}
			}
		})
	}
}

func TestNoBindingDeprecation(t *testing.T) {
	var binding any = new(Binding)
	if _, ok := binding.(interface {
		APILifecycleDeprecated(major, minor int)
	}); ok {
		t.Fatal("The Binding type must not marked as deprecated, it is still used for the binding sub-resource which is not deprecated.")
	}
}
