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
	"fmt"
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

func TestNoBindingDeprecation(t *testing.T) {
	var binding any = new(Binding)
	if _, ok := binding.(interface {
		APILifecycleDeprecated(major, minor int)
	}); ok {
		t.Fatal("The Binding type must not marked as deprecated, it is still used for the binding sub-resource which is not deprecated.")
	}
}

// TestResourceListUnmarshalJSON verifies that null values in ResourceList are properly
// handled by being omitted from the map rather than being converted to zero quantities.
// This ensures consistency between kubectl create and apply, and proper Server-Side Apply
// field ownership semantics. See: https://github.com/kubernetes/kubernetes/issues/135423
func TestResourceListUnmarshalJSON(t *testing.T) {
	tests := []struct {
		name     string
		json     string
		expected ResourceList
		wantErr  bool
	}{
		{
			name:     "null value should be omitted",
			json:     `{"cpu": null}`,
			expected: ResourceList{},
			wantErr:  false,
		},
		{
			name: "mixed null and valid values",
			json: `{"cpu": "100m", "memory": null}`,
			expected: ResourceList{
				ResourceCPU: resource.MustParse("100m"),
			},
			wantErr: false,
		},
		{
			name: "all valid values",
			json: `{"cpu": "100m", "memory": "256Mi"}`,
			expected: ResourceList{
				ResourceCPU:    resource.MustParse("100m"),
				ResourceMemory: resource.MustParse("256Mi"),
			},
			wantErr: false,
		},
		{
			name:     "empty object",
			json:     `{}`,
			expected: ResourceList{},
			wantErr:  false,
		},
		{
			name:     "all null values",
			json:     `{"cpu": null, "memory": null, "ephemeral-storage": null}`,
			expected: ResourceList{},
			wantErr:  false,
		},
		{
			name:    "invalid quantity",
			json:    `{"cpu": "invalid"}`,
			wantErr: true,
		},
		{
			name:    "invalid json",
			json:    `{cpu: 100m}`,
			wantErr: true,
		},
		{
			name: "complex quantities with null",
			json: `{"cpu": "2.5", "memory": "1Gi", "ephemeral-storage": null, "hugepages-2Mi": "512Mi"}`,
			expected: ResourceList{
				ResourceCPU:              resource.MustParse("2.5"),
				ResourceMemory:           resource.MustParse("1Gi"),
				ResourceName("hugepages-2Mi"): resource.MustParse("512Mi"),
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var rl ResourceList
			err := json.Unmarshal([]byte(tt.json), &rl)

			if (err != nil) != tt.wantErr {
				t.Errorf("UnmarshalJSON() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				if !reflect.DeepEqual(rl, tt.expected) {
					t.Errorf("UnmarshalJSON() = %v, want %v", rl, tt.expected)
				}

				// Verify that null values are truly omitted (not present in map)
				if strings.Contains(tt.json, "null") {
					// Check that keys with null values don't exist in the result
					for key := range rl {
						keyJSON := fmt.Sprintf(`"%s"`, key)
						if strings.Contains(tt.json, keyJSON+`: null`) {
							t.Errorf("Key %q with null value should not be in ResourceList, but was found", key)
						}
					}
				}
			}
		})
	}
}

// TestResourceListUnmarshalJSON_PodSpec tests ResourceList unmarshalling in the context
// of a complete Pod spec to ensure it works correctly with real-world usage.
func TestResourceListUnmarshalJSON_PodSpec(t *testing.T) {
	// This is the exact scenario from issue #135423
	podJSON := `{
		"apiVersion": "v1",
		"kind": "Pod",
		"metadata": {
			"name": "test-pod"
		},
		"spec": {
			"containers": [{
				"name": "test",
				"image": "nginx",
				"resources": {
					"requests": {
						"cpu": "100m"
					},
					"limits": {
						"cpu": null,
						"memory": "256Mi"
					}
				}
			}]
		}
	}`

	var pod Pod
	err := json.Unmarshal([]byte(podJSON), &pod)
	if err != nil {
		t.Fatalf("Failed to unmarshal Pod: %v", err)
	}

	// Verify that limits map doesn't contain cpu (null should omit it)
	limits := pod.Spec.Containers[0].Resources.Limits
	if len(limits) != 1 {
		t.Errorf("Expected 1 limit (memory only), got %d: %v", len(limits), limits)
	}

	if _, hasCPU := limits[ResourceCPU]; hasCPU {
		t.Errorf("CPU limit should not be in map (was null), but found: %v", limits[ResourceCPU])
	}

	expectedMemory := resource.MustParse("256Mi")
	if memory, ok := limits[ResourceMemory]; !ok {
		t.Error("Memory limit should be in map")
	} else if memory.Cmp(expectedMemory) != 0 {
		t.Errorf("Memory limit = %v, want %v", memory, expectedMemory)
	}

	// Verify that requests map has cpu
	requests := pod.Spec.Containers[0].Resources.Requests
	if len(requests) != 1 {
		t.Errorf("Expected 1 request (cpu only), got %d: %v", len(requests), requests)
	}

	expectedCPU := resource.MustParse("100m")
	if cpu, ok := requests[ResourceCPU]; !ok {
		t.Error("CPU request should be in map")
	} else if cpu.Cmp(expectedCPU) != 0 {
		t.Errorf("CPU request = %v, want %v", cpu, expectedCPU)
	}
}

// TestResourceListUnmarshalJSON_RoundTrip verifies that marshal/unmarshal is consistent
func TestResourceListUnmarshalJSON_RoundTrip(t *testing.T) {
	original := ResourceList{
		ResourceCPU:    resource.MustParse("100m"),
		ResourceMemory: resource.MustParse("256Mi"),
	}

	// Marshal
	data, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}

	// Unmarshal
	var result ResourceList
	if err := json.Unmarshal(data, &result); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}

	// Compare
	if !reflect.DeepEqual(original, result) {
		t.Errorf("Round-trip failed: original = %v, result = %v", original, result)
	}
}
