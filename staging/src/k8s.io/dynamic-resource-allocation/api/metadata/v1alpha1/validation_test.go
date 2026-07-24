/*
Copyright The Kubernetes Authors.

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

package v1alpha1

import (
	"context"
	"fmt"
	"strings"
	"testing"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/operation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestValidateDeviceMetadata(t *testing.T) {
	podClaimName := "claim"
	stringValue := "LATEST-GPU-MODEL"
	valid := DeviceMetadata{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "claim",
			Namespace:  "default",
			Generation: 1,
		},
		PodClaimName: &podClaimName,
		Requests: []DeviceMetadataRequest{{
			Name: "request/subrequest",
			Devices: []Device{{
				Driver: "gpu.example.com",
				Pool:   "worker-0",
				Name:   "gpu-0",
				Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"model": {StringValue: &stringValue},
				},
			}},
		}},
	}

	tooManyAttributes := make(map[resourceapi.QualifiedName]resourceapi.DeviceAttribute, 33)
	for i := range 33 {
		tooManyAttributes[resourceapi.QualifiedName(fmt.Sprintf("attribute%d", i))] = resourceapi.DeviceAttribute{
			StringValue: &stringValue,
		}
	}

	testcases := map[string]struct {
		mutate    func(*DeviceMetadata)
		wantField string
	}{
		"valid": {},
		"invalid pod claim name": {
			mutate: func(metadata *DeviceMetadata) {
				invalidName := "INVALID"
				metadata.PodClaimName = &invalidName
			},
			wantField: "podClaimName",
		},
		"request name required": {
			mutate: func(metadata *DeviceMetadata) {
				metadata.Requests[0].Name = ""
			},
			wantField: "requests[0].name",
		},
		"driver required": {
			mutate: func(metadata *DeviceMetadata) {
				metadata.Requests[0].Devices[0].Driver = ""
			},
			wantField: "requests[0].devices[0].driver",
		},
		"driver too long": {
			mutate: func(metadata *DeviceMetadata) {
				metadata.Requests[0].Devices[0].Driver = strings.Repeat("a", 64)
			},
			wantField: "requests[0].devices[0].driver",
		},
		"invalid pool": {
			mutate: func(metadata *DeviceMetadata) {
				metadata.Requests[0].Devices[0].Pool = "worker-0/"
			},
			wantField: "requests[0].devices[0].pool",
		},
		"invalid device name": {
			mutate: func(metadata *DeviceMetadata) {
				metadata.Requests[0].Devices[0].Name = "GPU-0"
			},
			wantField: "requests[0].devices[0].name",
		},
		"too many attributes": {
			mutate: func(metadata *DeviceMetadata) {
				metadata.Requests[0].Devices[0].Attributes = tooManyAttributes
			},
			wantField: "requests[0].devices[0].attributes",
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			metadata := valid.DeepCopy()
			if tc.mutate != nil {
				tc.mutate(metadata)
			}

			errs := Validate_DeviceMetadata(context.Background(), operation.Operation{}, nil, metadata, nil)
			if tc.wantField == "" {
				if len(errs) != 0 {
					t.Fatalf("expected no validation errors, got %v", errs)
				}
				return
			}
			if len(errs) != 1 {
				t.Fatalf("expected one validation error for %q, got %v", tc.wantField, errs)
			}
			if errs[0].Field != tc.wantField {
				t.Errorf("expected validation error for %q, got %q", tc.wantField, errs[0].Field)
			}
		})
	}
}
