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

package validation

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/resource"
)

const (
	dns1123SubdomainErrorMsg = "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"
)

func testResourcePoolStatusRequest(name string, spec resource.ResourcePoolStatusRequestSpec) *resource.ResourcePoolStatusRequest {
	return &resource.ResourcePoolStatusRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: *spec.DeepCopy(),
	}
}

func testResourcePoolStatusRequestForUpdate(name string, spec resource.ResourcePoolStatusRequestSpec) *resource.ResourcePoolStatusRequest {
	return &resource.ResourcePoolStatusRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name:            name,
			ResourceVersion: "1",
		},
		Spec: *spec.DeepCopy(),
	}
}

var validResourcePoolStatusRequestSpec = resource.ResourcePoolStatusRequestSpec{
	Driver:   "test.example.com",
	PoolName: new("pool-1"),
	Limit:    new(int32(100)),
}

func TestValidateResourcePoolStatusRequest(t *testing.T) {
	goodName := "my-request"
	badName := "!@#$%^"

	scenarios := map[string]struct {
		request      *resource.ResourcePoolStatusRequest
		wantFailures field.ErrorList
	}{
		"valid-request": {
			request: testResourcePoolStatusRequest(goodName, validResourcePoolStatusRequestSpec),
		},
		"valid-driver-only": {
			request: testResourcePoolStatusRequest(goodName, resource.ResourcePoolStatusRequestSpec{
				Driver: "test.example.com",
			}),
		},
		"valid-with-pool-name": {
			request: testResourcePoolStatusRequest(goodName, resource.ResourcePoolStatusRequestSpec{
				Driver:   "test.example.com",
				PoolName: new("node-1"),
			}),
		},
		"valid-with-limit": {
			request: testResourcePoolStatusRequest(goodName, resource.ResourcePoolStatusRequestSpec{
				Driver: "test.example.com",
				Limit:  new(int32(50)),
			}),
		},
		"valid-limit-min": {
			request: testResourcePoolStatusRequest(goodName, resource.ResourcePoolStatusRequestSpec{
				Driver: "test.example.com",
				Limit:  new(int32(1)),
			}),
		},
		"valid-limit-max": {
			request: testResourcePoolStatusRequest(goodName, resource.ResourcePoolStatusRequestSpec{
				Driver: "test.example.com",
				Limit:  new(int32(1000)),
			}),
		},
		"missing-name": {
			request:      testResourcePoolStatusRequest("", validResourcePoolStatusRequestSpec),
			wantFailures: field.ErrorList{field.Required(field.NewPath("metadata", "name"), "name or generateName is required")},
		},
		"bad-name": {
			request:      testResourcePoolStatusRequest(badName, validResourcePoolStatusRequestSpec),
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "name"), badName, dns1123SubdomainErrorMsg)},
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			errs := ValidateResourcePoolStatusRequest(scenario.request)
			assertFailures(t, scenario.wantFailures, errs)
		})
	}
}

func TestValidateResourcePoolStatusRequestUpdate(t *testing.T) {
	goodName := "my-request"

	scenarios := map[string]struct {
		oldRequest   *resource.ResourcePoolStatusRequest
		newRequest   *resource.ResourcePoolStatusRequest
		wantFailures field.ErrorList
	}{
		"no-change": {
			oldRequest: testResourcePoolStatusRequestForUpdate(goodName, validResourcePoolStatusRequestSpec),
			newRequest: testResourcePoolStatusRequestForUpdate(goodName, validResourcePoolStatusRequestSpec),
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			errs := ValidateResourcePoolStatusRequestUpdate(scenario.newRequest, scenario.oldRequest)
			assertFailures(t, scenario.wantFailures, errs)
		})
	}
}

func TestValidateResourcePoolStatusRequestStatusUpdate(t *testing.T) {
	goodName := "my-request"
	now := metav1.Now()

	baseRequest := func() *resource.ResourcePoolStatusRequest {
		r := testResourcePoolStatusRequestForUpdate(goodName, validResourcePoolStatusRequestSpec)
		return r
	}

	scenarios := map[string]struct {
		oldRequest   *resource.ResourcePoolStatusRequest
		newRequest   *resource.ResourcePoolStatusRequest
		wantFailures field.ErrorList
	}{
		"valid-initial-status-update": {
			oldRequest: baseRequest(),
			newRequest: func() *resource.ResourcePoolStatusRequest {
				r := baseRequest()
				r.Status = &resource.ResourcePoolStatusRequestStatus{
					PoolCount: new(int32(2)),
					Pools: []resource.PoolStatus{
						{
							Driver:             "test.example.com",
							PoolName:           "pool-1",
							NodeName:           new("node-1"),
							TotalDevices:       new(int32(4)),
							AllocatedDevices:   new(int32(2)),
							AvailableDevices:   new(int32(2)),
							UnavailableDevices: new(int32(0)),
							ResourceSliceCount: new(int32(1)),
							Generation:         1,
						},
					},
					Conditions: []metav1.Condition{
						{
							Type:               "Complete",
							Status:             metav1.ConditionTrue,
							LastTransitionTime: now,
							Reason:             "Calculated",
						},
					},
				}
				return r
			}(),
		},
		"valid-pool-with-validation-error": {
			oldRequest: baseRequest(),
			newRequest: func() *resource.ResourcePoolStatusRequest {
				r := baseRequest()
				validationErr := "pool data is inconsistent"
				r.Status = &resource.ResourcePoolStatusRequestStatus{
					PoolCount: new(int32(1)),
					Pools: []resource.PoolStatus{
						{
							Driver:          "test.example.com",
							PoolName:        "pool-1",
							Generation:      1,
							ValidationError: &validationErr,
						},
					},
					Conditions: []metav1.Condition{
						{
							Type:               "Complete",
							Status:             metav1.ConditionTrue,
							LastTransitionTime: now,
							Reason:             "Calculated",
						},
					},
				}
				return r
			}(),
		},
		"status-immutable-after-status-set": {
			oldRequest: func() *resource.ResourcePoolStatusRequest {
				r := baseRequest()
				r.Status = &resource.ResourcePoolStatusRequestStatus{
					PoolCount: new(int32(2)),
				}
				return r
			}(),
			newRequest: func() *resource.ResourcePoolStatusRequest {
				r := baseRequest()
				r.Status = &resource.ResourcePoolStatusRequestStatus{
					PoolCount: new(int32(5)), // changed
				}
				return r
			}(),
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("status"),
					&resource.ResourcePoolStatusRequestStatus{PoolCount: new(int32(5))},
					"field is immutable"),
			},
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			errs := ValidateResourcePoolStatusRequestStatusUpdate(scenario.newRequest, scenario.oldRequest)
			assertFailures(t, scenario.wantFailures, errs)
		})
	}
}

// new is a generic helper to create a pointer to a value.
func new[T any](v T) *T {
	return &v
}
