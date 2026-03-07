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
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/utils/ptr"
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
	PoolName: ptr.To("pool-1"),
	Limit:    ptr.To(int32(100)),
}

var validPoolStatus = resource.PoolStatus{
	Driver:             "test.example.com",
	PoolName:           "pool-1",
	TotalDevices:       ptr.To(int32(4)),
	AllocatedDevices:   ptr.To(int32(2)),
	AvailableDevices:   ptr.To(int32(2)),
	UnavailableDevices: ptr.To(int32(0)),
	SliceCount:         1,
	Generation:         ptr.To(int64(1)),
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
				PoolName: ptr.To("node-1"),
			}),
		},
		"valid-with-limit": {
			request: testResourcePoolStatusRequest(goodName, resource.ResourcePoolStatusRequestSpec{
				Driver: "test.example.com",
				Limit:  ptr.To(int32(50)),
			}),
		},
		"valid-limit-min": {
			request: testResourcePoolStatusRequest(goodName, resource.ResourcePoolStatusRequestSpec{
				Driver: "test.example.com",
				Limit:  ptr.To(int32(1)),
			}),
		},
		"valid-limit-max": {
			request: testResourcePoolStatusRequest(goodName, resource.ResourcePoolStatusRequestSpec{
				Driver: "test.example.com",
				Limit:  ptr.To(int32(1000)),
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
		"missing-driver": {
			request: testResourcePoolStatusRequest(goodName, resource.ResourcePoolStatusRequestSpec{
				Driver: "",
			}),
			wantFailures: field.ErrorList{field.Required(field.NewPath("spec", "driver"), "driver name is required").MarkCoveredByDeclarative()},
		},
		"invalid-driver-name": {
			request: testResourcePoolStatusRequest(goodName, resource.ResourcePoolStatusRequestSpec{
				Driver: "invalid driver!",
			}),
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec", "driver"), "invalid driver!", dns1123SubdomainErrorMsg).MarkCoveredByDeclarative()},
		},
		"invalid-pool-name": {
			request: testResourcePoolStatusRequest(goodName, resource.ResourcePoolStatusRequestSpec{
				Driver:   "test.example.com",
				PoolName: ptr.To("invalid pool!"),
			}),
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec", "poolName"), "invalid pool!", dns1123SubdomainErrorMsg).MarkCoveredByDeclarative()},
		},
		"empty-pool-name": {
			request: testResourcePoolStatusRequest(goodName, resource.ResourcePoolStatusRequestSpec{
				Driver:   "test.example.com",
				PoolName: ptr.To(""),
			}),
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec", "poolName"), "", "must not be empty when specified").MarkCoveredByDeclarative()},
		},
		"limit-zero": {
			request: testResourcePoolStatusRequest(goodName, resource.ResourcePoolStatusRequestSpec{
				Driver: "test.example.com",
				Limit:  ptr.To(int32(0)),
			}),
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec", "limit"), int32(0), "must be at least 1").MarkCoveredByDeclarative()},
		},
		"limit-negative": {
			request: testResourcePoolStatusRequest(goodName, resource.ResourcePoolStatusRequestSpec{
				Driver: "test.example.com",
				Limit:  ptr.To(int32(-1)),
			}),
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec", "limit"), int32(-1), "must be at least 1").MarkCoveredByDeclarative()},
		},
		"limit-exceeds-max": {
			request: testResourcePoolStatusRequest(goodName, resource.ResourcePoolStatusRequestSpec{
				Driver: "test.example.com",
				Limit:  ptr.To(int32(1001)),
			}),
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec", "limit"), int32(1001), "must not exceed 1000").MarkCoveredByDeclarative()},
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
		"spec-immutable-driver": {
			oldRequest: testResourcePoolStatusRequestForUpdate(goodName, resource.ResourcePoolStatusRequestSpec{
				Driver: "old.example.com",
			}),
			newRequest: testResourcePoolStatusRequestForUpdate(goodName, resource.ResourcePoolStatusRequestSpec{
				Driver: "new.example.com",
			}),
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec"), resource.ResourcePoolStatusRequestSpec{Driver: "new.example.com"}, "field is immutable").MarkCoveredByDeclarative(),
			},
		},
		"spec-immutable-pool-name": {
			oldRequest: testResourcePoolStatusRequestForUpdate(goodName, resource.ResourcePoolStatusRequestSpec{
				Driver:   "test.example.com",
				PoolName: ptr.To("old-pool"),
			}),
			newRequest: testResourcePoolStatusRequestForUpdate(goodName, resource.ResourcePoolStatusRequestSpec{
				Driver:   "test.example.com",
				PoolName: ptr.To("new-pool"),
			}),
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec"), resource.ResourcePoolStatusRequestSpec{Driver: "test.example.com", PoolName: ptr.To("new-pool")}, "field is immutable").MarkCoveredByDeclarative(),
			},
		},
		"spec-immutable-limit": {
			oldRequest: testResourcePoolStatusRequestForUpdate(goodName, resource.ResourcePoolStatusRequestSpec{
				Driver: "test.example.com",
				Limit:  ptr.To(int32(50)),
			}),
			newRequest: testResourcePoolStatusRequestForUpdate(goodName, resource.ResourcePoolStatusRequestSpec{
				Driver: "test.example.com",
				Limit:  ptr.To(int32(100)),
			}),
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec"), resource.ResourcePoolStatusRequestSpec{Driver: "test.example.com", Limit: ptr.To(int32(100))}, "field is immutable").MarkCoveredByDeclarative(),
			},
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
					ObservationTime:    &now,
					TotalMatchingPools: ptr.To(int32(2)),
					Pools: []resource.PoolStatus{
						{
							Driver:             "test.example.com",
							PoolName:           "pool-1",
							NodeName:           ptr.To("node-1"),
							TotalDevices:       ptr.To(int32(4)),
							AllocatedDevices:   ptr.To(int32(2)),
							AvailableDevices:   ptr.To(int32(2)),
							UnavailableDevices: ptr.To(int32(0)),
							SliceCount:         1,
							Generation:         ptr.To(int64(1)),
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
					ObservationTime:    &now,
					TotalMatchingPools: ptr.To(int32(2)),
				}
				return r
			}(),
			newRequest: func() *resource.ResourcePoolStatusRequest {
				r := baseRequest()
				r.Status = &resource.ResourcePoolStatusRequestStatus{
					ObservationTime:    &now,
					TotalMatchingPools: ptr.To(int32(5)), // changed
				}
				return r
			}(),
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("status"),
					&resource.ResourcePoolStatusRequestStatus{ObservationTime: &now, TotalMatchingPools: ptr.To(int32(5))},
					"field is immutable"),
			},
		},
		"invalid-pool-missing-driver": {
			oldRequest: baseRequest(),
			newRequest: func() *resource.ResourcePoolStatusRequest {
				r := baseRequest()
				pool := *validPoolStatus.DeepCopy()
				pool.Driver = ""
				r.Status = &resource.ResourcePoolStatusRequestStatus{
					Pools: []resource.PoolStatus{pool},
				}
				return r
			}(),
			wantFailures: field.ErrorList{field.Required(field.NewPath("status", "pools").Index(0).Child("driver"), "").MarkCoveredByDeclarative()},
		},
		"invalid-pool-missing-pool-name": {
			oldRequest: baseRequest(),
			newRequest: func() *resource.ResourcePoolStatusRequest {
				r := baseRequest()
				pool := *validPoolStatus.DeepCopy()
				pool.PoolName = ""
				r.Status = &resource.ResourcePoolStatusRequestStatus{
					Pools: []resource.PoolStatus{pool},
				}
				return r
			}(),
			wantFailures: field.ErrorList{field.Required(field.NewPath("status", "pools").Index(0).Child("poolName"), "").MarkCoveredByDeclarative()},
		},
		"invalid-pool-negative-total-devices": {
			oldRequest: baseRequest(),
			newRequest: func() *resource.ResourcePoolStatusRequest {
				r := baseRequest()
				pool := *validPoolStatus.DeepCopy()
				pool.TotalDevices = ptr.To(int32(-1))
				r.Status = &resource.ResourcePoolStatusRequestStatus{
					Pools: []resource.PoolStatus{pool},
				}
				return r
			}(),
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("status", "pools").Index(0).Child("totalDevices"), int32(-1), "must be non-negative").MarkCoveredByDeclarative()},
		},
		"invalid-pool-negative-allocated-devices": {
			oldRequest: baseRequest(),
			newRequest: func() *resource.ResourcePoolStatusRequest {
				r := baseRequest()
				pool := *validPoolStatus.DeepCopy()
				pool.AllocatedDevices = ptr.To(int32(-1))
				r.Status = &resource.ResourcePoolStatusRequestStatus{
					Pools: []resource.PoolStatus{pool},
				}
				return r
			}(),
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("status", "pools").Index(0).Child("allocatedDevices"), int32(-1), "must be non-negative").MarkCoveredByDeclarative()},
		},
		"invalid-pool-zero-slice-count": {
			oldRequest: baseRequest(),
			newRequest: func() *resource.ResourcePoolStatusRequest {
				r := baseRequest()
				pool := *validPoolStatus.DeepCopy()
				pool.SliceCount = 0
				r.Status = &resource.ResourcePoolStatusRequestStatus{
					Pools: []resource.PoolStatus{pool},
				}
				return r
			}(),
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("status", "pools").Index(0).Child("sliceCount"), int32(0), "must be at least 1").MarkCoveredByDeclarative()},
		},
		"invalid-pool-negative-generation": {
			oldRequest: baseRequest(),
			newRequest: func() *resource.ResourcePoolStatusRequest {
				r := baseRequest()
				pool := *validPoolStatus.DeepCopy()
				pool.Generation = ptr.To(int64(-1))
				r.Status = &resource.ResourcePoolStatusRequestStatus{
					Pools: []resource.PoolStatus{pool},
				}
				return r
			}(),
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("status", "pools").Index(0).Child("generation"), int64(-1), "must be non-negative").MarkCoveredByDeclarative()},
		},
		"invalid-negative-total-matching-pools": {
			oldRequest: baseRequest(),
			newRequest: func() *resource.ResourcePoolStatusRequest {
				r := baseRequest()
				r.Status = &resource.ResourcePoolStatusRequestStatus{
					TotalMatchingPools: ptr.To(int32(-1)),
				}
				return r
			}(),
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("status", "totalMatchingPools"), int32(-1), "must be non-negative").MarkCoveredByDeclarative()},
		},
		"too-many-validation-errors": {
			oldRequest: baseRequest(),
			newRequest: func() *resource.ResourcePoolStatusRequest {
				r := baseRequest()
				errors := make([]string, 11)
				for i := range errors {
					errors[i] = "error"
				}
				r.Status = &resource.ResourcePoolStatusRequestStatus{
					ValidationErrors: errors,
				}
				return r
			}(),
			wantFailures: field.ErrorList{field.TooMany(field.NewPath("status", "validationErrors"), 11, 10).MarkCoveredByDeclarative()},
		},
		"validation-error-too-long": {
			oldRequest: baseRequest(),
			newRequest: func() *resource.ResourcePoolStatusRequest {
				r := baseRequest()
				r.Status = &resource.ResourcePoolStatusRequestStatus{
					ValidationErrors: []string{strings.Repeat("x", 257)},
				}
				return r
			}(),
			wantFailures: field.ErrorList{field.TooLong(field.NewPath("status", "validationErrors").Index(0), strings.Repeat("x", 257), 256).MarkCoveredByDeclarative()},
		},
		"invalid-pool-nil-required-fields": {
			oldRequest: baseRequest(),
			newRequest: func() *resource.ResourcePoolStatusRequest {
				r := baseRequest()
				r.Status = &resource.ResourcePoolStatusRequestStatus{
					Pools: []resource.PoolStatus{
						{
							Driver:     "test.example.com",
							PoolName:   "pool-1",
							SliceCount: 1,
							// All required pointer fields are nil
						},
					},
				}
				return r
			}(),
			wantFailures: field.ErrorList{
				field.Required(field.NewPath("status", "pools").Index(0).Child("totalDevices"), "").MarkCoveredByDeclarative(),
				field.Required(field.NewPath("status", "pools").Index(0).Child("allocatedDevices"), "").MarkCoveredByDeclarative(),
				field.Required(field.NewPath("status", "pools").Index(0).Child("availableDevices"), "").MarkCoveredByDeclarative(),
				field.Required(field.NewPath("status", "pools").Index(0).Child("unavailableDevices"), "").MarkCoveredByDeclarative(),
				field.Required(field.NewPath("status", "pools").Index(0).Child("generation"), "").MarkCoveredByDeclarative(),
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
