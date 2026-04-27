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

package resourcepoolstatusrequest

import (
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/utils/ptr"

	// Ensure all API groups are registered with the scheme
	_ "k8s.io/kubernetes/pkg/apis/resource/install"
)

func TestDeclarativeValidate(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "resource.k8s.io",
		APIVersion:        "v1alpha3",
		Resource:          "resourcepoolstatusrequests",
		IsResourceRequest: true,
		Verb:              "create",
	})

	testCases := map[string]struct {
		input        resource.ResourcePoolStatusRequest
		expectedErrs field.ErrorList
	}{
		"valid": {
			input: mkValidRPSR(),
		},
		"empty driver": {
			// +k8s:required (standard DV)
			input:        mkValidRPSR(setDriver("")),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("spec", "driver"), "")},
		},
		"invalid driver format": {
			// +k8s:format=k8s-long-name-caseless (standard DV)
			input:        mkValidRPSR(setDriver("invalid driver!")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "driver"), nil, "").WithOrigin("format=k8s-long-name-caseless")},
		},
		"empty pool name": {
			// +k8s:format=k8s-resource-pool-name (standard DV)
			input:        mkValidRPSR(setPoolName("")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "poolName"), nil, "").WithOrigin("format=k8s-resource-pool-name")},
		},
		"invalid pool name format": {
			// +k8s:format=k8s-resource-pool-name (standard DV)
			input:        mkValidRPSR(setPoolName("invalid pool!")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "poolName"), nil, "").WithOrigin("format=k8s-resource-pool-name")},
		},
		"limit zero": {
			// +k8s:minimum=1 (standard DV)
			input:        mkValidRPSR(setLimit(0)),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "limit"), nil, "").WithOrigin("minimum")},
		},
		"limit negative": {
			// +k8s:minimum=1 (standard DV)
			input:        mkValidRPSR(setLimit(-1)),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "limit"), nil, "").WithOrigin("minimum")},
		},
		"limit too high": {
			// +k8s:maximum=1000 (standard DV)
			input:        mkValidRPSR(setLimit(1001)),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "limit"), nil, "").WithOrigin("maximum")},
		},
		"pool name nil": {
			// PoolName is optional, nil should be valid
			input: mkValidRPSR(func(r *resource.ResourcePoolStatusRequest) {
				r.Spec.PoolName = nil
			}),
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "apis",
		APIGroup:          "resource.k8s.io",
		APIVersion:        "v1alpha3",
		Resource:          "resourcepoolstatusrequests",
		Name:              "valid-rpsr",
		IsResourceRequest: true,
		Verb:              "update",
	})

	testCases := map[string]struct {
		oldObj       resource.ResourcePoolStatusRequest
		updateObj    resource.ResourcePoolStatusRequest
		expectedErrs field.ErrorList
	}{
		"valid no-change": {
			oldObj:    mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(),
		},
		"spec changed": {
			// +k8s:immutable
			oldObj:    mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(setDriver("other.example.com")),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec"), nil, "field is immutable").WithOrigin("immutable"),
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

func TestDeclarativeValidateStatusUpdate(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "apis",
		APIGroup:          "resource.k8s.io",
		APIVersion:        "v1alpha3",
		Resource:          "resourcepoolstatusrequests",
		Name:              "valid-rpsr",
		IsResourceRequest: true,
		Verb:              "update",
		Subresource:       "status",
	})

	testCases := map[string]struct {
		oldObj       resource.ResourcePoolStatusRequest
		updateObj    resource.ResourcePoolStatusRequest
		expectedErrs field.ErrorList
	}{
		"valid status": {
			oldObj:    mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withValidStatus()),
		},
		"pool empty driver": {
			// +k8s:required (standard DV)
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				pool := mkValidPoolStatus()
				pool.Driver = ""
				s.Pools = []resource.PoolStatus{pool}
			})),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("status", "pools").Index(0).Child("driver"), "")},
		},
		"pool invalid driver": {
			// +k8s:format=k8s-long-name-caseless (standard DV)
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				pool := mkValidPoolStatus()
				pool.Driver = "invalid driver!"
				s.Pools = []resource.PoolStatus{pool}
			})),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("status", "pools").Index(0).Child("driver"), nil, "").WithOrigin("format=k8s-long-name-caseless")},
		},
		"pool empty poolName": {
			// +k8s:required (standard DV)
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				pool := mkValidPoolStatus()
				pool.PoolName = ""
				s.Pools = []resource.PoolStatus{pool}
			})),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("status", "pools").Index(0).Child("poolName"), "")},
		},
		"pool invalid poolName": {
			// +k8s:format=k8s-resource-pool-name (standard DV)
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				pool := mkValidPoolStatus()
				pool.PoolName = "invalid pool!"
				s.Pools = []resource.PoolStatus{pool}
			})),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("status", "pools").Index(0).Child("poolName"), nil, "").WithOrigin("format=k8s-resource-pool-name")},
		},
		"pool nil optional fields valid": {
			// Device count fields are optional (may be unset when validationError is set)
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				s.Pools = []resource.PoolStatus{
					{
						Driver:             "test.example.com",
						PoolName:           "pool-1",
						ResourceSliceCount: ptr.To[int32](1),
						Generation:         1,
						// All optional pointer fields are nil - this is valid
					},
				}
			})),
		},
		"pool negative values": {
			// +k8s:minimum=0 (standard DV)
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				pool := mkValidPoolStatus()
				pool.TotalDevices = ptr.To[int32](-1)
				s.Pools = []resource.PoolStatus{pool}
			})),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("status", "pools").Index(0).Child("totalDevices"), nil, "").WithOrigin("minimum")},
		},
		"pool zero resourceSliceCount": {
			// +k8s:minimum=1 (standard DV)
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				pool := mkValidPoolStatus()
				pool.ResourceSliceCount = ptr.To[int32](0)
				s.Pools = []resource.PoolStatus{pool}
			})),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("status", "pools").Index(0).Child("resourceSliceCount"), nil, "").WithOrigin("minimum")},
		},
		"negative poolCount": {
			// +k8s:minimum=0 (standard DV)
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				s.PoolCount = ptr.To[int32](-1)
			})),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("status", "poolCount"), nil, "").WithOrigin("minimum")},
		},
		"pool validationError too long": {
			// +k8s:maxBytes=256
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				pool := mkValidPoolStatus()
				longErr := strings.Repeat("x", 257)
				pool.ValidationError = &longErr
				s.Pools = []resource.PoolStatus{pool}
			})),
			expectedErrs: field.ErrorList{field.TooLong(field.NewPath("status", "pools").Index(0).Child("validationError"), "", 256).WithOrigin("maxBytes")},
		},
		"poolCount zero": {
			// +k8s:minimum=0 boundary — zero should be valid
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				s.PoolCount = ptr.To[int32](0)
			})),
		},
		"pools maxItems exceeded": {
			// +k8s:maxItems=1000
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				s.PoolCount = ptr.To[int32](1001)
				pools := make([]resource.PoolStatus, 1001)
				for i := range pools {
					pools[i] = mkValidPoolStatus()
				}
				s.Pools = pools
			})),
			expectedErrs: field.ErrorList{field.TooMany(field.NewPath("status", "pools"), 1001, 1000).WithOrigin("maxItems")},
		},
		"conditions maxItems exceeded": {
			// +k8s:maxItems=10
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				now := metav1.Now()
				conditions := make([]metav1.Condition, 11)
				for i := range conditions {
					conditions[i] = metav1.Condition{
						Type:               "Type" + strings.Repeat("x", i),
						Status:             metav1.ConditionTrue,
						LastTransitionTime: now,
						Reason:             "Reason",
					}
				}
				s.Conditions = conditions
			})),
			expectedErrs: field.ErrorList{field.TooMany(field.NewPath("status", "conditions"), 11, 10).WithOrigin("maxItems")},
		},
		"pool negative allocatedDevices": {
			// +k8s:minimum=0 (standard DV)
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				pool := mkValidPoolStatus()
				pool.AllocatedDevices = ptr.To[int32](-1)
				s.Pools = []resource.PoolStatus{pool}
			})),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("status", "pools").Index(0).Child("allocatedDevices"), nil, "").WithOrigin("minimum")},
		},
		"pool negative availableDevices": {
			// +k8s:minimum=0 (standard DV)
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				pool := mkValidPoolStatus()
				pool.AvailableDevices = ptr.To[int32](-1)
				s.Pools = []resource.PoolStatus{pool}
			})),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("status", "pools").Index(0).Child("availableDevices"), nil, "").WithOrigin("minimum")},
		},
		"pool negative unavailableDevices": {
			// +k8s:minimum=0 (standard DV)
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				pool := mkValidPoolStatus()
				pool.UnavailableDevices = ptr.To[int32](-1)
				s.Pools = []resource.PoolStatus{pool}
			})),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("status", "pools").Index(0).Child("unavailableDevices"), nil, "").WithOrigin("minimum")},
		},
		"pool invalid nodeName": {
			// +k8s:format=k8s-long-name (standard DV)
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				pool := mkValidPoolStatus()
				pool.NodeName = ptr.To("invalid node!")
				s.Pools = []resource.PoolStatus{pool}
			})),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("status", "pools").Index(0).Child("nodeName"), nil, "").WithOrigin("format=k8s-long-name")},
		},
		"pool negative generation": {
			// +k8s:minimum=0 (standard DV)
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				pool := mkValidPoolStatus()
				pool.Generation = -1
				s.Pools = []resource.PoolStatus{pool}
			})),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("status", "pools").Index(0).Child("generation"), nil, "").WithOrigin("minimum")},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, StatusStrategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

// mkValidRPSR creates a valid ResourcePoolStatusRequest with optional tweaks.
func mkValidRPSR(tweaks ...func(*resource.ResourcePoolStatusRequest)) resource.ResourcePoolStatusRequest {
	obj := resource.ResourcePoolStatusRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-rpsr",
		},
		Spec: resource.ResourcePoolStatusRequestSpec{
			Driver:   "test.example.com",
			PoolName: ptr.To("pool-1"),
			Limit:    ptr.To[int32](100),
		},
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}

// mkValidRPSRForUpdate creates a valid ResourcePoolStatusRequest for update scenarios.
func mkValidRPSRForUpdate(tweaks ...func(*resource.ResourcePoolStatusRequest)) resource.ResourcePoolStatusRequest {
	obj := mkValidRPSR(tweaks...)
	obj.ResourceVersion = "1"
	return obj
}

func mkValidPoolStatus() resource.PoolStatus {
	return resource.PoolStatus{
		Driver:             "test.example.com",
		PoolName:           "pool-1",
		TotalDevices:       ptr.To[int32](4),
		AllocatedDevices:   ptr.To[int32](2),
		AvailableDevices:   ptr.To[int32](2),
		UnavailableDevices: ptr.To[int32](0),
		ResourceSliceCount: ptr.To[int32](1),
		Generation:         1,
	}
}

func setDriver(driver string) func(*resource.ResourcePoolStatusRequest) {
	return func(r *resource.ResourcePoolStatusRequest) {
		r.Spec.Driver = driver
	}
}

func setPoolName(name string) func(*resource.ResourcePoolStatusRequest) {
	return func(r *resource.ResourcePoolStatusRequest) {
		r.Spec.PoolName = &name
	}
}

func setLimit(limit int32) func(*resource.ResourcePoolStatusRequest) {
	return func(r *resource.ResourcePoolStatusRequest) {
		r.Spec.Limit = &limit
	}
}

func withValidStatus() func(*resource.ResourcePoolStatusRequest) {
	return withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
		s.Pools = []resource.PoolStatus{mkValidPoolStatus()}
		s.PoolCount = ptr.To[int32](1)
	})
}

// mkBaseStatus creates a status with all required fields populated.
func mkBaseStatus() *resource.ResourcePoolStatusRequestStatus {
	return &resource.ResourcePoolStatusRequestStatus{
		PoolCount: ptr.To[int32](0),
	}
}

func withStatus(fn func(*resource.ResourcePoolStatusRequestStatus)) func(*resource.ResourcePoolStatusRequest) {
	return func(r *resource.ResourcePoolStatusRequest) {
		if r.Status == nil {
			r.Status = mkBaseStatus()
		}
		fn(r.Status)
	}
}
