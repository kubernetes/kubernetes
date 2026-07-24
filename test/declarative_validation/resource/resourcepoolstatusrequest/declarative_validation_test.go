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

	apiresource "k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/kubernetes/pkg/features"
	registry "k8s.io/kubernetes/pkg/registry/resource/resourcepoolstatusrequest"
	"k8s.io/kubernetes/test/declarative_validation/meta"
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
		input                   resource.ResourcePoolStatusRequest
		enablePartitionTypeAttr bool
		expectedErrs            field.ErrorList
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
		"valid partitionTypeAttribute": {
			// +k8s:ifEnabled(DRAPartitionableDevicesType)=+k8s:optional
			input: mkValidRPSR(func(r *resource.ResourcePoolStatusRequest) {
				r.Spec.DefaultPartitionTypeAttribute = new("gpu.example.com/profile")
			}),
			enablePartitionTypeAttr: true,
		},
		"invalid partitionTypeAttribute format": {
			// +k8s:ifEnabled(DRAPartitionableDevicesType)=+k8s:format=k8s-resource-fully-qualified-name
			input: mkValidRPSR(func(r *resource.ResourcePoolStatusRequest) {
				r.Spec.DefaultPartitionTypeAttribute = new("invalid attr!")
			}),
			enablePartitionTypeAttr: true,
			expectedErrs:            field.ErrorList{field.Invalid(field.NewPath("spec", "defaultPartitionTypeAttribute"), nil, "").WithOrigin("format=k8s-resource-fully-qualified-name")},
		},
		"partitionTypeAttribute without domain": {
			// A bare name is a valid attribute key but not a valid reference:
			// the domain is required so the attribute is unambiguous.
			input: mkValidRPSR(func(r *resource.ResourcePoolStatusRequest) {
				r.Spec.DefaultPartitionTypeAttribute = new("profile")
			}),
			enablePartitionTypeAttr: true,
			expectedErrs:            field.ErrorList{field.Invalid(field.NewPath("spec", "defaultPartitionTypeAttribute"), nil, "").WithOrigin("format=k8s-resource-fully-qualified-name")},
		},
		"partitionTypeAttribute with feature disabled": {
			// +k8s:ifDisabled(DRAPartitionableDevicesType)=+k8s:forbidden
			input: mkValidRPSR(func(r *resource.ResourcePoolStatusRequest) {
				r.Spec.DefaultPartitionTypeAttribute = new("gpu.example.com/profile")
			}),
			expectedErrs: field.ErrorList{field.Forbidden(field.NewPath("spec", "defaultPartitionTypeAttribute"), "")},
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
		"limit nil": {
			// +k8s:required (standard DV) — pointer is nil
			input: mkValidRPSR(func(r *resource.ResourcePoolStatusRequest) {
				r.Spec.Limit = nil
			}),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("spec", "limit"), "")},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			// DRAPartitionableDevicesType depends on DRAResourcePoolStatus, so it
			// cannot be enabled on its own.
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAResourcePoolStatus, true)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAPartitionableDevicesType, tc.enablePartitionTypeAttr)
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, registry.Strategy, tc.expectedErrs)
		})
	}

	obj := mkValidRPSR()
	meta.RunObjectMetaTestCases(t, ctx, &obj, registry.Strategy, meta.WithStringentFinalizerValidation())
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
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, registry.Strategy, tc.expectedErrs)
		})
	}

	updateObj := mkValidRPSRForUpdate()
	meta.RunObjectMetaUpdateTestCases(t, ctx, &updateObj, registry.Strategy, meta.WithStringentFinalizerValidation())
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
						Message:            "Message",
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
		"poolCount nil": {
			// +k8s:required (standard DV) — pointer is nil
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				s.PoolCount = nil
			})),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("status", "poolCount"), "")},
		},
		"pool zero generation": {
			// +k8s:required (standard DV) — zero is the zero value for int64
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				pool := mkValidPoolStatus()
				pool.Generation = 0
				s.Pools = []resource.PoolStatus{pool}
			})),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("status", "pools").Index(0).Child("generation"), "")},
		},
		"partitionSummary maxItems exceeded": {
			// +k8s:maxItems=32
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				pool := mkValidPoolStatus()
				summary := make([]resource.PartitionTypeStatus, 33)
				for i := range summary {
					summary[i] = resource.PartitionTypeStatus{Attribute: "gpu.example.com/profile", Type: "Full" + strings.Repeat("x", i), Total: ptr.To[int32](0), Allocatable: ptr.To[int32](0)}
				}
				pool.PartitionSummary = summary
				s.Pools = []resource.PoolStatus{pool}
			})),
			expectedErrs: field.ErrorList{field.TooMany(field.NewPath("status", "pools").Index(0).Child("partitionSummary"), 33, 32).WithOrigin("maxItems")},
		},
		"partitionSummary empty type": {
			// +k8s:required (standard DV)
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				pool := mkValidPoolStatus()
				pool.PartitionSummary = []resource.PartitionTypeStatus{{Attribute: "gpu.example.com/profile", Type: "", Total: ptr.To[int32](0), Allocatable: ptr.To[int32](0)}}
				s.Pools = []resource.PoolStatus{pool}
			})),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("status", "pools").Index(0).Child("partitionSummary").Index(0).Child("type"), "")},
		},
		"partitionSummary negative total": {
			// +k8s:minimum=0 (standard DV)
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				pool := mkValidPoolStatus()
				pool.PartitionSummary = []resource.PartitionTypeStatus{{Attribute: "gpu.example.com/profile", Type: "Full", Total: ptr.To[int32](-1), Allocatable: ptr.To[int32](0)}}
				s.Pools = []resource.PoolStatus{pool}
			})),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("status", "pools").Index(0).Child("partitionSummary").Index(0).Child("total"), nil, "").WithOrigin("minimum")},
		},
		"partitionSummary negative allocatable": {
			// +k8s:minimum=0 (standard DV)
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				pool := mkValidPoolStatus()
				pool.PartitionSummary = []resource.PartitionTypeStatus{{Attribute: "gpu.example.com/profile", Type: "Full", Total: ptr.To[int32](0), Allocatable: ptr.To[int32](-1)}}
				s.Pools = []resource.PoolStatus{pool}
			})),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("status", "pools").Index(0).Child("partitionSummary").Index(0).Child("allocatable"), nil, "").WithOrigin("minimum")},
		},
		"partitionSummary total required": {
			// +k8s:required (standard DV)
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				pool := mkValidPoolStatus()
				pool.PartitionSummary = []resource.PartitionTypeStatus{{Attribute: "gpu.example.com/profile", Type: "Full", Allocatable: ptr.To[int32](0)}}
				s.Pools = []resource.PoolStatus{pool}
			})),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("status", "pools").Index(0).Child("partitionSummary").Index(0).Child("total"), "")},
		},
		"partitionSummary allocatable required": {
			// +k8s:required (standard DV)
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				pool := mkValidPoolStatus()
				pool.PartitionSummary = []resource.PartitionTypeStatus{{Attribute: "gpu.example.com/profile", Type: "Full", Total: ptr.To[int32](0)}}
				s.Pools = []resource.PoolStatus{pool}
			})),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("status", "pools").Index(0).Child("partitionSummary").Index(0).Child("allocatable"), "")},
		},
		"partitionSummary empty attribute": {
			// +k8s:required (standard DV)
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				pool := mkValidPoolStatus()
				pool.PartitionSummary = []resource.PartitionTypeStatus{{Type: "Full", Total: ptr.To[int32](0), Allocatable: ptr.To[int32](0)}}
				s.Pools = []resource.PoolStatus{pool}
			})),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("status", "pools").Index(0).Child("partitionSummary").Index(0).Child("attribute"), "")},
		},
		"partitionSummary duplicate attribute and type": {
			// +k8s:unique=map on (attribute, type)
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				pool := mkValidPoolStatus()
				entry := resource.PartitionTypeStatus{Attribute: "gpu.example.com/profile", Type: "Full", Total: ptr.To[int32](0), Allocatable: ptr.To[int32](0)}
				pool.PartitionSummary = []resource.PartitionTypeStatus{entry, entry}
				s.Pools = []resource.PoolStatus{pool}
			})),
			expectedErrs: field.ErrorList{field.Duplicate(field.NewPath("status", "pools").Index(0).Child("partitionSummary").Index(1), map[string]interface{}{"attribute": "gpu.example.com/profile", "type": "Full"})},
		},
		"shareableSummary capacity maxItems exceeded": {
			// +k8s:maxItems=32
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				pool := mkValidPoolStatus()
				caps := make([]resource.ShareableCapacityStatus, 33)
				for i := range caps {
					caps[i] = validShareableCapacity("mem")
				}
				pool.ShareableSummary = &resource.ShareableSummaryStatus{FullyAvailableDevices: ptr.To[int32](0), PartiallyAvailableDevices: ptr.To[int32](0), Capacity: caps}
				s.Pools = []resource.PoolStatus{pool}
			})),
			expectedErrs: field.ErrorList{field.TooMany(field.NewPath("status", "pools").Index(0).Child("shareableSummary").Child("capacity"), 33, 32).WithOrigin("maxItems")},
		},
		"shareableSummary capacity empty name": {
			// +k8s:required (standard DV)
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				pool := mkValidPoolStatus()
				cap := validShareableCapacity("")
				pool.ShareableSummary = &resource.ShareableSummaryStatus{FullyAvailableDevices: ptr.To[int32](0), PartiallyAvailableDevices: ptr.To[int32](0), Capacity: []resource.ShareableCapacityStatus{cap}}
				s.Pools = []resource.PoolStatus{pool}
			})),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("status", "pools").Index(0).Child("shareableSummary").Child("capacity").Index(0).Child("name"), "")},
		},
		"shareableSummary capacity total required": {
			// +k8s:required
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				pool := mkValidPoolStatus()
				cap := validShareableCapacity("mem")
				cap.Total = nil
				pool.ShareableSummary = &resource.ShareableSummaryStatus{FullyAvailableDevices: ptr.To[int32](0), PartiallyAvailableDevices: ptr.To[int32](0), Capacity: []resource.ShareableCapacityStatus{cap}}
				s.Pools = []resource.PoolStatus{pool}
			})),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("status", "pools").Index(0).Child("shareableSummary").Child("capacity").Index(0).Child("total"), "")},
		},
		"shareableSummary capacity consumed required": {
			// +k8s:required
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				pool := mkValidPoolStatus()
				cap := validShareableCapacity("mem")
				cap.Consumed = nil
				pool.ShareableSummary = &resource.ShareableSummaryStatus{FullyAvailableDevices: ptr.To[int32](0), PartiallyAvailableDevices: ptr.To[int32](0), Capacity: []resource.ShareableCapacityStatus{cap}}
				s.Pools = []resource.PoolStatus{pool}
			})),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("status", "pools").Index(0).Child("shareableSummary").Child("capacity").Index(0).Child("consumed"), "")},
		},
		"shareableSummary capacity available required": {
			// +k8s:required
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				pool := mkValidPoolStatus()
				cap := validShareableCapacity("mem")
				cap.Available = nil
				pool.ShareableSummary = &resource.ShareableSummaryStatus{FullyAvailableDevices: ptr.To[int32](0), PartiallyAvailableDevices: ptr.To[int32](0), Capacity: []resource.ShareableCapacityStatus{cap}}
				s.Pools = []resource.PoolStatus{pool}
			})),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("status", "pools").Index(0).Child("shareableSummary").Child("capacity").Index(0).Child("available"), "")},
		},
		"shareableSummary negative fullyAvailableDevices": {
			// +k8s:minimum=0 (standard DV)
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				pool := mkValidPoolStatus()
				pool.ShareableSummary = &resource.ShareableSummaryStatus{FullyAvailableDevices: ptr.To[int32](-1), PartiallyAvailableDevices: ptr.To[int32](0)}
				s.Pools = []resource.PoolStatus{pool}
			})),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("status", "pools").Index(0).Child("shareableSummary").Child("fullyAvailableDevices"), nil, "").WithOrigin("minimum")},
		},
		"shareableSummary negative partiallyAvailableDevices": {
			// +k8s:minimum=0 (standard DV)
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				pool := mkValidPoolStatus()
				pool.ShareableSummary = &resource.ShareableSummaryStatus{FullyAvailableDevices: ptr.To[int32](0), PartiallyAvailableDevices: ptr.To[int32](-1)}
				s.Pools = []resource.PoolStatus{pool}
			})),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("status", "pools").Index(0).Child("shareableSummary").Child("partiallyAvailableDevices"), nil, "").WithOrigin("minimum")},
		},
		"shareableSummary fullyAvailableDevices required": {
			// +k8s:required (standard DV)
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				pool := mkValidPoolStatus()
				pool.ShareableSummary = &resource.ShareableSummaryStatus{PartiallyAvailableDevices: ptr.To[int32](0)}
				s.Pools = []resource.PoolStatus{pool}
			})),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("status", "pools").Index(0).Child("shareableSummary").Child("fullyAvailableDevices"), "")},
		},
		"shareableSummary partiallyAvailableDevices required": {
			// +k8s:required (standard DV)
			oldObj: mkValidRPSRForUpdate(),
			updateObj: mkValidRPSRForUpdate(withStatus(func(s *resource.ResourcePoolStatusRequestStatus) {
				pool := mkValidPoolStatus()
				pool.ShareableSummary = &resource.ShareableSummaryStatus{FullyAvailableDevices: ptr.To[int32](0)}
				s.Pools = []resource.PoolStatus{pool}
			})),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("status", "pools").Index(0).Child("shareableSummary").Child("partiallyAvailableDevices"), "")},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, registry.StatusStrategy, tc.expectedErrs)
		})
	}

	meta.RunConditionTestCases(t, ctx, field.NewPath("status", "conditions"), &resource.ResourcePoolStatusRequest{}, registry.StatusStrategy, func(obj *resource.ResourcePoolStatusRequest, c []metav1.Condition) {
		*obj = mkValidRPSRForUpdate()
		if c != nil {
			withStatus(func(s *resource.ResourcePoolStatusRequestStatus) { s.Conditions = c })(obj)
		}
	})
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

func qty(s string) *apiresource.Quantity {
	q := apiresource.MustParse(s)
	return &q
}

func validShareableCapacity(name string) resource.ShareableCapacityStatus {
	return resource.ShareableCapacityStatus{Name: name, Total: qty("1"), Consumed: qty("0"), Available: qty("1")}
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
