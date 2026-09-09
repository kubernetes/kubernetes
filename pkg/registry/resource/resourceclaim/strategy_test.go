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

package resourceclaim

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"

	corev1 "k8s.io/api/core/v1"
	apiresource "k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes/fake"
	testclient "k8s.io/client-go/testing"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
)

var obj = &resource.ResourceClaim{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "valid-claim",
		Namespace: "kube-system",
	},
	Spec: resource.ResourceClaimSpec{
		Devices: resource.DeviceClaim{
			Requests: []resource.DeviceRequest{
				{
					Name: "req-0",
					Exactly: &resource.ExactDeviceRequest{
						DeviceClassName: "class",
						AllocationMode:  resource.DeviceAllocationModeAll,
					},
				},
			},
		},
	},
}

var objWithStatus = &resource.ResourceClaim{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "valid-claim",
		Namespace: "kube-system",
	},
	Spec: resource.ResourceClaimSpec{
		Devices: resource.DeviceClaim{
			Requests: []resource.DeviceRequest{
				{
					Name: "req-0",
					Exactly: &resource.ExactDeviceRequest{
						DeviceClassName: "class",
						AllocationMode:  resource.DeviceAllocationModeAll,
					},
				},
			},
		},
	},
	Status: resource.ResourceClaimStatus{
		Allocation: &resource.AllocationResult{
			Devices: resource.DeviceAllocationResult{
				Results: []resource.DeviceRequestAllocationResult{
					{
						Request: "req-0",
						Driver:  "dra.example.com",
						Pool:    "pool-0",
						Device:  "device-0",
					},
				},
			},
		},
	},
}

var objWithAdminAccess = &resource.ResourceClaim{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "valid-claim",
		Namespace: "kube-system",
	},
	Spec: resource.ResourceClaimSpec{
		Devices: resource.DeviceClaim{
			Requests: []resource.DeviceRequest{
				{
					Name: "req-0",
					Exactly: &resource.ExactDeviceRequest{
						DeviceClassName: "class",
						AllocationMode:  resource.DeviceAllocationModeAll,
						AdminAccess:     ptr.To(true),
					},
				},
			},
		},
	},
}

var objInNonAdminNamespace = &resource.ResourceClaim{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "valid-claim",
		Namespace: "default",
	},
	Spec: resource.ResourceClaimSpec{
		Devices: resource.DeviceClaim{
			Requests: []resource.DeviceRequest{
				{
					Name: "req-0",
					Exactly: &resource.ExactDeviceRequest{
						DeviceClassName: "class",
						AllocationMode:  resource.DeviceAllocationModeAll,
					},
				},
			},
		},
	},
}

var objWithAdminAccessInNonAdminNamespace = &resource.ResourceClaim{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "valid-claim",
		Namespace: "default",
	},
	Spec: resource.ResourceClaimSpec{
		Devices: resource.DeviceClaim{
			Requests: []resource.DeviceRequest{
				{
					Name: "req-0",
					Exactly: &resource.ExactDeviceRequest{
						DeviceClassName: "class",
						AllocationMode:  resource.DeviceAllocationModeAll,
						AdminAccess:     ptr.To(true),
					},
				},
			},
		},
	},
}

var objWithAdminAccessStatusInNonAdminNamespace = &resource.ResourceClaim{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "valid-claim",
		Namespace: "default",
	},
	Spec: resource.ResourceClaimSpec{
		Devices: resource.DeviceClaim{
			Requests: []resource.DeviceRequest{
				{
					Name: "req-0",
					Exactly: &resource.ExactDeviceRequest{
						DeviceClassName: "class",
						AllocationMode:  resource.DeviceAllocationModeAll,
						AdminAccess:     ptr.To(true),
					},
				},
			},
		},
	},
	Status: resource.ResourceClaimStatus{
		Allocation: &resource.AllocationResult{
			Devices: resource.DeviceAllocationResult{
				Results: []resource.DeviceRequestAllocationResult{
					{
						Request:     "req-0",
						Driver:      "dra.example.com",
						Pool:        "pool-0",
						Device:      "device-0",
						AdminAccess: ptr.To(true),
					},
				},
			},
		},
	},
}

var objWithDeviceTaints = &resource.ResourceClaim{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "valid-claim",
		Namespace: "kube-system",
	},
	Spec: resource.ResourceClaimSpec{
		Devices: resource.DeviceClaim{
			Requests: []resource.DeviceRequest{
				{
					Name: "req-0",
					Exactly: &resource.ExactDeviceRequest{
						DeviceClassName: "class",
						AllocationMode:  resource.DeviceAllocationModeAll,
						Tolerations:     []resource.DeviceToleration{{Key: "some-key", Operator: resource.DeviceTolerationOpExists}},
					},
				},
			},
		},
	},
}

var objWithPrioritizedList = &resource.ResourceClaim{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "valid-claim",
		Namespace: "kube-system",
	},
	Spec: resource.ResourceClaimSpec{
		Devices: resource.DeviceClaim{
			Requests: []resource.DeviceRequest{
				{
					Name: "req-0",
					FirstAvailable: []resource.DeviceSubRequest{
						{
							Name:            "subreq-0",
							DeviceClassName: "class",
							AllocationMode:  resource.DeviceAllocationModeExactCount,
							Count:           1,
						},
					},
				},
			},
		},
	},
}

var objWithDeviceTaintsInPrioritizedList = &resource.ResourceClaim{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "valid-claim",
		Namespace: "kube-system",
	},
	Spec: resource.ResourceClaimSpec{
		Devices: resource.DeviceClaim{
			Requests: []resource.DeviceRequest{
				{
					Name: "req-0",
					FirstAvailable: []resource.DeviceSubRequest{
						{
							Name:            "subreq-0",
							DeviceClassName: "class",
							AllocationMode:  resource.DeviceAllocationModeExactCount,
							Count:           1,
							Tolerations:     []resource.DeviceToleration{{Key: "some-key", Operator: resource.DeviceTolerationOpExists}},
						},
					},
				},
			},
		},
	},
}

var objWithAdminAccessStatus = &resource.ResourceClaim{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "valid-claim",
		Namespace: "kube-system",
	},
	Spec: resource.ResourceClaimSpec{
		Devices: resource.DeviceClaim{
			Requests: []resource.DeviceRequest{
				{
					Name: "req-0",
					Exactly: &resource.ExactDeviceRequest{
						DeviceClassName: "class",
						AllocationMode:  resource.DeviceAllocationModeAll,
						AdminAccess:     ptr.To(true),
					},
				},
			},
		},
	},
	Status: resource.ResourceClaimStatus{
		Allocation: &resource.AllocationResult{
			Devices: resource.DeviceAllocationResult{
				Results: []resource.DeviceRequestAllocationResult{
					{
						Request:     "req-0",
						Driver:      "dra.example.com",
						Pool:        "pool-0",
						Device:      "device-0",
						AdminAccess: ptr.To(true),
					},
				},
			},
		},
	},
}

var objWithDeviceBindingConditions = &resource.ResourceClaim{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "valid-claim",
		Namespace: "kube-system",
	},
	Spec: resource.ResourceClaimSpec{
		Devices: resource.DeviceClaim{
			Requests: []resource.DeviceRequest{
				{
					Name: "req-0",
					Exactly: &resource.ExactDeviceRequest{
						DeviceClassName: "class",
						AllocationMode:  resource.DeviceAllocationModeAll,
					},
				},
			},
		},
	},
	Status: resource.ResourceClaimStatus{
		Allocation: &resource.AllocationResult{
			Devices: resource.DeviceAllocationResult{
				Results: []resource.DeviceRequestAllocationResult{
					{
						Request:                  "req-0",
						Driver:                   "dra.example.com",
						Pool:                     "pool-0",
						Device:                   "device-0",
						BindingConditions:        []string{"condition-1", "condition-2"},
						BindingFailureConditions: []string{"condition-3", "condition-4"},
					},
				},
			},
		},
	},
}

var objWithSkipNodeOperationsStatus = &resource.ResourceClaim{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "valid-claim",
		Namespace: "kube-system",
	},
	Spec: resource.ResourceClaimSpec{
		Devices: resource.DeviceClaim{
			Requests: []resource.DeviceRequest{
				{
					Name: "req-0",
					Exactly: &resource.ExactDeviceRequest{
						DeviceClassName: "class",
						AllocationMode:  resource.DeviceAllocationModeAll,
					},
				},
			},
		},
	},
	Status: resource.ResourceClaimStatus{
		Allocation: &resource.AllocationResult{
			Devices: resource.DeviceAllocationResult{
				Results: []resource.DeviceRequestAllocationResult{
					{
						Request:            "req-0",
						Driver:             "dra.example.com",
						Pool:               "pool-0",
						Device:             "device-0",
						SkipNodeOperations: []resource.SkipNodeOperation{resource.SkipNodeOperationAll},
					},
				},
			},
		},
	},
}

var objWithCapacityRequests = &resource.ResourceClaim{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "valid-claim",
		Namespace: "kube-system",
	},
	Spec: resource.ResourceClaimSpec{
		Devices: resource.DeviceClaim{
			Requests: []resource.DeviceRequest{
				{
					Name: "req-0",
					Exactly: &resource.ExactDeviceRequest{
						DeviceClassName: "class",
						AllocationMode:  resource.DeviceAllocationModeAll,
						Capacity: &resource.CapacityRequirements{
							Requests: map[resource.QualifiedName]apiresource.Quantity{
								resource.QualifiedName("test-capacity"): apiresource.MustParse("1"),
							},
						},
					},
				},
			},
		},
	},
}

var objWithDerivedAttributes = &resource.ResourceClaim{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "valid-claim",
		Namespace: "kube-system",
	},
	Spec: resource.ResourceClaimSpec{
		Devices: resource.DeviceClaim{
			Requests: []resource.DeviceRequest{
				{
					Name: "req-0",
					Exactly: &resource.ExactDeviceRequest{
						DeviceClassName: "class",
						AllocationMode:  resource.DeviceAllocationModeAll,
						DerivedAttributes: []resource.DeviceDerivedAttribute{
							{
								Name:       "derived/sharedNumaNode",
								Expression: `device.attributes["dra.example.com"]["numa"]`,
							},
						},
					},
				},
			},
			Constraints: []resource.DeviceConstraint{
				{
					Requests:       []string{"req-0"},
					MatchAttribute: ptr.To(resource.FullyQualifiedName("derived/sharedNumaNode")),
				},
			},
		},
	},
}

var objWithDerivedAttributesInPrioritizedList = &resource.ResourceClaim{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "valid-claim",
		Namespace: "kube-system",
	},
	Spec: resource.ResourceClaimSpec{
		Devices: resource.DeviceClaim{
			Requests: []resource.DeviceRequest{
				{
					Name: "req-0",
					FirstAvailable: []resource.DeviceSubRequest{
						{
							Name:            "subreq-0",
							DeviceClassName: "class",
							AllocationMode:  resource.DeviceAllocationModeExactCount,
							Count:           1,
							DerivedAttributes: []resource.DeviceDerivedAttribute{
								{
									Name:       "derived/sharedNumaNode",
									Expression: `device.attributes["dra.example.com"]["numa"]`,
								},
							},
						},
					},
				},
			},
			Constraints: []resource.DeviceConstraint{
				{
					Requests:       []string{"req-0"},
					MatchAttribute: ptr.To(resource.FullyQualifiedName("derived/sharedNumaNode")),
				},
			},
		},
	},
}

var ns1 = &corev1.Namespace{
	ObjectMeta: metav1.ObjectMeta{
		Name:   "default",
		Labels: map[string]string{"key": "value"},
	},
}
var ns2 = &corev1.Namespace{
	ObjectMeta: metav1.ObjectMeta{
		Name:   "kube-system",
		Labels: map[string]string{resource.DRAAdminNamespaceLabelKey: "true"},
	},
}

var adminAccessError = "Forbidden: admin access to devices requires the `resource.kubernetes.io/admin-access: true` label"
var fieldImmutableError = "field is immutable"
var metadataError = "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters"
var deviceRequestError = "exactly one of `exactly` or `firstAvailable` is required"
var constraintError = "matchAttribute: Required value"
var bindingUpdateError = `User "test-user" cannot update resource "resourceclaims/binding" in API group "resource.k8s.io" at the cluster scope: denied`
var deviceAssociatedNodeUpdateError = `User "system:serviceaccount:kube-system:dra-driver" cannot associated-node:update resource "resourceclaims/driver" in API group "resource.k8s.io" in the namespace "default": denied`
var deviceArbitraryNodeUpdateError = `User "test-user" cannot arbitrary-node:update resource "resourceclaims/driver" in API group "resource.k8s.io" in the namespace "default": denied`

const (
	req0        = "req-0"
	subReq0     = "subreq-0"
	req0SubReq0 = "req-0/subreq-0"

	testRequest = "test-request"
	testDriver  = "test-driver"
	testPool    = "test-pool"
	testDevice  = "test-device"
	testUser    = "test-user"
)

var (
	testShareID = ptr.To(types.UID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"))
)

var testCapacity = map[resource.QualifiedName]apiresource.Quantity{
	resource.QualifiedName("test-capacity"): apiresource.MustParse("1"),
}

func TestStrategy(t *testing.T) {
	fakeClient := fake.NewSimpleClientset()
	mockNSClient := fakeClient.CoreV1().Namespaces()
	strategy := NewStrategy(mockNSClient, nil)
	if !strategy.NamespaceScoped() {
		t.Errorf("ResourceClaim must be namespace scoped")
	}
	if strategy.AllowCreateOnUpdate(context.Background()) {
		t.Errorf("ResourceClaim should not allow create on update")
	}
}

func TestStrategyCreate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	testcases := map[string]struct {
		obj                   *resource.ResourceClaim
		featureOverrides      featuregatetesting.FeatureOverrides
		emulatedVersion       string
		expectValidationError string
		expectObj             *resource.ResourceClaim
		verify                func(*testing.T, []testclient.Action)
	}{
		"simple": {
			obj:       obj,
			expectObj: obj,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"validation-error": {
			obj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				obj.Name = "%#@$%$"
				return obj
			}(),
			expectValidationError: metadataError,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"drop-fields-admin-access": {
			obj:              objWithAdminAccess,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAAdminAccess: false},
			emulatedVersion:  "1.35",
			expectObj:        obj,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-fields-admin-access": {
			obj:              objWithAdminAccess,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAAdminAccess: true},
			expectObj:        objWithAdminAccess,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 1 {
					t.Errorf("expected one action but got %d", len(as))
					return
				}
				ns := as[0].(testclient.GetAction).GetName()
				if ns != "kube-system" {
					t.Errorf("expected to get the kube-system namespace but got '%s'", ns)
				}
			},
		},
		"drop-fields-device-taints": {
			obj:              objWithDeviceTaints,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRADeviceTaints: false},
			emulatedVersion:  "1.35",
			expectObj:        obj,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-fields-device-taints": {
			obj:              objWithDeviceTaints,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRADeviceTaints: true},
			expectObj:        objWithDeviceTaints,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"drop-fields-device-taints-in-prioritized-list": {
			obj:              objWithDeviceTaintsInPrioritizedList,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRADeviceTaints: false, features.DRAPrioritizedList: true},
			emulatedVersion:  "1.35",
			expectObj:        objWithPrioritizedList,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-fields-device-taints-in-prioritized-list": {
			obj:              objWithDeviceTaintsInPrioritizedList,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRADeviceTaints: true, features.DRAPrioritizedList: true},
			expectObj:        objWithDeviceTaintsInPrioritizedList,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"drop-fields-prioritized-list": {
			obj:                   objWithPrioritizedList,
			featureOverrides:      featuregatetesting.FeatureOverrides{features.DRAPrioritizedList: false},
			emulatedVersion:       "1.36",
			expectValidationError: deviceRequestError,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-fields-prioritized-list": {
			obj:              objWithPrioritizedList,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPrioritizedList: true},
			expectObj:        objWithPrioritizedList,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"admin-access-admin-namespace": {
			obj:              objWithAdminAccess,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAAdminAccess: true},
			expectObj:        objWithAdminAccess,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 1 {
					t.Errorf("expected one action but got %d", len(as))
					return
				}
				ns := as[0].(testclient.GetAction).GetName()
				if ns != "kube-system" {
					t.Errorf("expected to get the kube-system namespace but got '%s'", ns)
				}
			},
		},
		"admin-access-non-admin-namespace": {
			obj:                   objWithAdminAccessInNonAdminNamespace,
			featureOverrides:      featuregatetesting.FeatureOverrides{features.DRAAdminAccess: true},
			expectObj:             objWithAdminAccessInNonAdminNamespace,
			expectValidationError: adminAccessError,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 1 {
					t.Errorf("expected one action but got %d", len(as))
					return
				}
				ns := as[0].(testclient.GetAction).GetName()
				if ns != "default" {
					t.Errorf("expected to get the default namespace but got '%s'", ns)
				}
			},
		},
		"keep-fields-consumable-capacity": {
			obj:              objWithCapacityRequests,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAConsumableCapacity: true},
			expectObj:        objWithCapacityRequests,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"drop-fields-consumable-capacity-disabled-feature": {
			obj:              objWithCapacityRequests,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAConsumableCapacity: false},
			expectObj:        obj,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"drop-fields-consumable-capacity-disabled-feature-with-distinct-attribute": {
			obj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				modifySpecDeviceRequestWithCapacityRequests(obj, nil, false)
				addDistinctAttribute(obj)
				return obj
			}(),
			featureOverrides:      featuregatetesting.FeatureOverrides{features.DRAConsumableCapacity: false},
			expectValidationError: constraintError,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"drop-fields-consumable-capacity-disabled-feature-with-prioritized-list": {
			obj: func() *resource.ResourceClaim {
				obj := objWithPrioritizedList.DeepCopy()
				modifySpecDeviceRequestWithCapacityRequests(obj, testCapacity, true)
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAConsumableCapacity: false, features.DRAPrioritizedList: true},
			expectObj: func() *resource.ResourceClaim {
				obj := objWithPrioritizedList.DeepCopy()
				modifySpecDeviceRequestWithCapacityRequests(obj, nil, true)
				return obj
			}(),
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			fakeClient := fake.NewSimpleClientset(ns1, ns2)
			mockNSClient := fakeClient.CoreV1().Namespaces()

			if tc.emulatedVersion != "" {
				featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse(tc.emulatedVersion))
			}
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, tc.featureOverrides)

			strategy := NewStrategy(mockNSClient, nil)

			obj := tc.obj.DeepCopy()
			strategy.PrepareForCreate(ctx, obj)
			if errs := strategy.Validate(ctx, obj); len(errs) != 0 {
				if tc.expectValidationError == "" {
					t.Fatalf("unexpected error(s): %v", errs)
				}
				assert.Len(t, errs, 1, "exactly one error expected")
				assert.ErrorContains(t, errs[0], tc.expectValidationError, "the error message should have contained the expected error message")
				return
			}
			if tc.expectValidationError != "" {
				t.Fatal("expected validation error(s), got none")
			}
			if warnings := strategy.WarningsOnCreate(ctx, obj); len(warnings) != 0 {
				t.Fatalf("unexpected warnings: %q", warnings)
			}
			strategy.Canonicalize(obj)
			assert.Equal(t, tc.expectObj, obj)
			tc.verify(t, fakeClient.Actions())
		})
	}
}

func TestStrategyUpdate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	testcases := map[string]struct {
		oldObj                *resource.ResourceClaim
		newObj                *resource.ResourceClaim
		featureOverrides      featuregatetesting.FeatureOverrides
		emulatedVersion       string
		expectValidationError string
		expectObj             *resource.ResourceClaim
		verify                func(*testing.T, []testclient.Action)
	}{
		"no-changes-okay": {
			oldObj:    obj,
			newObj:    obj,
			expectObj: obj,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"name-change-not-allowed": {
			oldObj: obj,
			newObj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				obj.Name += "-2"
				return obj
			}(),
			expectValidationError: fieldImmutableError,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"drop-fields-admin-access": {
			oldObj:           obj,
			newObj:           objWithAdminAccess,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAAdminAccess: false},
			emulatedVersion:  "1.35",
			expectObj:        obj,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-fields-admin-access": {
			oldObj:                obj,
			newObj:                objWithAdminAccess,
			featureOverrides:      featuregatetesting.FeatureOverrides{features.DRAAdminAccess: true},
			expectValidationError: fieldImmutableError, // Spec is immutable.
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-existing-fields-admin-access": {
			oldObj:           objWithAdminAccess,
			newObj:           objWithAdminAccess,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAAdminAccess: true},
			expectObj:        objWithAdminAccess,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"admin-access-admin-namespace": {
			oldObj:           objWithAdminAccess,
			newObj:           objWithAdminAccess,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAAdminAccess: true},
			expectObj:        objWithAdminAccess,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"admin-access-non-admin-namespace": {
			oldObj:                objInNonAdminNamespace,
			newObj:                objWithAdminAccessInNonAdminNamespace,
			featureOverrides:      featuregatetesting.FeatureOverrides{features.DRAAdminAccess: true},
			expectValidationError: fieldImmutableError,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"drop-fields-prioritized-list": {
			oldObj:                obj,
			newObj:                objWithPrioritizedList,
			featureOverrides:      featuregatetesting.FeatureOverrides{features.DRAPrioritizedList: false},
			emulatedVersion:       "1.36",
			expectValidationError: fieldImmutableError,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-fields-prioritized-list": {
			oldObj:                obj,
			newObj:                objWithPrioritizedList,
			featureOverrides:      featuregatetesting.FeatureOverrides{features.DRAPrioritizedList: true},
			expectValidationError: fieldImmutableError, // Spec is immutable.
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-existing-fields-prioritized-list": {
			oldObj:           objWithPrioritizedList,
			newObj:           objWithPrioritizedList,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPrioritizedList: true},
			expectObj:        objWithPrioritizedList,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-existing-fields-prioritized-list-disabled-feature": {
			oldObj:           objWithPrioritizedList,
			newObj:           objWithPrioritizedList,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPrioritizedList: false},
			emulatedVersion:  "1.36",
			expectObj:        objWithPrioritizedList,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-existing-fields-consumable-capacity": {
			oldObj:           objWithCapacityRequests,
			newObj:           objWithCapacityRequests,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAConsumableCapacity: true},
			expectObj:        objWithCapacityRequests,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-existing-fields-consumable-capacity-disabled-feature": {
			oldObj:           objWithCapacityRequests,
			newObj:           objWithCapacityRequests,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAConsumableCapacity: false},
			expectObj:        objWithCapacityRequests,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"drop-fields-consumable-capacity-disabled-feature": {
			oldObj:           obj,
			newObj:           objWithCapacityRequests,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAConsumableCapacity: false},
			expectObj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				modifySpecDeviceRequestWithCapacityRequests(obj, nil, false)
				return obj
			}(),
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"drop-fields-consumable-capacity-disabled-feature-with-prioritized-list": {
			oldObj: func() *resource.ResourceClaim {
				obj := objWithPrioritizedList.DeepCopy()
				modifySpecDeviceRequestWithCapacityRequests(obj, nil, true)
				return obj
			}(),
			newObj: func() *resource.ResourceClaim {
				obj := objWithPrioritizedList.DeepCopy()
				modifySpecDeviceRequestWithCapacityRequests(obj, testCapacity, true)
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAConsumableCapacity: false, features.DRAPrioritizedList: true},
			expectObj: func() *resource.ResourceClaim {
				obj := objWithPrioritizedList.DeepCopy()
				modifySpecDeviceRequestWithCapacityRequests(obj, nil, true)
				return obj
			}(),
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"drop-fields-device-taints": {
			oldObj:           obj,
			newObj:           objWithDeviceTaints,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRADeviceTaints: false, features.DRAPrioritizedList: true},
			emulatedVersion:  "1.35",
			expectObj:        obj,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-fields-device-taints": {
			oldObj:                obj,
			newObj:                objWithDeviceTaints,
			featureOverrides:      featuregatetesting.FeatureOverrides{features.DRADeviceTaints: true, features.DRAPrioritizedList: true},
			expectValidationError: fieldImmutableError, // Spec is immutable, cannot add tolerations.
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-existing-fields-device-taints": {
			oldObj:           objWithDeviceTaints,
			newObj:           objWithDeviceTaints,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRADeviceTaints: true, features.DRAPrioritizedList: true},
			expectObj:        objWithDeviceTaints,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-existing-fields-device-taints-disabled-feature": {
			oldObj:           objWithDeviceTaints,
			newObj:           objWithDeviceTaints,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRADeviceTaints: false, features.DRAPrioritizedList: true},
			emulatedVersion:  "1.35",
			expectObj:        objWithDeviceTaints,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"drop-fields-device-taints-in-prioritized-list": {
			oldObj:           objWithPrioritizedList,
			newObj:           objWithDeviceTaintsInPrioritizedList,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRADeviceTaints: false, features.DRAPrioritizedList: true},
			emulatedVersion:  "1.35",
			expectObj:        objWithPrioritizedList,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-fields-device-taints-in-prioritized-list": {
			oldObj:                objWithPrioritizedList,
			newObj:                objWithDeviceTaintsInPrioritizedList,
			featureOverrides:      featuregatetesting.FeatureOverrides{features.DRADeviceTaints: true, features.DRAPrioritizedList: true},
			expectValidationError: fieldImmutableError, // Spec is immutable, cannot add tolerations.
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-existing-fields-device-taints-in-prioritized-list": {
			oldObj:           objWithDeviceTaintsInPrioritizedList,
			newObj:           objWithDeviceTaintsInPrioritizedList,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRADeviceTaints: true, features.DRAPrioritizedList: true},
			expectObj:        objWithDeviceTaintsInPrioritizedList,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-existing-fields-device-taints-in-prioritized-list-disabled-feature": {
			oldObj:           objWithDeviceTaintsInPrioritizedList,
			newObj:           objWithDeviceTaintsInPrioritizedList,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRADeviceTaints: false, features.DRAPrioritizedList: true},
			emulatedVersion:  "1.35",
			expectObj:        objWithDeviceTaintsInPrioritizedList,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			fakeClient := fake.NewSimpleClientset(ns1, ns2)
			mockNSClient := fakeClient.CoreV1().Namespaces()

			if tc.emulatedVersion != "" {
				featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse(tc.emulatedVersion))
			}
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, tc.featureOverrides)

			strategy := NewStrategy(mockNSClient, nil)

			oldObj := tc.oldObj.DeepCopy()
			newObj := tc.newObj.DeepCopy()
			newObj.ResourceVersion = "4"

			strategy.PrepareForUpdate(ctx, newObj, oldObj)
			if errs := strategy.ValidateUpdate(ctx, newObj, oldObj); len(errs) != 0 {
				if tc.expectValidationError == "" {
					t.Fatalf("unexpected error(s): %v", errs)
				}
				assert.Len(t, errs, 1, "exactly one error expected")
				assert.ErrorContains(t, errs[0], tc.expectValidationError, "the error message should have contained the expected error message")
				return
			}
			if tc.expectValidationError != "" {
				t.Fatal("expected validation error(s), got none")
			}
			if warnings := strategy.WarningsOnUpdate(ctx, newObj, oldObj); len(warnings) != 0 {
				t.Fatalf("unexpected warnings: %q", warnings)
			}
			strategy.Canonicalize(newObj)
			expectObj := tc.expectObj.DeepCopy()
			expectObj.ResourceVersion = "4"
			assert.Equal(t, expectObj, newObj)
			tc.verify(t, fakeClient.Actions())
		})
	}
}

func stripDerivedAttributes(obj *resource.ResourceClaim) *resource.ResourceClaim {
	res := obj.DeepCopy()
	for i := range res.Spec.Devices.Requests {
		if res.Spec.Devices.Requests[i].Exactly != nil {
			res.Spec.Devices.Requests[i].Exactly.DerivedAttributes = nil
		}
		for j := range res.Spec.Devices.Requests[i].FirstAvailable {
			res.Spec.Devices.Requests[i].FirstAvailable[j].DerivedAttributes = nil
		}
	}
	return res
}

func TestStrategyCreateWithDerivedAttributes(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	testcases := []struct {
		name              string
		obj               *resource.ResourceClaim
		derivedAttributes bool // DRADerivedAttributes feature gate
		expectObj         *resource.ResourceClaim
	}{
		{
			name:              "should drop derivedAttributes during creation when the feature gate is disabled",
			obj:               objWithDerivedAttributes,
			derivedAttributes: false,
			expectObj:         stripDerivedAttributes(objWithDerivedAttributes),
		},
		{
			name:              "should preserve derivedAttributes during creation when the feature gate is enabled",
			obj:               objWithDerivedAttributes,
			derivedAttributes: true,
			expectObj:         objWithDerivedAttributes,
		},
		{
			name:              "should drop derivedAttributes in prioritized list during creation when the feature gate is disabled",
			obj:               objWithDerivedAttributesInPrioritizedList,
			derivedAttributes: false,
			expectObj:         stripDerivedAttributes(objWithDerivedAttributesInPrioritizedList),
		},
		{
			name:              "should preserve derivedAttributes in prioritized list during creation when the feature gate is enabled",
			obj:               objWithDerivedAttributesInPrioritizedList,
			derivedAttributes: true,
			expectObj:         objWithDerivedAttributesInPrioritizedList,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			fakeClient := fake.NewSimpleClientset(ns1, ns2)
			mockNSClient := fakeClient.CoreV1().Namespaces()
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRADerivedAttributes, tc.derivedAttributes)
			strategy := NewStrategy(mockNSClient, nil)

			obj := tc.obj.DeepCopy()
			strategy.PrepareForCreate(ctx, obj)
			if errs := strategy.Validate(ctx, obj); len(errs) != 0 {
				t.Fatalf("unexpected error(s): %v", errs)
			}
			strategy.Canonicalize(obj)
			assert.Equal(t, tc.expectObj, obj)
		})
	}
}

func TestStrategyUpdateWithDerivedAttributes(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	testcases := []struct {
		name                  string
		oldObj                *resource.ResourceClaim
		newObj                *resource.ResourceClaim
		derivedAttributes     bool // DRADerivedAttributes feature gate
		expectValidationError string
		expectObj             *resource.ResourceClaim
	}{
		{
			name:              "should drop derivedAttributes during update when the feature gate is disabled and they were not previously in use",
			oldObj:            stripDerivedAttributes(objWithDerivedAttributes),
			newObj:            objWithDerivedAttributes,
			derivedAttributes: false,
			expectObj:         stripDerivedAttributes(objWithDerivedAttributes),
		},
		{
			name:                  "should fail validation on update when the feature gate is enabled because spec is immutable",
			oldObj:                stripDerivedAttributes(objWithDerivedAttributes),
			newObj:                objWithDerivedAttributes,
			derivedAttributes:     true,
			expectValidationError: fieldImmutableError, // Spec is immutable, cannot add derived attributes.
		},
		{
			name:              "should preserve derivedAttributes during update when the feature gate is enabled and they were already in use",
			oldObj:            objWithDerivedAttributes,
			newObj:            objWithDerivedAttributes,
			derivedAttributes: true,
			expectObj:         objWithDerivedAttributes,
		},
		{
			name:              "should preserve derivedAttributes during update even if the feature gate is disabled because they were already in use",
			oldObj:            objWithDerivedAttributes,
			newObj:            objWithDerivedAttributes,
			derivedAttributes: false,
			expectObj:         objWithDerivedAttributes,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			fakeClient := fake.NewSimpleClientset(ns1, ns2)
			mockNSClient := fakeClient.CoreV1().Namespaces()
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRADerivedAttributes, tc.derivedAttributes)
			strategy := NewStrategy(mockNSClient, nil)

			oldObj := tc.oldObj.DeepCopy()
			newObj := tc.newObj.DeepCopy()
			newObj.ResourceVersion = "4"

			strategy.PrepareForUpdate(ctx, newObj, oldObj)
			if errs := strategy.ValidateUpdate(ctx, newObj, oldObj); len(errs) != 0 {
				if tc.expectValidationError == "" {
					t.Fatalf("unexpected error(s): %v", errs)
				}
				assert.Len(t, errs, 1, "exactly one error expected")
				assert.ErrorContains(t, errs[0], tc.expectValidationError, "the error message should have contained the expected error message")
				return
			}
			if tc.expectValidationError != "" {
				t.Fatal("expected validation error(s), got none")
			}
			strategy.Canonicalize(newObj)
			expectObj := tc.expectObj.DeepCopy()
			expectObj.ResourceVersion = "4"
			assert.Equal(t, expectObj, newObj)
		})
	}
}

func TestStatusStrategyUpdate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	ctx = genericapirequest.WithUser(ctx, &user.DefaultInfo{
		Name:   testUser,
		Groups: []string{"system:authenticated"},
	})
	ctx = genericapirequest.WithRequestInfo(ctx, &genericapirequest.RequestInfo{
		IsResourceRequest: true,
		Verb:              "update",
		APIGroup:          "resource.k8s.io",
		APIVersion:        "v1",
		Resource:          "resourceclaims",
		Subresource:       "status",
		Namespace:         metav1.NamespaceDefault,
	})
	testcases := map[string]struct {
		oldObj                 *resource.ResourceClaim
		newObj                 *resource.ResourceClaim
		authz                  authorizer.UnconditionalAuthorizer
		ctxOverride            func(context.Context) context.Context // if set, transforms the default ctx
		featureOverrides       featuregatetesting.FeatureOverrides
		emulatedVersion        string
		expectValidationErrors []string
		expectObj              *resource.ResourceClaim
		verify                 func(*testing.T, []testclient.Action)
	}{
		"no-changes-okay": {
			oldObj:    obj,
			newObj:    obj,
			expectObj: obj,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"name-change-not-allowed": {
			oldObj: obj,
			newObj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				obj.Name += "-2"
				return obj
			}(),
			expectValidationErrors: []string{fieldImmutableError},
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		// Cannot add finalizers, annotations and labels during status update.
		"drop-meta-changes": {
			oldObj: obj,
			newObj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				obj.Finalizers = []string{"foo"}
				obj.Annotations = map[string]string{"foo": "bar"}
				obj.Labels = map[string]string{"foo": "bar"}
				return obj
			}(),
			expectObj: obj,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"drop-fields-admin-access": {
			oldObj:           obj,
			newObj:           objWithAdminAccessStatus,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAAdminAccess: false},
			emulatedVersion:  "1.35",
			expectObj:        objWithStatus,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-fields-admin-access": {
			oldObj:           obj,
			newObj:           objWithAdminAccessStatus,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAAdminAccess: true},
			expectObj: func() *resource.ResourceClaim {
				expectObj := objWithAdminAccessStatus.DeepCopy()
				// Spec remains unchanged.
				expectObj.Spec = obj.Spec
				return expectObj
			}(),
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 1 {
					t.Errorf("expected one action but got %d", len(as))
					return
				}
				ns := as[0].(testclient.GetAction).GetName()
				if ns != "kube-system" {
					t.Errorf("expected to get the kube-system namespace but got '%s'", ns)
				}
			},
		},
		"keep-fields-admin-access-NonAdminNamespace": {
			oldObj:                 objInNonAdminNamespace,
			newObj:                 objWithAdminAccessStatusInNonAdminNamespace,
			featureOverrides:       featuregatetesting.FeatureOverrides{features.DRAAdminAccess: true},
			expectValidationErrors: []string{adminAccessError},
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 1 {
					t.Errorf("expected one action but got %d", len(as))
					return
				}
				ns := as[0].(testclient.GetAction).GetName()
				if ns != "default" {
					t.Errorf("expected to get the default namespace but got '%s'", ns)
				}
			},
		},
		"keep-fields-admin-access-because-of-spec": {
			oldObj:           objWithAdminAccess,
			newObj:           objWithAdminAccessStatus,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAAdminAccess: false},
			emulatedVersion:  "1.35",
			expectObj:        objWithAdminAccessStatus,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 1 {
					t.Errorf("expected one action but got %d", len(as))
					return
				}
				ns := as[0].(testclient.GetAction).GetName()
				if ns != "kube-system" {
					t.Errorf("expected to get the kube-system namespace but got '%s'", ns)
				}
			},
		},
		// Normally a claim without admin access in the spec shouldn't
		// have one in the status either, but it's not invalid and thus
		// let's test this.
		"keep-fields-admin-access-because-of-status": {
			oldObj: func() *resource.ResourceClaim {
				oldObj := objWithAdminAccessStatus.DeepCopy()
				oldObj.Spec.Devices.Requests[0].Exactly.AdminAccess = ptr.To(false)
				return oldObj
			}(),
			newObj:           objWithAdminAccessStatus,
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAAdminAccess: false},
			emulatedVersion:  "1.35",
			expectObj: func() *resource.ResourceClaim {
				oldObj := objWithAdminAccessStatus.DeepCopy()
				oldObj.Spec.Devices.Requests[0].Exactly.AdminAccess = ptr.To(false)
				return oldObj
			}(),
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"drop-fields-devices-status": {
			oldObj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, testRequest)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, testRequest, nil, nil)
				return obj
			}(),
			newObj: func() *resource.ResourceClaim { // Status is added
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, testRequest)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, testRequest, nil, nil)
				addStatusDevices(obj, testDriver, testPool, testDevice, nil)
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAResourceClaimDeviceStatus: false},
			emulatedVersion:  "1.36",
			expectObj: func() *resource.ResourceClaim { // Status is no longer there
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, testRequest)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, testRequest, nil, nil)
				return obj
			}(),
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-fields-devices-status-disable-feature-gate": {
			oldObj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, testRequest)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, testRequest, nil, nil)
				addStatusDevices(obj, testDriver, testPool, testDevice, nil)
				return obj
			}(),
			newObj: func() *resource.ResourceClaim { // Status is added
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, testRequest)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, testRequest, nil, nil)
				addStatusDevices(obj, testDriver, testPool, testDevice, nil)
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAResourceClaimDeviceStatus: false},
			emulatedVersion:  "1.36",
			expectObj: func() *resource.ResourceClaim { // Status is still there (as the status was set in the old object)
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, testRequest)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, testRequest, nil, nil)
				addStatusDevices(obj, testDriver, testPool, testDevice, nil)
				return obj
			}(),
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-fields-devices-status": {
			oldObj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, testRequest)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, testRequest, nil, nil)
				return obj
			}(),
			newObj: func() *resource.ResourceClaim { // Status is added
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, testRequest)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, testRequest, nil, nil)
				addStatusDevices(obj, testDriver, testPool, testDevice, nil)
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAResourceClaimDeviceStatus: true},
			expectObj: func() *resource.ResourceClaim { // Status is still there
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, testRequest)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, testRequest, nil, nil)
				addStatusDevices(obj, testDriver, testPool, testDevice, nil)
				return obj
			}(),
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"drop-status-deallocated-device": {
			oldObj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, testRequest)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, testRequest, nil, nil)
				addStatusDevices(obj, testDriver, testPool, testDevice, nil)
				return obj
			}(),
			newObj: func() *resource.ResourceClaim { // device is deallocated
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, testRequest)
				addStatusDevices(obj, testDriver, testPool, testDevice, nil)
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAResourceClaimDeviceStatus: true},
			expectObj: func() *resource.ResourceClaim { // Status is no longer there
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, testRequest)
				return obj
			}(),
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"fail-update-fields-devices-status-without-permissions": {
			oldObj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, testRequest)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, testRequest, nil, nil)
				return obj
			}(),
			newObj: func() *resource.ResourceClaim { // Status is added
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, testRequest)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, testRequest, nil, nil)
				addStatusDevices(obj, testDriver, testPool, testDevice, nil)
				return obj
			}(),
			featureOverrides:       featuregatetesting.FeatureOverrides{features.DRAResourceClaimDeviceStatus: true, features.DRAAdminAccess: true},
			expectValidationErrors: []string{deviceArbitraryNodeUpdateError},
			expectObj: func() *resource.ResourceClaim { // Status is not updated
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, testRequest)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, testRequest, nil, nil)
				return obj
			}(),
			authz: &fakeAuthorizer{false},
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"fail-update-fields-devices-status-associated-node-without-permissions": {
			oldObj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, testRequest)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, testRequest, nil, nil)
				obj.Status.Allocation.NodeSelector = &core.NodeSelector{
					NodeSelectorTerms: []core.NodeSelectorTerm{{
						MatchFields: []core.NodeSelectorRequirement{{
							Key: "metadata.name", Operator: core.NodeSelectorOpIn, Values: []string{"test-node"},
						}},
					}},
				}
				return obj
			}(),
			newObj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, testRequest)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, testRequest, nil, nil)
				obj.Status.Allocation.NodeSelector = &core.NodeSelector{
					NodeSelectorTerms: []core.NodeSelectorTerm{{
						MatchFields: []core.NodeSelectorRequirement{{
							Key: "metadata.name", Operator: core.NodeSelectorOpIn, Values: []string{"test-node"},
						}},
					}},
				}
				addStatusDevices(obj, testDriver, testPool, testDevice, nil)
				return obj
			}(),
			ctxOverride: func(ctx context.Context) context.Context {
				return genericapirequest.WithUser(ctx, &user.DefaultInfo{
					Name:   "system:serviceaccount:kube-system:dra-driver",
					Groups: []string{"system:authenticated"},
					Extra:  map[string][]string{"authentication.kubernetes.io/node-name": {"test-node"}},
				})
			},
			featureOverrides:       featuregatetesting.FeatureOverrides{features.DRAResourceClaimDeviceStatus: true, features.DRAAdminAccess: true},
			authz:                  &fakeAuthorizer{false},
			expectValidationErrors: []string{deviceAssociatedNodeUpdateError},
			expectObj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, testRequest)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, testRequest, nil, nil)
				obj.Status.Allocation.NodeSelector = &core.NodeSelector{
					NodeSelectorTerms: []core.NodeSelectorTerm{{
						MatchFields: []core.NodeSelectorRequirement{{
							Key: "metadata.name", Operator: core.NodeSelectorOpIn, Values: []string{"test-node"},
						}},
					}},
				}
				return obj
			}(),
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"fail-drop-status-deallocated-device-without-permissions": {
			oldObj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, testRequest)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, testRequest, nil, nil)
				addStatusDevices(obj, testDriver, testPool, testDevice, nil)
				return obj
			}(),
			newObj: func() *resource.ResourceClaim { // device is deallocated
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, testRequest)
				addStatusDevices(obj, testDriver, testPool, testDevice, nil)
				return obj
			}(),
			authz:                  &fakeAuthorizer{false},
			featureOverrides:       featuregatetesting.FeatureOverrides{features.DRAResourceClaimDeviceStatus: true, features.DRAAdminAccess: true},
			expectValidationErrors: []string{bindingUpdateError},
			expectObj: func() *resource.ResourceClaim { // Status is no longer there
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, testRequest)
				return obj
			}(),
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"drop-status-deallocated-device-disable-feature-gate": {
			oldObj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, testRequest)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, testRequest, nil, nil)
				addStatusDevices(obj, testDriver, testPool, testDevice, nil)
				return obj
			}(),
			newObj: func() *resource.ResourceClaim { // device is deallocated
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, testRequest)
				addStatusDevices(obj, testDriver, testPool, testDevice, nil)
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAResourceClaimDeviceStatus: false},
			emulatedVersion:  "1.36",
			expectObj: func() *resource.ResourceClaim { // Status is no longer there
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, testRequest)
				return obj
			}(),
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-fields-binding-conditions": {
			oldObj:    obj,
			newObj:    objWithDeviceBindingConditions,
			expectObj: objWithDeviceBindingConditions,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAResourceClaimDeviceStatus: true, features.DRADeviceBindingConditions: true},
		},
		"keep-exist-fields-disable-bindingconditions-feature-gate": {
			oldObj:    objWithDeviceBindingConditions,
			newObj:    objWithDeviceBindingConditions,
			expectObj: objWithDeviceBindingConditions,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAResourceClaimDeviceStatus: true, features.DRADeviceBindingConditions: false},
		},
		"drop-fields-binding-conditions": {
			oldObj:    obj,
			newObj:    objWithDeviceBindingConditions,
			expectObj: objWithStatus,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAResourceClaimDeviceStatus: true, features.DRADeviceBindingConditions: false},
		},
		"drop-fields-binding-conditions-disable-feature-gate": {
			oldObj:    obj,
			newObj:    objWithDeviceBindingConditions,
			expectObj: objWithStatus,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAResourceClaimDeviceStatus: false, features.DRADeviceBindingConditions: false},
			emulatedVersion:  "1.36",
		},
		"drop-fields-binding-conditions-disable-binding-conditions-feature-gate": {
			oldObj:    obj,
			newObj:    objWithDeviceBindingConditions,
			expectObj: objWithStatus,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAResourceClaimDeviceStatus: true, features.DRADeviceBindingConditions: false},
		},
		"keep-fields-optional-node-operations": {
			oldObj:    obj,
			newObj:    objWithSkipNodeOperationsStatus,
			expectObj: objWithSkipNodeOperationsStatus,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAOptionalNodeOperations: true},
		},
		"keep-existing-fields-disable-optional-node-operations-feature-gate": {
			oldObj:    objWithSkipNodeOperationsStatus,
			newObj:    objWithSkipNodeOperationsStatus,
			expectObj: objWithSkipNodeOperationsStatus,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAOptionalNodeOperations: false},
		},
		"drop-fields-optional-node-operations": {
			oldObj:    obj,
			newObj:    objWithSkipNodeOperationsStatus,
			expectObj: objWithStatus,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAOptionalNodeOperations: false},
		},
		"keep-fields-consumable-capacity-with-device-status": {
			oldObj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				modifySpecDeviceRequestWithCapacityRequests(obj, testCapacity, false)
				addDistinctAttribute(obj)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, req0, testShareID, testCapacity)
				return obj
			}(),
			newObj: func() *resource.ResourceClaim { // Status is added with share id and consumed capacities
				obj := obj.DeepCopy()
				modifySpecDeviceRequestWithCapacityRequests(obj, testCapacity, false)
				addDistinctAttribute(obj)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, req0, testShareID, testCapacity)
				addStatusDevices(obj, testDriver, testPool, testDevice, testShareID)
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAResourceClaimDeviceStatus: true, features.DRAConsumableCapacity: true, features.DRAPrioritizedList: false},
			emulatedVersion:  "1.36",
			expectObj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				modifySpecDeviceRequestWithCapacityRequests(obj, testCapacity, false)
				addDistinctAttribute(obj)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, req0, testShareID, testCapacity)
				addStatusDevices(obj, testDriver, testPool, testDevice, testShareID)
				return obj
			}(),
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-fields-consumable-capacity-disabled-feature-gate-with-device-status": {
			oldObj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				modifySpecDeviceRequestWithCapacityRequests(obj, testCapacity, false)
				addDistinctAttribute(obj)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, req0, testShareID, testCapacity)
				return obj
			}(),
			newObj: func() *resource.ResourceClaim { // Status is added with share id and consumed capacities
				obj := obj.DeepCopy()
				modifySpecDeviceRequestWithCapacityRequests(obj, testCapacity, false)
				addDistinctAttribute(obj)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, req0, testShareID, testCapacity)
				addStatusDevices(obj, testDriver, testPool, testDevice, testShareID)
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAResourceClaimDeviceStatus: true, features.DRAConsumableCapacity: false},
			expectObj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				modifySpecDeviceRequestWithCapacityRequests(obj, testCapacity, false)
				addDistinctAttribute(obj)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, req0, testShareID, testCapacity)
				addStatusDevices(obj, testDriver, testPool, testDevice, testShareID)
				return obj
			}(),
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-fields-consumable-capacity-with-device-status-disabled-feature-gate": {
			oldObj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				modifySpecDeviceRequestWithCapacityRequests(obj, testCapacity, false)
				addDistinctAttribute(obj)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, req0, testShareID, testCapacity)
				return obj
			}(),
			newObj: func() *resource.ResourceClaim { // Status should not be added
				obj := obj.DeepCopy()
				modifySpecDeviceRequestWithCapacityRequests(obj, testCapacity, false)
				addDistinctAttribute(obj)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, req0, testShareID, testCapacity)
				addStatusDevices(obj, testDriver, testPool, testDevice, testShareID)
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAResourceClaimDeviceStatus: false, features.DRAConsumableCapacity: true},
			emulatedVersion:  "1.36",
			expectObj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				modifySpecDeviceRequestWithCapacityRequests(obj, testCapacity, false)
				addDistinctAttribute(obj)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, req0, testShareID, testCapacity)
				return obj
			}(),
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-fields-consumable-capacity-with-device-status-with-prioritized-list-disabled-feature-gate": {
			oldObj: func() *resource.ResourceClaim {
				obj := objWithPrioritizedList.DeepCopy()
				modifySpecDeviceRequestWithCapacityRequests(obj, testCapacity, true)
				addDistinctAttribute(obj)
				return obj
			}(),
			newObj: func() *resource.ResourceClaim { // Status is added with share id and consumed capacities but FirstAvailable should not be set
				obj := objWithPrioritizedList.DeepCopy()
				modifySpecDeviceRequestWithCapacityRequests(obj, testCapacity, true)
				addDistinctAttribute(obj)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, req0SubReq0, testShareID, testCapacity)
				addStatusDevices(obj, testDriver, testPool, testDevice, testShareID)
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAResourceClaimDeviceStatus: true, features.DRAConsumableCapacity: true, features.DRAPrioritizedList: false},
			emulatedVersion:  "1.36",
			expectObj: func() *resource.ResourceClaim {
				obj := objWithPrioritizedList.DeepCopy()
				modifySpecDeviceRequestWithCapacityRequests(obj, testCapacity, true)
				addDistinctAttribute(obj)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, req0SubReq0, testShareID, testCapacity)
				addStatusDevices(obj, testDriver, testPool, testDevice, testShareID)
				return obj
			}(),
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"drop-fields-consumable-capacity-disabled-feature-gate": {
			oldObj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, req0)
				return obj
			}(),
			newObj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				modifySpecDeviceRequestWithCapacityRequests(obj, testCapacity, false)
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAConsumableCapacity: false},
			expectObj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, req0)
				return obj
			}(),
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"drop-fields-consumable-capacity-disabled-feature-gate-with-prioritized-list": {
			oldObj: objWithPrioritizedList,
			newObj: func() *resource.ResourceClaim {
				obj := objWithPrioritizedList.DeepCopy()
				modifySpecDeviceRequestWithCapacityRequests(obj, testCapacity, true)
				addDistinctAttribute(obj)
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAPrioritizedList: true, features.DRAConsumableCapacity: false},
			expectObj:        objWithPrioritizedList,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"drop-fields-consumable-capacity-disabled-feature-gate-with-device-status": {
			oldObj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, req0)
				return obj
			}(),
			newObj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				modifySpecDeviceRequestWithCapacityRequests(obj, testCapacity, false)
				addDistinctAttribute(obj)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, req0, testShareID, testCapacity)
				addStatusDevices(obj, testDriver, testPool, testDevice, testShareID)
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAResourceClaimDeviceStatus: true, features.DRAConsumableCapacity: false},
			expectObj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, req0)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, req0, nil, nil)
				addStatusDevices(obj, testDriver, testPool, testDevice, nil)
				return obj
			}(),
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"drop-fields-consumable-capacity-disabled-feature-gate-with-device-status-default-shareid": {
			oldObj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, req0)
				return obj
			}(),
			newObj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, req0)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, req0, nil, nil)
				addStatusDevices(obj, testDriver, testPool, testDevice, nil)
				return obj
			}(),
			featureOverrides: featuregatetesting.FeatureOverrides{features.DRAResourceClaimDeviceStatus: true, features.DRAConsumableCapacity: false},
			expectObj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, req0)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, req0, nil, nil)
				addStatusDevices(obj, testDriver, testPool, testDevice, nil)
				return obj
			}(),
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			fakeClient := fake.NewSimpleClientset(ns1, ns2)
			mockNSClient := fakeClient.CoreV1().Namespaces()
			authz := tc.authz
			if tc.authz == nil {
				authz = &fakeAuthorizer{true}
			}
			strategy := NewStrategy(mockNSClient, authz)

			if tc.emulatedVersion != "" {
				featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse(tc.emulatedVersion))
			}
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, tc.featureOverrides)

			statusStrategy := NewStatusStrategy(strategy)

			ctx := ctx
			if tc.ctxOverride != nil {
				ctx = tc.ctxOverride(ctx)
			}

			oldObj := tc.oldObj.DeepCopy()
			newObj := tc.newObj.DeepCopy()
			newObj.ResourceVersion = "4"

			statusStrategy.PrepareForUpdate(ctx, newObj, oldObj)
			if errs := statusStrategy.ValidateUpdate(ctx, newObj, oldObj); len(errs) != 0 {
				if len(tc.expectValidationErrors) == 0 {
					t.Fatalf("unexpected error(s): %v", errs)
				}
				assert.Len(t, errs, len(tc.expectValidationErrors), "wrong number of validation errors")
				for i, expectValidationError := range tc.expectValidationErrors {
					assert.ErrorContains(t, errs[i], expectValidationError, "the error message should have contained the expected error message")
				}
				return
			}
			if len(tc.expectValidationErrors) != 0 {
				t.Fatal("expected validation error(s), got none")
			}
			if warnings := statusStrategy.WarningsOnUpdate(ctx, newObj, oldObj); len(warnings) != 0 {
				t.Fatalf("unexpected warnings: %q", warnings)
			}
			statusStrategy.Canonicalize(newObj)

			expectObj := tc.expectObj.DeepCopy()
			expectObj.ResourceVersion = "4"
			assert.Equal(t, expectObj, newObj)
			tc.verify(t, fakeClient.Actions())
		})
	}
}

func addSpecDevicesRequest(resourceClaim *resource.ResourceClaim, request string) {
	resourceClaim.Spec.Devices.Requests = append(resourceClaim.Spec.Devices.Requests, resource.DeviceRequest{
		Name: request,
	})
}

func modifySpecDeviceRequestWithCapacityRequests(resourceClaim *resource.ResourceClaim,
	capacity map[resource.QualifiedName]apiresource.Quantity, prioritizedListFeature bool) {
	if capacity != nil {
		if prioritizedListFeature {
			resourceClaim.Spec.Devices.Requests[0].FirstAvailable[0].Capacity = &resource.CapacityRequirements{
				Requests: capacity,
			}
		} else {
			resourceClaim.Spec.Devices.Requests[0].Exactly.Capacity = &resource.CapacityRequirements{
				Requests: capacity,
			}
		}
	}
}

func addDistinctAttribute(resourceClaim *resource.ResourceClaim) {
	distinctConstraint := resource.DeviceConstraint{
		Requests:          []string{req0},
		DistinctAttribute: ptr.To(resource.FullyQualifiedName("driver-a/attr")),
	}
	resourceClaim.Spec.Devices.Constraints = append(resourceClaim.Spec.Devices.Constraints, distinctConstraint)
}

func addStatusAllocationDevicesResults(resourceClaim *resource.ResourceClaim, driver string, pool string, device string, request string,
	shareID *types.UID, consumedCapacity map[resource.QualifiedName]apiresource.Quantity) {
	if resourceClaim.Status.Allocation == nil {
		resourceClaim.Status.Allocation = &resource.AllocationResult{}
	}
	resourceClaim.Status.Allocation.Devices.Results = append(resourceClaim.Status.Allocation.Devices.Results, resource.DeviceRequestAllocationResult{
		Request:          request,
		Driver:           driver,
		Pool:             pool,
		Device:           device,
		ShareID:          shareID,
		ConsumedCapacity: consumedCapacity,
	})
}

func addStatusDevices(resourceClaim *resource.ResourceClaim, driver string, pool string, device string, shareID *types.UID) {
	resourceClaim.Status.Devices = append(resourceClaim.Status.Devices, resource.AllocatedDeviceStatus{
		Driver:  driver,
		Pool:    pool,
		Device:  device,
		ShareID: (*string)(shareID),
	})
}

type fakeAuthorizer struct {
	verdict bool
}

func (f *fakeAuthorizer) Authorize(ctx context.Context, a authorizer.Attributes) (authorizer.Decision, string, error) {
	if !f.verdict {
		return authorizer.DecisionDeny, "denied", nil
	}
	return authorizer.DecisionAllow, "default accept", nil
}
