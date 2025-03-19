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
	"testing"

	"github.com/stretchr/testify/assert"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes/fake"
	testclient "k8s.io/client-go/testing"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
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
					Name:            "req-0",
					DeviceClassName: "class",
					AllocationMode:  resource.DeviceAllocationModeAll,
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
					Name:            "req-0",
					DeviceClassName: "class",
					AllocationMode:  resource.DeviceAllocationModeAll,
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
					Name:            "req-0",
					DeviceClassName: "class",
					AllocationMode:  resource.DeviceAllocationModeAll,
					AdminAccess:     ptr.To(true),
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
					Name:            "req-0",
					DeviceClassName: "class",
					AllocationMode:  resource.DeviceAllocationModeAll,
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
					Name:            "req-0",
					DeviceClassName: "class",
					AllocationMode:  resource.DeviceAllocationModeAll,
					AdminAccess:     ptr.To(true),
				},
			},
		},
	},
}

var objStatusInNonAdminNamespace = &resource.ResourceClaim{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "valid-claim",
		Namespace: "default",
	},
	Spec: resource.ResourceClaimSpec{
		Devices: resource.DeviceClaim{
			Requests: []resource.DeviceRequest{
				{
					Name:            "req-0",
					DeviceClassName: "class",
					AllocationMode:  resource.DeviceAllocationModeAll,
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
var objWithAdminAccessStatusInNonAdminNamespace = &resource.ResourceClaim{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "valid-claim",
		Namespace: "default",
	},
	Spec: resource.ResourceClaimSpec{
		Devices: resource.DeviceClaim{
			Requests: []resource.DeviceRequest{
				{
					Name:            "req-0",
					DeviceClassName: "class",
					AllocationMode:  resource.DeviceAllocationModeAll,
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

var objWithPrioritizedList = &resource.ResourceClaim{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "valid-claim",
		Namespace: "default",
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

var objWithAdminAccessStatus = &resource.ResourceClaim{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "valid-claim",
		Namespace: "kube-system",
	},
	Spec: resource.ResourceClaimSpec{
		Devices: resource.DeviceClaim{
			Requests: []resource.DeviceRequest{
				{
					Name:            "req-0",
					DeviceClassName: "class",
					AllocationMode:  resource.DeviceAllocationModeAll,
					AdminAccess:     ptr.To(true),
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

var adminAccessError = "Forbidden: admin access to devices requires the `resource.k8s.io/admin-access: true` label"
var fieldImmutableError = "field is immutable"
var metadataError = "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters"
var deviceRequestError = "exactly one of `deviceClassName` or `firstAvailable` must be specified"

const (
	testRequest = "test-request"
	testDriver  = "test-driver"
	testPool    = "test-pool"
	testDevice  = "test-device"
)

func TestStrategy(t *testing.T) {
	fakeClient := fake.NewSimpleClientset()
	mockNSClient := fakeClient.CoreV1().Namespaces()
	strategy := NewStrategy(mockNSClient)
	if !strategy.NamespaceScoped() {
		t.Errorf("ResourceClaim must be namespace scoped")
	}
	if strategy.AllowCreateOnUpdate() {
		t.Errorf("ResourceClaim should not allow create on update")
	}
}

func TestStrategyCreate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	testcases := map[string]struct {
		obj                   *resource.ResourceClaim
		adminAccess           bool
		prioritizedList       bool
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
			obj:         objWithAdminAccess,
			adminAccess: false,
			expectObj:   obj,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-fields-admin-access": {
			obj:         objWithAdminAccess,
			adminAccess: true,
			expectObj:   objWithAdminAccess,
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
		"drop-fields-prioritized-list": {
			obj:                   objWithPrioritizedList,
			prioritizedList:       false,
			expectValidationError: deviceRequestError,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-fields-prioritized-list": {
			obj:             objWithPrioritizedList,
			prioritizedList: true,
			expectObj:       objWithPrioritizedList,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"admin-access-admin-namespace": {
			obj:         objWithAdminAccess,
			adminAccess: true,
			expectObj:   objWithAdminAccess,
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
			adminAccess:           true,
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
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			fakeClient := fake.NewSimpleClientset(ns1, ns2)
			mockNSClient := fakeClient.CoreV1().Namespaces()

			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAAdminAccess, tc.adminAccess)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAPrioritizedList, tc.prioritizedList)
			strategy := NewStrategy(mockNSClient)

			obj := tc.obj.DeepCopy()
			strategy.PrepareForCreate(ctx, obj)
			if errs := strategy.Validate(ctx, obj); len(errs) != 0 {
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
		adminAccess           bool
		expectValidationError string
		prioritizedList       bool
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
			oldObj:      obj,
			newObj:      objWithAdminAccess,
			adminAccess: false,
			expectObj:   obj,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-fields-admin-access": {
			oldObj:                obj,
			newObj:                objWithAdminAccess,
			adminAccess:           true,
			expectValidationError: fieldImmutableError, // Spec is immutable.
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-existing-fields-admin-access": {
			oldObj:      objWithAdminAccess,
			newObj:      objWithAdminAccess,
			adminAccess: true,
			expectObj:   objWithAdminAccess,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"admin-access-admin-namespace": {
			oldObj:      objWithAdminAccess,
			newObj:      objWithAdminAccess,
			adminAccess: true,
			expectObj:   objWithAdminAccess,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"admin-access-non-admin-namespace": {
			oldObj:                objInNonAdminNamespace,
			newObj:                objWithAdminAccessInNonAdminNamespace,
			adminAccess:           true,
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
			prioritizedList:       false,
			expectValidationError: deviceRequestError,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-fields-prioritized-list": {
			oldObj:                obj,
			newObj:                objWithPrioritizedList,
			prioritizedList:       true,
			expectValidationError: fieldImmutableError, // Spec is immutable.
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-existing-fields-prioritized-list": {
			oldObj:          objWithPrioritizedList,
			newObj:          objWithPrioritizedList,
			prioritizedList: true,
			expectObj:       objWithPrioritizedList,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-existing-fields-prioritized-list-disabled-feature": {
			oldObj:          objWithPrioritizedList,
			newObj:          objWithPrioritizedList,
			prioritizedList: false,
			expectObj:       objWithPrioritizedList,
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

			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAAdminAccess, tc.adminAccess)
			strategy := NewStrategy(mockNSClient)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAPrioritizedList, tc.prioritizedList)

			oldObj := tc.oldObj.DeepCopy()
			newObj := tc.newObj.DeepCopy()
			newObj.ResourceVersion = "4"

			strategy.PrepareForUpdate(ctx, newObj, oldObj)
			if errs := strategy.ValidateUpdate(ctx, newObj, oldObj); len(errs) != 0 {
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

func TestStatusStrategyUpdate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	testcases := map[string]struct {
		oldObj                  *resource.ResourceClaim
		newObj                  *resource.ResourceClaim
		adminAccess             bool
		deviceStatusFeatureGate bool
		expectValidationError   string
		expectObj               *resource.ResourceClaim
		verify                  func(*testing.T, []testclient.Action)
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
			oldObj:      obj,
			newObj:      objWithAdminAccessStatus,
			adminAccess: false,
			expectObj:   objWithStatus,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-fields-admin-access": {
			oldObj:      obj,
			newObj:      objWithAdminAccessStatus,
			adminAccess: true,
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
			oldObj:                objStatusInNonAdminNamespace,
			newObj:                objWithAdminAccessStatusInNonAdminNamespace,
			adminAccess:           true,
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
		"keep-fields-admin-access-because-of-spec": {
			oldObj:      objWithAdminAccess,
			newObj:      objWithAdminAccessStatus,
			adminAccess: false,
			expectObj:   objWithAdminAccessStatus,
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
				oldObj.Spec.Devices.Requests[0].AdminAccess = ptr.To(false)
				return oldObj
			}(),
			newObj:      objWithAdminAccessStatus,
			adminAccess: false,
			expectObj: func() *resource.ResourceClaim {
				oldObj := objWithAdminAccessStatus.DeepCopy()
				oldObj.Spec.Devices.Requests[0].AdminAccess = ptr.To(false)
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
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, testRequest)
				return obj
			}(),
			newObj: func() *resource.ResourceClaim { // Status is added
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, testRequest)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, testRequest)
				addStatusDevices(obj, testDriver, testPool, testDevice)
				return obj
			}(),
			deviceStatusFeatureGate: false,
			expectObj: func() *resource.ResourceClaim { // Status is no longer there
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, testRequest)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, testRequest)
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
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, testRequest)
				addStatusDevices(obj, testDriver, testPool, testDevice)
				return obj
			}(),
			newObj: func() *resource.ResourceClaim { // Status is added
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, testRequest)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, testRequest)
				addStatusDevices(obj, testDriver, testPool, testDevice)
				return obj
			}(),
			deviceStatusFeatureGate: false,
			expectObj: func() *resource.ResourceClaim { // Status is still there (as the status was set in the old object)
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, testRequest)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, testRequest)
				addStatusDevices(obj, testDriver, testPool, testDevice)
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
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, testRequest)
				return obj
			}(),
			newObj: func() *resource.ResourceClaim { // Status is added
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, testRequest)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, testRequest)
				addStatusDevices(obj, testDriver, testPool, testDevice)
				return obj
			}(),
			deviceStatusFeatureGate: true,
			expectObj: func() *resource.ResourceClaim { // Status is still there
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, testRequest)
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, testRequest)
				addStatusDevices(obj, testDriver, testPool, testDevice)
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
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, testRequest)
				addStatusDevices(obj, testDriver, testPool, testDevice)
				return obj
			}(),
			newObj: func() *resource.ResourceClaim { // device is deallocated
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, testRequest)
				addStatusDevices(obj, testDriver, testPool, testDevice)
				return obj
			}(),
			deviceStatusFeatureGate: true,
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
				addStatusAllocationDevicesResults(obj, testDriver, testPool, testDevice, testRequest)
				addStatusDevices(obj, testDriver, testPool, testDevice)
				return obj
			}(),
			newObj: func() *resource.ResourceClaim { // device is deallocated
				obj := obj.DeepCopy()
				addSpecDevicesRequest(obj, testRequest)
				addStatusDevices(obj, testDriver, testPool, testDevice)
				return obj
			}(),
			deviceStatusFeatureGate: false,
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
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			fakeClient := fake.NewSimpleClientset(ns1, ns2)
			mockNSClient := fakeClient.CoreV1().Namespaces()
			strategy := NewStrategy(mockNSClient)

			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAAdminAccess, tc.adminAccess)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAResourceClaimDeviceStatus, tc.deviceStatusFeatureGate)
			statusStrategy := NewStatusStrategy(strategy)

			oldObj := tc.oldObj.DeepCopy()
			newObj := tc.newObj.DeepCopy()
			newObj.ResourceVersion = "4"

			statusStrategy.PrepareForUpdate(ctx, newObj, oldObj)
			if errs := statusStrategy.ValidateUpdate(ctx, newObj, oldObj); len(errs) != 0 {
				assert.ErrorContains(t, errs[0], tc.expectValidationError, "the error message should have contained the expected error message")
				return
			}
			if tc.expectValidationError != "" {
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

func addStatusAllocationDevicesResults(resourceClaim *resource.ResourceClaim, driver string, pool string, device string, request string) {
	if resourceClaim.Status.Allocation == nil {
		resourceClaim.Status.Allocation = &resource.AllocationResult{}
	}
	resourceClaim.Status.Allocation.Devices.Results = append(resourceClaim.Status.Allocation.Devices.Results, resource.DeviceRequestAllocationResult{
		Request: request,
		Driver:  driver,
		Pool:    pool,
		Device:  device,
	})
}

func addStatusDevices(resourceClaim *resource.ResourceClaim, driver string, pool string, device string) {
	resourceClaim.Status.Devices = append(resourceClaim.Status.Devices, resource.AllocatedDeviceStatus{
		Driver: driver,
		Pool:   pool,
		Device: device,
	})
}
