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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/kubernetes/pkg/features"
)

var obj = &resource.ResourceClaim{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "valid-claim",
		Namespace: "default",
	},
}

var testRequest = "test-request"
var testDriver = "test-driver"
var testPool = "test-pool"
var testDevice = "test-device"

func TestStrategy(t *testing.T) {
	if !Strategy.NamespaceScoped() {
		t.Errorf("ResourceClaim must be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("ResourceClaim should not allow create on update")
	}
}

func TestStrategyCreate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()

	testcases := map[string]struct {
		obj                   *resource.ResourceClaim
		expectValidationError bool
		expectObj             *resource.ResourceClaim
	}{
		"simple": {
			obj:       obj,
			expectObj: obj,
		},
		"validation-error": {
			obj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				obj.Name = "%#@$%$"
				return obj
			}(),
			expectValidationError: true,
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			obj := tc.obj.DeepCopy()
			Strategy.PrepareForCreate(ctx, obj)
			if errs := Strategy.Validate(ctx, obj); len(errs) != 0 {
				if !tc.expectValidationError {
					t.Fatalf("unexpected validation errors: %q", errs)
				}
				return
			} else if tc.expectValidationError {
				t.Fatal("expected validation error(s), got none")
			}
			if warnings := Strategy.WarningsOnCreate(ctx, obj); len(warnings) != 0 {
				t.Fatalf("unexpected warnings: %q", warnings)
			}
			Strategy.Canonicalize(obj)
			assert.Equal(t, tc.expectObj, obj)
		})
	}
}

func TestStrategyUpdate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()

	testcases := map[string]struct {
		oldObj                *resource.ResourceClaim
		newObj                *resource.ResourceClaim
		expectValidationError bool
		expectObj             *resource.ResourceClaim
	}{
		"no-changes-okay": {
			oldObj:    obj,
			newObj:    obj,
			expectObj: obj,
		},
		"name-change-not-allowed": {
			oldObj: obj,
			newObj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				obj.Name += "-2"
				return obj
			}(),
			expectValidationError: true,
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			oldObj := tc.oldObj.DeepCopy()
			newObj := tc.newObj.DeepCopy()
			newObj.ResourceVersion = "4"

			Strategy.PrepareForUpdate(ctx, newObj, oldObj)
			if errs := Strategy.ValidateUpdate(ctx, newObj, oldObj); len(errs) != 0 {
				if !tc.expectValidationError {
					t.Fatalf("unexpected validation errors: %q", errs)
				}
				return
			} else if tc.expectValidationError {
				t.Fatal("expected validation error(s), got none")
			}
			if warnings := Strategy.WarningsOnUpdate(ctx, newObj, oldObj); len(warnings) != 0 {
				t.Fatalf("unexpected warnings: %q", warnings)
			}
			Strategy.Canonicalize(newObj)

			expectObj := tc.expectObj.DeepCopy()
			expectObj.ResourceVersion = "4"
			assert.Equal(t, expectObj, newObj)
		})
	}
}

func TestStatusStrategyUpdate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()

	testcases := map[string]struct {
		oldObj                  *resource.ResourceClaim
		newObj                  *resource.ResourceClaim
		expectValidationError   bool
		expectObj               *resource.ResourceClaim
		deviceStatusFeatureGate bool
	}{
		"no-changes-okay": {
			oldObj:    obj,
			newObj:    obj,
			expectObj: obj,
		},
		"name-change-not-allowed": {
			oldObj: obj,
			newObj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				obj.Name += "-2"
				return obj
			}(),
			expectValidationError: true,
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
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAResourceClaimDeviceStatus, tc.deviceStatusFeatureGate)

			oldObj := tc.oldObj.DeepCopy()
			newObj := tc.newObj.DeepCopy()
			newObj.ResourceVersion = "4"

			StatusStrategy.PrepareForUpdate(ctx, newObj, oldObj)
			if errs := StatusStrategy.ValidateUpdate(ctx, newObj, oldObj); len(errs) != 0 {
				if !tc.expectValidationError {
					t.Fatalf("unexpected validation errors: %q", errs)
				}
				return
			} else if tc.expectValidationError {
				t.Fatal("expected validation error(s), got none")
			}
			if warnings := StatusStrategy.WarningsOnUpdate(ctx, newObj, oldObj); len(warnings) != 0 {
				t.Fatalf("unexpected warnings: %q", warnings)
			}
			StatusStrategy.Canonicalize(newObj)

			expectObj := tc.expectObj.DeepCopy()
			expectObj.ResourceVersion = "4"
			assert.Equal(t, expectObj, newObj)
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
