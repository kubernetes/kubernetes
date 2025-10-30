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

package resourceclaimtemplate

import (
	"testing"

	"github.com/stretchr/testify/assert"

	corev1 "k8s.io/api/core/v1"
	apiresource "k8s.io/apimachinery/pkg/api/resource"
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

var obj = &resource.ResourceClaimTemplate{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "valid-claim-template",
		Namespace: "kube-system",
	},
	Spec: resource.ResourceClaimTemplateSpec{
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
	},
}

var objWithAdminAccess = &resource.ResourceClaimTemplate{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "valid-claim-template",
		Namespace: "kube-system",
	},
	Spec: resource.ResourceClaimTemplateSpec{
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
	},
}

var objInNonAdminNamespace = &resource.ResourceClaimTemplate{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "valid-claim-template",
		Namespace: "default",
	},
	Spec: resource.ResourceClaimTemplateSpec{
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
	},
}

var objWithAdminAccessInNonAdminNamespace = &resource.ResourceClaimTemplate{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "valid-claim-template",
		Namespace: "default",
	},
	Spec: resource.ResourceClaimTemplateSpec{
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
	},
}

var objWithDeviceTaints = &resource.ResourceClaimTemplate{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "valid-claim-template",
		Namespace: "kube-system",
	},
	Spec: resource.ResourceClaimTemplateSpec{
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
	},
}

var objWithPrioritizedList = &resource.ResourceClaimTemplate{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "valid-claim-template",
		Namespace: "kube-system",
	},
	Spec: resource.ResourceClaimTemplateSpec{
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
	},
}

var testCapacity = map[resource.QualifiedName]apiresource.Quantity{
	resource.QualifiedName("test-capacity"): apiresource.MustParse("1"),
}

var objWithCapacityRequests = func() *resource.ResourceClaimTemplate {
	obj := obj.DeepCopy()
	addSpecDeviceRequestWithCapacityRequests(obj, testCapacity, false)
	return obj
}()

func addSpecDeviceRequestWithCapacityRequests(resourceClaimTemplate *resource.ResourceClaimTemplate,
	capacity map[resource.QualifiedName]apiresource.Quantity, prioritizedListFeature bool) {
	r := resource.DeviceRequest{
		Name: "cap-req-0",
	}
	if prioritizedListFeature {
		r.FirstAvailable = []resource.DeviceSubRequest{
			{
				Name:            "subreq-0",
				DeviceClassName: "class",
				AllocationMode:  resource.DeviceAllocationModeExactCount,
				Count:           1,
			},
		}
	} else {
		r.Exactly = &resource.ExactDeviceRequest{
			DeviceClassName: "class",
			AllocationMode:  resource.DeviceAllocationModeAll,
		}
	}
	if capacity != nil {
		if prioritizedListFeature {
			r.FirstAvailable[0].Capacity = &resource.CapacityRequirements{
				Requests: capacity,
			}
		} else {
			r.Exactly.Capacity = &resource.CapacityRequirements{
				Requests: capacity,
			}
		}
	}
	resourceClaimTemplate.Spec.Spec.Devices.Requests = append(resourceClaimTemplate.Spec.Spec.Devices.Requests, r)
}

var objWithDeviceTaintsInPrioritizedList = &resource.ResourceClaimTemplate{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "valid-claim-template",
		Namespace: "kube-system",
	},
	Spec: resource.ResourceClaimTemplateSpec{
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
var adminAccessError = "Forbidden: admin access to devices requires the `resource.kubernetes.io/admin-access: true` label on the containing namespace"
var fieldImmutableError = "field is immutable"
var metadataError = "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters"
var deviceRequestError = "exactly one of `exactly` or `firstAvailable` is required"

func TestClaimTemplateStrategy(t *testing.T) {
	fakeClient := fake.NewSimpleClientset()
	mockNSClient := fakeClient.CoreV1().Namespaces()
	strategy := NewStrategy(mockNSClient)

	if !strategy.NamespaceScoped() {
		t.Errorf("ResourceClaimTemplate must be namespace scoped")
	}
	if strategy.AllowCreateOnUpdate() {
		t.Errorf("ResourceClaimTemplate should not allow create on update")
	}
}

func TestClaimTemplateStrategyCreate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()

	testcases := map[string]struct {
		obj                   *resource.ResourceClaimTemplate
		adminAccess           bool
		deviceTaints          bool
		prioritizedList       bool
		consumableCapacity    bool
		expectValidationError string
		expectObj             *resource.ResourceClaimTemplate
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
			obj: func() *resource.ResourceClaimTemplate {
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
		"drop-fields-device-taints": {
			obj:          objWithDeviceTaints,
			deviceTaints: false,
			expectObj:    obj,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-fields-device-taints": {
			obj:          objWithDeviceTaints,
			deviceTaints: true,
			expectObj:    objWithDeviceTaints,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"drop-fields-device-taints-in-prioritized-list": {
			obj:             objWithDeviceTaintsInPrioritizedList,
			deviceTaints:    false,
			prioritizedList: true,
			expectObj:       objWithPrioritizedList,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-fields-device-taints-in-prioritized-list": {
			obj:             objWithDeviceTaintsInPrioritizedList,
			deviceTaints:    true,
			prioritizedList: true,
			expectObj:       objWithDeviceTaintsInPrioritizedList,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
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
		"keep-consumable-capacity-fields": {
			obj:                objWithCapacityRequests,
			consumableCapacity: true,
			expectObj:          objWithCapacityRequests,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"drop-consumable-capacity-fields-disabled-feature": {
			obj:                objWithCapacityRequests,
			consumableCapacity: false,
			expectObj: func() *resource.ResourceClaimTemplate {
				obj := obj.DeepCopy()
				addSpecDeviceRequestWithCapacityRequests(obj, nil, false)
				return obj
			}(),
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"drop-consumable-capacity-fields-disabled-feature-with-prioritized-list": {
			obj: func() *resource.ResourceClaimTemplate {
				obj := obj.DeepCopy()
				addSpecDeviceRequestWithCapacityRequests(obj, testCapacity, true)
				return obj
			}(),
			consumableCapacity: false,
			prioritizedList:    true,
			expectObj: func() *resource.ResourceClaimTemplate {
				obj := obj.DeepCopy()
				addSpecDeviceRequestWithCapacityRequests(obj, nil, true)
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
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.DRAAdminAccess:        tc.adminAccess,
				features.DRADeviceTaints:       tc.deviceTaints,
				features.DRAPrioritizedList:    tc.prioritizedList,
				features.DRAConsumableCapacity: tc.consumableCapacity,
			})
			strategy := NewStrategy(mockNSClient)

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

func TestClaimTemplateStrategyUpdate(t *testing.T) {
	t.Run("no-changes-okay", func(t *testing.T) {
		ctx := genericapirequest.NewDefaultContext()
		fakeClient := fake.NewSimpleClientset(ns1, ns2)
		mockNSClient := fakeClient.CoreV1().Namespaces()

		strategy := NewStrategy(mockNSClient)
		resourceClaimTemplate := obj.DeepCopy()
		newClaimTemplate := resourceClaimTemplate.DeepCopy()
		newClaimTemplate.ResourceVersion = "4"

		strategy.PrepareForUpdate(ctx, newClaimTemplate, resourceClaimTemplate)
		errs := strategy.ValidateUpdate(ctx, newClaimTemplate, resourceClaimTemplate)
		if len(errs) != 0 {
			t.Errorf("unexpected validation errors: %v", errs)
		}
		if len(fakeClient.Actions()) != 0 {
			t.Errorf("expected no action to be taken")
		}
	})

	t.Run("name-change-not-allowed", func(t *testing.T) {
		ctx := genericapirequest.NewDefaultContext()
		fakeClient := fake.NewSimpleClientset(ns1, ns2)
		mockNSClient := fakeClient.CoreV1().Namespaces()
		strategy := NewStrategy(mockNSClient)
		resourceClaimTemplate := obj.DeepCopy()
		newClaimTemplate := resourceClaimTemplate.DeepCopy()
		newClaimTemplate.Name = "valid-class-2"
		newClaimTemplate.ResourceVersion = "4"

		strategy.PrepareForUpdate(ctx, newClaimTemplate, resourceClaimTemplate)
		errs := strategy.ValidateUpdate(ctx, newClaimTemplate, resourceClaimTemplate)
		if len(errs) == 0 {
			t.Errorf("expected a validation error")
		}
		if len(fakeClient.Actions()) != 0 {
			t.Errorf("expected no action to be taken")
		}
	})

	t.Run("adminaccess-update-not-allowed", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAAdminAccess, true)
		ctx := genericapirequest.NewDefaultContext()
		fakeClient := fake.NewSimpleClientset(ns1, ns2)
		mockNSClient := fakeClient.CoreV1().Namespaces()
		strategy := NewStrategy(mockNSClient)
		resourceClaimTemplate := obj.DeepCopy()
		newClaimTemplate := resourceClaimTemplate.DeepCopy()
		newClaimTemplate.ResourceVersion = "4"
		newClaimTemplate.Spec.Spec.Devices.Requests[0].Exactly.AdminAccess = ptr.To(true)

		strategy.PrepareForUpdate(ctx, newClaimTemplate, resourceClaimTemplate)
		errs := strategy.ValidateUpdate(ctx, newClaimTemplate, resourceClaimTemplate)
		if len(errs) != 0 {
			if fieldImmutableError == "" {
				t.Fatalf("unexpected error(s): %v", errs)
			}
			assert.ErrorContains(t, errs[0], fieldImmutableError, "the error message should have contained the expected error message")
			return
		}
		if len(errs) == 0 {
			t.Errorf("expected a validation error")
		}
		if len(fakeClient.Actions()) != 0 {
			t.Errorf("expected no action to be taken")
		}
	})
}

func TestStrategyUpdate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	testcases := map[string]struct {
		oldObj                 *resource.ResourceClaimTemplate
		newObj                 *resource.ResourceClaimTemplate
		adminAccess            bool
		deviceTaints           bool
		prioritizedList        bool
		expectValidationErrors []string
		expectObj              *resource.ResourceClaimTemplate
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
			newObj: func() *resource.ResourceClaimTemplate {
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
			oldObj:                 obj,
			newObj:                 objWithAdminAccess,
			adminAccess:            true,
			expectValidationErrors: []string{fieldImmutableError}, // Spec is immutable.
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
			oldObj:                 objInNonAdminNamespace,
			newObj:                 objWithAdminAccessInNonAdminNamespace,
			adminAccess:            true,
			expectValidationErrors: []string{fieldImmutableError},
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"drop-fields-prioritized-list": {
			oldObj:                 obj,
			newObj:                 objWithPrioritizedList,
			prioritizedList:        false,
			expectValidationErrors: []string{deviceRequestError, fieldImmutableError},
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-fields-prioritized-list": {
			oldObj:                 obj,
			newObj:                 objWithPrioritizedList,
			prioritizedList:        true,
			expectValidationErrors: []string{fieldImmutableError}, // Spec is immutable.
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
		"drop-fields-device-taints": {
			oldObj:          obj,
			newObj:          objWithDeviceTaints,
			deviceTaints:    false,
			prioritizedList: true,
			expectObj:       obj,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-fields-device-taints": {
			oldObj:                 obj,
			newObj:                 objWithDeviceTaints,
			deviceTaints:           true,
			prioritizedList:        true,
			expectValidationErrors: []string{fieldImmutableError}, // Spec is immutable, cannot add tolerations.
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-existing-fields-device-taints": {
			oldObj:          objWithDeviceTaints,
			newObj:          objWithDeviceTaints,
			deviceTaints:    true,
			prioritizedList: true,
			expectObj:       objWithDeviceTaints,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-existing-fields-device-taints-disabled-feature": {
			oldObj:          objWithDeviceTaints,
			newObj:          objWithDeviceTaints,
			deviceTaints:    false,
			prioritizedList: true,
			expectObj:       objWithDeviceTaints,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"drop-fields-device-taints-in-prioritized-list": {
			oldObj:          objWithPrioritizedList,
			newObj:          objWithDeviceTaintsInPrioritizedList,
			deviceTaints:    false,
			prioritizedList: true,
			expectObj:       objWithPrioritizedList,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-fields-device-taints-in-prioritized-list": {
			oldObj:                 objWithPrioritizedList,
			newObj:                 objWithDeviceTaintsInPrioritizedList,
			deviceTaints:           true,
			prioritizedList:        true,
			expectValidationErrors: []string{fieldImmutableError}, // Spec is immutable, cannot add tolerations.
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-existing-fields-device-taints-in-prioritized-list": {
			oldObj:          objWithDeviceTaintsInPrioritizedList,
			newObj:          objWithDeviceTaintsInPrioritizedList,
			deviceTaints:    true,
			prioritizedList: true,
			expectObj:       objWithDeviceTaintsInPrioritizedList,
			verify: func(t *testing.T, as []testclient.Action) {
				if len(as) != 0 {
					t.Errorf("expected no action to be taken")
				}
			},
		},
		"keep-existing-fields-device-taints-in-prioritized-list-disabled-feature": {
			oldObj:          objWithDeviceTaintsInPrioritizedList,
			newObj:          objWithDeviceTaintsInPrioritizedList,
			deviceTaints:    false,
			prioritizedList: true,
			expectObj:       objWithDeviceTaintsInPrioritizedList,
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
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRADeviceTaints, tc.deviceTaints)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAPrioritizedList, tc.prioritizedList)
			strategy := NewStrategy(mockNSClient)

			oldObj := tc.oldObj.DeepCopy()
			newObj := tc.newObj.DeepCopy()
			newObj.ResourceVersion = "4"

			strategy.PrepareForUpdate(ctx, newObj, oldObj)
			expectedErrLen := len(tc.expectValidationErrors)
			if errs := strategy.ValidateUpdate(ctx, newObj, oldObj); len(errs) != 0 {
				if assert.Len(t, errs, expectedErrLen, "exact number of errors expected") {
					for i, expectErr := range tc.expectValidationErrors {
						assert.ErrorContains(t, errs[i], expectErr, "the error message should have contained the expected error message")
					}
					return
				}
			}
			if expectedErrLen > 0 {
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
