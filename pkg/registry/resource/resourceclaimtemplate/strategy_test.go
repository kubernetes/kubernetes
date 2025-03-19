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
						Name:            "req-0",
						DeviceClassName: "class",
						AllocationMode:  resource.DeviceAllocationModeAll,
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
						Name:            "req-0",
						DeviceClassName: "class",
						AllocationMode:  resource.DeviceAllocationModeAll,
						AdminAccess:     ptr.To(true),
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
						Name:            "req-0",
						DeviceClassName: "class",
						AllocationMode:  resource.DeviceAllocationModeAll,
						AdminAccess:     ptr.To(true),
					},
				},
			},
		},
	},
}

var objWithPrioritizedList = &resource.ResourceClaimTemplate{
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
var adminAccessError = "Forbidden: admin access to devices requires the `resource.k8s.io/admin-access: true` label on the containing namespace"
var fieldImmutableError = "field is immutable"
var metadataError = "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters"
var deviceRequestError = "exactly one of `deviceClassName` or `firstAvailable` must be specified"

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
		expectValidationError string
		prioritizedList       bool
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
		newClaimTemplate.Spec.Spec.Devices.Requests[0].AdminAccess = ptr.To(true)

		strategy.PrepareForUpdate(ctx, newClaimTemplate, resourceClaimTemplate)
		errs := strategy.ValidateUpdate(ctx, newClaimTemplate, resourceClaimTemplate)
		if len(errs) != 0 {
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
