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

var objWithStatus = &resource.ResourceClaim{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "valid-claim",
		Namespace: "default",
	},
	Status: resource.ResourceClaimStatus{
		Allocation: &resource.AllocationResult{},
	},
}

var objWithGatedFields = &resource.ResourceClaim{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "valid-claim",
		Namespace: "default",
	},
	Spec: resource.ResourceClaimSpec{
		Controller: "dra.example.com",
	},
}

var objWithGatedStatusFields = &resource.ResourceClaim{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "valid-claim",
		Namespace: "default",
	},
	Spec: resource.ResourceClaimSpec{
		Controller: "dra.example.com",
	},
	Status: resource.ResourceClaimStatus{
		Allocation: &resource.AllocationResult{
			Controller: "dra.example.com",
		},
		DeallocationRequested: true,
	},
}

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
		obj                    *resource.ResourceClaim
		controlPlaneController bool
		expectValidationError  bool
		expectObj              *resource.ResourceClaim
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
		"drop-fields": {
			obj:                    objWithGatedFields,
			controlPlaneController: false,
			expectObj:              obj,
		},
		"keep-fields": {
			obj:                    objWithGatedFields,
			controlPlaneController: true,
			expectObj:              objWithGatedFields,
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAControlPlaneController, tc.controlPlaneController)

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
		oldObj                 *resource.ResourceClaim
		newObj                 *resource.ResourceClaim
		controlPlaneController bool
		expectValidationError  bool
		expectObj              *resource.ResourceClaim
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
		"drop-fields": {
			oldObj:                 obj,
			newObj:                 objWithGatedFields,
			controlPlaneController: false,
			expectObj:              obj,
		},
		"keep-fields": {
			oldObj:                 obj,
			newObj:                 objWithGatedFields,
			controlPlaneController: true,
			expectValidationError:  true, // Spec is immutable.
		},
		"keep-existing-fields": {
			oldObj:                 objWithGatedFields,
			newObj:                 objWithGatedFields,
			controlPlaneController: false,
			expectObj:              objWithGatedFields,
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAControlPlaneController, tc.controlPlaneController)
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
		oldObj                 *resource.ResourceClaim
		newObj                 *resource.ResourceClaim
		controlPlaneController bool
		expectValidationError  bool
		expectObj              *resource.ResourceClaim
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
		"drop-fields": {
			oldObj:                 obj,
			newObj:                 objWithGatedStatusFields,
			controlPlaneController: false,
			expectObj:              objWithStatus,
		},
		"keep-fields": {
			oldObj:                 obj,
			newObj:                 objWithGatedStatusFields,
			controlPlaneController: true,
			expectObj: func() *resource.ResourceClaim {
				expectObj := objWithGatedStatusFields.DeepCopy()
				// Spec remains unchanged.
				expectObj.Spec = obj.Spec
				return expectObj
			}(),
		},
		"keep-fields-because-of-spec": {
			oldObj:                 objWithGatedFields,
			newObj:                 objWithGatedStatusFields,
			controlPlaneController: false,
			expectObj:              objWithGatedStatusFields,
		},
		// Normally a claim without a controller in the spec shouldn't
		// have one in the status either, but it's not invalid and thus
		// let's test this.
		"keep-fields-because-of-status": {
			oldObj: func() *resource.ResourceClaim {
				oldObj := objWithGatedStatusFields.DeepCopy()
				oldObj.Spec.Controller = ""
				return oldObj
			}(),
			newObj:                 objWithGatedStatusFields,
			controlPlaneController: false,
			expectObj: func() *resource.ResourceClaim {
				oldObj := objWithGatedStatusFields.DeepCopy()
				oldObj.Spec.Controller = ""
				return oldObj
			}(),
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAControlPlaneController, tc.controlPlaneController)
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
