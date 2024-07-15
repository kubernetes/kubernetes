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

package deviceclass

import (
	"testing"

	"github.com/stretchr/testify/assert"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/kubernetes/pkg/features"
)

var obj = &resource.DeviceClass{
	ObjectMeta: metav1.ObjectMeta{
		Name:       "valid-class",
		Generation: 1,
	},
}

var objWithGatedFields = &resource.DeviceClass{
	ObjectMeta: metav1.ObjectMeta{
		Name:       "valid-class",
		Generation: 1,
	},
	Spec: resource.DeviceClassSpec{
		SuitableNodes: &core.NodeSelector{
			NodeSelectorTerms: []core.NodeSelectorTerm{{
				MatchExpressions: []core.NodeSelectorRequirement{{
					Key:      "foo",
					Operator: core.NodeSelectorOpExists,
				}},
			}},
		},
	},
}

func TestStrategy(t *testing.T) {
	if Strategy.NamespaceScoped() {
		t.Errorf("DeviceClass must not be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("DeviceClass should not allow create on update")
	}
}

func TestStrategyCreate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()

	testcases := map[string]struct {
		obj                    *resource.DeviceClass
		controlPlaneController bool
		expectValidationError  bool
		expectObj              *resource.DeviceClass
	}{
		"simple": {
			obj:       obj,
			expectObj: obj,
		},
		"validation-error": {
			obj: func() *resource.DeviceClass {
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
		oldObj                 *resource.DeviceClass
		newObj                 *resource.DeviceClass
		controlPlaneController bool
		expectValidationError  bool
		expectObj              *resource.DeviceClass
	}{
		"no-changes-okay": {
			oldObj:    obj,
			newObj:    obj,
			expectObj: obj,
		},
		"name-change-not-allowed": {
			oldObj: obj,
			newObj: func() *resource.DeviceClass {
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
			expectObj: func() *resource.DeviceClass {
				obj := objWithGatedFields.DeepCopy()
				// Spec changes -> generation gets bumped.
				obj.Generation++
				return obj
			}(),
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
