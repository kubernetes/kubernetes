/*
Copyright 2025 The Kubernetes Authors.

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

package devicetaintrule

import (
	"testing"

	"github.com/stretchr/testify/assert"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/resource"
)

var obj = &resource.DeviceTaintRule{
	ObjectMeta: metav1.ObjectMeta{
		Name:       "valid-patch",
		Generation: 1,
	},
	Spec: resource.DeviceTaintRuleSpec{
		Taint: resource.DeviceTaint{
			Key:    "example.com/tainted",
			Effect: resource.DeviceTaintEffectNoExecute,
		},
	},
}

var objWithStatus = &resource.DeviceTaintRule{
	ObjectMeta: metav1.ObjectMeta{
		Name:       "valid-patch",
		Generation: 1,
	},
	Spec: resource.DeviceTaintRuleSpec{
		Taint: resource.DeviceTaint{
			Key:    "example.com/tainted",
			Effect: resource.DeviceTaintEffectNoExecute,
		},
	},
	Status: resource.DeviceTaintRuleStatus{
		Conditions: []metav1.Condition{{
			Type:               "foo",
			Status:             metav1.ConditionFalse,
			LastTransitionTime: metav1.Now(),
			Reason:             "something",
			Message:            "else",
		}},
	},
}

var fieldImmutableError = "field is immutable"
var metadataError = "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters"

func TestDeviceTaintRuleStrategy(t *testing.T) {
	if Strategy.NamespaceScoped() {
		t.Errorf("DeviceTaintRule must not be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("DeviceTaintRule should not allow create on update")
	}
}

func TestDeviceTaintRuleStrategyCreate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	testcases := map[string]struct {
		obj                   *resource.DeviceTaintRule
		expectValidationError string
		expectObj             *resource.DeviceTaintRule
	}{
		"simple": {
			obj:       obj,
			expectObj: obj,
		},
		"validation-error": {
			obj: func() *resource.DeviceTaintRule {
				obj := obj.DeepCopy()
				obj.Name = "%#@$%$"
				return obj
			}(),
			expectValidationError: metadataError,
		},
		"drop-status": {
			obj:       objWithStatus,
			expectObj: obj,
		},
		"set-generation": {
			obj: func() *resource.DeviceTaintRule {
				obj := obj.DeepCopy()
				obj.Generation = 42 // Cannot be set by client on create, overwritten with 1.
				return obj
			}(),
			expectObj: obj,
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			obj := tc.obj.DeepCopy()
			Strategy.PrepareForCreate(ctx, obj)
			if errs := Strategy.Validate(ctx, obj); len(errs) != 0 {
				if tc.expectValidationError == "" {
					t.Fatalf("unexpected error(s): %v", errs)
				}
				assert.ErrorContains(t, errs[0], tc.expectValidationError, "the error message should have contained the expected error message")
				return
			}
			if tc.expectValidationError != "" {
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

func TestDeviceTaintRuleStrategyUpdate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()

	testcases := map[string]struct {
		oldObj                *resource.DeviceTaintRule
		newObj                *resource.DeviceTaintRule
		expectValidationError string
		expectObj             *resource.DeviceTaintRule
	}{
		"no-changes-okay": {
			oldObj:    obj,
			newObj:    obj,
			expectObj: obj,
		},
		"name-change-not-allowed": {
			oldObj: obj,
			newObj: func() *resource.DeviceTaintRule {
				obj := obj.DeepCopy()
				obj.Name += "-2"
				return obj
			}(),
			expectValidationError: fieldImmutableError,
		},
		"drop-status": {
			oldObj:    obj,
			newObj:    objWithStatus,
			expectObj: obj,
		},
		"bump-generation": {
			oldObj: obj,
			newObj: func() *resource.DeviceTaintRule {
				obj := obj.DeepCopy()
				obj.Spec.Taint.Effect = resource.DeviceTaintEffectNone
				return obj
			}(),
			expectObj: func() *resource.DeviceTaintRule {
				obj := obj.DeepCopy()
				obj.Spec.Taint.Effect = resource.DeviceTaintEffectNone
				obj.Generation++
				return obj
			}(),
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			oldObj := tc.oldObj.DeepCopy()
			newObj := tc.newObj.DeepCopy()
			newObj.ResourceVersion = "4"

			Strategy.PrepareForUpdate(ctx, newObj, oldObj)
			if errs := Strategy.ValidateUpdate(ctx, newObj, oldObj); len(errs) != 0 {
				if tc.expectValidationError == "" {
					t.Fatalf("unexpected error(s): %v", errs)
				}
				assert.ErrorContains(t, errs[0], tc.expectValidationError, "the error message should have contained the expected error message")
				return
			}
			if tc.expectValidationError != "" {
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
		oldObj                *resource.DeviceTaintRule
		newObj                *resource.DeviceTaintRule
		expectValidationError string
		expectObj             *resource.DeviceTaintRule
	}{
		"no-changes-okay": {
			oldObj:    obj,
			newObj:    obj,
			expectObj: obj,
		},
		"name-change-not-allowed": {
			oldObj: obj,
			newObj: func() *resource.DeviceTaintRule {
				obj := obj.DeepCopy()
				obj.Name += "-2"
				return obj
			}(),
			expectValidationError: fieldImmutableError,
		},
		// Cannot add finalizers, annotations and labels during status update.
		"drop-meta-changes": {
			oldObj: obj,
			newObj: func() *resource.DeviceTaintRule {
				obj := obj.DeepCopy()
				obj.Finalizers = []string{"foo"}
				obj.Annotations = map[string]string{"foo": "bar"}
				obj.Labels = map[string]string{"foo": "bar"}
				return obj
			}(),
			expectObj: obj,
		},
		"drop-spec": {
			oldObj: obj,
			newObj: func() *resource.DeviceTaintRule {
				obj := obj.DeepCopy()
				obj.Spec.Taint.Effect = resource.DeviceTaintEffectNone
				return obj
			}(),
			expectObj: obj,
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			oldObj := tc.oldObj.DeepCopy()
			newObj := tc.newObj.DeepCopy()
			newObj.ResourceVersion = "4"

			StatusStrategy.PrepareForUpdate(ctx, newObj, oldObj)
			if errs := StatusStrategy.ValidateUpdate(ctx, newObj, oldObj); len(errs) != 0 {
				if tc.expectValidationError == "" {
					t.Fatalf("unexpected error(s): %v", errs)
				}
				assert.ErrorContains(t, errs[0], tc.expectValidationError, "the error message should have contained the expected error message")
				return
			}
			if tc.expectValidationError != "" {
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
