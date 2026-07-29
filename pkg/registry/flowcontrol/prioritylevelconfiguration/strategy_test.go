/*
Copyright 2023 The Kubernetes Authors.

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

package prioritylevelconfiguration

import (
	"context"
	"testing"

	flowcontrolv1 "k8s.io/api/flowcontrol/v1"
	flowcontrolv1beta3 "k8s.io/api/flowcontrol/v1beta3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/flowcontrol"
	"k8s.io/utils/ptr"

	"github.com/google/go-cmp/cmp"
)

func TestPriorityLevelConfigurationValidation(t *testing.T) {
	v1ObjFn := func(v *int32) *flowcontrolv1.PriorityLevelConfiguration {
		return &flowcontrolv1.PriorityLevelConfiguration{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo",
			},
			Spec: flowcontrolv1.PriorityLevelConfigurationSpec{
				Type: flowcontrolv1.PriorityLevelEnablementLimited,
				Limited: &flowcontrolv1.LimitedPriorityLevelConfiguration{
					NominalConcurrencyShares: v,
					LimitResponse: flowcontrolv1.LimitResponse{
						Type: flowcontrolv1.LimitResponseTypeReject},
				},
			},
		}
	}
	v1beta3ObjFn := func(v int32, isZero bool) *flowcontrolv1beta3.PriorityLevelConfiguration {
		obj := &flowcontrolv1beta3.PriorityLevelConfiguration{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo",
			},
			Spec: flowcontrolv1beta3.PriorityLevelConfigurationSpec{
				Type: flowcontrolv1beta3.PriorityLevelEnablementLimited,
				Limited: &flowcontrolv1beta3.LimitedPriorityLevelConfiguration{
					NominalConcurrencyShares: v,
					LimitResponse: flowcontrolv1beta3.LimitResponse{
						Type: flowcontrolv1beta3.LimitResponseTypeReject},
				},
			},
		}
		if isZero && v == 0 {
			obj.ObjectMeta.Annotations = map[string]string{}
			obj.ObjectMeta.Annotations[flowcontrolv1beta3.PriorityLevelPreserveZeroConcurrencySharesKey] = ""
		}
		return obj
	}
	internalObjFn := func(v int32) *flowcontrol.PriorityLevelConfiguration {
		return &flowcontrol.PriorityLevelConfiguration{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo",
			},
			Spec: flowcontrol.PriorityLevelConfigurationSpec{
				Type: flowcontrol.PriorityLevelEnablementLimited,
				Limited: &flowcontrol.LimitedPriorityLevelConfiguration{
					NominalConcurrencyShares: v,
					LimitResponse: flowcontrol.LimitResponse{
						Type: flowcontrol.LimitResponseTypeReject},
				},
			},
		}
	}
	v1SchemeFn := func(t *testing.T) *runtime.Scheme {
		scheme := runtime.NewScheme()
		if err := flowcontrolv1.AddToScheme(scheme); err != nil {
			t.Fatalf("Failed to add to scheme: %v", err)
		}
		return scheme
	}
	v1beta3SchemeFn := func(t *testing.T) *runtime.Scheme {
		scheme := runtime.NewScheme()
		if err := flowcontrolv1beta3.AddToScheme(scheme); err != nil {
			t.Fatalf("Failed to add to scheme: %v", err)
		}
		return scheme
	}

	tests := []struct {
		name        string
		obj         runtime.Object
		old         *flowcontrol.PriorityLevelConfiguration // for UPDATE only
		scheme      *runtime.Scheme
		errExpected field.ErrorList
	}{
		{
			name:        "v1, create, zero value, no error expected",
			obj:         v1ObjFn(ptr.To(int32(0))),
			scheme:      v1SchemeFn(t),
			errExpected: nil,
		},
		{
			name:        "v1, create, unset, no error expected",
			obj:         v1ObjFn(nil),
			scheme:      v1SchemeFn(t),
			errExpected: nil,
		},
		{
			name:        "v1, create, non-zero, no error expected",
			obj:         v1ObjFn(ptr.To(int32(1))),
			scheme:      v1SchemeFn(t),
			errExpected: nil,
		},
		{
			name:        "v1beta3, create, zero value, no error expected",
			obj:         v1beta3ObjFn(0, true),
			scheme:      v1beta3SchemeFn(t),
			errExpected: nil,
		},
		{
			name:        "v1beta3, create, zero value without annotation, no error expected",
			obj:         v1beta3ObjFn(0, false),
			scheme:      v1beta3SchemeFn(t),
			errExpected: nil,
		},
		{
			name:        "v1beta3, create, non-zero, no error expected",
			obj:         v1beta3ObjFn(1, false),
			scheme:      v1beta3SchemeFn(t),
			errExpected: nil,
		},

		// the following use cases cover UPDATE
		{
			name:        "v1, update, zero value, existing has non-zero, no error expected",
			obj:         v1ObjFn(ptr.To(int32(0))),
			old:         internalObjFn(1),
			scheme:      v1SchemeFn(t),
			errExpected: nil,
		},
		{
			name:        "v1, update, zero value, existing has zero, no error expected",
			obj:         v1ObjFn(ptr.To(int32(0))),
			old:         internalObjFn(0),
			scheme:      v1SchemeFn(t),
			errExpected: nil,
		},
		{
			name:        "v1, update, non-zero value, existing has zero, no error expected",
			obj:         v1ObjFn(ptr.To(int32(1))),
			old:         internalObjFn(0),
			scheme:      v1SchemeFn(t),
			errExpected: nil,
		},
		{
			name:        "v1beta3, update, zero value, existing has non-zero, no error expected",
			obj:         v1beta3ObjFn(0, true),
			old:         internalObjFn(1),
			scheme:      v1beta3SchemeFn(t),
			errExpected: nil,
		},
		{
			name:        "v1beta3, update, zero value, existing has zero, no error expected",
			obj:         v1beta3ObjFn(0, true),
			old:         internalObjFn(0),
			scheme:      v1beta3SchemeFn(t),
			errExpected: nil,
		},
		{
			name:        "v1beta3, update, non-zero value, existing has zero, no error expected",
			obj:         v1beta3ObjFn(1, false),
			old:         internalObjFn(0),
			scheme:      v1beta3SchemeFn(t),
			errExpected: nil,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			scheme := test.scheme
			scheme.Default(test.obj)

			ctx := context.TODO()
			internal := &flowcontrol.PriorityLevelConfiguration{}
			if err := scheme.Convert(test.obj, internal, ctx); err != nil {
				t.Errorf("Expected no error while converting to internal type: %v", err)
			}

			err := func(obj, old *flowcontrol.PriorityLevelConfiguration) field.ErrorList {
				if old == nil {
					return Strategy.Validate(ctx, obj) // for create operation
				}
				return Strategy.ValidateUpdate(ctx, obj, old) // for update operation
			}(internal, test.old)

			if !cmp.Equal(test.errExpected, err) {
				t.Errorf("Expected error: %v, diff: %s", test.errExpected, cmp.Diff(test.errExpected, err))
			}
		})
	}
}
