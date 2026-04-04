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

package pod

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
)

var bindingApiVersions = []string{"v1"}

func mkBinding(tweaks ...func(*api.Binding)) api.Binding {
	b := api.Binding{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "mypod",
			Namespace: "default",
		},
		Target: api.ObjectReference{
			Kind: "Node",
			Name: "mynode",
		},
	}
	for _, f := range tweaks {
		f(&b)
	}
	return b
}

func TestBinding_DeclarativeValidate_Create(t *testing.T) {
	for _, apiVersion := range bindingApiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(
				genericapirequest.NewDefaultContext(),
				&genericapirequest.RequestInfo{
					APIGroup:   "",
					APIVersion: apiVersion,
				},
			)

			// These test cases exercise declarative validation parity.
			// All 4 sub-modes (Beta enabled/disabled, hand-written, All Rules Enforced)
			// must agree on the expected errors.
			tests := map[string]struct {
				obj          api.Binding
				expectedErrs field.ErrorList
			}{
				"valid binding": {
					obj:          mkBinding(),
					expectedErrs: field.ErrorList{},
				},
				"valid binding with empty kind": {
					obj: mkBinding(func(b *api.Binding) {
						b.Target.Kind = ""
					}),
					expectedErrs: field.ErrorList{},
				},
				"missing target name": {
					obj: mkBinding(func(b *api.Binding) {
						b.Target.Name = ""
					}),
					expectedErrs: field.ErrorList{
						field.Required(field.NewPath("target", "name"), "").MarkAlpha(),
					},
				},
			}

			for name, tc := range tests {
				t.Run(name, func(t *testing.T) {
					apitesting.VerifyValidationEquivalence(t, ctx, &tc.obj, BindingStrategy.Validate, tc.expectedErrs)
				})
			}

			// The kind constraint is not yet covered by declarative validation (no suitable tag exists
			// for constraining plain string fields to a set of allowed values). It is validated by
			// hand-written code in ValidatePodBinding with MarkAlpha(). Test it separately.
			t.Run("invalid target kind", func(t *testing.T) {
				obj := mkBinding(func(b *api.Binding) {
					b.Target.Kind = "Pod"
				})
				errs := BindingStrategy.Validate(ctx, &obj)
				expectedErrs := field.ErrorList{
					field.NotSupported(field.NewPath("target", "kind"), "Pod", []string{"Node", "<empty>"}).MarkAlpha(),
				}
				matcher := field.ErrorMatcher{}.ByType().ByOrigin().ByField()
				matcher.Test(t, expectedErrs, errs)
			})
		})
	}
}
