/*
Copyright The Kubernetes Authors.

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

// Package binding exercises declarative validation for the core/v1 Binding
// resource. Binding's custom BindingREST.Create bypasses genericregistry.Store
// (and therefore any RESTCreateStrategy), so declarative validation is invoked
// directly through corevalidation.ValidatePodBindingCreate rather than a strategy.
package binding

import (
	"context"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/core"
	corevalidation "k8s.io/kubernetes/pkg/apis/core/validation"
)

func TestDeclarativeValidate(t *testing.T) {
	for _, v := range apiVersions {
		t.Run("version="+v, func(t *testing.T) {
			testDeclarativeValidate(t, v)
		})
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(),
		&genericapirequest.RequestInfo{APIGroup: "", APIVersion: apiVersion, Resource: "pods", Subresource: "binding"})

	testCases := map[string]struct {
		obj          core.Binding
		expectedErrs field.ErrorList
	}{
		"valid": {
			obj: mkBinding(),
		},
		"valid: empty kind defaults to Node": {
			obj: mkBinding(func(b *core.Binding) { b.Target.Kind = "" }),
		},
		"missing target name": {
			obj: mkBinding(func(b *core.Binding) { b.Target.Name = "" }),
			expectedErrs: field.ErrorList{
				// Under normal enforcement this is the handwritten error (CoveredByDeclarative,
				// since the +k8s:alpha DV tag is never enforced outside of tests). Under the
				// test-only "All Rules Enforced" mode it is replaced by the DV-native error,
				// which carries the tag's own Alpha stability and a declarative source.
				field.Required(field.NewPath("target", "name"), "").MarkCoveredByDeclarative().MarkAlpha(),
			},
		},
		"unsupported target kind": {
			obj: mkBinding(func(b *core.Binding) { b.Target.Kind = "Pod" }),
			expectedErrs: field.ErrorList{
				// Kind has no declarative counterpart, so this stays a plain handwritten
				// (imperative) error in every enforcement mode.
				field.NotSupported(field.NewPath("target", "kind"), "Pod", []string{"Node", "<empty>"}).MarkAlpha().MarkFromImperative(),
			},
		},
	}

	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalenceFunc(t, ctx, &tc.obj, func(ctx context.Context, obj runtime.Object) field.ErrorList {
				binding := obj.(*core.Binding)
				return corevalidation.ValidatePodBindingCreate(ctx, legacyscheme.Scheme, binding)
			}, tc.expectedErrs)
		})
	}
}

func mkBinding(tweaks ...func(*core.Binding)) core.Binding {
	b := core.Binding{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "mypod",
			Namespace: "default",
		},
		Target: core.ObjectReference{
			Kind: "Node",
			Name: "mynode",
		},
	}
	for _, tweak := range tweaks {
		tweak(&b)
	}
	return b
}
