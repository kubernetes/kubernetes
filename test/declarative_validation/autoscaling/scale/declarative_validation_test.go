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

// Package scale exercises declarative validation for the autoscaling/v1.Scale
// subresource. Unlike other directories under test/declarative_validation, the
// Scale subresource has no RESTUpdateStrategy: each parent's ScaleREST.Update
// composes ValidateScale + rest.DeclarativeValidation inline. The test mirrors
// that composition through VerifyUpdateValidationEquivalenceFunc.
package scale

import (
	"context"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	autoscalingvalidation "k8s.io/kubernetes/pkg/apis/autoscaling/validation"
	rcstorage "k8s.io/kubernetes/pkg/registry/core/replicationcontroller/storage"
)

func TestDeclarativeValidateUpdate(t *testing.T) {
	testCases := map[string]struct {
		old          autoscaling.Scale
		update       autoscaling.Scale
		expectedErrs field.ErrorList
	}{
		"replicas: 0":        {old: mkScale(), update: mkScale(setReplicas(0))},
		"replicas: positive": {old: mkScale(), update: mkScale(setReplicas(100))},
		"replicas: negative": {
			old:    mkScale(),
			update: mkScale(setReplicas(-1)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec.replicas"), nil, "").WithOrigin("minimum").MarkAlpha(),
			},
		},
	}

	for _, v := range apiVersions {
		t.Run(v, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(),
				&genericapirequest.RequestInfo{APIGroup: "autoscaling", APIVersion: v, Subresource: "scale"})
			for k, tc := range testCases {
				t.Run(k, func(t *testing.T) {
					tc.old.ResourceVersion = "1"
					tc.update.ResourceVersion = "1"
					apitesting.VerifyUpdateValidationEquivalenceFunc(t, ctx, &tc.update, &tc.old, validateScale, tc.expectedErrs, apitesting.WithSubResources("scale"), apitesting.WithSkipGroupVersions("extensions/v1beta1"))
				})
			}
		})
	}
}

// validateScale invokes the same ValidateScaleUpdate helper that every
// ScaleREST.UpdatedObject calls in production (e.g.
// pkg/registry/core/replicationcontroller/storage/storage.go), passing the
// real type-specific GroupVersionKindProvider used by the
// replicationcontroller ScaleREST so the test's GVK mapping matches
// production.
func validateScale(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	scale := obj.(*autoscaling.Scale)
	var oldScale *autoscaling.Scale
	if old != nil {
		oldScale = old.(*autoscaling.Scale)
	}
	return autoscalingvalidation.ValidateScaleUpdate(ctx, scale, oldScale, legacyscheme.Scheme, &rcstorage.ScaleREST{})
}

func mkScale(tweaks ...func(*autoscaling.Scale)) autoscaling.Scale {
	s := autoscaling.Scale{
		ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
		Spec:       autoscaling.ScaleSpec{Replicas: 1},
	}
	for _, tweak := range tweaks {
		tweak(&s)
	}
	return s
}

func setReplicas(val int32) func(*autoscaling.Scale) {
	return func(s *autoscaling.Scale) { s.Spec.Replicas = val }
}
