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

// Package subjectaccessreview exercises declarative validation for the
// authorization.k8s.io SubjectAccessReview resource.
package subjectaccessreview

import (
	"context"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/authorization"
	authorizationvalidation "k8s.io/kubernetes/pkg/apis/authorization/validation"
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
		&genericapirequest.RequestInfo{APIGroup: "authorization.k8s.io", APIVersion: apiVersion, Resource: "subjectaccessreviews"})

	testCases := map[string]struct {
		obj          authorization.SubjectAccessReview
		expectedErrs field.ErrorList
	}{
		"valid": {
			obj: mkSAR(),
		},
		"neither": {
			obj:          mkSAR(clearResourceAttributes()),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec"), "", "").WithOrigin("union").MarkAlpha()},
		},
		"both": {
			obj: mkSAR(setNonResourceAttributes()),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec"), "", "").WithOrigin("union").MarkAlpha(),
			},
		},
	}

	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalenceFunc(t, ctx, &tc.obj, func(ctx context.Context, obj runtime.Object) field.ErrorList {
				sar := obj.(*authorization.SubjectAccessReview)
				return authorizationvalidation.ValidateSubjectAccessReviewCreate(ctx, legacyscheme.Scheme, sar)
			}, tc.expectedErrs)
		})
	}
}

func mkSAR(tweaks ...func(*authorization.SubjectAccessReview)) authorization.SubjectAccessReview {
	sar := authorization.SubjectAccessReview{
		ObjectMeta: metav1.ObjectMeta{},
		Spec: authorization.SubjectAccessReviewSpec{
			ResourceAttributes: &authorization.ResourceAttributes{
				Namespace: "default",
				Verb:      "get",
				Resource:  "pods",
			},
			User: "admin",
		},
	}
	for _, tweak := range tweaks {
		tweak(&sar)
	}
	return sar
}

func clearResourceAttributes() func(*authorization.SubjectAccessReview) {
	return func(sar *authorization.SubjectAccessReview) {
		sar.Spec.ResourceAttributes = nil
	}
}

func setNonResourceAttributes() func(*authorization.SubjectAccessReview) {
	return func(sar *authorization.SubjectAccessReview) {
		sar.Spec.NonResourceAttributes = &authorization.NonResourceAttributes{}
	}
}
