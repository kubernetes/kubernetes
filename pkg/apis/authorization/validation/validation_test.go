/*
Copyright 2014 The Kubernetes Authors.

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

package validation

import (
	"strings"
	"testing"

	authorizationapi "k8s.io/kubernetes/pkg/apis/authorization"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

func TestValidateSARSpec(t *testing.T) {
	successCases := []authorizationapi.SubjectAccessReviewSpec{
		{ResourceAttributes: &authorizationapi.ResourceAttributes{}, User: "me"},
		{NonResourceAttributes: &authorizationapi.NonResourceAttributes{}, Groups: []string{"my-group"}},
	}
	for _, successCase := range successCases {
		if errs := ValidateSubjectAccessReviewSpec(successCase, field.NewPath("spec")); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := []struct {
		name string
		obj  authorizationapi.SubjectAccessReviewSpec
		msg  string
	}{
		{
			name: "neither request",
			obj:  authorizationapi.SubjectAccessReviewSpec{User: "me"},
			msg:  "exactly one of nonResourceAttributes or resourceAttributes must be specified",
		},
		{
			name: "both requests",
			obj: authorizationapi.SubjectAccessReviewSpec{
				ResourceAttributes:    &authorizationapi.ResourceAttributes{},
				NonResourceAttributes: &authorizationapi.NonResourceAttributes{},
				User: "me",
			},
			msg: "cannot be specified in combination with resourceAttributes",
		},
		{
			name: "no subject",
			obj: authorizationapi.SubjectAccessReviewSpec{
				ResourceAttributes: &authorizationapi.ResourceAttributes{},
			},
			msg: `spec.user: Invalid value: "": at least one of user or group must be specified`,
		},
	}

	for _, c := range errorCases {
		errs := ValidateSubjectAccessReviewSpec(c.obj, field.NewPath("spec"))
		if len(errs) == 0 {
			t.Errorf("%s: expected failure for %q", c.name, c.msg)
		} else if !strings.Contains(errs[0].Error(), c.msg) {
			t.Errorf("%s: unexpected error: %q, expected: %q", c.name, errs[0], c.msg)
		}

		errs = ValidateSubjectAccessReview(&authorizationapi.SubjectAccessReview{Spec: c.obj})
		if len(errs) == 0 {
			t.Errorf("%s: expected failure for %q", c.name, c.msg)
		} else if !strings.Contains(errs[0].Error(), c.msg) {
			t.Errorf("%s: unexpected error: %q, expected: %q", c.name, errs[0], c.msg)
		}
		errs = ValidateLocalSubjectAccessReview(&authorizationapi.LocalSubjectAccessReview{Spec: c.obj})
		if len(errs) == 0 {
			t.Errorf("%s: expected failure for %q", c.name, c.msg)
		} else if !strings.Contains(errs[0].Error(), c.msg) {
			t.Errorf("%s: unexpected error: %q, expected: %q", c.name, errs[0], c.msg)
		}

	}
}

func TestValidateSelfSAR(t *testing.T) {
	successCases := []authorizationapi.SelfSubjectAccessReviewSpec{
		{ResourceAttributes: &authorizationapi.ResourceAttributes{}},
	}
	for _, successCase := range successCases {
		if errs := ValidateSelfSubjectAccessReviewSpec(successCase, field.NewPath("spec")); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := []struct {
		name string
		obj  authorizationapi.SelfSubjectAccessReviewSpec
		msg  string
	}{
		{
			name: "neither request",
			obj:  authorizationapi.SelfSubjectAccessReviewSpec{},
			msg:  "exactly one of nonResourceAttributes or resourceAttributes must be specified",
		},
		{
			name: "both requests",
			obj: authorizationapi.SelfSubjectAccessReviewSpec{
				ResourceAttributes:    &authorizationapi.ResourceAttributes{},
				NonResourceAttributes: &authorizationapi.NonResourceAttributes{},
			},
			msg: "cannot be specified in combination with resourceAttributes",
		},
	}

	for _, c := range errorCases {
		errs := ValidateSelfSubjectAccessReviewSpec(c.obj, field.NewPath("spec"))
		if len(errs) == 0 {
			t.Errorf("%s: expected failure for %q", c.name, c.msg)
		} else if !strings.Contains(errs[0].Error(), c.msg) {
			t.Errorf("%s: unexpected error: %q, expected: %q", c.name, errs[0], c.msg)
		}

		errs = ValidateSelfSubjectAccessReview(&authorizationapi.SelfSubjectAccessReview{Spec: c.obj})
		if len(errs) == 0 {
			t.Errorf("%s: expected failure for %q", c.name, c.msg)
		} else if !strings.Contains(errs[0].Error(), c.msg) {
			t.Errorf("%s: unexpected error: %q, expected: %q", c.name, errs[0], c.msg)
		}
	}
}
