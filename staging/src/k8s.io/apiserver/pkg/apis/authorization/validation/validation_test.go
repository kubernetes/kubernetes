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

	"github.com/google/go-cmp/cmp"

	authorizationv1 "k8s.io/api/authorization/v1"
	authorizationv1alpha1 "k8s.io/api/authorization/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

func TestValidateSARSpec(t *testing.T) {
	successCases := []authorizationv1.SubjectAccessReviewSpec{
		{ResourceAttributes: &authorizationv1.ResourceAttributes{}, User: "me"},
		{NonResourceAttributes: &authorizationv1.NonResourceAttributes{}, Groups: []string{"my-group"}},
		{ // field raw selector
			User: "me",
			ResourceAttributes: &authorizationv1.ResourceAttributes{
				FieldSelector: &authorizationv1.FieldSelectorAttributes{
					RawSelector: "***foo",
				},
			},
		},
		{ // label raw selector
			User: "me",
			ResourceAttributes: &authorizationv1.ResourceAttributes{
				LabelSelector: &authorizationv1.LabelSelectorAttributes{
					RawSelector: "***foo",
				},
			},
		},
		{ // unknown field operator
			User: "me",
			ResourceAttributes: &authorizationv1.ResourceAttributes{
				FieldSelector: &authorizationv1.FieldSelectorAttributes{
					Requirements: []metav1.FieldSelectorRequirement{
						{
							Key:      "k",
							Operator: metav1.FieldSelectorOperator("fake"),
							Values:   []string{"val"},
						},
					},
				},
			},
		},
		{ // unknown label operator
			User: "me",
			ResourceAttributes: &authorizationv1.ResourceAttributes{
				LabelSelector: &authorizationv1.LabelSelectorAttributes{
					Requirements: []metav1.LabelSelectorRequirement{
						{
							Key:      "k",
							Operator: metav1.LabelSelectorOperator("fake"),
							Values:   []string{"val"},
						},
					},
				},
			},
		},
		{ // unconditional authorization ok
			User: "me",
			ResourceAttributes: &authorizationv1.ResourceAttributes{
				Verb:     "create",
				Resource: "pods",
			},
			AuthorizationOptions: &authorizationv1.AuthorizationOptions{
				HandledDecisionTypes: []authorizationv1.ConditionsAwareDecisionType{
					authorizationv1.ConditionsAwareDecisionTypeAllow,
					authorizationv1.ConditionsAwareDecisionTypeNoOpinion,
					authorizationv1.ConditionsAwareDecisionTypeDeny,
				},
			},
		},
		{ // conditional authorization ok
			User: "me",
			ResourceAttributes: &authorizationv1.ResourceAttributes{
				Verb:     "create",
				Resource: "pods",
			},
			AuthorizationOptions: &authorizationv1.AuthorizationOptions{
				HandledDecisionTypes: []authorizationv1.ConditionsAwareDecisionType{
					authorizationv1.ConditionsAwareDecisionTypeAllow,
					authorizationv1.ConditionsAwareDecisionTypeNoOpinion,
					authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
					authorizationv1.ConditionsAwareDecisionTypeUnion,
					authorizationv1.ConditionsAwareDecisionTypeDeny,
				},
			},
		},
	}
	for _, successCase := range successCases {
		if errs := ValidateSubjectAccessReviewSpec(successCase, field.NewPath("spec")); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := []struct {
		name string
		obj  authorizationv1.SubjectAccessReviewSpec
		msg  string
	}{{
		name: "neither request",
		obj:  authorizationv1.SubjectAccessReviewSpec{User: "me"},
		msg:  "exactly one of nonResourceAttributes or resourceAttributes must be specified",
	}, {
		name: "both requests",
		obj: authorizationv1.SubjectAccessReviewSpec{
			ResourceAttributes:    &authorizationv1.ResourceAttributes{},
			NonResourceAttributes: &authorizationv1.NonResourceAttributes{},
			User:                  "me",
		},
		msg: "exactly one of nonResourceAttributes or resourceAttributes must be specified",
	}, {
		name: "no subject",
		obj: authorizationv1.SubjectAccessReviewSpec{
			ResourceAttributes: &authorizationv1.ResourceAttributes{},
		},
		msg: `spec.user: Invalid value: "": at least one of user or group must be specified`,
	}, {
		name: "resource attributes: field selector specify both",
		obj: authorizationv1.SubjectAccessReviewSpec{
			User: "me",
			ResourceAttributes: &authorizationv1.ResourceAttributes{
				FieldSelector: &authorizationv1.FieldSelectorAttributes{
					RawSelector: "foo",
					Requirements: []metav1.FieldSelectorRequirement{
						{},
					},
				},
			},
		},
		msg: `spec.resourceAttributes.fieldSelector.rawSelector: Invalid value: "foo": may not specified at the same time as requirements`,
	}, {
		name: "resource attributes: field selector specify neither",
		obj: authorizationv1.SubjectAccessReviewSpec{
			User: "me",
			ResourceAttributes: &authorizationv1.ResourceAttributes{
				FieldSelector: &authorizationv1.FieldSelectorAttributes{},
			},
		},
		msg: `spec.resourceAttributes.fieldSelector.requirements: Required value: when spec.resourceAttributes.fieldSelector is specified, requirements or rawSelector is required`,
	}, {
		name: "resource attributes: field selector no key",
		obj: authorizationv1.SubjectAccessReviewSpec{
			User: "me",
			ResourceAttributes: &authorizationv1.ResourceAttributes{
				FieldSelector: &authorizationv1.FieldSelectorAttributes{
					Requirements: []metav1.FieldSelectorRequirement{
						{
							Key: "",
						},
					},
				},
			},
		},
		msg: `spec.resourceAttributes.fieldSelector.requirements[0].key: Required value: must be specified`,
	}, {
		name: "resource attributes: field selector no value for in",
		obj: authorizationv1.SubjectAccessReviewSpec{
			User: "me",
			ResourceAttributes: &authorizationv1.ResourceAttributes{
				FieldSelector: &authorizationv1.FieldSelectorAttributes{
					Requirements: []metav1.FieldSelectorRequirement{
						{
							Key:      "k",
							Operator: metav1.FieldSelectorOpIn,
							Values:   []string{},
						},
					},
				},
			},
		},
		msg: "spec.resourceAttributes.fieldSelector.requirements[0].values: Required value: must be specified when `operator` is 'In' or 'NotIn'",
	}, {
		name: "resource attributes: field selector no value for not in",
		obj: authorizationv1.SubjectAccessReviewSpec{
			User: "me",
			ResourceAttributes: &authorizationv1.ResourceAttributes{
				FieldSelector: &authorizationv1.FieldSelectorAttributes{
					Requirements: []metav1.FieldSelectorRequirement{
						{
							Key:      "k",
							Operator: metav1.FieldSelectorOpNotIn,
							Values:   []string{},
						},
					},
				},
			},
		},
		msg: "spec.resourceAttributes.fieldSelector.requirements[0].values: Required value: must be specified when `operator` is 'In' or 'NotIn'",
	}, {
		name: "resource attributes: field selector values for exists",
		obj: authorizationv1.SubjectAccessReviewSpec{
			User: "me",
			ResourceAttributes: &authorizationv1.ResourceAttributes{
				FieldSelector: &authorizationv1.FieldSelectorAttributes{
					Requirements: []metav1.FieldSelectorRequirement{
						{
							Key:      "k",
							Operator: metav1.FieldSelectorOpExists,
							Values:   []string{"val"},
						},
					},
				},
			},
		},
		msg: "spec.resourceAttributes.fieldSelector.requirements[0].values: Forbidden: may not be specified when `operator` is 'Exists' or 'DoesNotExist'",
	}, {
		name: "resource attributes: field selector values for not exists",
		obj: authorizationv1.SubjectAccessReviewSpec{
			User: "me",
			ResourceAttributes: &authorizationv1.ResourceAttributes{
				FieldSelector: &authorizationv1.FieldSelectorAttributes{
					Requirements: []metav1.FieldSelectorRequirement{
						{
							Key:      "k",
							Operator: metav1.FieldSelectorOpDoesNotExist,
							Values:   []string{"val"},
						},
					},
				},
			},
		},
		msg: "spec.resourceAttributes.fieldSelector.requirements[0].values: Forbidden: may not be specified when `operator` is 'Exists' or 'DoesNotExist'",
	}, {
		name: "resource attributes: label selector specify both",
		obj: authorizationv1.SubjectAccessReviewSpec{
			User: "me",
			ResourceAttributes: &authorizationv1.ResourceAttributes{
				LabelSelector: &authorizationv1.LabelSelectorAttributes{
					RawSelector: "foo",
					Requirements: []metav1.LabelSelectorRequirement{
						{},
					},
				},
			},
		},
		msg: `spec.resourceAttributes.labelSelector.rawSelector: Invalid value: "foo": may not specified at the same time as requirements`,
	}, {
		name: "resource attributes: label selector specify neither",
		obj: authorizationv1.SubjectAccessReviewSpec{
			User: "me",
			ResourceAttributes: &authorizationv1.ResourceAttributes{
				LabelSelector: &authorizationv1.LabelSelectorAttributes{},
			},
		},
		msg: `spec.resourceAttributes.labelSelector.requirements: Required value: when spec.resourceAttributes.labelSelector is specified, requirements or rawSelector is required`,
	}, {
		name: "resource attributes: label selector no key",
		obj: authorizationv1.SubjectAccessReviewSpec{
			User: "me",
			ResourceAttributes: &authorizationv1.ResourceAttributes{
				LabelSelector: &authorizationv1.LabelSelectorAttributes{
					Requirements: []metav1.LabelSelectorRequirement{
						{
							Key: "",
						},
					},
				},
			},
		},
		msg: `spec.resourceAttributes.labelSelector.requirements[0].key: Invalid value: "": name part must be non-empty`,
	}, {
		name: "resource attributes: label selector invalid label name",
		obj: authorizationv1.SubjectAccessReviewSpec{
			User: "me",
			ResourceAttributes: &authorizationv1.ResourceAttributes{
				LabelSelector: &authorizationv1.LabelSelectorAttributes{
					Requirements: []metav1.LabelSelectorRequirement{
						{
							Key: "()foo",
						},
					},
				},
			},
		},
		msg: `spec.resourceAttributes.labelSelector.requirements[0].key: Invalid value: "()foo": name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')`,
	}, {
		name: "resource attributes: label selector no value for in",
		obj: authorizationv1.SubjectAccessReviewSpec{
			User: "me",
			ResourceAttributes: &authorizationv1.ResourceAttributes{
				LabelSelector: &authorizationv1.LabelSelectorAttributes{
					Requirements: []metav1.LabelSelectorRequirement{
						{
							Key:      "k",
							Operator: metav1.LabelSelectorOpIn,
							Values:   []string{},
						},
					},
				},
			},
		},
		msg: "spec.resourceAttributes.labelSelector.requirements[0].values: Required value: must be specified when `operator` is 'In' or 'NotIn'",
	}, {
		name: "resource attributes: label selector no value for not in",
		obj: authorizationv1.SubjectAccessReviewSpec{
			User: "me",
			ResourceAttributes: &authorizationv1.ResourceAttributes{
				LabelSelector: &authorizationv1.LabelSelectorAttributes{
					Requirements: []metav1.LabelSelectorRequirement{
						{
							Key:      "k",
							Operator: metav1.LabelSelectorOpNotIn,
							Values:   []string{},
						},
					},
				},
			},
		},
		msg: "spec.resourceAttributes.labelSelector.requirements[0].values: Required value: must be specified when `operator` is 'In' or 'NotIn'",
	}, {
		name: "resource attributes: label selector values for exists",
		obj: authorizationv1.SubjectAccessReviewSpec{
			User: "me",
			ResourceAttributes: &authorizationv1.ResourceAttributes{
				LabelSelector: &authorizationv1.LabelSelectorAttributes{
					Requirements: []metav1.LabelSelectorRequirement{
						{
							Key:      "k",
							Operator: metav1.LabelSelectorOpExists,
							Values:   []string{"val"},
						},
					},
				},
			},
		},
		msg: "spec.resourceAttributes.labelSelector.requirements[0].values: Forbidden: may not be specified when `operator` is 'Exists' or 'DoesNotExist'",
	}, {
		name: "resource attributes: label selector values for not exists",
		obj: authorizationv1.SubjectAccessReviewSpec{
			User: "me",
			ResourceAttributes: &authorizationv1.ResourceAttributes{
				LabelSelector: &authorizationv1.LabelSelectorAttributes{
					Requirements: []metav1.LabelSelectorRequirement{
						{
							Key:      "k",
							Operator: metav1.LabelSelectorOpDoesNotExist,
							Values:   []string{"val"},
						},
					},
				},
			},
		},
		msg: "spec.resourceAttributes.labelSelector.requirements[0].values: Forbidden: may not be specified when `operator` is 'Exists' or 'DoesNotExist'",
	}, {
		name: "authorization options: at least {Allow, Deny, NoOpinion} must be specified",
		obj: authorizationv1.SubjectAccessReviewSpec{
			User: "me",
			ResourceAttributes: &authorizationv1.ResourceAttributes{
				Verb:     "create",
				Resource: "pods",
			},
			AuthorizationOptions: &authorizationv1.AuthorizationOptions{
				HandledDecisionTypes: []authorizationv1.ConditionsAwareDecisionType{
					authorizationv1.ConditionsAwareDecisionTypeAllow,
					authorizationv1.ConditionsAwareDecisionTypeDeny,
				},
			},
		},
		msg: `spec.authorizationOptions.handledDecisionTypes: Invalid value: ["Allow","Deny"]: set must at least contain {Allow, Deny, NoOpinion}`,
	}}

	for _, c := range errorCases {
		t.Run(c.name, func(t *testing.T) {
			errs := ValidateSubjectAccessReviewSpec(c.obj, field.NewPath("spec"))
			if len(errs) == 0 {
				t.Errorf("%s: expected failure for %q", c.name, c.msg)
			} else if !strings.Contains(errs[0].Error(), c.msg) {
				t.Errorf("%s: unexpected error: %q, expected: %q", c.name, errs[0], c.msg)
			}

			errs = ValidateSubjectAccessReview(&authorizationv1.SubjectAccessReview{Spec: c.obj})
			if len(errs) == 0 {
				t.Errorf("%s: expected failure for %q", c.name, c.msg)
			} else if !strings.Contains(errs[0].Error(), c.msg) {
				t.Errorf("%s: unexpected error: %q, expected: %q", c.name, errs[0], c.msg)
			}
			errs = ValidateLocalSubjectAccessReview(&authorizationv1.LocalSubjectAccessReview{Spec: c.obj})
			if len(errs) == 0 {
				t.Errorf("%s: expected failure for %q", c.name, c.msg)
			} else if !strings.Contains(errs[0].Error(), c.msg) {
				t.Errorf("%s: unexpected error: %q, expected: %q", c.name, errs[0], c.msg)
			}
		})
	}
}

func TestValidateSelfSAR(t *testing.T) {
	successCases := []authorizationv1.SelfSubjectAccessReviewSpec{
		{ResourceAttributes: &authorizationv1.ResourceAttributes{}},
		{ // unconditional authorization ok
			ResourceAttributes: &authorizationv1.ResourceAttributes{
				Verb:     "create",
				Resource: "pods",
			},
			AuthorizationOptions: &authorizationv1.AuthorizationOptions{
				HandledDecisionTypes: []authorizationv1.ConditionsAwareDecisionType{
					authorizationv1.ConditionsAwareDecisionTypeAllow,
					authorizationv1.ConditionsAwareDecisionTypeNoOpinion,
					authorizationv1.ConditionsAwareDecisionTypeDeny,
				},
			},
		},
		{ // conditional authorization ok
			ResourceAttributes: &authorizationv1.ResourceAttributes{
				Verb:     "create",
				Resource: "pods",
			},
			AuthorizationOptions: &authorizationv1.AuthorizationOptions{
				HandledDecisionTypes: []authorizationv1.ConditionsAwareDecisionType{
					authorizationv1.ConditionsAwareDecisionTypeAllow,
					authorizationv1.ConditionsAwareDecisionTypeNoOpinion,
					authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
					authorizationv1.ConditionsAwareDecisionTypeUnion,
					authorizationv1.ConditionsAwareDecisionTypeDeny,
				},
			},
		},
	}
	for _, successCase := range successCases {
		if errs := ValidateSelfSubjectAccessReviewSpec(successCase, field.NewPath("spec")); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := []struct {
		name string
		obj  authorizationv1.SelfSubjectAccessReviewSpec
		msg  string
	}{{
		name: "neither request",
		obj:  authorizationv1.SelfSubjectAccessReviewSpec{},
		msg:  "exactly one of nonResourceAttributes or resourceAttributes must be specified",
	}, {
		name: "both requests",
		obj: authorizationv1.SelfSubjectAccessReviewSpec{
			ResourceAttributes:    &authorizationv1.ResourceAttributes{},
			NonResourceAttributes: &authorizationv1.NonResourceAttributes{},
		},
		msg: "exactly one of nonResourceAttributes or resourceAttributes must be specified",
	}, {
		// here we only test one to be sure the function is called.  The more exhaustive suite is tested above.
		name: "resource attributes: label selector specify both",
		obj: authorizationv1.SelfSubjectAccessReviewSpec{
			ResourceAttributes: &authorizationv1.ResourceAttributes{
				LabelSelector: &authorizationv1.LabelSelectorAttributes{
					RawSelector: "foo",
					Requirements: []metav1.LabelSelectorRequirement{
						{},
					},
				},
			},
		},
		msg: `spec.resourceAttributes.labelSelector.rawSelector: Invalid value: "foo": may not specified at the same time as requirements`,
	}, {
		name: "authorization options: at least {Allow, Deny, NoOpinion} must be specified",
		obj: authorizationv1.SelfSubjectAccessReviewSpec{
			ResourceAttributes: &authorizationv1.ResourceAttributes{
				Verb:     "create",
				Resource: "pods",
			},
			AuthorizationOptions: &authorizationv1.AuthorizationOptions{
				HandledDecisionTypes: []authorizationv1.ConditionsAwareDecisionType{
					authorizationv1.ConditionsAwareDecisionTypeAllow,
					authorizationv1.ConditionsAwareDecisionTypeDeny,
				},
			},
		},
		msg: `spec.authorizationOptions.handledDecisionTypes: Invalid value: ["Allow","Deny"]: set must at least contain {Allow, Deny, NoOpinion}`,
	}}

	for _, c := range errorCases {
		errs := ValidateSelfSubjectAccessReviewSpec(c.obj, field.NewPath("spec"))
		if len(errs) == 0 {
			t.Errorf("%s: expected failure for %q", c.name, c.msg)
		} else if !strings.Contains(errs[0].Error(), c.msg) {
			t.Errorf("%s: unexpected error: %q, expected: %q", c.name, errs[0], c.msg)
		}

		errs = ValidateSelfSubjectAccessReview(&authorizationv1.SelfSubjectAccessReview{Spec: c.obj})
		if len(errs) == 0 {
			t.Errorf("%s: expected failure for %q", c.name, c.msg)
		} else if !strings.Contains(errs[0].Error(), c.msg) {
			t.Errorf("%s: unexpected error: %q, expected: %q", c.name, errs[0], c.msg)
		}
	}
}

func TestValidateLocalSAR(t *testing.T) {
	successCases := []authorizationv1.LocalSubjectAccessReview{{
		Spec: authorizationv1.SubjectAccessReviewSpec{
			ResourceAttributes: &authorizationv1.ResourceAttributes{},
			User:               "user",
		},
	}}
	for _, successCase := range successCases {
		if errs := ValidateLocalSubjectAccessReview(&successCase); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := []struct {
		name string
		obj  *authorizationv1.LocalSubjectAccessReview
		msg  string
	}{{
		name: "name",
		obj: &authorizationv1.LocalSubjectAccessReview{
			ObjectMeta: metav1.ObjectMeta{Name: "a"},
			Spec: authorizationv1.SubjectAccessReviewSpec{
				ResourceAttributes: &authorizationv1.ResourceAttributes{},
				User:               "user",
			},
		},
		msg: "must be empty except for namespace",
	}, {
		name: "namespace conflict",
		obj: &authorizationv1.LocalSubjectAccessReview{
			ObjectMeta: metav1.ObjectMeta{Namespace: "a"},
			Spec: authorizationv1.SubjectAccessReviewSpec{
				ResourceAttributes: &authorizationv1.ResourceAttributes{},
				User:               "user",
			},
		},
		msg: "must match metadata.namespace",
	}, {
		name: "nonresource",
		obj: &authorizationv1.LocalSubjectAccessReview{
			ObjectMeta: metav1.ObjectMeta{Namespace: "a"},
			Spec: authorizationv1.SubjectAccessReviewSpec{
				NonResourceAttributes: &authorizationv1.NonResourceAttributes{},
				User:                  "user",
			},
		},
		msg: "disallowed on this kind of request",
	}, {
		// here we only test one to be sure the function is called.  The more exhaustive suite is tested above.
		name: "resource attributes: label selector specify both",
		obj: &authorizationv1.LocalSubjectAccessReview{
			Spec: authorizationv1.SubjectAccessReviewSpec{
				User: "user",
				ResourceAttributes: &authorizationv1.ResourceAttributes{
					LabelSelector: &authorizationv1.LabelSelectorAttributes{
						RawSelector: "foo",
						Requirements: []metav1.LabelSelectorRequirement{
							{},
						},
					},
				},
			},
		},
		msg: `spec.resourceAttributes.labelSelector.rawSelector: Invalid value: "foo": may not specified at the same time as requirements`,
	}}

	for _, c := range errorCases {
		errs := ValidateLocalSubjectAccessReview(c.obj)
		if len(errs) == 0 {
			t.Errorf("%s: expected failure for %q", c.name, c.msg)
		} else if !strings.Contains(errs[0].Error(), c.msg) {
			t.Errorf("%s: unexpected error: %q, expected: %q", c.name, errs[0], c.msg)
		}
	}
}

// TestValidateAuthorizationConditionsReview exercises the parts of
// ValidateAuthorizationConditionsReview that are handwritten (i.e. not covered
// by declarative validation): the ObjectMeta emptiness check, the
// domain-prefix separator check on every Condition.ID / Condition.Type across
// the request and response ConditionsMaps, and the MaxBytes checks on
// Condition.Condition and Condition.Description. Errors already covered by
// declarative validation (empty ID, invalid label-key format) are intentionally
// not fired by the handwritten path and are therefore not asserted here.
func TestValidateAuthorizationConditionsReview(t *testing.T) {
	emptyDecision := authorizationv1.ConditionsAwareDecision{}
	validConditionsMap := &authorizationv1.ConditionsMap{
		DenyConditions:      []authorizationv1.Condition{{ID: "example.com/deny-1", Type: "example.com/type-1"}},
		NoOpinionConditions: []authorizationv1.Condition{{ID: "example.com/no-op-1"}},
		AllowConditions:     []authorizationv1.Condition{{ID: "example.com/allow-1", Type: "example.io/allow-type"}},
	}

	successCases := []struct {
		name string
		obj  authorizationv1alpha1.AuthorizationConditionsReview
	}{{
		name: "empty request and response decisions",
		obj: authorizationv1alpha1.AuthorizationConditionsReview{
			Request:  &authorizationv1alpha1.AuthorizationConditionsRequest{Decision: emptyDecision},
			Response: &authorizationv1alpha1.AuthorizationConditionsResponse{Decision: emptyDecision},
		},
	}, {
		name: "conditions with valid domain-prefixed keys in all buckets",
		obj: authorizationv1alpha1.AuthorizationConditionsReview{
			Request: &authorizationv1alpha1.AuthorizationConditionsRequest{
				Decision: authorizationv1.ConditionsAwareDecision{ConditionsMap: validConditionsMap},
			},
			Response: &authorizationv1alpha1.AuthorizationConditionsResponse{
				Decision: authorizationv1.ConditionsAwareDecision{ConditionsMap: validConditionsMap},
			},
		},
	}, {
		name: "condition type unset is allowed",
		obj: authorizationv1alpha1.AuthorizationConditionsReview{
			Request: &authorizationv1alpha1.AuthorizationConditionsRequest{
				Decision: authorizationv1.ConditionsAwareDecision{
					ConditionsMap: &authorizationv1.ConditionsMap{
						AllowConditions: []authorizationv1.Condition{{ID: "example.com/allow"}},
					},
				},
			},
			Response: &authorizationv1alpha1.AuthorizationConditionsResponse{Decision: emptyDecision},
		},
	}, {
		name: "only ManagedFields on ObjectMeta is allowed",
		obj: authorizationv1alpha1.AuthorizationConditionsReview{
			ObjectMeta: metav1.ObjectMeta{
				ManagedFields: []metav1.ManagedFieldsEntry{{Manager: "test"}},
			},
			Request:  &authorizationv1alpha1.AuthorizationConditionsRequest{Decision: emptyDecision},
			Response: &authorizationv1alpha1.AuthorizationConditionsResponse{Decision: emptyDecision},
		},
	}, {
		name: "empty id is skipped by handwritten path (declarative covers it)",
		obj: authorizationv1alpha1.AuthorizationConditionsReview{
			Request: &authorizationv1alpha1.AuthorizationConditionsRequest{
				Decision: authorizationv1.ConditionsAwareDecision{
					ConditionsMap: &authorizationv1.ConditionsMap{
						DenyConditions: []authorizationv1.Condition{{ID: ""}},
					},
				},
			},
			Response: &authorizationv1alpha1.AuthorizationConditionsResponse{Decision: emptyDecision},
		},
	}, {
		name: "invalid label-key format is skipped by handwritten path (declarative covers it)",
		obj: authorizationv1alpha1.AuthorizationConditionsReview{
			Request: &authorizationv1alpha1.AuthorizationConditionsRequest{
				Decision: authorizationv1.ConditionsAwareDecision{
					ConditionsMap: &authorizationv1.ConditionsMap{
						AllowConditions: []authorizationv1.Condition{
							{ID: "example.com/foo/bar"},
							{ID: "example.com/_bad", Type: "example.com/e?"},
						},
					},
				},
			},
			Response: &authorizationv1alpha1.AuthorizationConditionsResponse{Decision: emptyDecision},
		},
	}, {
		name: "condition and description exactly at MaxBytes are allowed",
		obj: authorizationv1alpha1.AuthorizationConditionsReview{
			Request: &authorizationv1alpha1.AuthorizationConditionsRequest{
				Decision: authorizationv1.ConditionsAwareDecision{
					ConditionsMap: &authorizationv1.ConditionsMap{
						AllowConditions: []authorizationv1.Condition{{
							ID:          "example.com/foo",
							Condition:   strings.Repeat("a", authorizer.MaxConditionBytes),
							Description: strings.Repeat("b", authorizer.MaxConditionDescriptionBytes),
						}},
					},
				},
			},
			Response: &authorizationv1alpha1.AuthorizationConditionsResponse{Decision: emptyDecision},
		},
	}}

	for _, c := range successCases {
		t.Run("success/"+c.name, func(t *testing.T) {
			if errs := ValidateAuthorizationConditionsReview(&c.obj); len(errs) != 0 {
				t.Errorf("expected success, got: %v", errs)
			}
		})
	}

	errorCases := []struct {
		name string
		obj  authorizationv1alpha1.AuthorizationConditionsReview
		msgs []string
	}{{
		name: "non-empty name in ObjectMeta",
		obj: authorizationv1alpha1.AuthorizationConditionsReview{
			ObjectMeta: metav1.ObjectMeta{Name: "a-name"},
			Request:    &authorizationv1alpha1.AuthorizationConditionsRequest{Decision: emptyDecision},
			Response:   &authorizationv1alpha1.AuthorizationConditionsResponse{Decision: emptyDecision},
		},
		msgs: []string{`metadata: Invalid value:`, `must be empty`},
	}, {
		name: "non-empty namespace in ObjectMeta",
		obj: authorizationv1alpha1.AuthorizationConditionsReview{
			ObjectMeta: metav1.ObjectMeta{Namespace: "ns"},
			Request:    &authorizationv1alpha1.AuthorizationConditionsRequest{Decision: emptyDecision},
			Response:   &authorizationv1alpha1.AuthorizationConditionsResponse{Decision: emptyDecision},
		},
		msgs: []string{`metadata: Invalid value:`, `must be empty`},
	}, {
		name: "non-empty labels in ObjectMeta",
		obj: authorizationv1alpha1.AuthorizationConditionsReview{
			ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"k": "v"}},
			Request:    &authorizationv1alpha1.AuthorizationConditionsRequest{Decision: emptyDecision},
			Response:   &authorizationv1alpha1.AuthorizationConditionsResponse{Decision: emptyDecision},
		},
		msgs: []string{`metadata: Invalid value:`, `must be empty`},
	}, {
		name: "request allowConditions: id missing domain prefix",
		obj: authorizationv1alpha1.AuthorizationConditionsReview{
			Request: &authorizationv1alpha1.AuthorizationConditionsRequest{
				Decision: authorizationv1.ConditionsAwareDecision{
					ConditionsMap: &authorizationv1.ConditionsMap{
						AllowConditions: []authorizationv1.Condition{{ID: "no-slash"}},
					},
				},
			},
			Response: &authorizationv1alpha1.AuthorizationConditionsResponse{Decision: emptyDecision},
		},
		msgs: []string{`request.decision.conditionsMap.allowConditions[0].id: Invalid value: "no-slash": must be a domain-prefixed key`},
	}, {
		name: "request noOpinionConditions: type missing domain prefix",
		obj: authorizationv1alpha1.AuthorizationConditionsReview{
			Request: &authorizationv1alpha1.AuthorizationConditionsRequest{
				Decision: authorizationv1.ConditionsAwareDecision{
					ConditionsMap: &authorizationv1.ConditionsMap{
						NoOpinionConditions: []authorizationv1.Condition{{ID: "example.com/id", Type: "no-slash"}},
					},
				},
			},
			Response: &authorizationv1alpha1.AuthorizationConditionsResponse{Decision: emptyDecision},
		},
		msgs: []string{`request.decision.conditionsMap.noOpinionConditions[0].type: Invalid value: "no-slash": must be a domain-prefixed key`},
	}, {
		name: "response denyConditions: id at index reflects its position",
		obj: authorizationv1alpha1.AuthorizationConditionsReview{
			Request: &authorizationv1alpha1.AuthorizationConditionsRequest{Decision: emptyDecision},
			Response: &authorizationv1alpha1.AuthorizationConditionsResponse{
				Decision: authorizationv1.ConditionsAwareDecision{
					ConditionsMap: &authorizationv1.ConditionsMap{
						DenyConditions: []authorizationv1.Condition{
							{ID: "example.com/ok"},
							{ID: "bad"},
						},
					},
				},
			},
		},
		msgs: []string{`response.decision.conditionsMap.denyConditions[1].id: Invalid value: "bad": must be a domain-prefixed key`},
	}, {
		name: "condition body over MaxConditionBytes",
		obj: authorizationv1alpha1.AuthorizationConditionsReview{
			Request: &authorizationv1alpha1.AuthorizationConditionsRequest{
				Decision: authorizationv1.ConditionsAwareDecision{
					ConditionsMap: &authorizationv1.ConditionsMap{
						AllowConditions: []authorizationv1.Condition{{
							ID:        "example.com/foo",
							Condition: strings.Repeat("a", authorizer.MaxConditionBytes+1),
						}},
					},
				},
			},
			Response: &authorizationv1alpha1.AuthorizationConditionsResponse{Decision: emptyDecision},
		},
		msgs: []string{`request.decision.conditionsMap.allowConditions[0].condition: Too long`},
	}, {
		name: "description over MaxConditionDescriptionBytes",
		obj: authorizationv1alpha1.AuthorizationConditionsReview{
			Request: &authorizationv1alpha1.AuthorizationConditionsRequest{Decision: emptyDecision},
			Response: &authorizationv1alpha1.AuthorizationConditionsResponse{
				Decision: authorizationv1.ConditionsAwareDecision{
					ConditionsMap: &authorizationv1.ConditionsMap{
						DenyConditions: []authorizationv1.Condition{{
							ID:          "example.com/foo",
							Description: strings.Repeat("b", authorizer.MaxConditionDescriptionBytes+1),
						}},
					},
				},
			},
		},
		msgs: []string{`response.decision.conditionsMap.denyConditions[0].description: Too long`},
	}, {
		name: "traversal covers all three buckets in both request and response",
		obj: authorizationv1alpha1.AuthorizationConditionsReview{
			Request: &authorizationv1alpha1.AuthorizationConditionsRequest{
				Decision: authorizationv1.ConditionsAwareDecision{
					ConditionsMap: &authorizationv1.ConditionsMap{
						DenyConditions:      []authorizationv1.Condition{{ID: "bad-deny"}},
						NoOpinionConditions: []authorizationv1.Condition{{ID: "bad-noop"}},
						AllowConditions:     []authorizationv1.Condition{{ID: "bad-allow"}},
					},
				},
			},
			Response: &authorizationv1alpha1.AuthorizationConditionsResponse{
				Decision: authorizationv1.ConditionsAwareDecision{
					ConditionsMap: &authorizationv1.ConditionsMap{
						DenyConditions:      []authorizationv1.Condition{{ID: "bad-deny"}},
						NoOpinionConditions: []authorizationv1.Condition{{ID: "bad-noop"}},
						AllowConditions:     []authorizationv1.Condition{{ID: "bad-allow"}},
					},
				},
			},
		},
		msgs: []string{
			`request.decision.conditionsMap.denyConditions[0].id`,
			`request.decision.conditionsMap.noOpinionConditions[0].id`,
			`request.decision.conditionsMap.allowConditions[0].id`,
			`response.decision.conditionsMap.denyConditions[0].id`,
			`response.decision.conditionsMap.noOpinionConditions[0].id`,
			`response.decision.conditionsMap.allowConditions[0].id`,
		},
	}, {
		name: "nil ConditionsMap in decision is skipped by handwritten validation",
		obj: authorizationv1alpha1.AuthorizationConditionsReview{
			ObjectMeta: metav1.ObjectMeta{Name: "not-empty"},
			Request:    &authorizationv1alpha1.AuthorizationConditionsRequest{Decision: emptyDecision},
			Response:   &authorizationv1alpha1.AuthorizationConditionsResponse{Decision: emptyDecision},
		},
		msgs: []string{`metadata: Invalid value:`},
	}}

	for _, c := range errorCases {
		t.Run(c.name, func(t *testing.T) {
			errs := ValidateAuthorizationConditionsReview(&c.obj)
			if len(errs) == 0 {
				t.Fatalf("expected failure containing %q", c.msgs)
			}
			joined := errs.ToAggregate().Error()
			for _, msg := range c.msgs {
				if !strings.Contains(joined, msg) {
					t.Errorf("expected error containing %q, got: %s", msg, joined)
				}
			}
		})
	}
}

// TestValidateCondition exercises the handwritten checks in ValidateCondition
// in isolation: the domain-prefix separator on ID and Type (fires only when
// the key is otherwise a valid label key), and MaxBytes on Condition and
// Description. Empty IDs and invalid label-key formats are covered by
// declarative validation and are not asserted here.
func TestValidateCondition(t *testing.T) {
	testCases := []struct {
		name    string
		cond    authorizationv1.Condition
		wantErr bool
		msg     string
	}{{
		name: "valid id only",
		cond: authorizationv1.Condition{ID: "example.com/foo"},
	}, {
		name: "valid id and type",
		cond: authorizationv1.Condition{ID: "example.com/foo", Type: "example.io/bar"},
	}, {
		name: "empty id is skipped (declarative covers required)",
		cond: authorizationv1.Condition{ID: ""},
	}, {
		name: "id with too many slashes is skipped (declarative covers label-key format)",
		cond: authorizationv1.Condition{ID: "example.com/foo/bar"},
	}, {
		name: "id with malformed name part is skipped (declarative covers label-key format)",
		cond: authorizationv1.Condition{ID: "example.com/_bad"},
	}, {
		name: "type empty is skipped",
		cond: authorizationv1.Condition{ID: "example.com/foo", Type: ""},
	}, {
		name:    "id missing domain prefix",
		cond:    authorizationv1.Condition{ID: "no-slash"},
		wantErr: true,
		msg:     `id: Invalid value: "no-slash": must be a domain-prefixed key`,
	}, {
		name:    "type set but not domain-prefixed",
		cond:    authorizationv1.Condition{ID: "example.com/foo", Type: "bad"},
		wantErr: true,
		msg:     `type: Invalid value: "bad": must be a domain-prefixed key`,
	}, {
		name: "condition body at MaxBytes is allowed",
		cond: authorizationv1.Condition{
			ID:        "example.com/foo",
			Condition: strings.Repeat("a", authorizer.MaxConditionBytes),
		},
	}, {
		name: "description at MaxBytes is allowed",
		cond: authorizationv1.Condition{
			ID:          "example.com/foo",
			Description: strings.Repeat("b", authorizer.MaxConditionDescriptionBytes),
		},
	}, {
		name: "condition body just over MaxBytes",
		cond: authorizationv1.Condition{
			ID:        "example.com/foo",
			Condition: strings.Repeat("a", authorizer.MaxConditionBytes+1),
		},
		wantErr: true,
		msg:     `condition: Too long`,
	}, {
		name: "description just over MaxBytes",
		cond: authorizationv1.Condition{
			ID:          "example.com/foo",
			Description: strings.Repeat("b", authorizer.MaxConditionDescriptionBytes+1),
		},
		wantErr: true,
		msg:     `description: Too long`,
	}}

	for _, c := range testCases {
		t.Run(c.name, func(t *testing.T) {
			errs := ValidateCondition(&c.cond, field.NewPath("condition"))
			if c.wantErr {
				if len(errs) == 0 {
					t.Fatalf("expected failure containing %q", c.msg)
				}
				if !strings.Contains(errs.ToAggregate().Error(), c.msg) {
					t.Errorf("unexpected error: %v, expected: %q", errs, c.msg)
				}
			} else if len(errs) != 0 {
				t.Errorf("expected success, got: %v", errs)
			}
		})
	}
}

// assertErrors compares the full text of every error in errs against want, in order.
// Matching complete error strings rather than substrings means a change to an error's
// type (for example Invalid to Forbidden) or to its wording is caught here.
func assertErrors(t *testing.T, errs field.ErrorList, want []string) {
	t.Helper()
	var got []string
	for _, e := range errs {
		got = append(got, e.Error())
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("unexpected errors (-want +got):\n%s", diff)
	}
}

// TestValidateSARStatus covers status validation through the three top-level
// validators rather than through ValidateSubjectAccessReviewStatus directly, because
// whether status.conditionalDecision is permitted depends on the spec: each validator
// derives the client's conditions-awareness from spec.authorizationOptions and passes
// it down. Every case therefore pairs a status with the options the client asked for,
// on an otherwise valid spec, so the only errors reported are the status ones.
//
// The rules being exercised are:
//
//   - status.conditionalDecision may only be set when the client opted into
//     conditions-awareness; otherwise the server ignored the client's request for
//     unconditional answers.
//   - It may not carry the unconditional types (Allow/Deny/NoOpinion), which belong in
//     status.allowed / status.denied.
//   - When it carries a conditional type, the unconditional result fields must all be
//     empty, so there is exactly one place to look for the answer. This is what lets
//     the v1beta1 conversion overwrite those fields when it folds a conditional
//     decision down to unconditional form.
func TestValidateSARStatus(t *testing.T) {
	conditionsMap := &authorizationv1.ConditionsMap{
		AllowConditions: []authorizationv1.Condition{{ID: "example.com/allow", Type: "example.com/opaque"}},
	}
	conditionsMapDecision := &authorizationv1.ConditionsAwareDecision{
		Type:          authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
		ConditionsMap: conditionsMap,
	}
	unionDecision := &authorizationv1.ConditionsAwareDecision{
		Type: authorizationv1.ConditionsAwareDecisionTypeUnion,
		Union: []authorizationv1.NamedConditionsAwareDecision{{
			AuthorizerName: "cm",
			Decision: authorizationv1.ConditionsAwareDecision{
				Type:          authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
				ConditionsMap: conditionsMap,
			},
		}},
	}
	conditionalOptions := &authorizationv1.AuthorizationOptions{
		HandledDecisionTypes: []authorizationv1.ConditionsAwareDecisionType{
			authorizationv1.ConditionsAwareDecisionTypeAllow,
			authorizationv1.ConditionsAwareDecisionTypeDeny,
			authorizationv1.ConditionsAwareDecisionTypeNoOpinion,
			authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
			authorizationv1.ConditionsAwareDecisionTypeUnion,
		},
	}
	unconditionalOptions := &authorizationv1.AuthorizationOptions{
		HandledDecisionTypes: []authorizationv1.ConditionsAwareDecisionType{
			authorizationv1.ConditionsAwareDecisionTypeAllow,
			authorizationv1.ConditionsAwareDecisionTypeDeny,
			authorizationv1.ConditionsAwareDecisionTypeNoOpinion,
		},
	}
	const notOptedIn = "status.conditionalDecision: Forbidden: can only be set when the client opted into conditions-awareness"

	testCases := []struct {
		name string
		// options is what the client advertised it can handle, and is what decides
		// whether a conditional decision is allowed in the status at all.
		options *authorizationv1.AuthorizationOptions
		status  authorizationv1.SubjectAccessReviewStatus
		// msgs is the exact, ordered list of expected errors; nil means valid.
		msgs []string
	}{{
		name:   "empty status",
		status: authorizationv1.SubjectAccessReviewStatus{},
	}, {
		name:   "allowed only",
		status: authorizationv1.SubjectAccessReviewStatus{Allowed: true},
	}, {
		name:   "denied only",
		status: authorizationv1.SubjectAccessReviewStatus{Denied: true},
	}, {
		// reason and evaluationError are only constrained alongside a conditional
		// decision; on their own they are ordinary result metadata.
		name: "reason and evaluationError without a conditional decision",
		status: authorizationv1.SubjectAccessReviewStatus{
			Allowed:         true,
			Reason:          "rbac allowed",
			EvaluationError: "one authorizer was flaky",
		},
	}, {
		// The error echoes the whole status, so this case keeps the status minimal to
		// keep the expectation readable.
		name: "allowed and denied are mutually exclusive",
		status: authorizationv1.SubjectAccessReviewStatus{
			Allowed: true,
			Denied:  true,
		},
		msgs: []string{`status: Invalid value: {"allowed":true,"denied":true}: allowed and denied are mutually exclusive`},
	}, {
		// Nil options means unconditional-only, so a conditional answer was never
		// asked for.
		name:    "conditional decision with nil options is not opted in",
		options: nil,
		status: authorizationv1.SubjectAccessReviewStatus{
			ConditionalDecision: conditionsMapDecision,
		},
		msgs: []string{notOptedIn},
	}, {
		name:    "conditional decision with unconditional options is not opted in",
		options: unconditionalOptions,
		status: authorizationv1.SubjectAccessReviewStatus{
			ConditionalDecision: conditionsMapDecision,
		},
		msgs: []string{notOptedIn},
	}, {
		name:    "ConditionsMap decision for an opted-in client",
		options: conditionalOptions,
		status: authorizationv1.SubjectAccessReviewStatus{
			ConditionalDecision: conditionsMapDecision,
		},
	}, {
		name:    "Union decision for an opted-in client",
		options: conditionalOptions,
		status: authorizationv1.SubjectAccessReviewStatus{
			ConditionalDecision: unionDecision,
		},
	}, {
		name:    "conditionalDecision.type=Allow is rejected",
		options: conditionalOptions,
		status: authorizationv1.SubjectAccessReviewStatus{
			ConditionalDecision: &authorizationv1.ConditionsAwareDecision{
				Type:  authorizationv1.ConditionsAwareDecisionTypeAllow,
				Allow: &authorizationv1.UnconditionalDecision{Reason: "allowed"},
			},
		},
		msgs: []string{`status.conditionalDecision.type: Invalid value: "Allow": cannot be one of [Allow, Deny, NoOpinion], these decisions must be expressed using status.allowed and status.denied`},
	}, {
		name:    "conditionalDecision.type=Deny is rejected",
		options: conditionalOptions,
		status: authorizationv1.SubjectAccessReviewStatus{
			ConditionalDecision: &authorizationv1.ConditionsAwareDecision{
				Type: authorizationv1.ConditionsAwareDecisionTypeDeny,
				Deny: &authorizationv1.UnconditionalDecision{Reason: "denied"},
			},
		},
		msgs: []string{`status.conditionalDecision.type: Invalid value: "Deny": cannot be one of [Allow, Deny, NoOpinion], these decisions must be expressed using status.allowed and status.denied`},
	}, {
		name:    "conditionalDecision.type=NoOpinion is rejected",
		options: conditionalOptions,
		status: authorizationv1.SubjectAccessReviewStatus{
			ConditionalDecision: &authorizationv1.ConditionsAwareDecision{
				Type:      authorizationv1.ConditionsAwareDecisionTypeNoOpinion,
				NoOpinion: &authorizationv1.UnconditionalDecision{},
			},
		},
		msgs: []string{`status.conditionalDecision.type: Invalid value: "NoOpinion": cannot be one of [Allow, Deny, NoOpinion], these decisions must be expressed using status.allowed and status.denied`},
	}, {
		name:    "allowed must be false alongside a ConditionsMap",
		options: conditionalOptions,
		status: authorizationv1.SubjectAccessReviewStatus{
			Allowed:             true,
			ConditionalDecision: conditionsMapDecision,
		},
		msgs: []string{"status.allowed: Forbidden: must be false when status.conditionalDecision.type=ConditionsMap"},
	}, {
		name:    "denied must be false alongside a ConditionsMap",
		options: conditionalOptions,
		status: authorizationv1.SubjectAccessReviewStatus{
			Denied:              true,
			ConditionalDecision: conditionsMapDecision,
		},
		msgs: []string{"status.denied: Forbidden: must be false when status.conditionalDecision.type=ConditionsMap"},
	}, {
		name:    "evaluationError must be empty alongside a ConditionsMap",
		options: conditionalOptions,
		status: authorizationv1.SubjectAccessReviewStatus{
			EvaluationError:     "an authorizer failed",
			ConditionalDecision: conditionsMapDecision,
		},
		msgs: []string{"status.evaluationError: Forbidden: must be empty when status.conditionalDecision.type=ConditionsMap"},
	}, {
		name:    "reason must be empty alongside a ConditionsMap",
		options: conditionalOptions,
		status: authorizationv1.SubjectAccessReviewStatus{
			Reason:              "because",
			ConditionalDecision: conditionsMapDecision,
		},
		msgs: []string{"status.reason: Forbidden: must be empty when status.conditionalDecision.type=ConditionsMap"},
	}, {
		// The message names the offending type, so Union reads differently to
		// ConditionsMap.
		name:    "reason must be empty alongside a Union",
		options: conditionalOptions,
		status: authorizationv1.SubjectAccessReviewStatus{
			Reason:              "because",
			ConditionalDecision: unionDecision,
		},
		msgs: []string{"status.reason: Forbidden: must be empty when status.conditionalDecision.type=Union"},
	}, {
		// Each unconditional field is reported independently, in the order the
		// validator checks them. Denied is left unset here so the mutual-exclusion
		// error, which echoes the whole status, stays out of the expectation.
		name:    "several unconditional fields set alongside a ConditionsMap",
		options: conditionalOptions,
		status: authorizationv1.SubjectAccessReviewStatus{
			Allowed:             true,
			Reason:              "because",
			EvaluationError:     "an authorizer failed",
			ConditionalDecision: conditionsMapDecision,
		},
		msgs: []string{
			"status.allowed: Forbidden: must be false when status.conditionalDecision.type=ConditionsMap",
			"status.evaluationError: Forbidden: must be empty when status.conditionalDecision.type=ConditionsMap",
			"status.reason: Forbidden: must be empty when status.conditionalDecision.type=ConditionsMap",
		},
	}, {
		// Not opting in and sending a rejected type are independent failures.
		name:    "not opted in and an unconditional type together",
		options: nil,
		status: authorizationv1.SubjectAccessReviewStatus{
			ConditionalDecision: &authorizationv1.ConditionsAwareDecision{
				Type:  authorizationv1.ConditionsAwareDecisionTypeAllow,
				Allow: &authorizationv1.UnconditionalDecision{},
			},
		},
		msgs: []string{
			notOptedIn,
			`status.conditionalDecision.type: Invalid value: "Allow": cannot be one of [Allow, Deny, NoOpinion], these decisions must be expressed using status.allowed and status.denied`,
		},
	}, {
		// An unrecognized type matches no arm of the handwritten switch, so the
		// unconditional fields are not constrained; declarative validation is
		// responsible for rejecting the type itself.
		name:    "unrecognized type is left to declarative validation",
		options: conditionalOptions,
		status: authorizationv1.SubjectAccessReviewStatus{
			Allowed:             true,
			Reason:              "because",
			ConditionalDecision: &authorizationv1.ConditionsAwareDecision{Type: "SomeFutureType"},
		},
	}, {
		// The decision is descended into, so a malformed condition is reported at its
		// full path.
		name:    "condition errors inside the decision are reported",
		options: conditionalOptions,
		status: authorizationv1.SubjectAccessReviewStatus{
			ConditionalDecision: &authorizationv1.ConditionsAwareDecision{
				Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
				ConditionsMap: &authorizationv1.ConditionsMap{
					AllowConditions: []authorizationv1.Condition{{ID: "no-slash"}},
				},
			},
		},
		msgs: []string{`status.conditionalDecision.conditionsMap.allowConditions[0].id: Invalid value: "no-slash": must be a domain-prefixed key (such as "acme.io/foo")`},
	}, {
		// ValidateConditionsAwareDecision only descends into ConditionsMap, not into
		// Union members, so a malformed condition nested in a Union is not reported by
		// the handwritten path.
		name:    "condition errors nested in a Union are left to declarative validation",
		options: conditionalOptions,
		status: authorizationv1.SubjectAccessReviewStatus{
			ConditionalDecision: &authorizationv1.ConditionsAwareDecision{
				Type: authorizationv1.ConditionsAwareDecisionTypeUnion,
				Union: []authorizationv1.NamedConditionsAwareDecision{{
					AuthorizerName: "cm",
					Decision: authorizationv1.ConditionsAwareDecision{
						Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
						ConditionsMap: &authorizationv1.ConditionsMap{
							AllowConditions: []authorizationv1.Condition{{ID: "no-slash"}},
						},
					},
				}},
			},
		},
	}}

	for _, c := range testCases {
		t.Run(c.name, func(t *testing.T) {
			// A spec that is valid on its own, so every reported error comes from the
			// status. The namespace is set because LocalSubjectAccessReview requires
			// spec.resourceAttributes.namespace to match metadata.namespace.
			sarSpec := authorizationv1.SubjectAccessReviewSpec{
				ResourceAttributes:   &authorizationv1.ResourceAttributes{Namespace: "ns"},
				User:                 "me",
				AuthorizationOptions: c.options,
			}

			t.Run("SubjectAccessReview", func(t *testing.T) {
				assertErrors(t, ValidateSubjectAccessReview(&authorizationv1.SubjectAccessReview{
					Spec:   sarSpec,
					Status: c.status,
				}), c.msgs)
			})

			t.Run("SelfSubjectAccessReview", func(t *testing.T) {
				assertErrors(t, ValidateSelfSubjectAccessReview(&authorizationv1.SelfSubjectAccessReview{
					Spec: authorizationv1.SelfSubjectAccessReviewSpec{
						ResourceAttributes:   &authorizationv1.ResourceAttributes{Namespace: "ns"},
						AuthorizationOptions: c.options,
					},
					Status: c.status,
				}), c.msgs)
			})

			t.Run("LocalSubjectAccessReview", func(t *testing.T) {
				assertErrors(t, ValidateLocalSubjectAccessReview(&authorizationv1.LocalSubjectAccessReview{
					ObjectMeta: metav1.ObjectMeta{Namespace: "ns"},
					Spec:       sarSpec,
					Status:     c.status,
				}), c.msgs)
			})
		})
	}
}
