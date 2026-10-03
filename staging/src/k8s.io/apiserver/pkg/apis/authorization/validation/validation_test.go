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
	"context"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	admissionv1 "k8s.io/api/admission/v1"
	authorizationv1 "k8s.io/api/authorization/v1"
	authorizationv1alpha1 "k8s.io/api/authorization/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

// longHandledDecisionTypes has 33 entries, one past the 32-entry maximum that the
// handwritten superset check is willing to walk. None of them is an unconditional
// decision type, so neither this list nor its 32-entry prefix is a superset of
// {Allow, Deny, NoOpinion}.
var longHandledDecisionTypes = []authorizationv1.ConditionsAwareDecisionType{
	"T01", "T02", "T03", "T04", "T05", "T06", "T07", "T08", "T09", "T10", "T11",
	"T12", "T13", "T14", "T15", "T16", "T17", "T18", "T19", "T20", "T21", "T22",
	"T23", "T24", "T25", "T26", "T27", "T28", "T29", "T30", "T31", "T32", "T33",
}

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
		{ // over the 32-entry limit: not walked here, so the missing unconditional
			// types are not reported even though the set is not a superset. Declarative
			// validation reports the maxItems violation instead.
			User: "me",
			ResourceAttributes: &authorizationv1.ResourceAttributes{
				Verb:     "create",
				Resource: "pods",
			},
			AuthorizationOptions: &authorizationv1.AuthorizationOptions{
				HandledDecisionTypes: longHandledDecisionTypes,
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
	}, {
		// Exactly at the 32-entry limit, so the superset check still runs. One more
		// entry and it would be skipped, as the success cases above show.
		name: "authorization options: at the 32-entry limit the superset check still runs",
		obj: authorizationv1.SubjectAccessReviewSpec{
			User: "me",
			ResourceAttributes: &authorizationv1.ResourceAttributes{
				Verb:     "create",
				Resource: "pods",
			},
			AuthorizationOptions: &authorizationv1.AuthorizationOptions{
				HandledDecisionTypes: longHandledDecisionTypes[:32],
			},
		},
		msg: `spec.authorizationOptions.handledDecisionTypes: Invalid value: ["T01","T02","T03","T04","T05","T06","T07","T08","T09","T10","T11","T12","T13","T14","T15","T16","T17","T18","T19","T20","T21","T22","T23","T24","T25","T26","T27","T28","T29","T30","T31","T32"]: set must at least contain {Allow, Deny, NoOpinion}`,
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

	// The error cases above only carry a spec. Here we only test one status error to be
	// sure the status is validated for this kind too; the more exhaustive suite is
	// TestValidateSARStatus.
	t.Run("status is validated", func(t *testing.T) {
		errs := ValidateSelfSubjectAccessReview(&authorizationv1.SelfSubjectAccessReview{
			Spec: authorizationv1.SelfSubjectAccessReviewSpec{
				ResourceAttributes: &authorizationv1.ResourceAttributes{},
			},
			Status: authorizationv1.SubjectAccessReviewStatus{Allowed: true, Denied: true},
		})

		want := `status: Invalid value: {"allowed":true,"denied":true}: allowed and denied are mutually exclusive`
		assertErrors(t, errs, []string{want})
	})
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
	}, {
		// here we only test one to be sure the status is validated for this kind too.
		// The more exhaustive suite is TestValidateSARStatus.
		name: "status: allowed and denied are mutually exclusive",
		obj: &authorizationv1.LocalSubjectAccessReview{
			Spec: authorizationv1.SubjectAccessReviewSpec{
				ResourceAttributes: &authorizationv1.ResourceAttributes{},
				User:               "user",
			},
			Status: authorizationv1.SubjectAccessReviewStatus{Allowed: true, Denied: true},
		},
		msg: `status: Invalid value: {"allowed":true,"denied":true}: allowed and denied are mutually exclusive`,
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

// TestValidateAuthorizationConditionsReview exercises AuthorizationConditionsReview
// validation through CombinedValidateAuthorizationConditionsReviewCreate, which is what
// the webhook authorizer actually calls: the handwritten checks plus the generated
// declarative ones.
//
// Two rules are handwritten. ObjectMeta must be empty, since the review is a
// non-persisted request/response envelope, with ManagedFields exempt because the API
// machinery may set it. And response.decision.type must be an unconditional decision,
// because evaluating conditions can only ever produce Allow, Deny or NoOpinion.
// Everything else, including the contents of the conditions, comes from declarative
// validation.
func TestValidateAuthorizationConditionsReview(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ConditionalAuthorization, true)

	// A response must echo the UID of the request it is answering, so both sides of a
	// valid envelope carry the same one.
	const uid = "test-uid"

	conditionsMapDecision := authorizationv1.ConditionsAwareDecision{
		Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
		ConditionsMap: &authorizationv1.ConditionsMap{
			DenyConditions:      []authorizationv1.Condition{{ID: "example.com/deny-1", Type: "example.com/type-1"}},
			NoOpinionConditions: []authorizationv1.Condition{{ID: "example.com/no-op-1"}},
			AllowConditions:     []authorizationv1.Condition{{ID: "example.com/allow-1", Type: "example.io/allow-type"}},
		},
	}
	denyDecision := authorizationv1.ConditionsAwareDecision{
		Type: authorizationv1.ConditionsAwareDecisionTypeDeny,
		Deny: &authorizationv1.UnconditionalDecision{Reason: "denied"},
	}
	allowDecision := authorizationv1.ConditionsAwareDecision{
		Type:  authorizationv1.ConditionsAwareDecisionTypeAllow,
		Allow: &authorizationv1.UnconditionalDecision{Reason: "allowed"},
	}
	noOpinionDecision := authorizationv1.ConditionsAwareDecision{
		Type:      authorizationv1.ConditionsAwareDecisionTypeNoOpinion,
		NoOpinion: &authorizationv1.UnconditionalDecision{},
	}

	testCases := []struct {
		name             string
		objectMeta       metav1.ObjectMeta
		requestDecision  authorizationv1.ConditionsAwareDecision
		responseDecision authorizationv1.ConditionsAwareDecision
		// msgs is the exact, ordered list of expected errors; nil means valid.
		msgs []string
	}{{
		name:             "conditional request, unconditional deny response",
		requestDecision:  conditionsMapDecision,
		responseDecision: denyDecision,
	}, {
		name:             "unconditional allow response",
		requestDecision:  conditionsMapDecision,
		responseDecision: allowDecision,
	}, {
		name:             "unconditional no-opinion response",
		requestDecision:  conditionsMapDecision,
		responseDecision: noOpinionDecision,
	}, {
		name:             "only ManagedFields on ObjectMeta is allowed",
		objectMeta:       metav1.ObjectMeta{ManagedFields: []metav1.ManagedFieldsEntry{{Manager: "test"}}},
		requestDecision:  conditionsMapDecision,
		responseDecision: allowDecision,
	}, {
		name:             "non-empty ObjectMeta",
		objectMeta:       metav1.ObjectMeta{Name: "a-name"},
		requestDecision:  conditionsMapDecision,
		responseDecision: allowDecision,
		msgs:             []string{`metadata: Invalid value: {"name":"a-name"}: must be empty`},
	}, {
		// The type is required, which only declarative validation reports; the
		// handwritten unconditional-only check deliberately skips an unset type.
		name:             "unset response decision type",
		requestDecision:  conditionsMapDecision,
		responseDecision: authorizationv1.ConditionsAwareDecision{},
		msgs:             []string{"response.decision.type: Required value"},
	}, {
		// A conditional response cannot be the result of evaluating conditions.
		name:             "ConditionsMap response is not supported",
		requestDecision:  conditionsMapDecision,
		responseDecision: conditionsMapDecision,
		msgs:             []string{`response.decision.type: Invalid value: "ConditionsMap": currently must evaluate to an unconditional decision`},
	}, {
		name:            "Union response is not supported",
		requestDecision: conditionsMapDecision,
		responseDecision: authorizationv1.ConditionsAwareDecision{
			Type: authorizationv1.ConditionsAwareDecisionTypeUnion,
			Union: []authorizationv1.NamedConditionsAwareDecision{{
				AuthorizerName: "cm",
				Decision:       conditionsMapDecision,
			}},
		},
		msgs: []string{`response.decision.type: Invalid value: "Union": currently must evaluate to an unconditional decision`},
	}, {
		name:             "unrecognized response type is not supported",
		requestDecision:  conditionsMapDecision,
		responseDecision: authorizationv1.ConditionsAwareDecision{Type: "SomeFutureType"},
		msgs: []string{
			`response.decision.type: Invalid value: "SomeFutureType": currently must evaluate to an unconditional decision`,
			`response.decision.type: Unsupported value: "SomeFutureType": supported values: "Allow", "ConditionsMap", "Deny", "NoOpinion", "Union"`,
		},
	}, {
		// The conditions in the request are reported by declarative validation; the
		// handwritten validators no longer descend into a decision at all.
		name: "malformed request conditions are reported by declarative validation",
		requestDecision: authorizationv1.ConditionsAwareDecision{
			Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
			ConditionsMap: &authorizationv1.ConditionsMap{
				DenyConditions: []authorizationv1.Condition{{
					ID:          "no-slash",
					Type:        "also-no-slash",
					Condition:   strings.Repeat("a", authorizer.MaxConditionBytes+1),
					Description: strings.Repeat("b", authorizer.MaxConditionDescriptionBytes+1),
				}},
			},
		},
		responseDecision: allowDecision,
		msgs: []string{
			`request.decision.conditionsMap.denyConditions[0].id: Invalid value: "no-slash": must include a prefix (e.g. 'example.com/key')`,
			"request.decision.conditionsMap.denyConditions[0].condition: Too long: may not be more than 10240 bytes",
			`request.decision.conditionsMap.denyConditions[0].type: Invalid value: "also-no-slash": must include a prefix (e.g. 'example.com/key')`,
			"request.decision.conditionsMap.denyConditions[0].description: Too long: may not be more than 1024 bytes",
		},
	}}

	for _, c := range testCases {
		t.Run(c.name, func(t *testing.T) {
			acr := &authorizationv1alpha1.AuthorizationConditionsReview{
				ObjectMeta: c.objectMeta,
				Request: &authorizationv1alpha1.AuthorizationConditionsRequest{
					AdmissionRequest: &admissionv1.AdmissionRequest{UID: uid},
					Decision:         c.requestDecision,
				},
				Response: &authorizationv1alpha1.AuthorizationConditionsResponse{
					UID:      uid,
					Decision: c.responseDecision,
				},
			}
			assertErrors(t, CombinedValidateAuthorizationConditionsReviewCreate(context.Background(), acr), c.msgs)
		})
	}

	// The request and response are each optional on the wire, so an envelope carrying
	// neither is structurally valid; the webhook authorizer checks for a missing
	// response separately.
	t.Run("nil request and response", func(t *testing.T) {
		assertErrors(t, CombinedValidateAuthorizationConditionsReviewCreate(context.Background(),
			&authorizationv1alpha1.AuthorizationConditionsReview{}), nil)
	})
}

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
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ConditionalAuthorization, true)

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
		// unconditional fields alongside it are not constrained. Declarative validation
		// rejects the type itself.
		name:    "unrecognized type is rejected by declarative validation",
		options: conditionalOptions,
		status: authorizationv1.SubjectAccessReviewStatus{
			Allowed:             true,
			Reason:              "because",
			ConditionalDecision: &authorizationv1.ConditionsAwareDecision{Type: "SomeFutureType"},
		},
		msgs: []string{`status.conditionalDecision.type: Unsupported value: "SomeFutureType": supported values: "Allow", "ConditionsMap", "Deny", "NoOpinion", "Union"`},
	}, {
		// The handwritten validators do not descend into a decision, so the contents of
		// the conditions are reported by declarative validation, which reaches them even
		// when they are nested inside a Union.
		name:    "malformed condition contents are rejected by declarative validation",
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
		msgs: []string{`status.conditionalDecision.union[0].decision.conditionsMap.allowConditions[0].id: Invalid value: "no-slash": must include a prefix (e.g. 'example.com/key')`},
	}}

	for _, c := range testCases {
		t.Run(c.name, func(t *testing.T) {
			// A spec that is valid on its own, so every reported error comes from the
			// status.
			assertErrors(t, CombinedValidateSubjectAccessReviewCreate(context.Background(), &authorizationv1.SubjectAccessReview{
				Spec: authorizationv1.SubjectAccessReviewSpec{
					ResourceAttributes:   &authorizationv1.ResourceAttributes{Namespace: "ns"},
					User:                 "me",
					AuthorizationOptions: c.options,
				},
				Status: c.status,
			}), c.msgs)
		})
	}
}

// TestCombinedValidateSubjectAccessReviewCreate covers the wrapper the webhook
// authorizer uses to check a SubjectAccessReview it got back from a webhook. It runs the
// handwritten and the generated declarative validation together and aggregates both, and
// it guards against a panic in either so a validator crash surfaces as an internal error
// instead of taking down the request goroutine.
func TestCombinedValidateSubjectAccessReviewCreate(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ConditionalAuthorization, true)

	t.Run("recovers a panic", func(t *testing.T) {
		errs := CombinedValidateSubjectAccessReviewCreate(context.Background(), nil)

		if len(errs) != 1 {
			t.Fatalf("expected exactly 1 error, got: %v", errs)
		}
		if got, want := errs[0].Type, field.ErrorTypeInternal; got != want {
			t.Errorf("expected type %q, got %q", want, got)
		}
		if !strings.HasPrefix(errs[0].Detail, "panic during SAR validation: ") {
			t.Errorf("expected the detail to report a recovered panic, got %q", errs[0].Detail)
		}
	})

	t.Run("valid review", func(t *testing.T) {
		sar := &authorizationv1.SubjectAccessReview{
			Spec: authorizationv1.SubjectAccessReviewSpec{
				ResourceAttributes: &authorizationv1.ResourceAttributes{Verb: "create", Resource: "pods"},
				User:               "me",
			},
			Status: authorizationv1.SubjectAccessReviewStatus{Allowed: true},
		}
		assertErrors(t, CombinedValidateSubjectAccessReviewCreate(context.Background(), sar), nil)
	})

	t.Run("handwritten and declarative errors are combined", func(t *testing.T) {
		// The options are missing NoOpinion, which only the handwritten check reports.
		// The conditional decision claims a ConditionsMap without supplying one, which
		// only declarative validation reports.
		sar := &authorizationv1.SubjectAccessReview{
			Spec: authorizationv1.SubjectAccessReviewSpec{
				ResourceAttributes: &authorizationv1.ResourceAttributes{Verb: "create", Resource: "pods"},
				User:               "me",
				AuthorizationOptions: &authorizationv1.AuthorizationOptions{
					HandledDecisionTypes: []authorizationv1.ConditionsAwareDecisionType{
						authorizationv1.ConditionsAwareDecisionTypeAllow,
						authorizationv1.ConditionsAwareDecisionTypeDeny,
					},
				},
			},
			Status: authorizationv1.SubjectAccessReviewStatus{
				ConditionalDecision: &authorizationv1.ConditionsAwareDecision{
					Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
				},
			},
		}

		assertErrors(t, CombinedValidateSubjectAccessReviewCreate(context.Background(), sar), []string{
			`spec.authorizationOptions.handledDecisionTypes: Invalid value: ["Allow","Deny"]: set must at least contain {Allow, Deny, NoOpinion}`,
			"status.conditionalDecision: Forbidden: can only be set when the client opted into conditions-awareness",
			"status.conditionalDecision.conditionsMap: Invalid value: \"\": must be specified when `type` is \"ConditionsMap\"",
		})
	})
}

// TestCombinedValidateAuthorizationConditionsReviewCreate covers the same composition
// and panic guard for AuthorizationConditionsReview.
func TestCombinedValidateAuthorizationConditionsReviewCreate(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ConditionalAuthorization, true)

	conditionsMapDecision := authorizationv1.ConditionsAwareDecision{
		Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
		ConditionsMap: &authorizationv1.ConditionsMap{
			AllowConditions: []authorizationv1.Condition{{ID: "example.com/allow", Type: "example.com/opaque"}},
		},
	}

	t.Run("recovers a panic", func(t *testing.T) {
		errs := CombinedValidateAuthorizationConditionsReviewCreate(context.Background(), nil)

		if len(errs) != 1 {
			t.Fatalf("expected exactly 1 error, got: %v", errs)
		}
		if got, want := errs[0].Type, field.ErrorTypeInternal; got != want {
			t.Errorf("expected type %q, got %q", want, got)
		}
		if !strings.HasPrefix(errs[0].Detail, "panic during ACR validation: ") {
			t.Errorf("expected the detail to report a recovered panic, got %q", errs[0].Detail)
		}
	})

	t.Run("valid review", func(t *testing.T) {
		acr := &authorizationv1alpha1.AuthorizationConditionsReview{
			Request: &authorizationv1alpha1.AuthorizationConditionsRequest{
				AdmissionRequest: &admissionv1.AdmissionRequest{UID: "test-uid"},
				Decision:         conditionsMapDecision,
			},
			Response: &authorizationv1alpha1.AuthorizationConditionsResponse{
				UID: "test-uid",
				Decision: authorizationv1.ConditionsAwareDecision{
					Type:  authorizationv1.ConditionsAwareDecisionTypeAllow,
					Allow: &authorizationv1.UnconditionalDecision{Reason: "allowed"},
				},
			},
		}
		assertErrors(t, CombinedValidateAuthorizationConditionsReviewCreate(context.Background(), acr), nil)
	})

	t.Run("handwritten and declarative errors are combined", func(t *testing.T) {
		// A conditional response is rejected by the handwritten check, while the
		// non-domain-prefixed condition ID in the request is reported by declarative
		// validation, which is the only layer that still descends into conditions.
		acr := &authorizationv1alpha1.AuthorizationConditionsReview{
			Request: &authorizationv1alpha1.AuthorizationConditionsRequest{
				AdmissionRequest: &admissionv1.AdmissionRequest{UID: "test-uid"},
				Decision: authorizationv1.ConditionsAwareDecision{
					Type: authorizationv1.ConditionsAwareDecisionTypeConditionsMap,
					ConditionsMap: &authorizationv1.ConditionsMap{
						AllowConditions: []authorizationv1.Condition{{ID: "nodomain", Type: "example.com/opaque"}},
					},
				},
			},
			Response: &authorizationv1alpha1.AuthorizationConditionsResponse{
				UID:      "test-uid",
				Decision: conditionsMapDecision,
			},
		}

		assertErrors(t, CombinedValidateAuthorizationConditionsReviewCreate(context.Background(), acr), []string{
			`response.decision.type: Invalid value: "ConditionsMap": currently must evaluate to an unconditional decision`,
			`request.decision.conditionsMap.allowConditions[0].id: Invalid value: "nodomain": must include a prefix (e.g. 'example.com/key')`,
		})
	})
}
