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

	authorizationv1 "k8s.io/api/authorization/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
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
