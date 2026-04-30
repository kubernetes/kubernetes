/*
Copyright 2017 The Kubernetes Authors.

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

package util

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/selection"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	authorizationapi "k8s.io/kubernetes/pkg/apis/authorization"
)

func TestResourceAttributesFrom(t *testing.T) {
	knownResourceAttributesNames := sets.NewString(
		// Fields we copy in ResourceAttributesFrom
		"Verb",
		"Namespace",
		"Group",
		"Version",
		"Resource",
		"Subresource",
		"Name",

		// Fields we read and parse in ResourceAttributesFrom
		"FieldSelector",
		"LabelSelector",

		// Fields we copy in NonResourceAttributesFrom
		"Path",
		"Verb",
	)
	reflect.TypeOf(authorizationapi.ResourceAttributes{}).FieldByNameFunc(func(name string) bool {
		if !knownResourceAttributesNames.Has(name) {
			t.Errorf("authorizationapi.ResourceAttributes has a new field: %q. Add to ResourceAttributesFrom/NonResourceAttributesFrom as appropriate, then add to knownResourceAttributesNames", name)
		}
		return false
	})

	knownAttributesRecordFieldNames := sets.NewString(
		// Fields we set in ResourceAttributesFrom
		"User",
		"Verb",
		"Namespace",
		"APIGroup",
		"APIVersion",
		"Resource",
		"Subresource",
		"Name",
		"ResourceRequest",

		// Fields we compute and set in ResourceAttributesFrom
		"FieldSelectorRequirements",
		"FieldSelectorParsingErr",
		"LabelSelectorRequirements",
		"LabelSelectorParsingErr",

		// Fields we set in NonResourceAttributesFrom
		"User",
		"ResourceRequest",
		"Path",
		"Verb",
	)
	reflect.TypeOf(authorizer.AttributesRecord{}).FieldByNameFunc(func(name string) bool {
		if !knownAttributesRecordFieldNames.Has(name) {
			t.Errorf("authorizer.AttributesRecord has a new field: %q. Add to ResourceAttributesFrom/NonResourceAttributesFrom as appropriate, then add to knownAttributesRecordFieldNames", name)
		}
		return false
	})
}

func TestAuthorizationAttributesFrom(t *testing.T) {
	mustRequirement := func(key string, op selection.Operator, vals []string) labels.Requirement {
		ret, err := labels.NewRequirement(key, op, vals)
		if err != nil {
			panic(err)
		}
		return *ret
	}

	type args struct {
		spec authorizationapi.SubjectAccessReviewSpec
	}
	tests := []struct {
		name                        string
		args                        args
		want                        authorizer.AttributesRecord
		enableAuthorizationSelector bool
	}{
		{
			name: "nonresource",
			args: args{
				spec: authorizationapi.SubjectAccessReviewSpec{
					User:                  "bob",
					Groups:                []string{user.AllAuthenticated},
					NonResourceAttributes: &authorizationapi.NonResourceAttributes{Verb: "get", Path: "/mypath"},
					Extra:                 map[string]authorizationapi.ExtraValue{"scopes": {"scope-a", "scope-b"}},
				},
			},
			want: authorizer.AttributesRecord{
				User: &user.DefaultInfo{
					Name:   "bob",
					Groups: []string{user.AllAuthenticated},
					Extra:  map[string][]string{"scopes": {"scope-a", "scope-b"}},
				},
				Verb: "get",
				Path: "/mypath",
			},
		},
		{
			name: "resource",
			args: args{
				spec: authorizationapi.SubjectAccessReviewSpec{
					User: "bob",
					ResourceAttributes: &authorizationapi.ResourceAttributes{
						Namespace:   "myns",
						Verb:        "create",
						Group:       "extensions",
						Version:     "v1beta1",
						Resource:    "deployments",
						Subresource: "scale",
						Name:        "mydeployment",
					},
				},
			},
			want: authorizer.AttributesRecord{
				User: &user.DefaultInfo{
					Name: "bob",
				},
				APIGroup:        "extensions",
				APIVersion:      "v1beta1",
				Namespace:       "myns",
				Verb:            "create",
				Resource:        "deployments",
				Subresource:     "scale",
				Name:            "mydeployment",
				ResourceRequest: true,
			},
		},
		{
			name: "resource with no version",
			args: args{
				spec: authorizationapi.SubjectAccessReviewSpec{
					User: "bob",
					ResourceAttributes: &authorizationapi.ResourceAttributes{
						Namespace:   "myns",
						Verb:        "create",
						Group:       "extensions",
						Resource:    "deployments",
						Subresource: "scale",
						Name:        "mydeployment",
					},
				},
			},
			want: authorizer.AttributesRecord{
				User: &user.DefaultInfo{
					Name: "bob",
				},
				APIGroup:        "extensions",
				APIVersion:      "*",
				Namespace:       "myns",
				Verb:            "create",
				Resource:        "deployments",
				Subresource:     "scale",
				Name:            "mydeployment",
				ResourceRequest: true,
			},
		},
		{
			name: "field: ignore when featuregate off",
			args: args{
				spec: authorizationapi.SubjectAccessReviewSpec{
					ResourceAttributes: &authorizationapi.ResourceAttributes{
						FieldSelector: &authorizationapi.FieldSelectorAttributes{
							RawSelector: "foo=bar",
						},
					},
				},
			},
			want: authorizer.AttributesRecord{
				User:            &user.DefaultInfo{},
				ResourceRequest: true,
				APIVersion:      "*",
			},
		},
		{
			name: "field: raw selector",
			args: args{
				spec: authorizationapi.SubjectAccessReviewSpec{
					ResourceAttributes: &authorizationapi.ResourceAttributes{
						FieldSelector: &authorizationapi.FieldSelectorAttributes{
							RawSelector: "foo=bar",
						},
					},
				},
			},
			want: authorizer.AttributesRecord{
				User:            &user.DefaultInfo{},
				ResourceRequest: true,
				APIVersion:      "*",
				FieldSelectorRequirements: fields.Requirements{
					{Operator: "=", Field: "foo", Value: "bar"},
				},
			},
			enableAuthorizationSelector: true,
		},
		{
			name: "field: raw selector error",
			args: args{
				spec: authorizationapi.SubjectAccessReviewSpec{
					ResourceAttributes: &authorizationapi.ResourceAttributes{
						FieldSelector: &authorizationapi.FieldSelectorAttributes{
							RawSelector: "&foo",
						},
					},
				},
			},
			want: authorizer.AttributesRecord{
				User:                    &user.DefaultInfo{},
				ResourceRequest:         true,
				APIVersion:              "*",
				FieldSelectorParsingErr: errors.New("invalid selector: '&foo'; can't understand '&foo'"),
			},
			enableAuthorizationSelector: true,
		},
		{
			name: "field: requirements",
			args: args{
				spec: authorizationapi.SubjectAccessReviewSpec{
					ResourceAttributes: &authorizationapi.ResourceAttributes{
						FieldSelector: &authorizationapi.FieldSelectorAttributes{
							Requirements: []metav1.FieldSelectorRequirement{
								{
									Key:      "one",
									Operator: "In",
									Values:   []string{"apple"},
								},
								{
									Key:      "two",
									Operator: "NotIn",
									Values:   []string{"banana"},
								},
							},
						},
					},
				},
			},
			want: authorizer.AttributesRecord{
				User:            &user.DefaultInfo{},
				ResourceRequest: true,
				APIVersion:      "*",
				FieldSelectorRequirements: fields.Requirements{
					{Operator: "=", Field: "one", Value: "apple"},
					{Operator: "!=", Field: "two", Value: "banana"},
				},
			},
			enableAuthorizationSelector: true,
		},
		{
			name: "field: requirements too many values",
			args: args{
				spec: authorizationapi.SubjectAccessReviewSpec{
					ResourceAttributes: &authorizationapi.ResourceAttributes{
						FieldSelector: &authorizationapi.FieldSelectorAttributes{
							Requirements: []metav1.FieldSelectorRequirement{
								{
									Key:      "one",
									Operator: "In",
									Values:   []string{"apple", "other"},
								},
							},
						},
					},
				},
			},
			want: authorizer.AttributesRecord{
				User:                    &user.DefaultInfo{},
				ResourceRequest:         true,
				APIVersion:              "*",
				FieldSelectorParsingErr: utilerrors.NewAggregate([]error{errors.New("fieldSelectors do not yet support multiple values")}),
			},
			enableAuthorizationSelector: true,
		},
		{
			name: "field: requirements missing in value",
			args: args{
				spec: authorizationapi.SubjectAccessReviewSpec{
					ResourceAttributes: &authorizationapi.ResourceAttributes{
						FieldSelector: &authorizationapi.FieldSelectorAttributes{
							Requirements: []metav1.FieldSelectorRequirement{
								{
									Key:      "one",
									Operator: "In",
									Values:   []string{},
								},
							},
						},
					},
				},
			},
			want: authorizer.AttributesRecord{
				User:                    &user.DefaultInfo{},
				ResourceRequest:         true,
				APIVersion:              "*",
				FieldSelectorParsingErr: utilerrors.NewAggregate([]error{errors.New("fieldSelectors in must have one value")}),
			},
			enableAuthorizationSelector: true,
		},
		{
			name: "field: requirements missing notin value",
			args: args{
				spec: authorizationapi.SubjectAccessReviewSpec{
					ResourceAttributes: &authorizationapi.ResourceAttributes{
						FieldSelector: &authorizationapi.FieldSelectorAttributes{
							Requirements: []metav1.FieldSelectorRequirement{
								{
									Key:      "one",
									Operator: "NotIn",
									Values:   []string{},
								},
							},
						},
					},
				},
			},
			want: authorizer.AttributesRecord{
				User:                    &user.DefaultInfo{},
				ResourceRequest:         true,
				APIVersion:              "*",
				FieldSelectorParsingErr: utilerrors.NewAggregate([]error{errors.New("fieldSelectors not in must have one value")}),
			},
			enableAuthorizationSelector: true,
		},
		{
			name: "field: requirements exists",
			args: args{
				spec: authorizationapi.SubjectAccessReviewSpec{
					ResourceAttributes: &authorizationapi.ResourceAttributes{
						FieldSelector: &authorizationapi.FieldSelectorAttributes{
							Requirements: []metav1.FieldSelectorRequirement{
								{
									Key:      "one",
									Operator: "Exists",
									Values:   []string{},
								},
							},
						},
					},
				},
			},
			want: authorizer.AttributesRecord{
				User:                    &user.DefaultInfo{},
				ResourceRequest:         true,
				APIVersion:              "*",
				FieldSelectorParsingErr: utilerrors.NewAggregate([]error{errors.New("fieldSelectors do not yet support Exists")}),
			},
			enableAuthorizationSelector: true,
		},
		{
			name: "field: requirements DoesNotExist",
			args: args{
				spec: authorizationapi.SubjectAccessReviewSpec{
					ResourceAttributes: &authorizationapi.ResourceAttributes{
						FieldSelector: &authorizationapi.FieldSelectorAttributes{
							Requirements: []metav1.FieldSelectorRequirement{
								{
									Key:      "one",
									Operator: "DoesNotExist",
									Values:   []string{},
								},
							},
						},
					},
				},
			},
			want: authorizer.AttributesRecord{
				User:                    &user.DefaultInfo{},
				ResourceRequest:         true,
				APIVersion:              "*",
				FieldSelectorParsingErr: utilerrors.NewAggregate([]error{errors.New("fieldSelectors do not yet support DoesNotExist")}),
			},
			enableAuthorizationSelector: true,
		},
		{
			name: "field: requirements bad operator",
			args: args{
				spec: authorizationapi.SubjectAccessReviewSpec{
					ResourceAttributes: &authorizationapi.ResourceAttributes{
						FieldSelector: &authorizationapi.FieldSelectorAttributes{
							Requirements: []metav1.FieldSelectorRequirement{
								{
									Key:      "one",
									Operator: "bad",
									Values:   []string{},
								},
							},
						},
					},
				},
			},
			want: authorizer.AttributesRecord{
				User:                    &user.DefaultInfo{},
				ResourceRequest:         true,
				APIVersion:              "*",
				FieldSelectorParsingErr: utilerrors.NewAggregate([]error{errors.New("\"bad\" is not a valid field selector operator")}),
			},
			enableAuthorizationSelector: true,
		},

		{
			name: "label: ignore when featuregate off",
			args: args{
				spec: authorizationapi.SubjectAccessReviewSpec{
					ResourceAttributes: &authorizationapi.ResourceAttributes{
						LabelSelector: &authorizationapi.LabelSelectorAttributes{
							RawSelector: "foo=bar",
						},
					},
				},
			},
			want: authorizer.AttributesRecord{
				User:            &user.DefaultInfo{},
				ResourceRequest: true,
				APIVersion:      "*",
			},
		},
		{
			name: "label: raw selector",
			args: args{
				spec: authorizationapi.SubjectAccessReviewSpec{
					ResourceAttributes: &authorizationapi.ResourceAttributes{
						LabelSelector: &authorizationapi.LabelSelectorAttributes{
							RawSelector: "foo=bar",
						},
					},
				},
			},
			want: authorizer.AttributesRecord{
				User:            &user.DefaultInfo{},
				ResourceRequest: true,
				APIVersion:      "*",
				LabelSelectorRequirements: labels.Requirements{
					mustRequirement("foo", "=", []string{"bar"}),
				},
			},
			enableAuthorizationSelector: true,
		},
		{
			name: "label: raw selector error",
			args: args{
				spec: authorizationapi.SubjectAccessReviewSpec{
					ResourceAttributes: &authorizationapi.ResourceAttributes{
						LabelSelector: &authorizationapi.LabelSelectorAttributes{
							RawSelector: "&foo",
						},
					},
				},
			},
			want: authorizer.AttributesRecord{
				User:                    &user.DefaultInfo{},
				ResourceRequest:         true,
				APIVersion:              "*",
				LabelSelectorParsingErr: errors.New("unable to parse requirement: <nil>: Invalid value: \"&foo\": name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')"),
			},
			enableAuthorizationSelector: true,
		},
		{
			name: "label: requirements",
			args: args{
				spec: authorizationapi.SubjectAccessReviewSpec{
					ResourceAttributes: &authorizationapi.ResourceAttributes{
						LabelSelector: &authorizationapi.LabelSelectorAttributes{
							Requirements: []metav1.LabelSelectorRequirement{
								{
									Key:      "one",
									Operator: "In",
									Values:   []string{"apple"},
								},
								{
									Key:      "two",
									Operator: "NotIn",
									Values:   []string{"banana"},
								},
							},
						},
					},
				},
			},
			want: authorizer.AttributesRecord{
				User:            &user.DefaultInfo{},
				ResourceRequest: true,
				APIVersion:      "*",
				LabelSelectorRequirements: labels.Requirements{
					mustRequirement("one", "in", []string{"apple"}),
					mustRequirement("two", "notin", []string{"banana"}),
				},
			},
			enableAuthorizationSelector: true,
		},
		{
			name: "label: requirements multiple values",
			args: args{
				spec: authorizationapi.SubjectAccessReviewSpec{
					ResourceAttributes: &authorizationapi.ResourceAttributes{
						LabelSelector: &authorizationapi.LabelSelectorAttributes{
							Requirements: []metav1.LabelSelectorRequirement{
								{
									Key:      "one",
									Operator: "In",
									Values:   []string{"apple", "other"},
								},
								{
									Key:      "two",
									Operator: "NotIn",
									Values:   []string{"carrot", "donut"},
								},
							},
						},
					},
				},
			},
			want: authorizer.AttributesRecord{
				User:            &user.DefaultInfo{},
				ResourceRequest: true,
				APIVersion:      "*",
				LabelSelectorRequirements: labels.Requirements{
					mustRequirement("one", "in", []string{"apple", "other"}),
					mustRequirement("two", "notin", []string{"carrot", "donut"}),
				},
			},
			enableAuthorizationSelector: true,
		},
		{
			name: "label: requirements exists",
			args: args{
				spec: authorizationapi.SubjectAccessReviewSpec{
					ResourceAttributes: &authorizationapi.ResourceAttributes{
						LabelSelector: &authorizationapi.LabelSelectorAttributes{
							Requirements: []metav1.LabelSelectorRequirement{
								{
									Key:      "one",
									Operator: "Exists",
									Values:   []string{},
								},
							},
						},
					},
				},
			},
			want: authorizer.AttributesRecord{
				User:            &user.DefaultInfo{},
				ResourceRequest: true,
				APIVersion:      "*",
				LabelSelectorRequirements: labels.Requirements{
					mustRequirement("one", "exists", nil),
				},
			},
			enableAuthorizationSelector: true,
		},
		{
			name: "label: requirements DoesNotExist",
			args: args{
				spec: authorizationapi.SubjectAccessReviewSpec{
					ResourceAttributes: &authorizationapi.ResourceAttributes{
						LabelSelector: &authorizationapi.LabelSelectorAttributes{
							Requirements: []metav1.LabelSelectorRequirement{
								{
									Key:      "one",
									Operator: "DoesNotExist",
									Values:   []string{},
								},
							},
						},
					},
				},
			},
			want: authorizer.AttributesRecord{
				User:            &user.DefaultInfo{},
				ResourceRequest: true,
				APIVersion:      "*",
				LabelSelectorRequirements: labels.Requirements{
					mustRequirement("one", "!", nil),
				},
			},
			enableAuthorizationSelector: true,
		},
		{
			name: "label: requirements bad operator",
			args: args{
				spec: authorizationapi.SubjectAccessReviewSpec{
					ResourceAttributes: &authorizationapi.ResourceAttributes{
						LabelSelector: &authorizationapi.LabelSelectorAttributes{
							Requirements: []metav1.LabelSelectorRequirement{
								{
									Key:      "one",
									Operator: "bad",
									Values:   []string{},
								},
							},
						},
					},
				},
			},
			want: authorizer.AttributesRecord{
				User:                    &user.DefaultInfo{},
				ResourceRequest:         true,
				APIVersion:              "*",
				LabelSelectorParsingErr: utilerrors.NewAggregate([]error{errors.New("\"bad\" is not a valid label selector operator")}),
			},
			enableAuthorizationSelector: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if !tt.enableAuthorizationSelector {
				featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse("1.33"))
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.AuthorizeWithSelectors, false)
			}

			if got := AuthorizationAttributesFrom(tt.args.spec); !reflect.DeepEqual(got, tt.want) {
				if got.LabelSelectorParsingErr != nil {
					t.Logf("labelSelectorErr=%q", got.LabelSelectorParsingErr)
				}
				t.Errorf("AuthorizationAttributesFrom(), got:\n%#v\nwant:\n%#v", got, tt.want)
			}
		})
	}
}

// sarMockAuthorizer implements authorizer.Authorizer for testing SARStatusFromAuthorize.
type sarMockAuthorizer struct {
	// makeDecision is called after the feature gate is set, to construct the decision for the mock.
	// Required to be set, otherwise the authorizer panics.
	// Serves both the conditions-aware and conditions-unaware endpoint.
	// If one tries to return conditions to an unaware endpoint, it fails closed.
	makeDecision func() authorizer.ConditionsAwareDecision
}

func (m *sarMockAuthorizer) Authorize(ctx context.Context, a authorizer.Attributes) (authorizer.Decision, string, error) {
	return m.ConditionsAwareAuthorize(ctx, a).UnconditionalParts()
}

func (m *sarMockAuthorizer) ConditionsAwareAuthorize(ctx context.Context, a authorizer.Attributes) authorizer.ConditionsAwareDecision {
	return m.makeDecision()
}

func (m *sarMockAuthorizer) EvaluateConditions(ctx context.Context, decision authorizer.ConditionsAwareDecision, data authorizer.ConditionsData) (authorizer.Decision, string, error) {
	return authorizer.DecisionDeny, "not implemented", authorizer.ErrorConditionEvaluationNotSupported
}

func TestSARStatusFromAuthorize(t *testing.T) {
	// Helper to construct a conditions map decision. Must be called with the ConditionalAuthorization gate enabled.
	makeCondAllowDecision := func() authorizer.ConditionsAwareDecision {
		return authorizer.ConditionsAwareDecisionConditionsMap(
			authorizer.GenericCondition{ID: "cond1", Condition: "object.metadata.name == 'foo'", Effect: authorizer.ConditionEffectAllow, Type: "cel", Description: "allow foo"},
		)
	}

	makeCondDenyDecision := func() authorizer.ConditionsAwareDecision {
		return authorizer.ConditionsAwareDecisionConditionsMap(
			authorizer.GenericCondition{ID: "cond1", Condition: "object.metadata.name == 'foo'", Effect: authorizer.ConditionEffectDeny, Type: "cel", Description: "deny foo"},
		)
	}

	// Helper to construct a union decision. Must be called with the ConditionalAuthorization gate enabled.
	makeUnionDecision := func() authorizer.ConditionsAwareDecision {
		return authorizer.ConditionsAwareDecisionUnion(
			makeCondDenyDecision(),
			authorizer.ConditionsAwareDecisionNoOpinion("no-opinion-reason", fmt.Errorf("no-opinion-err")),
			authorizer.ConditionsAwareDecisionUnion(
				authorizer.ConditionsAwareDecisionNoOpinion("", nil),
				makeCondAllowDecision(),
			),
			authorizer.ConditionsAwareDecisionDeny("", nil),
		)
	}

	tests := []struct {
		name         string
		makeDecision func() authorizer.ConditionsAwareDecision
		modes        map[string]authorizationapi.SubjectAccessReviewStatus
	}{
		{
			name: "unconditional allow with evaluation error",
			makeDecision: func() authorizer.ConditionsAwareDecision {
				return authorizer.ConditionsAwareDecisionAllow("RBAC: allowed", fmt.Errorf("partial error"))
			},
			modes: map[string]authorizationapi.SubjectAccessReviewStatus{
				"featureOffClientOptOut": {
					Allowed:         true,
					Reason:          "RBAC: allowed",
					EvaluationError: "partial error",
				},
				"featureOffClientOptIn": {
					Allowed:         true,
					Reason:          "RBAC: allowed",
					EvaluationError: "partial error",
				},
				"featureOnClientOptOut": {
					Allowed:         true,
					Reason:          "RBAC: allowed",
					EvaluationError: "partial error",
				},
				"featureOnClientOptIn": {
					Allowed:         true,
					Reason:          "RBAC: allowed",
					EvaluationError: "partial error",
				},
			},
		},
		{
			name: "unconditional deny",
			makeDecision: func() authorizer.ConditionsAwareDecision {
				return authorizer.ConditionsAwareDecisionDeny("Node: denied", nil)
			},
			modes: map[string]authorizationapi.SubjectAccessReviewStatus{
				"featureOffClientOptOut": {
					Denied: true,
					Reason: "Node: denied",
				},
				"featureOffClientOptIn": {
					Denied: true,
					Reason: "Node: denied",
				},
				"featureOnClientOptOut": {
					Denied: true,
					Reason: "Node: denied",
				},
				"featureOnClientOptIn": {
					Denied: true,
					Reason: "Node: denied",
				},
			},
		},
		{
			name: "no opinion",
			makeDecision: func() authorizer.ConditionsAwareDecision {
				return authorizer.ConditionsAwareDecisionNoOpinion("no rules matched", nil)
			},
			modes: map[string]authorizationapi.SubjectAccessReviewStatus{
				"featureOffClientOptOut": {
					Reason: "no rules matched",
				},
				"featureOffClientOptIn": {
					Reason: "no rules matched",
				},
				"featureOnClientOptOut": {
					Reason: "no rules matched",
				},
				"featureOnClientOptIn": {
					Reason: "no rules matched",
				},
			},
		},
		{
			name:         "conditional allow",
			makeDecision: makeCondAllowDecision,
			modes: map[string]authorizationapi.SubjectAccessReviewStatus{
				"featureOffClientOptOut": {
					// NoOpinion
					Reason: "authorizer tried to return conditional decision, but the ConditionalAuthorization feature gate is disabled",
				},
				"featureOffClientOptIn": {
					// NoOpinion
					Reason: "authorizer tried to return conditional decision, but the ConditionalAuthorization feature gate is disabled",
				},
				"featureOnClientOptOut": {
					// NoOpinion
					Reason: "failed closed: tried to return conditional decision to conditions-unaware authorizer",
				},
				"featureOnClientOptIn": {
					ConditionalDecision: &authorizationapi.ConditionsAwareDecision{
						Type: authorizationapi.ConditionsAwareDecisionTypeConditionsMap,
						ConditionsMap: &authorizationapi.ConditionsMap{
							Conditions: []authorizationapi.Condition{
								{
									ID:          "cond1",
									Effect:      authorizationapi.ConditionEffectAllow,
									Condition:   "object.metadata.name == 'foo'",
									Type:        "cel",
									Description: "allow foo",
								},
							},
						},
					},
				},
			},
		},
		{
			name:         "conditional deny",
			makeDecision: makeCondDenyDecision,
			modes: map[string]authorizationapi.SubjectAccessReviewStatus{
				"featureOffClientOptOut": {
					Denied: true, // As the authorizer tried to construct a conditional deny, it fails closed as Deny
					Reason: "authorizer tried to return conditional decision, but the ConditionalAuthorization feature gate is disabled",
				},
				"featureOffClientOptIn": {
					Denied: true, // As the authorizer tried to construct a conditional deny, it fails closed as Deny
					Reason: "authorizer tried to return conditional decision, but the ConditionalAuthorization feature gate is disabled",
				},
				"featureOnClientOptOut": {
					Denied: true, // As the authorizer tried to construct a conditional deny, it fails closed as Deny
					Reason: "failed closed: tried to return conditional decision to conditions-unaware authorizer",
				},
				"featureOnClientOptIn": {
					ConditionalDecision: &authorizationapi.ConditionsAwareDecision{
						Type: authorizationapi.ConditionsAwareDecisionTypeConditionsMap,
						ConditionsMap: &authorizationapi.ConditionsMap{
							Conditions: []authorizationapi.Condition{
								{
									ID:          "cond1",
									Effect:      authorizationapi.ConditionEffectDeny,
									Condition:   "object.metadata.name == 'foo'",
									Type:        "cel",
									Description: "deny foo",
								},
							},
						},
					},
				},
			},
		},
		{
			name:         "union",
			makeDecision: makeUnionDecision,
			modes: map[string]authorizationapi.SubjectAccessReviewStatus{
				"featureOffClientOptOut": {
					Denied: true, // As the authorizer tried to construct a conditional deny, it fails closed as Deny
					Reason: "authorizer tried to return conditional decision, but the ConditionalAuthorization feature gate is disabled",
				},
				"featureOffClientOptIn": {
					Denied: true, // As the authorizer tried to construct a conditional deny, it fails closed as Deny
					Reason: "authorizer tried to return conditional decision, but the ConditionalAuthorization feature gate is disabled",
				},
				"featureOnClientOptOut": {
					Denied: true, // As the authorizer tried to construct a conditional deny, it fails closed as Deny
					Reason: "failed closed: tried to return conditional decision to conditions-unaware authorizer",
				},
				"featureOnClientOptIn": {
					ConditionalDecision: &authorizationapi.ConditionsAwareDecision{
						Type: authorizationapi.ConditionsAwareDecisionTypeUnion,
						Union: []authorizationapi.ConditionsAwareDecision{
							{
								Type: authorizationapi.ConditionsAwareDecisionTypeConditionsMap,
								ConditionsMap: &authorizationapi.ConditionsMap{
									Conditions: []authorizationapi.Condition{
										{
											ID:          "cond1",
											Effect:      authorizationapi.ConditionEffectDeny,
											Condition:   "object.metadata.name == 'foo'",
											Type:        "cel",
											Description: "deny foo",
										},
									},
								},
							},
							{
								Type:            authorizationapi.ConditionsAwareDecisionTypeNoOpinion,
								Reason:          "no-opinion-reason",
								EvaluationError: "no-opinion-err",
							},
							{
								Type: authorizationapi.ConditionsAwareDecisionTypeUnion,
								Union: []authorizationapi.ConditionsAwareDecision{
									{
										Type: authorizationapi.ConditionsAwareDecisionTypeNoOpinion,
									},
									{
										Type: authorizationapi.ConditionsAwareDecisionTypeConditionsMap,
										ConditionsMap: &authorizationapi.ConditionsMap{
											Conditions: []authorizationapi.Condition{
												{
													ID:          "cond1",
													Effect:      authorizationapi.ConditionEffectAllow,
													Condition:   "object.metadata.name == 'foo'",
													Type:        "cel",
													Description: "allow foo",
												},
											},
										},
									},
								},
							},
							{
								Type: authorizationapi.ConditionsAwareDecisionTypeDeny,
							},
						},
					},
				},
			},
		},
	}

	modesPerTestCase := map[string]func(t *testing.T) *authorizationapi.ConditionalAuthorizationOptions{
		"featureOffClientOptOut": func(t *testing.T) *authorizationapi.ConditionalAuthorizationOptions {
			return nil
		},
		"featureOffClientOptIn": func(t *testing.T) *authorizationapi.ConditionalAuthorizationOptions {
			return &authorizationapi.ConditionalAuthorizationOptions{
				Enabled: true,
			}
		},
		"featureOnClientOptOut": func(t *testing.T) *authorizationapi.ConditionalAuthorizationOptions {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ConditionalAuthorization, true)
			return &authorizationapi.ConditionalAuthorizationOptions{
				Enabled: false,
			}
		},
		"featureOnClientOptIn": func(t *testing.T) *authorizationapi.ConditionalAuthorizationOptions {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ConditionalAuthorization, true)
			return &authorizationapi.ConditionalAuthorizationOptions{
				Enabled: true,
			}
		},
	}

	for _, tt := range tests {
		for modeName, modeSetupFunc := range modesPerTestCase {
			t.Run(fmt.Sprintf("%s/%s", tt.name, modeName), func(t *testing.T) {
				conditionalOpts := modeSetupFunc(t)
				authz := &sarMockAuthorizer{
					makeDecision: tt.makeDecision,
				}

				attrs := authorizer.AttributesRecord{
					User: &user.DefaultInfo{
						Name: "foo",
					},
					Verb:     "create",
					Resource: "pods",
				}

				got := SARStatusFromAuthorize(t.Context(), authz, attrs, conditionalOpts)
				want := tt.modes[modeName]
				if diff := cmp.Diff(want, got); diff != "" {
					t.Errorf("Found diff=%s", diff)
				}
			})
		}
	}
}
