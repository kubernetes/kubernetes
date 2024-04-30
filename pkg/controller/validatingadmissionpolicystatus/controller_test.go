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

package validatingadmissionpolicystatus

import (
	"context"
	"strings"
	"testing"
	"time"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	"k8s.io/apimachinery/pkg/api/meta/testrestmapper"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	validatingadmissionpolicy "k8s.io/apiserver/pkg/admission/plugin/policy/validating"
	"k8s.io/apiserver/pkg/cel/openapi/resolver"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/kubernetes/pkg/generated/openapi"
)

func TestTypeChecking(t *testing.T) {
	for _, tc := range []struct {
		name           string
		policy         *admissionregistrationv1.ValidatingAdmissionPolicy
		assertFieldRef func(warnings []admissionregistrationv1.ExpressionWarning, t *testing.T) // warning.fieldRef
		assertWarnings func(warnings []admissionregistrationv1.ExpressionWarning, t *testing.T) // warning.warning
	}{
		{
			name: "deployment with correct expression",
			policy: withGVRMatch([]string{"apps"}, []string{"v1"}, []string{"deployments"}, withValidations([]admissionregistrationv1.Validation{
				{
					Expression: "object.spec.replicas > 1",
				},
			}, makePolicy("replicated-deployment"))),
			assertFieldRef: toHaveLengthOf(0),
			assertWarnings: toHaveLengthOf(0),
		},
		{
			name: "deployment with type confusion",
			policy: withGVRMatch([]string{"apps"}, []string{"v1"}, []string{"deployments"}, withValidations([]admissionregistrationv1.Validation{
				{
					Expression: "object.spec.replicas < 100", // this one passes
				},
				{
					Expression: "object.spec.replicas > '1'", // '1' should be int
				},
			}, makePolicy("confused-deployment"))),
			assertFieldRef: toBe("spec.validations[1].expression"),
			assertWarnings: toHaveSubstring(`found no matching overload for '_>_' applied to '(int, string)'`),
		},
		{
			name: "two expressions different type checking errors",
			policy: withGVRMatch([]string{"apps"}, []string{"v1"}, []string{"deployments"}, withValidations([]admissionregistrationv1.Validation{
				{
					Expression: "object.spec.nonExistingFirst > 1",
				},
				{
					Expression: "object.spec.replicas > '1'", // '1' should be int
				},
			}, makePolicy("confused-deployment"))),
			assertFieldRef: toBe("spec.validations[0].expression", "spec.validations[1].expression"),
			assertWarnings: toHaveSubstring(
				"undefined field 'nonExistingFirst'",
				`found no matching overload for '_>_' applied to '(int, string)'`,
			),
		},
		{
			name: "one expression, two warnings",
			policy: withGVRMatch([]string{"apps"}, []string{"v1"}, []string{"deployments"}, withValidations([]admissionregistrationv1.Validation{
				{
					Expression: "object.spec.replicas < 100", // this one passes
				},
				{
					Expression: "object.spec.replicas > '1' && object.spec.nonExisting == 1",
				},
			}, makePolicy("confused-deployment"))),
			assertFieldRef: toBe("spec.validations[1].expression"),
			assertWarnings: toHaveMultipleSubstrings([]string{"undefined field 'nonExisting'", `found no matching overload for '_>_' applied to '(int, string)'`}),
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
			defer cancel()
			policy := tc.policy.DeepCopy()
			policy.ObjectMeta.Generation = 1 // fake storage does not do this automatically
			client := fake.NewSimpleClientset(policy)
			informerFactory := informers.NewSharedInformerFactory(client, 0)
			typeChecker := &validatingadmissionpolicy.TypeChecker{
				SchemaResolver: resolver.NewDefinitionsSchemaResolver(openapi.GetOpenAPIDefinitions, scheme.Scheme),
				RestMapper:     testrestmapper.TestOnlyStaticRESTMapper(scheme.Scheme),
			}
			controller, err := NewController(
				informerFactory.Admissionregistration().V1().ValidatingAdmissionPolicies(),
				client.AdmissionregistrationV1().ValidatingAdmissionPolicies(),
				typeChecker,
			)
			if err != nil {
				t.Fatalf("cannot create controller: %v", err)
			}
			go informerFactory.Start(ctx.Done())
			go controller.Run(ctx, 1)
			err = wait.PollUntilContextCancel(ctx, time.Second, false, func(ctx context.Context) (done bool, err error) {
				name := policy.Name
				// wait until the typeChecking is set, which means the type checking
				// is complete.
				updated, err := client.AdmissionregistrationV1().ValidatingAdmissionPolicies().Get(ctx, name, metav1.GetOptions{})
				if err != nil {
					return false, err
				}
				if updated.Status.TypeChecking != nil {
					policy = updated
					return true, nil
				}
				return false, nil
			})
			if err != nil {
				t.Fatal(err)
			}
			tc.assertFieldRef(policy.Status.TypeChecking.ExpressionWarnings, t)
			tc.assertWarnings(policy.Status.TypeChecking.ExpressionWarnings, t)
			if err != nil {
				t.Fatalf("failed to initialize controller: %v", err)
			}
		})
	}

}

func toBe(expected ...string) func(warnings []admissionregistrationv1.ExpressionWarning, t *testing.T) {
	return func(warnings []admissionregistrationv1.ExpressionWarning, t *testing.T) {
		if len(expected) != len(warnings) {
			t.Fatalf("mismatched length, expect %d, got %d", len(expected), len(warnings))
		}
		for i := range expected {
			if expected[i] != warnings[i].FieldRef {
				t.Errorf("expected %q but got %q", expected[i], warnings[i].FieldRef)
			}
		}
	}
}

func toHaveSubstring(substrings ...string) func(warnings []admissionregistrationv1.ExpressionWarning, t *testing.T) {
	return func(warnings []admissionregistrationv1.ExpressionWarning, t *testing.T) {
		if len(substrings) != len(warnings) {
			t.Fatalf("mismatched length, expect %d, got %d", len(substrings), len(warnings))
		}
		for i := range substrings {
			if !strings.Contains(warnings[i].Warning, substrings[i]) {
				t.Errorf("missing expected substring %q in %v", substrings[i], warnings[i])
			}
		}
	}
}

func toHaveMultipleSubstrings(substrings ...[]string) func(warnings []admissionregistrationv1.ExpressionWarning, t *testing.T) {
	return func(warnings []admissionregistrationv1.ExpressionWarning, t *testing.T) {
		if len(substrings) != len(warnings) {
			t.Fatalf("mismatched length, expect %d, got %d", len(substrings), len(warnings))
		}
		for i, expectedSubstrings := range substrings {
			for _, s := range expectedSubstrings {
				if !strings.Contains(warnings[i].Warning, s) {
					t.Errorf("missing expected substring %q in %v", substrings[i], warnings[i])
				}
			}
		}
	}
}

func toHaveLengthOf(n int) func(warnings []admissionregistrationv1.ExpressionWarning, t *testing.T) {
	return func(warnings []admissionregistrationv1.ExpressionWarning, t *testing.T) {
		if n != len(warnings) {
			t.Fatalf("mismatched length, expect %d, got %d", n, len(warnings))
		}
	}
}

func withGVRMatch(groups []string, versions []string, resources []string, policy *admissionregistrationv1.ValidatingAdmissionPolicy) *admissionregistrationv1.ValidatingAdmissionPolicy {
	policy.Spec.MatchConstraints = &admissionregistrationv1.MatchResources{
		ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{
			{
				RuleWithOperations: admissionregistrationv1.RuleWithOperations{
					Operations: []admissionregistrationv1.OperationType{
						"*",
					},
					Rule: admissionregistrationv1.Rule{
						APIGroups:   groups,
						APIVersions: versions,
						Resources:   resources,
					},
				},
			},
		},
	}
	return policy
}

func withValidations(validations []admissionregistrationv1.Validation, policy *admissionregistrationv1.ValidatingAdmissionPolicy) *admissionregistrationv1.ValidatingAdmissionPolicy {
	policy.Spec.Validations = validations
	return policy
}

func makePolicy(name string) *admissionregistrationv1.ValidatingAdmissionPolicy {
	return &admissionregistrationv1.ValidatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{Name: name},
	}
}
