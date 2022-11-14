/*
Copyright 2022 The Kubernetes Authors.

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

package cel

import (
	"context"
	"strings"
	"testing"
	"time"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionsclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/etcd"
	"k8s.io/kubernetes/test/integration/framework"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"

	v1 "k8s.io/api/core/v1"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	admissionregistrationv1alpha1 "k8s.io/api/admissionregistration/v1alpha1"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
)

// Test_ValidateNamespace_NoParams tests a ValidatingAdmissionPolicy that validates creation of a Namespace with no params.
func Test_ValidateNamespace_NoParams(t *testing.T) {
	forbiddenReason := metav1.StatusReasonForbidden

	testcases := []struct {
		name          string
		policy        *admissionregistrationv1alpha1.ValidatingAdmissionPolicy
		policyBinding *admissionregistrationv1alpha1.ValidatingAdmissionPolicyBinding
		namespace     *v1.Namespace
		err           string
		failureReason metav1.StatusReason
	}{
		{
			name: "namespace name contains suffix enforced by validating admission policy, using object metadata fields",
			policy: withValidations([]admissionregistrationv1alpha1.Validation{
				{
					Expression: "object.metadata.name.endsWith('k8s')",
				},
			}, withFailurePolicy(admissionregistrationv1alpha1.Fail, withNamespaceMatch(makePolicy("validate-namespace-suffix")))),
			policyBinding: makeBinding("validate-namespace-suffix-binding", "validate-namespace-suffix", ""),
			namespace: &v1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-k8s",
				},
			},
			err: "",
		},
		{
			name: "namespace name does NOT contain suffix enforced by validating admission policyusing, object metadata fields",
			policy: withValidations([]admissionregistrationv1alpha1.Validation{
				{
					Expression: "object.metadata.name.endsWith('k8s')",
				},
			}, withFailurePolicy(admissionregistrationv1alpha1.Fail, withNamespaceMatch(makePolicy("validate-namespace-suffix")))),
			policyBinding: makeBinding("validate-namespace-suffix-binding", "validate-namespace-suffix", ""),
			namespace: &v1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-foobar",
				},
			},
			err:           "namespaces \"test-foobar\" is forbidden: ValidatingAdmissionPolicy 'validate-namespace-suffix' with binding 'validate-namespace-suffix-binding' denied request: failed expression: object.metadata.name.endsWith('k8s')",
			failureReason: metav1.StatusReasonInvalid,
		},
		{
			name: "namespace name does NOT contain suffix enforced by validating admission policy using object metadata fields, AND validating expression returns StatusReasonForbidden",
			policy: withValidations([]admissionregistrationv1alpha1.Validation{
				{
					Expression: "object.metadata.name.endsWith('k8s')",
					Reason:     &forbiddenReason,
				},
			}, withFailurePolicy(admissionregistrationv1alpha1.Fail, withNamespaceMatch(makePolicy("validate-namespace-suffix")))),
			policyBinding: makeBinding("validate-namespace-suffix-binding", "validate-namespace-suffix", ""),
			namespace: &v1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: "forbidden-test-foobar",
				},
			},
			err:           "namespaces \"forbidden-test-foobar\" is forbidden: ValidatingAdmissionPolicy 'validate-namespace-suffix' with binding 'validate-namespace-suffix-binding' denied request: failed expression: object.metadata.name.endsWith('k8s')",
			failureReason: metav1.StatusReasonForbidden,
		},
		{
			name: "namespace name contains suffix enforced by validating admission policy, using request field",
			policy: withValidations([]admissionregistrationv1alpha1.Validation{
				{
					Expression: "request.name.endsWith('k8s')",
				},
			}, withFailurePolicy(admissionregistrationv1alpha1.Fail, withNamespaceMatch(makePolicy("validate-namespace-suffix")))),
			policyBinding: makeBinding("validate-namespace-suffix-binding", "validate-namespace-suffix", ""),
			namespace: &v1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-k8s",
				},
			},
			err: "",
		},
		{
			name: "namespace name does NOT contains suffix enforced by validating admission policy, using request field",
			policy: withValidations([]admissionregistrationv1alpha1.Validation{
				{
					Expression: "request.name.endsWith('k8s')",
				},
			}, withFailurePolicy(admissionregistrationv1alpha1.Fail, withNamespaceMatch(makePolicy("validate-namespace-suffix")))),
			policyBinding: makeBinding("validate-namespace-suffix-binding", "validate-namespace-suffix", ""),
			namespace: &v1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-k8s",
				},
			},
			err: "",
		},
		{
			name: "runtime error when validating namespace, but failurePolicy=Ignore",
			policy: withValidations([]admissionregistrationv1alpha1.Validation{
				{
					Expression: "object.nonExistentProperty == 'someval'",
				},
			}, withFailurePolicy(admissionregistrationv1alpha1.Ignore, withNamespaceMatch(makePolicy("validate-namespace-suffix")))),
			policyBinding: makeBinding("validate-namespace-suffix-binding", "validate-namespace-suffix", ""),
			namespace: &v1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-k8s",
				},
			},
			err: "",
		},
		{
			name: "runtime error when validating namespace, but failurePolicy=Fail",
			policy: withValidations([]admissionregistrationv1alpha1.Validation{
				{
					Expression: "object.nonExistentProperty == 'someval'",
				},
			}, withFailurePolicy(admissionregistrationv1alpha1.Fail, withNamespaceMatch(makePolicy("validate-namespace-suffix")))),
			policyBinding: makeBinding("validate-namespace-suffix-binding", "validate-namespace-suffix", ""),
			namespace: &v1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-k8s",
				},
			},
			err:           "namespaces \"test-k8s\" is forbidden: ValidatingAdmissionPolicy 'validate-namespace-suffix' with binding 'validate-namespace-suffix-binding' denied request: expression 'object.nonExistentProperty == 'someval'' resulted in error: no such key: nonExistentProperty",
			failureReason: metav1.StatusReasonInvalid,
		},
		{
			name: "runtime error due to unguarded params",
			policy: withValidations([]admissionregistrationv1alpha1.Validation{
				{
					Expression: "object.metadata.name.startsWith(params.metadata.name)",
				},
			}, withParams(configParamKind(), withFailurePolicy(admissionregistrationv1alpha1.Fail, withNamespaceMatch(makePolicy("validate-namespace-suffix"))))),
			policyBinding: makeBinding("validate-namespace-suffix-binding", "validate-namespace-suffix", ""),
			namespace: &v1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-k8s",
				},
			},
			err:           "namespaces \"test-k8s\" is forbidden: ValidatingAdmissionPolicy 'validate-namespace-suffix' with binding 'validate-namespace-suffix-binding' denied request: expression 'object.metadata.name.startsWith(params.metadata.name)' resulted in error: no such key: metadata",
			failureReason: metav1.StatusReasonInvalid,
		},
		{
			name: "with check against unguarded params using has()",
			policy: withValidations([]admissionregistrationv1alpha1.Validation{
				{
					Expression: "has(params.metadata) && has(params.metadata.name) && object.metadata.name.endsWith(params.metadata.name)",
				},
			}, withParams(configParamKind(), withFailurePolicy(admissionregistrationv1alpha1.Fail, withNamespaceMatch(makePolicy("validate-namespace-suffix"))))),
			policyBinding: makeBinding("validate-namespace-suffix-binding", "validate-namespace-suffix", ""),
			namespace: &v1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-k8s",
				},
			},
			err:           "namespaces \"test-k8s\" is forbidden: ValidatingAdmissionPolicy 'validate-namespace-suffix' with binding 'validate-namespace-suffix-binding' denied request: expression 'has(params.metadata) && has(params.metadata.name) && object.metadata.name.endsWith(params.metadata.name)' resulted in error: invalid type for field selection.",
			failureReason: metav1.StatusReasonInvalid,
		},
		{
			name: "with check against null params",
			policy: withValidations([]admissionregistrationv1alpha1.Validation{
				{
					Expression: "(params != null && object.metadata.name.endsWith(params.metadata.name))",
				},
			}, withParams(configParamKind(), withFailurePolicy(admissionregistrationv1alpha1.Fail, withNamespaceMatch(makePolicy("validate-namespace-suffix"))))),
			policyBinding: makeBinding("validate-namespace-suffix-binding", "validate-namespace-suffix", ""),
			namespace: &v1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-k8s",
				},
			},
			err:           "namespaces \"test-k8s\" is forbidden: ValidatingAdmissionPolicy 'validate-namespace-suffix' with binding 'validate-namespace-suffix-binding' denied request: failed expression: (params != null && object.metadata.name.endsWith(params.metadata.name))",
			failureReason: metav1.StatusReasonInvalid,
		},
		{
			name: "with check against unguarded params using has() and default check",
			policy: withValidations([]admissionregistrationv1alpha1.Validation{
				{
					Expression: "(has(params.metadata) && has(params.metadata.name) && object.metadata.name.startsWith(params.metadata.name)) || object.metadata.name.endsWith('k8s')",
				},
			}, withParams(configParamKind(), withFailurePolicy(admissionregistrationv1alpha1.Fail, withNamespaceMatch(makePolicy("validate-namespace-suffix"))))),
			policyBinding: makeBinding("validate-namespace-suffix-binding", "validate-namespace-suffix", ""),
			namespace: &v1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-k8s",
				},
			},
			err:           "",
			failureReason: metav1.StatusReasonInvalid,
		},
		{
			name: "with check against null params and default check",
			policy: withValidations([]admissionregistrationv1alpha1.Validation{
				{
					Expression: "(params != null && object.metadata.name.startsWith(params.metadata.name)) || object.metadata.name.endsWith('k8s')",
				},
			}, withParams(configParamKind(), withFailurePolicy(admissionregistrationv1alpha1.Fail, withNamespaceMatch(makePolicy("validate-namespace-suffix"))))),
			policyBinding: makeBinding("validate-namespace-suffix-binding", "validate-namespace-suffix", ""),
			namespace: &v1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-k8s",
				},
			},
			err:           "",
			failureReason: metav1.StatusReasonInvalid,
		},
	}

	for _, testcase := range testcases {
		t.Run(testcase.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ValidatingAdmissionPolicy, true)()
			server, err := apiservertesting.StartTestServer(t, nil, []string{
				"--enable-admission-plugins", "ValidatingAdmissionPolicy",
			}, framework.SharedEtcd())
			if err != nil {
				t.Fatal(err)
			}
			defer server.TearDownFn()

			config := server.ClientConfig

			client, err := clientset.NewForConfig(config)
			if err != nil {
				t.Fatal(err)
			}
			policy := withWaitReadyConstraintAndExpression(testcase.policy)
			if _, err := client.AdmissionregistrationV1alpha1().ValidatingAdmissionPolicies().Create(context.TODO(), policy, metav1.CreateOptions{}); err != nil {
				t.Fatal(err)
			}
			if err := createAndWaitReady(t, client, testcase.policyBinding, nil); err != nil {
				t.Fatal(err)
			}

			_, err = client.CoreV1().Namespaces().Create(context.TODO(), testcase.namespace, metav1.CreateOptions{})
			if err == nil && testcase.err == "" {
				return
			}

			if err == nil && testcase.err != "" {
				t.Logf("actual error: %v", err)
				t.Logf("expected error: %v", testcase.err)
				t.Fatal("got nil error but expected an error")
			}

			if err != nil && testcase.err == "" {
				t.Logf("actual error: %v", err)
				t.Logf("expected error: %v", testcase.err)
				t.Fatal("got error but expected none")
			}

			if err.Error() != testcase.err {
				t.Logf("actual validation error: %v", err)
				t.Logf("expected validation error: %v", testcase.err)
				t.Error("unexpected validation error")
			}

			checkFailureReason(t, err, testcase.failureReason)
		})
	}
}

// Test_ValidateNamespace_WithConfigMapParams tests a ValidatingAdmissionPolicy that validates creation of a Namespace,
// using ConfigMap as a param reference.
func Test_ValidateNamespace_WithConfigMapParams(t *testing.T) {
	testcases := []struct {
		name          string
		policy        *admissionregistrationv1alpha1.ValidatingAdmissionPolicy
		policyBinding *admissionregistrationv1alpha1.ValidatingAdmissionPolicyBinding
		configMap     *v1.ConfigMap
		namespace     *v1.Namespace
		err           string
		failureReason metav1.StatusReason
	}{
		{
			name: "namespace name contains suffix enforced by validating admission policy",
			policy: withValidations([]admissionregistrationv1alpha1.Validation{
				{
					Expression: "object.metadata.name.endsWith(params.data.namespaceSuffix)",
				},
			}, withFailurePolicy(admissionregistrationv1alpha1.Fail, withParams(configParamKind(), withNamespaceMatch(makePolicy("validate-namespace-suffix"))))),
			policyBinding: makeBinding("validate-namespace-suffix-binding", "validate-namespace-suffix", "validate-namespace-suffix-param"),
			configMap: makeConfigParams("validate-namespace-suffix-param", map[string]string{
				"namespaceSuffix": "k8s",
			}),
			namespace: &v1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-k8s",
				},
			},
			err: "",
		},
		{
			name: "namespace name does NOT contain suffix enforced by validating admission policy",
			policy: withValidations([]admissionregistrationv1alpha1.Validation{
				{
					Expression: "object.metadata.name.endsWith(params.data.namespaceSuffix)",
				},
			}, withFailurePolicy(admissionregistrationv1alpha1.Fail, withParams(configParamKind(), withNamespaceMatch(makePolicy("validate-namespace-suffix"))))),
			policyBinding: makeBinding("validate-namespace-suffix-binding", "validate-namespace-suffix", "validate-namespace-suffix-param"),
			configMap: makeConfigParams("validate-namespace-suffix-param", map[string]string{
				"namespaceSuffix": "k8s",
			}),
			namespace: &v1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-foo",
				},
			},
			err:           "namespaces \"test-foo\" is forbidden: ValidatingAdmissionPolicy 'validate-namespace-suffix' with binding 'validate-namespace-suffix-binding' denied request: failed expression: object.metadata.name.endsWith(params.data.namespaceSuffix)",
			failureReason: metav1.StatusReasonInvalid,
		},
	}

	for _, testcase := range testcases {
		t.Run(testcase.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ValidatingAdmissionPolicy, true)()
			server, err := apiservertesting.StartTestServer(t, nil, []string{
				"--enable-admission-plugins", "ValidatingAdmissionPolicy",
			}, framework.SharedEtcd())
			if err != nil {
				t.Fatal(err)
			}
			defer server.TearDownFn()

			config := server.ClientConfig

			client, err := clientset.NewForConfig(config)
			if err != nil {
				t.Fatal(err)
			}

			if _, err := client.CoreV1().ConfigMaps("default").Create(context.TODO(), testcase.configMap, metav1.CreateOptions{}); err != nil {
				t.Fatal(err)
			}

			policy := withWaitReadyConstraintAndExpression(testcase.policy)
			if _, err := client.AdmissionregistrationV1alpha1().ValidatingAdmissionPolicies().Create(context.TODO(), policy, metav1.CreateOptions{}); err != nil {
				t.Fatal(err)
			}
			if err := createAndWaitReady(t, client, testcase.policyBinding, nil); err != nil {
				t.Fatal(err)
			}

			_, err = client.CoreV1().Namespaces().Create(context.TODO(), testcase.namespace, metav1.CreateOptions{})
			if err == nil && testcase.err == "" {
				return
			}

			if err == nil && testcase.err != "" {
				t.Logf("actual error: %v", err)
				t.Logf("expected error: %v", testcase.err)
				t.Fatal("got nil error but expected an error")
			}

			if err != nil && testcase.err == "" {
				t.Logf("actual error: %v", err)
				t.Logf("expected error: %v", testcase.err)
				t.Fatal("got error but expected none")
			}

			if err.Error() != testcase.err {
				t.Logf("actual validation error: %v", err)
				t.Logf("expected validation error: %v", testcase.err)
				t.Error("unexpected validation error")
			}

			checkFailureReason(t, err, testcase.failureReason)
		})
	}
}

func TestMultiplePolicyBindings(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ValidatingAdmissionPolicy, true)()
	server, err := apiservertesting.StartTestServer(t, nil, nil, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	defer server.TearDownFn()

	config := server.ClientConfig

	client, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	paramKind := &admissionregistrationv1alpha1.ParamKind{
		APIVersion: "v1",
		Kind:       "ConfigMap",
	}
	policy := withPolicyExistsLabels([]string{"paramIdent"}, withParams(paramKind, withPolicyMatch("secrets", withFailurePolicy(admissionregistrationv1alpha1.Fail, makePolicy("test-policy")))))
	policy.Spec.Validations = []admissionregistrationv1alpha1.Validation{
		{
			Expression: "params.data.autofail != 'true' && (params.data.conditional == 'false' || object.metadata.name.startsWith(params.data.check))",
		},
	}
	policy = withWaitReadyConstraintAndExpression(policy)
	if _, err := client.AdmissionregistrationV1alpha1().ValidatingAdmissionPolicies().Create(context.TODO(), policy, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	autoFailParams := makeConfigParams("autofail-params", map[string]string{
		"autofail": "true",
	})
	if _, err := client.CoreV1().ConfigMaps("default").Create(context.TODO(), autoFailParams, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	autofailBinding := withBindingExistsLabels([]string{"autofail-binding-label"}, policy, makeBinding("autofail-binding", "test-policy", "autofail-params"))
	if err := createAndWaitReady(t, client, autofailBinding, map[string]string{"paramIdent": "true", "autofail-binding-label": "true"}); err != nil {
		t.Fatal(err)
	}

	autoPassParams := makeConfigParams("autopass-params", map[string]string{
		"autofail":    "false",
		"conditional": "false",
	})
	if _, err := client.CoreV1().ConfigMaps("default").Create(context.TODO(), autoPassParams, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	autopassBinding := withBindingExistsLabels([]string{"autopass-binding-label"}, policy, makeBinding("autopass-binding", "test-policy", "autopass-params"))
	if err := createAndWaitReady(t, client, autopassBinding, map[string]string{"paramIdent": "true", "autopass-binding-label": "true"}); err != nil {
		t.Fatal(err)
	}

	condpassParams := makeConfigParams("condpass-params", map[string]string{
		"autofail":    "false",
		"conditional": "true",
		"check":       "prefix-",
	})
	if _, err := client.CoreV1().ConfigMaps("default").Create(context.TODO(), condpassParams, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	condpassBinding := withBindingExistsLabels([]string{"condpass-binding-label"}, policy, makeBinding("condpass-binding", "test-policy", "condpass-params"))
	if err := createAndWaitReady(t, client, condpassBinding, map[string]string{"paramIdent": "true", "condpass-binding-label": "true"}); err != nil {
		t.Fatal(err)
	}

	autofailingSecret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name: "autofailing-secret",
			Labels: map[string]string{
				"paramIdent":             "someVal",
				"autofail-binding-label": "true",
			},
		},
	}
	_, err = client.CoreV1().Secrets("default").Create(context.TODO(), autofailingSecret, metav1.CreateOptions{})
	if err == nil {
		t.Fatal("expected secret creation to fail due to autofail-binding")
	}
	checkForFailedRule(t, err)
	checkFailureReason(t, err, metav1.StatusReasonInvalid)

	autopassingSecret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name: "autopassing-secret",
			Labels: map[string]string{
				"paramIdent":             "someVal",
				"autopass-binding-label": "true",
			},
		},
	}
	if _, err := client.CoreV1().Secrets("default").Create(context.TODO(), autopassingSecret, metav1.CreateOptions{}); err != nil {
		t.Fatalf("expected secret creation to succeed, got: %s", err)
	}

	condpassingSecret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name: "prefix-condpassing-secret",
			Labels: map[string]string{
				"paramIdent":             "someVal",
				"condpass-binding-label": "true",
			},
		},
	}
	if _, err := client.CoreV1().Secrets("default").Create(context.TODO(), condpassingSecret, metav1.CreateOptions{}); err != nil {
		t.Fatalf("expected secret creation to succeed, got: %s", err)
	}

	condfailingSecret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name: "condfailing-secret",
			Labels: map[string]string{
				"paramIdent":             "someVal",
				"condpass-binding-label": "true",
			},
		},
	}
	_, err = client.CoreV1().Secrets("default").Create(context.TODO(), condfailingSecret, metav1.CreateOptions{})
	if err == nil {
		t.Fatal("expected secret creation to fail due to autofail-binding")
	}
	checkForFailedRule(t, err)
	checkFailureReason(t, err, metav1.StatusReasonInvalid)
}

// Test_PolicyExemption tests that ValidatingAdmissionPolicy and ValidatingAdmissionPolicyBinding resources
// are exempt from policy rules.
func Test_PolicyExemption(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ValidatingAdmissionPolicy, true)()
	server, err := apiservertesting.StartTestServer(t, nil, []string{
		"--enable-admission-plugins", "ValidatingAdmissionPolicy",
	}, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	defer server.TearDownFn()

	config := server.ClientConfig

	client, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	policy := makePolicy("test-policy")
	policy.Spec.MatchConstraints = &admissionregistrationv1alpha1.MatchResources{
		ResourceRules: []admissionregistrationv1alpha1.NamedRuleWithOperations{
			{
				RuleWithOperations: admissionregistrationv1alpha1.RuleWithOperations{
					Operations: []admissionregistrationv1.OperationType{
						"*",
					},
					Rule: admissionregistrationv1.Rule{
						APIGroups: []string{
							"*",
						},
						APIVersions: []string{
							"*",
						},
						Resources: []string{
							"*",
						},
					},
				},
			},
		},
	}

	policy.Spec.Validations = []admissionregistrationv1alpha1.Validation{{
		Expression: "false",
		Message:    "marker denied; policy is ready",
	}}

	policy, err = client.AdmissionregistrationV1alpha1().ValidatingAdmissionPolicies().Create(context.TODO(), policy, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	policyBinding := makeBinding("test-policy-binding", "test-policy", "")
	if err := createAndWaitReady(t, client, policyBinding, nil); err != nil {
		t.Fatal(err)
	}

	// validate that operations to ValidatingAdmissionPolicy are exempt from an existing policy that catches all resources
	policyCopy := policy.DeepCopy()
	ignoreFailurePolicy := admissionregistrationv1alpha1.Ignore
	policyCopy.Spec.FailurePolicy = &ignoreFailurePolicy
	_, err = client.AdmissionregistrationV1alpha1().ValidatingAdmissionPolicies().Update(context.TODO(), policyCopy, metav1.UpdateOptions{})
	if err != nil {
		t.Error(err)
	}

	policyBinding, err = client.AdmissionregistrationV1alpha1().ValidatingAdmissionPolicyBindings().Get(context.TODO(), policyBinding.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// validate that operations to ValidatingAdmissionPolicyBindings are exempt from an existing policy that catches all resources
	policyBindingCopy := policyBinding.DeepCopy()
	policyBindingCopy.Spec.PolicyName = "different-binding"
	_, err = client.AdmissionregistrationV1alpha1().ValidatingAdmissionPolicyBindings().Update(context.TODO(), policyBindingCopy, metav1.UpdateOptions{})
	if err != nil {
		t.Error(err)
	}
}

// Test_ValidatingAdmissionPolicy_UpdateParamKind validates the behavior of ValidatingAdmissionPolicy when
// only the ParamKind is updated. This test creates a policy where namespaces must have a prefix that matches
// the ParamKind set in the policy. Switching the ParamKind should result in only namespaces with prefixes matching
// the new ParamKind to be allowed. For example, when Paramkind is v1/ConfigMap, only namespaces prefixed with "configmap"
// is allowed and when ParamKind is updated to v1/Secret, only namespaces prefixed with "secret" is allowed, etc.
func Test_ValidatingAdmissionPolicy_UpdateParamKind(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ValidatingAdmissionPolicy, true)()
	server, err := apiservertesting.StartTestServer(t, nil, []string{
		"--enable-admission-plugins", "ValidatingAdmissionPolicy",
	}, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	defer server.TearDownFn()

	config := server.ClientConfig

	client, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	allowedPrefixesParamsConfigMap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name: "allowed-prefixes",
		},
	}
	if _, err := client.CoreV1().ConfigMaps("default").Create(context.TODO(), allowedPrefixesParamsConfigMap, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	allowedPrefixesParamSecret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name: "allowed-prefixes",
		},
	}
	if _, err := client.CoreV1().Secrets("default").Create(context.TODO(), allowedPrefixesParamSecret, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	paramKind := &admissionregistrationv1alpha1.ParamKind{
		APIVersion: "v1",
		Kind:       "ConfigMap",
	}

	policy := withValidations([]admissionregistrationv1alpha1.Validation{
		{
			Expression: "object.metadata.name.startsWith(params.kind.lowerAscii())",
			Message:    "wrong paramKind",
		},
	}, withParams(paramKind, withNamespaceMatch(withFailurePolicy(admissionregistrationv1alpha1.Fail, makePolicy("allowed-prefixes")))))
	policy = withWaitReadyConstraintAndExpression(policy)
	policy, err = client.AdmissionregistrationV1alpha1().ValidatingAdmissionPolicies().Create(context.TODO(), policy, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	allowedPrefixesBinding := makeBinding("allowed-prefixes-binding", "allowed-prefixes", "allowed-prefixes")
	if err := createAndWaitReady(t, client, allowedPrefixesBinding, nil); err != nil {
		t.Fatal(err)
	}

	// validate that namespaces starting with "configmap-" are allowed
	// and namespaces starting with "secret-" are disallowed
	allowedNamespace := &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "configmap-",
		},
	}
	_, err = client.CoreV1().Namespaces().Create(context.TODO(), allowedNamespace, metav1.CreateOptions{})
	if err != nil {
		t.Error(err)
	}

	disallowedNamespace := &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "secret-",
		},
	}
	_, err = client.CoreV1().Namespaces().Create(context.TODO(), disallowedNamespace, metav1.CreateOptions{})
	if err == nil {
		t.Error("unexpected nil error")
	}
	if !strings.Contains(err.Error(), "wrong paramKind") {
		t.Errorf("unexpected error message: %v", err)
	}
	checkFailureReason(t, err, metav1.StatusReasonInvalid)

	// update the policy ParamKind to reference a Secret
	paramKind = &admissionregistrationv1alpha1.ParamKind{
		APIVersion: "v1",
		Kind:       "Secret",
	}
	policyCopy := policy.DeepCopy()
	policyCopy.Spec.ParamKind = paramKind
	_, err = client.AdmissionregistrationV1alpha1().ValidatingAdmissionPolicies().Update(context.TODO(), policyCopy, metav1.UpdateOptions{})
	if err != nil {
		t.Error(err)
	}

	// validate that namespaces starting with "secret-" are allowed
	// and namespaces starting with "configmap-" are disallowed
	// wait loop is required here since ConfigMaps were previousy allowed and we need to wait for the new policy
	// to be enforced
	if waitErr := wait.PollImmediate(time.Millisecond*10, wait.ForeverTestTimeout, func() (bool, error) {
		disallowedNamespace = &v1.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "configmap-",
			},
		}

		_, err = client.CoreV1().Namespaces().Create(context.TODO(), disallowedNamespace, metav1.CreateOptions{})
		if err == nil {
			return false, nil
		}

		if strings.Contains(err.Error(), "not yet synced to use for admission") {
			return false, nil
		}

		if !strings.Contains(err.Error(), "wrong paramKind") {
			return false, err
		}

		return true, nil
	}); waitErr != nil {
		t.Errorf("timed out waiting: %v", err)
	}

	allowedNamespace = &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "secret-",
		},
	}
	_, err = client.CoreV1().Namespaces().Create(context.TODO(), allowedNamespace, metav1.CreateOptions{})
	if err != nil {
		t.Error(err)
	}
}

// Test_ValidatingAdmissionPolicy_UpdateParamRef validates the behavior of ValidatingAdmissionPolicy when
// only the ParamRef in the binding is updated. This test creates a policy where namespaces must have a prefix that matches
// the ParamRef set in the policy binding. The paramRef in the binding is then updated to a different object.
func Test_ValidatingAdmissionPolicy_UpdateParamRef(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ValidatingAdmissionPolicy, true)()
	server, err := apiservertesting.StartTestServer(t, nil, []string{
		"--enable-admission-plugins", "ValidatingAdmissionPolicy",
	}, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	defer server.TearDownFn()

	config := server.ClientConfig

	client, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	allowedPrefixesParamsConfigMap1 := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-1",
			Namespace: "default",
		},
	}
	if _, err := client.CoreV1().ConfigMaps("default").Create(context.TODO(), allowedPrefixesParamsConfigMap1, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	allowedPrefixesParamsConfigMap2 := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-2",
			Namespace: "default",
		},
	}
	if _, err := client.CoreV1().ConfigMaps("default").Create(context.TODO(), allowedPrefixesParamsConfigMap2, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	policy := withValidations([]admissionregistrationv1alpha1.Validation{
		{
			Expression: "object.metadata.name.startsWith(params.metadata.name)",
			Message:    "wrong paramRef",
		},
	}, withParams(configParamKind(), withNamespaceMatch(withFailurePolicy(admissionregistrationv1alpha1.Fail, makePolicy("allowed-prefixes")))))
	policy = withWaitReadyConstraintAndExpression(policy)
	_, err = client.AdmissionregistrationV1alpha1().ValidatingAdmissionPolicies().Create(context.TODO(), policy, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// validate that namespaces starting with "test-1" are allowed
	// and namespaces starting with "test-2-" are disallowed
	allowedPrefixesBinding := makeBinding("allowed-prefixes-binding", "allowed-prefixes", "test-1")
	if err := createAndWaitReady(t, client, allowedPrefixesBinding, nil); err != nil {
		t.Fatal(err)
	}

	if waitErr := wait.PollImmediate(time.Millisecond*10, wait.ForeverTestTimeout, func() (bool, error) {
		disallowedNamespace := &v1.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "test-2-",
			},
		}

		_, err = client.CoreV1().Namespaces().Create(context.TODO(), disallowedNamespace, metav1.CreateOptions{})
		if err == nil {
			return false, nil
		}

		if strings.Contains(err.Error(), "not yet synced to use for admission") {
			return false, nil
		}

		if !strings.Contains(err.Error(), "wrong paramRef") {
			return false, err
		}

		return true, nil
	}); waitErr != nil {
		t.Errorf("timed out waiting: %v", err)
	}

	allowedNamespace := &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-1-",
		},
	}
	_, err = client.CoreV1().Namespaces().Create(context.TODO(), allowedNamespace, metav1.CreateOptions{})
	if err != nil {
		t.Error(err)
	}

	// Update the paramRef in the policy binding to use the test-2 ConfigMap
	policyBinding, err := client.AdmissionregistrationV1alpha1().ValidatingAdmissionPolicyBindings().Get(context.TODO(), allowedPrefixesBinding.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}

	policyBindingCopy := policyBinding.DeepCopy()
	policyBindingCopy.Spec.ParamRef = &admissionregistrationv1alpha1.ParamRef{
		Name:      "test-2",
		Namespace: "default",
	}
	_, err = client.AdmissionregistrationV1alpha1().ValidatingAdmissionPolicyBindings().Update(context.TODO(), policyBindingCopy, metav1.UpdateOptions{})
	if err != nil {
		t.Error(err)
	}

	// validate that namespaces starting with "test-2" are allowed
	// and namespaces starting with "test-1" are disallowed
	if waitErr := wait.PollImmediate(time.Millisecond*10, wait.ForeverTestTimeout, func() (bool, error) {
		disallowedNamespace := &v1.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "test-1-",
			},
		}

		_, err = client.CoreV1().Namespaces().Create(context.TODO(), disallowedNamespace, metav1.CreateOptions{})
		if err == nil {
			return false, nil
		}

		if strings.Contains(err.Error(), "not yet synced to use for admission") {
			return false, nil
		}

		if !strings.Contains(err.Error(), "wrong paramRef") {
			return false, err
		}

		return true, nil
	}); waitErr != nil {
		t.Errorf("timed out waiting: %v", err)
	}

	allowedNamespace = &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-2-",
		},
	}
	_, err = client.CoreV1().Namespaces().Create(context.TODO(), allowedNamespace, metav1.CreateOptions{})
	if err != nil {
		t.Error(err)
	}
}

// Test_ValidatingAdmissionPolicy_UpdateParamResource validates behavior of a policy after updates to the param resource.
func Test_ValidatingAdmissionPolicy_UpdateParamResource(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ValidatingAdmissionPolicy, true)()
	server, err := apiservertesting.StartTestServer(t, nil, []string{
		"--enable-admission-plugins", "ValidatingAdmissionPolicy",
	}, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	defer server.TearDownFn()

	config := server.ClientConfig

	client, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	paramConfigMap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "allowed-prefix",
			Namespace: "default",
		},
		Data: map[string]string{
			"prefix": "test-1",
		},
	}
	paramConfigMap, err = client.CoreV1().ConfigMaps(paramConfigMap.Namespace).Create(context.TODO(), paramConfigMap, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	policy := withValidations([]admissionregistrationv1alpha1.Validation{
		{
			Expression: "object.metadata.name.startsWith(params.data['prefix'])",
			Message:    "wrong prefix",
		},
	}, withParams(configParamKind(), withNamespaceMatch(withFailurePolicy(admissionregistrationv1alpha1.Fail, makePolicy("allowed-prefixes")))))
	policy = withWaitReadyConstraintAndExpression(policy)
	_, err = client.AdmissionregistrationV1alpha1().ValidatingAdmissionPolicies().Create(context.TODO(), policy, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// validate that namespaces starting with "test-1" are allowed
	// and namespaces starting with "test-2-" are disallowed
	allowedPrefixesBinding := makeBinding("allowed-prefixes-binding", "allowed-prefixes", "allowed-prefix")
	if err := createAndWaitReady(t, client, allowedPrefixesBinding, nil); err != nil {
		t.Fatal(err)
	}

	if waitErr := wait.PollImmediate(time.Millisecond*10, wait.ForeverTestTimeout, func() (bool, error) {
		disallowedNamespace := &v1.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "test-2-",
			},
		}

		_, err = client.CoreV1().Namespaces().Create(context.TODO(), disallowedNamespace, metav1.CreateOptions{})
		if err == nil {
			return false, nil
		}

		if strings.Contains(err.Error(), "not yet synced to use for admission") {
			return false, nil
		}

		if !strings.Contains(err.Error(), "wrong prefix") {
			return false, err
		}

		return true, nil
	}); waitErr != nil {
		t.Errorf("timed out waiting: %v", err)
	}

	allowedNamespace := &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-1-",
		},
	}
	_, err = client.CoreV1().Namespaces().Create(context.TODO(), allowedNamespace, metav1.CreateOptions{})
	if err != nil {
		t.Error(err)
	}

	// Update the param resource to use "test-2" as the new allwoed prefix
	paramConfigMapCopy := paramConfigMap.DeepCopy()
	paramConfigMapCopy.Data = map[string]string{
		"prefix": "test-2",
	}
	_, err = client.CoreV1().ConfigMaps(paramConfigMapCopy.Namespace).Update(context.TODO(), paramConfigMapCopy, metav1.UpdateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// validate that namespaces starting with "test-2" are allowed
	// and namespaces starting with "test-1" are disallowed
	if waitErr := wait.PollImmediate(time.Millisecond*10, wait.ForeverTestTimeout, func() (bool, error) {
		disallowedNamespace := &v1.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "test-1-",
			},
		}

		_, err = client.CoreV1().Namespaces().Create(context.TODO(), disallowedNamespace, metav1.CreateOptions{})
		if err == nil {
			return false, nil
		}

		if strings.Contains(err.Error(), "not yet synced to use for admission") {
			return false, nil
		}

		if !strings.Contains(err.Error(), "wrong prefix") {
			return false, err
		}

		return true, nil
	}); waitErr != nil {
		t.Errorf("timed out waiting: %v", err)
	}

	allowedNamespace = &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-2-",
		},
	}
	_, err = client.CoreV1().Namespaces().Create(context.TODO(), allowedNamespace, metav1.CreateOptions{})
	if err != nil {
		t.Error(err)
	}
}

func Test_ValidatingAdmissionPolicy_MatchByObjectSelector(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ValidatingAdmissionPolicy, true)()
	server, err := apiservertesting.StartTestServer(t, nil, []string{
		"--enable-admission-plugins", "ValidatingAdmissionPolicy",
	}, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	defer server.TearDownFn()

	config := server.ClientConfig

	client, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	labelSelector := &metav1.LabelSelector{
		MatchLabels: map[string]string{
			"foo": "bar",
		},
	}

	policy := withValidations([]admissionregistrationv1alpha1.Validation{
		{
			Expression: "false",
			Message:    "matched by object selector!",
		},
	}, withConfigMapMatch(withFailurePolicy(admissionregistrationv1alpha1.Fail, makePolicy("match-by-object-selector"))))
	policy = withObjectSelector(labelSelector, policy)
	policy = withWaitReadyConstraintAndExpression(policy)
	_, err = client.AdmissionregistrationV1alpha1().ValidatingAdmissionPolicies().Create(context.TODO(), policy, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	policyBinding := makeBinding("match-by-object-selector-binding", "match-by-object-selector", "")
	if err := createAndWaitReady(t, client, policyBinding, map[string]string{"foo": "bar"}); err != nil {
		t.Fatal(err)
	}

	matchedConfigMap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "denied",
			Namespace: "default",
			Labels: map[string]string{
				"foo": "bar",
			},
		},
	}

	_, err = client.CoreV1().ConfigMaps(matchedConfigMap.Namespace).Create(context.TODO(), matchedConfigMap, metav1.CreateOptions{})
	if !strings.Contains(err.Error(), "matched by object selector!") {
		t.Errorf("unexpected error: %v", err)
	}

	allowedConfigMap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "allowed",
			Namespace: "default",
		},
	}

	if _, err := client.CoreV1().ConfigMaps(allowedConfigMap.Namespace).Create(context.TODO(), allowedConfigMap, metav1.CreateOptions{}); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func Test_ValidatingAdmissionPolicy_MatchByNamespaceSelector(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ValidatingAdmissionPolicy, true)()
	server, err := apiservertesting.StartTestServer(t, nil, []string{
		"--enable-admission-plugins", "ValidatingAdmissionPolicy",
	}, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	defer server.TearDownFn()

	config := server.ClientConfig

	client, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	// only configmaps in default will be allowed.
	labelSelector := &metav1.LabelSelector{
		MatchExpressions: []metav1.LabelSelectorRequirement{
			{
				Key:      "kubernetes.io/metadata.name",
				Operator: "NotIn",
				Values:   []string{"default"},
			},
		},
	}

	policy := withValidations([]admissionregistrationv1alpha1.Validation{
		{
			Expression: "false",
			Message:    "matched by namespace selector!",
		},
	}, withConfigMapMatch(withFailurePolicy(admissionregistrationv1alpha1.Fail, makePolicy("match-by-namespace-selector"))))
	policy = withNamespaceSelector(labelSelector, policy)
	policy = withWaitReadyConstraintAndExpression(policy)
	_, err = client.AdmissionregistrationV1alpha1().ValidatingAdmissionPolicies().Create(context.TODO(), policy, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	policyBinding := makeBinding("match-by-namespace-selector-binding", "match-by-namespace-selector", "")
	_, err = client.AdmissionregistrationV1alpha1().ValidatingAdmissionPolicyBindings().Create(context.TODO(), policyBinding, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	namespace := &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: "not-default",
		},
	}
	if _, err := client.CoreV1().Namespaces().Create(context.TODO(), namespace, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	if waitErr := wait.PollImmediate(time.Millisecond*10, wait.ForeverTestTimeout, func() (bool, error) {
		matchedConfigMap := &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "denied-",
				Namespace:    "not-default",
			},
		}

		_, err := client.CoreV1().ConfigMaps(matchedConfigMap.Namespace).Create(context.TODO(), matchedConfigMap, metav1.CreateOptions{})
		// policy not enforced yet, try again
		if err == nil {
			return false, nil
		}

		if !strings.Contains(err.Error(), "matched by namespace selector!") {
			return false, err
		}

		return true, nil

	}); waitErr != nil {
		t.Errorf("timed out waiting: %v", waitErr)
	}

	allowedConfigMap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "allowed",
			Namespace: "default",
		},
	}

	if _, err := client.CoreV1().ConfigMaps(allowedConfigMap.Namespace).Create(context.TODO(), allowedConfigMap, metav1.CreateOptions{}); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func Test_ValidatingAdmissionPolicy_MatchByResourceNames(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ValidatingAdmissionPolicy, true)()
	server, err := apiservertesting.StartTestServer(t, nil, []string{
		"--enable-admission-plugins", "ValidatingAdmissionPolicy",
	}, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	defer server.TearDownFn()

	config := server.ClientConfig

	client, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	policy := withValidations([]admissionregistrationv1alpha1.Validation{
		{
			Expression: "false",
			Message:    "matched by resource names!",
		},
	}, withConfigMapMatch(withFailurePolicy(admissionregistrationv1alpha1.Fail, makePolicy("match-by-resource-names"))))
	policy.Spec.MatchConstraints.ResourceRules[0].ResourceNames = []string{"matched-by-resource-name"}
	policy = withWaitReadyConstraintAndExpression(policy)
	_, err = client.AdmissionregistrationV1alpha1().ValidatingAdmissionPolicies().Create(context.TODO(), policy, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	policyBinding := makeBinding("match-by-resource-names-binding", "match-by-resource-names", "")
	if err := createAndWaitReady(t, client, policyBinding, nil); err != nil {
		t.Fatal(err)
	}

	matchedConfigMap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "matched-by-resource-name",
			Namespace: "default",
		},
	}

	_, err = client.CoreV1().ConfigMaps(matchedConfigMap.Namespace).Create(context.TODO(), matchedConfigMap, metav1.CreateOptions{})
	if !strings.Contains(err.Error(), "matched by resource names!") {
		t.Errorf("unexpected error: %v", err)
	}

	allowedConfigMap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "not-matched-by-resource-name",
			Namespace: "default",
		},
	}

	if _, err := client.CoreV1().ConfigMaps(allowedConfigMap.Namespace).Create(context.TODO(), allowedConfigMap, metav1.CreateOptions{}); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func Test_ValidatingAdmissionPolicy_MatchWithExcludeResources(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ValidatingAdmissionPolicy, true)()
	server, err := apiservertesting.StartTestServer(t, nil, []string{
		"--enable-admission-plugins", "ValidatingAdmissionPolicy",
	}, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	defer server.TearDownFn()

	config := server.ClientConfig

	client, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	policy := withValidations([]admissionregistrationv1alpha1.Validation{
		{
			Expression: "false",
			Message:    "not matched by exclude resources!",
		},
	}, withPolicyMatch("*", withFailurePolicy(admissionregistrationv1alpha1.Fail, makePolicy("match-by-resource-names"))))

	policy = withExcludePolicyMatch("configmaps", policy)
	_, err = client.AdmissionregistrationV1alpha1().ValidatingAdmissionPolicies().Create(context.TODO(), policy, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	policyBinding := makeBinding("match-by-resource-names-binding", "match-by-resource-names", "")
	_, err = client.AdmissionregistrationV1alpha1().ValidatingAdmissionPolicyBindings().Create(context.TODO(), policyBinding, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	if waitErr := wait.PollImmediate(time.Millisecond*10, wait.ForeverTestTimeout, func() (bool, error) {
		secret := &v1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "not-matched-by-exclude-resources",
				Namespace:    "default",
			},
		}

		_, err := client.CoreV1().Secrets(secret.Namespace).Create(context.TODO(), secret, metav1.CreateOptions{})
		// policy not enforced yet, try again
		if err == nil {
			return false, nil
		}

		if !strings.Contains(err.Error(), "not matched by exclude resources!") {
			return false, err
		}

		return true, nil

	}); waitErr != nil {
		t.Errorf("timed out waiting: %v", waitErr)
	}

	allowedConfigMap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "matched-by-exclude-resources",
			Namespace: "default",
		},
	}

	if _, err := client.CoreV1().ConfigMaps(allowedConfigMap.Namespace).Create(context.TODO(), allowedConfigMap, metav1.CreateOptions{}); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func Test_ValidatingAdmissionPolicy_MatchWithMatchPolicyEquivalent(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ValidatingAdmissionPolicy, true)()
	server, err := apiservertesting.StartTestServer(t, nil, []string{
		"--enable-admission-plugins", "ValidatingAdmissionPolicy",
	}, framework.SharedEtcd())
	etcd.CreateTestCRDs(t, apiextensionsclientset.NewForConfigOrDie(server.ClientConfig), false, versionedCustomResourceDefinition())
	if err != nil {
		t.Fatal(err)
	}
	defer server.TearDownFn()

	config := server.ClientConfig

	client, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	policy := withValidations([]admissionregistrationv1alpha1.Validation{
		{
			Expression: "false",
			Message:    "matched by equivalent match policy!",
		},
	}, withFailurePolicy(admissionregistrationv1alpha1.Fail, makePolicy("match-by-match-policy-equivalent")))
	policy.Spec.MatchConstraints = &admissionregistrationv1alpha1.MatchResources{
		ResourceRules: []admissionregistrationv1alpha1.NamedRuleWithOperations{
			{
				RuleWithOperations: admissionregistrationv1alpha1.RuleWithOperations{
					Operations: []admissionregistrationv1.OperationType{
						"*",
					},
					Rule: admissionregistrationv1.Rule{
						APIGroups: []string{
							"awesome.bears.com",
						},
						APIVersions: []string{
							"v1",
						},
						Resources: []string{
							"pandas",
						},
					},
				},
			},
		},
	}
	policy = withWaitReadyConstraintAndExpression(policy)
	if _, err := client.AdmissionregistrationV1alpha1().ValidatingAdmissionPolicies().Create(context.TODO(), policy, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	policyBinding := makeBinding("match-by-match-policy-equivalent-binding", "match-by-match-policy-equivalent", "")
	if err := createAndWaitReady(t, client, policyBinding, nil); err != nil {
		t.Fatal(err)
	}

	v1Resource := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "awesome.bears.com" + "/" + "v1",
			"kind":       "Panda",
			"metadata": map[string]interface{}{
				"name": "v1-bears",
			},
		},
	}

	v2Resource := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "awesome.bears.com" + "/" + "v2",
			"kind":       "Panda",
			"metadata": map[string]interface{}{
				"name": "v2-bears",
			},
		},
	}

	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	_, err = dynamicClient.Resource(schema.GroupVersionResource{Group: "awesome.bears.com", Version: "v1", Resource: "pandas"}).Create(context.TODO(), v1Resource, metav1.CreateOptions{})
	if !strings.Contains(err.Error(), "matched by equivalent match policy!") {
		t.Errorf("v1 panadas did not match against policy, err: %v", err)
	}

	_, err = dynamicClient.Resource(schema.GroupVersionResource{Group: "awesome.bears.com", Version: "v2", Resource: "pandas"}).Create(context.TODO(), v2Resource, metav1.CreateOptions{})
	if !strings.Contains(err.Error(), "matched by equivalent match policy!") {
		t.Errorf("v2 panadas did not match against policy, err: %v", err)
	}
}

func Test_ValidatingAdmissionPolicy_MatchWithMatchPolicyExact(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ValidatingAdmissionPolicy, true)()
	server, err := apiservertesting.StartTestServer(t, nil, []string{
		"--enable-admission-plugins", "ValidatingAdmissionPolicy",
	}, framework.SharedEtcd())
	etcd.CreateTestCRDs(t, apiextensionsclientset.NewForConfigOrDie(server.ClientConfig), false, versionedCustomResourceDefinition())
	if err != nil {
		t.Fatal(err)
	}
	defer server.TearDownFn()

	config := server.ClientConfig

	client, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	policy := withValidations([]admissionregistrationv1alpha1.Validation{
		{
			Expression: "false",
			Message:    "matched by exact match policy!",
		},
	}, withFailurePolicy(admissionregistrationv1alpha1.Fail, makePolicy("match-by-match-policy-exact")))
	matchPolicyExact := admissionregistrationv1alpha1.Exact
	policy.Spec.MatchConstraints = &admissionregistrationv1alpha1.MatchResources{
		MatchPolicy: &matchPolicyExact,
		ResourceRules: []admissionregistrationv1alpha1.NamedRuleWithOperations{
			{
				RuleWithOperations: admissionregistrationv1alpha1.RuleWithOperations{
					Operations: []admissionregistrationv1.OperationType{
						"*",
					},
					Rule: admissionregistrationv1.Rule{
						APIGroups: []string{
							"awesome.bears.com",
						},
						APIVersions: []string{
							"v1",
						},
						Resources: []string{
							"pandas",
						},
					},
				},
			},
		},
	}
	policy = withWaitReadyConstraintAndExpression(policy)
	if _, err := client.AdmissionregistrationV1alpha1().ValidatingAdmissionPolicies().Create(context.TODO(), policy, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	policyBinding := makeBinding("match-by-match-policy-exact-binding", "match-by-match-policy-exact", "")
	if err := createAndWaitReady(t, client, policyBinding, nil); err != nil {
		t.Fatal(err)
	}

	v1Resource := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "awesome.bears.com" + "/" + "v1",
			"kind":       "Panda",
			"metadata": map[string]interface{}{
				"name": "v1-bears",
			},
		},
	}

	v2Resource := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "awesome.bears.com" + "/" + "v2",
			"kind":       "Panda",
			"metadata": map[string]interface{}{
				"name": "v2-bears",
			},
		},
	}

	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	_, err = dynamicClient.Resource(schema.GroupVersionResource{Group: "awesome.bears.com", Version: "v1", Resource: "pandas"}).Create(context.TODO(), v1Resource, metav1.CreateOptions{})
	if !strings.Contains(err.Error(), "matched by exact match policy!") {
		t.Errorf("v1 panadas did not match against policy, err: %v", err)
	}

	// v2 panadas is allowed since policy specificed match policy Exact and only matched against v1
	_, err = dynamicClient.Resource(schema.GroupVersionResource{Group: "awesome.bears.com", Version: "v2", Resource: "pandas"}).Create(context.TODO(), v2Resource, metav1.CreateOptions{})
	if err != nil {
		t.Error(err)
	}
}

// Test_ValidatingAdmissionPolicy_PolicyDeletedThenRecreated validates that deleting a ValidatingAdmissionPolicy
// removes the policy from the apiserver admission chain and recreating it re-enables it.
func Test_ValidatingAdmissionPolicy_PolicyDeletedThenRecreated(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ValidatingAdmissionPolicy, true)()
	server, err := apiservertesting.StartTestServer(t, nil, []string{
		"--enable-admission-plugins", "ValidatingAdmissionPolicy",
	}, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	defer server.TearDownFn()

	config := server.ClientConfig

	client, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	policy := withValidations([]admissionregistrationv1alpha1.Validation{
		{
			Expression: "object.metadata.name.startsWith('test')",
			Message:    "wrong prefix",
		},
	}, withParams(configParamKind(), withNamespaceMatch(withFailurePolicy(admissionregistrationv1alpha1.Fail, makePolicy("allowed-prefixes")))))
	policy = withWaitReadyConstraintAndExpression(policy)
	_, err = client.AdmissionregistrationV1alpha1().ValidatingAdmissionPolicies().Create(context.TODO(), policy, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// validate that namespaces starting with "test" are allowed
	policyBinding := makeBinding("allowed-prefixes-binding", "allowed-prefixes", "")
	if err := createAndWaitReady(t, client, policyBinding, nil); err != nil {
		t.Fatal(err)
	}

	if waitErr := wait.PollImmediate(time.Millisecond*10, wait.ForeverTestTimeout, func() (bool, error) {
		disallowedNamespace := &v1.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "not-test-",
			},
		}

		_, err = client.CoreV1().Namespaces().Create(context.TODO(), disallowedNamespace, metav1.CreateOptions{})
		if err == nil {
			return false, nil
		}

		if strings.Contains(err.Error(), "not yet synced to use for admission") {
			return false, nil
		}

		if !strings.Contains(err.Error(), "wrong prefix") {
			return false, err
		}

		return true, nil
	}); waitErr != nil {
		t.Errorf("timed out waiting: %v", err)
	}

	// delete the binding object and validate that policy is not enforced
	if err := client.AdmissionregistrationV1alpha1().ValidatingAdmissionPolicies().Delete(context.TODO(), "allowed-prefixes", metav1.DeleteOptions{}); err != nil {
		t.Fatal(err)
	}

	if waitErr := wait.PollImmediate(time.Millisecond*10, wait.ForeverTestTimeout, func() (bool, error) {
		allowedNamespace := &v1.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "not-test-",
			},
		}
		_, err = client.CoreV1().Namespaces().Create(context.TODO(), allowedNamespace, metav1.CreateOptions{})
		if err == nil {
			return true, nil
		}

		// old policy is still enforced, try again
		if strings.Contains(err.Error(), "wrong prefix") {
			return false, nil
		}

		return false, err
	}); waitErr != nil {
		t.Errorf("timed out waiting: %v", err)
	}

	_, err = client.AdmissionregistrationV1alpha1().ValidatingAdmissionPolicies().Create(context.TODO(), policy, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if waitErr := wait.PollImmediate(time.Millisecond*10, wait.ForeverTestTimeout, func() (bool, error) {
		disallowedNamespace := &v1.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "not-test-",
			},
		}

		_, err = client.CoreV1().Namespaces().Create(context.TODO(), disallowedNamespace, metav1.CreateOptions{})
		if err == nil {
			return false, nil
		}

		if strings.Contains(err.Error(), "not yet synced to use for admission") {
			return false, nil
		}

		if !strings.Contains(err.Error(), "wrong prefix") {
			return false, err
		}

		return true, nil
	}); waitErr != nil {
		t.Errorf("timed out waiting: %v", err)
	}
}

// Test_ValidatingAdmissionPolicy_BindingDeletedThenRecreated validates that deleting a ValidatingAdmissionPolicyBinding
// removes the policy from the apiserver admission chain and recreating it re-enables it.
func Test_ValidatingAdmissionPolicy_BindingDeletedThenRecreated(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ValidatingAdmissionPolicy, true)()
	server, err := apiservertesting.StartTestServer(t, nil, []string{
		"--enable-admission-plugins", "ValidatingAdmissionPolicy",
	}, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	defer server.TearDownFn()

	config := server.ClientConfig

	client, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	policy := withValidations([]admissionregistrationv1alpha1.Validation{
		{
			Expression: "object.metadata.name.startsWith('test')",
			Message:    "wrong prefix",
		},
	}, withParams(configParamKind(), withNamespaceMatch(withFailurePolicy(admissionregistrationv1alpha1.Fail, makePolicy("allowed-prefixes")))))
	policy = withWaitReadyConstraintAndExpression(policy)
	_, err = client.AdmissionregistrationV1alpha1().ValidatingAdmissionPolicies().Create(context.TODO(), policy, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// validate that namespaces starting with "test" are allowed
	policyBinding := makeBinding("allowed-prefixes-binding", "allowed-prefixes", "")
	if err := createAndWaitReady(t, client, policyBinding, nil); err != nil {
		t.Fatal(err)
	}

	if waitErr := wait.PollImmediate(time.Millisecond*10, wait.ForeverTestTimeout, func() (bool, error) {
		disallowedNamespace := &v1.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "not-test-",
			},
		}

		_, err = client.CoreV1().Namespaces().Create(context.TODO(), disallowedNamespace, metav1.CreateOptions{})
		if err == nil {
			return false, nil
		}

		if strings.Contains(err.Error(), "not yet synced to use for admission") {
			return false, nil
		}

		if !strings.Contains(err.Error(), "wrong prefix") {
			return false, err
		}

		return true, nil
	}); waitErr != nil {
		t.Errorf("timed out waiting: %v", err)
	}

	// delete the binding object and validate that policy is not enforced
	if err := client.AdmissionregistrationV1alpha1().ValidatingAdmissionPolicyBindings().Delete(context.TODO(), "allowed-prefixes-binding", metav1.DeleteOptions{}); err != nil {
		t.Fatal(err)
	}

	if waitErr := wait.PollImmediate(time.Millisecond*10, wait.ForeverTestTimeout, func() (bool, error) {
		allowedNamespace := &v1.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "not-test-",
			},
		}
		_, err = client.CoreV1().Namespaces().Create(context.TODO(), allowedNamespace, metav1.CreateOptions{})
		if err == nil {
			return true, nil
		}

		// old policy is still enforced, try again
		if strings.Contains(err.Error(), "wrong prefix") {
			return false, nil
		}

		return false, err
	}); waitErr != nil {
		t.Errorf("timed out waiting: %v", err)
	}

	// recreate the policy binding and test that policy is enforced again
	_, err = client.AdmissionregistrationV1alpha1().ValidatingAdmissionPolicyBindings().Create(context.TODO(), policyBinding, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if waitErr := wait.PollImmediate(time.Millisecond*10, wait.ForeverTestTimeout, func() (bool, error) {
		disallowedNamespace := &v1.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "not-test-",
			},
		}

		_, err = client.CoreV1().Namespaces().Create(context.TODO(), disallowedNamespace, metav1.CreateOptions{})
		if err == nil {
			return false, nil
		}

		if strings.Contains(err.Error(), "not yet synced to use for admission") {
			return false, nil
		}

		if !strings.Contains(err.Error(), "wrong prefix") {
			return false, err
		}

		return true, nil
	}); waitErr != nil {
		t.Errorf("timed out waiting: %v", err)
	}
}

// Test_ValidatingAdmissionPolicy_ParamResourceDeletedThenRecreated validates that deleting a param resource referenced
// by a binding renders the policy as invalid. Recreating the param resource re-enables the policy.
func Test_ValidatingAdmissionPolicy_ParamResourceDeletedThenRecreated(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ValidatingAdmissionPolicy, true)()
	server, err := apiservertesting.StartTestServer(t, nil, []string{
		"--enable-admission-plugins", "ValidatingAdmissionPolicy",
	}, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	defer server.TearDownFn()

	config := server.ClientConfig

	client, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	param := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			Namespace: "default",
		},
	}
	if _, err := client.CoreV1().ConfigMaps("default").Create(context.TODO(), param, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	policy := withValidations([]admissionregistrationv1alpha1.Validation{
		{
			Expression: "object.metadata.name.startsWith(params.metadata.name)",
			Message:    "wrong prefix",
		},
	}, withParams(configParamKind(), withNamespaceMatch(withFailurePolicy(admissionregistrationv1alpha1.Fail, makePolicy("allowed-prefixes")))))
	policy = withWaitReadyConstraintAndExpression(policy)
	_, err = client.AdmissionregistrationV1alpha1().ValidatingAdmissionPolicies().Create(context.TODO(), policy, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// validate that namespaces starting with "test" are allowed
	policyBinding := makeBinding("allowed-prefixes-binding", "allowed-prefixes", "test")
	if err := createAndWaitReady(t, client, policyBinding, nil); err != nil {
		t.Fatal(err)
	}

	if waitErr := wait.PollImmediate(time.Millisecond*10, wait.ForeverTestTimeout, func() (bool, error) {
		disallowedNamespace := &v1.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "not-test-",
			},
		}

		_, err = client.CoreV1().Namespaces().Create(context.TODO(), disallowedNamespace, metav1.CreateOptions{})
		if err == nil {
			return false, nil
		}

		if strings.Contains(err.Error(), "not yet synced to use for admission") {
			return false, nil
		}

		if !strings.Contains(err.Error(), "wrong prefix") {
			return false, err
		}

		return true, nil
	}); waitErr != nil {
		t.Errorf("timed out waiting: %v", err)
	}

	// delete param object and validate that policy is invalid
	if err := client.CoreV1().ConfigMaps("default").Delete(context.TODO(), "test", metav1.DeleteOptions{}); err != nil {
		t.Fatal(err)
	}

	if waitErr := wait.PollImmediate(time.Millisecond*10, wait.ForeverTestTimeout, func() (bool, error) {
		allowedNamespace := &v1.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "not-test-",
			},
		}

		_, err = client.CoreV1().Namespaces().Create(context.TODO(), allowedNamespace, metav1.CreateOptions{})
		// old policy is still enforced, try again
		if strings.Contains(err.Error(), "wrong prefix") {
			return false, nil
		}

		if !strings.Contains(err.Error(), "failed to configure binding: test not found") {
			return false, err
		}

		return true, nil
	}); waitErr != nil {
		t.Errorf("timed out waiting: %v", err)
	}

	// recreate the param resource and validate namespace is disallowed again
	if _, err := client.CoreV1().ConfigMaps("default").Create(context.TODO(), param, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	if waitErr := wait.PollImmediate(time.Millisecond*10, wait.ForeverTestTimeout, func() (bool, error) {
		disallowedNamespace := &v1.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "not-test-",
			},
		}

		_, err = client.CoreV1().Namespaces().Create(context.TODO(), disallowedNamespace, metav1.CreateOptions{})
		// cache not synced with new object yet, try again
		if strings.Contains(err.Error(), "failed to configure binding: test not found") {
			return false, nil
		}

		if !strings.Contains(err.Error(), "wrong prefix") {
			return false, err
		}

		return true, nil
	}); waitErr != nil {
		t.Errorf("timed out waiting: %v", err)
	}
}

func withWaitReadyConstraintAndExpression(policy *admissionregistrationv1alpha1.ValidatingAdmissionPolicy) *admissionregistrationv1alpha1.ValidatingAdmissionPolicy {
	policy = policy.DeepCopy()
	policy.Spec.MatchConstraints.ResourceRules = append(policy.Spec.MatchConstraints.ResourceRules, admissionregistrationv1alpha1.NamedRuleWithOperations{
		ResourceNames: []string{"test-marker"},
		RuleWithOperations: admissionregistrationv1alpha1.RuleWithOperations{
			Operations: []admissionregistrationv1.OperationType{
				"UPDATE",
			},
			Rule: admissionregistrationv1.Rule{
				APIGroups: []string{
					"",
				},
				APIVersions: []string{
					"v1",
				},
				Resources: []string{
					"endpoints",
				},
			},
		},
	})
	policy.Spec.Validations = append([]admissionregistrationv1alpha1.Validation{{
		Expression: "object.metadata.name != 'test-marker'",
		Message:    "marker denied; policy is ready",
	}}, policy.Spec.Validations...)
	return policy
}

func createAndWaitReady(t *testing.T, client *clientset.Clientset, binding *admissionregistrationv1alpha1.ValidatingAdmissionPolicyBinding, matchLabels map[string]string) error {
	marker := &v1.Endpoints{ObjectMeta: metav1.ObjectMeta{Name: "test-marker", Namespace: "default", Labels: matchLabels}}
	defer func() {
		err := client.CoreV1().Endpoints("default").Delete(context.TODO(), marker.Name, metav1.DeleteOptions{})
		if err != nil {
			t.Logf("error deleting marker: %v", err)
		}
	}()
	marker, err := client.CoreV1().Endpoints("default").Create(context.TODO(), marker, metav1.CreateOptions{})
	if err != nil {
		return err
	}

	_, err = client.AdmissionregistrationV1alpha1().ValidatingAdmissionPolicyBindings().Create(context.TODO(), binding, metav1.CreateOptions{})
	if err != nil {
		return err
	}

	if waitErr := wait.PollImmediate(time.Millisecond*5, wait.ForeverTestTimeout, func() (bool, error) {
		_, err := client.CoreV1().Endpoints("default").Patch(context.TODO(), marker.Name, types.JSONPatchType, []byte("[]"), metav1.PatchOptions{})
		if err != nil && strings.Contains(err.Error(), "marker denied; policy is ready") {
			return true, nil
		} else {
			t.Logf("waiting for policy to be ready. Marker: %v, Last marker patch response: %v", marker, err)
			return false, err
		}
	}); waitErr != nil {
		return waitErr
	}
	return nil
}

func makePolicy(name string) *admissionregistrationv1alpha1.ValidatingAdmissionPolicy {
	return &admissionregistrationv1alpha1.ValidatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{Name: name},
	}
}

func withParams(params *admissionregistrationv1alpha1.ParamKind, policy *admissionregistrationv1alpha1.ValidatingAdmissionPolicy) *admissionregistrationv1alpha1.ValidatingAdmissionPolicy {
	policy.Spec.ParamKind = params
	return policy
}

func configParamKind() *admissionregistrationv1alpha1.ParamKind {
	return &admissionregistrationv1alpha1.ParamKind{
		APIVersion: "v1",
		Kind:       "ConfigMap",
	}
}

func withFailurePolicy(failure admissionregistrationv1alpha1.FailurePolicyType, policy *admissionregistrationv1alpha1.ValidatingAdmissionPolicy) *admissionregistrationv1alpha1.ValidatingAdmissionPolicy {
	policy.Spec.FailurePolicy = &failure
	return policy
}

func withNamespaceMatch(policy *admissionregistrationv1alpha1.ValidatingAdmissionPolicy) *admissionregistrationv1alpha1.ValidatingAdmissionPolicy {
	return withPolicyMatch("namespaces", policy)
}

func withConfigMapMatch(policy *admissionregistrationv1alpha1.ValidatingAdmissionPolicy) *admissionregistrationv1alpha1.ValidatingAdmissionPolicy {
	return withPolicyMatch("configmaps", policy)
}

func withObjectSelector(labelSelector *metav1.LabelSelector, policy *admissionregistrationv1alpha1.ValidatingAdmissionPolicy) *admissionregistrationv1alpha1.ValidatingAdmissionPolicy {
	policy.Spec.MatchConstraints.ObjectSelector = labelSelector
	return policy
}

func withNamespaceSelector(labelSelector *metav1.LabelSelector, policy *admissionregistrationv1alpha1.ValidatingAdmissionPolicy) *admissionregistrationv1alpha1.ValidatingAdmissionPolicy {
	policy.Spec.MatchConstraints.NamespaceSelector = labelSelector
	return policy
}

func withPolicyMatch(resource string, policy *admissionregistrationv1alpha1.ValidatingAdmissionPolicy) *admissionregistrationv1alpha1.ValidatingAdmissionPolicy {
	policy.Spec.MatchConstraints = &admissionregistrationv1alpha1.MatchResources{
		ResourceRules: []admissionregistrationv1alpha1.NamedRuleWithOperations{
			{
				RuleWithOperations: admissionregistrationv1alpha1.RuleWithOperations{
					Operations: []admissionregistrationv1.OperationType{
						"*",
					},
					Rule: admissionregistrationv1.Rule{
						APIGroups: []string{
							"",
						},
						APIVersions: []string{
							"*",
						},
						Resources: []string{
							resource,
						},
					},
				},
			},
		},
	}
	return policy
}

func withExcludePolicyMatch(resource string, policy *admissionregistrationv1alpha1.ValidatingAdmissionPolicy) *admissionregistrationv1alpha1.ValidatingAdmissionPolicy {
	policy.Spec.MatchConstraints.ExcludeResourceRules = []admissionregistrationv1alpha1.NamedRuleWithOperations{
		{
			RuleWithOperations: admissionregistrationv1alpha1.RuleWithOperations{
				Operations: []admissionregistrationv1.OperationType{
					"*",
				},
				Rule: admissionregistrationv1.Rule{
					APIGroups: []string{
						"",
					},
					APIVersions: []string{
						"*",
					},
					Resources: []string{
						resource,
					},
				},
			},
		},
	}
	return policy
}

func withPolicyExistsLabels(labels []string, policy *admissionregistrationv1alpha1.ValidatingAdmissionPolicy) *admissionregistrationv1alpha1.ValidatingAdmissionPolicy {
	if policy.Spec.MatchConstraints == nil {
		policy.Spec.MatchConstraints = &admissionregistrationv1alpha1.MatchResources{}
	}
	matchExprs := buildExistsSelector(labels)
	policy.Spec.MatchConstraints.ObjectSelector = &metav1.LabelSelector{
		MatchExpressions: matchExprs,
	}
	return policy
}

func withValidations(validations []admissionregistrationv1alpha1.Validation, policy *admissionregistrationv1alpha1.ValidatingAdmissionPolicy) *admissionregistrationv1alpha1.ValidatingAdmissionPolicy {
	policy.Spec.Validations = validations
	return policy
}

func makeBinding(name, policyName, paramName string) *admissionregistrationv1alpha1.ValidatingAdmissionPolicyBinding {
	var paramRef *admissionregistrationv1alpha1.ParamRef
	if paramName != "" {
		paramRef = &admissionregistrationv1alpha1.ParamRef{
			Name:      paramName,
			Namespace: "default",
		}
	}
	return &admissionregistrationv1alpha1.ValidatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: admissionregistrationv1alpha1.ValidatingAdmissionPolicyBindingSpec{
			PolicyName: policyName,
			ParamRef:   paramRef,
		},
	}
}

func withBindingExistsLabels(labels []string, policy *admissionregistrationv1alpha1.ValidatingAdmissionPolicy, binding *admissionregistrationv1alpha1.ValidatingAdmissionPolicyBinding) *admissionregistrationv1alpha1.ValidatingAdmissionPolicyBinding {
	if policy != nil {
		// shallow copy
		constraintsCopy := *policy.Spec.MatchConstraints
		binding.Spec.MatchResources = &constraintsCopy
	}
	matchExprs := buildExistsSelector(labels)
	binding.Spec.MatchResources.ObjectSelector = &metav1.LabelSelector{
		MatchExpressions: matchExprs,
	}
	return binding
}

func buildExistsSelector(labels []string) []metav1.LabelSelectorRequirement {
	matchExprs := make([]metav1.LabelSelectorRequirement, len(labels))
	for i := 0; i < len(labels); i++ {
		matchExprs[i].Key = labels[i]
		matchExprs[i].Operator = metav1.LabelSelectorOpExists
	}
	return matchExprs
}

func makeConfigParams(name string, data map[string]string) *v1.ConfigMap {
	return &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Data:       data,
	}
}

func checkForFailedRule(t *testing.T, err error) {
	if !strings.Contains(err.Error(), "failed expression") {
		t.Fatalf("unexpected error (expected to find \"failed expression\"): %s", err)
	}
	if strings.Contains(err.Error(), "evaluation error") {
		t.Fatalf("CEL rule evaluation failed: %s", err)
	}
}

func checkFailureReason(t *testing.T, err error, expectedReason metav1.StatusReason) {
	reason := err.(apierrors.APIStatus).Status().Reason
	if reason != expectedReason {
		t.Logf("actual error reason: %v", reason)
		t.Logf("expected failure reason: %v", expectedReason)
		t.Error("unexpected error reason")
	}
}

// Copied from etcd.GetCustomResourceDefinitionData
func versionedCustomResourceDefinition() *apiextensionsv1.CustomResourceDefinition {
	return &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pandas.awesome.bears.com",
		},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: "awesome.bears.com",
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
				{
					Name:    "v1",
					Served:  true,
					Storage: true,
					Schema:  fixtures.AllowAllSchema(),
					Subresources: &apiextensionsv1.CustomResourceSubresources{
						Status: &apiextensionsv1.CustomResourceSubresourceStatus{},
						Scale: &apiextensionsv1.CustomResourceSubresourceScale{
							SpecReplicasPath:   ".spec.replicas",
							StatusReplicasPath: ".status.replicas",
							LabelSelectorPath:  func() *string { path := ".status.selector"; return &path }(),
						},
					},
				},
				{
					Name:    "v2",
					Served:  true,
					Storage: false,
					Schema:  fixtures.AllowAllSchema(),
					Subresources: &apiextensionsv1.CustomResourceSubresources{
						Status: &apiextensionsv1.CustomResourceSubresourceStatus{},
						Scale: &apiextensionsv1.CustomResourceSubresourceScale{
							SpecReplicasPath:   ".spec.replicas",
							StatusReplicasPath: ".status.replicas",
							LabelSelectorPath:  func() *string { path := ".status.selector"; return &path }(),
						},
					},
				},
			},
			Scope: apiextensionsv1.ClusterScoped,
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Plural: "pandas",
				Kind:   "Panda",
			},
		},
	}
}
