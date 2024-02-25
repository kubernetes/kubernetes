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

package apimachinery

import (
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	admissionregistrationv1beta1 "k8s.io/api/admissionregistration/v1beta1"
	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionsclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/features"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/openapi3"
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("ValidatingAdmissionPolicy [Privileged:ClusterAdmin]", framework.WithFeatureGate(features.ValidatingAdmissionPolicy), func() {
	f := framework.NewDefaultFramework("validating-admission-policy")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	var client clientset.Interface
	var extensionsClient apiextensionsclientset.Interface

	ginkgo.BeforeEach(func() {
		var err error
		client, err = clientset.NewForConfig(f.ClientConfig())
		framework.ExpectNoError(err, "initializing client")
		_, err = client.AdmissionregistrationV1beta1().ValidatingAdmissionPolicies().List(context.Background(), metav1.ListOptions{})
		if apierrors.IsNotFound(err) {
			// TODO: feature check should fail after GA graduation
			ginkgo.Skip(fmt.Sprintf("server does not support ValidatingAdmissionPolicy v1beta1: %v, feature gate not enabled?", err))
		}
		extensionsClient, err = apiextensionsclientset.NewForConfig(f.ClientConfig())
		framework.ExpectNoError(err, "initializing api-extensions client")
	})

	ginkgo.BeforeEach(func(ctx context.Context) {
		// Make sure the namespace created for the test is labeled to be selected
		// in binding.spec.matchResources.namespaceSelector.matchLabels
		// By containing the tests within the marked namespace, they will not
		// disturb concurrent tests that run in other namespaces.
		labelNamespace(ctx, f, f.Namespace.Name)
	})

	ginkgo.It("should validate against a Deployment", func(ctx context.Context) {
		ginkgo.By("creating the policy", func() {
			policy := newValidatingAdmissionPolicyBuilder(f.UniqueName+".policy.example.com").
				MatchUniqueNamespace(f.UniqueName).
				StartResourceRule().
				MatchResource([]string{"apps"}, []string{"v1"}, []string{"deployments"}).
				EndResourceRule().
				WithValidation(admissionregistrationv1beta1.Validation{
					Expression:        "object.spec.replicas > 1",
					MessageExpression: "'wants replicas > 1, got ' + object.spec.replicas",
				}).
				WithValidation(admissionregistrationv1beta1.Validation{
					Expression: "namespaceObject.metadata.name == '" + f.UniqueName + "'",
					Message:    "Internal error! Other namespace should not be allowed.",
				}).
				Build()
			policy, err := client.AdmissionregistrationV1beta1().ValidatingAdmissionPolicies().Create(ctx, policy, metav1.CreateOptions{})
			framework.ExpectNoError(err, "create policy")
			ginkgo.DeferCleanup(func(ctx context.Context, name string) error {
				return client.AdmissionregistrationV1beta1().ValidatingAdmissionPolicies().Delete(ctx, name, metav1.DeleteOptions{})
			}, policy.Name)
			binding := createBinding(f.UniqueName+".binding.example.com", f.UniqueName, policy.Name)
			binding, err = client.AdmissionregistrationV1beta1().ValidatingAdmissionPolicyBindings().Create(ctx, binding, metav1.CreateOptions{})
			framework.ExpectNoError(err, "create policy binding")
			ginkgo.DeferCleanup(func(ctx context.Context, name string) error {
				return client.AdmissionregistrationV1beta1().ValidatingAdmissionPolicyBindings().Delete(ctx, name, metav1.DeleteOptions{})
			}, binding.Name)
		})
		ginkgo.By("waiting until the marker is denied", func() {
			deployment := basicDeployment("marker-deployment", 1)
			err := wait.PollUntilContextCancel(ctx, 100*time.Millisecond, true, func(ctx context.Context) (done bool, err error) {
				_, err = client.AppsV1().Deployments(f.Namespace.Name).Create(ctx, deployment, metav1.CreateOptions{})
				defer client.AppsV1().Deployments(f.Namespace.Name).Delete(ctx, deployment.Name, metav1.DeleteOptions{})
				if err != nil {
					if apierrors.IsInvalid(err) {
						return true, nil
					}
					return false, err
				}
				return false, nil
			})
			framework.ExpectNoError(err, "wait for marker")
		})
		ginkgo.By("testing a replicated Deployment to be allowed", func() {
			deployment := basicDeployment("replicated", 2)
			deployment, err := client.AppsV1().Deployments(f.Namespace.Name).Create(ctx, deployment, metav1.CreateOptions{})
			defer client.AppsV1().Deployments(f.Namespace.Name).Delete(ctx, deployment.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "create replicated Deployment")
		})
		ginkgo.By("testing a non-replicated ReplicaSet not to be denied", func() {
			replicaSet := basicReplicaSet("non-replicated", 1)
			replicaSet, err := client.AppsV1().ReplicaSets(f.Namespace.Name).Create(ctx, replicaSet, metav1.CreateOptions{})
			defer client.AppsV1().ReplicaSets(f.Namespace.Name).Delete(ctx, replicaSet.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "create non-replicated ReplicaSet")
		})
	})

	ginkgo.It("should type check validation expressions", func(ctx context.Context) {
		var policy *admissionregistrationv1beta1.ValidatingAdmissionPolicy
		ginkgo.By("creating the policy with correct types", func() {
			policy = newValidatingAdmissionPolicyBuilder(f.UniqueName+".correct-policy.example.com").
				MatchUniqueNamespace(f.UniqueName).
				StartResourceRule().
				MatchResource([]string{"apps"}, []string{"v1"}, []string{"deployments"}).
				EndResourceRule().
				WithValidation(admissionregistrationv1beta1.Validation{
					Expression: "object.spec.replicas > 1",
				}).
				Build()
			var err error
			policy, err = client.AdmissionregistrationV1beta1().ValidatingAdmissionPolicies().Create(ctx, policy, metav1.CreateOptions{})
			framework.ExpectNoError(err, "create policy")
			ginkgo.DeferCleanup(func(ctx context.Context, name string) error {
				return client.AdmissionregistrationV1beta1().ValidatingAdmissionPolicies().Delete(ctx, name, metav1.DeleteOptions{})
			}, policy.Name)
		})
		ginkgo.By("waiting for the type check to finish without any warnings", func() {
			err := wait.PollUntilContextCancel(ctx, 100*time.Millisecond, true, func(ctx context.Context) (done bool, err error) {
				policy, err = client.AdmissionregistrationV1beta1().ValidatingAdmissionPolicies().Get(ctx, policy.Name, metav1.GetOptions{})
				if err != nil {
					return false, err
				}
				if policy.Status.TypeChecking != nil { // non-nil TypeChecking indicates its completion
					return true, nil
				}
				return false, nil
			})
			framework.ExpectNoError(err, "wait for type checking")
			gomega.Expect(policy.Status.TypeChecking.ExpressionWarnings).To(gomega.BeEmpty())
		})
		ginkgo.By("creating the policy with type confusion", func() {
			policy = newValidatingAdmissionPolicyBuilder(f.UniqueName+".confused-policy.example.com").
				MatchUniqueNamespace(f.UniqueName).
				StartResourceRule().
				MatchResource([]string{"apps"}, []string{"v1"}, []string{"deployments"}).
				EndResourceRule().
				WithValidation(admissionregistrationv1beta1.Validation{
					Expression:        "object.spec.replicas > '1'",                        // confusion: int > string
					MessageExpression: "'wants replicas > 1, got ' + object.spec.replicas", // confusion: string + int
				}).
				Build()
			var err error
			policy, err = client.AdmissionregistrationV1beta1().ValidatingAdmissionPolicies().Create(ctx, policy, metav1.CreateOptions{})
			framework.ExpectNoError(err, "create policy")
			ginkgo.DeferCleanup(func(ctx context.Context, name string) error {
				return client.AdmissionregistrationV1beta1().ValidatingAdmissionPolicies().Delete(ctx, name, metav1.DeleteOptions{})
			}, policy.Name)
		})
		ginkgo.By("waiting for the type check to finish with warnings", func() {
			err := wait.PollUntilContextCancel(ctx, 100*time.Millisecond, true, func(ctx context.Context) (done bool, err error) {
				policy, err = client.AdmissionregistrationV1beta1().ValidatingAdmissionPolicies().Get(ctx, policy.Name, metav1.GetOptions{})
				if err != nil {
					return false, err
				}
				if policy.Status.TypeChecking != nil { // non-nil TypeChecking indicates its completion
					return true, nil
				}
				return false, nil
			})
			framework.ExpectNoError(err, "wait for type checking")

			// assert it to contain 2 warnings, first for expression and second for messageExpression
			gomega.Expect(policy.Status.TypeChecking.ExpressionWarnings).To(gomega.HaveLen(2))
			warning := policy.Status.TypeChecking.ExpressionWarnings[0]
			gomega.Expect(warning.FieldRef).To(gomega.Equal("spec.validations[0].expression"))
			gomega.Expect(warning.Warning).To(gomega.ContainSubstring("found no matching overload for '_>_' applied to '(int, string)'"))
			warning = policy.Status.TypeChecking.ExpressionWarnings[1]
			gomega.Expect(warning.FieldRef).To(gomega.Equal("spec.validations[0].messageExpression"))
			gomega.Expect(warning.Warning).To(gomega.ContainSubstring("found no matching overload for '_+_' applied to '(string, int)'"))
		})
	})

	ginkgo.It("should allow expressions to refer variables.", func(ctx context.Context) {
		ginkgo.By("creating a policy with variables", func() {
			policy := newValidatingAdmissionPolicyBuilder(f.UniqueName+".policy.example.com").
				MatchUniqueNamespace(f.UniqueName).
				StartResourceRule().
				MatchResource([]string{"apps"}, []string{"v1"}, []string{"deployments"}).
				EndResourceRule().
				WithVariable(admissionregistrationv1beta1.Variable{
					Name:       "replicas",
					Expression: "object.spec.replicas",
				}).
				WithVariable(admissionregistrationv1beta1.Variable{
					Name:       "oddReplicas",
					Expression: "variables.replicas % 2 == 1",
				}).
				WithValidation(admissionregistrationv1beta1.Validation{
					Expression: "variables.replicas > 1",
				}).
				WithValidation(admissionregistrationv1beta1.Validation{
					Expression: "variables.oddReplicas",
				}).
				Build()
			policy, err := client.AdmissionregistrationV1beta1().ValidatingAdmissionPolicies().Create(ctx, policy, metav1.CreateOptions{})
			framework.ExpectNoError(err, "create policy")
			ginkgo.DeferCleanup(func(ctx context.Context, name string) error {
				return client.AdmissionregistrationV1beta1().ValidatingAdmissionPolicies().Delete(ctx, name, metav1.DeleteOptions{})
			}, policy.Name)
			binding := createBinding(f.UniqueName+".binding.example.com", f.UniqueName, policy.Name)
			binding, err = client.AdmissionregistrationV1beta1().ValidatingAdmissionPolicyBindings().Create(ctx, binding, metav1.CreateOptions{})
			framework.ExpectNoError(err, "create policy binding")
			ginkgo.DeferCleanup(func(ctx context.Context, name string) error {
				return client.AdmissionregistrationV1beta1().ValidatingAdmissionPolicyBindings().Delete(ctx, name, metav1.DeleteOptions{})
			}, binding.Name)
		})
		ginkgo.By("waiting until the marker is denied", func() {
			deployment := basicDeployment("marker-deployment", 1)
			err := wait.PollUntilContextCancel(ctx, 100*time.Millisecond, true, func(ctx context.Context) (done bool, err error) {
				_, err = client.AppsV1().Deployments(f.Namespace.Name).Create(ctx, deployment, metav1.CreateOptions{})
				defer client.AppsV1().Deployments(f.Namespace.Name).Delete(ctx, deployment.Name, metav1.DeleteOptions{})
				if err != nil {
					if apierrors.IsInvalid(err) {
						return true, nil
					}
					return false, err
				}
				return false, nil
			})
			framework.ExpectNoError(err, "wait for marker")
		})
		ginkgo.By("testing a replicated Deployment to be allowed", func() {
			deployment := basicDeployment("replicated", 3)
			deployment, err := client.AppsV1().Deployments(f.Namespace.Name).Create(ctx, deployment, metav1.CreateOptions{})
			defer client.AppsV1().Deployments(f.Namespace.Name).Delete(ctx, deployment.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "create replicated Deployment")
		})
		ginkgo.By("testing a non-replicated ReplicaSet not to be denied", func() {
			replicaSet := basicReplicaSet("non-replicated", 1)
			replicaSet, err := client.AppsV1().ReplicaSets(f.Namespace.Name).Create(ctx, replicaSet, metav1.CreateOptions{})
			defer client.AppsV1().ReplicaSets(f.Namespace.Name).Delete(ctx, replicaSet.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "create non-replicated ReplicaSet")
		})
	})

	ginkgo.It("should type check a CRD", func(ctx context.Context) {
		crd := crontabExampleCRD()
		crd.Spec.Group = "stable." + f.UniqueName
		crd.Name = crd.Spec.Names.Plural + "." + crd.Spec.Group
		var policy *admissionregistrationv1beta1.ValidatingAdmissionPolicy
		ginkgo.By("creating the CRD", func() {
			var err error
			crd, err = extensionsClient.ApiextensionsV1().CustomResourceDefinitions().Create(ctx, crd, metav1.CreateOptions{})
			framework.ExpectNoError(err, "create CRD")
			err = wait.PollUntilContextCancel(ctx, 100*time.Millisecond, true, func(ctx context.Context) (done bool, err error) {
				// wait for the CRD to be published.
				root := openapi3.NewRoot(client.Discovery().OpenAPIV3())
				_, err = root.GVSpec(schema.GroupVersion{Group: crd.Spec.Group, Version: "v1"})
				return err == nil, nil
			})
			framework.ExpectNoError(err, "wait for CRD.")
			ginkgo.DeferCleanup(func(ctx context.Context, name string) error {
				return extensionsClient.ApiextensionsV1().CustomResourceDefinitions().Delete(ctx, name, metav1.DeleteOptions{})
			}, crd.Name)
		})
		ginkgo.By("creating a vaild policy for crontabs", func() {
			policy = newValidatingAdmissionPolicyBuilder(f.UniqueName+".correct-crd-policy.example.com").
				MatchUniqueNamespace(f.UniqueName).
				StartResourceRule().
				MatchResource([]string{crd.Spec.Group}, []string{"v1"}, []string{"crontabs"}).
				EndResourceRule().
				WithValidation(admissionregistrationv1beta1.Validation{
					Expression: "object.spec.replicas > 1",
				}).
				Build()
			policy, err := client.AdmissionregistrationV1beta1().ValidatingAdmissionPolicies().Create(ctx, policy, metav1.CreateOptions{})
			framework.ExpectNoError(err, "create policy")
			ginkgo.DeferCleanup(func(ctx context.Context, name string) error {
				return client.AdmissionregistrationV1beta1().ValidatingAdmissionPolicies().Delete(ctx, name, metav1.DeleteOptions{})
			}, policy.Name)
		})
		ginkgo.By("waiting for the type check to finish without warnings", func() {
			err := wait.PollUntilContextCancel(ctx, 100*time.Millisecond, true, func(ctx context.Context) (done bool, err error) {
				policy, err = client.AdmissionregistrationV1beta1().ValidatingAdmissionPolicies().Get(ctx, policy.Name, metav1.GetOptions{})
				if err != nil {
					return false, err
				}
				if policy.Status.TypeChecking != nil {
					return true, nil
				}
				return false, nil
			})
			framework.ExpectNoError(err, "wait for type checking")
			gomega.Expect(policy.Status.TypeChecking.ExpressionWarnings).To(gomega.BeEmpty(), "expect no warnings")
		})
		ginkgo.By("creating a policy with type-confused expressions for crontabs", func() {
			policy = newValidatingAdmissionPolicyBuilder(f.UniqueName+".confused-crd-policy.example.com").
				MatchUniqueNamespace(f.UniqueName).
				StartResourceRule().
				MatchResource([]string{crd.Spec.Group}, []string{"v1"}, []string{"crontabs"}).
				EndResourceRule().
				WithValidation(admissionregistrationv1beta1.Validation{
					Expression: "object.spec.replicas > '1'", // type confusion
				}).
				WithValidation(admissionregistrationv1beta1.Validation{
					Expression: "object.spec.maxRetries < 10", // not yet existing field
				}).
				Build()
			policy, err := client.AdmissionregistrationV1beta1().ValidatingAdmissionPolicies().Create(ctx, policy, metav1.CreateOptions{})
			framework.ExpectNoError(err, "create policy")
			ginkgo.DeferCleanup(func(ctx context.Context, name string) error {
				return client.AdmissionregistrationV1beta1().ValidatingAdmissionPolicies().Delete(ctx, name, metav1.DeleteOptions{})
			}, policy.Name)
		})
		ginkgo.By("waiting for the type check to finish with warnings", func() {
			err := wait.PollUntilContextCancel(ctx, 100*time.Millisecond, true, func(ctx context.Context) (done bool, err error) {
				policy, err = client.AdmissionregistrationV1beta1().ValidatingAdmissionPolicies().Get(ctx, policy.Name, metav1.GetOptions{})
				if err != nil {
					return false, err
				}
				if policy.Status.TypeChecking != nil {
					return true, nil
				}
				return false, nil
			})
			framework.ExpectNoError(err, "wait for type checking")

			gomega.Expect(policy.Status.TypeChecking.ExpressionWarnings).To(gomega.HaveLen(2))
			warning := policy.Status.TypeChecking.ExpressionWarnings[0]
			gomega.Expect(warning.FieldRef).To(gomega.Equal("spec.validations[0].expression"))
			gomega.Expect(warning.Warning).To(gomega.ContainSubstring("found no matching overload for '_>_' applied to '(int, string)'"))
			warning = policy.Status.TypeChecking.ExpressionWarnings[1]
			gomega.Expect(warning.FieldRef).To(gomega.Equal("spec.validations[1].expression"))
			gomega.Expect(warning.Warning).To(gomega.ContainSubstring("undefined field 'maxRetries'"))
		})
	})
})

func createBinding(bindingName string, uniqueLabel string, policyName string) *admissionregistrationv1beta1.ValidatingAdmissionPolicyBinding {
	return &admissionregistrationv1beta1.ValidatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{Name: bindingName},
		Spec: admissionregistrationv1beta1.ValidatingAdmissionPolicyBindingSpec{
			PolicyName: policyName,
			MatchResources: &admissionregistrationv1beta1.MatchResources{
				NamespaceSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{uniqueLabel: "true"},
				},
			},
			ValidationActions: []admissionregistrationv1beta1.ValidationAction{admissionregistrationv1beta1.Deny},
		},
	}
}

func basicDeployment(name string, replicas int32) *appsv1.Deployment {
	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: map[string]string{"app": "nginx"},
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"app": "nginx"},
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"app": "nginx"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "nginx",
							Image: "nginx:latest",
						},
					},
				},
			},
		}}
}

func basicReplicaSet(name string, replicas int32) *appsv1.ReplicaSet {
	return &appsv1.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: map[string]string{"app": "nginx"},
		},
		Spec: appsv1.ReplicaSetSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"app": "nginx"},
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"app": "nginx"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "nginx",
							Image: "nginx:latest",
						},
					},
				},
			},
		}}
}

type validatingAdmissionPolicyBuilder struct {
	policy *admissionregistrationv1beta1.ValidatingAdmissionPolicy
}

type resourceRuleBuilder struct {
	policyBuilder *validatingAdmissionPolicyBuilder
	resourceRule  *admissionregistrationv1beta1.NamedRuleWithOperations
}

func newValidatingAdmissionPolicyBuilder(policyName string) *validatingAdmissionPolicyBuilder {
	return &validatingAdmissionPolicyBuilder{
		policy: &admissionregistrationv1beta1.ValidatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{Name: policyName},
		},
	}
}

func (b *validatingAdmissionPolicyBuilder) MatchUniqueNamespace(uniqueLabel string) *validatingAdmissionPolicyBuilder {
	if b.policy.Spec.MatchConstraints == nil {
		b.policy.Spec.MatchConstraints = &admissionregistrationv1beta1.MatchResources{}
	}
	b.policy.Spec.MatchConstraints.NamespaceSelector = &metav1.LabelSelector{
		MatchLabels: map[string]string{
			uniqueLabel: "true",
		},
	}
	return b
}

func (b *validatingAdmissionPolicyBuilder) StartResourceRule() *resourceRuleBuilder {
	return &resourceRuleBuilder{
		policyBuilder: b,
		resourceRule: &admissionregistrationv1beta1.NamedRuleWithOperations{
			RuleWithOperations: admissionregistrationv1beta1.RuleWithOperations{
				Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create, admissionregistrationv1.Update},
				Rule: admissionregistrationv1beta1.Rule{
					APIGroups:   []string{"apps"},
					APIVersions: []string{"v1"},
					Resources:   []string{"deployments"},
				},
			},
		},
	}
}

func (rb *resourceRuleBuilder) CreateAndUpdate() *resourceRuleBuilder {
	rb.resourceRule.Operations = []admissionregistrationv1.OperationType{admissionregistrationv1.Create, admissionregistrationv1.Update}
	return rb
}

func (rb *resourceRuleBuilder) MatchResource(groups []string, versions []string, resources []string) *resourceRuleBuilder {
	rb.resourceRule.Rule = admissionregistrationv1beta1.Rule{
		APIGroups:   groups,
		APIVersions: versions,
		Resources:   resources,
	}
	return rb
}

func (rb *resourceRuleBuilder) EndResourceRule() *validatingAdmissionPolicyBuilder {
	b := rb.policyBuilder
	if b.policy.Spec.MatchConstraints == nil {
		b.policy.Spec.MatchConstraints = &admissionregistrationv1beta1.MatchResources{}
	}
	b.policy.Spec.MatchConstraints.ResourceRules = append(b.policy.Spec.MatchConstraints.ResourceRules, *rb.resourceRule)
	return b
}

func (b *validatingAdmissionPolicyBuilder) WithValidation(validation admissionregistrationv1beta1.Validation) *validatingAdmissionPolicyBuilder {
	b.policy.Spec.Validations = append(b.policy.Spec.Validations, validation)
	return b
}

func (b *validatingAdmissionPolicyBuilder) WithVariable(variable admissionregistrationv1beta1.Variable) *validatingAdmissionPolicyBuilder {
	b.policy.Spec.Variables = append(b.policy.Spec.Variables, variable)
	return b
}

func (b *validatingAdmissionPolicyBuilder) Build() *admissionregistrationv1beta1.ValidatingAdmissionPolicy {
	return b.policy
}

func crontabExampleCRD() *apiextensionsv1.CustomResourceDefinition {
	return &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: "crontabs.stable.example.com",
		},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: "stable.example.com",
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
				{
					Name:    "v1",
					Served:  true,
					Storage: true,
					Schema: &apiextensionsv1.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensionsv1.JSONSchemaProps{
								"spec": {
									Type: "object",
									Properties: map[string]apiextensionsv1.JSONSchemaProps{
										"cronSpec": {
											Type: "string",
										},
										"image": {
											Type: "string",
										},
										"replicas": {
											Type: "integer",
										},
									},
								},
							},
						}},
				},
			},
			Scope: apiextensionsv1.NamespaceScoped,
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Plural:     "crontabs",
				Singular:   "crontab",
				Kind:       "CronTab",
				ShortNames: []string{"ct"},
			},
		},
	}
}
