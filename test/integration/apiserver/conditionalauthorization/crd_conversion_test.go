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

package conditionalauthorization

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionsclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
)

func createAndWaitForCRD(t *testing.T, server *kubeapiservertesting.TestServer, conversionClientConfig *apiextensionsv1.WebhookClientConfig) {
	// Create a CRD with two versions for multi-version conditional authorization tests.
	// v1 (storage version) has spec.replicas as an integer.
	// v2 has spec.replicas as an object with a "max" integer field.
	apiExtClient, err := apiextensionsclientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatal(err)
	}
	crdDef := &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: "scalablewidgets.example.com",
		},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: "example.com",
			Scope: apiextensionsv1.NamespaceScoped,
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Plural:   "scalablewidgets",
				Singular: "scalablewidget",
				Kind:     "ScalableWidget",
				ListKind: "ScalableWidgetList",
			},
			Conversion: &apiextensionsv1.CustomResourceConversion{
				Strategy: apiextensionsv1.WebhookConverter,
				Webhook: &apiextensionsv1.WebhookConversion{
					ClientConfig:             conversionClientConfig,
					ConversionReviewVersions: []string{"v1"},
				},
			},
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
										"replicas": {Type: "integer"},
									},
								},
							},
						},
					},
				},
				{
					Name:    "v2",
					Served:  true,
					Storage: false,
					Schema: &apiextensionsv1.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensionsv1.JSONSchemaProps{
								"spec": {
									Type: "object",
									Properties: map[string]apiextensionsv1.JSONSchemaProps{
										"replicas": {
											Type: "object",
											Properties: map[string]apiextensionsv1.JSONSchemaProps{
												"max": {Type: "integer"},
											},
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}
	if _, err := apiExtClient.ApiextensionsV1().CustomResourceDefinitions().Create(context.TODO(), crdDef, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	if err := wait.PollUntilContextTimeout(context.TODO(), 500*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		crd, err := apiExtClient.ApiextensionsV1().CustomResourceDefinitions().Get(ctx, crdDef.Name, metav1.GetOptions{})
		if err != nil {
			return false, nil
		}
		for _, cond := range crd.Status.Conditions {
			if cond.Type == apiextensionsv1.Established && cond.Status == apiextensionsv1.ConditionTrue {
				return true, nil
			}
		}
		return false, nil
	}); err != nil {
		t.Fatalf("timed out waiting for CRD %s to be established: %v", crdDef.Name, err)
	}
}

// crdReplicasDecision is a decisionFunc for the ScalableWidget CRD test cases.
// It emits a ConditionsMap that only allows updates/creates whose replicas is
// convertScalableWidget converts a single ScalableWidget custom resource between
// v1 (spec.replicas: integer) and v2 (spec.replicas: {max: integer})
// representations for use as an ObjectConverterFunc in the CRD conversion webhook.
// Same-version passes through unchanged; missing replicas is propagated as an
// unset field so round-trips stay lossless for the shapes the tests exercise.
func convertScalableWidget(desiredAPIVersion string, in runtime.RawExtension) (runtime.RawExtension, error) {
	obj := &unstructured.Unstructured{}
	if _, _, err := unstructured.UnstructuredJSONScheme.Decode(in.Raw, nil, obj); err != nil {
		return runtime.RawExtension{}, fmt.Errorf("decode CR: %w", err)
	}
	currentAPIVersion := obj.GetAPIVersion()

	switch {
	case currentAPIVersion == desiredAPIVersion:
		// Same-version request: no field transformation needed.
	case currentAPIVersion == "example.com/v1" && desiredAPIVersion == "example.com/v2":
		replicas, found, err := unstructured.NestedInt64(obj.Object, "spec", "replicas")
		if err != nil {
			return runtime.RawExtension{}, fmt.Errorf("v1→v2: read spec.replicas: %w", err)
		}
		if found {
			if err := unstructured.SetNestedField(obj.Object, map[string]interface{}{"max": replicas}, "spec", "replicas"); err != nil {
				return runtime.RawExtension{}, fmt.Errorf("v1→v2: set spec.replicas: %w", err)
			}
		}
	case currentAPIVersion == "example.com/v2" && desiredAPIVersion == "example.com/v1":
		max, found, err := unstructured.NestedInt64(obj.Object, "spec", "replicas", "max")
		if err != nil {
			return runtime.RawExtension{}, fmt.Errorf("v2→v1: read spec.replicas.max: %w", err)
		}
		if found {
			if err := unstructured.SetNestedField(obj.Object, max, "spec", "replicas"); err != nil {
				return runtime.RawExtension{}, fmt.Errorf("v2→v1: set spec.replicas: %w", err)
			}
		} else {
			unstructured.RemoveNestedField(obj.Object, "spec", "replicas")
		}
	default:
		return runtime.RawExtension{}, fmt.Errorf("unsupported ScalableWidget conversion %s → %s", currentAPIVersion, desiredAPIVersion)
	}

	obj.SetAPIVersion(desiredAPIVersion)
	raw, err := json.Marshal(obj.Object)
	if err != nil {
		return runtime.RawExtension{}, fmt.Errorf("marshal CR: %w", err)
	}
	return runtime.RawExtension{Raw: raw}, nil
}

func crdTestCases(server *kubeapiservertesting.TestServer) []conditionalAuthzTestCase {
	return []conditionalAuthzTestCase{
		// Tests for a multi-version CRD (ScalableWidget) with version-specific schemas.
		// v1 has spec.replicas (integer), v2 has spec.replicas.max (integer in object).
		// The authorizer returns version-specific CEL conditions requiring replicas <= 10.
		// For updates, both the old and new objects must satisfy the condition.
		{
			name:        "crd v1 replicas - create allowed",
			user:        "alice-crd-v1-create-allow",
			authorizers: celConditionalAuthorizerVariants(crdReplicasDecision),
			makeRequest: func(t *testing.T, _ *clientset.Clientset, suffix string) error {
				config := rest.CopyConfig(server.ClientConfig)
				config.Impersonate.UserName = "alice-crd-v1-create-allow" + suffix
				dynClient, err := dynamic.NewForConfig(config)
				if err != nil {
					return err
				}
				gvr := schema.GroupVersionResource{Group: "example.com", Version: "v1", Resource: "scalablewidgets"}
				_, err = dynClient.Resource(gvr).Namespace("test-ns").Create(context.TODO(), &unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "example.com/v1",
						"kind":       "ScalableWidget",
						"metadata": map[string]interface{}{
							"name":      "v1-create-allow" + suffix,
							"namespace": "test-ns",
						},
						"spec": map[string]interface{}{
							"replicas": int64(5),
						},
					},
				}, metav1.CreateOptions{})
				return err
			},
			expectAllowed:             true,
			expectAllowedWhenDisabled: new(false),
		},
		{
			name:        "crd v1 replicas - create denied",
			user:        "alice-crd-v1-create-deny",
			authorizers: celConditionalAuthorizerVariants(crdReplicasDecision),
			makeRequest: func(t *testing.T, _ *clientset.Clientset, suffix string) error {
				config := rest.CopyConfig(server.ClientConfig)
				config.Impersonate.UserName = "alice-crd-v1-create-deny" + suffix
				dynClient, err := dynamic.NewForConfig(config)
				if err != nil {
					return err
				}
				gvr := schema.GroupVersionResource{Group: "example.com", Version: "v1", Resource: "scalablewidgets"}
				_, err = dynClient.Resource(gvr).Namespace("test-ns").Create(context.TODO(), &unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "example.com/v1",
						"kind":       "ScalableWidget",
						"metadata": map[string]interface{}{
							"name":      "v1-create-deny" + suffix,
							"namespace": "test-ns",
						},
						"spec": map[string]interface{}{
							"replicas": int64(15),
						},
					},
				}, metav1.CreateOptions{})
				return err
			},
			expectAllowed: false,
		},
		{
			name:        "crd v2 replicas.max - create allowed",
			user:        "alice-crd-v2-create-allow",
			authorizers: celConditionalAuthorizerVariants(crdReplicasDecision),
			makeRequest: func(t *testing.T, _ *clientset.Clientset, suffix string) error {
				config := rest.CopyConfig(server.ClientConfig)
				config.Impersonate.UserName = "alice-crd-v2-create-allow" + suffix
				dynClient, err := dynamic.NewForConfig(config)
				if err != nil {
					return err
				}
				gvr := schema.GroupVersionResource{Group: "example.com", Version: "v2", Resource: "scalablewidgets"}
				_, err = dynClient.Resource(gvr).Namespace("test-ns").Create(context.TODO(), &unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "example.com/v2",
						"kind":       "ScalableWidget",
						"metadata": map[string]interface{}{
							"name":      "v2-create-allow" + suffix,
							"namespace": "test-ns",
						},
						"spec": map[string]interface{}{
							"replicas": map[string]interface{}{
								"max": int64(5),
							},
						},
					},
				}, metav1.CreateOptions{})
				return err
			},
			expectAllowed:             true,
			expectAllowedWhenDisabled: new(false),
		},
		{
			name:        "crd v2 replicas.max - create denied",
			user:        "alice-crd-v2-create-deny",
			authorizers: celConditionalAuthorizerVariants(crdReplicasDecision),
			makeRequest: func(t *testing.T, _ *clientset.Clientset, suffix string) error {
				config := rest.CopyConfig(server.ClientConfig)
				config.Impersonate.UserName = "alice-crd-v2-create-deny" + suffix
				dynClient, err := dynamic.NewForConfig(config)
				if err != nil {
					return err
				}
				gvr := schema.GroupVersionResource{Group: "example.com", Version: "v2", Resource: "scalablewidgets"}
				_, err = dynClient.Resource(gvr).Namespace("test-ns").Create(context.TODO(), &unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "example.com/v2",
						"kind":       "ScalableWidget",
						"metadata": map[string]interface{}{
							"name":      "v2-create-deny" + suffix,
							"namespace": "test-ns",
						},
						"spec": map[string]interface{}{
							"replicas": map[string]interface{}{
								"max": int64(15),
							},
						},
					},
				}, metav1.CreateOptions{})
				return err
			},
			expectAllowed: false,
		},
		{
			name:        "crd v1 replicas - update allowed",
			user:        "alice-crd-v1-update-allow",
			authorizers: celConditionalAuthorizerVariants(crdReplicasDecision),
			makeRequest: func(t *testing.T, _ *clientset.Clientset, suffix string) error {
				config := rest.CopyConfig(server.ClientConfig)
				config.Impersonate.UserName = "alice-crd-v1-update-allow" + suffix
				dynClient, err := dynamic.NewForConfig(config)
				if err != nil {
					return err
				}
				gvr := schema.GroupVersionResource{Group: "example.com", Version: "v1", Resource: "scalablewidgets"}
				// Create with replicas=5 (allowed: 5 <= 10)
				created, err := dynClient.Resource(gvr).Namespace("test-ns").Create(context.TODO(), &unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "example.com/v1",
						"kind":       "ScalableWidget",
						"metadata": map[string]interface{}{
							"name":      "v1-update-allow" + suffix,
							"namespace": "test-ns",
						},
						"spec": map[string]interface{}{
							"replicas": int64(5),
						},
					},
				}, metav1.CreateOptions{})
				if err != nil {
					return fmt.Errorf("initial create should have succeeded: %w", err)
				}
				// Update to replicas=8 (allowed: new=8<=10 && old=5<=10)
				created.Object["spec"] = map[string]interface{}{"replicas": int64(8)}
				_, err = dynClient.Resource(gvr).Namespace("test-ns").Update(context.TODO(), created, metav1.UpdateOptions{})
				return err
			},
			expectAllowed:             true,
			expectAllowedWhenDisabled: new(false),
		},
		{
			name:        "crd v1 replicas - update denied",
			user:        "alice-crd-v1-update-deny",
			authorizers: celConditionalAuthorizerVariants(crdReplicasDecision),
			makeRequest: func(t *testing.T, _ *clientset.Clientset, suffix string) error {
				config := rest.CopyConfig(server.ClientConfig)
				config.Impersonate.UserName = "alice-crd-v1-update-deny" + suffix
				dynClient, err := dynamic.NewForConfig(config)
				if err != nil {
					return err
				}
				gvr := schema.GroupVersionResource{Group: "example.com", Version: "v1", Resource: "scalablewidgets"}
				// Create with replicas=5 (allowed: 5 <= 10)
				created, err := dynClient.Resource(gvr).Namespace("test-ns").Create(context.TODO(), &unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "example.com/v1",
						"kind":       "ScalableWidget",
						"metadata": map[string]interface{}{
							"name":      "v1-update-deny" + suffix,
							"namespace": "test-ns",
						},
						"spec": map[string]interface{}{
							"replicas": int64(5),
						},
					},
				}, metav1.CreateOptions{})
				if err != nil {
					return fmt.Errorf("initial create should have succeeded: %w", err)
				}
				// Update to replicas=15 (denied: new=15>10)
				created.Object["spec"] = map[string]interface{}{"replicas": int64(15)}
				_, err = dynClient.Resource(gvr).Namespace("test-ns").Update(context.TODO(), created, metav1.UpdateOptions{})
				return err
			},
			expectAllowed: false,
		},
		{
			name:        "crd v2 replicas.max - update allowed",
			user:        "alice-crd-v2-update-allow",
			authorizers: celConditionalAuthorizerVariants(crdReplicasDecision),
			makeRequest: func(t *testing.T, _ *clientset.Clientset, suffix string) error {
				config := rest.CopyConfig(server.ClientConfig)
				config.Impersonate.UserName = "alice-crd-v2-update-allow" + suffix
				dynClient, err := dynamic.NewForConfig(config)
				if err != nil {
					return err
				}
				gvr := schema.GroupVersionResource{Group: "example.com", Version: "v2", Resource: "scalablewidgets"}
				// Create with replicas.max=5 (allowed: 5 <= 10)
				created, err := dynClient.Resource(gvr).Namespace("test-ns").Create(context.TODO(), &unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "example.com/v2",
						"kind":       "ScalableWidget",
						"metadata": map[string]interface{}{
							"name":      "v2-update-allow" + suffix,
							"namespace": "test-ns",
						},
						"spec": map[string]interface{}{
							"replicas": map[string]interface{}{
								"max": int64(5),
							},
						},
					},
				}, metav1.CreateOptions{})
				if err != nil {
					return fmt.Errorf("initial create should have succeeded: %w", err)
				}
				// Update to replicas.max=8 (allowed: new=8<=10 && old=5<=10)
				created.Object["spec"] = map[string]interface{}{
					"replicas": map[string]interface{}{
						"max": int64(8),
					},
				}
				_, err = dynClient.Resource(gvr).Namespace("test-ns").Update(context.TODO(), created, metav1.UpdateOptions{})
				return err
			},
			expectAllowed:             true,
			expectAllowedWhenDisabled: new(false),
		},
		{
			name:        "crd v2 replicas.max - update denied",
			user:        "alice-crd-v2-update-deny",
			authorizers: celConditionalAuthorizerVariants(crdReplicasDecision),
			makeRequest: func(t *testing.T, _ *clientset.Clientset, suffix string) error {
				config := rest.CopyConfig(server.ClientConfig)
				config.Impersonate.UserName = "alice-crd-v2-update-deny" + suffix
				dynClient, err := dynamic.NewForConfig(config)
				if err != nil {
					return err
				}
				gvr := schema.GroupVersionResource{Group: "example.com", Version: "v2", Resource: "scalablewidgets"}
				// Create with replicas.max=5 (allowed: 5 <= 10)
				created, err := dynClient.Resource(gvr).Namespace("test-ns").Create(context.TODO(), &unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "example.com/v2",
						"kind":       "ScalableWidget",
						"metadata": map[string]interface{}{
							"name":      "v2-update-deny" + suffix,
							"namespace": "test-ns",
						},
						"spec": map[string]interface{}{
							"replicas": map[string]interface{}{
								"max": int64(5),
							},
						},
					},
				}, metav1.CreateOptions{})
				if err != nil {
					return fmt.Errorf("initial create should have succeeded: %w", err)
				}
				// Update to replicas.max=15 (denied: new=15>10)
				created.Object["spec"] = map[string]interface{}{
					"replicas": map[string]interface{}{
						"max": int64(15),
					},
				}
				_, err = dynClient.Resource(gvr).Namespace("test-ns").Update(context.TODO(), created, metav1.UpdateOptions{})
				return err
			},
			expectAllowed: false,
		},
	}
}

// at most 10. The CEL expression is version-specific: for v1 it checks
// spec.replicas (integer), for v2 it checks spec.replicas.max (integer nested
// in an object). For updates, both old and new objects must satisfy the
// condition.
func crdReplicasDecision(a authorizer.Attributes, conditionsType string) authorizer.ConditionsAwareDecision {
	if a.GetResource() != "scalablewidgets" {
		return authorizer.ConditionsAwareDecisionNoOpinion("", nil)
	}

	var objectCondition, oldObjectCondition string
	switch a.GetAPIVersion() {
	case "v1":
		objectCondition = `has(object.spec.replicas) && object.spec.replicas <= 10`
		oldObjectCondition = `has(oldObject.spec.replicas) && oldObject.spec.replicas <= 10`
	case "v2":
		objectCondition = `has(object.spec.replicas) && has(object.spec.replicas.max) && object.spec.replicas.max <= 10`
		oldObjectCondition = `has(oldObject.spec.replicas) && has(oldObject.spec.replicas.max) && oldObject.spec.replicas.max <= 10`
	default:
		return authorizer.ConditionsAwareDecisionNoOpinion("", nil)
	}

	var condition string
	switch a.GetVerb() {
	case "create":
		condition = objectCondition
	case "update":
		condition = objectCondition + " && " + oldObjectCondition
	default:
		return authorizer.ConditionsAwareDecisionNoOpinion("", nil)
	}

	return authorizer.ConditionsAwareDecisionConditionsMap(
		nil, nil,
		[]authorizer.Condition{
			authorizer.GenericCondition{
				ID:          "example.com/limit-replicas",
				Condition:   condition,
				Type:        conditionsType,
				Description: "only allow if replicas <= 10",
			},
		},
	)
}
