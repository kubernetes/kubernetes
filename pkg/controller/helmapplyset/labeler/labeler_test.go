/*
Copyright 2024 The Kubernetes Authors.

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

package labeler

import (
	"context"
	"testing"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	dynamicfake "k8s.io/client-go/dynamic/fake"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/klog/v2/ktesting"
)

// testHelmReleaseInfo is a test-only struct to avoid import cycles
type testHelmReleaseInfo struct {
	Name       string
	Namespace  string
	GroupKinds sets.Set[schema.GroupKind]
	Manifest   string
}

func TestParseManifestResources(t *testing.T) {
	labeler := &Labeler{}

	manifest := `apiVersion: v1
kind: Service
metadata:
  name: test-service
  namespace: default
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test-deployment
  namespace: default
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: test-config
  namespace: default
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: test-cluster-role
`

	resources, err := labeler.parseManifestResources(manifest)
	if err != nil {
		t.Fatalf("parseManifestResources() error = %v", err)
	}

	if len(resources) != 4 {
		t.Errorf("Expected 4 resources, got %d", len(resources))
	}

	// Verify Service
	foundService := false
	for _, res := range resources {
		if res.GroupKind.Kind == "Service" && res.Name == "test-service" {
			foundService = true
			if res.Namespace != "default" {
				t.Errorf("Service namespace = %v, want default", res.Namespace)
			}
			break
		}
	}
	if !foundService {
		t.Error("Service not found in parsed resources")
	}

	// Verify ClusterRole (cluster-scoped)
	foundClusterRole := false
	for _, res := range resources {
		if res.GroupKind.Kind == "ClusterRole" && res.Name == "test-cluster-role" {
			foundClusterRole = true
			if res.Namespace != "" {
				t.Errorf("ClusterRole namespace = %v, want empty", res.Namespace)
			}
			break
		}
	}
	if !foundClusterRole {
		t.Error("ClusterRole not found in parsed resources")
	}
}

func TestParseManifestResources_EmptyManifest(t *testing.T) {
	labeler := &Labeler{}

	resources, err := labeler.parseManifestResources("")
	if err != nil {
		t.Fatalf("parseManifestResources() error = %v", err)
	}
	if len(resources) != 0 {
		t.Errorf("Expected 0 resources, got %d", len(resources))
	}
}

func TestLabelResource(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	// Create a test resource
	testService := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Service",
			"metadata": map[string]interface{}{
				"name":      "test-service",
				"namespace": "default",
			},
		},
	}

	scheme := runtime.NewScheme()
	dynamicClient := dynamicfake.NewSimpleDynamicClient(scheme, testService)
	kubeClient := fake.NewSimpleClientset()

	// Create a simple REST mapper
	mapper := meta.NewDefaultRESTMapper([]schema.GroupVersion{{Group: "", Version: "v1"}})
	mapper.Add(schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Service"}, meta.RESTScopeNamespace)

	labeler := NewLabeler(dynamicClient, kubeClient, mapper, logger)

	applySetID := "applyset-test-id-v1"

	ctx := context.Background()

	// Get the resource client
	resourceClient := dynamicClient.Resource(schema.GroupVersionResource{
		Group:    "",
		Version:  "v1",
		Resource: "services",
	}).Namespace("default")

	err := labeler.labelResource(ctx, resourceClient, "test-service", applySetID)
	if err != nil {
		t.Fatalf("labelResource() error = %v", err)
	}

	// Verify label was added
	labeled, err := resourceClient.Get(ctx, "test-service", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get labeled resource: %v", err)
	}

	labels := labeled.GetLabels()
	if labels == nil {
		t.Fatal("Labels should not be nil")
	}

	if labels[ApplysetPartOfLabel] != applySetID {
		t.Errorf("ApplySet label = %v, want %v", labels[ApplysetPartOfLabel], applySetID)
	}
}

func TestLabelResource_AlreadyLabeled(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	applySetID := "applyset-test-id-v1"

	// Create a test resource with label already set
	testService := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Service",
			"metadata": map[string]interface{}{
				"name":      "test-service",
				"namespace": "default",
				"labels": map[string]interface{}{
					ApplysetPartOfLabel: applySetID,
				},
			},
		},
	}

	scheme := runtime.NewScheme()
	dynamicClient := dynamicfake.NewSimpleDynamicClient(scheme, testService)
	kubeClient := fake.NewSimpleClientset()

	mapper := meta.NewDefaultRESTMapper([]schema.GroupVersion{{Group: "", Version: "v1"}})
	mapper.Add(schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Service"}, meta.RESTScopeNamespace)

	labeler := NewLabeler(dynamicClient, kubeClient, mapper, logger)

	ctx := context.Background()
	resourceClient := dynamicClient.Resource(schema.GroupVersionResource{
		Group:    "",
		Version:  "v1",
		Resource: "services",
	}).Namespace("default")

	// Should not error if already labeled
	err := labeler.labelResource(ctx, resourceClient, "test-service", applySetID)
	if err != nil {
		t.Fatalf("labelResource() error = %v", err)
	}
}

func TestLabelResource_NotFound(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	scheme := runtime.NewScheme()
	dynamicClient := dynamicfake.NewSimpleDynamicClient(scheme)
	kubeClient := fake.NewSimpleClientset()

	mapper := meta.NewDefaultRESTMapper([]schema.GroupVersion{{Group: "", Version: "v1"}})
	mapper.Add(schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Service"}, meta.RESTScopeNamespace)

	labeler := NewLabeler(dynamicClient, kubeClient, mapper, logger)

	ctx := context.Background()
	resourceClient := dynamicClient.Resource(schema.GroupVersionResource{
		Group:    "",
		Version:  "v1",
		Resource: "services",
	}).Namespace("default")

	// Should not error if resource doesn't exist
	err := labeler.labelResource(ctx, resourceClient, "non-existent", "applyset-test-id-v1")
	if err != nil {
		t.Fatalf("labelResource() should not error on NotFound, got: %v", err)
	}
}

func TestLabelResourcesByHelmLabels(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	applySetID := "applyset-test-id-v1"
	releaseName := "test-release"

	// Create test resources with Helm labels
	testService := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Service",
			"metadata": map[string]interface{}{
				"name":      "test-service",
				"namespace": "default",
				"labels": map[string]interface{}{
					HelmManagedByLabel: "Helm",
					HelmInstanceLabel:  releaseName,
				},
			},
		},
	}

	testDeployment := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "apps/v1",
			"kind":       "Deployment",
			"metadata": map[string]interface{}{
				"name":      "test-deployment",
				"namespace": "default",
				"labels": map[string]interface{}{
					HelmManagedByLabel: "Helm",
					HelmInstanceLabel:  releaseName,
				},
			},
		},
	}

	scheme := runtime.NewScheme()
	dynamicClient := dynamicfake.NewSimpleDynamicClient(scheme, testService, testDeployment)
	kubeClient := fake.NewSimpleClientset()

	mapper := meta.NewDefaultRESTMapper([]schema.GroupVersion{
		{Group: "", Version: "v1"},
		{Group: "apps", Version: "v1"},
	})
	mapper.Add(schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Service"}, meta.RESTScopeNamespace)
	mapper.Add(schema.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"}, meta.RESTScopeNamespace)

	labeler := NewLabeler(dynamicClient, kubeClient, mapper, logger)

	groupKinds := sets.New[schema.GroupKind](
		schema.GroupKind{Group: "", Kind: "Service"},
		schema.GroupKind{Group: "apps", Kind: "Deployment"},
	)

	ctx := context.Background()
	err := labeler.labelResourcesByHelmLabels(ctx, releaseName, "default", applySetID, groupKinds)
	if err != nil {
		t.Fatalf("labelResourcesByHelmLabels() error = %v", err)
	}

	// Verify Service was labeled
	serviceClient := dynamicClient.Resource(schema.GroupVersionResource{
		Group:    "",
		Version:  "v1",
		Resource: "services",
	}).Namespace("default")

	labeledService, err := serviceClient.Get(ctx, "test-service", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get labeled service: %v", err)
	}

	if labeledService.GetLabels()[ApplysetPartOfLabel] != applySetID {
		t.Errorf("Service ApplySet label = %v, want %v",
			labeledService.GetLabels()[ApplysetPartOfLabel], applySetID)
	}

	// Verify Deployment was labeled
	deploymentClient := dynamicClient.Resource(schema.GroupVersionResource{
		Group:    "apps",
		Version:  "v1",
		Resource: "deployments",
	}).Namespace("default")

	labeledDeployment, err := deploymentClient.Get(ctx, "test-deployment", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get labeled deployment: %v", err)
	}

	if labeledDeployment.GetLabels()[ApplysetPartOfLabel] != applySetID {
		t.Errorf("Deployment ApplySet label = %v, want %v",
			labeledDeployment.GetLabels()[ApplysetPartOfLabel], applySetID)
	}
}

func TestLabelResources_ManifestParsing(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	applySetID := "applyset-test-id-v1"

	manifest := `apiVersion: v1
kind: Service
metadata:
  name: test-service
  namespace: default
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test-deployment
  namespace: default
`

	releaseInfo := &testHelmReleaseInfo{
		Name:      "test-release",
		Namespace: "default",
		Manifest:  manifest,
		GroupKinds: sets.New[schema.GroupKind](
			schema.GroupKind{Group: "", Kind: "Service"},
			schema.GroupKind{Group: "apps", Kind: "Deployment"},
		),
	}

	// Create test resources
	testService := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Service",
			"metadata": map[string]interface{}{
				"name":      "test-service",
				"namespace": "default",
			},
		},
	}

	testDeployment := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "apps/v1",
			"kind":       "Deployment",
			"metadata": map[string]interface{}{
				"name":      "test-deployment",
				"namespace": "default",
			},
		},
	}

	scheme := runtime.NewScheme()
	dynamicClient := dynamicfake.NewSimpleDynamicClient(scheme, testService, testDeployment)
	kubeClient := fake.NewSimpleClientset()

	mapper := meta.NewDefaultRESTMapper([]schema.GroupVersion{
		{Group: "", Version: "v1"},
		{Group: "apps", Version: "v1"},
	})
	mapper.Add(schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Service"}, meta.RESTScopeNamespace)
	mapper.Add(schema.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"}, meta.RESTScopeNamespace)

	labeler := NewLabeler(dynamicClient, kubeClient, mapper, logger)

	ctx := context.Background()
	err := labeler.LabelResources(ctx, releaseInfo.Name, releaseInfo.Namespace, releaseInfo.Manifest, releaseInfo.GroupKinds, applySetID)
	if err != nil {
		t.Fatalf("LabelResources() error = %v", err)
	}

	// Verify resources were labeled
	serviceClient := dynamicClient.Resource(schema.GroupVersionResource{
		Group:    "",
		Version:  "v1",
		Resource: "services",
	}).Namespace("default")

	labeledService, err := serviceClient.Get(ctx, "test-service", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get labeled service: %v", err)
	}

	if labeledService.GetLabels()[ApplysetPartOfLabel] != applySetID {
		t.Errorf("Service ApplySet label = %v, want %v",
			labeledService.GetLabels()[ApplysetPartOfLabel], applySetID)
	}
}

func TestLabelResourceFromManifest_ClusterScoped(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	applySetID := "applyset-test-id-v1"

	// Create cluster-scoped resource
	testClusterRole := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "rbac.authorization.k8s.io/v1",
			"kind":       "ClusterRole",
			"metadata": map[string]interface{}{
				"name": "test-cluster-role",
			},
		},
	}

	scheme := runtime.NewScheme()
	dynamicClient := dynamicfake.NewSimpleDynamicClient(scheme, testClusterRole)
	kubeClient := fake.NewSimpleClientset()

	mapper := meta.NewDefaultRESTMapper([]schema.GroupVersion{
		{Group: "rbac.authorization.k8s.io", Version: "v1"},
	})
	mapper.Add(schema.GroupVersionKind{
		Group:   "rbac.authorization.k8s.io",
		Version: "v1",
		Kind:    "ClusterRole",
	}, meta.RESTScopeRoot)

	labeler := NewLabeler(dynamicClient, kubeClient, mapper, logger)

	res := ResourceInfo{
		GroupKind: schema.GroupKind{
			Group: "rbac.authorization.k8s.io",
			Kind:  "ClusterRole",
		},
		Name:       "test-cluster-role",
		Namespace:  "", // Cluster-scoped
		APIVersion: "rbac.authorization.k8s.io/v1",
	}

	ctx := context.Background()
	err := labeler.labelResourceFromManifest(ctx, res, applySetID)
	if err != nil {
		t.Fatalf("labelResourceFromManifest() error = %v", err)
	}

	// Verify ClusterRole was labeled
	clusterRoleClient := dynamicClient.Resource(schema.GroupVersionResource{
		Group:    "rbac.authorization.k8s.io",
		Version:  "v1",
		Resource: "clusterroles",
	})

	labeledClusterRole, err := clusterRoleClient.Get(ctx, "test-cluster-role", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get labeled ClusterRole: %v", err)
	}

	if labeledClusterRole.GetLabels()[ApplysetPartOfLabel] != applySetID {
		t.Errorf("ClusterRole ApplySet label = %v, want %v",
			labeledClusterRole.GetLabels()[ApplysetPartOfLabel], applySetID)
	}
}

func TestRemoveLabels(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	applySetID := "applyset-test-id-v1"

	// Create test resource with ApplySet label
	testService := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Service",
			"metadata": map[string]interface{}{
				"name":      "test-service",
				"namespace": "default",
				"labels": map[string]interface{}{
					ApplysetPartOfLabel: applySetID,
				},
			},
		},
	}

	scheme := runtime.NewScheme()
	dynamicClient := dynamicfake.NewSimpleDynamicClient(scheme, testService)
	kubeClient := fake.NewSimpleClientset()

	mapper := meta.NewDefaultRESTMapper([]schema.GroupVersion{
		{Group: "", Version: "v1"},
	})
	mapper.Add(schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Service"}, meta.RESTScopeNamespace)

	labeler := NewLabeler(dynamicClient, kubeClient, mapper, logger)

	ctx := context.Background()
	err := labeler.RemoveLabels(ctx, "test-release", "default", applySetID, sets.New[schema.GroupKind](
		schema.GroupKind{Group: "", Kind: "Service"},
	))
	if err != nil {
		t.Fatalf("RemoveLabels() error = %v", err)
	}

	// Verify label was removed
	serviceClient := dynamicClient.Resource(schema.GroupVersionResource{
		Group:    "",
		Version:  "v1",
		Resource: "services",
	}).Namespace("default")

	unlabeledService, err := serviceClient.Get(ctx, "test-service", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get unlabeled service: %v", err)
	}

	labels := unlabeledService.GetLabels()
	if labels != nil {
		if _, ok := labels[ApplysetPartOfLabel]; ok {
			t.Error("ApplySet label should have been removed")
		}
	}
}
