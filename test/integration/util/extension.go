/*
Copyright 2021 The Kubernetes Authors.

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
	"testing"

	v1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionsclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	apiextensionstestserver "k8s.io/apiextensions-apiserver/test/integration/fixtures"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	cacheddiscovery "k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/metadata"
	"k8s.io/client-go/metadata/metadatainformer"
	"k8s.io/client-go/restmapper"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
)

// ExtensionTestContext holds state from ExtensionSetup
type ExtensionTestContext struct {
	StopCh             <-chan struct{}
	TearDown           func()
	ClientSet          clientset.Interface
	RESTMapper         *restmapper.DeferredDiscoveryRESTMapper
	APIExtensionClient apiextensionsclientset.Interface
	SharedInformers    informers.SharedInformerFactory
	MetadataInformers  metadatainformer.SharedInformerFactory
	MetadataClient     metadata.Interface
	DynamicClient      dynamic.Interface
}

// ExtensionSetup performs shared setup
func ExtensionSetup(t *testing.T, result *kubeapiservertesting.TestServer, workerCount int) *ExtensionTestContext {
	clientSet, err := clientset.NewForConfig(result.ClientConfig)
	if err != nil {
		t.Fatalf("error creating clientset: %v", err)
	}

	// Helpful stuff for testing CRD.
	apiExtensionClient, err := apiextensionsclientset.NewForConfig(result.ClientConfig)
	if err != nil {
		t.Fatalf("error creating extension clientset: %v", err)
	}
	// CreateCRDUsingRemovedAPI wants to use this namespace for verifying
	// namespace-scoped CRD creation.
	CreateNamespaceOrDie("aval", clientSet, t)

	discoveryClient := cacheddiscovery.NewMemCacheClient(clientSet.Discovery())
	restMapper := restmapper.NewDeferredDiscoveryRESTMapper(discoveryClient)
	restMapper.Reset()
	config := *result.ClientConfig
	metadataClient, err := metadata.NewForConfig(&config)
	if err != nil {
		t.Fatalf("failed to create metadataClient: %v", err)
	}
	dynamicClient, err := dynamic.NewForConfig(&config)
	if err != nil {
		t.Fatalf("failed to create dynamicClient: %v", err)
	}
	sharedInformers := informers.NewSharedInformerFactory(clientSet, 0)
	metadataInformers := metadatainformer.NewSharedInformerFactoryWithOptions(metadataClient, 0, metadatainformer.WithStopOnZeroEventHandlers(true))

	// controller creation
	stopCh := make(chan struct{})
	tearDown := func() {
		close(stopCh)
		result.TearDownFn()
	}

	return &ExtensionTestContext{
		StopCh:             stopCh,
		TearDown:           tearDown,
		RESTMapper:         restMapper,
		ClientSet:          clientSet,
		APIExtensionClient: apiExtensionClient,
		DynamicClient:      dynamicClient,
		SharedInformers:    sharedInformers,
		MetadataInformers:  metadataInformers,
		MetadataClient:     metadataClient,
	}
}

func NewCRDInstance(definition *apiextensionsv1.CustomResourceDefinition, namespace, name string) *unstructured.Unstructured {
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind":       definition.Spec.Names.Kind,
			"apiVersion": definition.Spec.Group + "/" + definition.Spec.Versions[0].Name,
			"metadata": map[string]interface{}{
				"name":      name,
				"namespace": namespace,
			},
		},
	}
}

func CreateRandomCustomResourceDefinition(t *testing.T, apiExtensionClient apiextensionsclientset.Interface, dynamicClient dynamic.Interface, namespace string) (*apiextensionsv1.CustomResourceDefinition, dynamic.ResourceInterface) {
	// Create a random custom resource definition and ensure it's available for
	// use.
	definition := apiextensionstestserver.NewRandomNameV1CustomResourceDefinition(apiextensionsv1.NamespaceScoped)

	definition, err := apiextensionstestserver.CreateNewV1CustomResourceDefinition(definition, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatalf("failed to create CustomResourceDefinition: %v", err)
	}

	// Get a client for the custom resource.
	gvr := schema.GroupVersionResource{Group: definition.Spec.Group, Version: definition.Spec.Versions[0].Name, Resource: definition.Spec.Names.Plural}

	resourceClient := dynamicClient.Resource(gvr).Namespace(namespace)

	return definition, resourceClient
}

func CreateNamespaceOrDie(name string, c clientset.Interface, t *testing.T) *v1.Namespace {
	ns := &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: name}}
	if _, err := c.CoreV1().Namespaces().Create(context.TODO(), ns, metav1.CreateOptions{}); err != nil {
		t.Fatalf("failed to create namespace: %v", err)
	}
	falseVar := false
	_, err := c.CoreV1().ServiceAccounts(ns.Name).Create(context.TODO(), &v1.ServiceAccount{
		ObjectMeta:                   metav1.ObjectMeta{Name: "default"},
		AutomountServiceAccountToken: &falseVar,
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create service account: %v", err)
	}
	return ns
}

func DeleteNamespaceOrDie(name string, c clientset.Interface, t *testing.T) {
	zero := int64(0)
	background := metav1.DeletePropagationBackground
	err := c.CoreV1().Namespaces().Delete(context.TODO(), name, metav1.DeleteOptions{GracePeriodSeconds: &zero, PropagationPolicy: &background})
	if err != nil {
		t.Fatalf("failed to delete namespace %q: %v", name, err)
	}
}
