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

package apidefinitions

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"

	v1 "k8s.io/api/core/v1"
	apiextensionsclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/etcd"
	"k8s.io/kubernetes/test/integration/framework"
)

type DefinitionTestFunc func(t *testing.T, setup Definition)

type Definition struct {
	Config        *rest.Config
	Client        kubernetes.Interface
	DynamicClient dynamic.Interface
	Mapping       *meta.RESTMapping
	Resource      metav1.APIResource
	Namespace     string
	StorageData   etcd.StorageData
	Subresources  []string
}

func (d *Definition) HasVerb(verb string) bool {
	return slices.Contains(d.Resource.Verbs, verb)
}

func (d *Definition) HasSubresource(subresource string) bool {
	return slices.Contains(d.Subresources, subresource)
}

func (d *Definition) HasStatus() bool {
	return d.HasSubresource("status")
}

// ResourceClient returns a dynamic resource client scoped to the appropriate namespace.
func (d *Definition) ResourceClient() dynamic.ResourceInterface {
	namespace := d.Namespace
	if d.Mapping.Scope == meta.RESTScopeRoot {
		namespace = ""
	}
	return d.DynamicClient.Resource(d.Mapping.Resource).Namespace(namespace)
}

// TestAllDefinitions starts an apiserver and runs testFunc against every
// discoverable resource. It registers a fixed set of CRDs so that a sample
// set of custom resources discoverable and tested.
func TestAllDefinitions(t *testing.T, testNamespace string, testFunc DefinitionTestFunc) {
	server, err := apiservertesting.StartTestServer(t, apiservertesting.NewDefaultTestServerOptions(), []string{"--disable-admission-plugins", "ServiceAccount,TaintNodesByCondition"}, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	defer server.TearDownFn()

	client, err := kubernetes.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatal(err)
	}
	dynamicClient, err := dynamic.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatal(err)
	}

	etcd.CreateTestCRDs(t, apiextensionsclientset.NewForConfigOrDie(server.ClientConfig), false, etcd.GetCustomResourceDefinitionData()...)

	if _, err := client.CoreV1().Namespaces().Create(context.TODO(), &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: testNamespace}}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	storageData := etcd.GetEtcdStorageDataForNamespace(testNamespace)

	_, resourceLists, err := client.Discovery().ServerGroupsAndResources()
	if err != nil {
		t.Fatalf("Failed to get ServerGroupsAndResources: %v", err)
	}

	for _, resourceList := range resourceLists {
		for _, resource := range resourceList.APIResources {
			// Iterate over root resources
			if strings.Contains(resource.Name, "/") {
				continue
			}

			mapping, err := CreateMapping(resourceList.GroupVersion, resource)
			if err != nil {
				t.Fatal(err)
			}

			t.Run(mapping.Resource.String(), func(t *testing.T) {

				storageData, ok := storageData[mapping.Resource]
				if !ok {
					t.Skipf("no test data for %s in etcd.GetEtcdStorageData, skipping", mapping.Resource)
				}

				var subresources []string
				for _, r := range resourceList.APIResources {
					if suffix, ok := strings.CutPrefix(r.Name, resource.Name+"/"); ok {
						subresources = append(subresources, suffix)
					}
				}

				setup := Definition{
					Config:        server.ClientConfig,
					Client:        client,
					DynamicClient: dynamicClient,
					Mapping:       mapping,
					Resource:      resource,
					Namespace:     testNamespace,
					StorageData:   storageData,
					Subresources:  subresources,
				}

				testFunc(t, setup)
			})
		}
	}
}

func CreateMapping(groupVersion string, resource metav1.APIResource) (*meta.RESTMapping, error) {
	gv, err := schema.ParseGroupVersion(groupVersion)
	if err != nil {
		return nil, err
	}
	if len(resource.Group) > 0 || len(resource.Version) > 0 {
		gv = schema.GroupVersion{
			Group:   resource.Group,
			Version: resource.Version,
		}
	}
	gvk := gv.WithKind(resource.Kind)
	gvr := gv.WithResource(resource.Name)

	scope := meta.RESTScopeRoot
	if resource.Namespaced {
		scope = meta.RESTScopeNamespace
	}
	return &meta.RESTMapping{
		Resource:         gvr,
		GroupVersionKind: gvk,
		Scope:            scope,
	}, nil
}

// ResourceString returns the kubectl-style "resource.version.group" representation
// of a GroupVersionResource (e.g. "deployments.v1.apps", "pods.v1").
func ResourceString(gvr schema.GroupVersionResource) string {
	if gvr.Group == "" {
		return gvr.Resource + "." + gvr.Version
	}
	return gvr.Resource + "." + gvr.Version + "." + gvr.Group
}

// matchesException returns true if gvr matches any entry in exceptions.
// Each entry must be a kubectl-style resource string in either
// "resource.group" form (e.g. "pods", "deployments.apps",
// "customresourcedefinitions.apiextensions.k8s.io"), which matches all
// versions of that resource, or "resource.version.group" form (e.g.
// "pods.v1", "deployments.v1.apps"), which matches a single version.
func matchesException(gvr schema.GroupVersionResource, exceptions sets.Set[string]) bool {
	if exceptions.Has(gvr.GroupResource().String()) {
		return true
	}
	return exceptions.Has(ResourceString(gvr))
}

// TestObj is a generic test helper that creates an Unstructured object from a creation stub
// and explicitly sets the status from a separate JSON payload.
func TestObj(t *testing.T, stub, status string, gvk schema.GroupVersionKind) *unstructured.Unstructured {
	t.Helper()
	obj := &unstructured.Unstructured{}
	if err := json.Unmarshal([]byte(stub), &obj.Object); err != nil {
		t.Fatal(err)
	}
	var statusObj map[string]interface{}
	if err := json.Unmarshal([]byte(status), &statusObj); err != nil {
		t.Fatal(err)
	}
	if s, ok := statusObj["status"]; ok {
		obj.Object["status"] = s
	}
	obj.SetAPIVersion(gvk.GroupVersion().String())
	obj.SetKind(gvk.Kind)
	return obj
}
