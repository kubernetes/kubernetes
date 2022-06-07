/*
Copyright 2019 The Kubernetes Authors.

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

package etcd

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/client-go/dynamic"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
)

// TestCrossGroupStorage tests to make sure that all objects stored in an expected location in etcd can be converted/read.
func TestCrossGroupStorage(t *testing.T) {
	apiServer := StartRealAPIServerOrDie(t, func(opts *options.ServerRunOptions) {
		// force enable all resources so we can check storage.
	})
	defer apiServer.Cleanup()

	etcdStorageData := GetEtcdStorageData()

	crossGroupResources := map[schema.GroupVersionKind][]Resource{}

	apiServer.Client.CoreV1().Namespaces().Create(context.TODO(), &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: testNamespace}}, metav1.CreateOptions{})

	// Group by persisted GVK
	for _, resourceToPersist := range apiServer.Resources {
		gvk := resourceToPersist.Mapping.GroupVersionKind
		data, exists := etcdStorageData[resourceToPersist.Mapping.Resource]
		if !exists {
			continue
		}
		storageGVK := gvk
		if data.ExpectedGVK != nil {
			storageGVK = *data.ExpectedGVK
		}
		crossGroupResources[storageGVK] = append(crossGroupResources[storageGVK], resourceToPersist)
	}

	// Clear any without cross-group sources
	for gvk, resources := range crossGroupResources {
		groups := sets.NewString()
		for _, resource := range resources {
			groups.Insert(resource.Mapping.GroupVersionKind.Group)
		}
		if len(groups) < 2 {
			delete(crossGroupResources, gvk)
		}
	}

	if len(crossGroupResources) == 0 {
		// Sanity check
		t.Fatal("no cross-group resources found")
	}

	// Test all potential cross-group sources can be watched and fetched from all other sources
	for gvk, resources := range crossGroupResources {
		t.Run(gvk.String(), func(t *testing.T) {
			// use the first one to create the initial object
			resource := resources[0]

			// compute namespace
			ns := ""
			if resource.Mapping.Scope.Name() == meta.RESTScopeNameNamespace {
				ns = testNamespace
			}

			data := etcdStorageData[resource.Mapping.Resource]
			// create object
			resourceClient, obj, err := JSONToUnstructured(data.Stub, ns, resource.Mapping, apiServer.Dynamic)
			if err != nil {
				t.Fatal(err)
			}
			actual, err := resourceClient.Create(context.TODO(), obj, metav1.CreateOptions{})
			if err != nil {
				t.Fatal(err)
			}
			name := actual.GetName()

			// Set up clients, versioned data, and watches for all versions
			var (
				clients       = map[schema.GroupVersionResource]dynamic.ResourceInterface{}
				versionedData = map[schema.GroupVersionResource]*unstructured.Unstructured{}
				watches       = map[schema.GroupVersionResource]watch.Interface{}
			)
			for _, resource := range resources {
				clients[resource.Mapping.Resource] = apiServer.Dynamic.Resource(resource.Mapping.Resource).Namespace(ns)
				versionedData[resource.Mapping.Resource], err = clients[resource.Mapping.Resource].Get(context.TODO(), name, metav1.GetOptions{})
				if err != nil {
					t.Fatalf("error finding resource via %s: %v", resource.Mapping.Resource.GroupVersion().String(), err)
				}
				watches[resource.Mapping.Resource], err = clients[resource.Mapping.Resource].Watch(context.TODO(), metav1.ListOptions{ResourceVersion: actual.GetResourceVersion()})
				if err != nil {
					t.Fatalf("error opening watch via %s: %v", resource.Mapping.Resource.GroupVersion().String(), err)
				}
			}

			versioner := storage.APIObjectVersioner{}
			for _, resource := range resources {
				// clear out the things cleared in etcd
				versioned := versionedData[resource.Mapping.Resource]
				if err := versioner.PrepareObjectForStorage(versioned); err != nil {
					t.Error(err)
					continue
				}
				versionedJSON, err := versioned.MarshalJSON()
				if err != nil {
					t.Error(err)
					continue
				}

				// Update in etcd
				if _, err := apiServer.KV.Put(context.Background(), data.ExpectedEtcdPath, string(versionedJSON)); err != nil {
					t.Error(err)
					continue
				}
				t.Logf("wrote %s to etcd", resource.Mapping.Resource.GroupVersion().String())

				// Ensure everyone gets a watch event with the right version
				for watchResource, watcher := range watches {
					select {
					case event, ok := <-watcher.ResultChan():
						if !ok {
							t.Fatalf("watch of %s closed in response to persisting %s", watchResource.GroupVersion().String(), resource.Mapping.Resource.GroupVersion().String())
						}
						if event.Type != watch.Modified {
							eventJSON, _ := json.Marshal(event)
							t.Errorf("unexpected watch event sent to watch of %s in response to persisting %s: %s", watchResource.GroupVersion().String(), resource.Mapping.Resource.GroupVersion().String(), string(eventJSON))
							continue
						}
						if event.Object.GetObjectKind().GroupVersionKind().GroupVersion() != watchResource.GroupVersion() {
							t.Errorf("unexpected group version object sent to watch of %s in response to persisting %s: %#v", watchResource.GroupVersion().String(), resource.Mapping.Resource.GroupVersion().String(), event.Object)
							continue
						}
						t.Logf("     received event for %s", watchResource.GroupVersion().String())
					case <-time.After(30 * time.Second):
						t.Errorf("timed out waiting for watch event for %s in response to persisting %s", watchResource.GroupVersion().String(), resource.Mapping.Resource.GroupVersion().String())
						continue
					}
				}

				// Ensure everyone can do a direct get and gets the right version
				for clientResource, client := range clients {
					obj, err := client.Get(context.TODO(), name, metav1.GetOptions{})
					if err != nil {
						t.Errorf("error looking up %s after persisting %s", clientResource.GroupVersion().String(), resource.Mapping.Resource.GroupVersion().String())
						continue
					}
					if obj.GetObjectKind().GroupVersionKind().GroupVersion() != clientResource.GroupVersion() {
						t.Errorf("unexpected group version retrieved from %s after persisting %s: %#v", clientResource.GroupVersion().String(), resource.Mapping.Resource.GroupVersion().String(), obj)
						continue
					}
					t.Logf("     fetched object for %s", clientResource.GroupVersion().String())
				}
			}
		})
	}
}
