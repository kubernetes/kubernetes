/*
Copyright 2016 The Kubernetes Authors.

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

package namespace

import (
	"fmt"
	"testing"
	"time"

	federation_api "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	federation_release_1_4 "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_4"
	fake_federation_release_1_4 "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_4/fake"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	api_v1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/stretchr/testify/assert"
)

func TestNamespaceController(t *testing.T) {
	cluster1 := mkCluster("cluster1", api_v1.ConditionTrue)
	cluster2 := mkCluster("cluster2", api_v1.ConditionTrue)

	fakeClient := &fake_federation_release_1_4.Clientset{}
	RegisterList("clusters", fakeClient, &federation_api.ClusterList{Items: []federation_api.Cluster{*cluster1}})
	RegisterList("namespaces", fakeClient, &api_v1.NamespaceList{Items: []api_v1.Namespace{}})
	namespaceWatch := RegisterWatch("namespaces", fakeClient)
	clusterWatch := RegisterWatch("clusters", fakeClient)

	cluster1Client := &fake_federation_release_1_4.Clientset{}
	cluster1Watch := RegisterWatch("namespaces", cluster1Client)
	RegisterList("namespaces", cluster1Client, &api_v1.NamespaceList{Items: []api_v1.Namespace{}})
	cluster1CreateChan := RegisterCopyOnCreate("namespaces", cluster1Client, cluster1Watch)
	cluster1UpdateChan := RegisterCopyOnUpdate("namespaces", cluster1Client, cluster1Watch)

	cluster2Client := &fake_federation_release_1_4.Clientset{}
	cluster2Watch := RegisterWatch("namespaces", cluster2Client)
	RegisterList("namespaces", cluster2Client, &api_v1.NamespaceList{Items: []api_v1.Namespace{}})
	cluster2CreateChan := RegisterCopyOnCreate("namespaces", cluster2Client, cluster2Watch)

	namespaceController := NewNamespaceController(fakeClient)
	informer := toFederatedInformerForTestOnly(namespaceController.namespaceFederatedInformer)
	informer.SetClientFactory(func(cluster *federation_api.Cluster) (federation_release_1_4.Interface, error) {
		switch cluster.Name {
		case cluster1.Name:
			return cluster1Client, nil
		case cluster2.Name:
			return cluster2Client, nil
		default:
			return nil, fmt.Errorf("Unknown cluster")
		}
	})
	namespaceController.clusterAvailableDelay = time.Second
	namespaceController.namespaceReviewDelay = 50 * time.Millisecond
	namespaceController.smallDelay = 20 * time.Millisecond
	namespaceController.updateTimeout = 5 * time.Second

	stop := make(chan struct{})
	namespaceController.Run(stop)

	ns1 := api_v1.Namespace{
		ObjectMeta: api_v1.ObjectMeta{
			Name: "test-namespace",
		},
	}

	// Test add federated namespace.
	namespaceWatch.Add(&ns1)
	createdNamespace := GetNamespaceFromChan(cluster1CreateChan)
	assert.NotNil(t, createdNamespace)
	assert.Equal(t, ns1.Name, createdNamespace.Name)

	// Test update federated namespace.
	ns1.Annotations = map[string]string{
		"A": "B",
	}
	namespaceWatch.Modify(&ns1)
	updatedNamespace := GetNamespaceFromChan(cluster1UpdateChan)
	assert.NotNil(t, updatedNamespace)
	assert.Equal(t, ns1.Name, updatedNamespace.Name)
	// assert.Contains(t, updatedNamespace.Annotations, "A")

	// Test add cluster
	clusterWatch.Add(cluster2)
	createdNamespace2 := GetNamespaceFromChan(cluster2CreateChan)
	assert.NotNil(t, createdNamespace2)
	assert.Equal(t, ns1.Name, createdNamespace2.Name)
	// assert.Contains(t, createdNamespace2.Annotations, "A")

	close(stop)
}

func toFederatedInformerForTestOnly(informer util.FederatedInformer) util.FederatedInformerForTestOnly {
	inter := informer.(interface{})
	return inter.(util.FederatedInformerForTestOnly)
}

func mkCluster(name string, readyStatus api_v1.ConditionStatus) *federation_api.Cluster {
	return &federation_api.Cluster{
		ObjectMeta: api_v1.ObjectMeta{
			Name: name,
		},
		Status: federation_api.ClusterStatus{
			Conditions: []federation_api.ClusterCondition{
				{Type: federation_api.ClusterReady, Status: readyStatus},
			},
		},
	}
}

func RegisterWatch(resource string, client *fake_federation_release_1_4.Clientset) *watch.FakeWatcher {
	watcher := watch.NewFake()
	client.AddWatchReactor(resource, func(action core.Action) (bool, watch.Interface, error) { return true, watcher, nil })
	return watcher
}

func RegisterList(resource string, client *fake_federation_release_1_4.Clientset, obj runtime.Object) {
	client.AddReactor("list", resource, func(action core.Action) (bool, runtime.Object, error) {
		return true, obj, nil
	})
}

func RegisterCopyOnCreate(resource string, client *fake_federation_release_1_4.Clientset, watcher *watch.FakeWatcher) chan runtime.Object {
	objChan := make(chan runtime.Object, 100)
	client.AddReactor("create", resource, func(action core.Action) (bool, runtime.Object, error) {
		createAction := action.(core.CreateAction)
		obj := createAction.GetObject()
		go func() {
			watcher.Add(obj)
			objChan <- obj
		}()
		return true, obj, nil
	})
	return objChan
}

func RegisterCopyOnUpdate(resource string, client *fake_federation_release_1_4.Clientset, watcher *watch.FakeWatcher) chan runtime.Object {
	objChan := make(chan runtime.Object, 100)
	client.AddReactor("update", resource, func(action core.Action) (bool, runtime.Object, error) {
		updateAction := action.(core.UpdateAction)
		obj := updateAction.GetObject()
		go func() {
			watcher.Modify(obj)
			objChan <- obj
		}()
		return true, obj, nil
	})
	return objChan
}

func GetNamespaceFromChan(c chan runtime.Object) *api_v1.Namespace {
	select {
	case obj := <-c:
		namespace := obj.(*api_v1.Namespace)
		return namespace
	case <-time.After(time.Minute):
		return nil
	}
}
