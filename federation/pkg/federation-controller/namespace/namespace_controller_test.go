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
	fake_fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_5/fake"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/deletionhelper"
	. "k8s.io/kubernetes/federation/pkg/federation-controller/util/test"
	"k8s.io/kubernetes/pkg/api/unversioned"
	api_v1 "k8s.io/kubernetes/pkg/api/v1"
	extensionsv1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	fake_kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/fake"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/wait"

	"github.com/stretchr/testify/assert"
)

func TestNamespaceController(t *testing.T) {
	cluster1 := NewCluster("cluster1", api_v1.ConditionTrue)
	cluster2 := NewCluster("cluster2", api_v1.ConditionTrue)
	ns1 := api_v1.Namespace{
		ObjectMeta: api_v1.ObjectMeta{
			Name:     "test-namespace",
			SelfLink: "/api/v1/namespaces/test-namespace",
		},
		Spec: api_v1.NamespaceSpec{
			Finalizers: []api_v1.FinalizerName{api_v1.FinalizerKubernetes},
		},
	}

	fakeClient := &fake_fedclientset.Clientset{}
	RegisterFakeList("clusters", &fakeClient.Fake, &federation_api.ClusterList{Items: []federation_api.Cluster{*cluster1}})
	RegisterFakeList("namespaces", &fakeClient.Fake, &api_v1.NamespaceList{Items: []api_v1.Namespace{}})
	namespaceWatch := RegisterFakeWatch("namespaces", &fakeClient.Fake)
	namespaceCreateChan := RegisterFakeCopyOnCreate("namespaces", &fakeClient.Fake, namespaceWatch)
	clusterWatch := RegisterFakeWatch("clusters", &fakeClient.Fake)

	cluster1Client := &fake_kubeclientset.Clientset{}
	cluster1Watch := RegisterFakeWatch("namespaces", &cluster1Client.Fake)
	RegisterFakeList("namespaces", &cluster1Client.Fake, &api_v1.NamespaceList{Items: []api_v1.Namespace{}})
	cluster1CreateChan := RegisterFakeCopyOnCreate("namespaces", &cluster1Client.Fake, cluster1Watch)
	//	cluster1UpdateChan := RegisterFakeCopyOnUpdate("namespaces", &cluster1Client.Fake, cluster1Watch)

	cluster2Client := &fake_kubeclientset.Clientset{}
	cluster2Watch := RegisterFakeWatch("namespaces", &cluster2Client.Fake)
	RegisterFakeList("namespaces", &cluster2Client.Fake, &api_v1.NamespaceList{Items: []api_v1.Namespace{}})
	cluster2CreateChan := RegisterFakeCopyOnCreate("namespaces", &cluster2Client.Fake, cluster2Watch)

	RegisterFakeList("replicasets", &fakeClient.Fake, &extensionsv1.ReplicaSetList{Items: []extensionsv1.ReplicaSet{
		{
			ObjectMeta: api_v1.ObjectMeta{
				Name:      "test-rs",
				Namespace: ns1.Namespace,
			}}}})
	RegisterFakeList("secrets", &fakeClient.Fake, &api_v1.SecretList{Items: []api_v1.Secret{
		{
			ObjectMeta: api_v1.ObjectMeta{
				Name:      "test-secret",
				Namespace: ns1.Namespace,
			}}}})
	RegisterFakeList("services", &fakeClient.Fake, &api_v1.ServiceList{Items: []api_v1.Service{
		{
			ObjectMeta: api_v1.ObjectMeta{
				Name:      "test-service",
				Namespace: ns1.Namespace,
			}}}})
	nsDeleteChan := RegisterDelete(&fakeClient.Fake, "namespaces")
	rsDeleteChan := RegisterDeleteCollection(&fakeClient.Fake, "replicasets")
	serviceDeleteChan := RegisterDeleteCollection(&fakeClient.Fake, "services")
	secretDeleteChan := RegisterDeleteCollection(&fakeClient.Fake, "secrets")

	namespaceController := NewNamespaceController(fakeClient)
	informerClientFactory := func(cluster *federation_api.Cluster) (kubeclientset.Interface, error) {
		switch cluster.Name {
		case cluster1.Name:
			return cluster1Client, nil
		case cluster2.Name:
			return cluster2Client, nil
		default:
			return nil, fmt.Errorf("Unknown cluster")
		}
	}
	setClientFactory(namespaceController.namespaceFederatedInformer, informerClientFactory)
	namespaceController.clusterAvailableDelay = time.Second
	namespaceController.namespaceReviewDelay = 50 * time.Millisecond
	namespaceController.smallDelay = 20 * time.Millisecond
	namespaceController.updateTimeout = 5 * time.Second

	stop := make(chan struct{})
	namespaceController.Run(stop)

	// Test add federated namespace.
	namespaceWatch.Add(&ns1)
	// Verify that the DeleteFromUnderlyingClusters finalizer is added to the namespace.
	// Note: finalize invokes the create action in Fake client.
	// TODO: Seems like a bug. Should invoke update. Fix it.
	updatedNamespace := GetNamespaceFromChan(namespaceCreateChan)
	assert.True(t, namespaceController.hasFinalizerFunc(updatedNamespace, deletionhelper.FinalizerDeleteFromUnderlyingClusters))
	ns1 = *updatedNamespace

	// Verify that the namespace is created in underlying cluster1.
	createdNamespace := GetNamespaceFromChan(cluster1CreateChan)
	assert.NotNil(t, createdNamespace)
	assert.Equal(t, ns1.Name, createdNamespace.Name)

	// Wait for the namespace to appear in the informer store
	err := WaitForStoreUpdate(
		namespaceController.namespaceFederatedInformer.GetTargetStore(),
		cluster1.Name, ns1.Name, wait.ForeverTestTimeout)
	assert.Nil(t, err, "namespace should have appeared in the informer store")

	/*
		        // TODO: Uncomment this once we have figured out why this is flaky.
			// Test update federated namespace.
			ns1.Annotations = map[string]string{
				"A": "B",
			}
			namespaceWatch.Modify(&ns1)
			updatedNamespace = GetNamespaceFromChan(cluster1UpdateChan)
			assert.NotNil(t, updatedNamespace)
			assert.Equal(t, ns1.Name, updatedNamespace.Name)
			// assert.Contains(t, updatedNamespace.Annotations, "A")
	*/

	// Test add cluster
	clusterWatch.Add(cluster2)
	createdNamespace2 := GetNamespaceFromChan(cluster2CreateChan)
	assert.NotNil(t, createdNamespace2)
	assert.Equal(t, ns1.Name, createdNamespace2.Name)
	// assert.Contains(t, createdNamespace2.Annotations, "A")

	// Delete the namespace with orphan finalizer (let namespaces
	// in underlying clusters be as is).
	// TODO: Add a test without orphan finalizer.
	ns1.ObjectMeta.Finalizers = append(ns1.ObjectMeta.Finalizers, api_v1.FinalizerOrphan)
	ns1.DeletionTimestamp = &unversioned.Time{Time: time.Now()}
	namespaceWatch.Modify(&ns1)
	assert.Equal(t, ns1.Name, GetStringFromChan(nsDeleteChan))
	assert.Equal(t, "all", GetStringFromChan(rsDeleteChan))
	assert.Equal(t, "all", GetStringFromChan(serviceDeleteChan))
	assert.Equal(t, "all", GetStringFromChan(secretDeleteChan))

	close(stop)
}

func setClientFactory(informer util.FederatedInformer, informerClientFactory func(*federation_api.Cluster) (kubeclientset.Interface, error)) {
	testInformer := ToFederatedInformerForTestOnly(informer)
	testInformer.SetClientFactory(informerClientFactory)
}

func RegisterDeleteCollection(client *core.Fake, resource string) chan string {
	deleteChan := make(chan string, 100)
	client.AddReactor("delete-collection", resource, func(action core.Action) (bool, runtime.Object, error) {
		deleteChan <- "all"
		return true, nil, nil
	})
	return deleteChan
}

func RegisterDelete(client *core.Fake, resource string) chan string {
	deleteChan := make(chan string, 100)
	client.AddReactor("delete", resource, func(action core.Action) (bool, runtime.Object, error) {
		deleteAction := action.(core.DeleteAction)
		deleteChan <- deleteAction.GetName()
		return true, nil, nil
	})
	return deleteChan
}

func GetStringFromChan(c chan string) string {
	select {
	case str := <-c:
		return str
	case <-time.After(5 * time.Second):
		return "timedout"
	}
}

func GetNamespaceFromChan(c chan runtime.Object) *api_v1.Namespace {
	namespace := GetObjectFromChan(c).(*api_v1.Namespace)
	return namespace

}
