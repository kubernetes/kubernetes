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
	fake_federation_release_1_4 "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_4/fake"
	. "k8s.io/kubernetes/federation/pkg/federation-controller/util/test"
	api_v1 "k8s.io/kubernetes/pkg/api/v1"
	kube_release_1_4 "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_4"
	fake_kube_release_1_4 "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_4/fake"
	"k8s.io/kubernetes/pkg/runtime"

	"github.com/stretchr/testify/assert"
)

func TestNamespaceController(t *testing.T) {
	cluster1 := NewCluster("cluster1", api_v1.ConditionTrue)
	cluster2 := NewCluster("cluster2", api_v1.ConditionTrue)

	fakeClient := &fake_federation_release_1_4.Clientset{}
	RegisterFakeList("clusters", &fakeClient.Fake, &federation_api.ClusterList{Items: []federation_api.Cluster{*cluster1}})
	RegisterFakeList("namespaces", &fakeClient.Fake, &api_v1.NamespaceList{Items: []api_v1.Namespace{}})
	namespaceWatch := RegisterFakeWatch("namespaces", &fakeClient.Fake)
	clusterWatch := RegisterFakeWatch("clusters", &fakeClient.Fake)

	cluster1Client := &fake_kube_release_1_4.Clientset{}
	cluster1Watch := RegisterFakeWatch("namespaces", &cluster1Client.Fake)
	RegisterFakeList("namespaces", &cluster1Client.Fake, &api_v1.NamespaceList{Items: []api_v1.Namespace{}})
	cluster1CreateChan := RegisterFakeCopyOnCreate("namespaces", &cluster1Client.Fake, cluster1Watch)
	cluster1UpdateChan := RegisterFakeCopyOnUpdate("namespaces", &cluster1Client.Fake, cluster1Watch)

	cluster2Client := &fake_kube_release_1_4.Clientset{}
	cluster2Watch := RegisterFakeWatch("namespaces", &cluster2Client.Fake)
	RegisterFakeList("namespaces", &cluster2Client.Fake, &api_v1.NamespaceList{Items: []api_v1.Namespace{}})
	cluster2CreateChan := RegisterFakeCopyOnCreate("namespaces", &cluster2Client.Fake, cluster2Watch)

	namespaceController := NewNamespaceController(fakeClient)
	informer := ToFederatedInformerForTestOnly(namespaceController.namespaceFederatedInformer)
	informer.SetClientFactory(func(cluster *federation_api.Cluster) (kube_release_1_4.Interface, error) {
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
			Name:     "test-namespace",
			SelfLink: "/api/v1/namespaces/test-namespace",
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

func GetNamespaceFromChan(c chan runtime.Object) *api_v1.Namespace {
	namespace := GetObjectFromChan(c).(*api_v1.Namespace)
	return namespace

}
