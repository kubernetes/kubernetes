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

package util

import (
	"testing"
	"time"

	federation_api "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fake_federation_release_1_4 "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_4/fake"
	api "k8s.io/kubernetes/pkg/api"
	api_v1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/cache"
	kube_release_1_4 "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_4"
	fake_kube_release_1_4 "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_4/fake"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/stretchr/testify/assert"
)

// Basic test for Federated Informer. Checks whether the subinformer are added and deleted
// when the corresponding cluster entries appear and dissapear from etcd.
func TestFederatedInformer(t *testing.T) {
	fakeFederationClient := &fake_federation_release_1_4.Clientset{}

	// Add a single cluster to federation and remove it when needed.
	cluster := federation_api.Cluster{
		ObjectMeta: api_v1.ObjectMeta{
			Name: "mycluster",
		},
		Status: federation_api.ClusterStatus{
			Conditions: []federation_api.ClusterCondition{
				{Type: federation_api.ClusterReady, Status: api_v1.ConditionTrue},
			},
		},
	}
	fakeFederationClient.AddReactor("list", "clusters", func(action core.Action) (bool, runtime.Object, error) {
		return true, &federation_api.ClusterList{Items: []federation_api.Cluster{cluster}}, nil
	})
	deleteChan := make(chan struct{})
	fakeFederationClient.AddWatchReactor("clusters", func(action core.Action) (bool, watch.Interface, error) {
		fakeWatch := watch.NewFake()
		go func() {
			<-deleteChan
			fakeWatch.Delete(&cluster)
		}()
		return true, fakeWatch, nil
	})

	fakeKubeClient := &fake_kube_release_1_4.Clientset{}
	// There is a single service ns1/s1 in cluster mycluster.
	service := api_v1.Service{
		ObjectMeta: api_v1.ObjectMeta{
			Namespace: "ns1",
			Name:      "s1",
		},
	}
	fakeKubeClient.AddReactor("list", "services", func(action core.Action) (bool, runtime.Object, error) {
		return true, &api_v1.ServiceList{Items: []api_v1.Service{service}}, nil
	})
	fakeKubeClient.AddWatchReactor("services", func(action core.Action) (bool, watch.Interface, error) {
		return true, watch.NewFake(), nil
	})

	targetInformerFactory := func(cluster *federation_api.Cluster, clientset kube_release_1_4.Interface) (cache.Store, framework.ControllerInterface) {
		return framework.NewInformer(
			&cache.ListWatch{
				ListFunc: func(options api.ListOptions) (runtime.Object, error) {
					return clientset.Core().Services(api_v1.NamespaceAll).List(options)
				},
				WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
					return clientset.Core().Services(api_v1.NamespaceAll).Watch(options)
				},
			},
			&api_v1.Service{},
			10*time.Second,
			framework.ResourceEventHandlerFuncs{})
	}

	addedClusters := make(chan string, 1)
	deletedClusters := make(chan string, 1)
	lifecycle := ClusterLifecycleHandlerFuncs{
		ClusterAvailable: func(cluster *federation_api.Cluster) {
			addedClusters <- cluster.Name
			close(addedClusters)
		},
		ClusterUnavailable: func(cluster *federation_api.Cluster, _ []interface{}) {
			deletedClusters <- cluster.Name
			close(deletedClusters)
		},
	}

	informer := NewFederatedInformer(fakeFederationClient, targetInformerFactory, &lifecycle).(*federatedInformerImpl)
	informer.clientFactory = func(cluster *federation_api.Cluster) (kube_release_1_4.Interface, error) {
		return fakeKubeClient, nil
	}
	assert.NotNil(t, informer)
	informer.Start()

	// Wait until mycluster is synced.
	for !informer.GetTargetStore().ClustersSynced([]*federation_api.Cluster{&cluster}) {
		time.Sleep(time.Millisecond * 100)
	}
	readyClusters, err := informer.GetReadyClusters()
	assert.NoError(t, err)
	assert.Contains(t, readyClusters, &cluster)
	serviceList, err := informer.GetTargetStore().List()
	assert.NoError(t, err)
	federatedService := FederatedObject{ClusterName: "mycluster", Object: &service}
	assert.Contains(t, serviceList, federatedService)
	service1, found, err := informer.GetTargetStore().GetByKey("mycluster", "ns1/s1")
	assert.NoError(t, err)
	assert.True(t, found)
	assert.EqualValues(t, &service, service1)
	assert.Equal(t, "mycluster", <-addedClusters)

	// All checked, lets delete the cluster.
	deleteChan <- struct{}{}
	for !informer.GetTargetStore().ClustersSynced([]*federation_api.Cluster{}) {
		time.Sleep(time.Millisecond * 100)
	}
	readyClusters, err = informer.GetReadyClusters()
	assert.NoError(t, err)
	assert.Empty(t, readyClusters)

	serviceList, err = informer.GetTargetStore().List()
	assert.NoError(t, err)
	assert.Empty(t, serviceList)

	assert.Equal(t, "mycluster", <-deletedClusters)

	// Test complete.
	informer.Stop()
}
