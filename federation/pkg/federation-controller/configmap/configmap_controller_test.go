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

package configmap

import (
	"fmt"
	"testing"
	"time"

	federation_api "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fake_fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_5/fake"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	. "k8s.io/kubernetes/federation/pkg/federation-controller/util/test"
	api_v1 "k8s.io/kubernetes/pkg/api/v1"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	fake_kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/fake"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/wait"

	"github.com/stretchr/testify/assert"
)

func TestConfigMapController(t *testing.T) {
	cluster1 := NewCluster("cluster1", api_v1.ConditionTrue)
	cluster2 := NewCluster("cluster2", api_v1.ConditionTrue)

	fakeClient := &fake_fedclientset.Clientset{}
	RegisterFakeList("clusters", &fakeClient.Fake, &federation_api.ClusterList{Items: []federation_api.Cluster{*cluster1}})
	RegisterFakeList("configmaps", &fakeClient.Fake, &api_v1.ConfigMapList{Items: []api_v1.ConfigMap{}})
	configmapWatch := RegisterFakeWatch("configmaps", &fakeClient.Fake)
	clusterWatch := RegisterFakeWatch("clusters", &fakeClient.Fake)

	cluster1Client := &fake_kubeclientset.Clientset{}
	cluster1Watch := RegisterFakeWatch("configmaps", &cluster1Client.Fake)
	RegisterFakeList("configmaps", &cluster1Client.Fake, &api_v1.ConfigMapList{Items: []api_v1.ConfigMap{}})
	cluster1CreateChan := RegisterFakeCopyOnCreate("configmaps", &cluster1Client.Fake, cluster1Watch)
	cluster1UpdateChan := RegisterFakeCopyOnUpdate("configmaps", &cluster1Client.Fake, cluster1Watch)

	cluster2Client := &fake_kubeclientset.Clientset{}
	cluster2Watch := RegisterFakeWatch("configmaps", &cluster2Client.Fake)
	RegisterFakeList("configmaps", &cluster2Client.Fake, &api_v1.ConfigMapList{Items: []api_v1.ConfigMap{}})
	cluster2CreateChan := RegisterFakeCopyOnCreate("configmaps", &cluster2Client.Fake, cluster2Watch)

	configmapController := NewConfigMapController(fakeClient)
	informer := ToFederatedInformerForTestOnly(configmapController.configmapFederatedInformer)
	informer.SetClientFactory(func(cluster *federation_api.Cluster) (kubeclientset.Interface, error) {
		switch cluster.Name {
		case cluster1.Name:
			return cluster1Client, nil
		case cluster2.Name:
			return cluster2Client, nil
		default:
			return nil, fmt.Errorf("Unknown cluster")
		}
	})

	configmapController.clusterAvailableDelay = time.Second
	configmapController.configmapReviewDelay = 50 * time.Millisecond
	configmapController.smallDelay = 20 * time.Millisecond
	configmapController.updateTimeout = 5 * time.Second

	stop := make(chan struct{})
	configmapController.Run(stop)

	configmap1 := &api_v1.ConfigMap{
		ObjectMeta: api_v1.ObjectMeta{
			Name:      "test-configmap",
			Namespace: "ns",
			SelfLink:  "/api/v1/namespaces/ns/configmaps/test-configmap",
		},
		Data: map[string]string{
			"A": "ala ma kota",
			"B": "quick brown fox",
		},
	}

	// Test add federated configmap.
	configmapWatch.Add(configmap1)
	createdConfigMap := GetConfigMapFromChan(cluster1CreateChan)
	assert.NotNil(t, createdConfigMap)
	assert.Equal(t, configmap1.Namespace, createdConfigMap.Namespace)
	assert.Equal(t, configmap1.Name, createdConfigMap.Name)
	assert.True(t, util.ConfigMapEquivalent(configmap1, createdConfigMap))

	// Wait for the configmap to appear in the informer store
	err := WaitForStoreUpdate(
		configmapController.configmapFederatedInformer.GetTargetStore(),
		cluster1.Name, types.NamespacedName{Namespace: configmap1.Namespace, Name: configmap1.Name}.String(), wait.ForeverTestTimeout)
	assert.Nil(t, err, "configmap should have appeared in the informer store")

	// Test update federated configmap.
	configmap1.Annotations = map[string]string{
		"A": "B",
	}
	configmapWatch.Modify(configmap1)
	updatedConfigMap := GetConfigMapFromChan(cluster1UpdateChan)
	assert.NotNil(t, updatedConfigMap)
	assert.Equal(t, configmap1.Name, updatedConfigMap.Name)
	assert.Equal(t, configmap1.Namespace, updatedConfigMap.Namespace)
	assert.True(t, util.ConfigMapEquivalent(configmap1, updatedConfigMap))

	// Test update federated configmap.
	configmap1.Data = map[string]string{
		"config": "myconfigurationfile",
	}
	configmapWatch.Modify(configmap1)
	updatedConfigMap2 := GetConfigMapFromChan(cluster1UpdateChan)
	assert.NotNil(t, updatedConfigMap)
	assert.Equal(t, configmap1.Name, updatedConfigMap.Name)
	assert.Equal(t, configmap1.Namespace, updatedConfigMap.Namespace)
	assert.True(t, util.ConfigMapEquivalent(configmap1, updatedConfigMap2))

	// Test add cluster
	clusterWatch.Add(cluster2)
	createdConfigMap2 := GetConfigMapFromChan(cluster2CreateChan)
	assert.NotNil(t, createdConfigMap2)
	assert.Equal(t, configmap1.Name, createdConfigMap2.Name)
	assert.Equal(t, configmap1.Namespace, createdConfigMap2.Namespace)
	assert.True(t, util.ConfigMapEquivalent(configmap1, createdConfigMap2))

	close(stop)
}

func GetConfigMapFromChan(c chan runtime.Object) *api_v1.ConfigMap {
	configmap := GetObjectFromChan(c).(*api_v1.ConfigMap)
	return configmap
}
