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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fakefedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/fake"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/deletionhelper"
	. "k8s.io/kubernetes/federation/pkg/federation-controller/util/test"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	fakekubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"

	"github.com/golang/glog"
	"github.com/stretchr/testify/assert"
)

const (
	configmaps       string = "configmaps"
	clusters         string = "clusters"
	informerStoreErr string = "configmap should have appeared in the informer store"
)

func TestConfigMapController(t *testing.T) {
	cluster1 := NewCluster("cluster1", apiv1.ConditionTrue)
	cluster2 := NewCluster("cluster2", apiv1.ConditionTrue)

	fakeClient := &fakefedclientset.Clientset{}
	RegisterFakeList(clusters, &fakeClient.Fake, &federationapi.ClusterList{Items: []federationapi.Cluster{*cluster1}})
	RegisterFakeList(configmaps, &fakeClient.Fake, &apiv1.ConfigMapList{Items: []apiv1.ConfigMap{}})
	configmapWatch := RegisterFakeWatch(configmaps, &fakeClient.Fake)
	configmapUpdateChan := RegisterFakeCopyOnUpdate(configmaps, &fakeClient.Fake, configmapWatch)
	clusterWatch := RegisterFakeWatch(clusters, &fakeClient.Fake)

	cluster1Client := &fakekubeclientset.Clientset{}
	cluster1Watch := RegisterFakeWatch(configmaps, &cluster1Client.Fake)
	RegisterFakeList(configmaps, &cluster1Client.Fake, &apiv1.ConfigMapList{Items: []apiv1.ConfigMap{}})
	cluster1CreateChan := RegisterFakeCopyOnCreate(configmaps, &cluster1Client.Fake, cluster1Watch)
	cluster1UpdateChan := RegisterFakeCopyOnUpdate(configmaps, &cluster1Client.Fake, cluster1Watch)

	cluster2Client := &fakekubeclientset.Clientset{}
	cluster2Watch := RegisterFakeWatch(configmaps, &cluster2Client.Fake)
	RegisterFakeList(configmaps, &cluster2Client.Fake, &apiv1.ConfigMapList{Items: []apiv1.ConfigMap{}})
	cluster2CreateChan := RegisterFakeCopyOnCreate(configmaps, &cluster2Client.Fake, cluster2Watch)

	configmapController := NewConfigMapController(fakeClient)
	informer := ToFederatedInformerForTestOnly(configmapController.configmapFederatedInformer)
	informer.SetClientFactory(func(cluster *federationapi.Cluster) (kubeclientset.Interface, error) {
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

	configmap1 := &apiv1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
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
	// There should be 2 updates to add both the finalizers.
	updatedConfigMap := GetConfigMapFromChan(configmapUpdateChan)
	assert.True(t, configmapController.hasFinalizerFunc(updatedConfigMap, deletionhelper.FinalizerDeleteFromUnderlyingClusters))
	assert.True(t, configmapController.hasFinalizerFunc(updatedConfigMap, metav1.FinalizerOrphanDependents))

	// Verify that the configmap is created in underlying cluster1.
	createdConfigMap := GetConfigMapFromChan(cluster1CreateChan)
	assert.NotNil(t, createdConfigMap)
	assert.Equal(t, configmap1.Namespace, createdConfigMap.Namespace)
	assert.Equal(t, configmap1.Name, createdConfigMap.Name)
	assert.True(t, util.ConfigMapEquivalent(configmap1, createdConfigMap))

	// Wait for the configmap to appear in the informer store
	err := WaitForStoreUpdate(
		configmapController.configmapFederatedInformer.GetTargetStore(),
		cluster1.Name, types.NamespacedName{Namespace: configmap1.Namespace, Name: configmap1.Name}.String(), wait.ForeverTestTimeout)
	assert.Nil(t, err, informerStoreErr)

	// Test update federated configmap.
	configmap1.Annotations = map[string]string{
		"A": "B",
	}
	configmapWatch.Modify(configmap1)
	updatedConfigMap = GetConfigMapFromChan(cluster1UpdateChan)
	assert.NotNil(t, updatedConfigMap)
	assert.Equal(t, configmap1.Name, updatedConfigMap.Name)
	assert.Equal(t, configmap1.Namespace, updatedConfigMap.Namespace)
	assert.True(t, util.ConfigMapEquivalent(configmap1, updatedConfigMap))

	// Wait for the configmap to appear in the informer store
	err = WaitForConfigMapStoreUpdate(
		configmapController.configmapFederatedInformer.GetTargetStore(),
		cluster1.Name, types.NamespacedName{Namespace: configmap1.Namespace, Name: configmap1.Name}.String(),
		configmap1, wait.ForeverTestTimeout)
	assert.Nil(t, err, informerStoreErr)

	// Test update federated configmap.
	configmap1.Data = map[string]string{
		"config": "myconfigurationfile",
	}

	configmapWatch.Modify(configmap1)
	for {
		updatedConfigMap := GetConfigMapFromChan(cluster1UpdateChan)
		assert.NotNil(t, updatedConfigMap)
		if updatedConfigMap == nil {
			break
		}
		assert.Equal(t, configmap1.Name, updatedConfigMap.Name)
		assert.Equal(t, configmap1.Namespace, updatedConfigMap.Namespace)
		if util.ConfigMapEquivalent(configmap1, updatedConfigMap) {
			break
		}
	}

	// Test add cluster
	clusterWatch.Add(cluster2)
	createdConfigMap2 := GetConfigMapFromChan(cluster2CreateChan)
	assert.NotNil(t, createdConfigMap2)
	assert.Equal(t, configmap1.Name, createdConfigMap2.Name)
	assert.Equal(t, configmap1.Namespace, createdConfigMap2.Namespace)
	assert.True(t, util.ConfigMapEquivalent(configmap1, createdConfigMap2))

	close(stop)
}

func GetConfigMapFromChan(c chan runtime.Object) *apiv1.ConfigMap {
	if configmap := GetObjectFromChan(c); configmap == nil {
		return nil
	} else {
		return configmap.(*apiv1.ConfigMap)
	}
}

// Wait till the store is updated with latest configmap.
func WaitForConfigMapStoreUpdate(store util.FederatedReadOnlyStore, clusterName, key string, desiredConfigMap *apiv1.ConfigMap, timeout time.Duration) error {
	retryInterval := 200 * time.Millisecond
	err := wait.PollImmediate(retryInterval, timeout, func() (bool, error) {
		obj, found, err := store.GetByKey(clusterName, key)
		if !found || err != nil {
			glog.Infof("%s is not in the store", key)
			return false, err
		}
		equal := util.ConfigMapEquivalent(obj.(*apiv1.ConfigMap), desiredConfigMap)
		if !equal {
			glog.Infof("wrong content in the store expected:\n%v\nactual:\n%v\n", *desiredConfigMap, *obj.(*apiv1.ConfigMap))
		}
		return equal, err
	})
	return err
}
