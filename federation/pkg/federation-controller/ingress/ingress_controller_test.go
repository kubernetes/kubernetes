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

package ingress

import (
	"fmt"
	"reflect"
	"testing"
	"time"

	federation_api "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fake_federation_release_1_4 "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_4/fake"
	. "k8s.io/kubernetes/federation/pkg/federation-controller/util/test"
	api_v1 "k8s.io/kubernetes/pkg/api/v1"
	extensions_v1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	kube_release_1_4 "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_4"
	fake_kube_release_1_4 "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_4/fake"
	"k8s.io/kubernetes/pkg/runtime"

	"github.com/stretchr/testify/assert"
)

func TestIngressController(t *testing.T) {
	fakeClusterList := federation_api.ClusterList{Items: []federation_api.Cluster{}}
	fakeConfigMapList1 := api_v1.ConfigMapList{Items: []api_v1.ConfigMap{}}
	fakeConfigMapList2 := api_v1.ConfigMapList{Items: []api_v1.ConfigMap{}}
	cluster1 := NewCluster("cluster1", api_v1.ConditionTrue)
	cluster2 := NewCluster("cluster2", api_v1.ConditionTrue)
	cfg1 := NewConfigMap("foo")
	cluster1.ObjectMeta.Annotations[uidAnnotationKey] = "foo"
	cfg2 := NewConfigMap("bar") // Different UID from cfg1, so that we can check that they get reconciled.

	t.Log("Creating watches")
	fedClient := &fake_federation_release_1_4.Clientset{}
	RegisterFakeList("clusters", &fedClient.Fake, &fakeClusterList)
	RegisterFakeList("ingresses", &fedClient.Fake, &extensions_v1beta1.IngressList{Items: []extensions_v1beta1.Ingress{}})
	fedIngressWatch := RegisterFakeWatch("ingresses", &fedClient.Fake)
	clusterWatch := RegisterFakeWatch("clusters", &fedClient.Fake)
	fedClusterCreateChan := RegisterFakeCopyOnCreate("clusters", &fedClient.Fake, clusterWatch)
	fedClusterUpdateChan := RegisterFakeCopyOnUpdate("clusters", &fedClient.Fake, clusterWatch)

	cluster1IngressClient := &fake_kube_release_1_4.Clientset{}
	cluster1ConfigMapClient := &fake_kube_release_1_4.Clientset{}
	RegisterFakeList("ingresses", &cluster1IngressClient.Fake, &extensions_v1beta1.IngressList{Items: []extensions_v1beta1.Ingress{}})
	RegisterFakeList("configmaps", &cluster1ConfigMapClient.Fake, &fakeConfigMapList1)
	cluster1IngressWatch := RegisterFakeWatch("ingresses", &cluster1IngressClient.Fake)
	cluster1ConfigMapWatch := RegisterFakeWatch("configmaps", &cluster1ConfigMapClient.Fake)
	cluster1IngressCreateChan := RegisterFakeCopyOnCreate("ingresses", &cluster1IngressClient.Fake, cluster1IngressWatch)
	cluster1IngressUpdateChan := RegisterFakeCopyOnUpdate("ingresses", &cluster1IngressClient.Fake, cluster1IngressWatch)
	cluster1ConfigMapCreateChan := RegisterFakeCopyOnCreate("configmaps", &cluster1ConfigMapClient.Fake, cluster1ConfigMapWatch)
	cluster1ConfigMapUpdateChan := RegisterFakeCopyOnUpdate("configmaps", &cluster1ConfigMapClient.Fake, cluster1ConfigMapWatch)

	cluster2IngressClient := &fake_kube_release_1_4.Clientset{}
	cluster2ConfigMapClient := &fake_kube_release_1_4.Clientset{}
	RegisterFakeList("ingresses", &cluster2IngressClient.Fake, &extensions_v1beta1.IngressList{Items: []extensions_v1beta1.Ingress{}})
	RegisterFakeList("configmaps", &cluster2ConfigMapClient.Fake, &fakeConfigMapList2)
	cluster2IngressWatch := RegisterFakeWatch("ingresses", &cluster2IngressClient.Fake)
	cluster2ConfigMapWatch := RegisterFakeWatch("configmaps", &cluster2ConfigMapClient.Fake)
	cluster2IngressCreateChan := RegisterFakeCopyOnCreate("ingresses", &cluster2IngressClient.Fake, cluster2IngressWatch)
	cluster2IngressUpdateChan := RegisterFakeCopyOnUpdate("ingresses", &cluster2IngressClient.Fake, cluster2IngressWatch)
	cluster2ConfigMapCreateChan := RegisterFakeCopyOnCreate("configmaps", &cluster2ConfigMapClient.Fake, cluster2ConfigMapWatch)
	cluster2ConfigMapUpdateChan := RegisterFakeCopyOnUpdate("configmaps", &cluster2ConfigMapClient.Fake, cluster2ConfigMapWatch)

	t.Log("Creating Ingress Controller")
	ingressClientFactoryFunc := func(cluster *federation_api.Cluster) (kube_release_1_4.Interface, error) {
		switch cluster.Name {
		case cluster1.Name:
			return cluster1IngressClient, nil
		case cluster2.Name:
			return cluster2IngressClient, nil
		default:
			return nil, fmt.Errorf("Unknown cluster")
		}
	}
	configMapClientFactoryFunc := func(cluster *federation_api.Cluster) (kube_release_1_4.Interface, error) {
		switch cluster.Name {
		case cluster1.Name:
			return cluster1ConfigMapClient, nil
		case cluster2.Name:
			return cluster2ConfigMapClient, nil
		default:
			return nil, fmt.Errorf("Unknown cluster")
		}
	}
	ingressController := NewIngressController(fedClient)
	ingressInformer := ToFederatedInformerForTestOnly(ingressController.ingressFederatedInformer)
	ingressInformer.SetClientFactory(ingressClientFactoryFunc)
	configMapInformer := ToFederatedInformerForTestOnly(ingressController.configMapFederatedInformer)
	configMapInformer.SetClientFactory(configMapClientFactoryFunc)
	ingressController.clusterAvailableDelay = time.Second
	ingressController.ingressReviewDelay = 500 * time.Millisecond
	ingressController.configMapReviewDelay = 500 * time.Millisecond
	ingressController.smallDelay = 20 * time.Millisecond
	ingressController.updateTimeout = 5 * time.Second

	stop := make(chan struct{})
	t.Log("Running Ingress Controller")
	ingressController.Run(stop)

	ing1 := extensions_v1beta1.Ingress{
		ObjectMeta: api_v1.ObjectMeta{
			Name:      "test-ingress",
			Namespace: "mynamespace",
			SelfLink:  "/api/v1/namespaces/mynamespace/ingress/test-ingress",
		},
	}

	t.Log("Adding cluster1")
	fakeClusterList.Items = append(fakeClusterList.Items, *cluster1)
	clusterWatch.Add(cluster1)

	t.Log("Adding Ingress UID ConfigMap for cluster 1")
	fakeConfigMapList1.Items = append(fakeConfigMapList1.Items, *cfg1)
	cluster1ConfigMapWatch.Add(cfg1)

	// Test add federated ingress.
	t.Log("Adding Federated Ingress")
	fedIngressWatch.Add(&ing1)
	createdIngress := GetIngressFromChan(cluster1IngressCreateChan)
	assert.NotNil(t, createdIngress)
	assert.True(t, reflect.DeepEqual(&ing1, createdIngress))
	// Test update federated ingress.
	ing1.Annotations = map[string]string{
		"A": "B",
	}
	t.Log("Modifying Federated Ingress")
	fedIngressWatch.Modify(&ing1)
	updatedIngress := GetIngressFromChan(cluster1IngressUpdateChan)
	assert.NotNil(t, updatedIngress)
	assert.True(t, reflect.DeepEqual(&ing1, updatedIngress))

	// Test add cluster
	t.Log("Adding a second cluster")
	ing1.Annotations[staticIPAnnotationKey] = "foo" // Make sure that the base object has a static IP name first.
	fedIngressWatch.Modify(&ing1)
	fakeClusterList.Items = append(fakeClusterList.Items, *cluster2)
	fakeConfigMapList2.Items = append(fakeConfigMapList2.Items, *cfg2)
	for i := 0; i < 10; i++ {
		clusterWatch.Add(cluster2)
	}
	cluster2ConfigMapWatch.Add(cfg2)
	t.Log("Checking that the ingress got created in cluster 2")
	createdIngress2 := GetIngressFromChan(cluster2IngressCreateChan)
	assert.NotNil(t, createdIngress2)
	assert.True(t, reflect.DeepEqual(&ing1, createdIngress2))

	_ = cfg1 // TODO REMOVE
	_ = cfg2
	_ = ing1
	_ = fedIngressWatch
	_ = clusterWatch
	_ = fedClusterCreateChan
	_ = fedClusterUpdateChan
	_ = cluster1ConfigMapWatch
	_ = cluster2ConfigMapWatch
	_ = cluster1ConfigMapCreateChan
	_ = cluster2ConfigMapCreateChan
	_ = cluster1ConfigMapUpdateChan
	_ = cluster2ConfigMapUpdateChan
	_ = cluster1IngressCreateChan
	_ = cluster2IngressCreateChan
	_ = cluster1IngressUpdateChan
	_ = cluster2IngressUpdateChan
	t.Log("Checking that the configmap in cluster 2 got updated.")
	updatedConfigMap2 := GetConfigMapFromChan(cluster2ConfigMapUpdateChan)
	assert.NotNil(t, updatedConfigMap2)
	if updatedConfigMap2 != nil {
		assert.Equal(t, cfg1.Data["uid"], updatedConfigMap2.Data["uid"], fmt.Sprintf("UID's in configmaps in cluster's 1 and 2 are not equal (%q != %q)", cfg1.Data["uid"], updatedConfigMap2.Data["uid"]))
	}

	/*
		updatedConfigMap1 := GetConfigMapFromChan(cluster1ConfigMapUpdateChan) // TODO: Remove this check - only for debugging purposes.
		assert.NotNil(t, updatedConfigMap1, "UID in configmap in cluster 1 was not updated")
		assert.Equal(t, cfg1.Data["uid"], updatedConfigMap1.Data["uid"], fmt.Sprintf("UID in configmaps in cluster 1 and 2 are not equal", cfg1.Data["uid"], updatedConfigMap2.Data["uid"]))
	*/
	time.Sleep(10 * time.Second) // Wait to see what other things the controller processes.
	// TODO defer close(stop)
}

func GetIngressFromChan(c chan runtime.Object) *extensions_v1beta1.Ingress {
	ingress, _ := GetObjectFromChan(c).(*extensions_v1beta1.Ingress)
	return ingress
}

func GetConfigMapFromChan(c chan runtime.Object) *api_v1.ConfigMap {
	configMap, _ := GetObjectFromChan(c).(*api_v1.ConfigMap)
	return configMap
}

func NewConfigMap(uid string) *api_v1.ConfigMap {
	return &api_v1.ConfigMap{
		ObjectMeta: api_v1.ObjectMeta{
			Name:        uidConfigMapName,
			Namespace:   uidConfigMapNamespace,
			SelfLink:    "/api/v1/namespaces/" + uidConfigMapNamespace + "/configmap/" + uidConfigMapName,
			Annotations: map[string]string{},
		},
		Data: map[string]string{
			uidKey: uid,
		},
	}
}
