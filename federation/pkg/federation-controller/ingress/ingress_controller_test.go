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
	t.Log("Creating clusters")
	cluster1 := NewCluster("cluster1", api_v1.ConditionTrue)
	cluster2 := NewCluster("cluster2", api_v1.ConditionTrue)
	cfg1 := NewConfigMap("foo")
	cfg2 := NewConfigMap("bar") // Different UID from cfg1, so that we can check that they get reconciled.

	t.Log("Creating watches")
	fedClient := &fake_federation_release_1_4.Clientset{}
	RegisterFakeList("clusters", &fedClient.Fake, &federation_api.ClusterList{Items: []federation_api.Cluster{*cluster1}})
	RegisterFakeList("ingresses", &fedClient.Fake, &extensions_v1beta1.IngressList{Items: []extensions_v1beta1.Ingress{}})
	// RegisterFakeList("configmaps", &fedClient.Fake, &api_v1.ConfigMapList{Items: []api_v1.ConfigMap{}})
	fedIngressWatch := RegisterFakeWatch("ingresses", &fedClient.Fake)
	clusterWatch := RegisterFakeWatch("clusters", &fedClient.Fake)

	cluster1Client := &fake_kube_release_1_4.Clientset{}
	RegisterFakeList("ingresses", &cluster1Client.Fake, &extensions_v1beta1.IngressList{Items: []extensions_v1beta1.Ingress{}})
	RegisterFakeList("configmaps", &cluster1Client.Fake, &api_v1.ConfigMapList{Items: []api_v1.ConfigMap{*cfg1}})
	cluster1IngressWatch := RegisterFakeWatch("ingresses", &cluster1Client.Fake)
	cluster1ConfigMapWatch := RegisterFakeWatch("configmaps", &cluster1Client.Fake)
	cluster1IngressCreateChan := RegisterFakeCopyOnCreate("ingresses", &cluster1Client.Fake, cluster1IngressWatch)
	cluster1IngressUpdateChan := RegisterFakeCopyOnUpdate("ingresses", &cluster1Client.Fake, cluster1IngressWatch)
	//cluster1ConfigMapCreateChan := RegisterFakeCopyOnCreate("configmaps", &cluster1Client.Fake, cluster1ConfigMapWatch)
	//cluster1ConfigMapUpdateChan := RegisterFakeCopyOnUpdate("configmaps", &cluster1Client.Fake, cluster1ConfigMapWatch)

	cluster2Client := &fake_kube_release_1_4.Clientset{}
	RegisterFakeList("ingresses", &cluster2Client.Fake, &extensions_v1beta1.IngressList{Items: []extensions_v1beta1.Ingress{}})
	RegisterFakeList("configmaps", &cluster2Client.Fake, &api_v1.ConfigMapList{Items: []api_v1.ConfigMap{*cfg2}})
	cluster2IngressWatch := RegisterFakeWatch("ingresses", &cluster2Client.Fake)
	cluster2ConfigMapWatch := RegisterFakeWatch("configmaps", &cluster2Client.Fake)
	cluster2IngressCreateChan := RegisterFakeCopyOnCreate("ingresses", &cluster2Client.Fake, cluster2IngressWatch)
	cluster2IngressUpdateChan := RegisterFakeCopyOnUpdate("ingresses", &cluster2Client.Fake, cluster2IngressWatch)
	//cluster2ConfigMapCreateChan := RegisterFakeCopyOnUpdate("configmaps", &cluster2Client.Fake, cluster2ConfigMapWatch)
	cluster2ConfigMapUpdateChan := RegisterFakeCopyOnUpdate("configmaps", &cluster2Client.Fake, cluster2ConfigMapWatch)

	t.Log("Creating Ingress Controller")
	ingressController := NewIngressController(fedClient)
	informer := ToFederatedInformerForTestOnly(ingressController.ingressFederatedInformer)
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
	ingressController.clusterAvailableDelay = time.Second
	ingressController.ingressReviewDelay = 50 * time.Millisecond
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
	clusterWatch.Add(cluster2)
	t.Log("Checking that the ingress got created in cluster 2")
	createdIngress2 := GetIngressFromChan(cluster2IngressCreateChan)
	assert.NotNil(t, createdIngress2)
	assert.True(t, reflect.DeepEqual(&ing1, createdIngress2))

	_ = cfg1 // TODO REMOVE
	_ = cfg2
	_ = cluster1ConfigMapWatch
	_ = cluster2ConfigMapWatch
	/*
		_ = cluster1ConfigMapCreateChan
		_ = cluster2ConfigMapCreateChan
		_ = cluster1ConfigMapUpdateChan
		_ = cluster2ConfigMapUpdateChan
	*/
	_ = cluster2IngressUpdateChan
	/*
		t.Log("Adding Ingress UID ConfigMaps for clusters")
		cluster1ConfigMapWatch.Add(&cfg1)
		cluster2ConfigMapWatch.Add(&cfg2)
	*/

	t.Log("Checking that the configmap in cluster 2 got updated.")
	updatedConfigMap2 := GetConfigMapFromChan(cluster2ConfigMapUpdateChan)
	assert.NotNil(t, updatedConfigMap2)
	assert.Equal(t, cfg1.Data["uid"], updatedConfigMap2.Data["uid"], fmt.Sprintf("UID's in configmaps in cluster's 1 and 2 are not equal (%q != %q)", cfg1.Data["uid"], updatedConfigMap2.Data["uid"]))
	/*
		updatedConfigMap1 := GetConfigMapFromChan(cluster1ConfigMapUpdateChan) // TODO: Remove this check - only for debugging purposes.
		assert.NotNil(t, updatedConfigMap1, "UID in configmap in cluster 1 was not updated")
		assert.Equal(t, cfg1.Data["uid"], updatedConfigMap1.Data["uid"], fmt.Sprintf("UID in configmaps in cluster 1 and 2 are not equal", cfg1.Data["uid"], updatedConfigMap2.Data["uid"]))
	*/
	close(stop)
}

func GetIngressFromChan(c chan runtime.Object) *extensions_v1beta1.Ingress {
	ingress := GetObjectFromChan(c).(*extensions_v1beta1.Ingress)
	return ingress
}

func GetConfigMapFromChan(c chan runtime.Object) *api_v1.ConfigMap {
	configMap := GetObjectFromChan(c).(*api_v1.ConfigMap)
	return configMap
}

func NewConfigMap(uid string) *api_v1.ConfigMap {
	return &api_v1.ConfigMap{
		ObjectMeta: api_v1.ObjectMeta{
			Name:      uidConfigMapName,
			Namespace: uidConfigMapNamespace,
			SelfLink:  "/api/v1/namespaces/" + uidConfigMapNamespace + "/configmap/" + uidConfigMapName,
		},
		Data: map[string]string{
			uidKey: uid,
		},
	}
}
