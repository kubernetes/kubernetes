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
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
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
	cfg2 := NewConfigMap("bar") // Different UID from cfg1, so that we can check that they get reconciled.

	t.Log("Creating fake infrastructure")
	fedClient := &fake_federation_release_1_4.Clientset{}
	RegisterFakeList("clusters", &fedClient.Fake, &fakeClusterList)
	RegisterFakeList("ingresses", &fedClient.Fake, &extensions_v1beta1.IngressList{Items: []extensions_v1beta1.Ingress{}})
	fedIngressWatch := RegisterFakeWatch("ingresses", &fedClient.Fake)
	clusterWatch := RegisterFakeWatch("clusters", &fedClient.Fake)
	fedClusterUpdateChan := RegisterFakeCopyOnUpdate("clusters", &fedClient.Fake, clusterWatch)
	fedIngressUpdateChan := RegisterFakeCopyOnUpdate("ingresses", &fedClient.Fake, fedIngressWatch)

	cluster1Client := &fake_kube_release_1_4.Clientset{}
	RegisterFakeList("ingresses", &cluster1Client.Fake, &extensions_v1beta1.IngressList{Items: []extensions_v1beta1.Ingress{}})
	RegisterFakeList("configmaps", &cluster1Client.Fake, &fakeConfigMapList1)
	cluster1IngressWatch := RegisterFakeWatch("ingresses", &cluster1Client.Fake)
	cluster1ConfigMapWatch := RegisterFakeWatch("configmaps", &cluster1Client.Fake)
	cluster1IngressCreateChan := RegisterFakeCopyOnCreate("ingresses", &cluster1Client.Fake, cluster1IngressWatch)
	cluster1IngressUpdateChan := RegisterFakeCopyOnUpdate("ingresses", &cluster1Client.Fake, cluster1IngressWatch)

	cluster2Client := &fake_kube_release_1_4.Clientset{}
	RegisterFakeList("ingresses", &cluster2Client.Fake, &extensions_v1beta1.IngressList{Items: []extensions_v1beta1.Ingress{}})
	RegisterFakeList("configmaps", &cluster2Client.Fake, &fakeConfigMapList2)
	cluster2IngressWatch := RegisterFakeWatch("ingresses", &cluster2Client.Fake)
	cluster2ConfigMapWatch := RegisterFakeWatch("configmaps", &cluster2Client.Fake)
	cluster2IngressCreateChan := RegisterFakeCopyOnCreate("ingresses", &cluster2Client.Fake, cluster2IngressWatch)
	cluster2ConfigMapUpdateChan := RegisterFakeCopyOnUpdate("configmaps", &cluster2Client.Fake, cluster2ConfigMapWatch)

	clientFactoryFunc := func(cluster *federation_api.Cluster) (kube_release_1_4.Interface, error) {
		switch cluster.Name {
		case cluster1.Name:
			return cluster1Client, nil
		case cluster2.Name:
			return cluster2Client, nil
		default:
			return nil, fmt.Errorf("Unknown cluster")
		}
	}
	ingressController := NewIngressController(fedClient)
	ingressInformer := ToFederatedInformerForTestOnly(ingressController.ingressFederatedInformer)
	ingressInformer.SetClientFactory(clientFactoryFunc)
	configMapInformer := ToFederatedInformerForTestOnly(ingressController.configMapFederatedInformer)
	configMapInformer.SetClientFactory(clientFactoryFunc)
	ingressController.clusterAvailableDelay = time.Second
	ingressController.ingressReviewDelay = 50 * time.Millisecond
	ingressController.configMapReviewDelay = 50 * time.Millisecond
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
			// TODO: Remove: Annotations: map[string]string{},
		},
		Status: extensions_v1beta1.IngressStatus{
			LoadBalancer: api_v1.LoadBalancerStatus{
				Ingress: make([]api_v1.LoadBalancerIngress, 0, 0),
			},
		},
	}

	t.Log("Adding cluster 1")
	clusterWatch.Add(cluster1)

	t.Log("Adding Ingress UID ConfigMap to cluster 1")
	cluster1ConfigMapWatch.Add(cfg1)

	t.Log("Checking that UID annotation on Cluster 1 annotation was correctly updated")
	cluster := GetClusterFromChan(fedClusterUpdateChan)
	assert.NotNil(t, cluster)
	assert.Equal(t, cluster.ObjectMeta.Annotations[uidAnnotationKey], cfg1.Data[uidKey])

	// Test add federated ingress.
	t.Log("Adding Federated Ingress")
	fedIngressWatch.Add(&ing1)
	t.Log("Checking that Ingress was correctly created in cluster 1")
	createdIngress := GetIngressFromChan(t, cluster1IngressCreateChan)
	assert.NotNil(t, createdIngress)
	assert.True(t, reflect.DeepEqual(ing1.Spec, createdIngress.Spec), "Spec of created ingress is not equal")
	assert.True(t, util.ObjectMetaEquivalent(ing1.ObjectMeta, createdIngress.ObjectMeta), "Metadata of created object is not equivalent")

	// Test that IP address gets transferred from cluster ingress to federated ingress.
	t.Log("Checking that IP address gets transferred from cluster ingress to federated ingress")
	createdIngress.Status.LoadBalancer.Ingress = append(createdIngress.Status.LoadBalancer.Ingress, api_v1.LoadBalancerIngress{IP: "1.2.3.4"})
	cluster1IngressWatch.Modify(createdIngress)
	updatedIngress := GetIngressFromChan(t, fedIngressUpdateChan)
	assert.NotNil(t, updatedIngress, "Cluster's ingress load balancer status was not correctly transferred to the federated ingress")
	if updatedIngress != nil {
		assert.True(t, reflect.DeepEqual(createdIngress.Status.LoadBalancer.Ingress, updatedIngress.Status.LoadBalancer.Ingress), fmt.Sprintf("Ingress IP was not transferred from cluster ingress to federated ingress.  %v is not equal to %v", createdIngress.Status.LoadBalancer.Ingress, updatedIngress.Status.LoadBalancer.Ingress))
	}

	// Test update federated ingress.
	if updatedIngress.ObjectMeta.Annotations == nil {
		updatedIngress.ObjectMeta.Annotations = make(map[string]string)
	}
	updatedIngress.ObjectMeta.Annotations["A"] = "B"
	t.Log("Modifying Federated Ingress")
	fedIngressWatch.Modify(updatedIngress)
	t.Log("Checking that Ingress was correctly updated in cluster 1")
	updatedIngress2 := GetIngressFromChan(t, cluster1IngressUpdateChan)
	assert.NotNil(t, updatedIngress2)
	assert.True(t, reflect.DeepEqual(updatedIngress2.Spec, updatedIngress.Spec), "Spec of updated ingress is not equal")
	assert.Equal(t, updatedIngress2.ObjectMeta.Annotations["A"], updatedIngress.ObjectMeta.Annotations["A"], "Updated annotation not transferred from federated to cluster ingress.")
	// Test add cluster
	t.Log("Adding a second cluster")
	ing1.Annotations[staticIPNameKeyWritable] = "foo" // Make sure that the base object has a static IP name first.
	fedIngressWatch.Modify(&ing1)
	clusterWatch.Add(cluster2)
	// First check that the original values are not equal - see above comment
	assert.NotEqual(t, cfg1.Data[uidKey], cfg2.Data[uidKey], fmt.Sprintf("ConfigMap in cluster 2 must initially not equal that in cluster 1 for this test - please fix test"))
	cluster2ConfigMapWatch.Add(cfg2)
	t.Log("Checking that the ingress got created in cluster 2")
	createdIngress2 := GetIngressFromChan(t, cluster2IngressCreateChan)
	assert.NotNil(t, createdIngress2)
	assert.True(t, reflect.DeepEqual(ing1.Spec, createdIngress2.Spec), "Spec of created ingress is not equal")
	assert.True(t, util.ObjectMetaEquivalent(ing1.ObjectMeta, createdIngress2.ObjectMeta), "Metadata of created object is not equivalent")

	t.Log("Checking that the configmap in cluster 2 got updated.")
	updatedConfigMap2 := GetConfigMapFromChan(cluster2ConfigMapUpdateChan)
	assert.NotNil(t, updatedConfigMap2, fmt.Sprintf("ConfigMap in cluster 2 was not updated (or more likely the test is broken and the API type written is wrong)"))
	if updatedConfigMap2 != nil {
		assert.Equal(t, cfg1.Data[uidKey], updatedConfigMap2.Data[uidKey],
			fmt.Sprintf("UID's in configmaps in cluster's 1 and 2 are not equal (%q != %q)", cfg1.Data["uid"], updatedConfigMap2.Data["uid"]))
	}

	close(stop)
}

func GetIngressFromChan(t *testing.T, c chan runtime.Object) *extensions_v1beta1.Ingress {
	obj := GetObjectFromChan(c)
	ingress, ok := obj.(*extensions_v1beta1.Ingress)
	if !ok {
		t.Logf("Object on channel was not of type *extensions_v1beta1.Ingress: %v", obj)
	}
	return ingress
}

func GetConfigMapFromChan(c chan runtime.Object) *api_v1.ConfigMap {
	configMap, _ := GetObjectFromChan(c).(*api_v1.ConfigMap)
	return configMap
}

func GetClusterFromChan(c chan runtime.Object) *federation_api.Cluster {
	cluster, _ := GetObjectFromChan(c).(*federation_api.Cluster)
	return cluster
}

func NewConfigMap(uid string) *api_v1.ConfigMap {
	return &api_v1.ConfigMap{
		ObjectMeta: api_v1.ObjectMeta{
			Name:      uidConfigMapName,
			Namespace: uidConfigMapNamespace,
			SelfLink:  "/api/v1/namespaces/" + uidConfigMapNamespace + "/configmap/" + uidConfigMapName,
			// TODO: Remove: Annotations: map[string]string{},
		},
		Data: map[string]string{
			uidKey: uid,
		},
	}
}
