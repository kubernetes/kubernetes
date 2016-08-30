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
	cluster1 := NewCluster("cluster1", api_v1.ConditionTrue)
	cluster2 := NewCluster("cluster2", api_v1.ConditionTrue)

	fakeClient := &fake_federation_release_1_4.Clientset{}
	RegisterFakeList("clusters", &fakeClient.Fake, &federation_api.ClusterList{Items: []federation_api.Cluster{*cluster1}})
	RegisterFakeList("ingresses", &fakeClient.Fake, &extensions_v1beta1.IngressList{Items: []extensions_v1beta1.Ingress{}})
	ingressWatch := RegisterFakeWatch("ingresses", &fakeClient.Fake)
	clusterWatch := RegisterFakeWatch("clusters", &fakeClient.Fake)

	cluster1Client := &fake_kube_release_1_4.Clientset{}
	cluster1Watch := RegisterFakeWatch("ingresses", &cluster1Client.Fake)
	RegisterFakeList("ingresses", &cluster1Client.Fake, &extensions_v1beta1.IngressList{Items: []extensions_v1beta1.Ingress{}})
	cluster1CreateChan := RegisterFakeCopyOnCreate("ingresses", &cluster1Client.Fake, cluster1Watch)
	cluster1UpdateChan := RegisterFakeCopyOnUpdate("ingresses", &cluster1Client.Fake, cluster1Watch)

	cluster2Client := &fake_kube_release_1_4.Clientset{}
	cluster2Watch := RegisterFakeWatch("ingresses", &cluster2Client.Fake)
	RegisterFakeList("ingresses", &cluster2Client.Fake, &extensions_v1beta1.IngressList{Items: []extensions_v1beta1.Ingress{}})
	cluster2CreateChan := RegisterFakeCopyOnCreate("ingresses", &cluster2Client.Fake, cluster2Watch)

	ingressController := NewIngressController(fakeClient)
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
	ingressController.Run(stop)

	ing1 := extensions_v1beta1.Ingress{
		ObjectMeta: api_v1.ObjectMeta{
			Name:      "test-ingress",
			Namespace: "mynamespace",
			SelfLink:  "/api/v1/namespaces/mynamespaces/ingress/test-ingress",
		},
	}

	// Test add federated ingress.
	ingressWatch.Add(&ing1)
	createdIngress := GetIngressFromChan(cluster1CreateChan)
	assert.NotNil(t, createdIngress)
	assert.True(t, reflect.DeepEqual(&ing1, createdIngress))

	// Test update federated ingress.
	ing1.Annotations = map[string]string{
		"A": "B",
	}
	ingressWatch.Modify(&ing1)
	updatedIngress := GetIngressFromChan(cluster1UpdateChan)
	assert.NotNil(t, updatedIngress)
	assert.True(t, reflect.DeepEqual(&ing1, updatedIngress))

	// Test add cluster
	ing1.Annotations[staticIPAnnotationKey] = "foo" // Make sure that the base object has a static IP name first.
	ingressWatch.Modify(&ing1)
	clusterWatch.Add(cluster2)
	createdIngress2 := GetIngressFromChan(cluster2CreateChan)
	assert.NotNil(t, createdIngress2)
	assert.True(t, reflect.DeepEqual(&ing1, createdIngress2))

	close(stop)
}

func GetIngressFromChan(c chan runtime.Object) *extensions_v1beta1.Ingress {
	ingress := GetObjectFromChan(c).(*extensions_v1beta1.Ingress)
	return ingress
}
