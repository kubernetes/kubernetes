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
	// federation_release_1_4 "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_4"
	fake_federation_release_1_4 "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_4/fake"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	api_v1 "k8s.io/kubernetes/pkg/api/v1"
	extensions_v1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	kube_release_1_4 "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_4"
	fake_kube_release_1_4 "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_4/fake"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/stretchr/testify/assert"
)

func TestIngressController(t *testing.T) {
	cluster1 := mkCluster("cluster1", api_v1.ConditionTrue)
	cluster2 := mkCluster("cluster2", api_v1.ConditionTrue)

	fakeClient := &fake_federation_release_1_4.Clientset{}
	RegisterList("clusters", &fakeClient.Fake, &federation_api.ClusterList{Items: []federation_api.Cluster{*cluster1}})
	RegisterList("ingresses", &fakeClient.Fake, &extensions_v1beta1.IngressList{Items: []extensions_v1beta1.Ingress{}})
	ingressWatch := RegisterWatch("ingresses", &fakeClient.Fake)
	clusterWatch := RegisterWatch("clusters", &fakeClient.Fake)

	cluster1Client := &fake_kube_release_1_4.Clientset{}
	cluster1Watch := RegisterWatch("ingresses", &cluster1Client.Fake)
	RegisterList("ingresses", &cluster1Client.Fake, &extensions_v1beta1.IngressList{Items: []extensions_v1beta1.Ingress{}})
	cluster1CreateChan := RegisterCopyOnCreate("ingresses", &cluster1Client.Fake, cluster1Watch)
	cluster1UpdateChan := RegisterCopyOnUpdate("ingresses", &cluster1Client.Fake, cluster1Watch)

	cluster2Client := &fake_kube_release_1_4.Clientset{}
	cluster2Watch := RegisterWatch("ingresses", &cluster2Client.Fake)
	RegisterList("ingresses", &cluster2Client.Fake, &extensions_v1beta1.IngressList{Items: []extensions_v1beta1.Ingress{}})
	cluster2CreateChan := RegisterCopyOnCreate("ingresses", &cluster2Client.Fake, cluster2Watch)

	ingressController := NewIngressController(fakeClient)
	informer := toFederatedInformerForTestOnly(ingressController.ingressFederatedInformer)
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

func RegisterWatch(resource string, client *core.Fake) *watch.FakeWatcher {
	watcher := watch.NewFake()
	client.AddWatchReactor(resource, func(action core.Action) (bool, watch.Interface, error) { return true, watcher, nil })
	return watcher
}

func RegisterList(resource string, client *core.Fake, obj runtime.Object) {
	client.AddReactor("list", resource, func(action core.Action) (bool, runtime.Object, error) {
		return true, obj, nil
	})
}

func RegisterCopyOnCreate(resource string, client *core.Fake, watcher *watch.FakeWatcher) chan runtime.Object {
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

func RegisterCopyOnUpdate(resource string, client *core.Fake, watcher *watch.FakeWatcher) chan runtime.Object {
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

func GetIngressFromChan(c chan runtime.Object) *extensions_v1beta1.Ingress {
	select {
	case obj := <-c:
		ingress := obj.(*extensions_v1beta1.Ingress)
		return ingress
	case <-time.After(time.Minute):
		return nil
	}
}
