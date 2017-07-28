/*
Copyright 2017 The Kubernetes Authors.

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

package proxy

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"k8s.io/apimachinery/pkg/api/errors"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	fedv1 "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fedclientfake "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/fake"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	kubeclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	kubeclientfake "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
)

const testNamespace = "test"

func TestStoreList(t *testing.T) {
	store := newTestProxyStoreRegistry(t)

	// list from fed and all clusters
	clusterSelector, _ := labels.Parse(fedv1.FederationClusterNameLabel)
	obj, err := store.List(NewTestContextWithNamespace(testNamespace), &metainternalversion.ListOptions{ClusterSelector: clusterSelector})
	assert.Nil(t, err)
	podList, ok := obj.(*api.PodList)
	assert.True(t, ok)
	assert.Equal(t, 5, len(podList.Items))
	dups := []api.Pod{}
	for _, p := range podList.Items {
		assert.NotEqual(t, "bar2", p.Name) // not in testNamespace
		if p.Name == "dup" {               // in federation and both clusters
			assert.Contains(t, []string{"foo", "bar", ""}, p.ClusterName)
			dups = append(dups, p)
		} else { // in one of the clusters
			assert.Contains(t, []string{"foo", "bar"}, p.ClusterName)
		}
	}
	assert.Equal(t, 3, len(dups))
	assert.NotEqual(t, dups[0].ClusterName, dups[1].ClusterName)
	assert.NotEqual(t, dups[1].ClusterName, dups[2].ClusterName)
	assert.NotEqual(t, dups[2].ClusterName, dups[0].ClusterName)

	// list from cluster bar only
	clusterSelector, _ = labels.Parse(fedv1.FederationClusterNameLabel + "=bar")
	obj, err = store.List(NewTestContextWithNamespace(testNamespace), &metainternalversion.ListOptions{ClusterSelector: clusterSelector})
	assert.Nil(t, err)
	podList, ok = obj.(*api.PodList)
	assert.True(t, ok)
	assert.Equal(t, 2, len(podList.Items))
	for _, p := range podList.Items {
		assert.Equal(t, "bar", p.ClusterName)
	}

	// list from cluster bar only with label selector
	clusterSelector, _ = labels.Parse(fedv1.FederationClusterNameLabel + "=bar")
	labelSelector, _ := labels.Parse("name=bar1")
	obj, err = store.List(NewTestContextWithNamespace(testNamespace), &metainternalversion.ListOptions{ClusterSelector: clusterSelector, LabelSelector: labelSelector})
	assert.Nil(t, err)
	podList, ok = obj.(*api.PodList)
	assert.True(t, ok)
	assert.Equal(t, 1, len(podList.Items))
	assert.Equal(t, "bar", podList.Items[0].ClusterName)
	assert.Equal(t, "bar1", podList.Items[0].Name)

	// list from federation only
	obj, err = store.List(NewTestContextWithNamespace(testNamespace), &metainternalversion.ListOptions{})
	assert.Nil(t, err)
	podList, ok = obj.(*api.PodList)
	assert.True(t, ok)
	assert.Equal(t, 1, len(podList.Items))
	assert.Equal(t, "dup", podList.Items[0].Name)
	assert.Equal(t, "", podList.Items[0].ClusterName)
}

func TestStoreGet(t *testing.T) {
	store := newTestProxyStoreRegistry(t)

	obj, err := store.Get(NewTestContextWithNamespace(testNamespace), "dup", &metav1.GetOptions{})
	assert.Nil(t, err)
	pod, ok := obj.(*api.Pod)
	assert.True(t, ok)
	assert.Equal(t, "dup", pod.Name)
	assert.Equal(t, "", pod.ClusterName)

	obj, err = store.Get(NewTestContextWithNamespace(testNamespace), "dup", &metav1.GetOptions{ClusterName: "foo"})
	assert.Nil(t, err)
	pod, ok = obj.(*api.Pod)
	assert.True(t, ok)
	assert.Equal(t, "dup", pod.Name)
	assert.Equal(t, "foo", pod.ClusterName)

	obj, err = store.Get(NewTestContextWithNamespace(testNamespace), "not-found", &metav1.GetOptions{})
	assert.True(t, errors.IsNotFound(err))
}

func listPods(kubeClientset kubeclient.Interface, namespace string, opts metav1.ListOptions) (runtime.Object, error) {
	return kubeClientset.Core().Pods(namespace).List(opts)
}
func getPod(kubeClientset kubeclient.Interface, namespace string, name string, opts metav1.GetOptions) (runtime.Object, error) {
	return kubeClientset.Core().Pods(namespace).Get(name, opts)
}

func newPod(name string, namespace string) *api.Pod {
	pod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels:    map[string]string{"name": name},
		},
	}
	return pod
}

func newTestProxyStoreRegistry(t *testing.T) *Store {

	fedClient := fedclientfake.NewSimpleClientset(NewTestCluster("foo"), NewTestCluster("bar"))
	clusterClientsets := map[string]kubeclient.Interface{
		"foo": kubeclientfake.NewSimpleClientset(newPod("foo1", testNamespace), newPod("dup", testNamespace)),
		"bar": kubeclientfake.NewSimpleClientset(newPod("bar1", testNamespace), newPod("dup", testNamespace), newPod("bar2", api.NamespaceDefault)),
	}

	RESTClientFunc := FakeRestClientFuncForClusters(testapi.Default, clusterClientsets, listPods, getPod)

	fedStore := &FakeStore{
		NewFunc:     func() runtime.Object { return &api.Pod{} },
		NewListFunc: func() runtime.Object { return &api.PodList{} },
		KubeClient:  kubeclientfake.NewSimpleClientset(newPod("dup", testNamespace)),
		ListFunc: func(kubeClient kubeclient.Interface, namespace string, opts metav1.ListOptions) (runtime.Object, error) {
			return kubeClient.Core().Pods(namespace).List(opts)
		},
		GetFunc: func(kubeClient kubeclient.Interface, namespace string, name string, opts metav1.GetOptions) (runtime.Object, error) {
			return kubeClient.Core().Pods(namespace).Get(name, opts)
		},
	}

	return &Store{
		NewFunc:           func() runtime.Object { return &api.Pod{} },
		NewListFunc:       func() runtime.Object { return &api.PodList{} },
		RESTClientFunc:    RESTClientFunc,
		QualifiedResource: api.Resource("pods"),
		NamespaceScoped:   true,
		FedClient:         fedClient,
		FedStore:          fedStore,
	}
}
