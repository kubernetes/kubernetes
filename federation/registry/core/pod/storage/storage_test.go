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

package storage

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"k8s.io/apimachinery/pkg/api/errors"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	fedv1 "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fedclient "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/fake"
	"k8s.io/kubernetes/federation/registry/proxy"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	kubeclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	kubeclientfake "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
)

const (
	testNamespace = "test"
)

func listPods(kubeClientset kubeclient.Interface, namespace string, opts metav1.ListOptions) (runtime.Object, error) {
	return kubeClientset.Core().Pods(namespace).List(opts)
}
func getPod(kubeClientset kubeclient.Interface, namespace string, name string, opts metav1.GetOptions) (runtime.Object, error) {
	return kubeClientset.Core().Pods(namespace).Get(name, opts)
}

func newStorage(t *testing.T) (*REST, *StatusREST) {
	fedClient := fedclient.NewSimpleClientset(proxy.NewTestCluster("foo"), proxy.NewTestCluster("bar"))
	clusterClientsets := map[string]kubeclient.Interface{
		"foo": kubeclientfake.NewSimpleClientset(newPod("foo1", testNamespace), newPod("dup", testNamespace)),
		"bar": kubeclientfake.NewSimpleClientset(newPod("bar1", testNamespace), newPod("dup", testNamespace), newPod("bar2", api.NamespaceDefault)),
	}

	rest, statusRest := NewREST(nil, fedClient)
	rest.RESTClientFunc = proxy.FakeRestClientFuncForClusters(testapi.Default, clusterClientsets, listPods, getPod)
	statusRest.store.RESTClientFunc = proxy.FakeRestClientFuncForClusters(testapi.Default, clusterClientsets, listPods, getPod)

	return rest, statusRest
}

func newPod(name string, namespace string) *api.Pod {
	pod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels:    map[string]string{"name": name},
		},
		Spec:   api.PodSpec{},
		Status: api.PodStatus{},
	}
	api.Scheme.Default(pod)
	return pod
}

func TestList(t *testing.T) {
	rest, _ := newStorage(t)
	selector, _ := labels.Parse(fedv1.FederationClusterNameLabel)
	obj, err := rest.List(proxy.NewTestContextWithNamespace(testNamespace), &metainternalversion.ListOptions{
		ClusterSelector: selector,
	})
	assert.Nil(t, err)
	podList, ok := obj.(*api.PodList)
	assert.True(t, ok)
	assert.Equal(t, 4, len(podList.Items))
	for _, p := range podList.Items {
		assert.Contains(t, []string{"foo", "bar"}, p.ClusterName)
		assert.NotEqual(t, "bar2", p.Name)
	}

	selector, _ = labels.Parse(fedv1.FederationClusterNameLabel + "=bar")
	obj, err = rest.List(proxy.NewTestContextWithNamespace(testNamespace), &metainternalversion.ListOptions{
		ClusterSelector: selector,
	})
	assert.Nil(t, err)
	podList, ok = obj.(*api.PodList)
	assert.True(t, ok)
	assert.Equal(t, 2, len(podList.Items))
	for _, p := range podList.Items {
		assert.Equal(t, "bar", p.ClusterName)
	}
}

func TestGet(t *testing.T) {
	rest, _ := newStorage(t)
	obj, err := rest.Get(proxy.NewTestContextWithNamespace(testNamespace), "foo1", &metav1.GetOptions{ClusterName: "foo"})
	assert.Nil(t, err)
	pod, ok := obj.(*api.Pod)
	assert.True(t, ok)
	assert.Equal(t, "foo1", pod.Name)
	assert.Equal(t, "foo", pod.ClusterName)

	obj, err = rest.Get(proxy.NewTestContextWithNamespace(testNamespace), "not-found", &metav1.GetOptions{ClusterName: "foo"})
	assert.True(t, errors.IsNotFound(err))

	obj, err = rest.Get(proxy.NewTestContextWithNamespace(api.NamespaceDefault), "bar1", &metav1.GetOptions{})
	assert.True(t, errors.IsNotFound(err))

	obj, err = rest.Get(proxy.NewTestContextWithNamespace(api.NamespaceDefault), "bar2", &metav1.GetOptions{ClusterName: "bar"})
	assert.Nil(t, err)
	pod, ok = obj.(*api.Pod)
	assert.True(t, ok)
	assert.Equal(t, "bar2", pod.Name)
	assert.Equal(t, "bar", pod.ClusterName)
}
