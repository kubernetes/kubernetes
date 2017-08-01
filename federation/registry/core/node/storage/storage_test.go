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

func listNodes(kubeClientset kubeclient.Interface, namespace string, opts metav1.ListOptions) (runtime.Object, error) {
	return kubeClientset.Core().Nodes().List(opts)
}
func getNode(kubeClientset kubeclient.Interface, namespace string, name string, opts metav1.GetOptions) (runtime.Object, error) {
	return kubeClientset.Core().Nodes().Get(name, opts)
}

func newStorage(t *testing.T) (*REST, *StatusREST) {
	fedClient := fedclient.NewSimpleClientset(proxy.NewTestCluster("foo"), proxy.NewTestCluster("bar"))
	clusterClientsets := map[string]kubeclient.Interface{
		"foo": kubeclientfake.NewSimpleClientset(newNode("foo1"), newNode("dup")),
		"bar": kubeclientfake.NewSimpleClientset(newNode("bar1"), newNode("dup")),
	}

	rest, statusRest := NewREST(nil, fedClient)
	rest.RESTClientFunc = proxy.FakeRestClientFuncForClusters(testapi.Default, clusterClientsets, listNodes, getNode)
	statusRest.store.RESTClientFunc = proxy.FakeRestClientFuncForClusters(testapi.Default, clusterClientsets, listNodes, getNode)

	return rest, statusRest
}

func newNode(name string) *api.Node {
	node := &api.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: map[string]string{"name": name},
		},
		Spec:   api.NodeSpec{},
		Status: api.NodeStatus{},
	}
	api.Scheme.Default(node)
	return node
}

func TestList(t *testing.T) {
	rest, _ := newStorage(t)
	selector, _ := labels.Parse(fedv1.FederationClusterNameLabel)
	obj, err := rest.List(proxy.NewTestContex(), &metainternalversion.ListOptions{
		ClusterSelector: selector,
	})
	assert.Nil(t, err)
	nodeList, ok := obj.(*api.NodeList)
	assert.True(t, ok)
	assert.Equal(t, 4, len(nodeList.Items))
	for _, p := range nodeList.Items {
		assert.Contains(t, []string{"foo", "bar"}, p.ClusterName)
		assert.NotEqual(t, "bar2", p.Name)
	}

	selector, _ = labels.Parse(fedv1.FederationClusterNameLabel + "=bar")
	obj, err = rest.List(proxy.NewTestContex(), &metainternalversion.ListOptions{
		ClusterSelector: selector,
	})
	assert.Nil(t, err)
	nodeList, ok = obj.(*api.NodeList)
	assert.True(t, ok)
	assert.Equal(t, 2, len(nodeList.Items))
	for _, p := range nodeList.Items {
		assert.Equal(t, "bar", p.ClusterName)
	}
}

func TestGet(t *testing.T) {
	rest, _ := newStorage(t)
	obj, err := rest.Get(proxy.NewTestContex(), "foo1", &metav1.GetOptions{ClusterName: "foo"})
	assert.Nil(t, err)
	node, ok := obj.(*api.Node)
	assert.True(t, ok)
	assert.Equal(t, "foo1", node.Name)
	assert.Equal(t, "foo", node.ClusterName)

	obj, err = rest.Get(proxy.NewTestContex(), "not-found", &metav1.GetOptions{})
	assert.True(t, errors.IsNotFound(err))

	obj, err = rest.Get(proxy.NewTestContex(), "bar1", &metav1.GetOptions{ClusterName: "foo"})
	assert.True(t, errors.IsNotFound(err))

	obj, err = rest.Get(proxy.NewTestContex(), "bar1", &metav1.GetOptions{ClusterName: "bar"})
	assert.Nil(t, err)
	node, ok = obj.(*api.Node)
	assert.True(t, ok)
	assert.Equal(t, "bar1", node.Name)
	assert.Equal(t, "bar", node.ClusterName)
}
