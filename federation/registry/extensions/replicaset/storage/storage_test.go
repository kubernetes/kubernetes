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
	fedclientfake "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/fake"
	"k8s.io/kubernetes/federation/registry/proxy"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/extensions"
	kubeclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	kubeclientfake "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
)

const testNamespace = "test"

func TestStoreList(t *testing.T) {
	store := newStorage(t)

	// list from fed and all clusters
	clusterSelector, _ := labels.Parse(fedv1.FederationClusterNameLabel)
	obj, err := store.List(proxy.NewTestContextWithNamespace(testNamespace), &metainternalversion.ListOptions{ClusterSelector: clusterSelector})
	assert.Nil(t, err)
	replicaSetList, ok := obj.(*extensions.ReplicaSetList)
	assert.True(t, ok)
	assert.Equal(t, 5, len(replicaSetList.Items))
	dups := []extensions.ReplicaSet{}
	for _, p := range replicaSetList.Items {
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
	obj, err = store.List(proxy.NewTestContextWithNamespace(testNamespace), &metainternalversion.ListOptions{ClusterSelector: clusterSelector})
	assert.Nil(t, err)
	replicaSetList, ok = obj.(*extensions.ReplicaSetList)
	assert.True(t, ok)
	assert.Equal(t, 2, len(replicaSetList.Items))
	for _, p := range replicaSetList.Items {
		assert.Equal(t, "bar", p.ClusterName)
	}

	// list from cluster bar only with label selector
	clusterSelector, _ = labels.Parse(fedv1.FederationClusterNameLabel + "=bar")
	labelSelector, _ := labels.Parse("name=bar1")
	obj, err = store.List(proxy.NewTestContextWithNamespace(testNamespace), &metainternalversion.ListOptions{ClusterSelector: clusterSelector, LabelSelector: labelSelector})
	assert.Nil(t, err)
	replicaSetList, ok = obj.(*extensions.ReplicaSetList)
	assert.True(t, ok)
	assert.Equal(t, 1, len(replicaSetList.Items))
	assert.Equal(t, "bar", replicaSetList.Items[0].ClusterName)
	assert.Equal(t, "bar1", replicaSetList.Items[0].Name)

	// list from federation only
	obj, err = store.List(proxy.NewTestContextWithNamespace(testNamespace), &metainternalversion.ListOptions{})
	assert.Nil(t, err)
	replicaSetList, ok = obj.(*extensions.ReplicaSetList)
	assert.True(t, ok)
	assert.Equal(t, 1, len(replicaSetList.Items))
	assert.Equal(t, "dup", replicaSetList.Items[0].Name)
	assert.Equal(t, "", replicaSetList.Items[0].ClusterName)
}

func TestStoreGet(t *testing.T) {
	store := newStorage(t)

	obj, err := store.Get(proxy.NewTestContextWithNamespace(testNamespace), "dup", &metav1.GetOptions{})
	assert.Nil(t, err)
	replicaSet, ok := obj.(*extensions.ReplicaSet)
	assert.True(t, ok)
	assert.Equal(t, "dup", replicaSet.Name)
	assert.Equal(t, "", replicaSet.ClusterName)

	obj, err = store.Get(proxy.NewTestContextWithNamespace(testNamespace), "dup", &metav1.GetOptions{ClusterName: "foo"})
	assert.Nil(t, err)
	replicaSet, ok = obj.(*extensions.ReplicaSet)
	assert.True(t, ok)
	assert.Equal(t, "dup", replicaSet.Name)
	assert.Equal(t, "foo", replicaSet.ClusterName)

	obj, err = store.Get(proxy.NewTestContextWithNamespace(testNamespace), "not-found", &metav1.GetOptions{})
	assert.True(t, errors.IsNotFound(err))
}

func listReplicaSets(kubeClientset kubeclient.Interface, namespace string, opts metav1.ListOptions) (runtime.Object, error) {
	return kubeClientset.Extensions().ReplicaSets(namespace).List(opts)
}
func getReplicaSet(kubeClientset kubeclient.Interface, namespace string, name string, opts metav1.GetOptions) (runtime.Object, error) {
	return kubeClientset.Extensions().ReplicaSets(namespace).Get(name, opts)
}

func newReplicaSet(name string, namespace string) *extensions.ReplicaSet {
	replicaSet := &extensions.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels:    map[string]string{"name": name},
		},
	}
	return replicaSet
}

func newStorage(t *testing.T) *REST {

	fedClient := fedclientfake.NewSimpleClientset(proxy.NewTestCluster("foo"), proxy.NewTestCluster("bar"))
	clusterClientsets := map[string]kubeclient.Interface{
		"foo": kubeclientfake.NewSimpleClientset(newReplicaSet("foo1", testNamespace), newReplicaSet("dup", testNamespace)),
		"bar": kubeclientfake.NewSimpleClientset(newReplicaSet("bar1", testNamespace), newReplicaSet("dup", testNamespace), newReplicaSet("bar2", api.NamespaceDefault)),
	}

	fedStore := &proxy.FakeStore{
		NewFunc:     func() runtime.Object { return &extensions.ReplicaSet{} },
		NewListFunc: func() runtime.Object { return &extensions.ReplicaSetList{} },
		KubeClient:  kubeclientfake.NewSimpleClientset(newReplicaSet("dup", testNamespace)),
		ListFunc: func(kubeClient kubeclient.Interface, namespace string, opts metav1.ListOptions) (runtime.Object, error) {
			return kubeClient.Extensions().ReplicaSets(namespace).List(opts)
		},
		GetFunc: func(kubeClient kubeclient.Interface, namespace string, name string, opts metav1.GetOptions) (runtime.Object, error) {
			return kubeClient.Extensions().ReplicaSets(namespace).Get(name, opts)
		},
	}

	rest := NewREST(nil, fedClient, fedStore)
	rest.RESTClientFunc = proxy.FakeRestClientFuncForClusters(testapi.Extensions, clusterClientsets, listReplicaSets, getReplicaSet)
	return rest
}
