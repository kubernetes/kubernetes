/*
Copyright 2015 The Kubernetes Authors.

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
	"fmt"
	"net/http"
	"net/url"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/api"
	k8s_api_v1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/kubelet/client"
	"k8s.io/kubernetes/pkg/registry/cachesize"
	"k8s.io/kubernetes/pkg/registry/core/node"
	noderest "k8s.io/kubernetes/pkg/registry/core/node/rest"
)

// NodeStorage includes storage for nodes and all sub resources
type NodeStorage struct {
	Node   *REST
	Status *StatusREST
	Proxy  *noderest.ProxyREST

	KubeletConnectionInfo client.ConnectionInfoGetter
}

type REST struct {
	*genericregistry.Store
	connection     client.ConnectionInfoGetter
	proxyTransport http.RoundTripper
}

// StatusREST implements the REST endpoint for changing the status of a pod.
type StatusREST struct {
	store *genericregistry.Store
}

func (r *StatusREST) New() runtime.Object {
	return &api.Node{}
}

// Get retrieves the object from the storage. It is required to support Patch.
func (r *StatusREST) Get(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	return r.store.Get(ctx, name, options)
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx genericapirequest.Context, name string, objInfo rest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	return r.store.Update(ctx, name, objInfo)
}

// NewStorage returns a NodeStorage object that will work against nodes.
func NewStorage(optsGetter generic.RESTOptionsGetter, kubeletClientConfig client.KubeletClientConfig, proxyTransport http.RoundTripper) (*NodeStorage, error) {
	store := &genericregistry.Store{
		Copier:                   api.Scheme,
		NewFunc:                  func() runtime.Object { return &api.Node{} },
		NewListFunc:              func() runtime.Object { return &api.NodeList{} },
		PredicateFunc:            node.MatchNode,
		DefaultQualifiedResource: api.Resource("nodes"),
		WatchCacheSize:           cachesize.GetWatchCacheSizeByResource("nodes"),

		CreateStrategy: node.Strategy,
		UpdateStrategy: node.Strategy,
		DeleteStrategy: node.Strategy,
		ExportStrategy: node.Strategy,
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter, AttrFunc: node.GetAttrs, TriggerFunc: node.NodeNameTriggerFunc}
	if err := store.CompleteWithOptions(options); err != nil {
		return nil, err
	}

	statusStore := *store
	statusStore.UpdateStrategy = node.StatusStrategy

	// Set up REST handlers
	nodeREST := &REST{Store: store, proxyTransport: proxyTransport}
	statusREST := &StatusREST{store: &statusStore}
	proxyREST := &noderest.ProxyREST{Store: store, ProxyTransport: proxyTransport}

	// Build a NodeGetter that looks up nodes using the REST handler
	nodeGetter := client.NodeGetterFunc(func(nodeName string, options metav1.GetOptions) (*v1.Node, error) {
		obj, err := nodeREST.Get(genericapirequest.NewContext(), nodeName, &options)
		if err != nil {
			return nil, err
		}
		node, ok := obj.(*api.Node)
		if !ok {
			return nil, fmt.Errorf("unexpected type %T", obj)
		}
		// TODO: Remove the conversion. Consider only return the NodeAddresses
		externalNode := &v1.Node{}
		err = k8s_api_v1.Convert_api_Node_To_v1_Node(node, externalNode, nil)
		if err != nil {
			return nil, fmt.Errorf("failed to convert to v1.Node: %v", err)
		}
		return externalNode, nil
	})
	connectionInfoGetter, err := client.NewNodeConnectionInfoGetter(nodeGetter, kubeletClientConfig)
	if err != nil {
		return nil, err
	}
	nodeREST.connection = connectionInfoGetter
	proxyREST.Connection = connectionInfoGetter

	return &NodeStorage{
		Node:   nodeREST,
		Status: statusREST,
		Proxy:  proxyREST,
		KubeletConnectionInfo: connectionInfoGetter,
	}, nil
}

// Implement Redirector.
var _ = rest.Redirector(&REST{})

// ResourceLocation returns a URL to which one can send traffic for the specified node.
func (r *REST) ResourceLocation(ctx genericapirequest.Context, id string) (*url.URL, http.RoundTripper, error) {
	return node.ResourceLocation(r, r.connection, r.proxyTransport, ctx, id)
}

// ShortNames implements the ShortNamesProvider interface. Returns a list of short names for a resource.
func (r *REST) ShortNames() []string {
	return []string{"no"}
}
