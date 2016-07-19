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

package etcd

import (
	"fmt"
	"net/http"
	"net/url"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/kubelet/client"
	"k8s.io/kubernetes/pkg/registry/cachesize"
	"k8s.io/kubernetes/pkg/registry/core/node"
	noderest "k8s.io/kubernetes/pkg/registry/core/node/rest"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/runtime"
)

// NodeStorage includes storage for nodes and all sub resources
type NodeStorage struct {
	Node   *REST
	Status *StatusREST
	Proxy  *noderest.ProxyREST
}

type REST struct {
	*registry.Store
	connection     client.KubeletClient
	proxyTransport http.RoundTripper
}

// StatusREST implements the REST endpoint for changing the status of a pod.
type StatusREST struct {
	store *registry.Store
}

func (r *StatusREST) New() runtime.Object {
	return &api.Node{}
}

// Get retrieves the object from the storage. It is required to support Patch.
func (r *StatusREST) Get(ctx api.Context, name string) (runtime.Object, error) {
	return r.store.Get(ctx, name)
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx api.Context, name string, objInfo rest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	return r.store.Update(ctx, name, objInfo)
}

// NewStorage returns a NodeStorage object that will work against nodes.
func NewStorage(opts generic.RESTOptions, connection client.KubeletClient, proxyTransport http.RoundTripper) NodeStorage {
	prefix := "/" + opts.ResourcePrefix

	newListFunc := func() runtime.Object { return &api.NodeList{} }
	storageInterface, dFunc := opts.Decorator(
		opts.StorageConfig,
		cachesize.GetWatchCacheSizeByResource(cachesize.Nodes),
		&api.Node{},
		prefix,
		node.Strategy,
		newListFunc,
		node.NodeNameTriggerFunc)

	store := &registry.Store{
		NewFunc:     func() runtime.Object { return &api.Node{} },
		NewListFunc: newListFunc,
		KeyRootFunc: func(ctx api.Context) string {
			return prefix
		},
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return registry.NoNamespaceKeyFunc(ctx, prefix, name)
		},
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*api.Node).Name, nil
		},
		PredicateFunc:           node.MatchNode,
		QualifiedResource:       api.Resource("nodes"),
		EnableGarbageCollection: opts.EnableGarbageCollection,
		DeleteCollectionWorkers: opts.DeleteCollectionWorkers,

		CreateStrategy: node.Strategy,
		UpdateStrategy: node.Strategy,
		DeleteStrategy: node.Strategy,
		ExportStrategy: node.Strategy,

		Storage:     storageInterface,
		DestroyFunc: dFunc,
	}

	statusStore := *store
	statusStore.UpdateStrategy = node.StatusStrategy

	nodeREST := &REST{store, connection, proxyTransport}

	return NodeStorage{
		Node:   nodeREST,
		Status: &StatusREST{store: &statusStore},
		Proxy:  &noderest.ProxyREST{Store: store, Connection: client.ConnectionInfoGetter(nodeREST), ProxyTransport: proxyTransport},
	}
}

// Implement Redirector.
var _ = rest.Redirector(&REST{})

// ResourceLocation returns a URL to which one can send traffic for the specified node.
func (r *REST) ResourceLocation(ctx api.Context, id string) (*url.URL, http.RoundTripper, error) {
	return node.ResourceLocation(r, r, r.proxyTransport, ctx, id)
}

var _ = client.ConnectionInfoGetter(&REST{})

func (r *REST) getKubeletHostPortPair(nodeName string) (string, int, error) {
	obj, err := r.Get(api.NewDefaultContext(), nodeName)
	if err != nil {
		return "", 0, err
	}
	node, ok := obj.(*api.Node)
	if !ok {
		return "", 0, fmt.Errorf("Unexpected object type: %#v", node)
	}
	port := int(node.Status.DaemonEndpoints.KubeletEndpoint.Port)
	return getNodeAddress(node), port, nil
}

func getNodeAddress(node *api.Node) string {
	nodeAddresses := node.Status.Addresses
	for _, address := range nodeAddresses {
		if address.Type == api.NodeHostName {
			return address.Address
		}
	}
	nodeName := node.Name
	glog.Warningf("Failed to retrieve Hostname for Node %s, falling back to NodeName", nodeName)
	return nodeName
}

func (c *REST) GetConnectionInfo(ctx api.Context, nodeName string) (*client.ConnectionInfo, error) {
	hostname, port, err := c.getKubeletHostPortPair(nodeName)
	if err != nil {
		return nil, err
	}
	connectionInfo, err := c.connection.GetRawConnectionInfo(ctx, hostname)
	if err != nil {
		return nil, err
	}
	connectionInfo.Port = uint(port)

	return connectionInfo, nil
}
