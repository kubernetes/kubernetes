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
	"context"
	"fmt"
	"net"
	"net/http"
	"net/url"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage"
	api "k8s.io/kubernetes/pkg/apis/core"
	k8s_api_v1 "k8s.io/kubernetes/pkg/apis/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/client"
	"k8s.io/kubernetes/pkg/printers"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
	printerstorage "k8s.io/kubernetes/pkg/printers/storage"
	proxyutil "k8s.io/kubernetes/pkg/proxy/util"
	"k8s.io/kubernetes/pkg/registry/core/node"
	restproxy "k8s.io/kubernetes/pkg/registry/core/rest/proxy"
)

// NodeStorage includes storage for nodes and all sub resources
type NodeStorage struct {
	Node   *REST
	Status *StatusREST
	Proxy  *restproxy.REST

	KubeletConnectionInfo client.ConnectionInfoGetter
}

type REST struct {
	*genericregistry.Store
	connection     client.ConnectionInfoGetter
	proxyTransport http.RoundTripper
}

// StatusREST implements the REST endpoint for changing the status of a node.
type StatusREST struct {
	store *genericregistry.Store
}

func (r *StatusREST) New() runtime.Object {
	return &api.Node{}
}

// Get retrieves the object from the storage. It is required to support Patch.
func (r *StatusREST) Get(ctx context.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	return r.store.Get(ctx, name, options)
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx context.Context, name string, objInfo rest.UpdatedObjectInfo, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc, forceAllowCreate bool, options *metav1.UpdateOptions) (runtime.Object, bool, error) {
	// We are explicitly setting forceAllowCreate to false in the call to the underlying storage because
	// subresources should never allow create on update.
	return r.store.Update(ctx, name, objInfo, createValidation, updateValidation, false, options)
}

// NewStorage returns a NodeStorage object that will work against nodes.
func NewStorage(optsGetter generic.RESTOptionsGetter, kubeletClientConfig client.KubeletClientConfig, proxyTransport http.RoundTripper) (*NodeStorage, error) {
	store := &genericregistry.Store{
		NewFunc:                  func() runtime.Object { return &api.Node{} },
		NewListFunc:              func() runtime.Object { return &api.NodeList{} },
		PredicateFunc:            node.MatchNode,
		DefaultQualifiedResource: api.Resource("nodes"),

		CreateStrategy: node.Strategy,
		UpdateStrategy: node.Strategy,
		DeleteStrategy: node.Strategy,
		ExportStrategy: node.Strategy,

		TableConvertor: printerstorage.TableConvertor{TableGenerator: printers.NewTableGenerator().With(printersinternal.AddHandlers)},
	}
	options := &generic.StoreOptions{
		RESTOptions: optsGetter,
		AttrFunc:    node.GetAttrs,
		TriggerFunc: map[string]storage.IndexerFunc{"metadata.name": node.NameTriggerFunc},
	}
	if err := store.CompleteWithOptions(options); err != nil {
		return nil, err
	}

	statusStore := *store
	statusStore.UpdateStrategy = node.StatusStrategy

	// Set up REST handlers
	nodeREST := &REST{Store: store, proxyTransport: proxyTransport}
	statusREST := &StatusREST{store: &statusStore}

	// Build a NodeGetter that looks up nodes using the REST handler
	nodeGetter := client.NodeGetterFunc(func(ctx context.Context, nodeName string, options metav1.GetOptions) (*v1.Node, error) {
		obj, err := nodeREST.Get(ctx, nodeName, &options)
		if err != nil {
			return nil, err
		}
		node, ok := obj.(*api.Node)
		if !ok {
			return nil, fmt.Errorf("unexpected type %T", obj)
		}
		// TODO: Remove the conversion. Consider only return the NodeAddresses
		externalNode := &v1.Node{}
		err = k8s_api_v1.Convert_core_Node_To_v1_Node(node, externalNode, nil)
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

	proxyREST := &restproxy.REST{Redirector: nodeREST, ProxyTransport: proxyTransport}

	return &NodeStorage{
		Node:                  nodeREST,
		Status:                statusREST,
		Proxy:                 proxyREST,
		KubeletConnectionInfo: connectionInfoGetter,
	}, nil
}

// Implement Redirector.
var _ = rest.Redirector(&REST{})

// ResourceLocation returns a URL to which one can send traffic for the specified node.
func (r *REST) ResourceLocation(ctx context.Context, id string) (*url.URL, http.RoundTripper, error) {
	schemeReq, name, portReq, valid := utilnet.SplitSchemeNamePort(id)
	if !valid {
		return nil, nil, errors.NewBadRequest(fmt.Sprintf("invalid node request %q", id))
	}

	info, err := r.connection.GetConnectionInfo(ctx, types.NodeName(name))
	if err != nil {
		return nil, nil, err
	}

	// We check if we want to get a default Kubelet's transport. It happens if either:
	// - no port is specified in request (Kubelet's port is default)
	// - the requested port matches the kubelet port for this node
	if portReq == "" || portReq == info.Port {
		return &url.URL{
				Scheme: info.Scheme,
				Host:   net.JoinHostPort(info.Hostname, info.Port),
			},
			info.Transport,
			nil
	}

	if err := proxyutil.IsProxyableHostname(ctx, &net.Resolver{}, info.Hostname); err != nil {
		return nil, nil, errors.NewBadRequest(err.Error())
	}

	// Otherwise, return the requested scheme and port, and the proxy transport
	return &url.URL{Scheme: schemeReq, Host: net.JoinHostPort(info.Hostname, portReq)}, r.proxyTransport, nil
}

// ShortNames implements the ShortNamesProvider interface. Returns a list of short names for a resource.
func (r *REST) ShortNames() []string {
	return []string{"no"}
}
