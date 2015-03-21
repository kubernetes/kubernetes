/*
Copyright 2014 Google Inc. All rights reserved.

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

package master

import (
	"errors"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/rest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// RESTStorageToNodes will take a RESTStorage object and return a client interface
// which will work for any use expecting a client.Nodes() interface. The advantage
// of doing this in server code is that it doesn't require an actual trip through
// the network.
//
// TODO: considering that the only difference between the various client types
// and RESTStorage type is the type of the arguments, maybe use "go generate" to
// write a specialized adaptor for every client type?
//
// TODO: this also means that pod and node API endpoints have to be colocated in the same
// process
func RESTStorageToNodes(storage rest.Storage) client.NodesInterface {
	return &nodeAdaptor{storage}
}

type nodeAdaptor struct {
	storage rest.Storage
}

func (n *nodeAdaptor) Nodes() client.NodeInterface {
	return n
}

// Create creates a new node.
func (n *nodeAdaptor) Create(minion *api.Node) (*api.Node, error) {
	return nil, errors.New("direct creation not implemented")
	// TODO: apiserver should expose newOperation to make this easier.
	// the actual code that should go here would start like this:
	//
	// ctx := api.NewDefaultContext()
	// out, err := n.storage.Create(ctx, minion)
	// if err != nil {
	//	return nil, err
	// }
}

// List lists all the nodes in the cluster.
func (n *nodeAdaptor) List() (*api.NodeList, error) {
	ctx := api.NewContext()
	obj, err := n.storage.(rest.Lister).List(ctx, labels.Everything(), fields.Everything())
	if err != nil {
		return nil, err
	}
	return obj.(*api.NodeList), nil
}

// Get gets an existing node.
func (n *nodeAdaptor) Get(name string) (*api.Node, error) {
	ctx := api.NewContext()
	obj, err := n.storage.(rest.Getter).Get(ctx, name)
	if err != nil {
		return nil, err
	}
	return obj.(*api.Node), nil
}

// Delete deletes an existing node.
// TODO: implement
func (n *nodeAdaptor) Delete(name string) error {
	return errors.New("direct deletion not implemented")
}

// Update updates an existing node.
func (n *nodeAdaptor) Update(minion *api.Node) (*api.Node, error) {
	return nil, errors.New("direct update not implemented")
}

// Watch watches for nodes.
func (n *nodeAdaptor) Watch(label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return nil, errors.New("direct watch not implemented")
}
