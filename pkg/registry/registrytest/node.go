/*
Copyright 2014 The Kubernetes Authors.

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

package registrytest

import (
	"sync"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/watch"
)

// NodeRegistry implements node.Registry interface.
type NodeRegistry struct {
	Err   error
	Node  string
	Nodes api.NodeList

	sync.Mutex
}

// MakeNodeList constructs api.NodeList from list of node names and a NodeResource.
func MakeNodeList(nodes []string, nodeResources api.NodeResources) *api.NodeList {
	list := api.NodeList{
		Items: make([]api.Node, len(nodes)),
	}
	for i := range nodes {
		list.Items[i].Name = nodes[i]
		list.Items[i].Status.Capacity = nodeResources.Capacity
	}
	return &list
}

func NewNodeRegistry(nodes []string, nodeResources api.NodeResources) *NodeRegistry {
	return &NodeRegistry{
		Nodes: *MakeNodeList(nodes, nodeResources),
	}
}

func (r *NodeRegistry) SetError(err error) {
	r.Lock()
	defer r.Unlock()
	r.Err = err
}

func (r *NodeRegistry) ListNodes(ctx api.Context, options *api.ListOptions) (*api.NodeList, error) {
	r.Lock()
	defer r.Unlock()
	return &r.Nodes, r.Err
}

func (r *NodeRegistry) CreateNode(ctx api.Context, node *api.Node) error {
	r.Lock()
	defer r.Unlock()
	r.Node = node.Name
	r.Nodes.Items = append(r.Nodes.Items, *node)
	return r.Err
}

func (r *NodeRegistry) UpdateNode(ctx api.Context, node *api.Node) error {
	r.Lock()
	defer r.Unlock()
	for i, item := range r.Nodes.Items {
		if item.Name == node.Name {
			r.Nodes.Items[i] = *node
			return r.Err
		}
	}
	return r.Err
}

func (r *NodeRegistry) GetNode(ctx api.Context, nodeID string, options *metav1.GetOptions) (*api.Node, error) {
	r.Lock()
	defer r.Unlock()
	if r.Err != nil {
		return nil, r.Err
	}
	for _, node := range r.Nodes.Items {
		if node.Name == nodeID {
			return &node, nil
		}
	}
	return nil, errors.NewNotFound(api.Resource("nodes"), nodeID)
}

func (r *NodeRegistry) DeleteNode(ctx api.Context, nodeID string) error {
	r.Lock()
	defer r.Unlock()
	var newList []api.Node
	for _, node := range r.Nodes.Items {
		if node.Name != nodeID {
			newList = append(newList, api.Node{ObjectMeta: api.ObjectMeta{Name: node.Name}})
		}
	}
	r.Nodes.Items = newList
	return r.Err
}

func (r *NodeRegistry) WatchNodes(ctx api.Context, options *api.ListOptions) (watch.Interface, error) {
	return nil, r.Err
}
