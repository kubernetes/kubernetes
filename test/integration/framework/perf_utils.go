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

package framework

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/rand"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	testutils "k8s.io/kubernetes/test/utils"
)

const (
	retries = 5
)

// NodeTemplate is responsible for creating a v1.Node instance that is ready
// to be sent to the API server.
type NodeTemplate interface {
	// GetNodeTemplate returns a node template for one out of many different nodes.
	// Nodes with numbers in the range [index, index+count-1] will be created
	// based on what GetNodeTemplate returns. It gets called multiple times
	// with a fixed index and increasing count parameters. This number can,
	// but doesn't have to be, used to modify parts of the node spec like
	// for example a named reference to some other object.
	GetNodeTemplate(index, count int) (*v1.Node, error)
}

// StaticNodeTemplate returns an implementation of NodeTemplate for a fixed node that is the same regardless of the index.
func StaticNodeTemplate(node *v1.Node) NodeTemplate {
	return (*staticNodeTemplate)(node)
}

type staticNodeTemplate v1.Node

// GetNodeTemplate implements [NodeTemplate.GetNodeTemplate] by returning the same node
// for each call.
func (s *staticNodeTemplate) GetNodeTemplate(index, count int) (*v1.Node, error) {
	return (*v1.Node)(s), nil
}

// IntegrationTestNodePreparer holds configuration information for the test node preparer.
type IntegrationTestNodePreparer struct {
	client          clientset.Interface
	countToStrategy []testutils.CountToStrategy
	nodeTemplate    NodeTemplate
}

// NewIntegrationTestNodePreparer creates an IntegrationTestNodePreparer with a given nodeTemplate.
func NewIntegrationTestNodePreparer(client clientset.Interface, countToStrategy []testutils.CountToStrategy, nodeTemplate NodeTemplate) testutils.TestNodePreparer {
	return &IntegrationTestNodePreparer{
		client:          client,
		countToStrategy: countToStrategy,
		nodeTemplate:    nodeTemplate,
	}
}

// PrepareNodes prepares countToStrategy test nodes.
func (p *IntegrationTestNodePreparer) PrepareNodes(ctx context.Context, nextNodeIndex int) error {
	numNodes := 0
	for _, v := range p.countToStrategy {
		numNodes += v.Count
	}

	klog.Infof("Making %d nodes", numNodes)

	for i := 0; i < numNodes; i++ {
		baseNode, err := p.nodeTemplate.GetNodeTemplate(i, numNodes)
		if err != nil {
			return fmt.Errorf("failed to get node template: %w", err)
		}
		for retry := 0; retry < retries; retry++ {
			// Create nodes with the usual kubernetes.io/hostname label.
			// For that we need to know the name in advance, if we want to
			// do it in one request.
			node := baseNode.DeepCopy()
			name := node.Name
			if name == "" {
				name = node.GenerateName + rand.String(5)
				node.Name = name
			}
			if node.Labels == nil {
				node.Labels = make(map[string]string)
			}
			node.Labels["kubernetes.io/hostname"] = name
			_, err = p.client.CoreV1().Nodes().Create(ctx, node, metav1.CreateOptions{})
			if err == nil {
				break
			}
		}
		if err != nil {
			return fmt.Errorf("creating node: %w", err)
		}
	}

	nodes, err := waitListAllNodes(p.client)
	if err != nil {
		return fmt.Errorf("listing nodes: %w", err)
	}
	index := nextNodeIndex
	for _, v := range p.countToStrategy {
		for i := 0; i < v.Count; i, index = i+1, index+1 {
			if err := testutils.DoPrepareNode(ctx, p.client, &nodes.Items[index], v.Strategy); err != nil {
				return fmt.Errorf("aborting node preparation: %w", err)
			}
		}
	}
	return nil
}

// CleanupNodes deletes existing test nodes.
func (p *IntegrationTestNodePreparer) CleanupNodes(ctx context.Context) error {
	// TODO(#93794): make CleanupNodes only clean up the nodes created by this
	// IntegrationTestNodePreparer to make this more intuitive.
	nodes, err := waitListAllNodes(p.client)
	if err != nil {
		klog.Fatalf("Error listing nodes: %v", err)
	}
	var errRet error
	for i := range nodes.Items {
		if err := p.client.CoreV1().Nodes().Delete(ctx, nodes.Items[i].Name, metav1.DeleteOptions{}); err != nil {
			klog.Errorf("Error while deleting Node: %v", err)
			errRet = err
		}
	}
	return errRet
}
