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

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/rand"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	testutils "k8s.io/kubernetes/test/utils"
)

const (
	retries = 5
)

// IntegrationTestNodePreparer holds configuration information for the test node preparer.
type IntegrationTestNodePreparer struct {
	client          clientset.Interface
	countToStrategy []testutils.CountToStrategy
	nodeNamePrefix  string
	nodeSpec        *v1.Node
}

// NewIntegrationTestNodePreparer creates an IntegrationTestNodePreparer configured with defaults.
func NewIntegrationTestNodePreparer(client clientset.Interface, countToStrategy []testutils.CountToStrategy, nodeNamePrefix string) testutils.TestNodePreparer {
	return &IntegrationTestNodePreparer{
		client:          client,
		countToStrategy: countToStrategy,
		nodeNamePrefix:  nodeNamePrefix,
	}
}

// NewIntegrationTestNodePreparerWithNodeSpec creates an IntegrationTestNodePreparer configured with nodespec.
func NewIntegrationTestNodePreparerWithNodeSpec(client clientset.Interface, countToStrategy []testutils.CountToStrategy, nodeSpec *v1.Node) testutils.TestNodePreparer {
	return &IntegrationTestNodePreparer{
		client:          client,
		countToStrategy: countToStrategy,
		nodeSpec:        nodeSpec,
	}
}

// PrepareNodes prepares countToStrategy test nodes.
func (p *IntegrationTestNodePreparer) PrepareNodes(ctx context.Context, nextNodeIndex int) error {
	numNodes := 0
	for _, v := range p.countToStrategy {
		numNodes += v.Count
	}

	klog.Infof("Making %d nodes", numNodes)
	baseNode := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: p.nodeNamePrefix,
		},
		Status: v1.NodeStatus{
			Capacity: v1.ResourceList{
				v1.ResourcePods:   *resource.NewQuantity(110, resource.DecimalSI),
				v1.ResourceCPU:    resource.MustParse("4"),
				v1.ResourceMemory: resource.MustParse("32Gi"),
			},
			Phase: v1.NodeRunning,
			Conditions: []v1.NodeCondition{
				{Type: v1.NodeReady, Status: v1.ConditionTrue},
			},
		},
	}

	if p.nodeSpec != nil {
		baseNode = p.nodeSpec
	}

	for i := 0; i < numNodes; i++ {
		var err error
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
			klog.Fatalf("Error creating node: %v", err)
		}
	}

	nodes, err := waitListAllNodes(p.client)
	if err != nil {
		klog.Fatalf("Error listing nodes: %v", err)
	}
	index := nextNodeIndex
	for _, v := range p.countToStrategy {
		for i := 0; i < v.Count; i, index = i+1, index+1 {
			if err := testutils.DoPrepareNode(ctx, p.client, &nodes.Items[index], v.Strategy); err != nil {
				klog.Errorf("Aborting node preparation: %v", err)
				return err
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
