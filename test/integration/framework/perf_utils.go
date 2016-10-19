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
	"time"

	"k8s.io/kubernetes/pkg/api"
	apierrs "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/resource"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	e2eframework "k8s.io/kubernetes/test/e2e/framework"
	testutils "k8s.io/kubernetes/test/utils"

	"github.com/golang/glog"
)

const (
	retries = 5
)

type IntegrationTestNodePreparer struct {
	client          clientset.Interface
	countToStrategy map[int]testutils.PrepareNodeStrategy
	nodeNamePrefix  string
}

func NewIntegrationTestNodePreparer(client clientset.Interface, countToStrategy map[int]testutils.PrepareNodeStrategy, nodeNamePrefix string) testutils.TestNodePreparer {
	return &IntegrationTestNodePreparer{
		client:          client,
		countToStrategy: countToStrategy,
		nodeNamePrefix:  nodeNamePrefix,
	}
}

func (p *IntegrationTestNodePreparer) PrepareNodes() error {
	numNodes := 0
	for k := range p.countToStrategy {
		numNodes += k
	}

	glog.Infof("Making %d nodes", numNodes)
	baseNode := &api.Node{
		ObjectMeta: api.ObjectMeta{
			GenerateName: p.nodeNamePrefix,
		},
		Spec: api.NodeSpec{
			ExternalID: "foo",
		},
		Status: api.NodeStatus{
			Capacity: api.ResourceList{
				api.ResourcePods:   *resource.NewQuantity(110, resource.DecimalSI),
				api.ResourceCPU:    resource.MustParse("4"),
				api.ResourceMemory: resource.MustParse("32Gi"),
			},
			Phase: api.NodeRunning,
			Conditions: []api.NodeCondition{
				{Type: api.NodeReady, Status: api.ConditionTrue},
			},
		},
	}
	for i := 0; i < numNodes; i++ {
		if _, err := p.client.Core().Nodes().Create(baseNode); err != nil {
			panic("error creating node: " + err.Error())
		}
	}

	nodes := e2eframework.GetReadySchedulableNodesOrDie(p.client)
	index := 0
	sum := 0
	for k, strategy := range p.countToStrategy {
		sum += k
		for ; index < sum; index++ {
			var err error
			patch := strategy.PreparePatch(&nodes.Items[index])
			if len(patch) == 0 {
				continue
			}
			for attempt := 0; attempt < retries; attempt++ {
				_, err = p.client.Core().Nodes().Patch(nodes.Items[index].Name, api.MergePatchType, []byte(patch))
				if err != nil {
					if !apierrs.IsConflict(err) {
						break
					}
				} else {
					break
				}
				time.Sleep(100 * time.Millisecond)
			}
			if err != nil {
				glog.Errorf("Can't apply patch %v to Node %v: %v", string(patch), nodes.Items[index].Name, err)
				return err
			}
		}
	}
	return nil
}

func (p *IntegrationTestNodePreparer) CleanupNodes() error {
	nodes := e2eframework.GetReadySchedulableNodesOrDie(p.client)
	for i := range nodes.Items {
		err := p.client.Core().Nodes().Delete(nodes.Items[i].Name, &api.DeleteOptions{})
		if err != nil {
			glog.Errorf("Error while deleting Node: %v", err)
		}
	}
	return nil
}
