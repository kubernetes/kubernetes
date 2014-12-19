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

package controller

import (
	"fmt"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

type NodeController struct {
	cloud           cloudprovider.Interface
	matchRE         string
	staticResources *api.NodeResources
	nodes           []string
	kubeClient      client.Interface
}

// NewNodeController returns a new node controller to sync instances from cloudprovider.
func NewNodeController(
	cloud cloudprovider.Interface,
	matchRE string,
	nodes []string,
	staticResources *api.NodeResources,
	kubeClient client.Interface) *NodeController {
	return &NodeController{
		cloud:           cloud,
		matchRE:         matchRE,
		nodes:           nodes,
		staticResources: staticResources,
		kubeClient:      kubeClient,
	}
}

// Run starts syncing instances from cloudprovider periodically, or create initial node list.
func (s *NodeController) Run(period time.Duration) {
	if s.cloud != nil && len(s.matchRE) > 0 {
		go util.Forever(func() {
			if err := s.SyncCloud(); err != nil {
				glog.Errorf("Error syncing cloud: %v", err)
			}
		}, period)
	} else {
		go s.SyncStatic(period)
	}
}

// SyncStatic registers list of machines from command line flag. It returns after successful
// registration of all machines.
func (s *NodeController) SyncStatic(period time.Duration) error {
	registered := util.NewStringSet()
	for {
		for _, nodeID := range s.nodes {
			if registered.Has(nodeID) {
				continue
			}
			_, err := s.kubeClient.Nodes().Create(&api.Node{
				ObjectMeta: api.ObjectMeta{Name: nodeID},
				Spec: api.NodeSpec{
					Capacity: s.staticResources.Capacity,
				},
			})
			if err == nil {
				registered.Insert(nodeID)
			}
		}
		if registered.Len() == len(s.nodes) {
			return nil
		}
		time.Sleep(period)
	}
}

// SyncCloud syncs list of instances from cloudprovider to master etcd registry.
func (s *NodeController) SyncCloud() error {
	matches, err := s.cloudNodes()
	if err != nil {
		return err
	}
	nodes, err := s.kubeClient.Nodes().List()
	if err != nil {
		return err
	}
	nodeMap := make(map[string]*api.Node)
	for _, node := range nodes.Items {
		nodeMap[node.Name] = &node
	}

	// Create or delete nodes from registry.
	for _, node := range matches.Items {
		if _, ok := nodeMap[node.Name]; !ok {
			glog.Infof("Create node in registry: %s", node.Name)
			_, err = s.kubeClient.Nodes().Create(&node)
			if err != nil {
				glog.Errorf("Create node error: %s", node.Name)
			}
		}
		delete(nodeMap, node.Name)
	}

	for nodeID := range nodeMap {
		glog.Infof("Delete node from registry: %s", nodeID)
		err = s.kubeClient.Nodes().Delete(nodeID)
		if err != nil {
			glog.Errorf("Delete node error: %s", nodeID)
		}
	}
	return nil
}

// cloudNodes constructs and returns api.NodeList from cloudprovider.
func (s *NodeController) cloudNodes() (*api.NodeList, error) {
	instances, ok := s.cloud.Instances()
	if !ok {
		return nil, fmt.Errorf("cloud doesn't support instances")
	}
	matches, err := instances.List(s.matchRE)
	if err != nil {
		return nil, err
	}
	result := &api.NodeList{
		Items: make([]api.Node, len(matches)),
	}
	for i := range matches {
		result.Items[i].Name = matches[i]
		hostIP, err := instances.IPAddress(matches[i])
		if err != nil {
			glog.Errorf("error getting instance ip address for %s: %v", matches[i], err)
		} else {
			result.Items[i].Status.HostIP = hostIP.String()
		}
		resources, err := instances.GetNodeResources(matches[i])
		if err != nil {
			return nil, err
		}
		if resources == nil {
			resources = s.staticResources
		}
		if resources != nil {
			result.Items[i].Spec.Capacity = resources.Capacity
		}
	}
	return result, nil
}
