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
	"errors"
	"fmt"
	"net"
	"reflect"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/probe"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

var (
	ErrRegistration = errors.New("unable to register all nodes.")
)

type NodeController struct {
	cloud           cloudprovider.Interface
	matchRE         string
	staticResources *api.NodeResources
	nodes           []string
	kubeClient      client.Interface
	kubeletClient   client.KubeletHealthChecker
}

// NewNodeController returns a new node controller to sync instances from cloudprovider.
// TODO: NodeController health checker should be a separate package other than
// kubeletclient, node health check != kubelet health check.
func NewNodeController(
	cloud cloudprovider.Interface,
	matchRE string,
	nodes []string,
	staticResources *api.NodeResources,
	kubeClient client.Interface,
	kubeletClient client.KubeletHealthChecker) *NodeController {
	return &NodeController{
		cloud:           cloud,
		matchRE:         matchRE,
		nodes:           nodes,
		staticResources: staticResources,
		kubeClient:      kubeClient,
		kubeletClient:   kubeletClient,
	}
}

// Run creates initial node list and start syncing instances from cloudprovider if any.
// It also starts syncing cluster node status.
func (s *NodeController) Run(period time.Duration, retryCount int) {
	// Register intial set of nodes with their status set.
	var nodes *api.NodeList
	var err error
	if s.isRunningCloudProvider() {
		nodes, err = s.CloudNodes()
		if err != nil {
			glog.Errorf("Error loading initial node from cloudprovider: %v", err)
		}
	} else {
		nodes, err = s.StaticNodes()
		if err != nil {
			glog.Errorf("Error loading initial static nodes")
		}
	}
	nodes = s.DoChecks(nodes)
	if err := s.RegisterNodes(nodes, retryCount, period); err != nil {
		glog.Errorf("Error registrying node list: %+v", nodes)
	}

	// Start syncing node list from cloudprovider.
	if s.isRunningCloudProvider() {
		go util.Forever(func() {
			if err = s.SyncCloud(); err != nil {
				glog.Errorf("Error syncing cloud: %v", err)
			}
		}, period)
	}

	// Start syncing node status.
	go util.Forever(func() {
		if err = s.SyncNodeStatus(); err != nil {
			glog.Errorf("Error syncing status: %v", err)
		}
	}, period)
}

// RegisterNodes registers the given list of nodes, it keeps retrying for `retryCount` times.
func (s *NodeController) RegisterNodes(nodes *api.NodeList, retryCount int, retryInterval time.Duration) error {
	registered := util.NewStringSet()
	for i := 0; i < retryCount; i++ {
		for _, node := range nodes.Items {
			if registered.Has(node.Name) {
				continue
			}
			_, err := s.kubeClient.Nodes().Create(&node)
			if err == nil {
				registered.Insert(node.Name)
				glog.Infof("Registered node in registry: %s", node.Name)
			} else {
				glog.Errorf("Error registrying node %s, retrying: %s", node.Name, err)
			}
			if registered.Len() == len(nodes.Items) {
				glog.Infof("Successfully Registered all nodes")
				return nil
			}
		}
		time.Sleep(retryInterval)
	}
	if registered.Len() != len(nodes.Items) {
		return ErrRegistration
	} else {
		return nil
	}
}

// SyncCloud synchronizes the list of instances from cloudprovider to master server.
func (s *NodeController) SyncCloud() error {
	matches, err := s.CloudNodes()
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

	// Create nodes which have been created in cloud, but not in kubernetes cluster.
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

	// Delete nodes which have been deleted from cloud, but not from kubernetes cluster.
	for nodeID := range nodeMap {
		glog.Infof("Delete node from registry: %s", nodeID)
		err = s.kubeClient.Nodes().Delete(nodeID)
		if err != nil {
			glog.Errorf("Delete node error: %s", nodeID)
		}
	}

	return nil
}

// SyncNodeStatus synchronizes cluster nodes status to master server.
func (s *NodeController) SyncNodeStatus() error {
	nodes, err := s.kubeClient.Nodes().List()
	if err != nil {
		return err
	}
	oldNodes := make(map[string]api.Node)
	for _, node := range nodes.Items {
		oldNodes[node.Name] = node
	}
	nodes = s.DoChecks(nodes)
	for _, node := range nodes.Items {
		if reflect.DeepEqual(node, oldNodes[node.Name]) {
			glog.V(2).Infof("skip updating node %v", node.Name)
			continue
		}
		glog.V(2).Infof("updating node %v", node.Name)
		_, err = s.kubeClient.Nodes().Update(&node)
		if err != nil {
			glog.Errorf("error updating node %s: %v", node.Name, err)
		}
	}
	return nil
}

// DoChecks performs health checking for given list of nodes.
func (s *NodeController) DoChecks(nodes *api.NodeList) *api.NodeList {
	var wg sync.WaitGroup
	wg.Add(len(nodes.Items))
	for i := range nodes.Items {
		go func(node *api.Node) {
			node.Status.Conditions = s.DoCheck(node)
			wg.Done()
		}(&nodes.Items[i])
	}
	wg.Wait()
	return nodes
}

// DoCheck performs health checking for given node.
func (s *NodeController) DoCheck(node *api.Node) []api.NodeCondition {
	var conditions []api.NodeCondition
	switch status, err := s.kubeletClient.HealthCheck(node.Name); {
	case err != nil:
		glog.V(2).Infof("NodeController: node %s health check error: %v", node.Name, err)
		conditions = append(conditions, api.NodeCondition{
			Kind:   api.NodeReady,
			Status: api.ConditionUnknown,
		})
	case status == probe.Failure:
		conditions = append(conditions, api.NodeCondition{
			Kind:   api.NodeReady,
			Status: api.ConditionNone,
		})
	default:
		conditions = append(conditions, api.NodeCondition{
			Kind:   api.NodeReady,
			Status: api.ConditionFull,
		})
	}
	glog.V(5).Infof("NodeController: node %q status was %+v", node.Name, conditions)
	return conditions
}

// StaticNodes constructs and returns api.NodeList for static nodes. If error
// occurs, an empty NodeList will be returned with a non-nil error info.
func (s *NodeController) StaticNodes() (*api.NodeList, error) {
	result := &api.NodeList{}
	for _, nodeID := range s.nodes {
		node := api.Node{
			ObjectMeta: api.ObjectMeta{Name: nodeID},
			Spec:       api.NodeSpec{Capacity: s.staticResources.Capacity},
		}
		addr := net.ParseIP(nodeID)
		if addr != nil {
			node.Status.HostIP = nodeID
		} else {
			addrs, err := net.LookupIP(nodeID)
			if err != nil {
				glog.Errorf("Can't get ip address of node %v", nodeID)
			} else if len(addrs) == 0 {
				glog.Errorf("No ip address for node %v", nodeID)
			} else {
				node.Status.HostIP = addrs[0].String()
			}
		}
		result.Items = append(result.Items, node)
	}
	return result, nil
}

// CloudNodes constructs and returns api.NodeList from cloudprovider. If error
// occurs, an empty NodeList will be returned with a non-nil error info.
func (s *NodeController) CloudNodes() (*api.NodeList, error) {
	result := &api.NodeList{}
	instances, ok := s.cloud.Instances()
	if !ok {
		return result, fmt.Errorf("cloud doesn't support instances")
	}
	matches, err := instances.List(s.matchRE)
	if err != nil {
		return result, err
	}
	for i := range matches {
		node := api.Node{}
		node.Name = matches[i]
		hostIP, err := instances.IPAddress(matches[i])
		if err != nil {
			glog.Errorf("error getting instance ip address for %s: %v", matches[i], err)
		} else {
			node.Status.HostIP = hostIP.String()
		}
		resources, err := instances.GetNodeResources(matches[i])
		if err != nil {
			return nil, err
		}
		if resources == nil {
			resources = s.staticResources
		}
		if resources != nil {
			node.Spec.Capacity = resources.Capacity
		}
		result.Items = append(result.Items, node)
	}
	return result, nil
}

// isRunningCloudProvider checks if cluster is running with cloud provider.
func (s *NodeController) isRunningCloudProvider() bool {
	return s.cloud != nil && len(s.matchRE) > 0
}
