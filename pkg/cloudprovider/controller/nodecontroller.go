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
	"strings"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	apierrors "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/probe"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

var (
	ErrRegistration   = errors.New("unable to register all nodes.")
	ErrQueryIPAddress = errors.New("unable to query IP address.")
	ErrCloudInstance  = errors.New("cloud provider doesn't support instances.")
)

type NodeController struct {
	cloud              cloudprovider.Interface
	matchRE            string
	staticResources    *api.NodeResources
	nodes              []string
	kubeClient         client.Interface
	kubeletClient      client.KubeletClient
	registerRetryCount int
	podEvictionTimeout time.Duration
	lookupIP           func(host string) ([]net.IP, error)
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
	kubeletClient client.KubeletClient,
	registerRetryCount int,
	podEvictionTimeout time.Duration) *NodeController {
	return &NodeController{
		cloud:              cloud,
		matchRE:            matchRE,
		nodes:              nodes,
		staticResources:    staticResources,
		kubeClient:         kubeClient,
		kubeletClient:      kubeletClient,
		registerRetryCount: registerRetryCount,
		podEvictionTimeout: podEvictionTimeout,
		lookupIP:           net.LookupIP,
	}
}

// Run creates initial node list and start syncing instances from cloudprovider if any.
// It also starts syncing cluster node status.
// 1. RegisterNodes() is called only once to register all initial nodes (from cloudprovider
//    or from command line flag). To make cluster bootstrap faster, node controller populates
//    node addresses.
// 2. SyncCloud() is called periodically (if enabled) to sync instances from cloudprovider.
//    Node created here will only have specs.
// 3. SyncNodeStatus() is called periodically (if enabled) to sync node status for nodes in
//    k8s cluster.
func (s *NodeController) Run(period time.Duration, syncNodeList, syncNodeStatus bool) {
	// Register intial set of nodes with their status set.
	var nodes *api.NodeList
	var err error
	if s.isRunningCloudProvider() {
		if syncNodeList {
			nodes, err = s.GetCloudNodesWithSpec()
			if err != nil {
				glog.Errorf("Error loading initial node from cloudprovider: %v", err)
			}
		} else {
			nodes = &api.NodeList{}
		}
	} else {
		nodes, err = s.GetStaticNodesWithSpec()
		if err != nil {
			glog.Errorf("Error loading initial static nodes: %v", err)
		}
	}
	nodes, err = s.PopulateAddresses(nodes)
	if err != nil {
		glog.Errorf("Error getting nodes ips: %v", err)
	}
	if err = s.RegisterNodes(nodes, s.registerRetryCount, period); err != nil {
		glog.Errorf("Error registering node list %+v: %v", nodes, err)
	}

	// Start syncing node list from cloudprovider.
	if syncNodeList && s.isRunningCloudProvider() {
		go util.Forever(func() {
			if err = s.SyncCloud(); err != nil {
				glog.Errorf("Error syncing cloud: %v", err)
			}
		}, period)
	}

	if syncNodeStatus {
		// Start syncing node status.
		go util.Forever(func() {
			if err = s.SyncNodeStatus(); err != nil {
				glog.Errorf("Error syncing status: %v", err)
			}
		}, period)
	} else {
		// Start checking node reachability and evicting timeouted pods.
		go util.Forever(func() {
			if err = s.EvictTimeoutedPods(); err != nil {
				glog.Errorf("Error evicting timeouted pods: %v", err)
			}
		}, period)
	}
}

// RegisterNodes registers the given list of nodes, it keeps retrying for `retryCount` times.
func (s *NodeController) RegisterNodes(nodes *api.NodeList, retryCount int, retryInterval time.Duration) error {
	registered := util.NewStringSet()
	nodes = s.canonicalizeName(nodes)
	for i := 0; i < retryCount; i++ {
		for _, node := range nodes.Items {
			if registered.Has(node.Name) {
				continue
			}
			_, err := s.kubeClient.Nodes().Create(&node)
			if err == nil || apierrors.IsAlreadyExists(err) {
				registered.Insert(node.Name)
				glog.Infof("Registered node in registry: %s", node.Name)
			} else {
				glog.Errorf("Error registering node %s, retrying: %s", node.Name, err)
			}
			if registered.Len() == len(nodes.Items) {
				glog.Infof("Successfully registered all nodes")
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
	matches, err := s.GetCloudNodesWithSpec()
	if err != nil {
		return err
	}
	nodes, err := s.kubeClient.Nodes().List()
	if err != nil {
		return err
	}
	nodeMap := make(map[string]*api.Node)
	for i := range nodes.Items {
		node := nodes.Items[i]
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
		s.deletePods(nodeID)
	}

	return nil
}

// SyncNodeStatus synchronizes cluster nodes status to master server.
func (s *NodeController) SyncNodeStatus() error {
	nodes, err := s.kubeClient.Nodes().List()
	if err != nil {
		return err
	}
	nodes = s.UpdateNodesStatus(nodes)
	nodes, err = s.PopulateAddresses(nodes)
	if err != nil {
		return err
	}
	for _, node := range nodes.Items {
		// We used to skip updating node when node status doesn't change, this is no longer
		// useful after we introduce per-probe status field, e.g. 'LastProbeTime', which will
		// differ in every call of the sync loop.
		glog.V(2).Infof("updating node %v", node.Name)
		_, err = s.kubeClient.Nodes().Update(&node)
		if err != nil {
			glog.Errorf("error updating node %s: %v", node.Name, err)
		}
	}
	return nil
}

// EvictTimeoutedPods verifies if nodes are reachable by checking the time of last probe
// and deletes pods from not reachable nodes.
func (s *NodeController) EvictTimeoutedPods() error {
	nodes, err := s.kubeClient.Nodes().List()
	if err != nil {
		return err
	}
	for _, node := range nodes.Items {
		if util.Now().After(latestReadyTime(&node).Add(s.podEvictionTimeout)) {
			s.deletePods(node.Name)
		}
	}
	return nil
}

func latestReadyTime(node *api.Node) util.Time {
	readyTime := node.ObjectMeta.CreationTimestamp
	for _, condition := range node.Status.Conditions {
		if condition.Type == api.NodeReady &&
			condition.Status == api.ConditionFull &&
			condition.LastProbeTime.After(readyTime.Time) {
			readyTime = condition.LastProbeTime
		}
	}
	return readyTime
}

// PopulateAddresses queries Address for given list of nodes.
func (s *NodeController) PopulateAddresses(nodes *api.NodeList) (*api.NodeList, error) {
	if s.isRunningCloudProvider() {
		instances, ok := s.cloud.Instances()
		if !ok {
			return nodes, ErrCloudInstance
		}
		for i := range nodes.Items {
			node := &nodes.Items[i]
			hostIP, err := instances.IPAddress(node.Name)
			if err != nil {
				glog.Errorf("error getting instance ip address for %s: %v", node.Name, err)
			} else {
				address := api.NodeAddress{Type: api.NodeLegacyHostIP, Address: hostIP.String()}
				api.AddToNodeAddresses(&node.Status.Addresses, address)
			}
		}
	} else {
		for i := range nodes.Items {
			node := &nodes.Items[i]
			addr := net.ParseIP(node.Name)
			if addr != nil {
				address := api.NodeAddress{Type: api.NodeLegacyHostIP, Address: addr.String()}
				api.AddToNodeAddresses(&node.Status.Addresses, address)
			} else {
				addrs, err := s.lookupIP(node.Name)
				if err != nil {
					glog.Errorf("Can't get ip address of node %s: %v", node.Name, err)
				} else if len(addrs) == 0 {
					glog.Errorf("No ip address for node %v", node.Name)
				} else {
					address := api.NodeAddress{Type: api.NodeLegacyHostIP, Address: addrs[0].String()}
					api.AddToNodeAddresses(&node.Status.Addresses, address)
				}
			}
		}
	}
	return nodes, nil
}

// UpdateNodesStatus performs health checking for given list of nodes.
func (s *NodeController) UpdateNodesStatus(nodes *api.NodeList) *api.NodeList {
	var wg sync.WaitGroup
	wg.Add(len(nodes.Items))
	for i := range nodes.Items {
		go func(node *api.Node) {
			node.Status.Conditions = s.DoCheck(node)
			if err := s.updateNodeInfo(node); err != nil {
				glog.Errorf("Can't collect information for node %s: %v", node.Name, err)
			}
			wg.Done()
		}(&nodes.Items[i])
	}
	wg.Wait()
	return nodes
}

func (s *NodeController) updateNodeInfo(node *api.Node) error {
	nodeInfo, err := s.kubeletClient.GetNodeInfo(node.Name)
	if err != nil {
		return err
	}
	for key, value := range nodeInfo.Capacity {
		node.Spec.Capacity[key] = value
	}
	node.Status.NodeInfo = nodeInfo.NodeSystemInfo
	return nil
}

// DoCheck performs health checking for given node.
func (s *NodeController) DoCheck(node *api.Node) []api.NodeCondition {
	var conditions []api.NodeCondition

	// Check Condition: NodeReady. TODO: More node conditions.
	oldReadyCondition := s.getCondition(node, api.NodeReady)
	newReadyCondition := s.checkNodeReady(node)
	if oldReadyCondition != nil && oldReadyCondition.Status == newReadyCondition.Status {
		// If node status doesn't change, transition time is same as last time.
		newReadyCondition.LastTransitionTime = oldReadyCondition.LastTransitionTime
	} else {
		// Set transition time to Now() if node status changes or `oldReadyCondition` is nil, which
		// happens only when the node is checked for the first time.
		newReadyCondition.LastTransitionTime = util.Now()
	}

	if newReadyCondition.Status != api.ConditionFull {
		// Node is not ready for this probe, we need to check if pods need to be deleted.
		if newReadyCondition.LastProbeTime.After(newReadyCondition.LastTransitionTime.Add(s.podEvictionTimeout)) {
			// As long as the node fails, we call delete pods to delete all pods. Node controller sync
			// is not a closed loop process, there is no feedback from other components regarding pod
			// status. Keep listing pods to sanity check if pods are all deleted makes more sense.
			s.deletePods(node.Name)
		}
	}

	conditions = append(conditions, *newReadyCondition)

	return conditions
}

// checkNodeReady checks raw node ready condition, without transition timestamp set.
func (s *NodeController) checkNodeReady(node *api.Node) *api.NodeCondition {
	switch status, err := s.kubeletClient.HealthCheck(node.Name); {
	case err != nil:
		glog.V(2).Infof("NodeController: node %s health check error: %v", node.Name, err)
		return &api.NodeCondition{
			Type:          api.NodeReady,
			Status:        api.ConditionUnknown,
			Reason:        fmt.Sprintf("Node health check error: %v", err),
			LastProbeTime: util.Now(),
		}
	case status == probe.Failure:
		return &api.NodeCondition{
			Type:          api.NodeReady,
			Status:        api.ConditionNone,
			Reason:        fmt.Sprintf("Node health check failed: kubelet /healthz endpoint returns not ok"),
			LastProbeTime: util.Now(),
		}
	default:
		return &api.NodeCondition{
			Type:          api.NodeReady,
			Status:        api.ConditionFull,
			Reason:        fmt.Sprintf("Node health check succeeded: kubelet /healthz endpoint returns ok"),
			LastProbeTime: util.Now(),
		}
	}
}

// deletePods will delete all pods from master running on given node.
func (s *NodeController) deletePods(nodeID string) error {
	glog.V(2).Infof("Delete all pods from %v", nodeID)
	// TODO: We don't yet have field selectors from client, see issue #1362.
	pods, err := s.kubeClient.Pods(api.NamespaceAll).List(labels.Everything())
	if err != nil {
		return err
	}
	for _, pod := range pods.Items {
		if pod.Status.Host != nodeID {
			continue
		}
		glog.V(2).Infof("Delete pod %v", pod.Name)
		if err := s.kubeClient.Pods(pod.Namespace).Delete(pod.Name); err != nil {
			glog.Errorf("Error deleting pod %v: %v", pod.Name, err)
		}
	}

	return nil
}

// GetStaticNodesWithSpec constructs and returns api.NodeList for static nodes. If error
// occurs, an empty NodeList will be returned with a non-nil error info. The
// method only constructs spec fields for nodes.
func (s *NodeController) GetStaticNodesWithSpec() (*api.NodeList, error) {
	result := &api.NodeList{}
	for _, nodeID := range s.nodes {
		node := api.Node{
			ObjectMeta: api.ObjectMeta{Name: nodeID},
			Spec:       api.NodeSpec{Capacity: s.staticResources.Capacity},
		}
		result.Items = append(result.Items, node)
	}
	return result, nil
}

// GetCloudNodesWithSpec constructs and returns api.NodeList from cloudprovider. If error
// occurs, an empty NodeList will be returned with a non-nil error info. The
// method only constructs spec fields for nodes.
func (s *NodeController) GetCloudNodesWithSpec() (*api.NodeList, error) {
	result := &api.NodeList{}
	instances, ok := s.cloud.Instances()
	if !ok {
		return result, ErrCloudInstance
	}
	matches, err := instances.List(s.matchRE)
	if err != nil {
		return result, err
	}
	for i := range matches {
		node := api.Node{}
		node.Name = matches[i]
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
		instanceID, err := instances.ExternalID(node.Name)
		if err != nil {
			glog.Errorf("error getting instance id for %s: %v", node.Name, err)
		} else {
			node.Spec.ExternalID = instanceID
		}
		result.Items = append(result.Items, node)
	}
	return result, nil
}

// isRunningCloudProvider checks if cluster is running with cloud provider.
func (s *NodeController) isRunningCloudProvider() bool {
	return s.cloud != nil && len(s.matchRE) > 0
}

// canonicalizeName takes a node list and lowercases all nodes' name.
func (s *NodeController) canonicalizeName(nodes *api.NodeList) *api.NodeList {
	for i := range nodes.Items {
		nodes.Items[i].Name = strings.ToLower(nodes.Items[i].Name)
	}
	return nodes
}

// getCondition returns a condition object for the specific condition
// type, nil if the condition is not set.
func (s *NodeController) getCondition(node *api.Node, conditionType api.NodeConditionType) *api.NodeCondition {
	for i := range node.Status.Conditions {
		if node.Status.Conditions[i].Type == conditionType {
			return &node.Status.Conditions[i]
		}
	}
	return nil
}
