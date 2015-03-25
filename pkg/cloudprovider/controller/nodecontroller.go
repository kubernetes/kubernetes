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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/record"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/probe"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

const (
	// The constant is used if sync_nodes_status=False. NodeController will not proactively
	// sync node status in this case, but will monitor node status updated from kubelet. If
	// it doesn't receive update for this amount of time, it will start posting "NodeReady==
	// ConditionUnknown". The amount of time before which NodeController start evicting pods
	// is controlled via flag 'pod_eviction_timeout'.
	// Note: be cautious when changing the constant, it must work with nodeStatusUpdateFrequency
	// in kubelet. There are several constraints:
	// 1. nodeMonitorGracePeriod must be N times more than nodeStatusUpdateFrequency, where
	//    N means number of retries allowed for kubelet to post node status. It is pointless
	//    to make nodeMonitorGracePeriod be less than nodeStatusUpdateFrequency, since there
	//    will only be fresh values from Kubelet at an interval of nodeStatusUpdateFrequency.
	//    The constant must be less than podEvictionTimeout.
	// 2. nodeMonitorGracePeriod can't be too large for user experience - larger value takes
	//    longer for user to see up-to-date node status.
	nodeMonitorGracePeriod = 8 * time.Second
	// The constant is used if sync_nodes_status=False, only for node startup. When node
	// is just created, e.g. cluster bootstrap or node creation, we give a longer grace period.
	nodeStartupGracePeriod = 30 * time.Second
	// The constant is used if sync_nodes_status=False. It controls NodeController monitoring
	// period, i.e. how often does NodeController check node status posted from kubelet.
	// Theoretically, this value should be lower than nodeMonitorGracePeriod.
	// TODO: Change node status monitor to watch based.
	nodeMonitorPeriod = 5 * time.Second
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
	recorder           record.EventRecorder
	registerRetryCount int
	podEvictionTimeout time.Duration
	// Method for easy mocking in unittest.
	lookupIP func(host string) ([]net.IP, error)
	now      func() util.Time
}

// NewNodeController returns a new node controller to sync instances from cloudprovider.
func NewNodeController(
	cloud cloudprovider.Interface,
	matchRE string,
	nodes []string,
	staticResources *api.NodeResources,
	kubeClient client.Interface,
	kubeletClient client.KubeletClient,
	recorder record.EventRecorder,
	registerRetryCount int,
	podEvictionTimeout time.Duration) *NodeController {
	return &NodeController{
		cloud:              cloud,
		matchRE:            matchRE,
		nodes:              nodes,
		staticResources:    staticResources,
		kubeClient:         kubeClient,
		kubeletClient:      kubeletClient,
		recorder:           recorder,
		registerRetryCount: registerRetryCount,
		podEvictionTimeout: podEvictionTimeout,
		lookupIP:           net.LookupIP,
		now:                util.Now,
	}
}

// Run creates initial node list and start syncing instances from cloudprovider, if any.
// It also starts syncing or monitoring cluster node status.
// 1. RegisterNodes() is called only once to register all initial nodes (from cloudprovider
//    or from command line flag). To make cluster bootstrap faster, node controller populates
//    node addresses.
// 2. SyncCloudNodes() is called periodically (if enabled) to sync instances from cloudprovider.
//    Node created here will only have specs.
// 3. Depending on how k8s is configured, there are two ways of syncing the node status:
//   3.1 SyncProbedNodeStatus() is called periodically to trigger master to probe kubelet,
//       and incorporate the resulting node status.
//   3.2 MonitorNodeStatus() is called periodically to incorporate the results of node status
//       pushed from kubelet to master.
func (nc *NodeController) Run(period time.Duration, syncNodeList, syncNodeStatus bool) {
	// Register intial set of nodes with their status set.
	var nodes *api.NodeList
	var err error
	record.StartRecording(nc.kubeClient.Events(""))
	if nc.isRunningCloudProvider() {
		if syncNodeList {
			if nodes, err = nc.GetCloudNodesWithSpec(); err != nil {
				glog.Errorf("Error loading initial node from cloudprovider: %v", err)
			}
		} else {
			nodes = &api.NodeList{}
		}
	} else {
		if nodes, err = nc.GetStaticNodesWithSpec(); err != nil {
			glog.Errorf("Error loading initial static nodes: %v", err)
		}
	}
	if nodes, err = nc.PopulateAddresses(nodes); err != nil {
		glog.Errorf("Error getting nodes ips: %v", err)
	}
	if err = nc.RegisterNodes(nodes, nc.registerRetryCount, period); err != nil {
		glog.Errorf("Error registering node list %+v: %v", nodes, err)
	}

	// Start syncing node list from cloudprovider.
	if syncNodeList && nc.isRunningCloudProvider() {
		go util.Forever(func() {
			if err = nc.SyncCloudNodes(); err != nil {
				glog.Errorf("Error syncing cloud: %v", err)
			}
		}, period)
	}

	// Start syncing or monitoring node status.
	if syncNodeStatus {
		go util.Forever(func() {
			if err = nc.SyncProbedNodeStatus(); err != nil {
				glog.Errorf("Error syncing status: %v", err)
			}
		}, period)
	} else {
		go util.Forever(func() {
			if err = nc.MonitorNodeStatus(); err != nil {
				glog.Errorf("Error monitoring node status: %v", err)
			}
		}, nodeMonitorPeriod)
	}
}

// RegisterNodes registers the given list of nodes, it keeps retrying for `retryCount` times.
func (nc *NodeController) RegisterNodes(nodes *api.NodeList, retryCount int, retryInterval time.Duration) error {
	if len(nodes.Items) == 0 {
		return nil
	}

	registered := util.NewStringSet()
	nodes = nc.canonicalizeName(nodes)
	for i := 0; i < retryCount; i++ {
		for _, node := range nodes.Items {
			if registered.Has(node.Name) {
				continue
			}
			_, err := nc.kubeClient.Nodes().Create(&node)
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

// SyncCloudNodes synchronizes the list of instances from cloudprovider to master server.
func (nc *NodeController) SyncCloudNodes() error {
	matches, err := nc.GetCloudNodesWithSpec()
	if err != nil {
		return err
	}
	nodes, err := nc.kubeClient.Nodes().List()
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
			_, err = nc.kubeClient.Nodes().Create(&node)
			if err != nil {
				glog.Errorf("Create node %s error: %v", node.Name, err)
			}
		}
		delete(nodeMap, node.Name)
	}

	// Delete nodes which have been deleted from cloud, but not from kubernetes cluster.
	for nodeID := range nodeMap {
		glog.Infof("Delete node from registry: %s", nodeID)
		err = nc.kubeClient.Nodes().Delete(nodeID)
		if err != nil {
			glog.Errorf("Delete node %s error: %v", nodeID, err)
		}
		nc.deletePods(nodeID)
	}

	return nil
}

// SyncProbedNodeStatus synchronizes cluster nodes status to master server.
func (nc *NodeController) SyncProbedNodeStatus() error {
	nodes, err := nc.kubeClient.Nodes().List()
	if err != nil {
		return err
	}
	nodes, err = nc.PopulateNodesStatus(nodes)
	if err != nil {
		return err
	}
	for _, node := range nodes.Items {
		// We used to skip updating node when node status doesn't change, this is no longer
		// useful after we introduce per-probe status field, e.g. 'LastProbeTime', which will
		// differ in every call of the sync loop.
		glog.V(2).Infof("updating node %v", node.Name)
		_, err = nc.kubeClient.Nodes().Update(&node)
		if err != nil {
			glog.Errorf("error updating node %s: %v", node.Name, err)
		}
	}
	return nil
}

// PopulateNodesStatus populates node status for given list of nodes.
func (nc *NodeController) PopulateNodesStatus(nodes *api.NodeList) (*api.NodeList, error) {
	var wg sync.WaitGroup
	wg.Add(len(nodes.Items))
	for i := range nodes.Items {
		go func(node *api.Node) {
			node.Status.Conditions = nc.DoCheck(node)
			if err := nc.populateNodeInfo(node); err != nil {
				glog.Errorf("Can't collect information for node %s: %v", node.Name, err)
			}
			wg.Done()
		}(&nodes.Items[i])
	}
	wg.Wait()
	return nc.PopulateAddresses(nodes)
}

// populateNodeInfo gets node info from kubelet and update the node.
func (nc *NodeController) populateNodeInfo(node *api.Node) error {
	nodeInfo, err := nc.kubeletClient.GetNodeInfo(node.Name)
	if err != nil {
		return err
	}
	for key, value := range nodeInfo.Capacity {
		node.Spec.Capacity[key] = value
	}
	if node.Status.NodeInfo.BootID != "" &&
		node.Status.NodeInfo.BootID != nodeInfo.NodeSystemInfo.BootID {
		ref := &api.ObjectReference{
			Kind:      "Minion",
			Name:      node.Name,
			UID:       node.UID,
			Namespace: api.NamespaceDefault,
		}
		nc.recorder.Eventf(ref, "rebooted", "Node %s has been rebooted", node.Name)
	}
	node.Status.NodeInfo = nodeInfo.NodeSystemInfo
	return nil
}

// DoCheck performs various condition checks for given node.
func (nc *NodeController) DoCheck(node *api.Node) []api.NodeCondition {
	var conditions []api.NodeCondition

	// Check Condition: NodeReady. TODO: More node conditions.
	oldReadyCondition := nc.getCondition(node, api.NodeReady)
	newReadyCondition := nc.checkNodeReady(node)
	nc.updateLastTransitionTime(oldReadyCondition, newReadyCondition)
	if newReadyCondition.Status != api.ConditionTrue {
		// Node is not ready for this probe, we need to check if pods need to be deleted.
		if newReadyCondition.LastProbeTime.After(newReadyCondition.LastTransitionTime.Add(nc.podEvictionTimeout)) {
			// As long as the node fails, we call delete pods to delete all pods. Node controller sync
			// is not a closed loop process, there is no feedback from other components regarding pod
			// status. Keep listing pods to sanity check if pods are all deleted makes more sense.
			nc.deletePods(node.Name)
		}
	}
	conditions = append(conditions, *newReadyCondition)

	// Check Condition: NodeSchedulable
	oldSchedulableCondition := nc.getCondition(node, api.NodeSchedulable)
	newSchedulableCondition := nc.checkNodeSchedulable(node)
	nc.updateLastTransitionTime(oldSchedulableCondition, newSchedulableCondition)
	conditions = append(conditions, *newSchedulableCondition)

	return conditions
}

// updateLastTransitionTime updates LastTransitionTime for the newCondition based on oldCondition.
func (nc *NodeController) updateLastTransitionTime(oldCondition, newCondition *api.NodeCondition) {
	if oldCondition != nil && oldCondition.Status == newCondition.Status {
		// If node status doesn't change, transition time is same as last time.
		newCondition.LastTransitionTime = oldCondition.LastTransitionTime
	} else {
		// Set transition time to Now() if node status changes or `oldCondition` is nil, which
		// happens only when the node is checked for the first time.
		newCondition.LastTransitionTime = nc.now()
	}
}

// checkNodeSchedulable checks node schedulable condition, without transition timestamp set.
func (nc *NodeController) checkNodeSchedulable(node *api.Node) *api.NodeCondition {
	if node.Spec.Unschedulable {
		return &api.NodeCondition{
			Type:          api.NodeSchedulable,
			Status:        api.ConditionFalse,
			Reason:        "User marked unschedulable during node create/update",
			LastProbeTime: nc.now(),
		}
	} else {
		return &api.NodeCondition{
			Type:          api.NodeSchedulable,
			Status:        api.ConditionTrue,
			Reason:        "Node is schedulable by default",
			LastProbeTime: nc.now(),
		}
	}
}

// checkNodeReady checks raw node ready condition, without transition timestamp set.
func (nc *NodeController) checkNodeReady(node *api.Node) *api.NodeCondition {
	switch status, err := nc.kubeletClient.HealthCheck(node.Name); {
	case err != nil:
		glog.V(2).Infof("NodeController: node %s health check error: %v", node.Name, err)
		return &api.NodeCondition{
			Type:          api.NodeReady,
			Status:        api.ConditionUnknown,
			Reason:        fmt.Sprintf("Node health check error: %v", err),
			LastProbeTime: nc.now(),
		}
	case status == probe.Failure:
		return &api.NodeCondition{
			Type:          api.NodeReady,
			Status:        api.ConditionFalse,
			Reason:        fmt.Sprintf("Node health check failed: kubelet /healthz endpoint returns not ok"),
			LastProbeTime: nc.now(),
		}
	default:
		return &api.NodeCondition{
			Type:          api.NodeReady,
			Status:        api.ConditionTrue,
			Reason:        fmt.Sprintf("Node health check succeeded: kubelet /healthz endpoint returns ok"),
			LastProbeTime: nc.now(),
		}
	}
}

// PopulateAddresses queries Address for given list of nodes.
func (nc *NodeController) PopulateAddresses(nodes *api.NodeList) (*api.NodeList, error) {
	if nc.isRunningCloudProvider() {
		instances, ok := nc.cloud.Instances()
		if !ok {
			return nodes, ErrCloudInstance
		}
		for i := range nodes.Items {
			node := &nodes.Items[i]
			nodeAddresses, err := instances.NodeAddresses(node.Name)
			if err != nil {
				glog.Errorf("error getting instance addresses for %s: %v", node.Name, err)
			} else {
				node.Status.Addresses = nodeAddresses
			}
		}
	} else {
		for i := range nodes.Items {
			node := &nodes.Items[i]
			addr := net.ParseIP(node.Name)
			if addr != nil {
				address := api.NodeAddress{Type: api.NodeLegacyHostIP, Address: addr.String()}
				node.Status.Addresses = []api.NodeAddress{address}
			} else {
				addrs, err := nc.lookupIP(node.Name)
				if err != nil {
					glog.Errorf("Can't get ip address of node %s: %v", node.Name, err)
				} else if len(addrs) == 0 {
					glog.Errorf("No ip address for node %v", node.Name)
				} else {
					address := api.NodeAddress{Type: api.NodeLegacyHostIP, Address: addrs[0].String()}
					node.Status.Addresses = []api.NodeAddress{address}
				}
			}
		}
	}
	return nodes, nil
}

// MonitorNodeStatus verifies node status are constantly updated by kubelet, and if not,
// post "NodeReady==ConditionUnknown". It also evicts all pods if node is not ready or
// not reachable for a long period of time.
func (nc *NodeController) MonitorNodeStatus() error {
	nodes, err := nc.kubeClient.Nodes().List()
	if err != nil {
		return err
	}
	for i := range nodes.Items {
		var gracePeriod time.Duration
		var lastReadyCondition api.NodeCondition
		node := &nodes.Items[i]
		readyCondition := nc.getCondition(node, api.NodeReady)
		if readyCondition == nil {
			// If ready condition is nil, then kubelet (or nodecontroller) never posted node status.
			// A fake ready condition is created, where LastProbeTime and LastTransitionTime is set
			// to node.CreationTimestamp to avoid handle the corner case.
			lastReadyCondition = api.NodeCondition{
				Type:               api.NodeReady,
				Status:             api.ConditionUnknown,
				LastProbeTime:      node.CreationTimestamp,
				LastTransitionTime: node.CreationTimestamp,
			}
			gracePeriod = nodeStartupGracePeriod
		} else {
			// If ready condition is not nil, make a copy of it, since we may modify it in place later.
			lastReadyCondition = *readyCondition
			gracePeriod = nodeMonitorGracePeriod
		}

		// Check last time when NodeReady was updated.
		if nc.now().After(lastReadyCondition.LastProbeTime.Add(gracePeriod)) {
			// NodeReady condition was last set longer ago than gracePeriod, so update it to Unknown
			// (regardless of its current value) in the master, without contacting kubelet.
			if readyCondition == nil {
				glog.V(2).Infof("node %v is never updated by kubelet")
				node.Status.Conditions = append(node.Status.Conditions, api.NodeCondition{
					Type:               api.NodeReady,
					Status:             api.ConditionUnknown,
					Reason:             fmt.Sprintf("Kubelet never posted node status"),
					LastProbeTime:      node.CreationTimestamp,
					LastTransitionTime: nc.now(),
				})
			} else {
				glog.V(2).Infof("node %v hasn't been updated for %+v. Last ready condition is: %+v",
					node.Name, nc.now().Time.Sub(lastReadyCondition.LastProbeTime.Time), lastReadyCondition)
				if lastReadyCondition.Status != api.ConditionUnknown {
					readyCondition.Status = api.ConditionUnknown
					readyCondition.Reason = fmt.Sprintf("Kubelet stopped posting node status")
					// LastProbeTime is the last time we heard from kubelet.
					readyCondition.LastProbeTime = lastReadyCondition.LastProbeTime
					readyCondition.LastTransitionTime = nc.now()
				}
			}
			_, err = nc.kubeClient.Nodes().Update(node)
			if err != nil {
				glog.Errorf("error updating node %s: %v", node.Name, err)
			}
		}

		if readyCondition != nil {
			// Check eviction timeout.
			if lastReadyCondition.Status == api.ConditionFalse &&
				nc.now().After(lastReadyCondition.LastTransitionTime.Add(nc.podEvictionTimeout)) {
				// Node stays in not ready for at least 'podEvictionTimeout' - evict all pods on the unhealthy node.
				nc.deletePods(node.Name)
			}
			if lastReadyCondition.Status == api.ConditionUnknown &&
				nc.now().After(lastReadyCondition.LastProbeTime.Add(nc.podEvictionTimeout-gracePeriod)) {
				// Same as above. Note however, since condition unknown is posted by node controller, which means we
				// need to substract monitoring grace period in order to get the real 'podEvictionTimeout'.
				nc.deletePods(node.Name)
			}
		}
	}
	return nil
}

// GetStaticNodesWithSpec constructs and returns api.NodeList for static nodes. If error
// occurs, an empty NodeList will be returned with a non-nil error info. The method only
// constructs spec fields for nodes.
func (nc *NodeController) GetStaticNodesWithSpec() (*api.NodeList, error) {
	result := &api.NodeList{}
	for _, nodeID := range nc.nodes {
		node := api.Node{
			ObjectMeta: api.ObjectMeta{Name: nodeID},
			Spec: api.NodeSpec{
				Capacity:   nc.staticResources.Capacity,
				ExternalID: nodeID},
		}
		result.Items = append(result.Items, node)
	}
	return result, nil
}

// GetCloudNodesWithSpec constructs and returns api.NodeList from cloudprovider. If error
// occurs, an empty NodeList will be returned with a non-nil error info. The method only
// constructs spec fields for nodes.
func (nc *NodeController) GetCloudNodesWithSpec() (*api.NodeList, error) {
	result := &api.NodeList{}
	instances, ok := nc.cloud.Instances()
	if !ok {
		return result, ErrCloudInstance
	}
	matches, err := instances.List(nc.matchRE)
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
			resources = nc.staticResources
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

// deletePods will delete all pods from master running on given node.
func (nc *NodeController) deletePods(nodeID string) error {
	glog.V(2).Infof("Delete all pods from %v", nodeID)
	// TODO: We don't yet have field selectors from client, see issue #1362.
	pods, err := nc.kubeClient.Pods(api.NamespaceAll).List(labels.Everything())
	if err != nil {
		return err
	}
	for _, pod := range pods.Items {
		if pod.Status.Host != nodeID {
			continue
		}
		glog.V(2).Infof("Delete pod %v", pod.Name)
		if err := nc.kubeClient.Pods(pod.Namespace).Delete(pod.Name); err != nil {
			glog.Errorf("Error deleting pod %v: %v", pod.Name, err)
		}
	}

	return nil
}

// isRunningCloudProvider checks if cluster is running with cloud provider.
func (nc *NodeController) isRunningCloudProvider() bool {
	return nc.cloud != nil && len(nc.matchRE) > 0
}

// canonicalizeName takes a node list and lowercases all nodes' name.
func (nc *NodeController) canonicalizeName(nodes *api.NodeList) *api.NodeList {
	for i := range nodes.Items {
		nodes.Items[i].Name = strings.ToLower(nodes.Items[i].Name)
	}
	return nodes
}

// getCondition returns a condition object for the specific condition
// type, nil if the condition is not set.
func (nc *NodeController) getCondition(node *api.Node, conditionType api.NodeConditionType) *api.NodeCondition {
	for i := range node.Status.Conditions {
		if node.Status.Conditions[i].Type == conditionType {
			return &node.Status.Conditions[i]
		}
	}
	return nil
}
