/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package nodecontroller

import (
	"errors"
	"fmt"
	"net"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	apierrors "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/record"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

var (
	ErrRegistration   = errors.New("unable to register all nodes.")
	ErrQueryIPAddress = errors.New("unable to query IP address.")
	ErrCloudInstance  = errors.New("cloud provider doesn't support instances.")
)

// nodeStatusUpdateRetry controls the number of retries of writing NodeStatus update.
const nodeStatusUpdateRetry = 5

type nodeStatusData struct {
	probeTimestamp           util.Time
	readyTransitionTimestamp util.Time
	status                   api.NodeStatus
}

type NodeController struct {
	cloud                   cloudprovider.Interface
	matchRE                 string
	staticResources         *api.NodeResources
	nodes                   []string
	kubeClient              client.Interface
	recorder                record.EventRecorder
	registerRetryCount      int
	podEvictionTimeout      time.Duration
	deletingPodsRateLimiter util.RateLimiter
	// per Node map storing last observed Status together with a local time when it was observed.
	// This timestamp is to be used instead of LastProbeTime stored in Condition. We do this
	// to aviod the problem with time skew across the cluster.
	nodeStatusMap map[string]nodeStatusData
	// Value used if sync_nodes_status=False. NodeController will not proactively
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
	nodeMonitorGracePeriod time.Duration
	// Value used if sync_nodes_status=False, only for node startup. When node
	// is just created, e.g. cluster bootstrap or node creation, we give a longer grace period.
	nodeStartupGracePeriod time.Duration
	// Value controlling NodeController monitoring period, i.e. how often does NodeController
	// check node status posted from kubelet. This value should be lower than nodeMonitorGracePeriod.
	// TODO: Change node status monitor to watch based.
	nodeMonitorPeriod time.Duration
	clusterName       string
	clusterCIDR       *net.IPNet
	allocateNodeCIDRs bool
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
	registerRetryCount int,
	podEvictionTimeout time.Duration,
	deletingPodsRateLimiter util.RateLimiter,
	nodeMonitorGracePeriod time.Duration,
	nodeStartupGracePeriod time.Duration,
	nodeMonitorPeriod time.Duration,
	clusterName string,
	clusterCIDR *net.IPNet,
	allocateNodeCIDRs bool) *NodeController {
	eventBroadcaster := record.NewBroadcaster()
	recorder := eventBroadcaster.NewRecorder(api.EventSource{Component: "controllermanager"})
	if kubeClient != nil {
		glog.Infof("Sending events to api server.")
		eventBroadcaster.StartRecordingToSink(kubeClient.Events(""))
	} else {
		glog.Infof("No api server defined - no events will be sent to API server.")
	}
	if allocateNodeCIDRs && clusterCIDR == nil {
		glog.Fatal("NodeController: Must specify clusterCIDR if allocateNodeCIDRs == true.")
	}
	return &NodeController{
		cloud:                   cloud,
		matchRE:                 matchRE,
		nodes:                   nodes,
		staticResources:         staticResources,
		kubeClient:              kubeClient,
		recorder:                recorder,
		registerRetryCount:      registerRetryCount,
		podEvictionTimeout:      podEvictionTimeout,
		deletingPodsRateLimiter: deletingPodsRateLimiter,
		nodeStatusMap:           make(map[string]nodeStatusData),
		nodeMonitorGracePeriod:  nodeMonitorGracePeriod,
		nodeMonitorPeriod:       nodeMonitorPeriod,
		nodeStartupGracePeriod:  nodeStartupGracePeriod,
		lookupIP:                net.LookupIP,
		now:                     util.Now,
		clusterName:             clusterName,
		clusterCIDR:             clusterCIDR,
		allocateNodeCIDRs:       allocateNodeCIDRs,
	}
}

// Generates num pod CIDRs that could be assigned to nodes.
func (nc *NodeController) generateCIDRs(num int) util.StringSet {
	res := util.NewStringSet()
	cidrIP := nc.clusterCIDR.IP.To4()
	for i := 0; i < num; i++ {
		// TODO: Make the CIDRs configurable.
		b1 := byte(i >> 8)
		b2 := byte(i % 256)
		res.Insert(fmt.Sprintf("%d.%d.%d.0/24", cidrIP[0], cidrIP[1]+b1, cidrIP[2]+b2))
	}
	return res
}

// For each node from newNodes, finds its current spec in registeredNodes.
// If it is not there, it gets a new valid CIDR assigned.
func (nc *NodeController) reconcilePodCIDRs(newNodes, registeredNodes *api.NodeList) *api.NodeList {
	registeredCIDRs := make(map[string]string)
	availableCIDRs := nc.generateCIDRs(len(newNodes.Items) + len(registeredNodes.Items))
	for _, node := range registeredNodes.Items {
		registeredCIDRs[node.Name] = node.Spec.PodCIDR
		availableCIDRs.Delete(node.Spec.PodCIDR)
	}
	for i, node := range newNodes.Items {
		podCIDR, registered := registeredCIDRs[node.Name]
		if !registered {
			podCIDR, _ = availableCIDRs.PopAny()
		}
		newNodes.Items[i].Spec.PodCIDR = podCIDR
	}
	return newNodes
}

func (nc *NodeController) configureNodeCIDR(node *api.Node) {
	instances, ok := nc.cloud.Instances()
	if !ok {
		glog.Errorf("Error configuring node %s: CloudProvider does not support Instances()", node.Name)
		return
	}
	err := instances.Configure(node.Name, &node.Spec)
	if err != nil {
		glog.Errorf("Error configuring node %s: %s", node.Name, err)
		// The newly assigned CIDR was not properly configured, so don't save it in the API server.
		node.Spec.PodCIDR = ""
	}
}

func (nc *NodeController) unassignNodeCIDR(nodeName string) {
	instances, ok := nc.cloud.Instances()
	if !ok {
		glog.Errorf("Error deconfiguring node %s: CloudProvider does not support Instances()", nodeName)
		return
	}
	err := instances.Release(nodeName)
	if err != nil {
		glog.Errorf("Error deconfiguring node %s: %s", nodeName, err)
	}
}

// Run creates initial node list and start syncing instances from cloudprovider, if any.
// It also starts syncing or monitoring cluster node status.
// 1. registerNodes() is called only once to register all initial nodes (from cloudprovider
//    or from command line flag). To make cluster bootstrap faster, node controller populates
//    node addresses.
// 2. syncCloudNodes() is called periodically (if enabled) to sync instances from cloudprovider.
//    Node created here will only have specs.
// 3. monitorNodeStatus() is called periodically to incorporate the results of node status
//    pushed from kubelet to master.
func (nc *NodeController) Run(period time.Duration, syncNodeList bool) {
	// Register intial set of nodes with their status set.
	var nodes *api.NodeList
	var err error
	if nc.isRunningCloudProvider() {
		if syncNodeList {
			if nodes, err = nc.getCloudNodesWithSpec(); err != nil {
				glog.Errorf("Error loading initial node from cloudprovider: %v", err)
			}
		} else {
			nodes = &api.NodeList{}
		}
	} else {
		if nodes, err = nc.getStaticNodesWithSpec(); err != nil {
			glog.Errorf("Error loading initial static nodes: %v", err)
		}
	}

	if nodes, err = nc.populateAddresses(nodes); err != nil {
		glog.Errorf("Error getting nodes ips: %v", err)
	}
	if nc.isRunningCloudProvider() && nc.allocateNodeCIDRs {
		nc.reconcilePodCIDRs(nodes, &api.NodeList{})
	}
	if err := nc.registerNodes(nodes, nc.registerRetryCount, period); err != nil {
		glog.Errorf("Error registering node list %+v: %v", nodes, err)
	}

	// Start syncing node list from cloudprovider.
	if syncNodeList && nc.isRunningCloudProvider() {
		go util.Forever(func() {
			if err := nc.syncCloudNodes(); err != nil {
				glog.Errorf("Error syncing cloud: %v", err)
			}
		}, period)
	}

	// Start monitoring node status.
	go util.Forever(func() {
		if err := nc.monitorNodeStatus(); err != nil {
			glog.Errorf("Error monitoring node status: %v", err)
		}
	}, nc.nodeMonitorPeriod)
}

// registerNodes registers the given list of nodes, it keeps retrying for `retryCount` times.
func (nc *NodeController) registerNodes(nodes *api.NodeList, retryCount int, retryInterval time.Duration) error {
	if len(nodes.Items) == 0 {
		return nil
	}
	nodes = nc.canonicalizeName(nodes)
	toRegister := util.NewStringSet()
	var wg sync.WaitGroup
	var successfullyRegistered int32 = 0
	for i := range nodes.Items {
		node := &nodes.Items[i]
		if !toRegister.Has(node.Name) {
			wg.Add(1)
			toRegister.Insert(node.Name)
			go func(n *api.Node) {
				defer wg.Done()
				for i := 0; i < retryCount; i++ {
					if nc.isRunningCloudProvider() && nc.allocateNodeCIDRs {
						nc.configureNodeCIDR(n)
					}
					_, err := nc.kubeClient.Nodes().Create(n)
					if err == nil || apierrors.IsAlreadyExists(err) {
						glog.Infof("Registered node in registry: %v", n.Name)
						atomic.AddInt32(&successfullyRegistered, 1)
						return
					} else {
						glog.Errorf("Error registering node %v (retries left: %v): %v", n.Name, retryCount-i-1, err)
					}
					time.Sleep(retryInterval)
				}
				glog.Errorf("Unable to register node %v", n.Name)
			}(node)
		}
	}
	wg.Wait()
	if int32(toRegister.Len()) != atomic.LoadInt32(&successfullyRegistered) {
		return ErrRegistration
	} else {
		return nil
	}
}

// syncCloudNodes synchronizes the list of instances from cloudprovider to master server.
func (nc *NodeController) syncCloudNodes() error {
	matches, err := nc.getCloudNodesWithSpec()
	if err != nil {
		return err
	}
	nodes, err := nc.kubeClient.Nodes().List(labels.Everything(), fields.Everything())
	if err != nil {
		return err
	}
	nodeMap := make(map[string]*api.Node)
	nodeMapLock := sync.Mutex{}
	for i := range nodes.Items {
		node := nodes.Items[i]
		nodeMapLock.Lock()
		nodeMap[node.Name] = &node
		nodeMapLock.Unlock()
	}
	if nc.allocateNodeCIDRs {
		nc.reconcilePodCIDRs(matches, nodes)
	}
	var wg sync.WaitGroup
	wg.Add(len(matches.Items))
	// Create nodes which have been created in cloud, but not in kubernetes cluster
	// Skip nodes if we hit an error while trying to get their addresses.
	for i := range matches.Items {
		go func(node *api.Node) {
			defer wg.Done()
			nodeMapLock.Lock()
			_, ok := nodeMap[node.Name]
			nodeMapLock.Unlock()
			if !ok {
				glog.V(3).Infof("Querying addresses for new node: %s", node.Name)
				nodeList := &api.NodeList{}
				nodeList.Items = []api.Node{*node}
				_, err = nc.populateAddresses(nodeList)
				if err != nil {
					glog.Errorf("Error fetching addresses for new node %s: %v", node.Name, err)
					return
				}
				node.Status.Addresses = nodeList.Items[0].Status.Addresses
				if nc.allocateNodeCIDRs {
					nc.configureNodeCIDR(node)
				}
				glog.Infof("Create node in registry: %s", node.Name)
				_, err = nc.kubeClient.Nodes().Create(node)
				if err != nil {
					glog.Errorf("Create node %s error: %v", node.Name, err)
				}
			}
			nodeMapLock.Lock()
			delete(nodeMap, node.Name)
			nodeMapLock.Unlock()
		}(&matches.Items[i])
	}
	wg.Wait()

	wg.Add(len(nodeMap))
	// Delete nodes which have been deleted from cloud, but not from kubernetes cluster.
	for nodeID := range nodeMap {
		go func(nodeID string) {
			defer wg.Done()
			if nc.allocateNodeCIDRs {
				nc.unassignNodeCIDR(nodeID)
			}
			glog.Infof("Delete node from registry: %s", nodeID)
			err = nc.kubeClient.Nodes().Delete(nodeID)
			if err != nil {
				glog.Errorf("Delete node %s error: %v", nodeID, err)
			}
			nc.deletePods(nodeID)
		}(nodeID)
	}
	wg.Wait()

	return nil
}

// populateAddresses queries Address for given list of nodes.
func (nc *NodeController) populateAddresses(nodes *api.NodeList) (*api.NodeList, error) {
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

func (nc *NodeController) recordNodeEvent(node *api.Node, event string) {
	ref := &api.ObjectReference{
		Kind:      "Node",
		Name:      node.Name,
		UID:       types.UID(node.Name),
		Namespace: "",
	}
	glog.V(2).Infof("Recording %s event message for node %s", event, node.Name)
	// TODO: This requires a transaction, either both node status is updated
	// and event is recorded or neither should happen, see issue #6055.
	nc.recorder.Eventf(ref, event, "Node %s is now %s", node.Name, event)
}

// For a given node checks its conditions and tries to update it. Returns grace period to which given node
// is entitled, state of current and last observed Ready Condition, and an error if it ocured.
func (nc *NodeController) tryUpdateNodeStatus(node *api.Node) (time.Duration, api.NodeCondition, *api.NodeCondition, error) {
	var err error
	var gracePeriod time.Duration
	var lastReadyCondition api.NodeCondition
	readyCondition := nc.getCondition(&node.Status, api.NodeReady)
	if readyCondition == nil {
		// If ready condition is nil, then kubelet (or nodecontroller) never posted node status.
		// A fake ready condition is created, where LastProbeTime and LastTransitionTime is set
		// to node.CreationTimestamp to avoid handle the corner case.
		lastReadyCondition = api.NodeCondition{
			Type:               api.NodeReady,
			Status:             api.ConditionUnknown,
			LastHeartbeatTime:  node.CreationTimestamp,
			LastTransitionTime: node.CreationTimestamp,
		}
		gracePeriod = nc.nodeStartupGracePeriod
		nc.nodeStatusMap[node.Name] = nodeStatusData{
			status:                   node.Status,
			probeTimestamp:           node.CreationTimestamp,
			readyTransitionTimestamp: node.CreationTimestamp,
		}
	} else {
		// If ready condition is not nil, make a copy of it, since we may modify it in place later.
		lastReadyCondition = *readyCondition
		gracePeriod = nc.nodeMonitorGracePeriod
	}

	savedNodeStatus, found := nc.nodeStatusMap[node.Name]
	// There are following cases to check:
	// - both saved and new status have no Ready Condition set - we leave everything as it is,
	// - saved status have no Ready Condition, but current one does - NodeController was restarted with Node data already present in etcd,
	// - saved status have some Ready Condition, but current one does not - it's an error, but we fill it up because that's probably a good thing to do,
	// - both saved and current statuses have Ready Conditions and they have the same LastProbeTime - nothing happened on that Node, it may be
	//   unresponsive, so we leave it as it is,
	// - both saved and current statuses have Ready Conditions, they have different LastProbeTimes, but the same Ready Condition State -
	//   everything's in order, no transition occurred, we update only probeTimestamp,
	// - both saved and current statuses have Ready Conditions, different LastProbeTimes and different Ready Condition State -
	//   Ready Condition changed it state since we last seen it, so we update both probeTimestamp and readyTransitionTimestamp.
	// TODO: things to consider:
	//   - if 'LastProbeTime' have gone back in time its probably and error, currently we ignore it,
	//   - currently only correct Ready State transition outside of Node Controller is marking it ready by Kubelet, we don't check
	//     if that's the case, but it does not seem necessary.
	savedCondition := nc.getCondition(&savedNodeStatus.status, api.NodeReady)
	observedCondition := nc.getCondition(&node.Status, api.NodeReady)
	if !found {
		glog.Warningf("Missing timestamp for Node %s. Assuming now as a timestamp.", node.Name)
		savedNodeStatus = nodeStatusData{
			status:                   node.Status,
			probeTimestamp:           nc.now(),
			readyTransitionTimestamp: nc.now(),
		}
		nc.nodeStatusMap[node.Name] = savedNodeStatus
	} else if savedCondition == nil && observedCondition != nil {
		glog.V(1).Infof("Creating timestamp entry for newly observed Node %s", node.Name)
		savedNodeStatus = nodeStatusData{
			status:                   node.Status,
			probeTimestamp:           nc.now(),
			readyTransitionTimestamp: nc.now(),
		}
		nc.nodeStatusMap[node.Name] = savedNodeStatus
	} else if savedCondition != nil && observedCondition == nil {
		glog.Errorf("ReadyCondition was removed from Status of Node %s", node.Name)
		// TODO: figure out what to do in this case. For now we do the same thing as above.
		savedNodeStatus = nodeStatusData{
			status:                   node.Status,
			probeTimestamp:           nc.now(),
			readyTransitionTimestamp: nc.now(),
		}
		nc.nodeStatusMap[node.Name] = savedNodeStatus
	} else if savedCondition != nil && observedCondition != nil && savedCondition.LastHeartbeatTime != observedCondition.LastHeartbeatTime {
		var transitionTime util.Time
		// If ReadyCondition changed since the last time we checked, we update the transition timestamp to "now",
		// otherwise we leave it as it is.
		if savedCondition.LastTransitionTime != observedCondition.LastTransitionTime {
			glog.V(3).Infof("ReadyCondition for Node %s transitioned from %v to %v", node.Name, savedCondition.Status, observedCondition)

			transitionTime = nc.now()
		} else {
			transitionTime = savedNodeStatus.readyTransitionTimestamp
		}
		glog.V(3).Infof("Nodes ReadyCondition updated. Updating timestamp: %+v\n vs %+v.", savedNodeStatus.status, node.Status)
		savedNodeStatus = nodeStatusData{
			status:                   node.Status,
			probeTimestamp:           nc.now(),
			readyTransitionTimestamp: transitionTime,
		}
		nc.nodeStatusMap[node.Name] = savedNodeStatus
	}

	if nc.now().After(savedNodeStatus.probeTimestamp.Add(gracePeriod)) {
		// NodeReady condition was last set longer ago than gracePeriod, so update it to Unknown
		// (regardless of its current value) in the master, without contacting kubelet.
		if readyCondition == nil {
			glog.V(2).Infof("node %v is never updated by kubelet", node.Name)
			node.Status.Conditions = append(node.Status.Conditions, api.NodeCondition{
				Type:               api.NodeReady,
				Status:             api.ConditionUnknown,
				Reason:             fmt.Sprintf("Kubelet never posted node status."),
				LastHeartbeatTime:  node.CreationTimestamp,
				LastTransitionTime: nc.now(),
			})
		} else {
			glog.V(2).Infof("node %v hasn't been updated for %+v. Last ready condition is: %+v",
				node.Name, nc.now().Time.Sub(savedNodeStatus.probeTimestamp.Time), lastReadyCondition)
			if lastReadyCondition.Status != api.ConditionUnknown {
				readyCondition.Status = api.ConditionUnknown
				readyCondition.Reason = fmt.Sprintf("Kubelet stopped posting node status.")
				// LastProbeTime is the last time we heard from kubelet.
				readyCondition.LastHeartbeatTime = lastReadyCondition.LastHeartbeatTime
				readyCondition.LastTransitionTime = nc.now()
			}
		}
		if !api.Semantic.DeepEqual(nc.getCondition(&node.Status, api.NodeReady), lastReadyCondition) {
			if _, err = nc.kubeClient.Nodes().UpdateStatus(node); err != nil {
				glog.Errorf("Error updating node %s: %v", node.Name, err)
				return gracePeriod, lastReadyCondition, readyCondition, err
			} else {
				nc.nodeStatusMap[node.Name] = nodeStatusData{
					status:                   node.Status,
					probeTimestamp:           nc.nodeStatusMap[node.Name].probeTimestamp,
					readyTransitionTimestamp: nc.now(),
				}
				return gracePeriod, lastReadyCondition, readyCondition, nil
			}
		}
	}

	return gracePeriod, lastReadyCondition, readyCondition, err
}

// monitorNodeStatus verifies node status are constantly updated by kubelet, and if not,
// post "NodeReady==ConditionUnknown". It also evicts all pods if node is not ready or
// not reachable for a long period of time.
func (nc *NodeController) monitorNodeStatus() error {
	nodes, err := nc.kubeClient.Nodes().List(labels.Everything(), fields.Everything())
	if err != nil {
		return err
	}
	for i := range nodes.Items {
		var gracePeriod time.Duration
		var lastReadyCondition api.NodeCondition
		var readyCondition *api.NodeCondition
		node := &nodes.Items[i]
		for rep := 0; rep < nodeStatusUpdateRetry; rep++ {
			gracePeriod, lastReadyCondition, readyCondition, err = nc.tryUpdateNodeStatus(node)
			if err == nil {
				break
			}
			name := node.Name
			node, err = nc.kubeClient.Nodes().Get(name)
			if err != nil {
				glog.Errorf("Failed while getting a Node to retry updating NodeStatus. Probably Node %s was deleted.", name)
				break
			}
		}
		if err != nil {
			glog.Errorf("Update status  of Node %v from NodeController exceeds retry count."+
				"Skipping - no pods will be evicted.", node.Name)
			continue
		}

		if readyCondition != nil {
			// Check eviction timeout.
			if lastReadyCondition.Status == api.ConditionFalse &&
				nc.now().After(nc.nodeStatusMap[node.Name].readyTransitionTimestamp.Add(nc.podEvictionTimeout)) {
				// Node stays in not ready for at least 'podEvictionTimeout' - evict all pods on the unhealthy node.
				// Makes sure we are not removing pods from to many nodes in the same time.
				glog.Infof("Evicting pods: %v is later than %v + %v", nc.now(), nc.nodeStatusMap[node.Name].readyTransitionTimestamp, nc.podEvictionTimeout)
				if nc.deletingPodsRateLimiter.CanAccept() {
					nc.deletePods(node.Name)
				}
			}
			if lastReadyCondition.Status == api.ConditionUnknown &&
				nc.now().After(nc.nodeStatusMap[node.Name].probeTimestamp.Add(nc.podEvictionTimeout-gracePeriod)) {
				// Same as above. Note however, since condition unknown is posted by node controller, which means we
				// need to substract monitoring grace period in order to get the real 'podEvictionTimeout'.
				glog.Infof("Evicting pods2: %v is later than %v + %v", nc.now(), nc.nodeStatusMap[node.Name].readyTransitionTimestamp, nc.podEvictionTimeout-gracePeriod)
				if nc.deletingPodsRateLimiter.CanAccept() {
					nc.deletePods(node.Name)
				}
			}

			// Report node events.
			if readyCondition.Status == api.ConditionTrue && lastReadyCondition.Status != api.ConditionTrue {
				nc.recordNodeEvent(node, "ready")
			}
			if readyCondition.Status == api.ConditionFalse && lastReadyCondition.Status != api.ConditionFalse {
				nc.recordNodeEvent(node, "not_ready")
			}
			if readyCondition.Status == api.ConditionUnknown && lastReadyCondition.Status != api.ConditionUnknown {
				nc.recordNodeEvent(node, "unknown")
			}
		}
	}
	return nil
}

// getStaticNodesWithSpec constructs and returns api.NodeList for static nodes. If error
// occurs, an empty NodeList will be returned with a non-nil error info. The method only
// constructs spec fields for nodes.
func (nc *NodeController) getStaticNodesWithSpec() (*api.NodeList, error) {
	result := &api.NodeList{}
	for _, nodeID := range nc.nodes {
		node := api.Node{
			ObjectMeta: api.ObjectMeta{Name: nodeID},
			Spec: api.NodeSpec{
				ExternalID: nodeID,
			},
			Status: api.NodeStatus{
				Capacity: nc.staticResources.Capacity,
			},
		}
		result.Items = append(result.Items, node)
	}
	return result, nil
}

// getCloudNodesWithSpec constructs and returns api.NodeList from cloudprovider. If error
// occurs, an empty NodeList will be returned with a non-nil error info. The method only
// constructs spec fields for nodes.
func (nc *NodeController) getCloudNodesWithSpec() (*api.NodeList, error) {
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
			node.Status.Capacity = resources.Capacity
		}
		instanceID, err := instances.ExternalID(node.Name)
		if err != nil {
			glog.Errorf("Error getting instance id for %s: %v", node.Name, err)
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
	pods, err := nc.kubeClient.Pods(api.NamespaceAll).List(labels.Everything(),
		fields.OneTermEqualSelector(client.PodHost, nodeID))
	if err != nil {
		return err
	}
	for _, pod := range pods.Items {
		// Defensive check, also needed for tests.
		if pod.Spec.Host != nodeID {
			continue
		}
		glog.V(2).Infof("Delete pod %v", pod.Name)
		if err := nc.kubeClient.Pods(pod.Namespace).Delete(pod.Name, nil); err != nil {
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
func (nc *NodeController) getCondition(status *api.NodeStatus, conditionType api.NodeConditionType) *api.NodeCondition {
	if status == nil {
		return nil
	}
	for i := range status.Conditions {
		if status.Conditions[i].Type == conditionType {
			return &status.Conditions[i]
		}
	}
	return nil
}
