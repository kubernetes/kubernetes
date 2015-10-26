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

package node

import (
	"errors"
	"fmt"
	"net"
	"strings"
	"sync"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/record"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/watch"
)

var (
	ErrCloudInstance = errors.New("cloud provider doesn't support instances.")
)

const (
	// nodeStatusUpdateRetry controls the number of retries of writing NodeStatus update.
	nodeStatusUpdateRetry = 5
	// controls how often NodeController will try to evict Pods from non-responsive Nodes.
	nodeEvictionPeriod = 100 * time.Millisecond
)

type nodeStatusData struct {
	probeTimestamp           unversioned.Time
	readyTransitionTimestamp unversioned.Time
	status                   api.NodeStatus
}

type NodeController struct {
	allocateNodeCIDRs       bool
	cloud                   cloudprovider.Interface
	clusterCIDR             *net.IPNet
	deletingPodsRateLimiter util.RateLimiter
	knownNodeSet            sets.String
	kubeClient              client.Interface
	// Method for easy mocking in unittest.
	lookupIP func(host string) ([]net.IP, error)
	// Value used if sync_nodes_status=False. NodeController will not proactively
	// sync node status in this case, but will monitor node status updated from kubelet. If
	// it doesn't receive update for this amount of time, it will start posting "NodeReady==
	// ConditionUnknown". The amount of time before which NodeController start evicting pods
	// is controlled via flag 'pod-eviction-timeout'.
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
	// Value controlling NodeController monitoring period, i.e. how often does NodeController
	// check node status posted from kubelet. This value should be lower than nodeMonitorGracePeriod.
	// TODO: Change node status monitor to watch based.
	nodeMonitorPeriod time.Duration
	// Value used if sync_nodes_status=False, only for node startup. When node
	// is just created, e.g. cluster bootstrap or node creation, we give a longer grace period.
	nodeStartupGracePeriod time.Duration
	// per Node map storing last observed Status together with a local time when it was observed.
	// This timestamp is to be used instead of LastProbeTime stored in Condition. We do this
	// to aviod the problem with time skew across the cluster.
	nodeStatusMap map[string]nodeStatusData
	now           func() unversioned.Time
	// Lock to access evictor workers
	evictorLock *sync.Mutex
	// workers that evicts pods from unresponsive nodes.
	podEvictor         *RateLimitedTimedQueue
	terminationEvictor *RateLimitedTimedQueue
	podEvictionTimeout time.Duration
	// The maximum duration before a pod evicted from a node can be forcefully terminated.
	maximumGracePeriod time.Duration
	recorder           record.EventRecorder
	// Pod framework and store
	podController *framework.Controller
	podStore      cache.StoreToPodLister
	// Node framework and store
	nodeController *framework.Controller
	nodeStore      cache.StoreToNodeLister

	forcefullyDeletePod func(*api.Pod)
}

// NewNodeController returns a new node controller to sync instances from cloudprovider.
func NewNodeController(
	cloud cloudprovider.Interface,
	kubeClient client.Interface,
	podEvictionTimeout time.Duration,
	deletionEvictionLimiter util.RateLimiter,
	terminationEvictionLimiter util.RateLimiter,
	nodeMonitorGracePeriod time.Duration,
	nodeStartupGracePeriod time.Duration,
	nodeMonitorPeriod time.Duration,
	clusterCIDR *net.IPNet,
	allocateNodeCIDRs bool) *NodeController {
	eventBroadcaster := record.NewBroadcaster()
	recorder := eventBroadcaster.NewRecorder(api.EventSource{Component: "controllermanager"})
	eventBroadcaster.StartLogging(glog.Infof)
	if kubeClient != nil {
		glog.Infof("Sending events to api server.")
		eventBroadcaster.StartRecordingToSink(kubeClient.Events(""))
	} else {
		glog.Infof("No api server defined - no events will be sent to API server.")
	}
	if allocateNodeCIDRs && clusterCIDR == nil {
		glog.Fatal("NodeController: Must specify clusterCIDR if allocateNodeCIDRs == true.")
	}
	evictorLock := sync.Mutex{}

	nc := &NodeController{
		cloud:                  cloud,
		knownNodeSet:           make(sets.String),
		kubeClient:             kubeClient,
		recorder:               recorder,
		podEvictionTimeout:     podEvictionTimeout,
		maximumGracePeriod:     5 * time.Minute,
		evictorLock:            &evictorLock,
		podEvictor:             NewRateLimitedTimedQueue(deletionEvictionLimiter),
		terminationEvictor:     NewRateLimitedTimedQueue(terminationEvictionLimiter),
		nodeStatusMap:          make(map[string]nodeStatusData),
		nodeMonitorGracePeriod: nodeMonitorGracePeriod,
		nodeMonitorPeriod:      nodeMonitorPeriod,
		nodeStartupGracePeriod: nodeStartupGracePeriod,
		lookupIP:               net.LookupIP,
		now:                    unversioned.Now,
		clusterCIDR:            clusterCIDR,
		allocateNodeCIDRs:      allocateNodeCIDRs,
		forcefullyDeletePod:    func(p *api.Pod) { forcefullyDeletePod(kubeClient, p) },
	}

	nc.podStore.Store, nc.podController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return nc.kubeClient.Pods(api.NamespaceAll).List(labels.Everything(), fields.Everything())
			},
			WatchFunc: func(rv string) (watch.Interface, error) {
				return nc.kubeClient.Pods(api.NamespaceAll).Watch(labels.Everything(), fields.Everything(), rv)
			},
		},
		&api.Pod{},
		controller.NoResyncPeriodFunc(),
		framework.ResourceEventHandlerFuncs{
			AddFunc:    nc.maybeDeleteTerminatingPod,
			UpdateFunc: func(_, obj interface{}) { nc.maybeDeleteTerminatingPod(obj) },
		},
	)
	nc.nodeStore.Store, nc.nodeController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return nc.kubeClient.Nodes().List(labels.Everything(), fields.Everything())
			},
			WatchFunc: func(rv string) (watch.Interface, error) {
				return nc.kubeClient.Nodes().Watch(labels.Everything(), fields.Everything(), rv)
			},
		},
		&api.Node{},
		controller.NoResyncPeriodFunc(),
		framework.ResourceEventHandlerFuncs{},
	)
	return nc
}

// Run starts an asynchronous loop that monitors the status of cluster nodes.
func (nc *NodeController) Run(period time.Duration) {
	go nc.nodeController.Run(util.NeverStop)
	go nc.podController.Run(util.NeverStop)
	// Incorporate the results of node status pushed from kubelet to master.
	go util.Until(func() {
		if err := nc.monitorNodeStatus(); err != nil {
			glog.Errorf("Error monitoring node status: %v", err)
		}
	}, nc.nodeMonitorPeriod, util.NeverStop)

	// Managing eviction of nodes:
	// 1. when we delete pods off a node, if the node was not empty at the time we then
	//    queue a termination watcher
	//    a. If we hit an error, retry deletion
	// 2. The terminator loop ensures that pods are eventually cleaned and we never
	//    terminate a pod in a time period less than nc.maximumGracePeriod. AddedAt
	//    is the time from which we measure "has this pod been terminating too long",
	//    after which we will delete the pod with grace period 0 (force delete).
	//    a. If we hit errors, retry instantly
	//    b. If there are no pods left terminating, exit
	//    c. If there are pods still terminating, wait for their estimated completion
	//       before retrying
	go util.Until(func() {
		nc.evictorLock.Lock()
		defer nc.evictorLock.Unlock()
		nc.podEvictor.Try(func(value TimedValue) (bool, time.Duration) {
			remaining, err := nc.deletePods(value.Value)
			if err != nil {
				util.HandleError(fmt.Errorf("unable to evict node %q: %v", value.Value, err))
				return false, 0
			}

			if remaining {
				nc.terminationEvictor.Add(value.Value)
			}
			return true, 0
		})
	}, nodeEvictionPeriod, util.NeverStop)

	// TODO: replace with a controller that ensures pods that are terminating complete
	// in a particular time period
	go util.Until(func() {
		nc.evictorLock.Lock()
		defer nc.evictorLock.Unlock()
		nc.terminationEvictor.Try(func(value TimedValue) (bool, time.Duration) {
			completed, remaining, err := nc.terminatePods(value.Value, value.AddedAt)
			if err != nil {
				util.HandleError(fmt.Errorf("unable to terminate pods on node %q: %v", value.Value, err))
				return false, 0
			}

			if completed {
				glog.Infof("All pods terminated on %s", value.Value)
				nc.recordNodeEvent(value.Value, "TerminatedAllPods", fmt.Sprintf("Terminated all Pods on Node %s.", value.Value))
				return true, 0
			}

			glog.V(2).Infof("Pods terminating since %s on %q, estimated completion %s", value.AddedAt, value.Value, remaining)
			// clamp very short intervals
			if remaining < nodeEvictionPeriod {
				remaining = nodeEvictionPeriod
			}
			return false, remaining
		})
	}, nodeEvictionPeriod, util.NeverStop)
}

// Generates num pod CIDRs that could be assigned to nodes.
func generateCIDRs(clusterCIDR *net.IPNet, num int) sets.String {
	res := sets.NewString()
	cidrIP := clusterCIDR.IP.To4()
	for i := 0; i < num; i++ {
		// TODO: Make the CIDRs configurable.
		b1 := byte(i >> 8)
		b2 := byte(i % 256)
		res.Insert(fmt.Sprintf("%d.%d.%d.0/24", cidrIP[0], cidrIP[1]+b1, cidrIP[2]+b2))
	}
	return res
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

// maybeDeleteTerminatingPod non-gracefully deletes pods that are terminating
// that should not be gracefully terminated.
func (nc *NodeController) maybeDeleteTerminatingPod(obj interface{}) {
	pod, ok := obj.(*api.Pod)
	if !ok {
		return
	}

	// consider only terminating pods
	if pod.DeletionTimestamp == nil {
		return
	}

	// delete terminating pods that have not yet been scheduled
	if len(pod.Spec.NodeName) == 0 {
		nc.forcefullyDeletePod(pod)
		return
	}

	nodeObj, found, err := nc.nodeStore.GetByKey(pod.Spec.NodeName)
	if err != nil {
		// this can only happen if the Store.KeyFunc has a problem creating
		// a key for the pod. If it happens once, it will happen again so
		// don't bother requeuing the pod.
		util.HandleError(err)
		return
	}

	// delete terminating pods that have been scheduled on
	// nonexistant nodes
	if !found {
		nc.forcefullyDeletePod(pod)
		return
	}

	// delete terminating pods that have been scheduled on
	// nodes that do not support graceful termination
	// TODO(mikedanese): this can be removed when we no longer
	// guarantee backwards compatibility of master API to kubelets with
	// versions less than 1.1.0
	node := nodeObj.(*api.Node)
	if strings.HasPrefix(node.Status.NodeInfo.KubeletVersion, "v1.0") {
		nc.forcefullyDeletePod(pod)
		return
	}
}

func forcefullyDeletePod(c client.Interface, pod *api.Pod) {
	var zero int64
	err := c.Pods(pod.Namespace).Delete(pod.Name, &api.DeleteOptions{GracePeriodSeconds: &zero})
	if err != nil {
		util.HandleError(err)
	}
}

// monitorNodeStatus verifies node status are constantly updated by kubelet, and if not,
// post "NodeReady==ConditionUnknown". It also evicts all pods if node is not ready or
// not reachable for a long period of time.
func (nc *NodeController) monitorNodeStatus() error {
	nodes, err := nc.kubeClient.Nodes().List(labels.Everything(), fields.Everything())
	if err != nil {
		return err
	}
	for _, node := range nodes.Items {
		if !nc.knownNodeSet.Has(node.Name) {
			glog.V(1).Infof("NodeController observed a new Node: %#v", node)
			nc.recordNodeEvent(node.Name, "RegisteredNode", fmt.Sprintf("Registered Node %v in NodeController", node.Name))
			nc.cancelPodEviction(node.Name)
			nc.knownNodeSet.Insert(node.Name)
		}
	}
	// If there's a difference between lengths of known Nodes and observed nodes
	// we must have removed some Node.
	if len(nc.knownNodeSet) != len(nodes.Items) {
		observedSet := make(sets.String)
		for _, node := range nodes.Items {
			observedSet.Insert(node.Name)
		}
		deleted := nc.knownNodeSet.Difference(observedSet)
		for nodeName := range deleted {
			glog.V(1).Infof("NodeController observed a Node deletion: %v", nodeName)
			nc.recordNodeEvent(nodeName, "RemovingNode", fmt.Sprintf("Removing Node %v from NodeController", nodeName))
			nc.evictPods(nodeName)
			nc.knownNodeSet.Delete(nodeName)
		}
	}

	if nc.allocateNodeCIDRs {
		// TODO (cjcullen): Use pkg/controller/framework to watch nodes and
		// reduce lists/decouple this from monitoring status.
		nc.reconcileNodeCIDRs(nodes)
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

		decisionTimestamp := nc.now()

		if readyCondition != nil {
			// Check eviction timeout against decisionTimestamp
			if lastReadyCondition.Status == api.ConditionFalse &&
				decisionTimestamp.After(nc.nodeStatusMap[node.Name].readyTransitionTimestamp.Add(nc.podEvictionTimeout)) {
				if nc.evictPods(node.Name) {
					glog.Infof("Evicting pods on node %s: %v is later than %v + %v", node.Name, decisionTimestamp, nc.nodeStatusMap[node.Name].readyTransitionTimestamp, nc.podEvictionTimeout)
				}
			}
			if lastReadyCondition.Status == api.ConditionUnknown &&
				decisionTimestamp.After(nc.nodeStatusMap[node.Name].probeTimestamp.Add(nc.podEvictionTimeout-gracePeriod)) {
				if nc.evictPods(node.Name) {
					glog.Infof("Evicting pods on node %s: %v is later than %v + %v", node.Name, decisionTimestamp, nc.nodeStatusMap[node.Name].readyTransitionTimestamp, nc.podEvictionTimeout-gracePeriod)
				}
			}
			if lastReadyCondition.Status == api.ConditionTrue {
				if nc.cancelPodEviction(node.Name) {
					glog.Infof("Node %s is ready again, cancelled pod eviction", node.Name)
				}
			}

			// Report node event.
			if readyCondition.Status != api.ConditionTrue && lastReadyCondition.Status == api.ConditionTrue {
				nc.recordNodeStatusChange(node, "NodeNotReady")
			}

			// Check with the cloud provider to see if the node still exists. If it
			// doesn't, delete the node and all pods scheduled on the node.
			if readyCondition.Status != api.ConditionTrue && nc.cloud != nil {
				instances, ok := nc.cloud.Instances()
				if !ok {
					glog.Errorf("%v", ErrCloudInstance)
					continue
				}
				if _, err := instances.ExternalID(node.Name); err != nil && err == cloudprovider.InstanceNotFound {
					glog.Infof("Deleting node (no longer present in cloud provider): %s", node.Name)
					nc.recordNodeEvent(node.Name, "DeletingNode", fmt.Sprintf("Deleting Node %v because it's not present according to cloud provider", node.Name))

					remaining, err := nc.hasPods(node.Name)
					if err != nil {
						glog.Errorf("Unable to determine whether node %s has pods, will retry: %v", node.Name, err)
						continue
					}
					if remaining {
						// queue eviction of the pods on the node
						glog.V(2).Infof("Deleting node %s is delayed while pods are evicted", node.Name)
						nc.evictPods(node.Name)
						continue
					}

					if err := nc.kubeClient.Nodes().Delete(node.Name); err != nil {
						glog.Errorf("Unable to delete node %s: %v", node.Name, err)
						continue
					}
				}
			}
		}
	}
	return nil
}

// reconcileNodeCIDRs looks at each node and assigns it a valid CIDR
// if it doesn't currently have one.
func (nc *NodeController) reconcileNodeCIDRs(nodes *api.NodeList) {
	glog.V(4).Infof("Reconciling cidrs for %d nodes", len(nodes.Items))
	// TODO(roberthbailey): This seems inefficient. Why re-calculate CIDRs
	// on each sync period?
	availableCIDRs := generateCIDRs(nc.clusterCIDR, len(nodes.Items))
	for _, node := range nodes.Items {
		if node.Spec.PodCIDR != "" {
			glog.V(4).Infof("CIDR %s is already being used by node %s", node.Spec.PodCIDR, node.Name)
			availableCIDRs.Delete(node.Spec.PodCIDR)
		}
	}
	for _, node := range nodes.Items {
		if node.Spec.PodCIDR == "" {
			podCIDR, found := availableCIDRs.PopAny()
			if !found {
				nc.recordNodeStatusChange(&node, "CIDRNotAvailable")
				continue
			}
			glog.V(4).Infof("Assigning node %s CIDR %s", node.Name, podCIDR)
			node.Spec.PodCIDR = podCIDR
			if _, err := nc.kubeClient.Nodes().Update(&node); err != nil {
				nc.recordNodeStatusChange(&node, "CIDRAssignmentFailed")
			}
		}
	}
}

func (nc *NodeController) recordNodeEvent(nodeName string, reason string, event string) {
	ref := &api.ObjectReference{
		Kind:      "Node",
		Name:      nodeName,
		UID:       types.UID(nodeName),
		Namespace: "",
	}
	glog.V(2).Infof("Recording %s event message for node %s", event, nodeName)
	nc.recorder.Eventf(ref, reason, "Node %s event: %s", nodeName, event)
}

func (nc *NodeController) recordNodeStatusChange(node *api.Node, new_status string) {
	ref := &api.ObjectReference{
		Kind:      "Node",
		Name:      node.Name,
		UID:       types.UID(node.Name),
		Namespace: "",
	}
	glog.V(2).Infof("Recording status change %s event message for node %s", new_status, node.Name)
	// TODO: This requires a transaction, either both node status is updated
	// and event is recorded or neither should happen, see issue #6055.
	nc.recorder.Eventf(ref, new_status, "Node %s status is now: %s", node.Name, new_status)
}

// For a given node checks its conditions and tries to update it. Returns grace period to which given node
// is entitled, state of current and last observed Ready Condition, and an error if it occurred.
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
	//   - if 'LastProbeTime' have gone back in time its probably an error, currently we ignore it,
	//   - currently only correct Ready State transition outside of Node Controller is marking it ready by Kubelet, we don't check
	//     if that's the case, but it does not seem necessary.
	var savedCondition *api.NodeCondition
	if found {
		savedCondition = nc.getCondition(&savedNodeStatus.status, api.NodeReady)
	}
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
		var transitionTime unversioned.Time
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
				Reason:             "NodeStatusNeverUpdated",
				Message:            fmt.Sprintf("Kubelet never posted node status."),
				LastHeartbeatTime:  node.CreationTimestamp,
				LastTransitionTime: nc.now(),
			})
		} else {
			glog.V(2).Infof("node %v hasn't been updated for %+v. Last ready condition is: %+v",
				node.Name, nc.now().Time.Sub(savedNodeStatus.probeTimestamp.Time), lastReadyCondition)
			if lastReadyCondition.Status != api.ConditionUnknown {
				readyCondition.Status = api.ConditionUnknown
				readyCondition.Reason = "NodeStatusUnknown"
				readyCondition.Message = fmt.Sprintf("Kubelet stopped posting node status.")
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

// returns true if the provided node still has pods scheduled to it, or an error if
// the server could not be contacted.
func (nc *NodeController) hasPods(nodeName string) (bool, error) {
	pods, err := nc.kubeClient.Pods(api.NamespaceAll).List(labels.Everything(), fields.OneTermEqualSelector(client.PodHost, nodeName))
	if err != nil {
		return false, err
	}
	return len(pods.Items) > 0, nil
}

// evictPods queues an eviction for the provided node name, and returns false if the node is already
// queued for eviction.
func (nc *NodeController) evictPods(nodeName string) bool {
	nc.evictorLock.Lock()
	defer nc.evictorLock.Unlock()
	return nc.podEvictor.Add(nodeName)
}

// cancelPodEviction removes any queued evictions, typically because the node is available again. It
// returns true if an eviction was queued.
func (nc *NodeController) cancelPodEviction(nodeName string) bool {
	nc.evictorLock.Lock()
	defer nc.evictorLock.Unlock()
	wasDeleting := nc.podEvictor.Remove(nodeName)
	wasTerminating := nc.terminationEvictor.Remove(nodeName)
	if wasDeleting || wasTerminating {
		glog.V(2).Infof("Cancelling pod Eviction on Node: %v", nodeName)
		return true
	}
	return false
}

// deletePods will delete all pods from master running on given node, and return true
// if any pods were deleted.
func (nc *NodeController) deletePods(nodeName string) (bool, error) {
	remaining := false
	pods, err := nc.kubeClient.Pods(api.NamespaceAll).List(labels.Everything(), fields.OneTermEqualSelector(client.PodHost, nodeName))
	if err != nil {
		return remaining, err
	}

	if len(pods.Items) > 0 {
		nc.recordNodeEvent(nodeName, "DeletingAllPods", fmt.Sprintf("Deleting all Pods from Node %v.", nodeName))
	}

	for _, pod := range pods.Items {
		// Defensive check, also needed for tests.
		if pod.Spec.NodeName != nodeName {
			continue
		}
		// if the pod has already been deleted, ignore it
		if pod.DeletionGracePeriodSeconds != nil {
			continue
		}

		glog.V(2).Infof("Starting deletion of pod %v", pod.Name)
		nc.recorder.Eventf(&pod, "NodeControllerEviction", "Marking for deletion Pod %s from Node %s", pod.Name, nodeName)
		if err := nc.kubeClient.Pods(pod.Namespace).Delete(pod.Name, nil); err != nil {
			return false, err
		}
		remaining = true
	}
	return remaining, nil
}

// terminatePods will ensure all pods on the given node that are in terminating state are eventually
// cleaned up. Returns true if the node has no pods in terminating state, a duration that indicates how
// long before we should check again (the next deadline for a pod to complete), or an error.
func (nc *NodeController) terminatePods(nodeName string, since time.Time) (bool, time.Duration, error) {
	// the time before we should try again
	nextAttempt := time.Duration(0)
	// have we deleted all pods
	complete := true

	pods, err := nc.kubeClient.Pods(api.NamespaceAll).List(labels.Everything(),
		fields.OneTermEqualSelector(client.PodHost, nodeName))
	if err != nil {
		return false, nextAttempt, err
	}

	now := time.Now()
	elapsed := now.Sub(since)
	for _, pod := range pods.Items {
		// Defensive check, also needed for tests.
		if pod.Spec.NodeName != nodeName {
			continue
		}
		// only clean terminated pods
		if pod.DeletionGracePeriodSeconds == nil {
			continue
		}

		// the user's requested grace period
		grace := time.Duration(*pod.DeletionGracePeriodSeconds) * time.Second
		if grace > nc.maximumGracePeriod {
			grace = nc.maximumGracePeriod
		}

		// the time remaining before the pod should have been deleted
		remaining := grace - elapsed
		if remaining < 0 {
			remaining = 0
			glog.V(2).Infof("Removing pod %v after %s grace period", pod.Name, grace)
			nc.recordNodeEvent(nodeName, "TerminatingEvictedPod", fmt.Sprintf("Pod %s has exceeded the grace period for deletion after being evicted from Node %q and is being force killed", pod.Name, nodeName))
			if err := nc.kubeClient.Pods(pod.Namespace).Delete(pod.Name, api.NewDeleteOptions(0)); err != nil {
				glog.Errorf("Error completing deletion of pod %s: %v", pod.Name, err)
				complete = false
			}
		} else {
			glog.V(2).Infof("Pod %v still terminating, requested grace period %s, %s remaining", pod.Name, grace, remaining)
			complete = false
		}

		if nextAttempt < remaining {
			nextAttempt = remaining
		}
	}
	return complete, nextAttempt, nil
}
