/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package kubelet

import (
	"errors"
	"fmt"
	"net"
	"strings"
	"sync"
	"time"

	"github.com/golang/glog"
	cadvisorApi "github.com/google/cadvisor/info/v1"
	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/record"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
	"k8s.io/kubernetes/pkg/version"
	"k8s.io/kubernetes/pkg/watch"
)

const (
	// nodeStatusUpdateRetry specifies how many times kubelet retries when posting node status failed.
	nodeStatusUpdateRetry = 5
)

type infoGetter interface {
	GetMachineInfo() (*cadvisorApi.MachineInfo, error)
	ContainerRuntimeUp() bool
	NetworkConfigured() bool
	GetVersionInfo() (*cadvisorApi.VersionInfo, error)
}

type nodeManager interface {
	Start()
	GetNode() (*api.Node, error)
	GetPodCIDR() string
	GetHostIP() (net.IP, error)
}

type realNodeManager struct {
	// apiserver client.
	client client.Interface

	nodeLister nodeLister

	// Set to true to have the node register itself with the apiserver.
	registerNode bool

	// nodeStatusUpdateFrequency specifies how often kubelet posts node status to master.
	// Note: be cautious when changing the constant, it must work with nodeMonitorGracePeriod
	// in nodecontroller. There are several constraints:
	// 1. nodeMonitorGracePeriod must be N times more than nodeStatusUpdateFrequency, where
	//    N means number of retries allowed for kubelet to post node status. It is pointless
	//    to make nodeMonitorGracePeriod be less than nodeStatusUpdateFrequency, since there
	//    will only be fresh values from Kubelet at an interval of nodeStatusUpdateFrequency.
	//    The constant must be less than podEvictionTimeout.
	// 2. nodeStatusUpdateFrequency needs to be large enough for kubelet to generate node
	//    status. Kubelet may fail to update node status reliably if the value is too small,
	//    as it takes time to gather all necessary node information.
	nodeStatusUpdateFrequency time.Duration

	// Cloud provider interface
	cloud cloudprovider.Interface

	nodeName string
	hostname string

	// Number of Pods which can be run by this Kubelet.
	pods int

	// The EventRecorder to use
	recorder record.EventRecorder

	// Information about the ports which are opened by daemons on Node running this Kubelet server.
	daemonEndpoints *api.NodeDaemonEndpoints

	// Interface to get machine and version info.
	infoGetter infoGetter

	// Reference to this node.
	nodeRef *api.ObjectReference

	// podCIDR may be updated by node.Spec.
	podCIDR string

	// for internal book keeping; access only from within registerWithApiserver
	registrationCompleted bool

	lock sync.RWMutex
}

func newRealNodeManager(client client.Interface, cloud cloudprovider.Interface, registerNode bool,
	nodeStatusUpdateFrequency time.Duration, recorder record.EventRecorder, nodeName, hostname, podCIDR string,
	pods int, infoGetter infoGetter, daemonEndpoints *api.NodeDaemonEndpoints, nodeRef *api.ObjectReference) *realNodeManager {
	return &realNodeManager{
		client:                    client,
		cloud:                     cloud,
		registerNode:              registerNode,
		nodeStatusUpdateFrequency: nodeStatusUpdateFrequency,
		recorder:                  recorder,
		nodeName:                  nodeName,
		hostname:                  hostname,
		podCIDR:                   podCIDR,
		pods:                      pods,
		infoGetter:                infoGetter,
		daemonEndpoints:           daemonEndpoints,
		nodeRef:                   nodeRef,
	}
}

type nodeLister interface {
	List() (machines api.NodeList, err error)
	GetNodeInfo(id string) (*api.Node, error)
}

func (nm *realNodeManager) Start() {
	if nm.client == nil {
		return
	}
	nm.setNodeLister()
	go util.Until(nm.syncNodeStatus, nm.nodeStatusUpdateFrequency, util.NeverStop)
}

func (nm *realNodeManager) GetPodCIDR() string {
	nm.lock.RLock()
	defer nm.lock.RUnlock()
	return nm.podCIDR
}

func (nm *realNodeManager) GetNode() (*api.Node, error) {
	if nm.client == nil {
		return nil, errors.New("unable to get node entry because apiserver client is nil")
	}
	return nm.nodeLister.GetNodeInfo(nm.nodeName)
}

// Returns host IP or nil in case of error.
func (nm *realNodeManager) GetHostIP() (net.IP, error) {
	if nm.client == nil {
		return nil, errors.New("unable to get node entry because apiserver client is nil")
	}
	node, err := nm.GetNode()
	if err != nil {
		return nil, fmt.Errorf("cannot get node: %v", err)
	}
	return nodeutil.GetNodeHostIP(node)
}

func (nm *realNodeManager) setNodeLister() {
	nodeStore := cache.NewStore(cache.MetaNamespaceKeyFunc)
	// TODO: cache.NewListWatchFromClient is limited as it takes a client implementation rather
	// than an interface. There is no way to construct a list+watcher using resource name.
	fieldSelector := fields.Set{client.ObjectNameField: nm.nodeName}.AsSelector()
	listWatch := &cache.ListWatch{
		ListFunc: func() (runtime.Object, error) {
			return nm.client.Nodes().List(labels.Everything(), fieldSelector)
		},
		WatchFunc: func(resourceVersion string) (watch.Interface, error) {
			return nm.client.Nodes().Watch(labels.Everything(), fieldSelector, resourceVersion)
		},
	}
	cache.NewReflector(listWatch, &api.Node{}, nodeStore, 0).Run()
	nm.nodeLister = &cache.StoreToNodeLister{Store: nodeStore}
}

// syncNodeStatus should be called periodically from a goroutine.
// It synchronizes node status to master, registering the kubelet first if
// necessary.
func (nm *realNodeManager) syncNodeStatus() {

	if nm.registerNode {
		// This will exit immediately if it doesn't need to do anything.
		nm.registerWithApiserver()
	}
	if err := nm.updateNodeStatus(); err != nil {
		glog.Errorf("Unable to update node status: %v", err)
	}
}

func (nm *realNodeManager) initialNodeStatus() (*api.Node, error) {
	node := &api.Node{
		ObjectMeta: api.ObjectMeta{
			Name:   nm.nodeName,
			Labels: map[string]string{"kubernetes.io/hostname": nm.hostname},
		},
	}
	if nm.cloud != nil {
		instances, ok := nm.cloud.Instances()
		if !ok {
			return nil, fmt.Errorf("failed to get instances from cloud provider")
		}

		// TODO(roberthbailey): Can we do this without having credentials to talk
		// to the cloud provider?
		// TODO: ExternalID is deprecated, we'll have to drop this code
		externalID, err := instances.ExternalID(nm.nodeName)
		if err != nil {
			return nil, fmt.Errorf("failed to get external ID from cloud provider: %v", err)
		}
		node.Spec.ExternalID = externalID

		// TODO: We can't assume that the node has credentials to talk to the
		// cloudprovider from arbitrary nodes. At most, we should talk to a
		// local metadata server here.
		node.Spec.ProviderID, err = cloudprovider.GetInstanceProviderID(nm.cloud, nm.nodeName)
		if err != nil {
			return nil, err
		}
	} else {
		node.Spec.ExternalID = nm.hostname
	}
	if err := nm.setNodeStatus(node); err != nil {
		return nil, err
	}
	return node, nil
}

// registerWithApiserver registers the node with the cluster master. It is safe
// to call multiple times, but not concurrently (nm.registrationCompleted is
// not locked).
func (nm *realNodeManager) registerWithApiserver() {
	if nm.registrationCompleted {
		return
	}
	step := 100 * time.Millisecond
	for {
		time.Sleep(step)
		step = step * 2
		if step >= 7*time.Second {
			step = 7 * time.Second
		}

		node, err := nm.initialNodeStatus()
		if err != nil {
			glog.Errorf("Unable to construct api.Node object for kubelet: %v", err)
			continue
		}
		glog.V(2).Infof("Attempting to register node %s", node.Name)
		if _, err := nm.client.Nodes().Create(node); err != nil {
			if !apierrors.IsAlreadyExists(err) {
				glog.V(2).Infof("Unable to register %s with the apiserver: %v", node.Name, err)
				continue
			}
			currentNode, err := nm.client.Nodes().Get(nm.nodeName)
			if err != nil {
				glog.Errorf("error getting node %q: %v", nm.nodeName, err)
				continue
			}
			if currentNode == nil {
				glog.Errorf("no node instance returned for %q", nm.nodeName)
				continue
			}
			if currentNode.Spec.ExternalID == node.Spec.ExternalID {
				glog.Infof("Node %s was previously registered", node.Name)
				nm.registrationCompleted = true
				return
			}
			glog.Errorf(
				"Previously %q had externalID %q; now it is %q; will delete and recreate.",
				nm.nodeName, node.Spec.ExternalID, currentNode.Spec.ExternalID,
			)
			if err := nm.client.Nodes().Delete(node.Name); err != nil {
				glog.Errorf("Unable to delete old node: %v", err)
			} else {
				glog.Errorf("Deleted old node object %q", nm.nodeName)
			}
			continue
		}
		glog.Infof("Successfully registered node %s", node.Name)
		nm.registrationCompleted = true
		return
	}
}

func (nm *realNodeManager) setUpdatedAddressesFromCloud(node *api.Node) {
	instances, ok := nm.cloud.Instances()
	if !ok {
		glog.Errorf("Failed to get instances from cloud provider, so node addresses will be stale")
		return
	}
	// TODO(roberthbailey): Can we do this without having credentials to talk to the cloud provider?
	// TODO(justinsb): We can if CurrentNodeName() was actually CurrentNode() and returned an interface
	nodeAddresses, err := instances.NodeAddresses(nm.nodeName)
	if err != nil {
		glog.Errorf("Failed to get addresses from cloud provider, so node addresses will be stale: %v", err)
		return
	}
	node.Status.Addresses = nodeAddresses
}

func (nm *realNodeManager) setUpdatedAddressesFromHostname(node *api.Node) {
	addr := net.ParseIP(nm.hostname)
	if addr == nil {
		addrs, err := net.LookupIP(node.Name)

		if err != nil {
			glog.Errorf("Can't get ip address of node %s, so node addresses will be stale: %v", node.Name, err)
			return
		}

		if len(addrs) == 0 {
			glog.Errorf("No ip address for node %v, so node addresses will be stale", node.Name)
			return
		}

		// check all ip addresses for this node.Name and try to find the first non-loopback IPv4 address.
		// If no match is found, it uses the IP of the interface with gateway on it.
		for _, ip := range addrs {
			if !ip.IsLoopback() && ip.To4() != nil {
				addr = ip
				break
			}
		}

		if addr == nil {
			ip, err := util.ChooseHostInterface()
			if err != nil {
				glog.Errorf("Failed choosing host interface, so node addresses will be stale: %v", err)
				return
			}
			addr = ip
		}
	}

	node.Status.Addresses = []api.NodeAddress{
		{Type: api.NodeLegacyHostIP, Address: addr.String()},
		{Type: api.NodeInternalIP, Address: addr.String()},
	}
}

// setNodeStatus fills in the Status fields of the given Node, overwriting
// any fields that are currently set.
func (nm *realNodeManager) setNodeStatus(node *api.Node) error {

	// Set addresses for the node. These addresses may be stale if there is an
	// error retrieving an updated value, such as the cloudprovider API being
	// unavailable.
	if nm.cloud != nil {
		nm.setUpdatedAddressesFromCloud(node)
	} else {
		nm.setUpdatedAddressesFromHostname(node)
	}

	// TODO: Post NotReady if we cannot get MachineInfo from cAdvisor. This needs to start
	// cAdvisor locally, e.g. for test-cmd.sh, and in integration test.
	info, err := nm.infoGetter.GetMachineInfo()
	if err != nil {
		// TODO(roberthbailey): This is required for test-cmd.sh to pass.
		// See if the test should be updated instead.
		node.Status.Capacity = api.ResourceList{
			api.ResourceCPU:    *resource.NewMilliQuantity(0, resource.DecimalSI),
			api.ResourceMemory: resource.MustParse("0Gi"),
			api.ResourcePods:   *resource.NewQuantity(int64(nm.pods), resource.DecimalSI),
		}
		glog.Errorf("Error getting machine info: %v", err)
	} else {
		node.Status.NodeInfo.MachineID = info.MachineID
		node.Status.NodeInfo.SystemUUID = info.SystemUUID
		node.Status.Capacity = CapacityFromMachineInfo(info)
		node.Status.Capacity[api.ResourcePods] = *resource.NewQuantity(
			int64(nm.pods), resource.DecimalSI)
		if node.Status.NodeInfo.BootID != "" &&
			node.Status.NodeInfo.BootID != info.BootID {
			// TODO: This requires a transaction, either both node status is updated
			// and event is recorded or neither should happen, see issue #6055.
			nm.recorder.Eventf(nm.nodeRef, "Rebooted",
				"Node %s has been rebooted, boot id: %s", nm.nodeName, info.BootID)
		}
		node.Status.NodeInfo.BootID = info.BootID
	}

	verinfo, err := nm.infoGetter.GetVersionInfo()
	if err != nil {
		glog.Errorf("Error getting version info: %v", err)
	} else {
		node.Status.NodeInfo.KernelVersion = verinfo.KernelVersion
		node.Status.NodeInfo.OsImage = verinfo.ContainerOsVersion
		// TODO: Determine the runtime is docker or rocket
		node.Status.NodeInfo.ContainerRuntimeVersion = "docker://" + verinfo.DockerVersion
		node.Status.NodeInfo.KubeletVersion = version.Get().String()
		// TODO: kube-proxy might be different version from kubelet in the future
		node.Status.NodeInfo.KubeProxyVersion = version.Get().String()
	}

	node.Status.DaemonEndpoints = *nm.daemonEndpoints

	// Check whether container runtime can be reported as up.
	containerRuntimeUp := nm.infoGetter.ContainerRuntimeUp()
	// Check whether network is configured properly
	networkConfigured := nm.infoGetter.NetworkConfigured()

	currentTime := util.Now()
	var newNodeReadyCondition api.NodeCondition
	var oldNodeReadyConditionStatus api.ConditionStatus
	if containerRuntimeUp && networkConfigured {
		newNodeReadyCondition = api.NodeCondition{
			Type:              api.NodeReady,
			Status:            api.ConditionTrue,
			Reason:            "KubeletReady",
			Message:           "kubelet is posting ready status",
			LastHeartbeatTime: currentTime,
		}
	} else {
		var reasons []string
		var messages []string
		if !containerRuntimeUp {
			messages = append(messages, "container runtime is down")
		}
		if !networkConfigured {
			messages = append(reasons, "network not configured correctly")
		}
		newNodeReadyCondition = api.NodeCondition{
			Type:              api.NodeReady,
			Status:            api.ConditionFalse,
			Reason:            "KubeletNotReady",
			Message:           strings.Join(messages, ","),
			LastHeartbeatTime: currentTime,
		}
	}

	updated := false
	for i := range node.Status.Conditions {
		if node.Status.Conditions[i].Type == api.NodeReady {
			oldNodeReadyConditionStatus = node.Status.Conditions[i].Status
			if oldNodeReadyConditionStatus == newNodeReadyCondition.Status {
				newNodeReadyCondition.LastTransitionTime = node.Status.Conditions[i].LastTransitionTime
			} else {
				newNodeReadyCondition.LastTransitionTime = currentTime
			}
			node.Status.Conditions[i] = newNodeReadyCondition
			updated = true
		}
	}
	if !updated {
		newNodeReadyCondition.LastTransitionTime = currentTime
		node.Status.Conditions = append(node.Status.Conditions, newNodeReadyCondition)
	}
	if !updated || oldNodeReadyConditionStatus != newNodeReadyCondition.Status {
		if newNodeReadyCondition.Status == api.ConditionTrue {
			nm.recordNodeStatusEvent("NodeReady")
		} else {
			nm.recordNodeStatusEvent("NodeNotReady")
		}
	}
	if oldNodeUnschedulable != node.Spec.Unschedulable {
		if node.Spec.Unschedulable {
			nm.recordNodeStatusEvent("NodeNotSchedulable")
		} else {
			nm.recordNodeStatusEvent("NodeSchedulable")
		}
		oldNodeUnschedulable = node.Spec.Unschedulable
	}
	return nil
}

// updateNodeStatus updates node status to master with retries.
func (nm *realNodeManager) updateNodeStatus() error {
	for i := 0; i < nodeStatusUpdateRetry; i++ {
		if err := nm.tryUpdateNodeStatus(); err != nil {
			glog.Errorf("Error updating node status, will retry: %v", err)
		} else {
			return nil
		}
	}
	return fmt.Errorf("update node status exceeds retry count")
}

func (nm *realNodeManager) recordNodeStatusEvent(event string) {
	glog.V(2).Infof("Recording %s event message for node %s", event, nm.nodeName)
	// TODO: This requires a transaction, either both node status is updated
	// and event is recorded or neither should happen, see issue #6055.
	nm.recorder.Eventf(nm.nodeRef, event, "Node %s status is now: %s", nm.nodeName, event)
}

// tryUpdateNodeStatus tries to update node status to master. If ReconcileCBR0
// is set, this function will also confirm that cbr0 is configured correctly.
func (nm *realNodeManager) tryUpdateNodeStatus() error {
	node, err := nm.client.Nodes().Get(nm.nodeName)
	if err != nil {
		return fmt.Errorf("error getting node %q: %v", nm.nodeName, err)
	}
	if node == nil {
		return fmt.Errorf("no node instance returned for %q", nm.nodeName)
	}
	nm.lock.Lock()
	defer nm.lock.Unlock()
	nm.podCIDR = node.Spec.PodCIDR

	if err := nm.setNodeStatus(node); err != nil {
		return err
	}
	// Update the current status on the API server
	_, err = nm.client.Nodes().UpdateStatus(node)
	return err
}
