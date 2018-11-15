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

package cloud

import (
	"context"
	"errors"
	"fmt"
	"time"

	"k8s.io/klog"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	clientretry "k8s.io/client-go/util/retry"
	cloudprovider "k8s.io/cloud-provider"
	nodeutilv1 "k8s.io/kubernetes/pkg/api/v1/node"
	"k8s.io/kubernetes/pkg/controller"
	nodectrlutil "k8s.io/kubernetes/pkg/controller/util/node"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
)

var UpdateNodeSpecBackoff = wait.Backoff{
	Steps:    20,
	Duration: 50 * time.Millisecond,
	Jitter:   1.0,
}

type CloudNodeController struct {
	nodeInformer coreinformers.NodeInformer
	kubeClient   clientset.Interface
	recorder     record.EventRecorder

	cloud cloudprovider.Interface

	// Value controlling NodeController monitoring period, i.e. how often does NodeController
	// check node status posted from kubelet. This value should be lower than nodeMonitorGracePeriod
	// set in controller-manager
	nodeMonitorPeriod time.Duration

	nodeStatusUpdateFrequency time.Duration
}

const (
	// nodeStatusUpdateRetry controls the number of retries of writing NodeStatus update.
	nodeStatusUpdateRetry = 5

	// The amount of time the nodecontroller should sleep between retrying NodeStatus updates
	retrySleepTime = 20 * time.Millisecond
)

// NewCloudNodeController creates a CloudNodeController object
func NewCloudNodeController(
	nodeInformer coreinformers.NodeInformer,
	kubeClient clientset.Interface,
	cloud cloudprovider.Interface,
	nodeMonitorPeriod time.Duration,
	nodeStatusUpdateFrequency time.Duration) *CloudNodeController {

	eventBroadcaster := record.NewBroadcaster()
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "cloud-node-controller"})
	eventBroadcaster.StartLogging(klog.Infof)
	if kubeClient != nil {
		klog.V(0).Infof("Sending events to api server.")
		eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: kubeClient.CoreV1().Events("")})
	} else {
		klog.V(0).Infof("No api server defined - no events will be sent to API server.")
	}

	cnc := &CloudNodeController{
		nodeInformer:              nodeInformer,
		kubeClient:                kubeClient,
		recorder:                  recorder,
		cloud:                     cloud,
		nodeMonitorPeriod:         nodeMonitorPeriod,
		nodeStatusUpdateFrequency: nodeStatusUpdateFrequency,
	}

	// Use shared informer to listen to add/update of nodes. Note that any nodes
	// that exist before node controller starts will show up in the update method
	cnc.nodeInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    cnc.AddCloudNode,
		UpdateFunc: cnc.UpdateCloudNode,
	})

	return cnc
}

// This controller deletes a node if kubelet is not reporting
// and the node is gone from the cloud provider.
func (cnc *CloudNodeController) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()

	// The following loops run communicate with the APIServer with a worst case complexity
	// of O(num_nodes) per cycle. These functions are justified here because these events fire
	// very infrequently. DO NOT MODIFY this to perform frequent operations.

	// Start a loop to periodically update the node addresses obtained from the cloud
	go wait.Until(cnc.UpdateNodeStatus, cnc.nodeStatusUpdateFrequency, stopCh)

	// Start a loop to periodically check if any nodes have been deleted from cloudprovider
	go wait.Until(cnc.MonitorNode, cnc.nodeMonitorPeriod, stopCh)
}

// UpdateNodeStatus updates the node status, such as node addresses
func (cnc *CloudNodeController) UpdateNodeStatus() {
	instances, ok := cnc.cloud.Instances()
	if !ok {
		utilruntime.HandleError(fmt.Errorf("failed to get instances from cloud provider"))
		return
	}

	nodes, err := cnc.kubeClient.CoreV1().Nodes().List(metav1.ListOptions{ResourceVersion: "0"})
	if err != nil {
		klog.Errorf("Error monitoring node status: %v", err)
		return
	}

	for i := range nodes.Items {
		cnc.updateNodeAddress(&nodes.Items[i], instances)
	}
}

// UpdateNodeAddress updates the nodeAddress of a single node
func (cnc *CloudNodeController) updateNodeAddress(node *v1.Node, instances cloudprovider.Instances) {
	// Do not process nodes that are still tainted
	cloudTaint := getCloudTaint(node.Spec.Taints)
	if cloudTaint != nil {
		klog.V(5).Infof("This node %s is still tainted. Will not process.", node.Name)
		return
	}
	// Node that isn't present according to the cloud provider shouldn't have its address updated
	exists, err := ensureNodeExistsByProviderID(instances, node)
	if err != nil {
		// Continue to update node address when not sure the node is not exists
		klog.Errorf("%v", err)
	} else if !exists {
		klog.V(4).Infof("The node %s is no longer present according to the cloud provider, do not process.", node.Name)
		return
	}

	nodeAddresses, err := getNodeAddressesByProviderIDOrName(instances, node)
	if err != nil {
		klog.Errorf("%v", err)
		return
	}

	if len(nodeAddresses) == 0 {
		klog.V(5).Infof("Skipping node address update for node %q since cloud provider did not return any", node.Name)
		return
	}

	// Check if a hostname address exists in the cloud provided addresses
	hostnameExists := false
	for i := range nodeAddresses {
		if nodeAddresses[i].Type == v1.NodeHostName {
			hostnameExists = true
		}
	}
	// If hostname was not present in cloud provided addresses, use the hostname
	// from the existing node (populated by kubelet)
	if !hostnameExists {
		for _, addr := range node.Status.Addresses {
			if addr.Type == v1.NodeHostName {
				nodeAddresses = append(nodeAddresses, addr)
			}
		}
	}
	// If nodeIP was suggested by user, ensure that
	// it can be found in the cloud as well (consistent with the behaviour in kubelet)
	if nodeIP, ok := ensureNodeProvidedIPExists(node, nodeAddresses); ok {
		if nodeIP == nil {
			klog.Errorf("Specified Node IP not found in cloudprovider")
			return
		}
	}
	newNode := node.DeepCopy()
	newNode.Status.Addresses = nodeAddresses
	if !nodeAddressesChangeDetected(node.Status.Addresses, newNode.Status.Addresses) {
		return
	}
	_, _, err = nodeutil.PatchNodeStatus(cnc.kubeClient.CoreV1(), types.NodeName(node.Name), node, newNode)
	if err != nil {
		klog.Errorf("Error patching node with cloud ip addresses = [%v]", err)
	}
}

// Monitor node queries the cloudprovider for non-ready nodes and deletes them
// if they cannot be found in the cloud provider
func (cnc *CloudNodeController) MonitorNode() {
	instances, ok := cnc.cloud.Instances()
	if !ok {
		utilruntime.HandleError(fmt.Errorf("failed to get instances from cloud provider"))
		return
	}

	nodes, err := cnc.kubeClient.CoreV1().Nodes().List(metav1.ListOptions{ResourceVersion: "0"})
	if err != nil {
		klog.Errorf("Error monitoring node status: %v", err)
		return
	}

	for i := range nodes.Items {
		var currentReadyCondition *v1.NodeCondition
		node := &nodes.Items[i]
		// Try to get the current node status
		// If node status is empty, then kubelet has not posted ready status yet. In this case, process next node
		for rep := 0; rep < nodeStatusUpdateRetry; rep++ {
			_, currentReadyCondition = nodeutilv1.GetNodeCondition(&node.Status, v1.NodeReady)
			if currentReadyCondition != nil {
				break
			}
			name := node.Name
			node, err = cnc.kubeClient.CoreV1().Nodes().Get(name, metav1.GetOptions{})
			if err != nil {
				klog.Errorf("Failed while getting a Node to retry updating NodeStatus. Probably Node %s was deleted.", name)
				break
			}
			time.Sleep(retrySleepTime)
		}
		if currentReadyCondition == nil {
			klog.Errorf("Update status of Node %v from CloudNodeController exceeds retry count or the Node was deleted.", node.Name)
			continue
		}
		// If the known node status says that Node is NotReady, then check if the node has been removed
		// from the cloud provider. If node cannot be found in cloudprovider, then delete the node immediately
		if currentReadyCondition != nil {
			if currentReadyCondition.Status != v1.ConditionTrue {
				// we need to check this first to get taint working in similar in all cloudproviders
				// current problem is that shutdown nodes are not working in similar way ie. all cloudproviders
				// does not delete node from kubernetes cluster when instance it is shutdown see issue #46442
				shutdown, err := nodectrlutil.ShutdownInCloudProvider(context.TODO(), cnc.cloud, node)
				if err != nil {
					klog.Errorf("Error checking if node %s is shutdown: %v", node.Name, err)
				}

				if shutdown && err == nil {
					// if node is shutdown add shutdown taint
					err = controller.AddOrUpdateTaintOnNode(cnc.kubeClient, node.Name, controller.ShutdownTaint)
					if err != nil {
						klog.Errorf("Error patching node taints: %v", err)
					}
					// Continue checking the remaining nodes since the current one is shutdown.
					continue
				}

				// Check with the cloud provider to see if the node still exists. If it
				// doesn't, delete the node immediately.
				exists, err := ensureNodeExistsByProviderID(instances, node)
				if err != nil {
					klog.Errorf("Error checking if node %s exists: %v", node.Name, err)
					continue
				}

				if exists {
					// Continue checking the remaining nodes since the current one is fine.
					continue
				}

				klog.V(2).Infof("Deleting node since it is no longer present in cloud provider: %s", node.Name)

				ref := &v1.ObjectReference{
					Kind:      "Node",
					Name:      node.Name,
					UID:       types.UID(node.UID),
					Namespace: "",
				}
				klog.V(2).Infof("Recording %s event message for node %s", "DeletingNode", node.Name)

				cnc.recorder.Eventf(ref, v1.EventTypeNormal, fmt.Sprintf("Deleting Node %v because it's not present according to cloud provider", node.Name), "Node %s event: %s", node.Name, "DeletingNode")

				go func(nodeName string) {
					defer utilruntime.HandleCrash()
					if err := cnc.kubeClient.CoreV1().Nodes().Delete(nodeName, nil); err != nil {
						klog.Errorf("unable to delete node %q: %v", nodeName, err)
					}
				}(node.Name)

			} else {
				// if taint exist remove taint
				err = controller.RemoveTaintOffNode(cnc.kubeClient, node.Name, node, controller.ShutdownTaint)
				if err != nil {
					klog.Errorf("Error patching node taints: %v", err)
				}
			}
		}
	}
}

func (cnc *CloudNodeController) UpdateCloudNode(_, newObj interface{}) {
	if _, ok := newObj.(*v1.Node); !ok {
		utilruntime.HandleError(fmt.Errorf("unexpected object type: %v", newObj))
		return
	}
	cnc.AddCloudNode(newObj)
}

// This processes nodes that were added into the cluster, and cloud initialize them if appropriate
func (cnc *CloudNodeController) AddCloudNode(obj interface{}) {
	node := obj.(*v1.Node)

	cloudTaint := getCloudTaint(node.Spec.Taints)
	if cloudTaint == nil {
		klog.V(2).Infof("This node %s is registered without the cloud taint. Will not process.", node.Name)
		return
	}

	instances, ok := cnc.cloud.Instances()
	if !ok {
		utilruntime.HandleError(fmt.Errorf("failed to get instances from cloud provider"))
		return
	}

	err := clientretry.RetryOnConflict(UpdateNodeSpecBackoff, func() error {
		// TODO(wlan0): Move this logic to the route controller using the node taint instead of condition
		// Since there are node taints, do we still need this?
		// This condition marks the node as unusable until routes are initialized in the cloud provider
		if cnc.cloud.ProviderName() == "gce" {
			if err := nodeutil.SetNodeCondition(cnc.kubeClient, types.NodeName(node.Name), v1.NodeCondition{
				Type:               v1.NodeNetworkUnavailable,
				Status:             v1.ConditionTrue,
				Reason:             "NoRouteCreated",
				Message:            "Node created without a route",
				LastTransitionTime: metav1.Now(),
			}); err != nil {
				return err
			}
		}

		curNode, err := cnc.kubeClient.CoreV1().Nodes().Get(node.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}

		if curNode.Spec.ProviderID == "" {
			providerID, err := cloudprovider.GetInstanceProviderID(context.TODO(), cnc.cloud, types.NodeName(curNode.Name))
			if err == nil {
				curNode.Spec.ProviderID = providerID
			} else {
				// we should attempt to set providerID on curNode, but
				// we can continue if we fail since we will attempt to set
				// node addresses given the node name in getNodeAddressesByProviderIDOrName
				klog.Errorf("failed to set node provider id: %v", err)
			}
		}

		nodeAddresses, err := getNodeAddressesByProviderIDOrName(instances, curNode)
		if err != nil {
			return err
		}

		// If user provided an IP address, ensure that IP address is found
		// in the cloud provider before removing the taint on the node
		if nodeIP, ok := ensureNodeProvidedIPExists(curNode, nodeAddresses); ok {
			if nodeIP == nil {
				return errors.New("failed to find kubelet node IP from cloud provider")
			}
		}

		if instanceType, err := getInstanceTypeByProviderIDOrName(instances, curNode); err != nil {
			return err
		} else if instanceType != "" {
			klog.V(2).Infof("Adding node label from cloud provider: %s=%s", kubeletapis.LabelInstanceType, instanceType)
			curNode.ObjectMeta.Labels[kubeletapis.LabelInstanceType] = instanceType
		}

		if zones, ok := cnc.cloud.Zones(); ok {
			zone, err := getZoneByProviderIDOrName(zones, curNode)
			if err != nil {
				return fmt.Errorf("failed to get zone from cloud provider: %v", err)
			}
			if zone.FailureDomain != "" {
				klog.V(2).Infof("Adding node label from cloud provider: %s=%s", kubeletapis.LabelZoneFailureDomain, zone.FailureDomain)
				curNode.ObjectMeta.Labels[kubeletapis.LabelZoneFailureDomain] = zone.FailureDomain
			}
			if zone.Region != "" {
				klog.V(2).Infof("Adding node label from cloud provider: %s=%s", kubeletapis.LabelZoneRegion, zone.Region)
				curNode.ObjectMeta.Labels[kubeletapis.LabelZoneRegion] = zone.Region
			}
		}

		curNode.Spec.Taints = excludeTaintFromList(curNode.Spec.Taints, *cloudTaint)

		_, err = cnc.kubeClient.CoreV1().Nodes().Update(curNode)
		if err != nil {
			return err
		}
		// After adding, call UpdateNodeAddress to set the CloudProvider provided IPAddresses
		// So that users do not see any significant delay in IP addresses being filled into the node
		cnc.updateNodeAddress(curNode, instances)
		return nil
	})
	if err != nil {
		utilruntime.HandleError(err)
		return
	}

	klog.Infof("Successfully initialized node %s with cloud provider", node.Name)
}

func getCloudTaint(taints []v1.Taint) *v1.Taint {
	for _, taint := range taints {
		if taint.Key == schedulerapi.TaintExternalCloudProvider {
			return &taint
		}
	}
	return nil
}

func excludeTaintFromList(taints []v1.Taint, toExclude v1.Taint) []v1.Taint {
	newTaints := []v1.Taint{}
	for _, taint := range taints {
		if toExclude.MatchTaint(&taint) {
			continue
		}
		newTaints = append(newTaints, taint)
	}
	return newTaints
}

// ensureNodeExistsByProviderID checks if the instance exists by the provider id,
// If provider id in spec is empty it calls instanceId with node name to get provider id
func ensureNodeExistsByProviderID(instances cloudprovider.Instances, node *v1.Node) (bool, error) {
	providerID := node.Spec.ProviderID
	if providerID == "" {
		var err error
		providerID, err = instances.InstanceID(context.TODO(), types.NodeName(node.Name))
		if err != nil {
			if err == cloudprovider.InstanceNotFound {
				return false, nil
			}
			return false, err
		}

		if providerID == "" {
			klog.Warningf("Cannot find valid providerID for node name %q, assuming non existence", node.Name)
			return false, nil
		}
	}

	return instances.InstanceExistsByProviderID(context.TODO(), providerID)
}

func getNodeAddressesByProviderIDOrName(instances cloudprovider.Instances, node *v1.Node) ([]v1.NodeAddress, error) {
	nodeAddresses, err := instances.NodeAddressesByProviderID(context.TODO(), node.Spec.ProviderID)
	if err != nil {
		providerIDErr := err
		nodeAddresses, err = instances.NodeAddresses(context.TODO(), types.NodeName(node.Name))
		if err != nil {
			return nil, fmt.Errorf("NodeAddress: Error fetching by providerID: %v Error fetching by NodeName: %v", providerIDErr, err)
		}
	}
	return nodeAddresses, nil
}

func nodeAddressesChangeDetected(addressSet1, addressSet2 []v1.NodeAddress) bool {
	if len(addressSet1) != len(addressSet2) {
		return true
	}
	addressMap1 := map[v1.NodeAddressType]string{}
	addressMap2 := map[v1.NodeAddressType]string{}

	for i := range addressSet1 {
		addressMap1[addressSet1[i].Type] = addressSet1[i].Address
		addressMap2[addressSet2[i].Type] = addressSet2[i].Address
	}

	for k, v := range addressMap1 {
		if addressMap2[k] != v {
			return true
		}
	}
	return false
}

func ensureNodeProvidedIPExists(node *v1.Node, nodeAddresses []v1.NodeAddress) (*v1.NodeAddress, bool) {
	var nodeIP *v1.NodeAddress
	nodeIPExists := false
	if providedIP, ok := node.ObjectMeta.Annotations[kubeletapis.AnnotationProvidedIPAddr]; ok {
		nodeIPExists = true
		for i := range nodeAddresses {
			if nodeAddresses[i].Address == providedIP {
				nodeIP = &nodeAddresses[i]
				break
			}
		}
	}
	return nodeIP, nodeIPExists
}

func getInstanceTypeByProviderIDOrName(instances cloudprovider.Instances, node *v1.Node) (string, error) {
	instanceType, err := instances.InstanceTypeByProviderID(context.TODO(), node.Spec.ProviderID)
	if err != nil {
		providerIDErr := err
		instanceType, err = instances.InstanceType(context.TODO(), types.NodeName(node.Name))
		if err != nil {
			return "", fmt.Errorf("InstanceType: Error fetching by providerID: %v Error fetching by NodeName: %v", providerIDErr, err)
		}
	}
	return instanceType, err
}

// getZoneByProviderIDorName will attempt to get the zone of node using its providerID
// then it's name. If both attempts fail, an error is returned
func getZoneByProviderIDOrName(zones cloudprovider.Zones, node *v1.Node) (cloudprovider.Zone, error) {
	zone, err := zones.GetZoneByProviderID(context.TODO(), node.Spec.ProviderID)
	if err != nil {
		providerIDErr := err
		zone, err = zones.GetZoneByNodeName(context.TODO(), types.NodeName(node.Name))
		if err != nil {
			return cloudprovider.Zone{}, fmt.Errorf("Zone: Error fetching by providerID: %v Error fetching by NodeName: %v", providerIDErr, err)
		}
	}

	return zone, nil
}
