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
	"fmt"
	"time"

	"github.com/golang/glog"

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
	nodeutilv1 "k8s.io/kubernetes/pkg/api/v1/node"
	"k8s.io/kubernetes/pkg/cloudprovider"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
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
	eventBroadcaster.StartLogging(glog.Infof)
	if kubeClient != nil {
		glog.V(0).Infof("Sending events to api server.")
		eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: v1core.New(kubeClient.CoreV1().RESTClient()).Events("")})
	} else {
		glog.V(0).Infof("No api server defined - no events will be sent to API server.")
	}

	cnc := &CloudNodeController{
		nodeInformer:              nodeInformer,
		kubeClient:                kubeClient,
		recorder:                  recorder,
		cloud:                     cloud,
		nodeMonitorPeriod:         nodeMonitorPeriod,
		nodeStatusUpdateFrequency: nodeStatusUpdateFrequency,
	}

	nodeInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: cnc.AddCloudNode,
	})

	return cnc
}

// This controller deletes a node if kubelet is not reporting
// and the node is gone from the cloud provider.
func (cnc *CloudNodeController) Run() {
	defer utilruntime.HandleCrash()

	// The following loops run communicate with the APIServer with a worst case complexity
	// of O(num_nodes) per cycle. These functions are justified here because these events fire
	// very infrequently. DO NOT MODIFY this to perform frequent operations.

	// Start a loop to periodically update the node addresses obtained from the cloud
	go wait.Until(cnc.UpdateNodeStatus, cnc.nodeStatusUpdateFrequency, wait.NeverStop)

	// Start a loop to periodically check if any nodes have been deleted from cloudprovider
	go wait.Until(cnc.MonitorNode, cnc.nodeMonitorPeriod, wait.NeverStop)
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
		glog.Errorf("Error monitoring node status: %v", err)
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
		glog.V(5).Infof("This node %s is still tainted. Will not process.", node.Name)
		return
	}

	nodeAddresses, err := getNodeAddressesByProviderIDOrName(instances, node)
	if err != nil {
		glog.Errorf("%v", err)
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
			glog.Errorf("Specified Node IP not found in cloudprovider")
			return
		}
		nodeAddresses = []v1.NodeAddress{*nodeIP}
	}
	newNode := node.DeepCopy()
	newNode.Status.Addresses = nodeAddresses
	if !nodeAddressesChangeDetected(node.Status.Addresses, newNode.Status.Addresses) {
		return
	}
	_, err = nodeutil.PatchNodeStatus(cnc.kubeClient.CoreV1(), types.NodeName(node.Name), node, newNode)
	if err != nil {
		glog.Errorf("Error patching node with cloud ip addresses = [%v]", err)
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
		glog.Errorf("Error monitoring node status: %v", err)
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
				glog.Errorf("Failed while getting a Node to retry updating NodeStatus. Probably Node %s was deleted.", name)
				break
			}
			time.Sleep(retrySleepTime)
		}
		if currentReadyCondition == nil {
			glog.Errorf("Update status of Node %v from CloudNodeController exceeds retry count or the Node was deleted.", node.Name)
			continue
		}
		// If the known node status says that Node is NotReady, then check if the node has been removed
		// from the cloud provider. If node cannot be found in cloudprovider, then delete the node immediately
		if currentReadyCondition != nil {
			if currentReadyCondition.Status != v1.ConditionTrue {
				// Check with the cloud provider to see if the node still exists. If it
				// doesn't, delete the node immediately.
				exists, err := ensureNodeExistsByProviderIDOrExternalID(instances, node)
				if err != nil {
					glog.Errorf("Error getting data for node %s from cloud: %v", node.Name, err)
					continue
				}

				if exists {
					// Continue checking the remaining nodes since the current one is fine.
					continue
				}

				glog.V(2).Infof("Deleting node since it is no longer present in cloud provider: %s", node.Name)

				ref := &v1.ObjectReference{
					Kind:      "Node",
					Name:      node.Name,
					UID:       types.UID(node.UID),
					Namespace: "",
				}
				glog.V(2).Infof("Recording %s event message for node %s", "DeletingNode", node.Name)

				cnc.recorder.Eventf(ref, v1.EventTypeNormal, fmt.Sprintf("Deleting Node %v because it's not present according to cloud provider", node.Name), "Node %s event: %s", node.Name, "DeletingNode")

				go func(nodeName string) {
					defer utilruntime.HandleCrash()
					if err := cnc.kubeClient.CoreV1().Nodes().Delete(nodeName, nil); err != nil {
						glog.Errorf("unable to delete node %q: %v", nodeName, err)
					}
				}(node.Name)

			}
		}
	}
}

// This processes nodes that were added into the cluster, and cloud initialize them if appropriate
func (cnc *CloudNodeController) AddCloudNode(obj interface{}) {
	node := obj.(*v1.Node)

	instances, ok := cnc.cloud.Instances()
	if !ok {
		utilruntime.HandleError(fmt.Errorf("failed to get instances from cloud provider"))
		return
	}

	cloudTaint := getCloudTaint(node.Spec.Taints)
	if cloudTaint == nil {
		glog.V(2).Infof("This node %s is registered without the cloud taint. Will not process.", node.Name)
		return
	}

	err := clientretry.RetryOnConflict(UpdateNodeSpecBackoff, func() error {
		curNode, err := cnc.kubeClient.CoreV1().Nodes().Get(node.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}

		if curNode.Spec.ProviderID == "" {
			providerID, err := cloudprovider.GetInstanceProviderID(cnc.cloud, types.NodeName(curNode.Name))
			if err == nil {
				curNode.Spec.ProviderID = providerID
			} else {
				// we should attempt to set providerID on curNode, but
				// we can continue if we fail since we will attempt to set
				// node addresses given the node name in getNodeAddressesByProviderIDOrName
				glog.Errorf("failed to set node provider id: %v", err)
			}
		}

		nodeAddresses, err := getNodeAddressesByProviderIDOrName(instances, curNode)
		if err != nil {
			glog.Errorf("%v", err)
			return nil
		}

		// If user provided an IP address, ensure that IP address is found
		// in the cloud provider before removing the taint on the node
		if nodeIP, ok := ensureNodeProvidedIPExists(curNode, nodeAddresses); ok {
			if nodeIP == nil {
				glog.Errorf("failed to get specified nodeIP in cloudprovider")
				return nil
			}
		}

		if instanceType, err := getInstanceTypeByProviderIDOrName(instances, curNode); err != nil {
			glog.Errorf("%v", err)
			return err
		} else if instanceType != "" {
			glog.Infof("Adding node label from cloud provider: %s=%s", kubeletapis.LabelInstanceType, instanceType)
			curNode.ObjectMeta.Labels[kubeletapis.LabelInstanceType] = instanceType
		}

		// TODO(wlan0): Move this logic to the route controller using the node taint instead of condition
		// Since there are node taints, do we still need this?
		// This condition marks the node as unusable until routes are initialized in the cloud provider
		if cnc.cloud.ProviderName() == "gce" {
			curNode.Status.Conditions = append(node.Status.Conditions, v1.NodeCondition{
				Type:               v1.NodeNetworkUnavailable,
				Status:             v1.ConditionTrue,
				Reason:             "NoRouteCreated",
				Message:            "Node created without a route",
				LastTransitionTime: metav1.Now(),
			})
		}

		if zones, ok := cnc.cloud.Zones(); ok {
			zone, err := getZoneByProviderIDOrName(zones, curNode)
			if err != nil {
				return fmt.Errorf("failed to get zone from cloud provider: %v", err)
			}
			if zone.FailureDomain != "" {
				glog.Infof("Adding node label from cloud provider: %s=%s", kubeletapis.LabelZoneFailureDomain, zone.FailureDomain)
				curNode.ObjectMeta.Labels[kubeletapis.LabelZoneFailureDomain] = zone.FailureDomain
			}
			if zone.Region != "" {
				glog.Infof("Adding node label from cloud provider: %s=%s", kubeletapis.LabelZoneRegion, zone.Region)
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
}

func getCloudTaint(taints []v1.Taint) *v1.Taint {
	for _, taint := range taints {
		if taint.Key == algorithm.TaintExternalCloudProvider {
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

// ensureNodeExistsByProviderIDOrExternalID first checks if the instance exists by the provider id and then by calling external id with node name
func ensureNodeExistsByProviderIDOrExternalID(instances cloudprovider.Instances, node *v1.Node) (bool, error) {
	exists, err := instances.InstanceExistsByProviderID(node.Spec.ProviderID)
	if err != nil {
		providerIDErr := err
		_, err = instances.ExternalID(types.NodeName(node.Name))
		if err == nil {
			return true, nil
		}
		if err == cloudprovider.InstanceNotFound {
			return false, nil
		}

		return false, fmt.Errorf("InstanceExistsByProviderID: Error fetching by providerID: %v Error fetching by NodeName: %v", providerIDErr, err)
	}

	return exists, nil
}

func getNodeAddressesByProviderIDOrName(instances cloudprovider.Instances, node *v1.Node) ([]v1.NodeAddress, error) {
	nodeAddresses, err := instances.NodeAddressesByProviderID(node.Spec.ProviderID)
	if err != nil {
		providerIDErr := err
		nodeAddresses, err = instances.NodeAddresses(types.NodeName(node.Name))
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
	instanceType, err := instances.InstanceTypeByProviderID(node.Spec.ProviderID)
	if err != nil {
		providerIDErr := err
		instanceType, err = instances.InstanceType(types.NodeName(node.Name))
		if err != nil {
			return "", fmt.Errorf("InstanceType: Error fetching by providerID: %v Error fetching by NodeName: %v", providerIDErr, err)
		}
	}
	return instanceType, err
}

// getZoneByProviderIDorName will attempt to get the zone of node using its providerID
// then it's name. If both attempts fail, an error is returned
func getZoneByProviderIDOrName(zones cloudprovider.Zones, node *v1.Node) (cloudprovider.Zone, error) {
	zone, err := zones.GetZoneByProviderID(node.Spec.ProviderID)
	if err != nil {
		providerIDErr := err
		zone, err = zones.GetZoneByNodeName(types.NodeName(node.Name))
		if err != nil {
			return cloudprovider.Zone{}, fmt.Errorf("Zone: Error fetching by providerID: %v Error fetching by NodeName: %v", providerIDErr, err)
		}
	}

	return zone, nil
}
