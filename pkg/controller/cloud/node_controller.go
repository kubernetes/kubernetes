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

	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
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
	cloudproviderapi "k8s.io/cloud-provider/api"
	cloudnodeutil "k8s.io/cloud-provider/node/helpers"
	"k8s.io/klog/v2"
)

// labelReconcileInfo lists Node labels to reconcile, and how to reconcile them.
// primaryKey and secondaryKey are keys of labels to reconcile.
//   - If both keys exist, but their values don't match. Use the value from the
//   primaryKey as the source of truth to reconcile.
//   - If ensureSecondaryExists is true, and the secondaryKey does not
//   exist, secondaryKey will be added with the value of the primaryKey.
var labelReconcileInfo = []struct {
	primaryKey            string
	secondaryKey          string
	ensureSecondaryExists bool
}{
	{
		// Reconcile the beta and the GA zone label using the beta label as
		// the source of truth
		// TODO: switch the primary key to GA labels in v1.21
		primaryKey:            v1.LabelZoneFailureDomain,
		secondaryKey:          v1.LabelZoneFailureDomainStable,
		ensureSecondaryExists: true,
	},
	{
		// Reconcile the beta and the stable region label using the beta label as
		// the source of truth
		// TODO: switch the primary key to GA labels in v1.21
		primaryKey:            v1.LabelZoneRegion,
		secondaryKey:          v1.LabelZoneRegionStable,
		ensureSecondaryExists: true,
	},
	{
		// Reconcile the beta and the stable instance-type label using the beta label as
		// the source of truth
		// TODO: switch the primary key to GA labels in v1.21
		primaryKey:            v1.LabelInstanceType,
		secondaryKey:          v1.LabelInstanceTypeStable,
		ensureSecondaryExists: true,
	},
}

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

	nodeStatusUpdateFrequency time.Duration
}

// NewCloudNodeController creates a CloudNodeController object
func NewCloudNodeController(
	nodeInformer coreinformers.NodeInformer,
	kubeClient clientset.Interface,
	cloud cloudprovider.Interface,
	nodeStatusUpdateFrequency time.Duration) (*CloudNodeController, error) {

	eventBroadcaster := record.NewBroadcaster()
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "cloud-node-controller"})
	eventBroadcaster.StartLogging(klog.Infof)

	klog.Infof("Sending events to api server.")
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: kubeClient.CoreV1().Events("")})

	if _, ok := cloud.Instances(); !ok {
		return nil, errors.New("cloud provider does not support instances")
	}

	cnc := &CloudNodeController{
		nodeInformer:              nodeInformer,
		kubeClient:                kubeClient,
		recorder:                  recorder,
		cloud:                     cloud,
		nodeStatusUpdateFrequency: nodeStatusUpdateFrequency,
	}

	// Use shared informer to listen to add/update of nodes. Note that any nodes
	// that exist before node controller starts will show up in the update method
	cnc.nodeInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    func(obj interface{}) { cnc.AddCloudNode(context.TODO(), obj) },
		UpdateFunc: func(oldObj, newObj interface{}) { cnc.UpdateCloudNode(context.TODO(), oldObj, newObj) },
	})

	return cnc, nil
}

// This controller updates newly registered nodes with information
// from the cloud provider. This call is blocking so should be called
// via a goroutine
func (cnc *CloudNodeController) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()

	// The following loops run communicate with the APIServer with a worst case complexity
	// of O(num_nodes) per cycle. These functions are justified here because these events fire
	// very infrequently. DO NOT MODIFY this to perform frequent operations.

	// Start a loop to periodically update the node addresses obtained from the cloud
	wait.Until(func() { cnc.UpdateNodeStatus(context.TODO()) }, cnc.nodeStatusUpdateFrequency, stopCh)
}

// UpdateNodeStatus updates the node status, such as node addresses
func (cnc *CloudNodeController) UpdateNodeStatus(ctx context.Context) {
	instances, ok := cnc.cloud.Instances()
	if !ok {
		utilruntime.HandleError(fmt.Errorf("failed to get instances from cloud provider"))
		return
	}

	nodes, err := cnc.kubeClient.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{ResourceVersion: "0"})
	if err != nil {
		klog.Errorf("Error monitoring node status: %v", err)
		return
	}

	for i := range nodes.Items {
		cnc.updateNodeAddress(ctx, &nodes.Items[i], instances)
	}

	for _, node := range nodes.Items {
		err = cnc.reconcileNodeLabels(node.Name)
		if err != nil {
			klog.Errorf("Error reconciling node labels for node %q, err: %v", node.Name, err)
		}
	}
}

// reconcileNodeLabels reconciles node labels transitioning from beta to GA
func (cnc *CloudNodeController) reconcileNodeLabels(nodeName string) error {
	node, err := cnc.nodeInformer.Lister().Get(nodeName)
	if err != nil {
		// If node not found, just ignore it.
		if apierrors.IsNotFound(err) {
			return nil
		}

		return err
	}

	if node.Labels == nil {
		// Nothing to reconcile.
		return nil
	}

	labelsToUpdate := map[string]string{}
	for _, r := range labelReconcileInfo {
		primaryValue, primaryExists := node.Labels[r.primaryKey]
		secondaryValue, secondaryExists := node.Labels[r.secondaryKey]

		if !primaryExists {
			// The primary label key does not exist. This should not happen
			// within our supported version skew range, when no external
			// components/factors modifying the node object. Ignore this case.
			continue
		}
		if secondaryExists && primaryValue != secondaryValue {
			// Secondary label exists, but not consistent with the primary
			// label. Need to reconcile.
			labelsToUpdate[r.secondaryKey] = primaryValue

		} else if !secondaryExists && r.ensureSecondaryExists {
			// Apply secondary label based on primary label.
			labelsToUpdate[r.secondaryKey] = primaryValue
		}
	}

	if len(labelsToUpdate) == 0 {
		return nil
	}

	if !cloudnodeutil.AddOrUpdateLabelsOnNode(cnc.kubeClient, labelsToUpdate, node) {
		return fmt.Errorf("failed update labels for node %+v", node)
	}

	return nil
}

// UpdateNodeAddress updates the nodeAddress of a single node
func (cnc *CloudNodeController) updateNodeAddress(ctx context.Context, node *v1.Node, instances cloudprovider.Instances) {
	// Do not process nodes that are still tainted
	cloudTaint := getCloudTaint(node.Spec.Taints)
	if cloudTaint != nil {
		klog.V(5).Infof("This node %s is still tainted. Will not process.", node.Name)
		return
	}
	// Node that isn't present according to the cloud provider shouldn't have its address updated
	exists, err := ensureNodeExistsByProviderID(ctx, instances, node)
	if err != nil {
		// Continue to update node address when not sure the node is not exists
		klog.Errorf("%v", err)
	} else if !exists {
		klog.V(4).Infof("The node %s is no longer present according to the cloud provider, do not process.", node.Name)
		return
	}

	nodeAddresses, err := getNodeAddressesByProviderIDOrName(ctx, instances, node.Spec.ProviderID, node.Name)
	if err != nil {
		klog.Errorf("Error getting node addresses for node %q: %v", node.Name, err)
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
			break
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
			klog.Errorf("Specified Node IP not found in cloudprovider for node %q", node.Name)
			return
		}
	}
	if !nodeAddressesChangeDetected(node.Status.Addresses, nodeAddresses) {
		return
	}
	newNode := node.DeepCopy()
	newNode.Status.Addresses = nodeAddresses
	_, _, err = cloudnodeutil.PatchNodeStatus(cnc.kubeClient.CoreV1(), types.NodeName(node.Name), node, newNode)
	if err != nil {
		klog.Errorf("Error patching node with cloud ip addresses = [%v]", err)
	}
}

// nodeModifier is used to carry changes to node objects across multiple attempts to update them
// in a retry-if-conflict loop.
type nodeModifier func(*v1.Node)

func (cnc *CloudNodeController) UpdateCloudNode(ctx context.Context, _, newObj interface{}) {
	node, ok := newObj.(*v1.Node)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("unexpected object type: %v", newObj))
		return
	}

	cloudTaint := getCloudTaint(node.Spec.Taints)
	if cloudTaint == nil {
		// The node has already been initialized so nothing to do.
		return
	}

	cnc.initializeNode(ctx, node)
}

// AddCloudNode handles initializing new nodes registered with the cloud taint.
func (cnc *CloudNodeController) AddCloudNode(ctx context.Context, obj interface{}) {
	node := obj.(*v1.Node)

	cloudTaint := getCloudTaint(node.Spec.Taints)
	if cloudTaint == nil {
		klog.V(2).Infof("This node %s is registered without the cloud taint. Will not process.", node.Name)
		return
	}

	cnc.initializeNode(ctx, node)
}

// This processes nodes that were added into the cluster, and cloud initialize them if appropriate
func (cnc *CloudNodeController) initializeNode(ctx context.Context, node *v1.Node) {
	klog.Infof("Initializing node %s with cloud provider", node.Name)

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
			if err := cloudnodeutil.SetNodeCondition(cnc.kubeClient, types.NodeName(node.Name), v1.NodeCondition{
				Type:               v1.NodeNetworkUnavailable,
				Status:             v1.ConditionTrue,
				Reason:             "NoRouteCreated",
				Message:            "Node created without a route",
				LastTransitionTime: metav1.Now(),
			}); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		utilruntime.HandleError(err)
		return
	}

	curNode, err := cnc.kubeClient.CoreV1().Nodes().Get(context.TODO(), node.Name, metav1.GetOptions{})
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("failed to get node %s: %v", node.Name, err))
		return
	}

	cloudTaint := getCloudTaint(curNode.Spec.Taints)
	if cloudTaint == nil {
		// Node object received from event had the cloud taint but was outdated,
		// the node has actually already been initialized.
		return
	}

	nodeModifiers, err := cnc.getNodeModifiersFromCloudProvider(ctx, curNode, instances)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("failed to initialize node %s at cloudprovider: %v", node.Name, err))
		return
	}

	nodeModifiers = append(nodeModifiers, func(n *v1.Node) {
		n.Spec.Taints = excludeCloudTaint(n.Spec.Taints)
	})

	err = clientretry.RetryOnConflict(UpdateNodeSpecBackoff, func() error {
		curNode, err := cnc.kubeClient.CoreV1().Nodes().Get(context.TODO(), node.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}

		for _, modify := range nodeModifiers {
			modify(curNode)
		}

		_, err = cnc.kubeClient.CoreV1().Nodes().Update(context.TODO(), curNode, metav1.UpdateOptions{})
		if err != nil {
			return err
		}

		// After adding, call UpdateNodeAddress to set the CloudProvider provided IPAddresses
		// So that users do not see any significant delay in IP addresses being filled into the node
		cnc.updateNodeAddress(ctx, curNode, instances)

		klog.Infof("Successfully initialized node %s with cloud provider", node.Name)
		return nil
	})
	if err != nil {
		utilruntime.HandleError(err)
		return
	}
}

// getNodeModifiersFromCloudProvider returns a slice of nodeModifiers that update
// a node object with provider-specific information.
// All of the returned functions are idempotent, because they are used in a retry-if-conflict
// loop, meaning they could get called multiple times.
func (cnc *CloudNodeController) getNodeModifiersFromCloudProvider(ctx context.Context, node *v1.Node, instances cloudprovider.Instances) ([]nodeModifier, error) {
	var (
		nodeModifiers []nodeModifier
		providerID    string
		err           error
	)

	if node.Spec.ProviderID == "" {
		providerID, err = cloudprovider.GetInstanceProviderID(ctx, cnc.cloud, types.NodeName(node.Name))
		if err == nil {
			nodeModifiers = append(nodeModifiers, func(n *v1.Node) {
				if n.Spec.ProviderID == "" {
					n.Spec.ProviderID = providerID
				}
			})
		} else if err == cloudprovider.NotImplemented {
			// if the cloud provider being used does not support provider IDs,
			// we can safely continue since we will attempt to set node
			// addresses given the node name in getNodeAddressesByProviderIDOrName
			klog.Warningf("cloud provider does not set node provider ID, using node name to discover node %s", node.Name)
		} else {
			// if the cloud provider being used supports provider IDs, we want
			// to propagate the error so that we re-try in the future; if we
			// do not, the taint will be removed, and this will not be retried
			return nil, err
		}
	} else {
		providerID = node.Spec.ProviderID
	}

	nodeAddresses, err := getNodeAddressesByProviderIDOrName(ctx, instances, providerID, node.Name)
	if err != nil {
		return nil, err
	}

	// If user provided an IP address, ensure that IP address is found
	// in the cloud provider before removing the taint on the node
	if nodeIP, ok := ensureNodeProvidedIPExists(node, nodeAddresses); ok {
		if nodeIP == nil {
			return nil, errors.New("failed to find kubelet node IP from cloud provider")
		}
	}

	if instanceType, err := getInstanceTypeByProviderIDOrName(ctx, instances, providerID, node.Name); err != nil {
		return nil, err
	} else if instanceType != "" {
		klog.V(2).Infof("Adding node label from cloud provider: %s=%s", v1.LabelInstanceType, instanceType)
		klog.V(2).Infof("Adding node label from cloud provider: %s=%s", v1.LabelInstanceTypeStable, instanceType)
		nodeModifiers = append(nodeModifiers, func(n *v1.Node) {
			if n.Labels == nil {
				n.Labels = map[string]string{}
			}
			n.Labels[v1.LabelInstanceType] = instanceType
			n.Labels[v1.LabelInstanceTypeStable] = instanceType
		})
	}

	if zones, ok := cnc.cloud.Zones(); ok {
		zone, err := getZoneByProviderIDOrName(ctx, zones, providerID, node.Name)
		if err != nil {
			return nil, fmt.Errorf("failed to get zone from cloud provider: %v", err)
		}
		if zone.FailureDomain != "" {
			klog.V(2).Infof("Adding node label from cloud provider: %s=%s", v1.LabelZoneFailureDomain, zone.FailureDomain)
			klog.V(2).Infof("Adding node label from cloud provider: %s=%s", v1.LabelZoneFailureDomainStable, zone.FailureDomain)
			nodeModifiers = append(nodeModifiers, func(n *v1.Node) {
				if n.Labels == nil {
					n.Labels = map[string]string{}
				}
				n.Labels[v1.LabelZoneFailureDomain] = zone.FailureDomain
				n.Labels[v1.LabelZoneFailureDomainStable] = zone.FailureDomain
			})
		}
		if zone.Region != "" {
			klog.V(2).Infof("Adding node label from cloud provider: %s=%s", v1.LabelZoneRegion, zone.Region)
			klog.V(2).Infof("Adding node label from cloud provider: %s=%s", v1.LabelZoneRegionStable, zone.Region)
			nodeModifiers = append(nodeModifiers, func(n *v1.Node) {
				if n.Labels == nil {
					n.Labels = map[string]string{}
				}
				n.Labels[v1.LabelZoneRegion] = zone.Region
				n.Labels[v1.LabelZoneRegionStable] = zone.Region
			})
		}
	}
	return nodeModifiers, nil
}

func getCloudTaint(taints []v1.Taint) *v1.Taint {
	for _, taint := range taints {
		if taint.Key == cloudproviderapi.TaintExternalCloudProvider {
			return &taint
		}
	}
	return nil
}

func excludeCloudTaint(taints []v1.Taint) []v1.Taint {
	newTaints := []v1.Taint{}
	for _, taint := range taints {
		if taint.Key == cloudproviderapi.TaintExternalCloudProvider {
			continue
		}
		newTaints = append(newTaints, taint)
	}
	return newTaints
}

// ensureNodeExistsByProviderID checks if the instance exists by the provider id,
// If provider id in spec is empty it calls instanceId with node name to get provider id
func ensureNodeExistsByProviderID(ctx context.Context, instances cloudprovider.Instances, node *v1.Node) (bool, error) {
	providerID := node.Spec.ProviderID
	if providerID == "" {
		var err error
		providerID, err = instances.InstanceID(ctx, types.NodeName(node.Name))
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

	return instances.InstanceExistsByProviderID(ctx, providerID)
}

func getNodeAddressesByProviderIDOrName(ctx context.Context, instances cloudprovider.Instances, providerID, nodeName string) ([]v1.NodeAddress, error) {
	nodeAddresses, err := instances.NodeAddressesByProviderID(ctx, providerID)
	if err != nil {
		providerIDErr := err
		nodeAddresses, err = instances.NodeAddresses(ctx, types.NodeName(nodeName))
		if err != nil {
			return nil, fmt.Errorf("error fetching node by provider ID: %v, and error by node name: %v", providerIDErr, err)
		}
	}
	return nodeAddresses, nil
}

func nodeAddressesChangeDetected(addressSet1, addressSet2 []v1.NodeAddress) bool {
	if len(addressSet1) != len(addressSet2) {
		return true
	}
	addressMap1 := map[v1.NodeAddressType]string{}

	for i := range addressSet1 {
		addressMap1[addressSet1[i].Type] = addressSet1[i].Address
	}

	for _, v := range addressSet2 {
		if addressMap1[v.Type] != v.Address {
			return true
		}
	}
	return false
}

func ensureNodeProvidedIPExists(node *v1.Node, nodeAddresses []v1.NodeAddress) (*v1.NodeAddress, bool) {
	var nodeIP *v1.NodeAddress
	nodeIPExists := false
	if providedIP, ok := node.ObjectMeta.Annotations[cloudproviderapi.AnnotationAlphaProvidedIPAddr]; ok {
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

// getInstanceTypeByProviderIDOrName will attempt to get the instance type of node using its providerID
// then it's name. If both attempts fail, an error is returned.
func getInstanceTypeByProviderIDOrName(ctx context.Context, instances cloudprovider.Instances, providerID, nodeName string) (string, error) {
	instanceType, err := instances.InstanceTypeByProviderID(ctx, providerID)
	if err != nil {
		providerIDErr := err
		instanceType, err = instances.InstanceType(ctx, types.NodeName(nodeName))
		if err != nil {
			return "", fmt.Errorf("InstanceType: Error fetching by providerID: %v Error fetching by NodeName: %v", providerIDErr, err)
		}
	}
	return instanceType, err
}

// getZoneByProviderIDorName will attempt to get the zone of node using its providerID
// then it's name. If both attempts fail, an error is returned.
func getZoneByProviderIDOrName(ctx context.Context, zones cloudprovider.Zones, providerID, nodeName string) (cloudprovider.Zone, error) {
	zone, err := zones.GetZoneByProviderID(ctx, providerID)
	if err != nil {
		providerIDErr := err
		zone, err = zones.GetZoneByNodeName(ctx, types.NodeName(nodeName))
		if err != nil {
			return cloudprovider.Zone{}, fmt.Errorf("Zone: Error fetching by providerID: %v Error fetching by NodeName: %v", providerIDErr, err)
		}
	}

	return zone, nil
}
