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

package kubelet

import (
	"context"
	"fmt"
	"net"
	goruntime "runtime"
	"sort"
	"strings"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	cloudprovider "k8s.io/cloud-provider"
	cloudproviderapi "k8s.io/cloud-provider/api"
	"k8s.io/klog/v2"

	api "k8s.io/kubernetes/pkg/apis/core"
	k8s_api_v1 "k8s.io/kubernetes/pkg/apis/core/v1"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/nodestatus"
	"k8s.io/kubernetes/pkg/kubelet/util"
	"k8s.io/kubernetes/pkg/kubelet/volumemanager"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
	taintutil "k8s.io/kubernetes/pkg/util/taints"
	volutil "k8s.io/kubernetes/pkg/volume/util"
)

const (
	// nodeStatusUpdateRetry specifies how many times kubelet retries when posting node status failed.
	nodeStatusUpdateRetry = 5
)

// NodeStatusManager manage node status.
type NodeStatusManager struct {
	// hostname is kubelet hostname.
	hostname string
	// nodeName is kubelet nodeName.
	nodeName types.NodeName
	// clock records node status updated time.
	clock clock.Clock
	// registerNode Set to true means node register itself with the apiserver.
	registerNode bool
	// kubeClient used to register node.
	kubeClient clientset.Interface
	// heartbeatClient used to sync node status.
	heartbeatClient clientset.Interface
	// for internal book keeping; access only from within registerWithApiserver
	registrationCompleted bool
	// If non-nil, this is a unique identifier for the node in an external database, eg. cloudprovider
	providerID string
	// externalCloudProvider is kubelet externalCloudProvider.
	externalCloudProvider bool
	// containerManager is kubelet containerManager.
	containerManager cm.ContainerManager
	// volumeManager is kubelet volumeManager.
	volumeManager volumemanager.VolumeManager
	// cloud is kubelet cloud.
	cloud cloudprovider.Interface
	// syncNodeStatusMux is a lock on updating the node status, because this path is not thread-safe.
	// This lock is used by Kubelet.syncNodeStatus function and shouldn't be used anywhere else.
	syncNodeStatusMux sync.Mutex
	// nodeLabels is a list of node labels to register.
	nodeLabels map[string]string
	// Set to true to have the node register itself as schedulable.
	registerSchedulable bool
	// List of taints to add to a node object when the kubelet registers itself.
	registerWithTaints []api.Taint
	// enableControllerAttachDetach indicates the Attach/Detach controller
	// should manage attachment/detachment of volumes scheduled to this node,
	// and disable kubelet from executing any attach/detach operations
	enableControllerAttachDetach bool
	// onRepeatedHeartbeatFailure is called when a heartbeat operation fails more than once. optional.
	onRepeatedHeartbeatFailure func()
	// nodeStatusUpdateFrequency is kubelet nodeStatusUpdateFrequency.
	nodeStatusUpdateFrequency time.Duration
	// nodeStatusReportFrequency is the frequency that kubelet posts node
	// status to master. It is only used when node lease feature is enabled.
	nodeStatusReportFrequency time.Duration
	// lastStatusReportTime is the time when node status was last reported.
	lastStatusReportTime time.Time
	// handlers called during the tryUpdateNodeStatus cycle
	nodeStatusFuncs []func(*v1.Node) error
	// getNodeFunc is kubelet GetNode function.
	getNodeFunc func() (*v1.Node, error)
	// updateRuntimeUpFunc is kubelet updateRuntimeUp function.
	updateRuntimeUpFunc func()
	// updatePodCIDRFunc is kubelet updatePodCIDR function.
	updatePodCIDRFunc func(string) (bool, error)
	// setLastObservedNodeAddressesFunc is kubelet setLastObservedNodeAddresses function.
	setLastObservedNodeAddressesFunc func(addresses []v1.NodeAddress)
	// keepTerminatedPodVolumes is kubelet keepTerminatedPodVolumes.
	keepTerminatedPodVolumes bool
}

// NewNodeStatusManager build nodeStatus Manager.
// TODO (follow up work)move it and related tests to kubelet/nodestatus package.
func NewNodeStatusManager(
	hostname string,
	nodeName types.NodeName,
	clock clock.Clock,
	nodeLabels map[string]string,
	registerNode bool,
	kubeClient clientset.Interface,
	heartbeatClient clientset.Interface,
	providerID string,
	externalCloudProvider bool,
	containerManager cm.ContainerManager,
	volumeManager volumemanager.VolumeManager,
	cloud cloudprovider.Interface,
	registerSchedulable bool,
	registerWithTaints []api.Taint,
	enableControllerAttachDetach bool,
	onRepeatedHeartbeatFailure func(),
	nodeStatusUpdateFrequency time.Duration,
	nodeStatusReportFrequency time.Duration,
	keepTerminatedPodVolumes bool,
	getNodeFunc func() (*v1.Node, error),
	updateRuntimeUpFunc func(),
	updatePodCIDRFunc func(string) (bool, error),
	setLastObservedNodeAddressesFunc func(addresses []v1.NodeAddress),
	nodeStatusFuncs []func(*v1.Node) error,
) *NodeStatusManager {
	return &NodeStatusManager{
		hostname:                         hostname,
		nodeName:                         nodeName,
		nodeLabels:                       nodeLabels,
		clock:                            clock,
		registerNode:                     registerNode,
		kubeClient:                       kubeClient,
		heartbeatClient:                  heartbeatClient,
		providerID:                       providerID,
		externalCloudProvider:            externalCloudProvider,
		containerManager:                 containerManager,
		volumeManager:                    volumeManager,
		cloud:                            cloud,
		registerSchedulable:              registerSchedulable,
		registerWithTaints:               registerWithTaints,
		enableControllerAttachDetach:     enableControllerAttachDetach,
		onRepeatedHeartbeatFailure:       onRepeatedHeartbeatFailure,
		nodeStatusUpdateFrequency:        nodeStatusUpdateFrequency,
		nodeStatusReportFrequency:        nodeStatusReportFrequency,
		keepTerminatedPodVolumes:         keepTerminatedPodVolumes,
		getNodeFunc:                      getNodeFunc,
		updateRuntimeUpFunc:              updateRuntimeUpFunc,
		updatePodCIDRFunc:                updatePodCIDRFunc,
		setLastObservedNodeAddressesFunc: setLastObservedNodeAddressesFunc,
		nodeStatusFuncs:                  nodeStatusFuncs,
	}
}

// Run sync node status.
func (m *NodeStatusManager) Run(stopCh <-chan struct{}) {
	go m.fastStatusUpdateOnce()
	wait.Until(m.syncNodeStatus, m.nodeStatusUpdateFrequency, stopCh)
}

// fastStatusUpdateOnce starts a loop that checks the internal node indexer cache for when a CIDR
// is applied  and tries to update pod CIDR immediately. After pod CIDR is updated it fires off
// a runtime update and a node status update. Function returns after one successful node status update.
// Function is executed only during Kubelet start which improves latency to ready node by updating
// pod CIDR, runtime status and node statuses ASAP.
func (m *NodeStatusManager) fastStatusUpdateOnce() {
	for {
		time.Sleep(100 * time.Millisecond)
		node, err := m.getNodeFunc()
		if err != nil {
			klog.Errorf(err.Error())
			continue
		}
		if len(node.Spec.PodCIDRs) != 0 {
			podCIDRs := strings.Join(node.Spec.PodCIDRs, ",")
			if _, err := m.updatePodCIDRFunc(podCIDRs); err != nil {
				klog.Errorf("Pod CIDR update to %v failed %v", podCIDRs, err)
				continue
			}
			m.updateRuntimeUpFunc()
			m.syncNodeStatus()
			return
		}
	}
}

// registerWithAPIServer registers the node with the cluster master. It is safe
// to call multiple times, but not concurrently (kl.registrationCompleted is
// not locked).
func (m *NodeStatusManager) registerWithAPIServer() {
	if m.registrationCompleted {
		return
	}
	step := 100 * time.Millisecond

	for {
		time.Sleep(step)
		step = step * 2
		if step >= 7*time.Second {
			step = 7 * time.Second
		}

		node, err := m.InitialNode(context.TODO())
		if err != nil {
			klog.Errorf("Unable to construct v1.Node object for kubelet: %v", err)
			continue
		}

		klog.Infof("Attempting to register node %s", node.Name)
		registered := m.tryRegisterWithAPIServer(node)
		if registered {
			klog.Infof("Successfully registered node %s", node.Name)
			m.registrationCompleted = true
			return
		}
	}
}

// tryRegisterWithAPIServer makes an attempt to register the given node with
// the API server, returning a boolean indicating whether the attempt was
// successful.  If a node with the same name already exists, it reconciles the
// value of the annotation for controller-managed attach-detach of attachable
// persistent volumes for the node.
func (m *NodeStatusManager) tryRegisterWithAPIServer(node *v1.Node) bool {
	_, err := m.kubeClient.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{})
	if err == nil {
		return true
	}

	if !apierrors.IsAlreadyExists(err) {
		klog.Errorf("Unable to register node %q with API server: %v", m.nodeName, err)
		return false
	}

	existingNode, err := m.kubeClient.CoreV1().Nodes().Get(context.TODO(), string(m.nodeName), metav1.GetOptions{})
	if err != nil {
		klog.Errorf("Unable to register node %q with API server: error getting existing node: %v", m.nodeName, err)
		return false
	}
	if existingNode == nil {
		klog.Errorf("Unable to register node %q with API server: no node instance returned", m.nodeName)
		return false
	}

	originalNode := existingNode.DeepCopy()
	if originalNode == nil {
		klog.Errorf("Nil %q node object", m.nodeName)
		return false
	}

	klog.Infof("Node %s was previously registered", m.nodeName)

	// Edge case: the node was previously registered; reconcile
	// the value of the controller-managed attach-detach
	// annotation.
	requiresUpdate := reconcileCMADAnnotationWithExistingNode(node, existingNode)
	requiresUpdate = updateDefaultLabels(node, existingNode) || requiresUpdate
	requiresUpdate = m.reconcileExtendedResource(node, existingNode) || requiresUpdate
	requiresUpdate = reconcileHugePageResource(node, existingNode) || requiresUpdate
	if requiresUpdate {
		if _, _, err := nodeutil.PatchNodeStatus(m.kubeClient.CoreV1(), types.NodeName(m.nodeName), originalNode, existingNode); err != nil {
			klog.Errorf("Unable to reconcile node %q with API server: error updating node: %v", m.nodeName, err)
			return false
		}
	}

	return true
}

// reconcileHugePageResource will update huge page capacity for each page size and remove huge page sizes no longer supported
func reconcileHugePageResource(initialNode, existingNode *v1.Node) bool {
	requiresUpdate := false
	supportedHugePageResources := sets.String{}

	for resourceName := range initialNode.Status.Capacity {
		if !v1helper.IsHugePageResourceName(resourceName) {
			continue
		}
		supportedHugePageResources.Insert(string(resourceName))

		initialCapacity := initialNode.Status.Capacity[resourceName]
		initialAllocatable := initialNode.Status.Allocatable[resourceName]

		capacity, resourceIsSupported := existingNode.Status.Capacity[resourceName]
		allocatable := existingNode.Status.Allocatable[resourceName]

		// Add or update capacity if it the size was previously unsupported or has changed
		if !resourceIsSupported || capacity.Cmp(initialCapacity) != 0 {
			existingNode.Status.Capacity[resourceName] = initialCapacity.DeepCopy()
			requiresUpdate = true
		}

		// Add or update allocatable if it the size was previously unsupported or has changed
		if !resourceIsSupported || allocatable.Cmp(initialAllocatable) != 0 {
			existingNode.Status.Allocatable[resourceName] = initialAllocatable.DeepCopy()
			requiresUpdate = true
		}

	}

	for resourceName := range existingNode.Status.Capacity {
		if !v1helper.IsHugePageResourceName(resourceName) {
			continue
		}

		// If huge page size no longer is supported, we remove it from the node
		if !supportedHugePageResources.Has(string(resourceName)) {
			delete(existingNode.Status.Capacity, resourceName)
			delete(existingNode.Status.Allocatable, resourceName)
			klog.Infof("Removing now unsupported huge page resource named: %s", resourceName)
			requiresUpdate = true
		}
	}
	return requiresUpdate
}

// Zeros out extended resource capacity during reconciliation.
func (m *NodeStatusManager) reconcileExtendedResource(initialNode, node *v1.Node) bool {
	requiresUpdate := false
	// Check with the device manager to see if node has been recreated, in which case extended resources should be zeroed until they are available
	if m.containerManager.ShouldResetExtendedResourceCapacity() {
		for k := range node.Status.Capacity {
			if v1helper.IsExtendedResourceName(k) {
				klog.Infof("Zero out resource %s capacity in existing node.", k)
				node.Status.Capacity[k] = *resource.NewQuantity(int64(0), resource.DecimalSI)
				node.Status.Allocatable[k] = *resource.NewQuantity(int64(0), resource.DecimalSI)
				requiresUpdate = true
			}
		}
	}
	return requiresUpdate
}

// updateDefaultLabels will set the default labels on the node
func updateDefaultLabels(initialNode, existingNode *v1.Node) bool {
	defaultLabels := []string{
		v1.LabelHostname,
		v1.LabelZoneFailureDomainStable,
		v1.LabelZoneRegionStable,
		v1.LabelZoneFailureDomain,
		v1.LabelZoneRegion,
		v1.LabelInstanceTypeStable,
		v1.LabelInstanceType,
		v1.LabelOSStable,
		v1.LabelArchStable,
		v1.LabelWindowsBuild,
	}

	needsUpdate := false
	if existingNode.Labels == nil {
		existingNode.Labels = make(map[string]string)
	}
	//Set default labels but make sure to not set labels with empty values
	for _, label := range defaultLabels {
		if _, hasInitialValue := initialNode.Labels[label]; !hasInitialValue {
			continue
		}

		if existingNode.Labels[label] != initialNode.Labels[label] {
			existingNode.Labels[label] = initialNode.Labels[label]
			needsUpdate = true
		}

		if existingNode.Labels[label] == "" {
			delete(existingNode.Labels, label)
		}
	}

	return needsUpdate
}

// reconcileCMADAnnotationWithExistingNode reconciles the controller-managed
// attach-detach annotation on a new node and the existing node, returning
// whether the existing node must be updated.
func reconcileCMADAnnotationWithExistingNode(node, existingNode *v1.Node) bool {
	var (
		existingCMAAnnotation    = existingNode.Annotations[volutil.ControllerManagedAttachAnnotation]
		newCMAAnnotation, newSet = node.Annotations[volutil.ControllerManagedAttachAnnotation]
	)

	if newCMAAnnotation == existingCMAAnnotation {
		return false
	}

	// If the just-constructed node and the existing node do
	// not have the same value, update the existing node with
	// the correct value of the annotation.
	if !newSet {
		klog.Info("Controller attach-detach setting changed to false; updating existing Node")
		delete(existingNode.Annotations, volutil.ControllerManagedAttachAnnotation)
	} else {
		klog.Info("Controller attach-detach setting changed to true; updating existing Node")
		if existingNode.Annotations == nil {
			existingNode.Annotations = make(map[string]string)
		}
		existingNode.Annotations[volutil.ControllerManagedAttachAnnotation] = newCMAAnnotation
	}

	return true
}

// InitialNode constructs the initial v1.Node for this Kubelet, incorporating node
// labels, information from the cloud provider, and Kubelet configuration.
func (m *NodeStatusManager) InitialNode(ctx context.Context) (*v1.Node, error) {
	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: string(m.nodeName),
			Labels: map[string]string{
				v1.LabelHostname:   m.hostname,
				v1.LabelOSStable:   goruntime.GOOS,
				v1.LabelArchStable: goruntime.GOARCH,
			},
		},
		Spec: v1.NodeSpec{
			Unschedulable: !m.registerSchedulable,
		},
	}
	osLabels, err := getOSSpecificLabels()
	if err != nil {
		return nil, err
	}
	for label, value := range osLabels {
		node.Labels[label] = value
	}

	nodeTaints := make([]v1.Taint, 0)
	if len(m.registerWithTaints) > 0 {
		taints := make([]v1.Taint, len(m.registerWithTaints))
		for i := range m.registerWithTaints {
			if err := k8s_api_v1.Convert_core_Taint_To_v1_Taint(&m.registerWithTaints[i], &taints[i], nil); err != nil {
				return nil, err
			}
		}
		nodeTaints = append(nodeTaints, taints...)
	}

	unschedulableTaint := v1.Taint{
		Key:    v1.TaintNodeUnschedulable,
		Effect: v1.TaintEffectNoSchedule,
	}

	// Taint node with TaintNodeUnschedulable when initializing
	// node to avoid race condition; refer to #63897 for more detail.
	if node.Spec.Unschedulable &&
		!taintutil.TaintExists(nodeTaints, &unschedulableTaint) {
		nodeTaints = append(nodeTaints, unschedulableTaint)
	}

	if m.externalCloudProvider {
		taint := v1.Taint{
			Key:    cloudproviderapi.TaintExternalCloudProvider,
			Value:  "true",
			Effect: v1.TaintEffectNoSchedule,
		}

		nodeTaints = append(nodeTaints, taint)
	}
	if len(nodeTaints) > 0 {
		node.Spec.Taints = nodeTaints
	}
	// Initially, set NodeNetworkUnavailable to true.
	if providerRequiresNetworkingConfiguration(m.cloud) {
		node.Status.Conditions = append(node.Status.Conditions, v1.NodeCondition{
			Type:               v1.NodeNetworkUnavailable,
			Status:             v1.ConditionTrue,
			Reason:             "NoRouteCreated",
			Message:            "Node created without a route",
			LastTransitionTime: metav1.NewTime(m.clock.Now()),
		})
	}

	if m.enableControllerAttachDetach {
		if node.Annotations == nil {
			node.Annotations = make(map[string]string)
		}

		klog.V(2).Infof("Setting node annotation to enable volume controller attach/detach")
		node.Annotations[volutil.ControllerManagedAttachAnnotation] = "true"
	} else {
		klog.V(2).Infof("Controller attach/detach is disabled for this node; Kubelet will attach and detach volumes")
	}

	if m.keepTerminatedPodVolumes {
		if node.Annotations == nil {
			node.Annotations = make(map[string]string)
		}
		klog.V(2).Infof("Setting node annotation to keep pod volumes of terminated pods attached to the node")
		node.Annotations[volutil.KeepTerminatedPodVolumesAnnotation] = "true"
	}

	// @question: should this be place after the call to the cloud provider? which also applies labels
	for k, v := range m.nodeLabels {
		if cv, found := node.ObjectMeta.Labels[k]; found {
			klog.Warningf("the node label %s=%s will overwrite default setting %s", k, v, cv)
		}
		node.ObjectMeta.Labels[k] = v
	}

	if m.providerID != "" {
		node.Spec.ProviderID = m.providerID
	}

	if m.cloud != nil {
		instances, ok := m.cloud.Instances()
		if !ok {
			return nil, fmt.Errorf("failed to get instances from cloud provider")
		}

		// TODO: We can't assume that the node has credentials to talk to the
		// cloudprovider from arbitrary nodes. At most, we should talk to a
		// local metadata server here.
		var err error
		if node.Spec.ProviderID == "" {
			node.Spec.ProviderID, err = cloudprovider.GetInstanceProviderID(ctx, m.cloud, m.nodeName)
			if err != nil {
				return nil, err
			}
		}

		instanceType, err := instances.InstanceType(ctx, m.nodeName)
		if err != nil {
			return nil, err
		}
		if instanceType != "" {
			klog.Infof("Adding node label from cloud provider: %s=%s", v1.LabelInstanceType, instanceType)
			node.ObjectMeta.Labels[v1.LabelInstanceType] = instanceType
			klog.Infof("Adding node label from cloud provider: %s=%s", v1.LabelInstanceTypeStable, instanceType)
			node.ObjectMeta.Labels[v1.LabelInstanceTypeStable] = instanceType
		}
		// If the cloud has zone information, label the node with the zone information
		zones, ok := m.cloud.Zones()
		if ok {
			zone, err := zones.GetZone(ctx)
			if err != nil {
				return nil, fmt.Errorf("failed to get zone from cloud provider: %v", err)
			}
			if zone.FailureDomain != "" {
				klog.Infof("Adding node label from cloud provider: %s=%s", v1.LabelZoneFailureDomain, zone.FailureDomain)
				node.ObjectMeta.Labels[v1.LabelZoneFailureDomain] = zone.FailureDomain
				klog.Infof("Adding node label from cloud provider: %s=%s", v1.LabelZoneFailureDomainStable, zone.FailureDomain)
				node.ObjectMeta.Labels[v1.LabelZoneFailureDomainStable] = zone.FailureDomain
			}
			if zone.Region != "" {
				klog.Infof("Adding node label from cloud provider: %s=%s", v1.LabelZoneRegion, zone.Region)
				node.ObjectMeta.Labels[v1.LabelZoneRegion] = zone.Region
				klog.Infof("Adding node label from cloud provider: %s=%s", v1.LabelZoneRegionStable, zone.Region)
				node.ObjectMeta.Labels[v1.LabelZoneRegionStable] = zone.Region
			}
		}
	}

	m.setNodeStatus(node)

	return node, nil
}

// SyncNodeStatus should be called periodically from a goroutine.
// It synchronizes node status to master if there is any change or enough time
// passed from the last sync, registering the kubelet first if necessary.
func (m *NodeStatusManager) syncNodeStatus() {
	m.syncNodeStatusMux.Lock()
	defer m.syncNodeStatusMux.Unlock()

	if m.kubeClient == nil || m.heartbeatClient == nil {
		return
	}
	if m.registerNode {
		// This will exit immediately if it doesn't need to do anything.
		m.registerWithAPIServer()
	}
	if err := m.UpdateNodeStatus(); err != nil {
		klog.Errorf("Unable to update node status: %v", err)
	}
}

// UpdateNodeStatus updates node status to master with retries if there is any
// change or enough time passed from the last sync.
// Make UpdateNodeStatus public to be better tests with current kubelet defaultNodeStatusFuncs.
func (m *NodeStatusManager) UpdateNodeStatus() error {
	klog.V(5).Infof("Updating node status")
	for i := 0; i < nodeStatusUpdateRetry; i++ {
		if err := m.tryUpdateNodeStatus(i); err != nil {
			if i > 0 && m.onRepeatedHeartbeatFailure != nil {
				m.onRepeatedHeartbeatFailure()
			}
			klog.Errorf("Error updating node status, will retry: %v", err)
		} else {
			return nil
		}
	}
	return fmt.Errorf("update node status exceeds retry count")
}

// tryUpdateNodeStatus tries to update node status to master if there is any
// change or enough time passed from the last sync.
func (m *NodeStatusManager) tryUpdateNodeStatus(tryNumber int) error {
	// In large clusters, GET and PUT operations on Node objects coming
	// from here are the majority of load on apiserver and etcd.
	// To reduce the load on etcd, we are serving GET operations from
	// apiserver cache (the data might be slightly delayed but it doesn't
	// seem to cause more conflict - the delays are pretty small).
	// If it result in a conflict, all retries are served directly from etcd.
	opts := metav1.GetOptions{}
	if tryNumber == 0 {
		util.FromApiserverCache(&opts)
	}
	node, err := m.heartbeatClient.CoreV1().Nodes().Get(context.TODO(), string(m.nodeName), opts)
	if err != nil {
		return fmt.Errorf("error getting node %q: %v", m.nodeName, err)
	}

	originalNode := node.DeepCopy()
	if originalNode == nil {
		return fmt.Errorf("nil %q node object", m.nodeName)
	}

	podCIDRChanged := false
	if len(node.Spec.PodCIDRs) != 0 {
		// Pod CIDR could have been updated before, so we cannot rely on
		// node.Spec.PodCIDR being non-empty. We also need to know if pod CIDR is
		// actually changed.
		podCIDRs := strings.Join(node.Spec.PodCIDRs, ",")
		if podCIDRChanged, err = m.updatePodCIDRFunc(podCIDRs); err != nil {
			klog.Errorf(err.Error())
		}
	}

	m.setNodeStatus(node)

	now := m.clock.Now()
	if now.Before(m.lastStatusReportTime.Add(m.nodeStatusReportFrequency)) {
		if !podCIDRChanged && !nodeStatusHasChanged(&originalNode.Status, &node.Status) {
			// We must mark the volumes as ReportedInUse in volume manager's dsw even
			// if no changes were made to the node status (no volumes were added or removed
			// from the VolumesInUse list).
			//
			// The reason is that on a kubelet restart, the volume manager's dsw is
			// repopulated and the volume ReportedInUse is initialized to false, while the
			// VolumesInUse list from the Node object still contains the state from the
			// previous kubelet instantiation.
			//
			// Once the volumes are added to the dsw, the ReportedInUse field needs to be
			// synced from the VolumesInUse list in the Node.Status.
			//
			// The MarkVolumesAsReportedInUse() call cannot be performed in dsw directly
			// because it does not have access to the Node object.
			// This also cannot be populated on node status manager init because the volume
			// may not have been added to dsw at that time.
			m.volumeManager.MarkVolumesAsReportedInUse(node.Status.VolumesInUse)
			return nil
		}
	}

	// Patch the current status on the API server
	updatedNode, _, err := nodeutil.PatchNodeStatus(m.heartbeatClient.CoreV1(), m.nodeName, originalNode, node)
	if err != nil {
		return err
	}
	m.lastStatusReportTime = now
	m.setLastObservedNodeAddressesFunc(updatedNode.Status.Addresses)
	// If update finishes successfully, mark the volumeInUse as reportedInUse to indicate
	// those volumes are already updated in the node's status
	m.volumeManager.MarkVolumesAsReportedInUse(updatedNode.Status.VolumesInUse)
	return nil
}

// recordNodeStatusEvent records an event of the given type with the given
// message for the node.
func (kl *Kubelet) recordNodeStatusEvent(eventType, event string) {
	klog.V(2).Infof("Recording %s event message for node %s", event, kl.nodeName)
	// TODO: This requires a transaction, either both node status is updated
	// and event is recorded or neither should happen, see issue #6055.
	kl.recorder.Eventf(kl.nodeRef, eventType, event, "Node %s status is now: %s", kl.nodeName, event)
}

// recordEvent records an event for this node, the Kubelet's nodeRef is passed to the recorder
func (kl *Kubelet) recordEvent(eventType, event, message string) {
	kl.recorder.Eventf(kl.nodeRef, eventType, event, message)
}

// record if node schedulable change.
func (kl *Kubelet) recordNodeSchedulableEvent(node *v1.Node) error {
	kl.lastNodeUnschedulableLock.Lock()
	defer kl.lastNodeUnschedulableLock.Unlock()
	if kl.lastNodeUnschedulable != node.Spec.Unschedulable {
		if node.Spec.Unschedulable {
			kl.recordNodeStatusEvent(v1.EventTypeNormal, events.NodeNotSchedulable)
		} else {
			kl.recordNodeStatusEvent(v1.EventTypeNormal, events.NodeSchedulable)
		}
		kl.lastNodeUnschedulable = node.Spec.Unschedulable
	}
	return nil
}

// setNodeStatus fills in the Status fields of the given Node, overwriting
// any fields that are currently set.
// TODO(madhusudancs): Simplify the logic for setting node conditions and
// refactor the node status condition code out to a different file.
func (m *NodeStatusManager) setNodeStatus(node *v1.Node) {
	for i, f := range m.nodeStatusFuncs {
		klog.V(5).Infof("Setting node status at position %v", i)
		if err := f(node); err != nil {
			klog.Errorf("Failed to set some node status fields: %s", err)
		}
	}
}

func (kl *Kubelet) setLastObservedNodeAddresses(addresses []v1.NodeAddress) {
	kl.lastObservedNodeAddressesMux.Lock()
	defer kl.lastObservedNodeAddressesMux.Unlock()
	kl.lastObservedNodeAddresses = addresses
}
func (kl *Kubelet) getLastObservedNodeAddresses() []v1.NodeAddress {
	kl.lastObservedNodeAddressesMux.RLock()
	defer kl.lastObservedNodeAddressesMux.RUnlock()
	return kl.lastObservedNodeAddresses
}

// defaultNodeStatusFuncs is a factory that generates the default set of
// setNodeStatus funcs
func (kl *Kubelet) defaultNodeStatusFuncs() []func(*v1.Node) error {
	// if cloud is not nil, we expect the cloud resource sync manager to exist
	var nodeAddressesFunc func() ([]v1.NodeAddress, error)
	if kl.cloud != nil {
		nodeAddressesFunc = kl.cloudResourceSyncManager.NodeAddresses
	}
	var validateHostFunc func() error
	if kl.appArmorValidator != nil {
		validateHostFunc = kl.appArmorValidator.ValidateHost
	}
	var setters []func(n *v1.Node) error
	setters = append(setters,
		nodestatus.NodeAddress(kl.nodeIP, ValidateNodeIP, kl.hostname, kl.hostnameOverridden, kl.externalCloudProvider, kl.cloud, nodeAddressesFunc),
		nodestatus.MachineInfo(string(kl.nodeName), kl.maxPods, kl.podsPerCore, kl.GetCachedMachineInfo, kl.containerManager.GetCapacity,
			kl.containerManager.GetDevicePluginResourceCapacity, kl.containerManager.GetNodeAllocatableReservation, kl.recordEvent),
		nodestatus.VersionInfo(kl.cadvisor.VersionInfo, kl.containerRuntime.Type, kl.containerRuntime.Version),
		nodestatus.DaemonEndpoints(kl.daemonEndpoints),
		nodestatus.Images(kl.nodeStatusMaxImages, kl.imageManager.GetImageList),
		nodestatus.GoRuntime(),
	)
	// Volume limits
	setters = append(setters, nodestatus.VolumeLimits(kl.volumePluginMgr.ListVolumePluginWithLimits))

	setters = append(setters,
		nodestatus.MemoryPressureCondition(kl.clock.Now, kl.evictionManager.IsUnderMemoryPressure, kl.recordNodeStatusEvent),
		nodestatus.DiskPressureCondition(kl.clock.Now, kl.evictionManager.IsUnderDiskPressure, kl.recordNodeStatusEvent),
		nodestatus.PIDPressureCondition(kl.clock.Now, kl.evictionManager.IsUnderPIDPressure, kl.recordNodeStatusEvent),
		nodestatus.ReadyCondition(kl.clock.Now, kl.runtimeState.runtimeErrors, kl.runtimeState.networkErrors, kl.runtimeState.storageErrors, validateHostFunc, kl.containerManager.Status, kl.recordNodeStatusEvent),
		nodestatus.VolumesInUse(kl.volumeManager.ReconcilerStatesHasBeenSynced, kl.volumeManager.GetVolumesInUse),
		// TODO(mtaufen): I decided not to move this setter for now, since all it does is send an event
		// and record state back to the Kubelet runtime object. In the future, I'd like to isolate
		// these side-effects by decoupling the decisions to send events and partial status recording
		// from the Node setters.
		kl.recordNodeSchedulableEvent,
	)
	return setters
}

// ValidateNodeIP validate given node IP belongs to the current host.
func ValidateNodeIP(nodeIP net.IP) error {
	// Honor IP limitations set in setNodeStatus()
	if nodeIP.To4() == nil && nodeIP.To16() == nil {
		return fmt.Errorf("nodeIP must be a valid IP address")
	}
	if nodeIP.IsLoopback() {
		return fmt.Errorf("nodeIP can't be loopback address")
	}
	if nodeIP.IsMulticast() {
		return fmt.Errorf("nodeIP can't be a multicast address")
	}
	if nodeIP.IsLinkLocalUnicast() {
		return fmt.Errorf("nodeIP can't be a link-local unicast address")
	}
	if nodeIP.IsUnspecified() {
		return fmt.Errorf("nodeIP can't be an all zeros address")
	}

	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return err
	}
	for _, addr := range addrs {
		var ip net.IP
		switch v := addr.(type) {
		case *net.IPNet:
			ip = v.IP
		case *net.IPAddr:
			ip = v.IP
		}
		if ip != nil && ip.Equal(nodeIP) {
			return nil
		}
	}
	return fmt.Errorf("node IP: %q not found in the host's network interfaces", nodeIP.String())
}

// nodeStatusHasChanged compares the original node and current node's status and
// returns true if any change happens. The heartbeat timestamp is ignored.
func nodeStatusHasChanged(originalStatus *v1.NodeStatus, status *v1.NodeStatus) bool {
	if originalStatus == nil && status == nil {
		return false
	}
	if originalStatus == nil || status == nil {
		return true
	}

	// Compare node conditions here because we need to ignore the heartbeat timestamp.
	if nodeConditionsHaveChanged(originalStatus.Conditions, status.Conditions) {
		return true
	}

	// Compare other fields of NodeStatus.
	originalStatusCopy := originalStatus.DeepCopy()
	statusCopy := status.DeepCopy()
	originalStatusCopy.Conditions = nil
	statusCopy.Conditions = nil
	return !apiequality.Semantic.DeepEqual(originalStatusCopy, statusCopy)
}

// nodeConditionsHaveChanged compares the original node and current node's
// conditions and returns true if any change happens. The heartbeat timestamp is
// ignored.
func nodeConditionsHaveChanged(originalConditions []v1.NodeCondition, conditions []v1.NodeCondition) bool {
	if len(originalConditions) != len(conditions) {
		return true
	}

	originalConditionsCopy := make([]v1.NodeCondition, 0, len(originalConditions))
	originalConditionsCopy = append(originalConditionsCopy, originalConditions...)
	conditionsCopy := make([]v1.NodeCondition, 0, len(conditions))
	conditionsCopy = append(conditionsCopy, conditions...)

	sort.SliceStable(originalConditionsCopy, func(i, j int) bool { return originalConditionsCopy[i].Type < originalConditionsCopy[j].Type })
	sort.SliceStable(conditionsCopy, func(i, j int) bool { return conditionsCopy[i].Type < conditionsCopy[j].Type })

	replacedheartbeatTime := metav1.Time{}
	for i := range conditionsCopy {
		originalConditionsCopy[i].LastHeartbeatTime = replacedheartbeatTime
		conditionsCopy[i].LastHeartbeatTime = replacedheartbeatTime
		if !apiequality.Semantic.DeepEqual(&originalConditionsCopy[i], &conditionsCopy[i]) {
			return true
		}
	}
	return false
}

// providerRequiresNetworkingConfiguration returns whether the cloud provider
// requires special networking configuration.
func providerRequiresNetworkingConfiguration(cloud cloudprovider.Interface) bool {
	// TODO: We should have a mechanism to say whether native cloud provider
	// is used or whether we are using overlay networking. We should return
	// true for cloud providers if they implement Routes() interface and
	// we are not using overlay networking.
	if cloud == nil || cloud.ProviderName() != "gce" {
		return false
	}
	_, supported := cloud.Routes()
	return supported
}
