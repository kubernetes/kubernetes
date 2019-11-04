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

package nodestatusmanager

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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/record"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/klog"
	api "k8s.io/kubernetes/pkg/apis/core"
	k8s_api_v1 "k8s.io/kubernetes/pkg/apis/core/v1"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/features"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/eviction"
	"k8s.io/kubernetes/pkg/kubelet/images"
	"k8s.io/kubernetes/pkg/kubelet/util"
	"k8s.io/kubernetes/pkg/kubelet/volumemanager"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
	taintutil "k8s.io/kubernetes/pkg/util/taints"
	"k8s.io/kubernetes/pkg/volume"
	volutil "k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/utils/clock"
)

const (
	// nodeStatusUpdateRetry specifies how many times kubelet retries when posting node status failed.
	nodeStatusUpdateRetry = 5
)

type NodeStatusManager struct {
	// hostname is the hostname the kubelet detected or was given via flag/config
	hostname                     string
	lastObservedNodeAddressesMux sync.RWMutex
	lastObservedNodeAddresses    []v1.NodeAddress

	nodeName        types.NodeName
	heartbeatClient clientset.Interface
	kubeClient      clientset.Interface

	// This flag, if set, instructs the kubelet to keep volumes from terminated pods mounted to the node.
	// This can be useful for debugging volume related issues.
	keepTerminatedPodVolumes bool // DEPRECATED

	// enableControllerAttachDetach indicates the Attach/Detach controller
	// should manage attachment/detachment of volumes scheduled to this node,
	// and disable kubelet from executing any attach/detach operations
	enableControllerAttachDetach bool

	// Volume plugins.
	volumePluginMgr *volume.VolumePluginMgr

	// Manager of non-Runtime containers.
	containerManager cm.ContainerManager

	// Needed to observe and respond to situations that could impact node stability
	evictionManager eviction.Manager

	// Manager for image garbage collection.
	imageManager images.ImageGCManager

	// Set to true to have the node register itself with the apiserver.
	registerNode bool
	// List of taints to add to a node object when the kubelet registers itself.
	registerWithTaints []api.Taint
	// Set to true to have the node register itself as schedulable.
	registerSchedulable bool
	// for internal book keeping; access only from within registerWithApiserver
	registrationCompleted bool

	// Indicates that the node initialization happens in an external cloud controller
	externalCloudProvider bool

	// Reference to this node.
	nodeRef *v1.ObjectReference

	// VolumeManager runs a set of asynchronous loops that figure out which
	// volumes need to be attached/mounted/unmounted/detached based on the pods
	// scheduled on this node and makes it so.
	volumeManager volumemanager.VolumeManager

	// The EventRecorder to use
	recorder record.EventRecorder

	// Cloud provider interface.
	cloud cloudprovider.Interface

	// If non-nil, use this IP address for the node
	nodeIP net.IP

	// nodeStatusReportFrequency is the frequency that kubelet posts node
	// status to master. It is only used when node lease feature is enabled.
	nodeStatusReportFrequency time.Duration

	// lastStatusReportTime is the time when node status was last reported.
	lastStatusReportTime time.Time

	// syncNodeStatusMux is a lock on updating the node status, because this path is not thread-safe.
	// This lock is used by Kubelet.syncNodeStatus function and shouldn't be used anywhere else.
	syncNodeStatusMux sync.Mutex

	// If non-nil, this is a unique identifier for the node in an external database, eg. cloudprovider
	providerID string

	// clock is an interface that provides time related functionality in a way that makes it
	// easy to test the code.
	clock clock.Clock

	// handlers called during the tryUpdateNodeStatus cycle
	setNodeStatusFuncs []func(*v1.Node) error

	lastNodeUnschedulableLock sync.Mutex
	// maintains Node.Spec.Unschedulable value from previous run of tryUpdateNodeStatus()
	lastNodeUnschedulable bool

	// onRepeatedHeartbeatFailure is called when a heartbeat operation fails more than once. optional.
	onRepeatedHeartbeatFailure func()

	// a list of node labels to register
	nodeLabels map[string]string

	// TODO(dranveness): remove the following fields to break the circular dependency with the kubelet.

	// updatePodCIDR updates the pod CIDR in the runtime state if it is different
	// from the current CIDR. Return true if pod CIDR is actually changed.
	updatePodCIDR func(cidr string) (bool, error)
}

// NewNodeStatusManager returns a node status manager.
func NewNodeStatusManager(
	hostname string,
	nodeName types.NodeName,
	heartbeatClient clientset.Interface,
	kubeClient clientset.Interface,
	keepTerminatedPodVolumes bool, // DEPRECATED
	enableControllerAttachDetach bool,
	containerManager cm.ContainerManager,
	evictionManager eviction.Manager,
	imageManager images.ImageGCManager,
	registerNode bool,
	registerWithTaints []api.Taint,
	registerSchedulable bool,
	externalCloudProvider bool,
	nodeRef *v1.ObjectReference,
	recorder record.EventRecorder,
	cloud cloudprovider.Interface,
	nodeIP net.IP,
	nodeStatusReportFrequency time.Duration,
	providerID string,
	onRepeatedHeartbeatFailure func(),
	nodeLabels map[string]string,
	volumePluginMgr *volume.VolumePluginMgr,
	volumeManager volumemanager.VolumeManager,
	setNodeStatusFuncs []func(*v1.Node) error,
	updatePodCIDR func(cidr string) (bool, error),
) (*NodeStatusManager, error) {
	return &NodeStatusManager{
		hostname:                     hostname,
		nodeName:                     nodeName,
		heartbeatClient:              heartbeatClient,
		kubeClient:                   kubeClient,
		keepTerminatedPodVolumes:     keepTerminatedPodVolumes, // DEPRECATED
		enableControllerAttachDetach: enableControllerAttachDetach,
		volumePluginMgr:              volumePluginMgr,
		containerManager:             containerManager,
		evictionManager:              evictionManager,
		imageManager:                 imageManager,
		registerNode:                 registerNode,
		registerWithTaints:           registerWithTaints,
		registerSchedulable:          registerSchedulable,
		externalCloudProvider:        externalCloudProvider,
		nodeRef:                      nodeRef,
		volumeManager:                volumeManager,
		recorder:                     recorder,
		cloud:                        cloud,
		nodeIP:                       nodeIP,
		nodeStatusReportFrequency:    nodeStatusReportFrequency,
		providerID:                   providerID,
		clock:                        clock.RealClock{},
		setNodeStatusFuncs:           setNodeStatusFuncs,
		onRepeatedHeartbeatFailure:   onRepeatedHeartbeatFailure,
		nodeLabels:                   nodeLabels,
		updatePodCIDR:                updatePodCIDR,
	}, nil
}

// registerWithAPIServer registers the node with the cluster master. It is safe
// to call multiple times, but not concurrently (kl.registrationCompleted is
// not locked).
func (kl *NodeStatusManager) registerWithAPIServer() {
	if kl.registrationCompleted {
		return
	}
	step := 100 * time.Millisecond

	for {
		time.Sleep(step)
		step = step * 2
		if step >= 7*time.Second {
			step = 7 * time.Second
		}

		node, err := kl.initialNode(context.TODO())
		if err != nil {
			klog.Errorf("Unable to construct v1.Node object for kubelet: %v", err)
			continue
		}

		klog.Infof("Attempting to register node %s", node.Name)
		registered := kl.tryRegisterWithAPIServer(node)
		if registered {
			klog.Infof("Successfully registered node %s", node.Name)
			kl.registrationCompleted = true
			return
		}
	}
}

// tryRegisterWithAPIServer makes an attempt to register the given node with
// the API server, returning a boolean indicating whether the attempt was
// successful.  If a node with the same name already exists, it reconciles the
// value of the annotation for controller-managed attach-detach of attachable
// persistent volumes for the node.
func (kl *NodeStatusManager) tryRegisterWithAPIServer(node *v1.Node) bool {
	_, err := kl.kubeClient.CoreV1().Nodes().Create(node)
	if err == nil {
		return true
	}

	if !apierrors.IsAlreadyExists(err) {
		klog.Errorf("Unable to register node %q with API server: %v", kl.nodeName, err)
		return false
	}

	existingNode, err := kl.kubeClient.CoreV1().Nodes().Get(string(kl.nodeName), metav1.GetOptions{})
	if err != nil {
		klog.Errorf("Unable to register node %q with API server: error getting existing node: %v", kl.nodeName, err)
		return false
	}
	if existingNode == nil {
		klog.Errorf("Unable to register node %q with API server: no node instance returned", kl.nodeName)
		return false
	}

	originalNode := existingNode.DeepCopy()
	if originalNode == nil {
		klog.Errorf("Nil %q node object", kl.nodeName)
		return false
	}

	klog.Infof("Node %s was previously registered", kl.nodeName)

	// Edge case: the node was previously registered; reconcile
	// the value of the controller-managed attach-detach
	// annotation.
	requiresUpdate := kl.reconcileCMADAnnotationWithExistingNode(node, existingNode)
	requiresUpdate = kl.updateDefaultLabels(node, existingNode) || requiresUpdate
	requiresUpdate = kl.reconcileExtendedResource(node, existingNode) || requiresUpdate
	if requiresUpdate {
		if _, _, err := nodeutil.PatchNodeStatus(kl.kubeClient.CoreV1(), types.NodeName(kl.nodeName), originalNode, existingNode); err != nil {
			klog.Errorf("Unable to reconcile node %q with API server: error updating node: %v", kl.nodeName, err)
			return false
		}
	}

	return true
}

// Zeros out extended resource capacity during reconciliation.
func (kl *NodeStatusManager) reconcileExtendedResource(initialNode, node *v1.Node) bool {
	requiresUpdate := false
	// Check with the device manager to see if node has been recreated, in which case extended resources should be zeroed until they are available
	if kl.containerManager.ShouldResetExtendedResourceCapacity() {
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
func (kl *NodeStatusManager) updateDefaultLabels(initialNode, existingNode *v1.Node) bool {
	defaultLabels := []string{
		v1.LabelHostname,
		v1.LabelZoneFailureDomain,
		v1.LabelZoneRegion,
		v1.LabelInstanceType,
		v1.LabelOSStable,
		v1.LabelArchStable,
		kubeletapis.LabelOS,
		kubeletapis.LabelArch,
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
func (kl *NodeStatusManager) reconcileCMADAnnotationWithExistingNode(node, existingNode *v1.Node) bool {
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

// initialNode constructs the initial v1.Node for this NodeStatusManager, incorporating node
// labels, information from the cloud provider, and NodeStatusManager configuration.
func (kl *NodeStatusManager) initialNode(ctx context.Context) (*v1.Node, error) {
	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: string(kl.nodeName),
			Labels: map[string]string{
				v1.LabelHostname:      kl.hostname,
				v1.LabelOSStable:      goruntime.GOOS,
				v1.LabelArchStable:    goruntime.GOARCH,
				kubeletapis.LabelOS:   goruntime.GOOS,
				kubeletapis.LabelArch: goruntime.GOARCH,
			},
		},
		Spec: v1.NodeSpec{
			Unschedulable: !kl.registerSchedulable,
		},
	}
	nodeTaints := make([]v1.Taint, 0)
	if len(kl.registerWithTaints) > 0 {
		taints := make([]v1.Taint, len(kl.registerWithTaints))
		for i := range kl.registerWithTaints {
			if err := k8s_api_v1.Convert_core_Taint_To_v1_Taint(&kl.registerWithTaints[i], &taints[i], nil); err != nil {
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

	if kl.externalCloudProvider {
		taint := v1.Taint{
			Key:    schedulerapi.TaintExternalCloudProvider,
			Value:  "true",
			Effect: v1.TaintEffectNoSchedule,
		}

		nodeTaints = append(nodeTaints, taint)
	}
	if len(nodeTaints) > 0 {
		node.Spec.Taints = nodeTaints
	}
	// Initially, set NodeNetworkUnavailable to true.
	if kl.providerRequiresNetworkingConfiguration() {
		node.Status.Conditions = append(node.Status.Conditions, v1.NodeCondition{
			Type:               v1.NodeNetworkUnavailable,
			Status:             v1.ConditionTrue,
			Reason:             "NoRouteCreated",
			Message:            "Node created without a route",
			LastTransitionTime: metav1.NewTime(kl.clock.Now()),
		})
	}

	if kl.enableControllerAttachDetach {
		if node.Annotations == nil {
			node.Annotations = make(map[string]string)
		}

		klog.Infof("Setting node annotation to enable volume controller attach/detach")
		node.Annotations[volutil.ControllerManagedAttachAnnotation] = "true"
	} else {
		klog.Infof("Controller attach/detach is disabled for this node; NodeStatusManager will attach and detach volumes")
	}

	if kl.keepTerminatedPodVolumes {
		if node.Annotations == nil {
			node.Annotations = make(map[string]string)
		}
		klog.Infof("Setting node annotation to keep pod volumes of terminated pods attached to the node")
		node.Annotations[volutil.KeepTerminatedPodVolumesAnnotation] = "true"
	}

	// @question: should this be place after the call to the cloud provider? which also applies labels
	for k, v := range kl.nodeLabels {
		if cv, found := node.ObjectMeta.Labels[k]; found {
			klog.Warningf("the node label %s=%s will overwrite default setting %s", k, v, cv)
		}
		node.ObjectMeta.Labels[k] = v
	}

	if kl.providerID != "" {
		node.Spec.ProviderID = kl.providerID
	}

	if kl.cloud != nil {
		instances, ok := kl.cloud.Instances()
		if !ok {
			return nil, fmt.Errorf("failed to get instances from cloud provider")
		}

		// TODO: We can't assume that the node has credentials to talk to the
		// cloudprovider from arbitrary nodes. At most, we should talk to a
		// local metadata server here.
		var err error
		if node.Spec.ProviderID == "" {
			node.Spec.ProviderID, err = cloudprovider.GetInstanceProviderID(ctx, kl.cloud, kl.nodeName)
			if err != nil {
				return nil, err
			}
		}

		instanceType, err := instances.InstanceType(ctx, kl.nodeName)
		if err != nil {
			return nil, err
		}
		if instanceType != "" {
			klog.Infof("Adding node label from cloud provider: %s=%s", v1.LabelInstanceType, instanceType)
			node.ObjectMeta.Labels[v1.LabelInstanceType] = instanceType
		}
		// If the cloud has zone information, label the node with the zone information
		zones, ok := kl.cloud.Zones()
		if ok {
			zone, err := zones.GetZone(ctx)
			if err != nil {
				return nil, fmt.Errorf("failed to get zone from cloud provider: %v", err)
			}
			if zone.FailureDomain != "" {
				klog.Infof("Adding node label from cloud provider: %s=%s", v1.LabelZoneFailureDomain, zone.FailureDomain)
				node.ObjectMeta.Labels[v1.LabelZoneFailureDomain] = zone.FailureDomain
			}
			if zone.Region != "" {
				klog.Infof("Adding node label from cloud provider: %s=%s", v1.LabelZoneRegion, zone.Region)
				node.ObjectMeta.Labels[v1.LabelZoneRegion] = zone.Region
			}
		}
	}

	kl.setNodeStatus(node)

	return node, nil
}

// syncNodeStatus should be called periodically from a goroutine.
// It synchronizes node status to master if there is any change or enough time
// passed from the last sync, registering the kubelet first if necessary.
func (kl *NodeStatusManager) syncNodeStatus() {
	kl.syncNodeStatusMux.Lock()
	defer kl.syncNodeStatusMux.Unlock()

	if kl.kubeClient == nil || kl.heartbeatClient == nil {
		return
	}
	if kl.registerNode {
		// This will exit immediately if it doesn't need to do anything.
		kl.registerWithAPIServer()
	}
	if err := kl.updateNodeStatus(); err != nil {
		klog.Errorf("Unable to update node status: %v", err)
	}
}

// updateNodeStatus updates node status to master with retries if there is any
// change or enough time passed from the last sync.
func (kl *NodeStatusManager) updateNodeStatus() error {
	klog.V(5).Infof("Updating node status")
	for i := 0; i < nodeStatusUpdateRetry; i++ {
		if err := kl.tryUpdateNodeStatus(i); err != nil {
			if i > 0 && kl.onRepeatedHeartbeatFailure != nil {
				kl.onRepeatedHeartbeatFailure()
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
func (kl *NodeStatusManager) tryUpdateNodeStatus(tryNumber int) error {
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
	node, err := kl.heartbeatClient.CoreV1().Nodes().Get(string(kl.nodeName), opts)
	if err != nil {
		return fmt.Errorf("error getting node %q: %v", kl.nodeName, err)
	}

	originalNode := node.DeepCopy()
	if originalNode == nil {
		return fmt.Errorf("nil %q node object", kl.nodeName)
	}

	podCIDRChanged := false
	if len(node.Spec.PodCIDRs) != 0 {
		// Pod CIDR could have been updated before, so we cannot rely on
		// node.Spec.PodCIDR being non-empty. We also need to know if pod CIDR is
		// actually changed.
		podCIDRs := strings.Join(node.Spec.PodCIDRs, ",")
		if podCIDRChanged, err = kl.updatePodCIDR(podCIDRs); err != nil {
			klog.Errorf(err.Error())
		}
	}

	kl.setNodeStatus(node)

	now := kl.clock.Now()
	if utilfeature.DefaultFeatureGate.Enabled(features.NodeLease) && now.Before(kl.lastStatusReportTime.Add(kl.nodeStatusReportFrequency)) {
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
			kl.volumeManager.MarkVolumesAsReportedInUse(node.Status.VolumesInUse)
			return nil
		}
	}

	// Patch the current status on the API server
	updatedNode, _, err := nodeutil.PatchNodeStatus(kl.heartbeatClient.CoreV1(), types.NodeName(kl.nodeName), originalNode, node)
	if err != nil {
		return err
	}
	kl.lastStatusReportTime = now
	kl.setLastObservedNodeAddresses(updatedNode.Status.Addresses)
	// If update finishes successfully, mark the volumeInUse as reportedInUse to indicate
	// those volumes are already updated in the node's status
	kl.volumeManager.MarkVolumesAsReportedInUse(updatedNode.Status.VolumesInUse)
	return nil
}

// recordNodeStatusEvent records an event of the given type with the given
// message for the node.
func (kl *NodeStatusManager) recordNodeStatusEvent(eventType, event string) {
	klog.V(2).Infof("Recording %s event message for node %s", event, kl.nodeName)
	// TODO: This requires a transaction, either both node status is updated
	// and event is recorded or neither should happen, see issue #6055.
	kl.recorder.Eventf(kl.nodeRef, eventType, event, "Node %s status is now: %s", kl.nodeName, event)
}

// recordEvent records an event for this node, the NodeStatusManager's nodeRef is passed to the recorder
func (kl *NodeStatusManager) recordEvent(eventType, event, message string) {
	kl.recorder.Eventf(kl.nodeRef, eventType, event, message)
}

// record if node schedulable change.
func (kl *NodeStatusManager) recordNodeSchedulableEvent(node *v1.Node) error {
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
func (kl *NodeStatusManager) setNodeStatus(node *v1.Node) {
	for i, f := range kl.setNodeStatusFuncs {
		klog.V(5).Infof("Setting node status at position %v", i)
		if err := f(node); err != nil {
			klog.Errorf("Failed to set some node status fields: %s", err)
		}
	}
}

func (kl *NodeStatusManager) setLastObservedNodeAddresses(addresses []v1.NodeAddress) {
	kl.lastObservedNodeAddressesMux.Lock()
	defer kl.lastObservedNodeAddressesMux.Unlock()
	kl.lastObservedNodeAddresses = addresses
}
func (kl *NodeStatusManager) getLastObservedNodeAddresses() []v1.NodeAddress {
	kl.lastObservedNodeAddressesMux.RLock()
	defer kl.lastObservedNodeAddressesMux.RUnlock()
	return kl.lastObservedNodeAddresses
}

// Validate given node IP belongs to the current host
func validateNodeIP(nodeIP net.IP) error {
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
func (kl *NodeStatusManager) providerRequiresNetworkingConfiguration() bool {
	// TODO: We should have a mechanism to say whether native cloud provider
	// is used or whether we are using overlay networking. We should return
	// true for cloud providers if they implement Routes() interface and
	// we are not using overlay networking.
	if kl.cloud == nil || kl.cloud.ProviderName() != "gce" {
		return false
	}
	_, supported := kl.cloud.Routes()
	return supported
}
