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
	"fmt"
	"math"
	"net"
	goRuntime "runtime"
	"sort"
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/util/sliceutils"
	utilnet "k8s.io/kubernetes/pkg/util/net"
	"k8s.io/kubernetes/pkg/version"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
)

// registerWithApiServer registers the node with the cluster master. It is safe
// to call multiple times, but not concurrently (kl.registrationCompleted is
// not locked).
func (kl *Kubelet) registerWithApiServer() {
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

		node, err := kl.initialNode()
		if err != nil {
			glog.Errorf("Unable to construct api.Node object for kubelet: %v", err)
			continue
		}

		glog.Infof("Attempting to register node %s", node.Name)
		registered := kl.tryRegisterWithApiServer(node)
		if registered {
			glog.Infof("Successfully registered node %s", node.Name)
			kl.registrationCompleted = true
			return
		}
	}
}

// tryRegisterWithApiServer makes an attempt to register the given node with
// the API server, returning a boolean indicating whether the attempt was
// successful.  If a node with the same name already exists, it reconciles the
// value of the annotation for controller-managed attach-detach of attachable
// persistent volumes for the node.  If a node of the same name exists but has
// a different externalID value, it attempts to delete that node so that a
// later attempt can recreate it.
func (kl *Kubelet) tryRegisterWithApiServer(node *api.Node) bool {
	_, err := kl.kubeClient.Core().Nodes().Create(node)
	if err == nil {
		return true
	}

	if !apierrors.IsAlreadyExists(err) {
		glog.Errorf("Unable to register node %q with API server: %v", kl.nodeName, err)
		return false
	}

	existingNode, err := kl.kubeClient.Core().Nodes().Get(kl.nodeName)
	if err != nil {
		glog.Errorf("Unable to register node %q with API server: error getting existing node: %v", kl.nodeName, err)
		return false
	}
	if existingNode == nil {
		glog.Errorf("Unable to register node %q with API server: no node instance returned", kl.nodeName)
		return false
	}

	if existingNode.Spec.ExternalID == node.Spec.ExternalID {
		glog.Infof("Node %s was previously registered", kl.nodeName)

		// Edge case: the node was previously registered; reconcile
		// the value of the controller-managed attach-detach
		// annotation.
		requiresUpdate := kl.reconcileCMADAnnotationWithExistingNode(node, existingNode)
		if requiresUpdate {
			if _, err := kl.kubeClient.Core().Nodes().UpdateStatus(existingNode); err != nil {
				glog.Errorf("Unable to reconcile node %q with API server: error updating node: %v", kl.nodeName, err)
				return false
			}
		}

		return true
	}

	glog.Errorf(
		"Previously node %q had externalID %q; now it is %q; will delete and recreate.",
		kl.nodeName, node.Spec.ExternalID, existingNode.Spec.ExternalID,
	)
	if err := kl.kubeClient.Core().Nodes().Delete(node.Name, nil); err != nil {
		glog.Errorf("Unable to register node %q with API server: error deleting old node: %v", kl.nodeName, err)
	} else {
		glog.Info("Deleted old node object %q", kl.nodeName)
	}

	return false
}

// reconcileCMADAnnotationWithExistingNode reconciles the controller-managed
// attach-detach annotation on a new node and the existing node, returning
// whether the existing node must be updated.
func (kl *Kubelet) reconcileCMADAnnotationWithExistingNode(node, existingNode *api.Node) bool {
	var (
		existingCMAAnnotation    = existingNode.Annotations[volumehelper.ControllerManagedAttachAnnotation]
		newCMAAnnotation, newSet = node.Annotations[volumehelper.ControllerManagedAttachAnnotation]
	)

	if newCMAAnnotation == existingCMAAnnotation {
		return false
	}

	// If the just-constructed node and the existing node do
	// not have the same value, update the existing node with
	// the correct value of the annotation.
	if !newSet {
		glog.Info("Controller attach-detach setting changed to false; updating existing Node")
		delete(existingNode.Annotations, volumehelper.ControllerManagedAttachAnnotation)
	} else {
		glog.Info("Controller attach-detach setting changed to true; updating existing Node")
		if existingNode.Annotations == nil {
			existingNode.Annotations = make(map[string]string)
		}
		existingNode.Annotations[volumehelper.ControllerManagedAttachAnnotation] = newCMAAnnotation
	}

	return true
}

// initialNode constructs the initial api.Node for this Kubelet, incorporating node
// labels, information from the cloud provider, and Kubelet configuration.
func (kl *Kubelet) initialNode() (*api.Node, error) {
	node := &api.Node{
		ObjectMeta: api.ObjectMeta{
			Name: kl.nodeName,
			Labels: map[string]string{
				unversioned.LabelHostname: kl.hostname,
				unversioned.LabelOS:       goRuntime.GOOS,
				unversioned.LabelArch:     goRuntime.GOARCH,
			},
		},
		Spec: api.NodeSpec{
			Unschedulable: !kl.registerSchedulable,
		},
	}
	// Initially, set NodeNetworkUnavailable to true.
	if kl.providerRequiresNetworkingConfiguration() {
		node.Status.Conditions = append(node.Status.Conditions, api.NodeCondition{
			Type:               api.NodeNetworkUnavailable,
			Status:             api.ConditionTrue,
			Reason:             "NoRouteCreated",
			Message:            "Node created without a route",
			LastTransitionTime: unversioned.NewTime(kl.clock.Now()),
		})
	}

	if kl.enableControllerAttachDetach {
		if node.Annotations == nil {
			node.Annotations = make(map[string]string)
		}

		glog.Infof("Setting node annotation to enable volume controller attach/detach")
		node.Annotations[volumehelper.ControllerManagedAttachAnnotation] = "true"
	} else {
		glog.Infof("Controller attach/detach is disabled for this node; Kubelet will attach and detach volumes")
	}

	// @question: should this be place after the call to the cloud provider? which also applies labels
	for k, v := range kl.nodeLabels {
		if cv, found := node.ObjectMeta.Labels[k]; found {
			glog.Warningf("the node label %s=%s will overwrite default setting %s", k, v, cv)
		}
		node.ObjectMeta.Labels[k] = v
	}

	if kl.cloud != nil {
		instances, ok := kl.cloud.Instances()
		if !ok {
			return nil, fmt.Errorf("failed to get instances from cloud provider")
		}

		// TODO(roberthbailey): Can we do this without having credentials to talk
		// to the cloud provider?
		// TODO: ExternalID is deprecated, we'll have to drop this code
		externalID, err := instances.ExternalID(kl.nodeName)
		if err != nil {
			return nil, fmt.Errorf("failed to get external ID from cloud provider: %v", err)
		}
		node.Spec.ExternalID = externalID

		// TODO: We can't assume that the node has credentials to talk to the
		// cloudprovider from arbitrary nodes. At most, we should talk to a
		// local metadata server here.
		node.Spec.ProviderID, err = cloudprovider.GetInstanceProviderID(kl.cloud, kl.nodeName)
		if err != nil {
			return nil, err
		}

		instanceType, err := instances.InstanceType(kl.nodeName)
		if err != nil {
			return nil, err
		}
		if instanceType != "" {
			glog.Infof("Adding node label from cloud provider: %s=%s", unversioned.LabelInstanceType, instanceType)
			node.ObjectMeta.Labels[unversioned.LabelInstanceType] = instanceType
		}
		// If the cloud has zone information, label the node with the zone information
		zones, ok := kl.cloud.Zones()
		if ok {
			zone, err := zones.GetZone()
			if err != nil {
				return nil, fmt.Errorf("failed to get zone from cloud provider: %v", err)
			}
			if zone.FailureDomain != "" {
				glog.Infof("Adding node label from cloud provider: %s=%s", unversioned.LabelZoneFailureDomain, zone.FailureDomain)
				node.ObjectMeta.Labels[unversioned.LabelZoneFailureDomain] = zone.FailureDomain
			}
			if zone.Region != "" {
				glog.Infof("Adding node label from cloud provider: %s=%s", unversioned.LabelZoneRegion, zone.Region)
				node.ObjectMeta.Labels[unversioned.LabelZoneRegion] = zone.Region
			}
		}
	} else {
		node.Spec.ExternalID = kl.hostname
		if kl.autoDetectCloudProvider {
			// If no cloud provider is defined - use the one detected by cadvisor
			info, err := kl.GetCachedMachineInfo()
			if err == nil {
				kl.updateCloudProviderFromMachineInfo(node, info)
			}
		}
	}
	if err := kl.setNodeStatus(node); err != nil {
		return nil, err
	}

	return node, nil
}

// syncNodeStatus should be called periodically from a goroutine.
// It synchronizes node status to master, registering the kubelet first if
// necessary.
func (kl *Kubelet) syncNodeStatus() {
	if kl.kubeClient == nil {
		return
	}
	if kl.registerNode {
		// This will exit immediately if it doesn't need to do anything.
		kl.registerWithApiServer()
	}
	if err := kl.updateNodeStatus(); err != nil {
		glog.Errorf("Unable to update node status: %v", err)
	}
}

// updateNodeStatus updates node status to master with retries.
func (kl *Kubelet) updateNodeStatus() error {
	for i := 0; i < nodeStatusUpdateRetry; i++ {
		if err := kl.tryUpdateNodeStatus(); err != nil {
			glog.Errorf("Error updating node status, will retry: %v", err)
		} else {
			return nil
		}
	}
	return fmt.Errorf("update node status exceeds retry count")
}

// tryUpdateNodeStatus tries to update node status to master. If ReconcileCBR0
// is set, this function will also confirm that cbr0 is configured correctly.
func (kl *Kubelet) tryUpdateNodeStatus() error {
	node, err := kl.kubeClient.Core().Nodes().Get(kl.nodeName)
	if err != nil {
		return fmt.Errorf("error getting node %q: %v", kl.nodeName, err)
	}
	if node == nil {
		return fmt.Errorf("no node instance returned for %q", kl.nodeName)
	}

	// Flannel is the authoritative source of pod CIDR, if it's running.
	// This is a short term compromise till we get flannel working in
	// reservation mode.
	if kl.flannelExperimentalOverlay {
		flannelPodCIDR := kl.runtimeState.podCIDR()
		if node.Spec.PodCIDR != flannelPodCIDR {
			node.Spec.PodCIDR = flannelPodCIDR
			glog.Infof("Updating podcidr to %v", node.Spec.PodCIDR)
			if updatedNode, err := kl.kubeClient.Core().Nodes().Update(node); err != nil {
				glog.Warningf("Failed to update podCIDR: %v", err)
			} else {
				// Update the node resourceVersion so the status update doesn't fail.
				node = updatedNode
			}
		}
	} else if kl.reconcileCIDR {
		kl.updatePodCIDR(node.Spec.PodCIDR)
	}

	if err := kl.setNodeStatus(node); err != nil {
		return err
	}
	// Update the current status on the API server
	updatedNode, err := kl.kubeClient.Core().Nodes().UpdateStatus(node)
	if err == nil {
		kl.volumeManager.MarkVolumesAsReportedInUse(
			updatedNode.Status.VolumesInUse)
	}
	return err
}

// recordNodeStatusEvent records an event of the given type with the given
// message for the node.
func (kl *Kubelet) recordNodeStatusEvent(eventtype, event string) {
	glog.V(2).Infof("Recording %s event message for node %s", event, kl.nodeName)
	// TODO: This requires a transaction, either both node status is updated
	// and event is recorded or neither should happen, see issue #6055.
	kl.recorder.Eventf(kl.nodeRef, eventtype, event, "Node %s status is now: %s", kl.nodeName, event)
}

// Set IP addresses for the node.
func (kl *Kubelet) setNodeAddress(node *api.Node) error {
	if kl.cloud != nil {
		instances, ok := kl.cloud.Instances()
		if !ok {
			return fmt.Errorf("failed to get instances from cloud provider")
		}
		// TODO(roberthbailey): Can we do this without having credentials to talk
		// to the cloud provider?
		// TODO(justinsb): We can if CurrentNodeName() was actually CurrentNode() and returned an interface
		// TODO: If IP addresses couldn't be fetched from the cloud provider, should kubelet fallback on the other methods for getting the IP below?
		nodeAddresses, err := instances.NodeAddresses(kl.nodeName)
		if err != nil {
			return fmt.Errorf("failed to get node address from cloud provider: %v", err)
		}
		node.Status.Addresses = nodeAddresses
	} else {
		var ipAddr net.IP
		var err error

		// 1) Use nodeIP if set
		// 2) If the user has specified an IP to HostnameOverride, use it
		// 3) Lookup the IP from node name by DNS and use the first non-loopback ipv4 address
		// 4) Try to get the IP from the network interface used as default gateway
		if kl.nodeIP != nil {
			ipAddr = kl.nodeIP
		} else if addr := net.ParseIP(kl.hostname); addr != nil {
			ipAddr = addr
		} else {
			var addrs []net.IP
			addrs, err = net.LookupIP(node.Name)
			for _, addr := range addrs {
				if !addr.IsLoopback() && addr.To4() != nil {
					ipAddr = addr
					break
				}
			}

			if ipAddr == nil {
				ipAddr, err = utilnet.ChooseHostInterface()
			}
		}

		if ipAddr == nil {
			// We tried everything we could, but the IP address wasn't fetchable; error out
			return fmt.Errorf("can't get ip address of node %s. error: %v", node.Name, err)
		} else {
			node.Status.Addresses = []api.NodeAddress{
				{Type: api.NodeLegacyHostIP, Address: ipAddr.String()},
				{Type: api.NodeInternalIP, Address: ipAddr.String()},
			}
		}
	}
	return nil
}

func (kl *Kubelet) setNodeStatusMachineInfo(node *api.Node) {
	// TODO: Post NotReady if we cannot get MachineInfo from cAdvisor. This needs to start
	// cAdvisor locally, e.g. for test-cmd.sh, and in integration test.
	info, err := kl.GetCachedMachineInfo()
	if err != nil {
		// TODO(roberthbailey): This is required for test-cmd.sh to pass.
		// See if the test should be updated instead.
		node.Status.Capacity = api.ResourceList{
			api.ResourceCPU:       *resource.NewMilliQuantity(0, resource.DecimalSI),
			api.ResourceMemory:    resource.MustParse("0Gi"),
			api.ResourcePods:      *resource.NewQuantity(int64(kl.maxPods), resource.DecimalSI),
			api.ResourceNvidiaGPU: *resource.NewQuantity(int64(kl.nvidiaGPUs), resource.DecimalSI),
		}
		glog.Errorf("Error getting machine info: %v", err)
	} else {
		node.Status.NodeInfo.MachineID = info.MachineID
		node.Status.NodeInfo.SystemUUID = info.SystemUUID
		node.Status.Capacity = cadvisor.CapacityFromMachineInfo(info)
		if kl.podsPerCore > 0 {
			node.Status.Capacity[api.ResourcePods] = *resource.NewQuantity(
				int64(math.Min(float64(info.NumCores*kl.podsPerCore), float64(kl.maxPods))), resource.DecimalSI)
		} else {
			node.Status.Capacity[api.ResourcePods] = *resource.NewQuantity(
				int64(kl.maxPods), resource.DecimalSI)
		}
		node.Status.Capacity[api.ResourceNvidiaGPU] = *resource.NewQuantity(
			int64(kl.nvidiaGPUs), resource.DecimalSI)
		if node.Status.NodeInfo.BootID != "" &&
			node.Status.NodeInfo.BootID != info.BootID {
			// TODO: This requires a transaction, either both node status is updated
			// and event is recorded or neither should happen, see issue #6055.
			kl.recorder.Eventf(kl.nodeRef, api.EventTypeWarning, events.NodeRebooted,
				"Node %s has been rebooted, boot id: %s", kl.nodeName, info.BootID)
		}
		node.Status.NodeInfo.BootID = info.BootID
	}

	// Set Allocatable.
	node.Status.Allocatable = make(api.ResourceList)
	for k, v := range node.Status.Capacity {
		value := *(v.Copy())
		if kl.reservation.System != nil {
			value.Sub(kl.reservation.System[k])
		}
		if kl.reservation.Kubernetes != nil {
			value.Sub(kl.reservation.Kubernetes[k])
		}
		if value.Sign() < 0 {
			// Negative Allocatable resources don't make sense.
			value.Set(0)
		}
		node.Status.Allocatable[k] = value
	}
}

// Set versioninfo for the node.
func (kl *Kubelet) setNodeStatusVersionInfo(node *api.Node) {
	verinfo, err := kl.cadvisor.VersionInfo()
	if err != nil {
		glog.Errorf("Error getting version info: %v", err)
	} else {
		node.Status.NodeInfo.KernelVersion = verinfo.KernelVersion
		node.Status.NodeInfo.OSImage = verinfo.ContainerOsVersion

		runtimeVersion := "Unknown"
		if runtimeVer, err := kl.containerRuntime.Version(); err == nil {
			runtimeVersion = runtimeVer.String()
		}
		node.Status.NodeInfo.ContainerRuntimeVersion = fmt.Sprintf("%s://%s", kl.containerRuntime.Type(), runtimeVersion)

		node.Status.NodeInfo.KubeletVersion = version.Get().String()
		// TODO: kube-proxy might be different version from kubelet in the future
		node.Status.NodeInfo.KubeProxyVersion = version.Get().String()
	}

}

// Set daemonEndpoints for the node.
func (kl *Kubelet) setNodeStatusDaemonEndpoints(node *api.Node) {
	node.Status.DaemonEndpoints = *kl.daemonEndpoints
}

// Set images list for the node
func (kl *Kubelet) setNodeStatusImages(node *api.Node) {
	// Update image list of this node
	var imagesOnNode []api.ContainerImage
	containerImages, err := kl.imageManager.GetImageList()
	if err != nil {
		glog.Errorf("Error getting image list: %v", err)
	} else {
		// sort the images from max to min, and only set top N images into the node status.
		sort.Sort(sliceutils.ByImageSize(containerImages))
		if maxImagesInNodeStatus < len(containerImages) {
			containerImages = containerImages[0:maxImagesInNodeStatus]
		}

		for _, image := range containerImages {
			imagesOnNode = append(imagesOnNode, api.ContainerImage{
				Names:     append(image.RepoTags, image.RepoDigests...),
				SizeBytes: image.Size,
			})
		}
	}
	node.Status.Images = imagesOnNode
}

// Set the GOOS and GOARCH for this node
func (kl *Kubelet) setNodeStatusGoRuntime(node *api.Node) {
	node.Status.NodeInfo.OperatingSystem = goRuntime.GOOS
	node.Status.NodeInfo.Architecture = goRuntime.GOARCH
}

// Set status for the node.
func (kl *Kubelet) setNodeStatusInfo(node *api.Node) {
	kl.setNodeStatusMachineInfo(node)
	kl.setNodeStatusVersionInfo(node)
	kl.setNodeStatusDaemonEndpoints(node)
	kl.setNodeStatusImages(node)
	kl.setNodeStatusGoRuntime(node)
}

// Set Ready condition for the node.
func (kl *Kubelet) setNodeReadyCondition(node *api.Node) {
	// NOTE(aaronlevy): NodeReady condition needs to be the last in the list of node conditions.
	// This is due to an issue with version skewed kubelet and master components.
	// ref: https://github.com/kubernetes/kubernetes/issues/16961
	currentTime := unversioned.NewTime(kl.clock.Now())
	var newNodeReadyCondition api.NodeCondition
	if rs := kl.runtimeState.errors(); len(rs) == 0 {
		newNodeReadyCondition = api.NodeCondition{
			Type:              api.NodeReady,
			Status:            api.ConditionTrue,
			Reason:            "KubeletReady",
			Message:           "kubelet is posting ready status",
			LastHeartbeatTime: currentTime,
		}
	} else {
		newNodeReadyCondition = api.NodeCondition{
			Type:              api.NodeReady,
			Status:            api.ConditionFalse,
			Reason:            "KubeletNotReady",
			Message:           strings.Join(rs, ","),
			LastHeartbeatTime: currentTime,
		}
	}

	// Append AppArmor status if it's enabled.
	// TODO(timstclair): This is a temporary message until node feature reporting is added.
	if newNodeReadyCondition.Status == api.ConditionTrue &&
		kl.appArmorValidator != nil && kl.appArmorValidator.ValidateHost() == nil {
		newNodeReadyCondition.Message = fmt.Sprintf("%s. AppArmor enabled", newNodeReadyCondition.Message)
	}

	// Record any soft requirements that were not met in the container manager.
	status := kl.containerManager.Status()
	if status.SoftRequirements != nil {
		newNodeReadyCondition.Message = fmt.Sprintf("%s. WARNING: %s", newNodeReadyCondition.Message, status.SoftRequirements.Error())
	}

	readyConditionUpdated := false
	needToRecordEvent := false
	for i := range node.Status.Conditions {
		if node.Status.Conditions[i].Type == api.NodeReady {
			if node.Status.Conditions[i].Status == newNodeReadyCondition.Status {
				newNodeReadyCondition.LastTransitionTime = node.Status.Conditions[i].LastTransitionTime
			} else {
				newNodeReadyCondition.LastTransitionTime = currentTime
				needToRecordEvent = true
			}
			node.Status.Conditions[i] = newNodeReadyCondition
			readyConditionUpdated = true
			break
		}
	}
	if !readyConditionUpdated {
		newNodeReadyCondition.LastTransitionTime = currentTime
		node.Status.Conditions = append(node.Status.Conditions, newNodeReadyCondition)
	}
	if needToRecordEvent {
		if newNodeReadyCondition.Status == api.ConditionTrue {
			kl.recordNodeStatusEvent(api.EventTypeNormal, events.NodeReady)
		} else {
			kl.recordNodeStatusEvent(api.EventTypeNormal, events.NodeNotReady)
		}
	}
}

// setNodeMemoryPressureCondition for the node.
// TODO: this needs to move somewhere centralized...
func (kl *Kubelet) setNodeMemoryPressureCondition(node *api.Node) {
	currentTime := unversioned.NewTime(kl.clock.Now())
	var condition *api.NodeCondition

	// Check if NodeMemoryPressure condition already exists and if it does, just pick it up for update.
	for i := range node.Status.Conditions {
		if node.Status.Conditions[i].Type == api.NodeMemoryPressure {
			condition = &node.Status.Conditions[i]
		}
	}

	newCondition := false
	// If the NodeMemoryPressure condition doesn't exist, create one
	if condition == nil {
		condition = &api.NodeCondition{
			Type:   api.NodeMemoryPressure,
			Status: api.ConditionUnknown,
		}
		// cannot be appended to node.Status.Conditions here because it gets
		// copied to the slice. So if we append to the slice here none of the
		// updates we make below are reflected in the slice.
		newCondition = true
	}

	// Update the heartbeat time
	condition.LastHeartbeatTime = currentTime

	// Note: The conditions below take care of the case when a new NodeMemoryPressure condition is
	// created and as well as the case when the condition already exists. When a new condition
	// is created its status is set to api.ConditionUnknown which matches either
	// condition.Status != api.ConditionTrue or
	// condition.Status != api.ConditionFalse in the conditions below depending on whether
	// the kubelet is under memory pressure or not.
	if kl.evictionManager.IsUnderMemoryPressure() {
		if condition.Status != api.ConditionTrue {
			condition.Status = api.ConditionTrue
			condition.Reason = "KubeletHasInsufficientMemory"
			condition.Message = "kubelet has insufficient memory available"
			condition.LastTransitionTime = currentTime
			kl.recordNodeStatusEvent(api.EventTypeNormal, "NodeHasInsufficientMemory")
		}
	} else {
		if condition.Status != api.ConditionFalse {
			condition.Status = api.ConditionFalse
			condition.Reason = "KubeletHasSufficientMemory"
			condition.Message = "kubelet has sufficient memory available"
			condition.LastTransitionTime = currentTime
			kl.recordNodeStatusEvent(api.EventTypeNormal, "NodeHasSufficientMemory")
		}
	}

	if newCondition {
		node.Status.Conditions = append(node.Status.Conditions, *condition)
	}
}

// setNodeDiskPressureCondition for the node.
// TODO: this needs to move somewhere centralized...
func (kl *Kubelet) setNodeDiskPressureCondition(node *api.Node) {
	currentTime := unversioned.NewTime(kl.clock.Now())
	var condition *api.NodeCondition

	// Check if NodeDiskPressure condition already exists and if it does, just pick it up for update.
	for i := range node.Status.Conditions {
		if node.Status.Conditions[i].Type == api.NodeDiskPressure {
			condition = &node.Status.Conditions[i]
		}
	}

	newCondition := false
	// If the NodeDiskPressure condition doesn't exist, create one
	if condition == nil {
		condition = &api.NodeCondition{
			Type:   api.NodeDiskPressure,
			Status: api.ConditionUnknown,
		}
		// cannot be appended to node.Status.Conditions here because it gets
		// copied to the slice. So if we append to the slice here none of the
		// updates we make below are reflected in the slice.
		newCondition = true
	}

	// Update the heartbeat time
	condition.LastHeartbeatTime = currentTime

	// Note: The conditions below take care of the case when a new NodeDiskressure condition is
	// created and as well as the case when the condition already exists. When a new condition
	// is created its status is set to api.ConditionUnknown which matches either
	// condition.Status != api.ConditionTrue or
	// condition.Status != api.ConditionFalse in the conditions below depending on whether
	// the kubelet is under disk pressure or not.
	if kl.evictionManager.IsUnderDiskPressure() {
		if condition.Status != api.ConditionTrue {
			condition.Status = api.ConditionTrue
			condition.Reason = "KubeletHasDiskPressure"
			condition.Message = "kubelet has disk pressure"
			condition.LastTransitionTime = currentTime
			kl.recordNodeStatusEvent(api.EventTypeNormal, "NodeHasDiskPressure")
		}
	} else {
		if condition.Status != api.ConditionFalse {
			condition.Status = api.ConditionFalse
			condition.Reason = "KubeletHasNoDiskPressure"
			condition.Message = "kubelet has no disk pressure"
			condition.LastTransitionTime = currentTime
			kl.recordNodeStatusEvent(api.EventTypeNormal, "NodeHasNoDiskPressure")
		}
	}

	if newCondition {
		node.Status.Conditions = append(node.Status.Conditions, *condition)
	}
}

// Set OODcondition for the node.
func (kl *Kubelet) setNodeOODCondition(node *api.Node) {
	currentTime := unversioned.NewTime(kl.clock.Now())
	var nodeOODCondition *api.NodeCondition

	// Check if NodeOutOfDisk condition already exists and if it does, just pick it up for update.
	for i := range node.Status.Conditions {
		if node.Status.Conditions[i].Type == api.NodeOutOfDisk {
			nodeOODCondition = &node.Status.Conditions[i]
		}
	}

	newOODCondition := false
	// If the NodeOutOfDisk condition doesn't exist, create one.
	if nodeOODCondition == nil {
		nodeOODCondition = &api.NodeCondition{
			Type:   api.NodeOutOfDisk,
			Status: api.ConditionUnknown,
		}
		// nodeOODCondition cannot be appended to node.Status.Conditions here because it gets
		// copied to the slice. So if we append nodeOODCondition to the slice here none of the
		// updates we make to nodeOODCondition below are reflected in the slice.
		newOODCondition = true
	}

	// Update the heartbeat time irrespective of all the conditions.
	nodeOODCondition.LastHeartbeatTime = currentTime

	// Note: The conditions below take care of the case when a new NodeOutOfDisk condition is
	// created and as well as the case when the condition already exists. When a new condition
	// is created its status is set to api.ConditionUnknown which matches either
	// nodeOODCondition.Status != api.ConditionTrue or
	// nodeOODCondition.Status != api.ConditionFalse in the conditions below depending on whether
	// the kubelet is out of disk or not.
	if kl.isOutOfDisk() {
		if nodeOODCondition.Status != api.ConditionTrue {
			nodeOODCondition.Status = api.ConditionTrue
			nodeOODCondition.Reason = "KubeletOutOfDisk"
			nodeOODCondition.Message = "out of disk space"
			nodeOODCondition.LastTransitionTime = currentTime
			kl.recordNodeStatusEvent(api.EventTypeNormal, "NodeOutOfDisk")
		}
	} else {
		if nodeOODCondition.Status != api.ConditionFalse {
			// Update the out of disk condition when the condition status is unknown even if we
			// are within the outOfDiskTransitionFrequency duration. We do this to set the
			// condition status correctly at kubelet startup.
			if nodeOODCondition.Status == api.ConditionUnknown || kl.clock.Since(nodeOODCondition.LastTransitionTime.Time) >= kl.outOfDiskTransitionFrequency {
				nodeOODCondition.Status = api.ConditionFalse
				nodeOODCondition.Reason = "KubeletHasSufficientDisk"
				nodeOODCondition.Message = "kubelet has sufficient disk space available"
				nodeOODCondition.LastTransitionTime = currentTime
				kl.recordNodeStatusEvent(api.EventTypeNormal, "NodeHasSufficientDisk")
			} else {
				glog.Infof("Node condition status for OutOfDisk is false, but last transition time is less than %s", kl.outOfDiskTransitionFrequency)
			}
		}
	}

	if newOODCondition {
		node.Status.Conditions = append(node.Status.Conditions, *nodeOODCondition)
	}
}

// Maintains Node.Spec.Unschedulable value from previous run of tryUpdateNodeStatus()
// TODO: why is this a package var?
var oldNodeUnschedulable bool

// record if node schedulable change.
func (kl *Kubelet) recordNodeSchedulableEvent(node *api.Node) {
	if oldNodeUnschedulable != node.Spec.Unschedulable {
		if node.Spec.Unschedulable {
			kl.recordNodeStatusEvent(api.EventTypeNormal, events.NodeNotSchedulable)
		} else {
			kl.recordNodeStatusEvent(api.EventTypeNormal, events.NodeSchedulable)
		}
		oldNodeUnschedulable = node.Spec.Unschedulable
	}
}

// Update VolumesInUse field in Node Status
func (kl *Kubelet) setNodeVolumesInUseStatus(node *api.Node) {
	node.Status.VolumesInUse = kl.volumeManager.GetVolumesInUse()
}

// setNodeStatus fills in the Status fields of the given Node, overwriting
// any fields that are currently set.
// TODO(madhusudancs): Simplify the logic for setting node conditions and
// refactor the node status condition code out to a different file.
func (kl *Kubelet) setNodeStatus(node *api.Node) error {
	for _, f := range kl.setNodeStatusFuncs {
		if err := f(node); err != nil {
			return err
		}
	}
	return nil
}

// defaultNodeStatusFuncs is a factory that generates the default set of
// setNodeStatus funcs
func (kl *Kubelet) defaultNodeStatusFuncs() []func(*api.Node) error {
	// initial set of node status update handlers, can be modified by Option's
	withoutError := func(f func(*api.Node)) func(*api.Node) error {
		return func(n *api.Node) error {
			f(n)
			return nil
		}
	}
	return []func(*api.Node) error{
		kl.setNodeAddress,
		withoutError(kl.setNodeStatusInfo),
		withoutError(kl.setNodeOODCondition),
		withoutError(kl.setNodeMemoryPressureCondition),
		withoutError(kl.setNodeDiskPressureCondition),
		withoutError(kl.setNodeReadyCondition),
		withoutError(kl.setNodeVolumesInUseStatus),
		withoutError(kl.recordNodeSchedulableEvent),
	}
}

// SetNodeStatus returns a functional Option that adds the given node status
// update handler to the Kubelet
func SetNodeStatus(f func(*api.Node) error) Option {
	return func(k *Kubelet) {
		k.setNodeStatusFuncs = append(k.setNodeStatusFuncs, f)
	}
}
