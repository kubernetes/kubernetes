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
	"math"
	"net"
	goruntime "runtime"
	"strings"
	"sync"
	"time"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	k8s_api_v1 "k8s.io/kubernetes/pkg/apis/core/v1"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/features"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/util"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
	"k8s.io/kubernetes/pkg/version"
	volutil "k8s.io/kubernetes/pkg/volume/util"
)

const (
	// maxNamesPerImageInNodeStatus is max number of names per image stored in
	// the node status.
	maxNamesPerImageInNodeStatus = 5
)

// registerWithAPIServer registers the node with the cluster master. It is safe
// to call multiple times, but not concurrently (kl.registrationCompleted is
// not locked).
func (kl *Kubelet) registerWithAPIServer() {
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
			glog.Errorf("Unable to construct v1.Node object for kubelet: %v", err)
			continue
		}

		glog.Infof("Attempting to register node %s", node.Name)
		registered := kl.tryRegisterWithAPIServer(node)
		if registered {
			glog.Infof("Successfully registered node %s", node.Name)
			kl.registrationCompleted = true
			return
		}
	}
}

// tryRegisterWithAPIServer makes an attempt to register the given node with
// the API server, returning a boolean indicating whether the attempt was
// successful.  If a node with the same name already exists, it reconciles the
// value of the annotation for controller-managed attach-detach of attachable
// persistent volumes for the node.  If a node of the same name exists but has
// a different externalID value, it attempts to delete that node so that a
// later attempt can recreate it.
func (kl *Kubelet) tryRegisterWithAPIServer(node *v1.Node) bool {
	_, err := kl.kubeClient.CoreV1().Nodes().Create(node)
	if err == nil {
		return true
	}

	if !apierrors.IsAlreadyExists(err) {
		glog.Errorf("Unable to register node %q with API server: %v", kl.nodeName, err)
		return false
	}

	existingNode, err := kl.kubeClient.CoreV1().Nodes().Get(string(kl.nodeName), metav1.GetOptions{})
	if err != nil {
		glog.Errorf("Unable to register node %q with API server: error getting existing node: %v", kl.nodeName, err)
		return false
	}
	if existingNode == nil {
		glog.Errorf("Unable to register node %q with API server: no node instance returned", kl.nodeName)
		return false
	}

	originalNode := existingNode.DeepCopy()
	if originalNode == nil {
		glog.Errorf("Nil %q node object", kl.nodeName)
		return false
	}

	glog.Infof("Node %s was previously registered", kl.nodeName)

	// Edge case: the node was previously registered; reconcile
	// the value of the controller-managed attach-detach
	// annotation.
	requiresUpdate := kl.reconcileCMADAnnotationWithExistingNode(node, existingNode)
	requiresUpdate = kl.updateDefaultLabels(node, existingNode) || requiresUpdate
	requiresUpdate = kl.reconcileExtendedResource(node, existingNode) || requiresUpdate
	if requiresUpdate {
		if _, _, err := nodeutil.PatchNodeStatus(kl.kubeClient.CoreV1(), types.NodeName(kl.nodeName), originalNode, existingNode); err != nil {
			glog.Errorf("Unable to reconcile node %q with API server: error updating node: %v", kl.nodeName, err)
			return false
		}
	}

	return true
}

// Zeros out extended resource capacity during reconciliation.
func (kl *Kubelet) reconcileExtendedResource(initialNode, node *v1.Node) bool {
	requiresUpdate := false
	for k := range node.Status.Capacity {
		if v1helper.IsExtendedResourceName(k) {
			node.Status.Capacity[k] = *resource.NewQuantity(int64(0), resource.DecimalSI)
			node.Status.Allocatable[k] = *resource.NewQuantity(int64(0), resource.DecimalSI)
			requiresUpdate = true
		}
	}
	return requiresUpdate
}

// updateDefaultLabels will set the default labels on the node
func (kl *Kubelet) updateDefaultLabels(initialNode, existingNode *v1.Node) bool {
	defaultLabels := []string{
		kubeletapis.LabelHostname,
		kubeletapis.LabelZoneFailureDomain,
		kubeletapis.LabelZoneRegion,
		kubeletapis.LabelInstanceType,
		kubeletapis.LabelOS,
		kubeletapis.LabelArch,
	}

	var needsUpdate bool = false
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
func (kl *Kubelet) reconcileCMADAnnotationWithExistingNode(node, existingNode *v1.Node) bool {
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
		glog.Info("Controller attach-detach setting changed to false; updating existing Node")
		delete(existingNode.Annotations, volutil.ControllerManagedAttachAnnotation)
	} else {
		glog.Info("Controller attach-detach setting changed to true; updating existing Node")
		if existingNode.Annotations == nil {
			existingNode.Annotations = make(map[string]string)
		}
		existingNode.Annotations[volutil.ControllerManagedAttachAnnotation] = newCMAAnnotation
	}

	return true
}

// initialNode constructs the initial v1.Node for this Kubelet, incorporating node
// labels, information from the cloud provider, and Kubelet configuration.
func (kl *Kubelet) initialNode() (*v1.Node, error) {
	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: string(kl.nodeName),
			Labels: map[string]string{
				kubeletapis.LabelHostname: kl.hostname,
				kubeletapis.LabelOS:       goruntime.GOOS,
				kubeletapis.LabelArch:     goruntime.GOARCH,
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
	if kl.externalCloudProvider {
		taint := v1.Taint{
			Key:    algorithm.TaintExternalCloudProvider,
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

		glog.Infof("Setting node annotation to enable volume controller attach/detach")
		node.Annotations[volutil.ControllerManagedAttachAnnotation] = "true"
	} else {
		glog.Infof("Controller attach/detach is disabled for this node; Kubelet will attach and detach volumes")
	}

	if kl.keepTerminatedPodVolumes {
		if node.Annotations == nil {
			node.Annotations = make(map[string]string)
		}
		glog.Infof("Setting node annotation to keep pod volumes of terminated pods attached to the node")
		node.Annotations[volutil.KeepTerminatedPodVolumesAnnotation] = "true"
	}

	// @question: should this be place after the call to the cloud provider? which also applies labels
	for k, v := range kl.nodeLabels {
		if cv, found := node.ObjectMeta.Labels[k]; found {
			glog.Warningf("the node label %s=%s will overwrite default setting %s", k, v, cv)
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
			node.Spec.ProviderID, err = cloudprovider.GetInstanceProviderID(context.TODO(), kl.cloud, kl.nodeName)
			if err != nil {
				return nil, err
			}
		}

		instanceType, err := instances.InstanceType(context.TODO(), kl.nodeName)
		if err != nil {
			return nil, err
		}
		if instanceType != "" {
			glog.Infof("Adding node label from cloud provider: %s=%s", kubeletapis.LabelInstanceType, instanceType)
			node.ObjectMeta.Labels[kubeletapis.LabelInstanceType] = instanceType
		}
		// If the cloud has zone information, label the node with the zone information
		zones, ok := kl.cloud.Zones()
		if ok {
			zone, err := zones.GetZone(context.TODO())
			if err != nil {
				return nil, fmt.Errorf("failed to get zone from cloud provider: %v", err)
			}
			if zone.FailureDomain != "" {
				glog.Infof("Adding node label from cloud provider: %s=%s", kubeletapis.LabelZoneFailureDomain, zone.FailureDomain)
				node.ObjectMeta.Labels[kubeletapis.LabelZoneFailureDomain] = zone.FailureDomain
			}
			if zone.Region != "" {
				glog.Infof("Adding node label from cloud provider: %s=%s", kubeletapis.LabelZoneRegion, zone.Region)
				node.ObjectMeta.Labels[kubeletapis.LabelZoneRegion] = zone.Region
			}
		}
	}

	kl.setNodeStatus(node)

	return node, nil
}

// setVolumeLimits updates volume limits on the node
func (kl *Kubelet) setVolumeLimits(node *v1.Node) {
	if node.Status.Capacity == nil {
		node.Status.Capacity = v1.ResourceList{}
	}

	if node.Status.Allocatable == nil {
		node.Status.Allocatable = v1.ResourceList{}
	}

	pluginWithLimits := kl.volumePluginMgr.ListVolumePluginWithLimits()
	for _, volumePlugin := range pluginWithLimits {
		attachLimits, err := volumePlugin.GetVolumeLimits()
		if err != nil {
			glog.V(4).Infof("Error getting volume limit for plugin %s", volumePlugin.GetPluginName())
			continue
		}
		for limitKey, value := range attachLimits {
			node.Status.Capacity[v1.ResourceName(limitKey)] = *resource.NewQuantity(value, resource.DecimalSI)
			node.Status.Allocatable[v1.ResourceName(limitKey)] = *resource.NewQuantity(value, resource.DecimalSI)
		}
	}
}

// syncNodeStatus should be called periodically from a goroutine.
// It synchronizes node status to master, registering the kubelet first if
// necessary.
func (kl *Kubelet) syncNodeStatus() {
	if kl.kubeClient == nil || kl.heartbeatClient == nil {
		return
	}
	if kl.registerNode {
		// This will exit immediately if it doesn't need to do anything.
		kl.registerWithAPIServer()
	}
	if err := kl.updateNodeStatus(); err != nil {
		glog.Errorf("Unable to update node status: %v", err)
	}
}

// updateNodeStatus updates node status to master with retries.
func (kl *Kubelet) updateNodeStatus() error {
	glog.V(5).Infof("Updating node status")
	for i := 0; i < nodeStatusUpdateRetry; i++ {
		if err := kl.tryUpdateNodeStatus(i); err != nil {
			if i > 0 && kl.onRepeatedHeartbeatFailure != nil {
				kl.onRepeatedHeartbeatFailure()
			}
			glog.Errorf("Error updating node status, will retry: %v", err)
		} else {
			return nil
		}
	}
	return fmt.Errorf("update node status exceeds retry count")
}

// tryUpdateNodeStatus tries to update node status to master. If ReconcileCBR0
// is set, this function will also confirm that cbr0 is configured correctly.
func (kl *Kubelet) tryUpdateNodeStatus(tryNumber int) error {
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
	node, err := kl.heartbeatClient.Nodes().Get(string(kl.nodeName), opts)
	if err != nil {
		return fmt.Errorf("error getting node %q: %v", kl.nodeName, err)
	}

	originalNode := node.DeepCopy()
	if originalNode == nil {
		return fmt.Errorf("nil %q node object", kl.nodeName)
	}

	if node.Spec.PodCIDR != "" {
		kl.updatePodCIDR(node.Spec.PodCIDR)
	}

	kl.setNodeStatus(node)
	// Patch the current status on the API server
	updatedNode, _, err := nodeutil.PatchNodeStatus(kl.heartbeatClient, types.NodeName(kl.nodeName), originalNode, node)
	if err != nil {
		return err
	}
	// If update finishes successfully, mark the volumeInUse as reportedInUse to indicate
	// those volumes are already updated in the node's status
	kl.volumeManager.MarkVolumesAsReportedInUse(updatedNode.Status.VolumesInUse)
	return nil
}

// recordNodeStatusEvent records an event of the given type with the given
// message for the node.
func (kl *Kubelet) recordNodeStatusEvent(eventType, event string) {
	glog.V(2).Infof("Recording %s event message for node %s", event, kl.nodeName)
	// TODO: This requires a transaction, either both node status is updated
	// and event is recorded or neither should happen, see issue #6055.
	kl.recorder.Eventf(kl.nodeRef, eventType, event, "Node %s status is now: %s", kl.nodeName, event)
}

// Set IP and hostname addresses for the node.
func (kl *Kubelet) setNodeAddress(node *v1.Node) error {
	if kl.nodeIP != nil {
		if err := kl.nodeIPValidator(kl.nodeIP); err != nil {
			return fmt.Errorf("failed to validate nodeIP: %v", err)
		}
		glog.V(2).Infof("Using node IP: %q", kl.nodeIP.String())
	}

	if kl.externalCloudProvider {
		if kl.nodeIP != nil {
			if node.ObjectMeta.Annotations == nil {
				node.ObjectMeta.Annotations = make(map[string]string)
			}
			node.ObjectMeta.Annotations[kubeletapis.AnnotationProvidedIPAddr] = kl.nodeIP.String()
		}
		// We rely on the external cloud provider to supply the addresses.
		return nil
	}
	if kl.cloud != nil {
		instances, ok := kl.cloud.Instances()
		if !ok {
			return fmt.Errorf("failed to get instances from cloud provider")
		}
		// TODO(roberthbailey): Can we do this without having credentials to talk
		// to the cloud provider?
		// TODO(justinsb): We can if CurrentNodeName() was actually CurrentNode() and returned an interface
		// TODO: If IP addresses couldn't be fetched from the cloud provider, should kubelet fallback on the other methods for getting the IP below?
		var nodeAddresses []v1.NodeAddress
		var err error

		// Make sure the instances.NodeAddresses returns even if the cloud provider API hangs for a long time
		func() {
			kl.cloudproviderRequestMux.Lock()
			if len(kl.cloudproviderRequestParallelism) > 0 {
				kl.cloudproviderRequestMux.Unlock()
				return
			}
			kl.cloudproviderRequestParallelism <- 0
			kl.cloudproviderRequestMux.Unlock()

			go func() {
				nodeAddresses, err = instances.NodeAddresses(context.TODO(), kl.nodeName)

				kl.cloudproviderRequestMux.Lock()
				<-kl.cloudproviderRequestParallelism
				kl.cloudproviderRequestMux.Unlock()

				kl.cloudproviderRequestSync <- 0
			}()
		}()

		select {
		case <-kl.cloudproviderRequestSync:
		case <-time.After(kl.cloudproviderRequestTimeout):
			err = fmt.Errorf("Timeout after %v", kl.cloudproviderRequestTimeout)
		}

		if err != nil {
			return fmt.Errorf("failed to get node address from cloud provider: %v", err)
		}
		if kl.nodeIP != nil {
			enforcedNodeAddresses := []v1.NodeAddress{}

			var nodeIPType v1.NodeAddressType
			for _, nodeAddress := range nodeAddresses {
				if nodeAddress.Address == kl.nodeIP.String() {
					enforcedNodeAddresses = append(enforcedNodeAddresses, v1.NodeAddress{Type: nodeAddress.Type, Address: nodeAddress.Address})
					nodeIPType = nodeAddress.Type
					break
				}
			}
			if len(enforcedNodeAddresses) > 0 {
				for _, nodeAddress := range nodeAddresses {
					if nodeAddress.Type != nodeIPType && nodeAddress.Type != v1.NodeHostName {
						enforcedNodeAddresses = append(enforcedNodeAddresses, v1.NodeAddress{Type: nodeAddress.Type, Address: nodeAddress.Address})
					}
				}

				enforcedNodeAddresses = append(enforcedNodeAddresses, v1.NodeAddress{Type: v1.NodeHostName, Address: kl.GetHostname()})
				node.Status.Addresses = enforcedNodeAddresses
				return nil
			}
			return fmt.Errorf("failed to get node address from cloud provider that matches ip: %v", kl.nodeIP)
		}

		// Only add a NodeHostName address if the cloudprovider did not specify one
		// (we assume the cloudprovider knows best)
		var addressNodeHostName *v1.NodeAddress
		for i := range nodeAddresses {
			if nodeAddresses[i].Type == v1.NodeHostName {
				addressNodeHostName = &nodeAddresses[i]
				break
			}
		}
		if addressNodeHostName == nil {
			hostnameAddress := v1.NodeAddress{Type: v1.NodeHostName, Address: kl.GetHostname()}
			nodeAddresses = append(nodeAddresses, hostnameAddress)
		} else {
			glog.V(2).Infof("Using Node Hostname from cloudprovider: %q", addressNodeHostName.Address)
		}
		node.Status.Addresses = nodeAddresses
	} else {
		var ipAddr net.IP
		var err error

		// 1) Use nodeIP if set
		// 2) If the user has specified an IP to HostnameOverride, use it
		// 3) Lookup the IP from node name by DNS and use the first valid IPv4 address.
		//    If the node does not have a valid IPv4 address, use the first valid IPv6 address.
		// 4) Try to get the IP from the network interface used as default gateway
		if kl.nodeIP != nil {
			ipAddr = kl.nodeIP
		} else if addr := net.ParseIP(kl.hostname); addr != nil {
			ipAddr = addr
		} else {
			var addrs []net.IP
			addrs, _ = net.LookupIP(node.Name)
			for _, addr := range addrs {
				if err = kl.nodeIPValidator(addr); err == nil {
					if addr.To4() != nil {
						ipAddr = addr
						break
					}
					if addr.To16() != nil && ipAddr == nil {
						ipAddr = addr
					}
				}
			}

			if ipAddr == nil {
				ipAddr, err = utilnet.ChooseHostInterface()
			}
		}

		if ipAddr == nil {
			// We tried everything we could, but the IP address wasn't fetchable; error out
			return fmt.Errorf("can't get ip address of node %s. error: %v", node.Name, err)
		}
		node.Status.Addresses = []v1.NodeAddress{
			{Type: v1.NodeInternalIP, Address: ipAddr.String()},
			{Type: v1.NodeHostName, Address: kl.GetHostname()},
		}
	}
	return nil
}

func (kl *Kubelet) setNodeStatusMachineInfo(node *v1.Node) {
	// Note: avoid blindly overwriting the capacity in case opaque
	//       resources are being advertised.
	if node.Status.Capacity == nil {
		node.Status.Capacity = v1.ResourceList{}
	}

	var devicePluginAllocatable v1.ResourceList
	var devicePluginCapacity v1.ResourceList
	var removedDevicePlugins []string

	// TODO: Post NotReady if we cannot get MachineInfo from cAdvisor. This needs to start
	// cAdvisor locally, e.g. for test-cmd.sh, and in integration test.
	info, err := kl.GetCachedMachineInfo()
	if err != nil {
		// TODO(roberthbailey): This is required for test-cmd.sh to pass.
		// See if the test should be updated instead.
		node.Status.Capacity[v1.ResourceCPU] = *resource.NewMilliQuantity(0, resource.DecimalSI)
		node.Status.Capacity[v1.ResourceMemory] = resource.MustParse("0Gi")
		node.Status.Capacity[v1.ResourcePods] = *resource.NewQuantity(int64(kl.maxPods), resource.DecimalSI)
		glog.Errorf("Error getting machine info: %v", err)
	} else {
		node.Status.NodeInfo.MachineID = info.MachineID
		node.Status.NodeInfo.SystemUUID = info.SystemUUID

		for rName, rCap := range cadvisor.CapacityFromMachineInfo(info) {
			node.Status.Capacity[rName] = rCap
		}

		if kl.podsPerCore > 0 {
			node.Status.Capacity[v1.ResourcePods] = *resource.NewQuantity(
				int64(math.Min(float64(info.NumCores*kl.podsPerCore), float64(kl.maxPods))), resource.DecimalSI)
		} else {
			node.Status.Capacity[v1.ResourcePods] = *resource.NewQuantity(
				int64(kl.maxPods), resource.DecimalSI)
		}

		if node.Status.NodeInfo.BootID != "" &&
			node.Status.NodeInfo.BootID != info.BootID {
			// TODO: This requires a transaction, either both node status is updated
			// and event is recorded or neither should happen, see issue #6055.
			kl.recorder.Eventf(kl.nodeRef, v1.EventTypeWarning, events.NodeRebooted,
				"Node %s has been rebooted, boot id: %s", kl.nodeName, info.BootID)
		}
		node.Status.NodeInfo.BootID = info.BootID

		if utilfeature.DefaultFeatureGate.Enabled(features.LocalStorageCapacityIsolation) {
			// TODO: all the node resources should use GetCapacity instead of deriving the
			// capacity for every node status request
			initialCapacity := kl.containerManager.GetCapacity()
			if initialCapacity != nil {
				node.Status.Capacity[v1.ResourceEphemeralStorage] = initialCapacity[v1.ResourceEphemeralStorage]
			}
		}

		devicePluginCapacity, devicePluginAllocatable, removedDevicePlugins = kl.containerManager.GetDevicePluginResourceCapacity()
		if devicePluginCapacity != nil {
			for k, v := range devicePluginCapacity {
				if old, ok := node.Status.Capacity[k]; !ok || old.Value() != v.Value() {
					glog.V(2).Infof("Update capacity for %s to %d", k, v.Value())
				}
				node.Status.Capacity[k] = v
			}
		}

		for _, removedResource := range removedDevicePlugins {
			glog.V(2).Infof("Set capacity for %s to 0 on device removal", removedResource)
			// Set the capacity of the removed resource to 0 instead of
			// removing the resource from the node status. This is to indicate
			// that the resource is managed by device plugin and had been
			// registered before.
			//
			// This is required to differentiate the device plugin managed
			// resources and the cluster-level resources, which are absent in
			// node status.
			node.Status.Capacity[v1.ResourceName(removedResource)] = *resource.NewQuantity(int64(0), resource.DecimalSI)
		}
	}

	// Set Allocatable.
	if node.Status.Allocatable == nil {
		node.Status.Allocatable = make(v1.ResourceList)
	}
	// Remove extended resources from allocatable that are no longer
	// present in capacity.
	for k := range node.Status.Allocatable {
		_, found := node.Status.Capacity[k]
		if !found && v1helper.IsExtendedResourceName(k) {
			delete(node.Status.Allocatable, k)
		}
	}
	allocatableReservation := kl.containerManager.GetNodeAllocatableReservation()
	for k, v := range node.Status.Capacity {
		value := *(v.Copy())
		if res, exists := allocatableReservation[k]; exists {
			value.Sub(res)
		}
		if value.Sign() < 0 {
			// Negative Allocatable resources don't make sense.
			value.Set(0)
		}
		node.Status.Allocatable[k] = value
	}

	if devicePluginAllocatable != nil {
		for k, v := range devicePluginAllocatable {
			if old, ok := node.Status.Allocatable[k]; !ok || old.Value() != v.Value() {
				glog.V(2).Infof("Update allocatable for %s to %d", k, v.Value())
			}
			node.Status.Allocatable[k] = v
		}
	}
	// for every huge page reservation, we need to remove it from allocatable memory
	for k, v := range node.Status.Capacity {
		if v1helper.IsHugePageResourceName(k) {
			allocatableMemory := node.Status.Allocatable[v1.ResourceMemory]
			value := *(v.Copy())
			allocatableMemory.Sub(value)
			if allocatableMemory.Sign() < 0 {
				// Negative Allocatable resources don't make sense.
				allocatableMemory.Set(0)
			}
			node.Status.Allocatable[v1.ResourceMemory] = allocatableMemory
		}
	}
}

// Set versioninfo for the node.
func (kl *Kubelet) setNodeStatusVersionInfo(node *v1.Node) {
	verinfo, err := kl.cadvisor.VersionInfo()
	if err != nil {
		glog.Errorf("Error getting version info: %v", err)
		return
	}

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

// Set daemonEndpoints for the node.
func (kl *Kubelet) setNodeStatusDaemonEndpoints(node *v1.Node) {
	node.Status.DaemonEndpoints = *kl.daemonEndpoints
}

// Set images list for the node
func (kl *Kubelet) setNodeStatusImages(node *v1.Node) {
	// Update image list of this node
	var imagesOnNode []v1.ContainerImage
	containerImages, err := kl.imageManager.GetImageList()
	if err != nil {
		glog.Errorf("Error getting image list: %v", err)
		node.Status.Images = imagesOnNode
		return
	}
	// sort the images from max to min, and only set top N images into the node status.
	if int(kl.nodeStatusMaxImages) > -1 &&
		int(kl.nodeStatusMaxImages) < len(containerImages) {
		containerImages = containerImages[0:kl.nodeStatusMaxImages]
	}

	for _, image := range containerImages {
		names := append(image.RepoDigests, image.RepoTags...)
		// Report up to maxNamesPerImageInNodeStatus names per image.
		if len(names) > maxNamesPerImageInNodeStatus {
			names = names[0:maxNamesPerImageInNodeStatus]
		}
		imagesOnNode = append(imagesOnNode, v1.ContainerImage{
			Names:     names,
			SizeBytes: image.Size,
		})
	}

	node.Status.Images = imagesOnNode
}

// Set the GOOS and GOARCH for this node
func (kl *Kubelet) setNodeStatusGoRuntime(node *v1.Node) {
	node.Status.NodeInfo.OperatingSystem = goruntime.GOOS
	node.Status.NodeInfo.Architecture = goruntime.GOARCH
}

// Set status for the node.
func (kl *Kubelet) setNodeStatusInfo(node *v1.Node) {
	kl.setNodeStatusMachineInfo(node)
	kl.setNodeStatusVersionInfo(node)
	kl.setNodeStatusDaemonEndpoints(node)
	kl.setNodeStatusImages(node)
	kl.setNodeStatusGoRuntime(node)
	if utilfeature.DefaultFeatureGate.Enabled(features.AttachVolumeLimit) {
		kl.setVolumeLimits(node)
	}
}

// Set Ready condition for the node.
func (kl *Kubelet) setNodeReadyCondition(node *v1.Node) {
	// NOTE(aaronlevy): NodeReady condition needs to be the last in the list of node conditions.
	// This is due to an issue with version skewed kubelet and master components.
	// ref: https://github.com/kubernetes/kubernetes/issues/16961
	currentTime := metav1.NewTime(kl.clock.Now())
	newNodeReadyCondition := v1.NodeCondition{
		Type:              v1.NodeReady,
		Status:            v1.ConditionTrue,
		Reason:            "KubeletReady",
		Message:           "kubelet is posting ready status",
		LastHeartbeatTime: currentTime,
	}
	rs := append(kl.runtimeState.runtimeErrors(), kl.runtimeState.networkErrors()...)
	requiredCapacities := []v1.ResourceName{v1.ResourceCPU, v1.ResourceMemory, v1.ResourcePods}
	if utilfeature.DefaultFeatureGate.Enabled(features.LocalStorageCapacityIsolation) {
		requiredCapacities = append(requiredCapacities, v1.ResourceEphemeralStorage)
	}
	missingCapacities := []string{}
	for _, resource := range requiredCapacities {
		if _, found := node.Status.Capacity[resource]; !found {
			missingCapacities = append(missingCapacities, string(resource))
		}
	}
	if len(missingCapacities) > 0 {
		rs = append(rs, fmt.Sprintf("Missing node capacity for resources: %s", strings.Join(missingCapacities, ", ")))
	}
	if len(rs) > 0 {
		newNodeReadyCondition = v1.NodeCondition{
			Type:              v1.NodeReady,
			Status:            v1.ConditionFalse,
			Reason:            "KubeletNotReady",
			Message:           strings.Join(rs, ","),
			LastHeartbeatTime: currentTime,
		}
	}
	// Append AppArmor status if it's enabled.
	// TODO(tallclair): This is a temporary message until node feature reporting is added.
	if newNodeReadyCondition.Status == v1.ConditionTrue &&
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
		if node.Status.Conditions[i].Type == v1.NodeReady {
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
		if newNodeReadyCondition.Status == v1.ConditionTrue {
			kl.recordNodeStatusEvent(v1.EventTypeNormal, events.NodeReady)
		} else {
			kl.recordNodeStatusEvent(v1.EventTypeNormal, events.NodeNotReady)
			glog.Infof("Node became not ready: %+v", newNodeReadyCondition)
		}
	}
}

// setNodeMemoryPressureCondition for the node.
// TODO: this needs to move somewhere centralized...
func (kl *Kubelet) setNodeMemoryPressureCondition(node *v1.Node) {
	currentTime := metav1.NewTime(kl.clock.Now())
	var condition *v1.NodeCondition

	// Check if NodeMemoryPressure condition already exists and if it does, just pick it up for update.
	for i := range node.Status.Conditions {
		if node.Status.Conditions[i].Type == v1.NodeMemoryPressure {
			condition = &node.Status.Conditions[i]
		}
	}

	newCondition := false
	// If the NodeMemoryPressure condition doesn't exist, create one
	if condition == nil {
		condition = &v1.NodeCondition{
			Type:   v1.NodeMemoryPressure,
			Status: v1.ConditionUnknown,
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
	// is created its status is set to v1.ConditionUnknown which matches either
	// condition.Status != v1.ConditionTrue or
	// condition.Status != v1.ConditionFalse in the conditions below depending on whether
	// the kubelet is under memory pressure or not.
	if kl.evictionManager.IsUnderMemoryPressure() {
		if condition.Status != v1.ConditionTrue {
			condition.Status = v1.ConditionTrue
			condition.Reason = "KubeletHasInsufficientMemory"
			condition.Message = "kubelet has insufficient memory available"
			condition.LastTransitionTime = currentTime
			kl.recordNodeStatusEvent(v1.EventTypeNormal, "NodeHasInsufficientMemory")
		}
	} else if condition.Status != v1.ConditionFalse {
		condition.Status = v1.ConditionFalse
		condition.Reason = "KubeletHasSufficientMemory"
		condition.Message = "kubelet has sufficient memory available"
		condition.LastTransitionTime = currentTime
		kl.recordNodeStatusEvent(v1.EventTypeNormal, "NodeHasSufficientMemory")
	}

	if newCondition {
		node.Status.Conditions = append(node.Status.Conditions, *condition)
	}
}

// setNodePIDPressureCondition for the node.
// TODO: this needs to move somewhere centralized...
func (kl *Kubelet) setNodePIDPressureCondition(node *v1.Node) {
	currentTime := metav1.NewTime(kl.clock.Now())
	var condition *v1.NodeCondition

	// Check if NodePIDPressure condition already exists and if it does, just pick it up for update.
	for i := range node.Status.Conditions {
		if node.Status.Conditions[i].Type == v1.NodePIDPressure {
			condition = &node.Status.Conditions[i]
		}
	}

	newCondition := false
	// If the NodePIDPressure condition doesn't exist, create one
	if condition == nil {
		condition = &v1.NodeCondition{
			Type:   v1.NodePIDPressure,
			Status: v1.ConditionUnknown,
		}
		// cannot be appended to node.Status.Conditions here because it gets
		// copied to the slice. So if we append to the slice here none of the
		// updates we make below are reflected in the slice.
		newCondition = true
	}

	// Update the heartbeat time
	condition.LastHeartbeatTime = currentTime

	// Note: The conditions below take care of the case when a new NodePIDPressure condition is
	// created and as well as the case when the condition already exists. When a new condition
	// is created its status is set to v1.ConditionUnknown which matches either
	// condition.Status != v1.ConditionTrue or
	// condition.Status != v1.ConditionFalse in the conditions below depending on whether
	// the kubelet is under PID pressure or not.
	if kl.evictionManager.IsUnderPIDPressure() {
		if condition.Status != v1.ConditionTrue {
			condition.Status = v1.ConditionTrue
			condition.Reason = "KubeletHasInsufficientPID"
			condition.Message = "kubelet has insufficient PID available"
			condition.LastTransitionTime = currentTime
			kl.recordNodeStatusEvent(v1.EventTypeNormal, "NodeHasInsufficientPID")
		}
	} else if condition.Status != v1.ConditionFalse {
		condition.Status = v1.ConditionFalse
		condition.Reason = "KubeletHasSufficientPID"
		condition.Message = "kubelet has sufficient PID available"
		condition.LastTransitionTime = currentTime
		kl.recordNodeStatusEvent(v1.EventTypeNormal, "NodeHasSufficientPID")
	}

	if newCondition {
		node.Status.Conditions = append(node.Status.Conditions, *condition)
	}
}

// setNodeDiskPressureCondition for the node.
// TODO: this needs to move somewhere centralized...
func (kl *Kubelet) setNodeDiskPressureCondition(node *v1.Node) {
	currentTime := metav1.NewTime(kl.clock.Now())
	var condition *v1.NodeCondition

	// Check if NodeDiskPressure condition already exists and if it does, just pick it up for update.
	for i := range node.Status.Conditions {
		if node.Status.Conditions[i].Type == v1.NodeDiskPressure {
			condition = &node.Status.Conditions[i]
		}
	}

	newCondition := false
	// If the NodeDiskPressure condition doesn't exist, create one
	if condition == nil {
		condition = &v1.NodeCondition{
			Type:   v1.NodeDiskPressure,
			Status: v1.ConditionUnknown,
		}
		// cannot be appended to node.Status.Conditions here because it gets
		// copied to the slice. So if we append to the slice here none of the
		// updates we make below are reflected in the slice.
		newCondition = true
	}

	// Update the heartbeat time
	condition.LastHeartbeatTime = currentTime

	// Note: The conditions below take care of the case when a new NodeDiskPressure condition is
	// created and as well as the case when the condition already exists. When a new condition
	// is created its status is set to v1.ConditionUnknown which matches either
	// condition.Status != v1.ConditionTrue or
	// condition.Status != v1.ConditionFalse in the conditions below depending on whether
	// the kubelet is under disk pressure or not.
	if kl.evictionManager.IsUnderDiskPressure() {
		if condition.Status != v1.ConditionTrue {
			condition.Status = v1.ConditionTrue
			condition.Reason = "KubeletHasDiskPressure"
			condition.Message = "kubelet has disk pressure"
			condition.LastTransitionTime = currentTime
			kl.recordNodeStatusEvent(v1.EventTypeNormal, "NodeHasDiskPressure")
		}
	} else if condition.Status != v1.ConditionFalse {
		condition.Status = v1.ConditionFalse
		condition.Reason = "KubeletHasNoDiskPressure"
		condition.Message = "kubelet has no disk pressure"
		condition.LastTransitionTime = currentTime
		kl.recordNodeStatusEvent(v1.EventTypeNormal, "NodeHasNoDiskPressure")
	}

	if newCondition {
		node.Status.Conditions = append(node.Status.Conditions, *condition)
	}
}

// Set OODCondition for the node.
func (kl *Kubelet) setNodeOODCondition(node *v1.Node) {
	currentTime := metav1.NewTime(kl.clock.Now())
	var nodeOODCondition *v1.NodeCondition

	// Check if NodeOutOfDisk condition already exists and if it does, just pick it up for update.
	for i := range node.Status.Conditions {
		if node.Status.Conditions[i].Type == v1.NodeOutOfDisk {
			nodeOODCondition = &node.Status.Conditions[i]
		}
	}

	newOODCondition := nodeOODCondition == nil
	if newOODCondition {
		nodeOODCondition = &v1.NodeCondition{}
	}
	if nodeOODCondition.Status != v1.ConditionFalse {
		nodeOODCondition.Type = v1.NodeOutOfDisk
		nodeOODCondition.Status = v1.ConditionFalse
		nodeOODCondition.Reason = "KubeletHasSufficientDisk"
		nodeOODCondition.Message = "kubelet has sufficient disk space available"
		nodeOODCondition.LastTransitionTime = currentTime
		kl.recordNodeStatusEvent(v1.EventTypeNormal, "NodeHasSufficientDisk")
	}

	// Update the heartbeat time irrespective of all the conditions.
	nodeOODCondition.LastHeartbeatTime = currentTime

	if newOODCondition {
		node.Status.Conditions = append(node.Status.Conditions, *nodeOODCondition)
	}
}

// Maintains Node.Spec.Unschedulable value from previous run of tryUpdateNodeStatus()
// TODO: why is this a package var?
var (
	oldNodeUnschedulable     bool
	oldNodeUnschedulableLock sync.Mutex
)

// record if node schedulable change.
func (kl *Kubelet) recordNodeSchedulableEvent(node *v1.Node) {
	oldNodeUnschedulableLock.Lock()
	defer oldNodeUnschedulableLock.Unlock()
	if oldNodeUnschedulable != node.Spec.Unschedulable {
		if node.Spec.Unschedulable {
			kl.recordNodeStatusEvent(v1.EventTypeNormal, events.NodeNotSchedulable)
		} else {
			kl.recordNodeStatusEvent(v1.EventTypeNormal, events.NodeSchedulable)
		}
		oldNodeUnschedulable = node.Spec.Unschedulable
	}
}

// Update VolumesInUse field in Node Status only after states are synced up at least once
// in volume reconciler.
func (kl *Kubelet) setNodeVolumesInUseStatus(node *v1.Node) {
	// Make sure to only update node status after reconciler starts syncing up states
	if kl.volumeManager.ReconcilerStatesHasBeenSynced() {
		node.Status.VolumesInUse = kl.volumeManager.GetVolumesInUse()
	}
}

// setNodeStatus fills in the Status fields of the given Node, overwriting
// any fields that are currently set.
// TODO(madhusudancs): Simplify the logic for setting node conditions and
// refactor the node status condition code out to a different file.
func (kl *Kubelet) setNodeStatus(node *v1.Node) {
	for i, f := range kl.setNodeStatusFuncs {
		glog.V(5).Infof("Setting node status at position %v", i)
		if err := f(node); err != nil {
			glog.Warningf("Failed to set some node status fields: %s", err)
		}
	}
}

// defaultNodeStatusFuncs is a factory that generates the default set of
// setNodeStatus funcs
func (kl *Kubelet) defaultNodeStatusFuncs() []func(*v1.Node) error {
	// initial set of node status update handlers, can be modified by Option's
	withoutError := func(f func(*v1.Node)) func(*v1.Node) error {
		return func(n *v1.Node) error {
			f(n)
			return nil
		}
	}
	return []func(*v1.Node) error{
		kl.setNodeAddress,
		withoutError(kl.setNodeStatusInfo),
		withoutError(kl.setNodeOODCondition),
		withoutError(kl.setNodeMemoryPressureCondition),
		withoutError(kl.setNodeDiskPressureCondition),
		withoutError(kl.setNodePIDPressureCondition),
		withoutError(kl.setNodeReadyCondition),
		withoutError(kl.setNodeVolumesInUseStatus),
		withoutError(kl.recordNodeSchedulableEvent),
	}
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
	return fmt.Errorf("Node IP: %q not found in the host's network interfaces", nodeIP.String())
}
