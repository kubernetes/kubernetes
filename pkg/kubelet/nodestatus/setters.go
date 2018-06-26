/*
Copyright 2018 The Kubernetes Authors.

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

package nodestatus

import (
	"fmt"
	"net"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/kubernetes/pkg/cloudprovider"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"

	"github.com/golang/glog"
)

// Setter modifies the node in-place, and returns an error if the modification failed.
// Setters may partially mutate the node before returning an error.
type Setter func(node *v1.Node) error

// NodeAddress returns a Setter that updates address-related information on the node.
func NodeAddress(nodeIP net.IP, // typically Kubelet.nodeIP
	validateNodeIPFunc func(net.IP) error, // typically Kubelet.nodeIPValidator
	hostname string, // typically Kubelet.hostname
	externalCloudProvider bool, // typically Kubelet.externalCloudProvider
	cloud cloudprovider.Interface, // typically Kubelet.cloud
	nodeAddressesFunc func() ([]v1.NodeAddress, error), // typically Kubelet.cloudResourceSyncManager.NodeAddresses
) Setter {
	return func(node *v1.Node) error {
		if nodeIP != nil {
			if err := validateNodeIPFunc(nodeIP); err != nil {
				return fmt.Errorf("failed to validate nodeIP: %v", err)
			}
			glog.V(2).Infof("Using node IP: %q", nodeIP.String())
		}

		if externalCloudProvider {
			if nodeIP != nil {
				if node.ObjectMeta.Annotations == nil {
					node.ObjectMeta.Annotations = make(map[string]string)
				}
				node.ObjectMeta.Annotations[kubeletapis.AnnotationProvidedIPAddr] = nodeIP.String()
			}
			// We rely on the external cloud provider to supply the addresses.
			return nil
		}
		if cloud != nil {
			nodeAddresses, err := nodeAddressesFunc()
			if err != nil {
				return err
			}
			if nodeIP != nil {
				enforcedNodeAddresses := []v1.NodeAddress{}

				var nodeIPType v1.NodeAddressType
				for _, nodeAddress := range nodeAddresses {
					if nodeAddress.Address == nodeIP.String() {
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

					enforcedNodeAddresses = append(enforcedNodeAddresses, v1.NodeAddress{Type: v1.NodeHostName, Address: hostname})
					node.Status.Addresses = enforcedNodeAddresses
					return nil
				}
				return fmt.Errorf("failed to get node address from cloud provider that matches ip: %v", nodeIP)
			}

			// Only add a NodeHostName address if the cloudprovider did not specify any addresses.
			// (we assume the cloudprovider is authoritative if it specifies any addresses)
			if len(nodeAddresses) == 0 {
				nodeAddresses = []v1.NodeAddress{{Type: v1.NodeHostName, Address: hostname}}
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
			if nodeIP != nil {
				ipAddr = nodeIP
			} else if addr := net.ParseIP(hostname); addr != nil {
				ipAddr = addr
			} else {
				var addrs []net.IP
				addrs, _ = net.LookupIP(node.Name)
				for _, addr := range addrs {
					if err = validateNodeIPFunc(addr); err == nil {
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
				{Type: v1.NodeHostName, Address: hostname},
			}
		}
		return nil
	}
}

// MemoryPressureCondition returns a Setter that updates the v1.NodeMemoryPressure condition on the node.
func MemoryPressureCondition(nowFunc func() time.Time, // typically Kubelet.clock.Now
	pressureFunc func() bool, // typically Kubelet.evictionManager.IsUnderMemoryPressure
	recordEventFunc func(eventType, event string), // typically Kubelet.recordNodeStatusEvent
) Setter {
	return func(node *v1.Node) error {
		currentTime := metav1.NewTime(nowFunc())
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
		if pressureFunc() {
			if condition.Status != v1.ConditionTrue {
				condition.Status = v1.ConditionTrue
				condition.Reason = "KubeletHasInsufficientMemory"
				condition.Message = "kubelet has insufficient memory available"
				condition.LastTransitionTime = currentTime
				recordEventFunc(v1.EventTypeNormal, "NodeHasInsufficientMemory")
			}
		} else if condition.Status != v1.ConditionFalse {
			condition.Status = v1.ConditionFalse
			condition.Reason = "KubeletHasSufficientMemory"
			condition.Message = "kubelet has sufficient memory available"
			condition.LastTransitionTime = currentTime
			recordEventFunc(v1.EventTypeNormal, "NodeHasSufficientMemory")
		}

		if newCondition {
			node.Status.Conditions = append(node.Status.Conditions, *condition)
		}
		return nil
	}
}

// OutOfDiskCondition returns a Setter that updates the v1.NodeOutOfDisk condition on the node.
// TODO(#65658): remove this condition
func OutOfDiskCondition(nowFunc func() time.Time, // typically Kubelet.clock.Now
	recordEventFunc func(eventType, event string), // typically Kubelet.recordNodeStatusEvent
) Setter {
	return func(node *v1.Node) error {
		currentTime := metav1.NewTime(nowFunc())
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
			recordEventFunc(v1.EventTypeNormal, "NodeHasSufficientDisk")
		}

		// Update the heartbeat time irrespective of all the conditions.
		nodeOODCondition.LastHeartbeatTime = currentTime

		if newOODCondition {
			node.Status.Conditions = append(node.Status.Conditions, *nodeOODCondition)
		}
		return nil
	}
}
