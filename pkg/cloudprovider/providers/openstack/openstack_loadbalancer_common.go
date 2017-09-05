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

package openstack

import (
	"fmt"
	"strings"
	"time"

	"github.com/golang/glog"
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/layer3/floatingips"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas/pools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas/vips"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas_v2/listeners"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas_v2/loadbalancers"
	v2pools "github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas_v2/pools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/security/rules"
	"github.com/gophercloud/gophercloud/pagination"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"
)

// Note: when creating a new Loadbalancer (VM), it can take some time before it is ready for use,
// this timeout is used for waiting until the Loadbalancer provisioning status goes to ACTIVE state.
const (
	// loadbalancerActive* is configuration of exponential backoff for
	// going into ACTIVE loadbalancer provisioning status. Starting with 1
	// seconds, multiplying by 1.2 with each step and taking 19 steps at maximum
	// it will time out after 128s, which roughly corresponds to 120s
	loadbalancerActiveInitDealy = 1 * time.Second
	loadbalancerActiveFactor    = 1.2
	loadbalancerActiveSteps     = 19

	// loadbalancerDelete* is configuration of exponential backoff for
	// waiting for delete operation to complete. Starting with 1
	// seconds, multiplying by 1.2 with each step and taking 13 steps at maximum
	// it will time out after 32s, which roughly corresponds to 30s
	loadbalancerDeleteInitDealy = 1 * time.Second
	loadbalancerDeleteFactor    = 1.2
	loadbalancerDeleteSteps     = 13

	activeStatus = "ACTIVE"
	errorStatus  = "ERROR"

	ServiceAnnotationLoadBalancerFloatingNetworkId = "loadbalancer.openstack.org/floating-network-id"

	// ServiceAnnotationLoadBalancerInternal is the annotation used on the service
	// to indicate that we want an internal loadbalancer service.
	// If the value of ServiceAnnotationLoadBalancerInternal is false, it indicates that we want an external loadbalancer service. Default to true.
	ServiceAnnotationLoadBalancerInternal = "service.beta.kubernetes.io/openstack-internal-load-balancer"
)

func getFloatingIPByPortID(client *gophercloud.ServiceClient, portID string) (*floatingips.FloatingIP, error) {
	opts := floatingips.ListOpts{
		PortID: portID,
	}
	pager := floatingips.List(client, opts)

	floatingIPList := make([]floatingips.FloatingIP, 0, 1)

	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		f, err := floatingips.ExtractFloatingIPs(page)
		if err != nil {
			return false, err
		}
		floatingIPList = append(floatingIPList, f...)
		if len(floatingIPList) > 1 {
			return false, ErrMultipleResults
		}
		return true, nil
	})
	if err != nil {
		if isNotFound(err) {
			return nil, ErrNotFound
		}
		return nil, err
	}

	if len(floatingIPList) == 0 {
		return nil, ErrNotFound
	} else if len(floatingIPList) > 1 {
		return nil, ErrMultipleResults
	}

	return &floatingIPList[0], nil
}

func getPoolByName(client *gophercloud.ServiceClient, name string) (*pools.Pool, error) {
	opts := pools.ListOpts{
		Name: name,
	}
	pager := pools.List(client, opts)

	poolList := make([]pools.Pool, 0, 1)

	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		p, err := pools.ExtractPools(page)
		if err != nil {
			return false, err
		}
		poolList = append(poolList, p...)
		if len(poolList) > 1 {
			return false, ErrMultipleResults
		}
		return true, nil
	})
	if err != nil {
		if isNotFound(err) {
			return nil, ErrNotFound
		}
		return nil, err
	}

	if len(poolList) == 0 {
		return nil, ErrNotFound
	} else if len(poolList) > 1 {
		return nil, ErrMultipleResults
	}

	return &poolList[0], nil
}

func getVipByName(client *gophercloud.ServiceClient, name string) (*vips.VirtualIP, error) {
	opts := vips.ListOpts{
		Name: name,
	}
	pager := vips.List(client, opts)

	vipList := make([]vips.VirtualIP, 0, 1)

	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		v, err := vips.ExtractVIPs(page)
		if err != nil {
			return false, err
		}
		vipList = append(vipList, v...)
		if len(vipList) > 1 {
			return false, ErrMultipleResults
		}
		return true, nil
	})
	if err != nil {
		if isNotFound(err) {
			return nil, ErrNotFound
		}
		return nil, err
	}

	if len(vipList) == 0 {
		return nil, ErrNotFound
	} else if len(vipList) > 1 {
		return nil, ErrMultipleResults
	}

	return &vipList[0], nil
}

func getLoadbalancerByName(client *gophercloud.ServiceClient, name string) (*loadbalancers.LoadBalancer, error) {
	opts := loadbalancers.ListOpts{
		Name: name,
	}
	pager := loadbalancers.List(client, opts)

	loadbalancerList := make([]loadbalancers.LoadBalancer, 0, 1)

	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		v, err := loadbalancers.ExtractLoadBalancers(page)
		if err != nil {
			return false, err
		}
		loadbalancerList = append(loadbalancerList, v...)
		if len(loadbalancerList) > 1 {
			return false, ErrMultipleResults
		}
		return true, nil
	})
	if err != nil {
		if isNotFound(err) {
			return nil, ErrNotFound
		}
		return nil, err
	}

	if len(loadbalancerList) == 0 {
		return nil, ErrNotFound
	} else if len(loadbalancerList) > 1 {
		return nil, ErrMultipleResults
	}

	return &loadbalancerList[0], nil
}

func getListenersByLoadBalancerID(client *gophercloud.ServiceClient, id string) ([]listeners.Listener, error) {
	var existingListeners []listeners.Listener
	err := listeners.List(client, listeners.ListOpts{LoadbalancerID: id}).EachPage(func(page pagination.Page) (bool, error) {
		listenerList, err := listeners.ExtractListeners(page)
		if err != nil {
			return false, err
		}
		for _, l := range listenerList {
			for _, lb := range l.Loadbalancers {
				if lb.ID == id {
					existingListeners = append(existingListeners, l)
					break
				}
			}
		}

		return true, nil
	})
	if err != nil {
		return nil, err
	}

	return existingListeners, nil
}

// get listener for a port or nil if does not exist
func getListenerForPort(existingListeners []listeners.Listener, port v1.ServicePort) *listeners.Listener {
	for _, l := range existingListeners {
		if listeners.Protocol(l.Protocol) == toListenersProtocol(port.Protocol) && l.ProtocolPort == int(port.Port) {
			return &l
		}
	}

	return nil
}

// Get pool for a listener. A listener always has exactly one pool.
func getPoolByListenerID(client *gophercloud.ServiceClient, loadbalancerID string, listenerID string) (*v2pools.Pool, error) {
	listenerPools := make([]v2pools.Pool, 0, 1)
	err := v2pools.List(client, v2pools.ListOpts{LoadbalancerID: loadbalancerID}).EachPage(func(page pagination.Page) (bool, error) {
		poolsList, err := v2pools.ExtractPools(page)
		if err != nil {
			return false, err
		}
		for _, p := range poolsList {
			for _, l := range p.Listeners {
				if l.ID == listenerID {
					listenerPools = append(listenerPools, p)
				}
			}
		}
		if len(listenerPools) > 1 {
			return false, ErrMultipleResults
		}
		return true, nil
	})
	if err != nil {
		if isNotFound(err) {
			return nil, ErrNotFound
		}
		return nil, err
	}

	if len(listenerPools) == 0 {
		return nil, ErrNotFound
	} else if len(listenerPools) > 1 {
		return nil, ErrMultipleResults
	}

	return &listenerPools[0], nil
}

func getMembersByPoolID(client *gophercloud.ServiceClient, id string) ([]v2pools.Member, error) {
	var members []v2pools.Member
	err := v2pools.ListMembers(client, id, v2pools.ListMembersOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		membersList, err := v2pools.ExtractMembers(page)
		if err != nil {
			return false, err
		}
		members = append(members, membersList...)

		return true, nil
	})
	if err != nil {
		return nil, err
	}

	return members, nil
}

// Check if a member exists for node
func memberExists(members []v2pools.Member, addr string, port int) bool {
	for _, member := range members {
		if member.Address == addr && member.ProtocolPort == port {
			return true
		}
	}

	return false
}

func popListener(existingListeners []listeners.Listener, id string) []listeners.Listener {
	for i, existingListener := range existingListeners {
		if existingListener.ID == id {
			existingListeners[i] = existingListeners[len(existingListeners)-1]
			existingListeners = existingListeners[:len(existingListeners)-1]
			break
		}
	}

	return existingListeners
}

func popMember(members []v2pools.Member, addr string, port int) []v2pools.Member {
	for i, member := range members {
		if member.Address == addr && member.ProtocolPort == port {
			members[i] = members[len(members)-1]
			members = members[:len(members)-1]
		}
	}

	return members
}

func getSecurityGroupName(clusterName string, service *v1.Service) string {
	return fmt.Sprintf("lb-sg-%s-%v", clusterName, service.Name)
}

func getSecurityGroupRules(client *gophercloud.ServiceClient, opts rules.ListOpts) ([]rules.SecGroupRule, error) {

	pager := rules.List(client, opts)

	var securityRules []rules.SecGroupRule

	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		ruleList, err := rules.ExtractRules(page)
		if err != nil {
			return false, err
		}
		securityRules = append(securityRules, ruleList...)
		return true, nil
	})

	if err != nil {
		return nil, err
	}

	return securityRules, nil
}

func waitLoadbalancerActiveProvisioningStatus(client *gophercloud.ServiceClient, loadbalancerID string) (string, error) {
	backoff := wait.Backoff{
		Duration: loadbalancerActiveInitDealy,
		Factor:   loadbalancerActiveFactor,
		Steps:    loadbalancerActiveSteps,
	}

	var provisioningStatus string
	err := wait.ExponentialBackoff(backoff, func() (bool, error) {
		loadbalancer, err := loadbalancers.Get(client, loadbalancerID).Extract()
		if err != nil {
			return false, err
		}
		provisioningStatus = loadbalancer.ProvisioningStatus
		if loadbalancer.ProvisioningStatus == activeStatus {
			return true, nil
		} else if loadbalancer.ProvisioningStatus == errorStatus {
			return true, fmt.Errorf("Loadbalancer has gone into ERROR state")
		} else {
			return false, nil
		}

	})

	if err == wait.ErrWaitTimeout {
		err = fmt.Errorf("Loadbalancer failed to go into ACTIVE provisioning status within alloted time")
	}
	return provisioningStatus, err
}

func waitLoadbalancerDeleted(client *gophercloud.ServiceClient, loadbalancerID string) error {
	backoff := wait.Backoff{
		Duration: loadbalancerDeleteInitDealy,
		Factor:   loadbalancerDeleteFactor,
		Steps:    loadbalancerDeleteSteps,
	}
	err := wait.ExponentialBackoff(backoff, func() (bool, error) {
		_, err := loadbalancers.Get(client, loadbalancerID).Extract()
		if err != nil {
			if err == ErrNotFound {
				return true, nil
			} else {
				return false, err
			}
		} else {
			return false, nil
		}
	})

	if err == wait.ErrWaitTimeout {
		err = fmt.Errorf("Loadbalancer failed to delete within the alloted time")
	}

	return err
}

func toRuleProtocol(protocol v1.Protocol) rules.RuleProtocol {
	switch protocol {
	case v1.ProtocolTCP:
		return rules.ProtocolTCP
	case v1.ProtocolUDP:
		return rules.ProtocolUDP
	default:
		return rules.RuleProtocol(strings.ToLower(string(protocol)))
	}
}

func toListenersProtocol(protocol v1.Protocol) listeners.Protocol {
	switch protocol {
	case v1.ProtocolTCP:
		return listeners.ProtocolTCP
	default:
		return listeners.Protocol(string(protocol))
	}
}

func createNodeSecurityGroup(client *gophercloud.ServiceClient, nodeSecurityGroupID string, port int, protocol v1.Protocol, lbSecGroup string) error {
	v4NodeSecGroupRuleCreateOpts := rules.CreateOpts{
		Direction:     rules.DirIngress,
		PortRangeMax:  port,
		PortRangeMin:  port,
		Protocol:      toRuleProtocol(protocol),
		RemoteGroupID: lbSecGroup,
		SecGroupID:    nodeSecurityGroupID,
		EtherType:     rules.EtherType4,
	}

	v6NodeSecGroupRuleCreateOpts := rules.CreateOpts{
		Direction:     rules.DirIngress,
		PortRangeMax:  port,
		PortRangeMin:  port,
		Protocol:      toRuleProtocol(protocol),
		RemoteGroupID: lbSecGroup,
		SecGroupID:    nodeSecurityGroupID,
		EtherType:     rules.EtherType6,
	}

	_, err := rules.Create(client, v4NodeSecGroupRuleCreateOpts).Extract()

	if err != nil {
		return err
	}

	_, err = rules.Create(client, v6NodeSecGroupRuleCreateOpts).Extract()

	if err != nil {
		return err
	}
	return nil
}

// The LB needs to be configured with instance addresses on the same
// subnet as the LB (aka opts.SubnetId).  Currently we're just
// guessing that the node's InternalIP is the right address - and that
// should be sufficient for all "normal" cases.
func nodeAddressForLB(node *v1.Node) (string, error) {
	addrs := node.Status.Addresses
	if len(addrs) == 0 {
		return "", ErrNoAddressFound
	}

	for _, addr := range addrs {
		if addr.Type == v1.NodeInternalIP {
			return addr.Address, nil
		}
	}

	return addrs[0].Address, nil
}

//getStringFromServiceAnnotation searches a given v1.Service for a specific annotationKey and either returns the annotation's value or a specified defaultSetting
func getStringFromServiceAnnotation(service *v1.Service, annotationKey string, defaultSetting string) string {
	glog.V(4).Infof("getStringFromServiceAnnotation(%v, %v, %v)", service, annotationKey, defaultSetting)
	if annotationValue, ok := service.Annotations[annotationKey]; ok {
		//if there is an annotation for this setting, set the "setting" var to it
		// annotationValue can be empty, it is working as designed
		// it makes possible for instance provisioning loadbalancer without floatingip
		glog.V(4).Infof("Found a Service Annotation: %v = %v", annotationKey, annotationValue)
		return annotationValue
	}
	//if there is no annotation, set "settings" var to the value from cloud config
	glog.V(4).Infof("Could not find a Service Annotation; falling back on cloud-config setting: %v = %v", annotationKey, defaultSetting)
	return defaultSetting
}

// getSubnetIDForLB returns subnet-id for a specific node
func getSubnetIDForLB(compute *gophercloud.ServiceClient, node v1.Node) (string, error) {
	ipAddress, err := nodeAddressForLB(&node)
	if err != nil {
		return "", err
	}

	instanceID := node.Spec.ProviderID
	if ind := strings.LastIndex(instanceID, "/"); ind >= 0 {
		instanceID = instanceID[(ind + 1):]
	}

	interfaces, err := getAttachedInterfacesByID(compute, instanceID)
	if err != nil {
		return "", err
	}

	for _, intf := range interfaces {
		for _, fixedIP := range intf.FixedIPs {
			if fixedIP.IPAddress == ipAddress {
				return intf.NetID, nil
			}
		}
	}

	return "", ErrNotFound
}
