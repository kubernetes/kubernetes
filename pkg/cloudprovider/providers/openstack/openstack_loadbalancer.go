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
	"net"
	"strings"
	"time"

	"github.com/golang/glog"
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/layer3/floatingips"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas_v2/listeners"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas_v2/loadbalancers"
	v2monitors "github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas_v2/monitors"
	v2pools "github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/lbaas_v2/pools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/security/groups"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/security/rules"
	neutronports "github.com/gophercloud/gophercloud/openstack/networking/v2/ports"
	"github.com/gophercloud/gophercloud/pagination"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/api/v1/service"
	"k8s.io/kubernetes/pkg/cloudprovider"
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
	// If the value of ServiceAnnotationLoadBalancerInternal is false, it indicates that we want an external loadbalancer service. Default to false.
	ServiceAnnotationLoadBalancerInternal = "service.beta.kubernetes.io/openstack-internal-load-balancer"
)

// LoadBalancer implementation for LBaaS v2
type LbaasV2 struct {
	LoadBalancer
}

type empty struct{}

func networkExtensions(client *gophercloud.ServiceClient) (map[string]bool, error) {
	seen := make(map[string]bool)

	pager := extensions.List(client)
	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		exts, err := extensions.ExtractExtensions(page)
		if err != nil {
			return false, err
		}
		for _, ext := range exts {
			seen[ext.Alias] = true
		}
		return true, nil
	})

	return seen, err
}

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
	return fmt.Sprintf("lb-sg-%s-%s-%s", clusterName, service.Namespace, service.Name)
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

func (lbaas *LbaasV2) createLoadBalancer(service *v1.Service, name string, internalAnnotation bool) (*loadbalancers.LoadBalancer, error) {
	createOpts := loadbalancers.CreateOpts{
		Name:        name,
		Description: fmt.Sprintf("Kubernetes external service %s", name),
		VipSubnetID: lbaas.opts.SubnetId,
		Provider:    lbaas.opts.LBProvider,
	}

	loadBalancerIP := service.Spec.LoadBalancerIP
	if loadBalancerIP != "" && internalAnnotation {
		createOpts.VipAddress = loadBalancerIP
	}

	loadbalancer, err := loadbalancers.Create(lbaas.network, createOpts).Extract()
	if err != nil {
		return nil, fmt.Errorf("Error creating loadbalancer %v: %v", createOpts, err)
	}
	return loadbalancer, nil
}

func (lbaas *LbaasV2) GetLoadBalancer(clusterName string, service *v1.Service) (*v1.LoadBalancerStatus, bool, error) {
	loadBalancerName := cloudprovider.GetLoadBalancerName(service)
	loadbalancer, err := getLoadbalancerByName(lbaas.network, loadBalancerName)
	if err == ErrNotFound {
		return nil, false, nil
	}
	if loadbalancer == nil {
		return nil, false, err
	}

	status := &v1.LoadBalancerStatus{}

	portID := loadbalancer.VipPortID
	if portID != "" {
		floatIP, err := getFloatingIPByPortID(lbaas.network, portID)
		if err != nil {
			return nil, false, fmt.Errorf("Error getting floating ip for port %s: %v", portID, err)
		}
		status.Ingress = []v1.LoadBalancerIngress{{IP: floatIP.FloatingIP}}
	} else {
		status.Ingress = []v1.LoadBalancerIngress{{IP: loadbalancer.VipAddress}}
	}

	return status, true, err
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

// getNodeSecurityGroupIDForLB lists node-security-groups for specific nodes
func getNodeSecurityGroupIDForLB(compute *gophercloud.ServiceClient, nodes []*v1.Node) ([]string, error) {
	nodeSecurityGroupIDs := sets.NewString()

	for _, node := range nodes {
		nodeName := types.NodeName(node.Name)
		srv, err := getServerByName(compute, nodeName)
		if err != nil {
			return nodeSecurityGroupIDs.List(), err
		}

		// use the first node-security-groups
		// case 0: node1:SG1  node2:SG1  return SG1
		// case 1: node1:SG1  node2:SG2  return SG1,SG2
		// case 2: node1:SG1,SG2  node2:SG3,SG4  return SG1,SG3
		// case 3: node1:SG1,SG2  node2:SG2,SG3  return SG1,SG2
		securityGroupName := srv.SecurityGroups[0]["name"]
		nodeSecurityGroupIDs.Insert(securityGroupName.(string))
	}

	return nodeSecurityGroupIDs.List(), nil
}

// TODO: This code currently ignores 'region' and always creates a
// loadbalancer in only the current OpenStack region.  We should take
// a list of regions (from config) and query/create loadbalancers in
// each region.

func (lbaas *LbaasV2) EnsureLoadBalancer(clusterName string, apiService *v1.Service, nodes []*v1.Node) (*v1.LoadBalancerStatus, error) {
	glog.V(4).Infof("EnsureLoadBalancer(%v, %v, %v, %v, %v, %v, %v)", clusterName, apiService.Namespace, apiService.Name, apiService.Spec.LoadBalancerIP, apiService.Spec.Ports, nodes, apiService.Annotations)

	if len(nodes) == 0 {
		return nil, fmt.Errorf("there are no available nodes for LoadBalancer service %s/%s", apiService.Namespace, apiService.Name)
	}

	if len(lbaas.opts.SubnetId) == 0 {
		// Get SubnetId automatically.
		// The LB needs to be configured with instance addresses on the same subnet, so get SubnetId by one node.
		subnetID, err := getSubnetIDForLB(lbaas.compute, *nodes[0])
		if err != nil {
			glog.Warningf("Failed to find subnet-id for loadbalancer service %s/%s: %v", apiService.Namespace, apiService.Name, err)
			return nil, fmt.Errorf("no subnet-id for service %s/%s : subnet-id not set in cloud provider config, "+
				"and failed to find subnet-id from OpenStack: %v", apiService.Namespace, apiService.Name, err)
		}
		lbaas.opts.SubnetId = subnetID
	}

	ports := apiService.Spec.Ports
	if len(ports) == 0 {
		return nil, fmt.Errorf("no ports provided to openstack load balancer")
	}

	floatingPool := getStringFromServiceAnnotation(apiService, ServiceAnnotationLoadBalancerFloatingNetworkId, lbaas.opts.FloatingNetworkId)
	glog.V(4).Infof("EnsureLoadBalancer using floatingPool: %v", floatingPool)

	var internalAnnotation bool
	internal := getStringFromServiceAnnotation(apiService, ServiceAnnotationLoadBalancerInternal, "false")
	switch internal {
	case "true":
		glog.V(4).Infof("Ensure an internal loadbalancer service.")
		internalAnnotation = true
	case "false":
		if len(floatingPool) != 0 {
			glog.V(4).Infof("Ensure an external loadbalancer service.")
			internalAnnotation = false
		} else {
			return nil, fmt.Errorf("floating-network-id or loadbalancer.openstack.org/floating-network-id should be specified when ensuring an external loadbalancer service")
		}
	default:
		return nil, fmt.Errorf("unknown service.beta.kubernetes.io/openstack-internal-load-balancer annotation: %v, specify \"true\" or \"false\" ",
			internal)
	}

	// Check for TCP protocol on each port
	// TODO: Convert all error messages to use an event recorder
	for _, port := range ports {
		if port.Protocol != v1.ProtocolTCP {
			return nil, fmt.Errorf("only TCP LoadBalancer is supported for openstack load balancers")
		}
	}

	sourceRanges, err := service.GetLoadBalancerSourceRanges(apiService)
	if err != nil {
		return nil, fmt.Errorf("failed to get source ranges for loadbalancer service %s/%s: %v", apiService.Namespace, apiService.Name, err)
	}

	if !service.IsAllowAll(sourceRanges) && !lbaas.opts.ManageSecurityGroups {
		return nil, fmt.Errorf("source range restrictions are not supported for openstack load balancers without managing security groups")
	}

	affinity := apiService.Spec.SessionAffinity
	var persistence *v2pools.SessionPersistence
	switch affinity {
	case v1.ServiceAffinityNone:
		persistence = nil
	case v1.ServiceAffinityClientIP:
		persistence = &v2pools.SessionPersistence{Type: "SOURCE_IP"}
	default:
		return nil, fmt.Errorf("unsupported load balancer affinity: %v", affinity)
	}

	name := cloudprovider.GetLoadBalancerName(apiService)
	loadbalancer, err := getLoadbalancerByName(lbaas.network, name)
	if err != nil {
		if err != ErrNotFound {
			return nil, fmt.Errorf("error getting loadbalancer %s: %v", name, err)
		}
		glog.V(2).Infof("Creating loadbalancer %s", name)
		loadbalancer, err = lbaas.createLoadBalancer(apiService, name, internalAnnotation)
		if err != nil {
			// Unknown error, retry later
			return nil, fmt.Errorf("error creating loadbalancer %s: %v", name, err)
		}
	} else {
		glog.V(2).Infof("LoadBalancer %s already exists", name)
	}

	waitLoadbalancerActiveProvisioningStatus(lbaas.network, loadbalancer.ID)

	lbmethod := v2pools.LBMethod(lbaas.opts.LBMethod)
	if lbmethod == "" {
		lbmethod = v2pools.LBMethodRoundRobin
	}

	oldListeners, err := getListenersByLoadBalancerID(lbaas.network, loadbalancer.ID)
	if err != nil {
		return nil, fmt.Errorf("error getting LB %s listeners: %v", name, err)
	}
	for portIndex, port := range ports {
		listener := getListenerForPort(oldListeners, port)
		if listener == nil {
			glog.V(4).Infof("Creating listener for port %d", int(port.Port))
			listener, err = listeners.Create(lbaas.network, listeners.CreateOpts{
				Name:           fmt.Sprintf("listener_%s_%d", name, portIndex),
				Protocol:       listeners.Protocol(port.Protocol),
				ProtocolPort:   int(port.Port),
				LoadbalancerID: loadbalancer.ID,
			}).Extract()
			if err != nil {
				// Unknown error, retry later
				return nil, fmt.Errorf("error creating LB listener: %v", err)
			}
			waitLoadbalancerActiveProvisioningStatus(lbaas.network, loadbalancer.ID)
		}

		glog.V(4).Infof("Listener for %s port %d: %s", string(port.Protocol), int(port.Port), listener.ID)

		// After all ports have been processed, remaining listeners are removed as obsolete.
		// Pop valid listeners.
		oldListeners = popListener(oldListeners, listener.ID)
		pool, err := getPoolByListenerID(lbaas.network, loadbalancer.ID, listener.ID)
		if err != nil && err != ErrNotFound {
			// Unknown error, retry later
			return nil, fmt.Errorf("error getting pool for listener %s: %v", listener.ID, err)
		}
		if pool == nil {
			glog.V(4).Infof("Creating pool for listener %s", listener.ID)
			pool, err = v2pools.Create(lbaas.network, v2pools.CreateOpts{
				Name:        fmt.Sprintf("pool_%s_%d", name, portIndex),
				Protocol:    v2pools.Protocol(port.Protocol),
				LBMethod:    lbmethod,
				ListenerID:  listener.ID,
				Persistence: persistence,
			}).Extract()
			if err != nil {
				// Unknown error, retry later
				return nil, fmt.Errorf("error creating pool for listener %s: %v", listener.ID, err)
			}
			waitLoadbalancerActiveProvisioningStatus(lbaas.network, loadbalancer.ID)
		}

		glog.V(4).Infof("Pool for listener %s: %s", listener.ID, pool.ID)
		members, err := getMembersByPoolID(lbaas.network, pool.ID)
		if err != nil && !isNotFound(err) {
			return nil, fmt.Errorf("error getting pool members %s: %v", pool.ID, err)
		}
		for _, node := range nodes {
			addr, err := nodeAddressForLB(node)
			if err != nil {
				if err == ErrNotFound {
					// Node failure, do not create member
					glog.Warningf("Failed to create LB pool member for node %s: %v", node.Name, err)
					continue
				} else {
					return nil, fmt.Errorf("error getting address for node %s: %v", node.Name, err)
				}
			}

			if !memberExists(members, addr, int(port.NodePort)) {
				glog.V(4).Infof("Creating member for pool %s", pool.ID)
				_, err := v2pools.CreateMember(lbaas.network, pool.ID, v2pools.CreateMemberOpts{
					ProtocolPort: int(port.NodePort),
					Address:      addr,
					SubnetID:     lbaas.opts.SubnetId,
				}).Extract()
				if err != nil {
					return nil, fmt.Errorf("error creating LB pool member for node: %s, %v", node.Name, err)
				}

				waitLoadbalancerActiveProvisioningStatus(lbaas.network, loadbalancer.ID)
			} else {
				// After all members have been processed, remaining members are deleted as obsolete.
				members = popMember(members, addr, int(port.NodePort))
			}

			glog.V(4).Infof("Ensured pool %s has member for %s at %s", pool.ID, node.Name, addr)
		}

		// Delete obsolete members for this pool
		for _, member := range members {
			glog.V(4).Infof("Deleting obsolete member %s for pool %s address %s", member.ID, pool.ID, member.Address)
			err := v2pools.DeleteMember(lbaas.network, pool.ID, member.ID).ExtractErr()
			if err != nil && !isNotFound(err) {
				return nil, fmt.Errorf("error deleting obsolete member %s for pool %s address %s: %v", member.ID, pool.ID, member.Address, err)
			}
			waitLoadbalancerActiveProvisioningStatus(lbaas.network, loadbalancer.ID)
		}

		monitorID := pool.MonitorID
		if monitorID == "" && lbaas.opts.CreateMonitor {
			glog.V(4).Infof("Creating monitor for pool %s", pool.ID)
			monitor, err := v2monitors.Create(lbaas.network, v2monitors.CreateOpts{
				PoolID:     pool.ID,
				Type:       string(port.Protocol),
				Delay:      int(lbaas.opts.MonitorDelay.Duration.Seconds()),
				Timeout:    int(lbaas.opts.MonitorTimeout.Duration.Seconds()),
				MaxRetries: int(lbaas.opts.MonitorMaxRetries),
			}).Extract()
			if err != nil {
				return nil, fmt.Errorf("error creating LB pool healthmonitor: %v", err)
			}
			waitLoadbalancerActiveProvisioningStatus(lbaas.network, loadbalancer.ID)
			monitorID = monitor.ID
		} else if lbaas.opts.CreateMonitor == false {
			glog.V(4).Infof("Do not create monitor for pool %s when create-monitor is false", pool.ID)
		}

		if monitorID != "" {
			glog.V(4).Infof("Monitor for pool %s: %s", pool.ID, monitorID)
		}
	}

	// All remaining listeners are obsolete, delete
	for _, listener := range oldListeners {
		glog.V(4).Infof("Deleting obsolete listener %s:", listener.ID)
		// get pool for listener
		pool, err := getPoolByListenerID(lbaas.network, loadbalancer.ID, listener.ID)
		if err != nil && err != ErrNotFound {
			return nil, fmt.Errorf("error getting pool for obsolete listener %s: %v", listener.ID, err)
		}
		if pool != nil {
			// get and delete monitor
			monitorID := pool.MonitorID
			if monitorID != "" {
				glog.V(4).Infof("Deleting obsolete monitor %s for pool %s", monitorID, pool.ID)
				err = v2monitors.Delete(lbaas.network, monitorID).ExtractErr()
				if err != nil && !isNotFound(err) {
					return nil, fmt.Errorf("error deleting obsolete monitor %s for pool %s: %v", monitorID, pool.ID, err)
				}
				waitLoadbalancerActiveProvisioningStatus(lbaas.network, loadbalancer.ID)
			}
			// get and delete pool members
			members, err := getMembersByPoolID(lbaas.network, pool.ID)
			if err != nil && !isNotFound(err) {
				return nil, fmt.Errorf("error getting members for pool %s: %v", pool.ID, err)
			}
			if members != nil {
				for _, member := range members {
					glog.V(4).Infof("Deleting obsolete member %s for pool %s address %s", member.ID, pool.ID, member.Address)
					err := v2pools.DeleteMember(lbaas.network, pool.ID, member.ID).ExtractErr()
					if err != nil && !isNotFound(err) {
						return nil, fmt.Errorf("error deleting obsolete member %s for pool %s address %s: %v", member.ID, pool.ID, member.Address, err)
					}
					waitLoadbalancerActiveProvisioningStatus(lbaas.network, loadbalancer.ID)
				}
			}
			glog.V(4).Infof("Deleting obsolete pool %s for listener %s", pool.ID, listener.ID)
			// delete pool
			err = v2pools.Delete(lbaas.network, pool.ID).ExtractErr()
			if err != nil && !isNotFound(err) {
				return nil, fmt.Errorf("error deleting obsolete pool %s for listener %s: %v", pool.ID, listener.ID, err)
			}
			waitLoadbalancerActiveProvisioningStatus(lbaas.network, loadbalancer.ID)
		}
		// delete listener
		err = listeners.Delete(lbaas.network, listener.ID).ExtractErr()
		if err != nil && !isNotFound(err) {
			return nil, fmt.Errorf("error deleteting obsolete listener: %v", err)
		}
		waitLoadbalancerActiveProvisioningStatus(lbaas.network, loadbalancer.ID)
		glog.V(2).Infof("Deleted obsolete listener: %s", listener.ID)
	}

	portID := loadbalancer.VipPortID
	floatIP, err := getFloatingIPByPortID(lbaas.network, portID)
	if err != nil && err != ErrNotFound {
		return nil, fmt.Errorf("error getting floating ip for port %s: %v", portID, err)
	}
	if floatIP == nil && floatingPool != "" && !internalAnnotation {
		glog.V(4).Infof("Creating floating ip for loadbalancer %s port %s", loadbalancer.ID, portID)
		floatIPOpts := floatingips.CreateOpts{
			FloatingNetworkID: floatingPool,
			PortID:            portID,
		}

		loadBalancerIP := apiService.Spec.LoadBalancerIP
		if loadBalancerIP != "" {
			floatIPOpts.FloatingIP = loadBalancerIP
		}

		floatIP, err = floatingips.Create(lbaas.network, floatIPOpts).Extract()
		if err != nil {
			return nil, fmt.Errorf("error creating LB floatingip %+v: %v", floatIPOpts, err)
		}
	}

	status := &v1.LoadBalancerStatus{}

	if floatIP != nil {
		status.Ingress = []v1.LoadBalancerIngress{{IP: floatIP.FloatingIP}}
	} else {
		status.Ingress = []v1.LoadBalancerIngress{{IP: loadbalancer.VipAddress}}
	}

	if lbaas.opts.ManageSecurityGroups {
		err := lbaas.ensureSecurityGroup(clusterName, apiService, nodes, loadbalancer)
		if err != nil {
			// cleanup what was created so far
			_ = lbaas.EnsureLoadBalancerDeleted(clusterName, apiService)
			return status, err
		}
	}

	return status, nil
}

// ensureSecurityGroup ensures security group exist for specific loadbalancer service.
// Creating security group for specific loadbalancer service when it does not exist.
func (lbaas *LbaasV2) ensureSecurityGroup(clusterName string, apiService *v1.Service, nodes []*v1.Node, loadbalancer *loadbalancers.LoadBalancer) error {
	// find node-security-group for service
	var err error
	if len(lbaas.opts.NodeSecurityGroupIDs) == 0 {
		lbaas.opts.NodeSecurityGroupIDs, err = getNodeSecurityGroupIDForLB(lbaas.compute, nodes)
		if err != nil {
			return fmt.Errorf("failed to find node-security-group for loadbalancer service %s/%s: %v", apiService.Namespace, apiService.Name, err)
		}
	}
	glog.V(4).Infof("find node-security-group %v for loadbalancer service %s/%s", lbaas.opts.NodeSecurityGroupIDs, apiService.Namespace, apiService.Name)

	// get service ports
	ports := apiService.Spec.Ports
	if len(ports) == 0 {
		return fmt.Errorf("no ports provided to openstack load balancer")
	}

	// get service source ranges
	sourceRanges, err := service.GetLoadBalancerSourceRanges(apiService)
	if err != nil {
		return fmt.Errorf("failed to get source ranges for loadbalancer service %s/%s: %v", apiService.Namespace, apiService.Name, err)
	}

	// ensure security group for LB
	lbSecGroupName := getSecurityGroupName(clusterName, apiService)
	lbSecGroupID, err := groups.IDFromName(lbaas.network, lbSecGroupName)
	if err != nil {
		// check whether security group does not exist
		_, ok := err.(*gophercloud.ErrResourceNotFound)
		if ok {
			// create it later
			lbSecGroupID = ""
		} else {
			return fmt.Errorf("error occurred finding security group: %s: %v", lbSecGroupName, err)
		}
	}
	if len(lbSecGroupID) == 0 {
		// create security group
		lbSecGroupCreateOpts := groups.CreateOpts{
			Name:        getSecurityGroupName(clusterName, apiService),
			Description: fmt.Sprintf("Securty Group for loadbalancer service %s/%s", apiService.Namespace, apiService.Name),
		}

		lbSecGroup, err := groups.Create(lbaas.network, lbSecGroupCreateOpts).Extract()
		if err != nil {
			return fmt.Errorf("failed to create Security Group for loadbalancer service %s/%s: %v", apiService.Namespace, apiService.Name, err)
		}
		lbSecGroupID = lbSecGroup.ID

		//add rule in security group
		for _, port := range ports {
			for _, sourceRange := range sourceRanges.StringSlice() {
				ethertype := rules.EtherType4
				network, _, err := net.ParseCIDR(sourceRange)

				if err != nil {
					return fmt.Errorf("error parsing source range %s as a CIDR: %v", sourceRange, err)
				}

				if network.To4() == nil {
					ethertype = rules.EtherType6
				}

				lbSecGroupRuleCreateOpts := rules.CreateOpts{
					Direction:      rules.DirIngress,
					PortRangeMax:   int(port.Port),
					PortRangeMin:   int(port.Port),
					Protocol:       toRuleProtocol(port.Protocol),
					RemoteIPPrefix: sourceRange,
					SecGroupID:     lbSecGroup.ID,
					EtherType:      ethertype,
				}

				_, err = rules.Create(lbaas.network, lbSecGroupRuleCreateOpts).Extract()

				if err != nil {
					return fmt.Errorf("error occured creating rule for SecGroup %s: %v", lbSecGroup.ID, err)
				}
			}
		}

		lbSecGroupRuleCreateOpts := rules.CreateOpts{
			Direction:      rules.DirIngress,
			PortRangeMax:   4, // ICMP: Code -  Values for ICMP  "Destination Unreachable: Fragmentation Needed and Don't Fragment was Set"
			PortRangeMin:   3, // ICMP: Type
			Protocol:       rules.ProtocolICMP,
			RemoteIPPrefix: "0.0.0.0/0", // The Fragmentation packet can come from anywhere along the path back to the sourceRange - we need to all this from all
			SecGroupID:     lbSecGroup.ID,
			EtherType:      rules.EtherType4,
		}

		_, err = rules.Create(lbaas.network, lbSecGroupRuleCreateOpts).Extract()

		if err != nil {
			return fmt.Errorf("error occured creating rule for SecGroup %s: %v", lbSecGroup.ID, err)
		}

		lbSecGroupRuleCreateOpts = rules.CreateOpts{
			Direction:      rules.DirIngress,
			PortRangeMax:   0, // ICMP: Code - Values for ICMP "Packet Too Big"
			PortRangeMin:   2, // ICMP: Type
			Protocol:       rules.ProtocolICMP,
			RemoteIPPrefix: "::/0", // The Fragmentation packet can come from anywhere along the path back to the sourceRange - we need to all this from all
			SecGroupID:     lbSecGroup.ID,
			EtherType:      rules.EtherType6,
		}

		_, err = rules.Create(lbaas.network, lbSecGroupRuleCreateOpts).Extract()
		if err != nil {
			return fmt.Errorf("error occured creating rule for SecGroup %s: %v", lbSecGroup.ID, err)
		}

		// get security groups of port
		portID := loadbalancer.VipPortID
		port, err := getPortByID(lbaas.network, portID)
		if err != nil {
			return err
		}

		// ensure the vip port has the security groups
		found := false
		for _, portSecurityGroups := range port.SecurityGroups {
			if portSecurityGroups == lbSecGroup.ID {
				found = true
				break
			}
		}

		// update loadbalancer vip port
		if !found {
			port.SecurityGroups = append(port.SecurityGroups, lbSecGroup.ID)
			update_opts := neutronports.UpdateOpts{SecurityGroups: &port.SecurityGroups}
			res := neutronports.Update(lbaas.network, portID, update_opts)
			if res.Err != nil {
				msg := fmt.Sprintf("Error occured updating port %s for loadbalancer service %s/%s: %v", portID, apiService.Namespace, apiService.Name, res.Err)
				return fmt.Errorf(msg)
			}
		}
	}

	// ensure rules for every node security group
	for _, port := range ports {
		for _, nodeSecurityGroupID := range lbaas.opts.NodeSecurityGroupIDs {
			opts := rules.ListOpts{
				Direction:     string(rules.DirIngress),
				SecGroupID:    nodeSecurityGroupID,
				RemoteGroupID: lbSecGroupID,
				PortRangeMax:  int(port.NodePort),
				PortRangeMin:  int(port.NodePort),
				Protocol:      string(port.Protocol),
			}
			secGroupRules, err := getSecurityGroupRules(lbaas.network, opts)
			if err != nil && !isNotFound(err) {
				msg := fmt.Sprintf("Error finding rules for remote group id %s in security group id %s: %v", lbSecGroupID, nodeSecurityGroupID, err)
				return fmt.Errorf(msg)
			}
			if len(secGroupRules) != 0 {
				// Do not add rule when find rules for remote group in the Node Security Group
				continue
			}

			// Add the rules in the Node Security Group
			err = createNodeSecurityGroup(lbaas.network, nodeSecurityGroupID, int(port.NodePort), port.Protocol, lbSecGroupID)
			if err != nil {
				return fmt.Errorf("error occured creating security group for loadbalancer service %s/%s: %v", apiService.Namespace, apiService.Name, err)
			}
		}
	}

	return nil
}

func (lbaas *LbaasV2) UpdateLoadBalancer(clusterName string, service *v1.Service, nodes []*v1.Node) error {
	loadBalancerName := cloudprovider.GetLoadBalancerName(service)
	glog.V(4).Infof("UpdateLoadBalancer(%v, %v, %v)", clusterName, loadBalancerName, nodes)

	if len(lbaas.opts.SubnetId) == 0 && len(nodes) > 0 {
		// Get SubnetId automatically.
		// The LB needs to be configured with instance addresses on the same subnet, so get SubnetId by one node.
		subnetID, err := getSubnetIDForLB(lbaas.compute, *nodes[0])
		if err != nil {
			glog.Warningf("Failed to find subnet-id for loadbalancer service %s/%s: %v", service.Namespace, service.Name, err)
			return fmt.Errorf("no subnet-id for service %s/%s : subnet-id not set in cloud provider config, "+
				"and failed to find subnet-id from OpenStack: %v", service.Namespace, service.Name, err)
		}
		lbaas.opts.SubnetId = subnetID
	}

	ports := service.Spec.Ports
	if len(ports) == 0 {
		return fmt.Errorf("no ports provided to openstack load balancer")
	}

	loadbalancer, err := getLoadbalancerByName(lbaas.network, loadBalancerName)
	if err != nil {
		return err
	}
	if loadbalancer == nil {
		return fmt.Errorf("loadbalancer %s does not exist", loadBalancerName)
	}

	// Get all listeners for this loadbalancer, by "port key".
	type portKey struct {
		Protocol listeners.Protocol
		Port     int
	}
	var listenerIDs []string
	lbListeners := make(map[portKey]listeners.Listener)
	allListeners, err := getListenersByLoadBalancerID(lbaas.network, loadbalancer.ID)
	if err != nil {
		return fmt.Errorf("error getting listeners for LB %s: %v", loadBalancerName, err)
	}
	for _, l := range allListeners {
		key := portKey{Protocol: listeners.Protocol(l.Protocol), Port: l.ProtocolPort}
		lbListeners[key] = l
		listenerIDs = append(listenerIDs, l.ID)
	}

	// Get all pools for this loadbalancer, by listener ID.
	lbPools := make(map[string]v2pools.Pool)
	for _, listenerID := range listenerIDs {
		pool, err := getPoolByListenerID(lbaas.network, loadbalancer.ID, listenerID)
		if err != nil {
			return fmt.Errorf("error getting pool for listener %s: %v", listenerID, err)
		}
		lbPools[listenerID] = *pool
	}

	// Compose Set of member (addresses) that _should_ exist
	addrs := map[string]empty{}
	for _, node := range nodes {
		addr, err := nodeAddressForLB(node)
		if err != nil {
			return err
		}
		addrs[addr] = empty{}
	}

	// Check for adding/removing members associated with each port
	for _, port := range ports {
		// Get listener associated with this port
		listener, ok := lbListeners[portKey{
			Protocol: toListenersProtocol(port.Protocol),
			Port:     int(port.Port),
		}]
		if !ok {
			return fmt.Errorf("loadbalancer %s does not contain required listener for port %d and protocol %s", loadBalancerName, port.Port, port.Protocol)
		}

		// Get pool associated with this listener
		pool, ok := lbPools[listener.ID]
		if !ok {
			return fmt.Errorf("loadbalancer %s does not contain required pool for listener %s", loadBalancerName, listener.ID)
		}

		// Find existing pool members (by address) for this port
		getMembers, err := getMembersByPoolID(lbaas.network, pool.ID)
		if err != nil {
			return fmt.Errorf("error getting pool members %s: %v", pool.ID, err)
		}
		members := make(map[string]v2pools.Member)
		for _, member := range getMembers {
			members[member.Address] = member
		}

		// Add any new members for this port
		for addr := range addrs {
			if _, ok := members[addr]; ok && members[addr].ProtocolPort == int(port.NodePort) {
				// Already exists, do not create member
				continue
			}
			_, err := v2pools.CreateMember(lbaas.network, pool.ID, v2pools.CreateMemberOpts{
				Address:      addr,
				ProtocolPort: int(port.NodePort),
				SubnetID:     lbaas.opts.SubnetId,
			}).Extract()
			if err != nil {
				return err
			}
			waitLoadbalancerActiveProvisioningStatus(lbaas.network, loadbalancer.ID)
		}

		// Remove any old members for this port
		for _, member := range members {
			if _, ok := addrs[member.Address]; ok && member.ProtocolPort == int(port.NodePort) {
				// Still present, do not delete member
				continue
			}
			err = v2pools.DeleteMember(lbaas.network, pool.ID, member.ID).ExtractErr()
			if err != nil && !isNotFound(err) {
				return err
			}
			waitLoadbalancerActiveProvisioningStatus(lbaas.network, loadbalancer.ID)
		}
	}

	if lbaas.opts.ManageSecurityGroups {
		err := lbaas.updateSecurityGroup(clusterName, service, nodes, loadbalancer)
		if err != nil {
			return fmt.Errorf("failed to update Securty Group for loadbalancer service %s/%s: %v", service.Namespace, service.Name, err)
		}
	}

	return nil
}

// updateSecurityGroup updating security group for specific loadbalancer service.
func (lbaas *LbaasV2) updateSecurityGroup(clusterName string, apiService *v1.Service, nodes []*v1.Node, loadbalancer *loadbalancers.LoadBalancer) error {
	originalNodeSecurityGroupIDs := lbaas.opts.NodeSecurityGroupIDs

	var err error
	lbaas.opts.NodeSecurityGroupIDs, err = getNodeSecurityGroupIDForLB(lbaas.compute, nodes)
	if err != nil {
		return fmt.Errorf("failed to find node-security-group for loadbalancer service %s/%s: %v", apiService.Namespace, apiService.Name, err)
	}
	glog.V(4).Infof("find node-security-group %v for loadbalancer service %s/%s", lbaas.opts.NodeSecurityGroupIDs, apiService.Namespace, apiService.Name)

	original := sets.NewString(originalNodeSecurityGroupIDs...)
	current := sets.NewString(lbaas.opts.NodeSecurityGroupIDs...)
	removals := original.Difference(current)

	// Generate Name
	lbSecGroupName := getSecurityGroupName(clusterName, apiService)
	lbSecGroupID, err := groups.IDFromName(lbaas.network, lbSecGroupName)
	if err != nil {
		return fmt.Errorf("error occurred finding security group: %s: %v", lbSecGroupName, err)
	}

	ports := apiService.Spec.Ports
	if len(ports) == 0 {
		return fmt.Errorf("no ports provided to openstack load balancer")
	}

	for _, port := range ports {
		for removal := range removals {
			// Delete the rules in the Node Security Group
			opts := rules.ListOpts{
				Direction:     string(rules.DirIngress),
				SecGroupID:    removal,
				RemoteGroupID: lbSecGroupID,
				PortRangeMax:  int(port.NodePort),
				PortRangeMin:  int(port.NodePort),
				Protocol:      string(port.Protocol),
			}
			secGroupRules, err := getSecurityGroupRules(lbaas.network, opts)
			if err != nil && !isNotFound(err) {
				return fmt.Errorf("error finding rules for remote group id %s in security group id %s: %v", lbSecGroupID, removal, err)
			}

			for _, rule := range secGroupRules {
				res := rules.Delete(lbaas.network, rule.ID)
				if res.Err != nil && !isNotFound(res.Err) {
					return fmt.Errorf("error occurred deleting security group rule: %s: %v", rule.ID, res.Err)
				}
			}
		}

		for _, nodeSecurityGroupID := range lbaas.opts.NodeSecurityGroupIDs {
			opts := rules.ListOpts{
				Direction:     string(rules.DirIngress),
				SecGroupID:    nodeSecurityGroupID,
				RemoteGroupID: lbSecGroupID,
				PortRangeMax:  int(port.NodePort),
				PortRangeMin:  int(port.NodePort),
				Protocol:      string(port.Protocol),
			}
			secGroupRules, err := getSecurityGroupRules(lbaas.network, opts)
			if err != nil && !isNotFound(err) {
				return fmt.Errorf("error finding rules for remote group id %s in security group id %s: %v", lbSecGroupID, nodeSecurityGroupID, err)
			}
			if len(secGroupRules) != 0 {
				// Do not add rule when find rules for remote group in the Node Security Group
				continue
			}

			// Add the rules in the Node Security Group
			err = createNodeSecurityGroup(lbaas.network, nodeSecurityGroupID, int(port.NodePort), port.Protocol, lbSecGroupID)
			if err != nil {
				return fmt.Errorf("error occured creating security group for loadbalancer service %s/%s: %v", apiService.Namespace, apiService.Name, err)
			}
		}
	}

	return nil
}

func (lbaas *LbaasV2) EnsureLoadBalancerDeleted(clusterName string, service *v1.Service) error {
	loadBalancerName := cloudprovider.GetLoadBalancerName(service)
	glog.V(4).Infof("EnsureLoadBalancerDeleted(%v, %v)", clusterName, loadBalancerName)

	loadbalancer, err := getLoadbalancerByName(lbaas.network, loadBalancerName)
	if err != nil && err != ErrNotFound {
		return err
	}
	if loadbalancer == nil {
		return nil
	}

	if loadbalancer != nil && loadbalancer.VipPortID != "" {
		portID := loadbalancer.VipPortID
		floatingIP, err := getFloatingIPByPortID(lbaas.network, portID)
		if err != nil && err != ErrNotFound {
			return err
		}
		if floatingIP != nil {
			err = floatingips.Delete(lbaas.network, floatingIP.ID).ExtractErr()
			if err != nil && !isNotFound(err) {
				return err
			}
		}
	}

	// get all listeners associated with this loadbalancer
	listenerList, err := getListenersByLoadBalancerID(lbaas.network, loadbalancer.ID)
	if err != nil {
		return fmt.Errorf("error getting LB %s listeners: %v", loadbalancer.ID, err)
	}

	// get all pools (and health monitors) associated with this loadbalancer
	var poolIDs []string
	var monitorIDs []string
	for _, listener := range listenerList {
		pool, err := getPoolByListenerID(lbaas.network, loadbalancer.ID, listener.ID)
		if err != nil && err != ErrNotFound {
			return fmt.Errorf("error getting pool for listener %s: %v", listener.ID, err)
		}
		if pool != nil {
			poolIDs = append(poolIDs, pool.ID)
			// If create-monitor of cloud-config is false, pool has not monitor.
			if pool.MonitorID != "" {
				monitorIDs = append(monitorIDs, pool.MonitorID)
			}
		}
	}

	// get all members associated with each poolIDs
	var memberIDs []string
	for _, pool := range poolIDs {
		membersList, err := getMembersByPoolID(lbaas.network, pool)
		if err != nil && !isNotFound(err) {
			return fmt.Errorf("error getting pool members %s: %v", pool, err)
		}
		for _, member := range membersList {
			memberIDs = append(memberIDs, member.ID)
		}
	}

	// delete all monitors
	for _, monitorID := range monitorIDs {
		err := v2monitors.Delete(lbaas.network, monitorID).ExtractErr()
		if err != nil && !isNotFound(err) {
			return err
		}
		waitLoadbalancerActiveProvisioningStatus(lbaas.network, loadbalancer.ID)
	}

	// delete all members and pools
	for _, poolID := range poolIDs {
		// delete all members for this pool
		for _, memberID := range memberIDs {
			err := v2pools.DeleteMember(lbaas.network, poolID, memberID).ExtractErr()
			if err != nil && !isNotFound(err) {
				return err
			}
			waitLoadbalancerActiveProvisioningStatus(lbaas.network, loadbalancer.ID)
		}

		// delete pool
		err := v2pools.Delete(lbaas.network, poolID).ExtractErr()
		if err != nil && !isNotFound(err) {
			return err
		}
		waitLoadbalancerActiveProvisioningStatus(lbaas.network, loadbalancer.ID)
	}

	// delete all listeners
	for _, listener := range listenerList {
		err := listeners.Delete(lbaas.network, listener.ID).ExtractErr()
		if err != nil && !isNotFound(err) {
			return err
		}
		waitLoadbalancerActiveProvisioningStatus(lbaas.network, loadbalancer.ID)
	}

	// delete loadbalancer
	err = loadbalancers.Delete(lbaas.network, loadbalancer.ID).ExtractErr()
	if err != nil && !isNotFound(err) {
		return err
	}
	waitLoadbalancerDeleted(lbaas.network, loadbalancer.ID)

	// Delete the Security Group
	if lbaas.opts.ManageSecurityGroups {
		// Generate Name
		lbSecGroupName := getSecurityGroupName(clusterName, service)
		lbSecGroupID, err := groups.IDFromName(lbaas.network, lbSecGroupName)
		if err != nil {
			// check whether security group does not exist
			_, ok := err.(*gophercloud.ErrResourceNotFound)
			if ok {
				// It is OK when the security group has been deleted by others.
				return nil
			} else {
				return fmt.Errorf("error occurred finding security group: %s: %v", lbSecGroupName, err)
			}
		}

		lbSecGroup := groups.Delete(lbaas.network, lbSecGroupID)
		if lbSecGroup.Err != nil && !isNotFound(lbSecGroup.Err) {
			return lbSecGroup.Err
		}

		if len(lbaas.opts.NodeSecurityGroupIDs) == 0 {
			// Just happen when nodes have not Security Group, or should not happen
			// UpdateLoadBalancer and EnsureLoadBalancer can set lbaas.opts.NodeSecurityGroupIDs when it is empty
			// And service controller call UpdateLoadBalancer to set lbaas.opts.NodeSecurityGroupIDs when controller manager service is restarted.
			glog.Warningf("Can not find node-security-group from all the nodes of this cluser when delete loadbalancer service %s/%s",
				service.Namespace, service.Name)
		} else {
			// Delete the rules in the Node Security Group
			for _, nodeSecurityGroupID := range lbaas.opts.NodeSecurityGroupIDs {
				opts := rules.ListOpts{
					SecGroupID:    nodeSecurityGroupID,
					RemoteGroupID: lbSecGroupID,
				}
				secGroupRules, err := getSecurityGroupRules(lbaas.network, opts)

				if err != nil && !isNotFound(err) {
					msg := fmt.Sprintf("Error finding rules for remote group id %s in security group id %s: %v", lbSecGroupID, nodeSecurityGroupID, err)
					return fmt.Errorf(msg)
				}

				for _, rule := range secGroupRules {
					res := rules.Delete(lbaas.network, rule.ID)
					if res.Err != nil && !isNotFound(res.Err) {
						return fmt.Errorf("error occurred deleting security group rule: %s: %v", rule.ID, res.Err)
					}
				}
			}
		}
	}

	return nil
}
