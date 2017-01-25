package openstack

import (
	"fmt"
	"github.com/golang/glog"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions/security/groups"
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions/security/rules"
	"github.com/rackspace/gophercloud/openstack/networking/v2/subnets"
	"github.com/rackspace/gophercloud/pagination"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/api/v1/service"
	"net"
	"strings"
)


func (lbaas *LbaasV2) EnsureLoadBalancerSecurityGroups(clusterName string, service *v1.Service, nodes []*v1.Node, LoadBalancerExists bool) error {
	// If k8 is not managing the rules - nothing to do here, exit.
	if !lbaas.opts.ManageSecurityGroups {
		return nil
	}

	// Check if the security group for this loadbalancer has been created, if not, create it
	LBGroup, err := getOrCreateLBGroup(lbaas, clusterName, service)
	if err != nil {
		// If this is an update, do not destroy the LB. If it is a new LB, wipe it and start again.
		if !LoadBalancerExists {
			lbaas.EnsureLoadBalancerDeleted(clusterName, service)
		}
		return err
	}

	// Create the security group rules for inbound traffic to the loadbalancer.
	err = ensureExternalRules(lbaas, service, LBGroup)
	if err != nil {
		// If this is an update, do not destroy the LB. If it is a new LB, wipe it and start again.
		if !LoadBalancerExists {
			lbaas.EnsureLoadBalancerDeleted(clusterName, service)
		}
		return err
	}

	// If we are using octavia - we will have a single Internal Rule that allows all traffic from the VIP Subnet to the K8 Nodes.
	if lbaas.opts.AllowAllTrafficFromVIPSubnet {
		err = ensureAllowAllRule(lbaas)
	} else {
		err = ensureInternalRules(lbaas, service, LBGroup)
	}
	if err != nil {
		// If this is an update, do not destroy the LB. If it is a new LB, wipe it and start again.
		if !LoadBalancerExists {
			lbaas.EnsureLoadBalancerDeleted(clusterName, service)
		}
		return err
	}

	return nil
}

func (lbaas *LbaasV2) EnsureLoadBalancerSecurityGroupsDeleted(clusterName string, service *v1.Service, LoadBalancerExists bool) error {
	// If k8 is not managing the rules - nothing to do here, exit.
	if !lbaas.opts.ManageSecurityGroups {
		return nil
	}

	lbSecGroupName := getSecurityGroupName(clusterName, service)
	lbSecGroupID, err := groups.IDFromName(lbaas.network, lbSecGroupName)
	if err != nil {
		glog.V(1).Infof("Error occurred finding security group: %s: %v", lbSecGroupName, err)
		return nil
	}

	lbSecGroup := groups.Delete(lbaas.network, lbSecGroupID)
	if lbSecGroup.Err != nil && !isNotFound(lbSecGroup.Err) {
		return lbSecGroup.Err
	}

	if !lbaas.opts.AllowAllTrafficFromVIPSubnet {

		// Delete the rules in the Node Security Group
		opts := rules.ListOpts{
			SecGroupID:    lbaas.opts.NodeSecurityGroupID,
			RemoteGroupID: lbSecGroupID,
		}
		secGroupRules, err := getSecurityGroupRules(lbaas.network, opts)

		if err != nil && !isNotFound(err) {
			glog.Errorf("Error finding rules for remote group id %s in security group id %s", lbSecGroupID, lbaas.opts.NodeSecurityGroupID)
			return err
		}

		for _, rule := range secGroupRules {
			res := rules.Delete(lbaas.network, rule.ID)
			if res.Err != nil && !isNotFound(res.Err) {
				glog.V(1).Infof("Error occurred deleting security group rule: %s: %v", rule.ID, res.Err)
			}
		}
	}
	return nil
}


// Utility Functions

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

func getSecurityGroupName(clusterName string, service *v1.Service) string {
	return fmt.Sprintf("lb-sg-%s-%v", clusterName, service.Name)
}

func getRules(client *gophercloud.ServiceClient, opts rules.ListOpts) ([]rules.SecGroupRule, error) {

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



func getOrCreateLBGroup(lbaas *LbaasV2, clusterName string, service *v1.Service) (*groups.SecGroup, error) {

	lbGroupName := getSecurityGroupName(clusterName, service)
	glog.V(3).Infof("Searching for security group for loadbalancer, named %s", lbGroupName)
	lbGroupID, err := groups.IDFromName(lbaas.network, lbGroupName)
	// Doesn't Exist, create it
	if isNotFound(err) {
		glog.V(3).Infof("Creating security group for loadbalancer, named %s", lbGroupName)
		opts := groups.CreateOpts{
			Name:        lbGroupName,
			Description: fmt.Sprintf("Securty Group for %v Service LoadBalancer", service.Name),
		}
		return groups.Create(lbaas.network, opts).Extract()

	}
	if err != nil && isNotFound(err) {
		glog.V(1).Infof("Error occurred finding security group: %s: %v", lbGroupName, err)
		return nil, err
	}

	lbGroup := groups.Get(lbaas.network, lbGroupID)
	if lbGroup.Err != nil {
		return nil, lbGroup.Err
	}
	return lbGroup.Extract()
}

func ensureAllowAllRule(lbaas *LbaasV2) error {
	subnetResult := subnets.Get(lbaas.compute, lbaas.opts.SubnetId)
	if subnetResult.Err != nil {
		return subnetResult.Err
	}
	subnet, _ := subnetResult.Extract()
	glog.V(3).Infof("Searching for internal security group rule to allow traffic from %s to %s on all ports", subnet.CIDR, lbaas.opts.NodeSecurityGroupID)
	opts := rules.ListOpts{
		SecGroupID:     lbaas.opts.NodeSecurityGroupID,
		RemoteIPPrefix: subnet.CIDR,
		PortRangeMax:   65535,
		PortRangeMin:   0,
		EtherType:      fmt.Sprintf("IPv%d", subnet.IPVersion),
	}
	_, err := getRules(lbaas.network, opts)
	if isNotFound(err) {
		// Doesn't exist, create it.
		// the opts we used to find the group are the same, just use them
		glog.V(3).Infof("Creating for internal security group rule to allow traffic from %s to %s on all ports", subnet.CIDR, lbaas.opts.NodeSecurityGroupID)
		createOpts := rules.CreateOpts{
			SecGroupID:     lbaas.opts.NodeSecurityGroupID,
			RemoteIPPrefix: subnet.CIDR,
			PortRangeMax:   65535,
			PortRangeMin:   0,
			EtherType:      fmt.Sprintf("IPv%d", subnet.IPVersion),
		}
		rule := rules.Create(lbaas.network, createOpts)
		if rule.Err != nil {
			glog.V(1).Infof("Error creating for internal security group rule to allow traffic from %s to %s on all ports", subnet.CIDR, lbaas.opts.NodeSecurityGroupID)
			return rule.Err
		}
	}
	return nil
}

func ensureInternalRules(lbaas *LbaasV2, apiService *v1.Service, LBSecGroup *groups.SecGroup) error {
	ports := apiService.Spec.Ports

	// Get the rules in the Node Security Group
	opts := rules.ListOpts{
		SecGroupID:    lbaas.opts.NodeSecurityGroupID,
		RemoteGroupID: LBSecGroup.ID,
	}
	glog.V(3).Infof("Searching for internal security rules to allow traffic from group %s to %s", LBSecGroup.ID, lbaas.opts.NodeSecurityGroupID)
	secGroupRules, err := getSecurityGroupRules(lbaas.network, opts)

	if err != nil && !isNotFound(err) {
		glog.V(1).Infof("Error searching for internal security rules to allow traffic from group %s to %s", LBSecGroup.ID, lbaas.opts.NodeSecurityGroupID)
		return err
	}

	existingNodePorts := make(map[int]bool)
	requiredNodePorts := make(map[int]bool)
	requiredNodePortsServicePorts := make(map[int]v1.ServicePort)
	nodePortsToDelete := make(map[int]bool)
	nodePortsToAdd := make(map[int]bool)

	// Get the current list of rules
	for _, rule := range secGroupRules {
		existingNodePorts[rule.PortRangeMax] = true
	}
	// Get the new list of rules
	for _, port := range ports {
		requiredNodePorts[int(port.NodePort)] = true
		requiredNodePortsServicePorts[int(port.NodePort)] = port
	}
	// Get the list of rules to delete
	for port := range requiredNodePorts {
		if !existingNodePorts[port] {
			nodePortsToAdd[port] = true
		}
	}
	// Get the list of rules to add
	for port := range existingNodePorts {
		if !requiredNodePorts[port] {
			nodePortsToDelete[port] = true
		}
	}
	// Delete the old rules
	for port := range nodePortsToDelete {
		opts := rules.ListOpts{
			SecGroupID:    lbaas.opts.NodeSecurityGroupID,
			RemoteGroupID: LBSecGroup.ID,
			PortRangeMax:  port,
		}
		secGroupRule, _ := getSecurityGroupRules(lbaas.network, opts)
		for _, rule := range secGroupRule {
			glog.V(3).Infof("Deleting internal security rule to allow traffic from group %s to %s on port %d", LBSecGroup.ID, lbaas.opts.NodeSecurityGroupID, port)
			res := rules.Delete(lbaas.network, rule.ID)
			if res.Err != nil && !isNotFound(res.Err) {
				glog.V(1).Infof("Error occured deleting internal security rule to allow traffic from group %s to %s on port %d", LBSecGroup.ID, lbaas.opts.NodeSecurityGroupID, port)
			}
		}
	}
	// Add the new rules
	for port := range nodePortsToAdd {
		glog.V(3).Infof("Creating internal security rule to allow traffic from group %s to %s on port %d", LBSecGroup.ID, lbaas.opts.NodeSecurityGroupID, port)
		err := createInternalRule(lbaas.network, lbaas.opts.NodeSecurityGroupID, port, string(requiredNodePortsServicePorts[port].Protocol), LBSecGroup.ID)
		if err != nil {
			glog.V(1).Infof("Error occured creating internal security rule to allow traffic from group %s to %s on port %d", LBSecGroup.ID, lbaas.opts.NodeSecurityGroupID, port)
		}
	}

	return nil
}


func createInternalRule(client *gophercloud.ServiceClient, nodeSecurityGroupID string, port int, protocol string, lbSecGroup string) error {
	v4NodeSecGroupRuleCreateOpts := rules.CreateOpts{
		Direction:     "ingress",
		PortRangeMax:  port,
		PortRangeMin:  port,
		Protocol:      strings.ToLower(protocol),
		RemoteGroupID: lbSecGroup,
		SecGroupID:    nodeSecurityGroupID,
		EtherType:     "IPv4",
	}

	v6NodeSecGroupRuleCreateOpts := rules.CreateOpts{
		Direction:     "ingress",
		PortRangeMax:  port,
		PortRangeMin:  port,
		Protocol:      strings.ToLower(protocol),
		RemoteGroupID: lbSecGroup,
		SecGroupID:    nodeSecurityGroupID,
		EtherType:     "IPv6",
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

func ensureExternalRules(lbaas *LbaasV2, apiService *v1.Service, LBSecGroup *groups.SecGroup) error {
	ports := apiService.Spec.Ports
	sourceRanges, _ := service.GetLoadBalancerSourceRanges(apiService)

	for _, port := range ports {

		for _, sourceRange := range sourceRanges.StringSlice() {
			err := ensureExternalRule(lbaas.network, int(port.Port), strings.ToLower(string(port.Protocol)), LBSecGroup.ID, sourceRange)
			if err != nil {
				return err
			}
		}
	}

	return nil
}


func ensureExternalRule(client *gophercloud.ServiceClient, port int, protocol string, lbSecGroup string, sourceRange string) error {
	ethertype := "IPv4"
	network, _, err := net.ParseCIDR(sourceRange)

	if err != nil {
		// cleanup what was created so far
		glog.Errorf("Error parsing source range %s as a CIDR", sourceRange)
		return err
	}

	if network.To4() == nil {
		ethertype = "IPv6"
	}

	lbSecGroupRuleCreateOpts := rules.CreateOpts{
		Direction:      "ingress",
		PortRangeMax:   int(port),
		PortRangeMin:   int(port),
		Protocol:       strings.ToLower(protocol),
		RemoteIPPrefix: sourceRange,
		SecGroupID:     lbSecGroup,
		EtherType:      ethertype,
	}

	lbSecGroupRuleListOpts := rules.ListOpts{
		Direction:      "ingress",
		PortRangeMax:   int(port),
		PortRangeMin:   int(port),
		Protocol:       strings.ToLower(protocol),
		RemoteIPPrefix: sourceRange,
		SecGroupID:     lbSecGroup,
		EtherType:      ethertype,
	}

	glog.V(3).Infof("Searching for inbound security group rule to allow traffic from %s to %s on port %d", sourceRange, lbSecGroup, port)

	ruleList, err := getRules(client, lbSecGroupRuleListOpts)

	if err != nil || len(ruleList) == 0{

		if isNotFound(err) || len(ruleList) == 0 {
			glog.V(3).Infof("Creating inbound security group rule to allow traffic from %s to %s on port %d", sourceRange, lbSecGroup, port)
			_, err = rules.Create(client, lbSecGroupRuleCreateOpts).Extract()

			if err != nil {
				glog.V(1).Infof("Error creating inbound security group rule to allow traffic from %s to %s on port %d", sourceRange, lbSecGroup, port)
				return err
			}
		} else {
			glog.V(1).Infof("Error searching for inbound security group rule to allow traffic from %s to %s on port %d", sourceRange, lbSecGroup, port)
			return err
		}
	}

	return nil

}