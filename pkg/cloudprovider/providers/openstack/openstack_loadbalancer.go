/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"io"
	"net"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/openstack/compute/v2/extensions/secgroups"
	"github.com/rackspace/gophercloud/openstack/compute/v2/servers"
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions/layer3/floatingips"
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas/members"
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas/monitors"
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas/pools"
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas/vips"
	"github.com/rackspace/gophercloud/pagination"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/service"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

func (os *OpenStack) LoadBalancer() (cloudprovider.LoadBalancer, bool) {
	glog.V(4).Info("openstack.LoadBalancer() called")

	glog.V(1).Info("Claiming to support LoadBalancer")

	return os, true
}

func (os *OpenStack) GetLoadBalancer(service *api.Service) (*api.LoadBalancerStatus, bool, error) {
	loadBalancerName := cloudprovider.GetLoadBalancerName(service)
	vip, err := getVipByName(os.network, loadBalancerName)
	if err == ErrNotFound {
		return nil, false, nil
	}
	if vip == nil {
		return nil, false, err
	}
	fip, err := getFloatingIPByPortID(os.network, vip.PortID)
	if err != nil {
		return nil, false, err
	}

	status := &api.LoadBalancerStatus{}
	status.Ingress = []api.LoadBalancerIngress{{IP: fip.FloatingIP}}

	return status, true, err
}

// TODO: This code currently ignores 'region' and always creates a
// loadbalancer in only the current OpenStack region.  We should take
// a list of regions (from config) and query/create loadbalancers in
// each region.

func (os *OpenStack) EnsureLoadBalancer(apiService *api.Service, hosts []string, annotations map[string]string) (*api.LoadBalancerStatus, error) {
	glog.V(4).Infof("EnsureLoadBalancer(%v, %v, %v, %v, %v, %v, %v)", apiService.Namespace, apiService.Name, apiService.Spec.LoadBalancerIP, apiService.Spec.Ports, annotations)

	name := cloudprovider.GetLoadBalancerName(apiService)
	glog.V(2).Infof("Ensuring load balancer %s", name)
	// Sucess variable to trigger deferred clean up functions on failure.
	success := false

	ports := apiService.Spec.Ports
	if len(ports) > 1 {
		return nil, fmt.Errorf("multiple ports are not yet supported in openstack load balancers")
	}

	sourceRanges, err := service.GetLoadBalancerSourceRanges(annotations)
	if err != nil {
		return nil, err
	}

	if !service.IsAllowAll(sourceRanges) {
		return nil, fmt.Errorf("Source range restrictions are not supported for openstack load balancers")
	}

	affinity := apiService.Spec.SessionAffinity
	var persistence *vips.SessionPersistence
	switch affinity {
	case api.ServiceAffinityNone:
		persistence = nil
	case api.ServiceAffinityClientIP:
		persistence = &vips.SessionPersistence{Type: "SOURCE_IP"}
	default:
		return nil, fmt.Errorf("unsupported load balancer affinity: %v", affinity)
	}

	glog.V(4).Infof("Checking if openstack load balancer already exists: %s", name)
	_, exists, err := os.GetLoadBalancer(apiService)
	if err != nil {
		return nil, fmt.Errorf("error checking if openstack load balancer already exists: %v", err)
	}

	// TODO: Implement a more efficient update strategy for common changes than delete & create
	// In particular, if we implement hosts update, we can get rid of UpdateHosts
	if exists {
		glog.V(4).Infof("Removing existing OpenStack load balancer: %s", name)
		err := os.EnsureLoadBalancerDeleted(apiService)
		if err != nil {
			return nil, fmt.Errorf("error deleting existing openstack load balancer: %v", err)
		}
	}

	secgroupOpts := secgroups.CreateOpts{
		Name:        name,
		Description: "Security group for Kubernetes Load Balancer"}
	secgroup, err := secgroups.Create(os.compute, secgroupOpts).Extract()
	if err != nil {
		glog.Errorf("Security Group not created: %s", err)
		return nil, err
	}
	defer func() {
		if !success {
			err = secgroups.Delete(os.compute, secgroup.ID).ExtractErr()
			if err != nil {
				glog.Errorf("Security Group %s has failed to delete: %s", name, err)
			}
		}
	}()
	glog.V(4).Infof("Creating Security Group: %s", name)

	secgroupRuleOpts := secgroups.CreateRuleOpts{
		ParentGroupID: secgroup.ID,
		FromPort:      ports[0].NodePort,
		ToPort:        ports[0].NodePort,
		IPProtocol:    "TCP",
		CIDR:          "0.0.0.0/0",
	}
	rule, err := secgroups.CreateRule(os.compute, secgroupRuleOpts).Extract()
	if err != nil {
		glog.Errorf("Security Group Rule not created: %s", err)
		return nil, err
	}
	defer func() {
		if !success {
			secgroups.DeleteRule(os.compute, rule.ID)
		}
	}()
	glog.V(4).Info("Creating Security Group rule for port: %s %s", ports[0].NodePort, name)

	lbmethod := os.lbOpts.LBMethod
	if lbmethod == "" {
		lbmethod = pools.LBMethodRoundRobin
	}
	pool, err := pools.Create(os.network, pools.CreateOpts{
		Name:     name,
		Protocol: pools.ProtocolTCP,
		SubnetID: os.lbOpts.SubnetId,
		LBMethod: lbmethod,
	}).Extract()
	if err != nil {
		return nil, err
	}
	defer func() {
		if !success {
			pools.Delete(os.network, pool.ID)
		}
	}()
	glog.V(4).Infof("OpenStack loadbalancer pool created: %s", name)

	for _, host := range hosts {
		// Handles the instance where hostname is overridden to be the IP of the node.
		// Perhaps should move the route options HostnameOverride to Global options and
		// use that here
		var addr string
		var server *servers.Server
		ip := net.ParseIP(host)
		if ip != nil {
			glog.V(4).Infof("Searching for server with address: %s", host)
			server, err = getServerByAddress(os.compute, host)
			if err != nil {
				glog.Errorf("Server %s not found by name: %s", host, err)
				return nil, err
			}
			addr = ip.String()
			glog.V(4).Infof("Found server with address: %s", host)
		} else {
			glog.V(4).Infof("Searching for server with name: %s", host)
			server, err = getServerByName(os.compute, host)
			if err != nil {
				glog.Errorf("Server %s not found by name: %s", host, err)
				return nil, err
			}

			addr, err = getAddressByName(os.compute, host)
			glog.V(4).Infof("Found server with name %s and address %s", host, addr)
		}

		glog.V(4).Infof("Adding instance %s added to SecurityGroup %s", server.ID, name)
		err = secgroups.AddServerToGroup(os.compute, server.ID, secgroup.Name).ExtractErr()
		if err != nil && err != io.EOF {
			glog.V(4).Infof("Failed to add instance %s added to SecurityGroup %s: %s", host, name, err)
			return nil, err
		}
		defer func() {
			if !success {
				secgroups.RemoveServerFromGroup(os.compute, server.ID, secgroup.Name)
			}
		}()
		glog.V(4).Infof("OpenStack instance %s added to SecurityGroup %s", host, name)

		_, err = members.Create(os.network, members.CreateOpts{
			PoolID:       pool.ID,
			ProtocolPort: ports[0].NodePort, //TODO: need to handle multi-port
			Address:      addr,
		}).Extract()
		if err != nil {
			return nil, err
		}
		glog.V(4).Infof("OpenStack instance %s added to OpenStack LB pool %s", name)
	}

	glog.V(4).Infof("CreateLoadBalancer monitor")
	var mon *monitors.Monitor
	if os.lbOpts.CreateMonitor {
		mon, err = monitors.Create(os.network, monitors.CreateOpts{
			Type:       monitors.TypeTCP,
			Delay:      int(os.lbOpts.MonitorDelay.Duration.Seconds()),
			Timeout:    int(os.lbOpts.MonitorTimeout.Duration.Seconds()),
			MaxRetries: int(os.lbOpts.MonitorMaxRetries),
		}).Extract()
		if err != nil {
			return nil, err
		}
		defer func() {
			if !success {
				monitors.Delete(os.network, mon.ID)
			}
		}()

		_, err = pools.AssociateMonitor(os.network, pool.ID, mon.ID).Extract()
		if err != nil {
			return nil, err
		}
		glog.V(4).Infof("OpenStack lb monitor created: %s", name)
	}

	createOpts := vips.CreateOpts{
		Name:         name,
		Description:  fmt.Sprintf("Kubernetes external service %s", name),
		Protocol:     "TCP",
		ProtocolPort: ports[0].Port, //TODO: need to handle multi-port
		PoolID:       pool.ID,
		SubnetID:     os.lbOpts.SubnetId,
		Persistence:  persistence,
	}

	loadBalancerIP := apiService.Spec.LoadBalancerIP
	if loadBalancerIP != "" {
		createOpts.Address = loadBalancerIP
	}

	vip, err := vips.Create(os.network, createOpts).Extract()
	if err != nil {
		glog.Errorf("VIP for %s not created: %s", name, err)
		return nil, err
	}
	defer func() {
		if !success {
			vips.Delete(os.network, vip.ID)
		}
	}()
	glog.V(4).Infof("OpenStack VIP created: %s", name)

	fipCreateOpts := floatingips.CreateOpts{
		FloatingNetworkID: os.lbOpts.FloatingNetworkId,
		PortID:            vip.PortID,
	}
	fip, err := floatingips.Create(os.network, fipCreateOpts).Extract()
	if err != nil {
		return nil, err
	}
	glog.V(4).Infof("OpenStack FIP created: %s", name)

	status := &api.LoadBalancerStatus{}
	status.Ingress = []api.LoadBalancerIngress{
		{IP: vip.Address},
		{IP: fip.FloatingIP},
	}

	success = true

	return status, nil
}

func (os *OpenStack) UpdateLoadBalancer(service *api.Service, hosts []string) error {
	loadBalancerName := cloudprovider.GetLoadBalancerName(service)
	glog.V(4).Infof("UpdateLoadBalancer(%v, %v)", loadBalancerName, hosts)

	vip, err := getVipByName(os.network, loadBalancerName)
	if err != nil {
		return err
	}

	// Set of member (addresses) that _should_ exist
	addrs := map[string]bool{}
	for _, host := range hosts {
		var addr string
		ip := net.ParseIP(host)
		if ip != nil {
			addr = ip.String()
		} else {
			addr, err = getAddressByName(os.compute, host)
			if err != nil {
				return err
			}
		}

		addrs[addr] = true
	}

	// TODO Remove members from security group
	// Iterate over members that _do_ exist
	var port int
	pager := members.List(os.network, members.ListOpts{PoolID: vip.PoolID})
	err = pager.EachPage(func(page pagination.Page) (bool, error) {
		memList, err := members.ExtractMembers(page)
		if err != nil {
			return false, err
		}

		for _, member := range memList {
			if _, found := addrs[member.Address]; found {
				port = member.ProtocolPort
				// Member already exists
				delete(addrs, member.Address)
			} else {
				// Member needs to be deleted
				err = members.Delete(os.network, member.ID).ExtractErr()
				if err != nil {
					return false, err
				}
				server, err := getServerByAddress(os.compute, member.Address)
				if err != nil {
					return false, err
				}
				err = secgroups.RemoveServerFromGroup(os.compute, server.ID, service.Name).ExtractErr()
				if err != nil {
					return false, err
				}
			}
		}

		return true, nil
	})
	if err != nil {
		return err
	}

	// Anything left in addrs is a new member that needs to be added
	for addr := range addrs {
		_, err := members.Create(os.network, members.CreateOpts{
			PoolID:       vip.PoolID,
			Address:      addr,
			ProtocolPort: port,
		}).Extract()
		if err != nil {
			return err
		}
		server, err := getServerByAddress(os.compute, addr)
		if err != nil {
			return err
		}
		err = secgroups.AddServerToGroup(os.compute, server.ID, service.Name).ExtractErr()
		if err != nil {
			return err
		}
	}

	return nil
}

func (os *OpenStack) EnsureLoadBalancerDeleted(service *api.Service) error {
	loadBalancerName := cloudprovider.GetLoadBalancerName(service)
	glog.V(4).Infof("EnsureLoadBalancerDeleted(%v)", loadBalancerName)

	vip, err := getVipByName(os.network, loadBalancerName)
	if err != nil && err != ErrNotFound {
		return err
	}

	// We have to delete the VIP before the pool can be deleted,
	// so no point continuing if this fails.
	if vip != nil {
		err := vips.Delete(os.network, vip.ID).ExtractErr()
		if err != nil && !isNotFound(err) {
			return err
		}
	}

	var pool *pools.Pool
	if vip != nil {
		pool, err = pools.Get(os.network, vip.PoolID).Extract()
		if err != nil && !isNotFound(err) {
			return err
		}
	} else {
		// The VIP is gone, but it is conceivable that a Pool
		// still exists that we failed to delete on some
		// previous occasion.  Make a best effort attempt to
		// cleanup any pools with the same name as the VIP.
		pool, err = getPoolByName(os.network, service.Name)
		if err != nil && err != ErrNotFound {
			return err
		}
	}

	if pool != nil {
		for _, monId := range pool.MonitorIDs {
			_, err = pools.DisassociateMonitor(os.network, pool.ID, monId).Extract()
			if err != nil {
				return err
			}

			err = monitors.Delete(os.network, monId).ExtractErr()
			if err != nil && !isNotFound(err) {
				return err
			}
		}
		err = pools.Delete(os.network, pool.ID).ExtractErr()
		if err != nil && !isNotFound(err) {
			return err
		}
	}

	secgroup, err := getSecGroupByName(os.compute, service.Name)
	if err != nil && err != ErrNotFound {
		return err
	}
	if secgroup != nil {
		os.removeServersFromSecgroup(secgroup.Name)
		err := secgroups.Delete(os.compute, secgroup.ID).ExtractErr()
		if err != nil && !isNotFound(err) {
			return err
		}
	}

	return nil
}

func (os *OpenStack) removeServerFromSecgroup(serverID string, groupName string) error {
	pager := secgroups.ListByServer(os.compute, serverID)

	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		v, err := secgroups.ExtractSecurityGroups(page)
		if err != nil {
			return false, err
		}
		for _, group := range v {
			if group.Name == groupName {
				err := secgroups.RemoveServerFromGroup(os.compute, serverID, groupName).ExtractErr()
				if err != nil {
					return false, err
				}
			}
		}
		return true, nil
	})
	if err != nil {
		return err
	}

	return nil
}

func (os *OpenStack) removeServersFromSecgroup(groupName string) error {
	pager := servers.List(os.compute, nil)

	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		v, err := servers.ExtractServers(page)
		if err != nil {
			return false, err
		}
		for _, s := range v {
			err := os.removeServerFromSecgroup(s.ID, groupName)
			if err != nil {
				return false, err
			}
		}
		return true, nil
	})
	if err != nil {
		return err
	}

	return nil
}

func getSecGroupByName(client *gophercloud.ServiceClient, name string) (*secgroups.SecurityGroup, error) {
	pager := secgroups.List(client)

	secGroupList := make([]secgroups.SecurityGroup, 0, 1)

	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		v, err := secgroups.ExtractSecurityGroups(page)
		if err != nil {
			return false, err
		}
		for _, sg := range v {
			if sg.Name == name {
				secGroupList = append(secGroupList, sg)
				if len(secGroupList) > 1 {
					return false, ErrMultipleResults
				}
			}
		}
		return true, nil
	})
	if err != nil {
		if isNotFound(err) {
			return nil, ErrNotFound
		}
		return nil, err
	}

	if len(secGroupList) == 0 {
		return nil, ErrNotFound
	} else if len(secGroupList) > 1 {
		return nil, ErrMultipleResults
	}

	return &secGroupList[0], nil

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
