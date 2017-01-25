package openstack

import (
	"fmt"
	"time"
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions/layer3/floatingips"
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas_v2/loadbalancers"
	"github.com/rackspace/gophercloud/pagination"

	"github.com/golang/glog"
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas_v2/listeners"
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas_v2/monitors"
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas_v2/pools"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/api/v1/service"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"github.com/rackspace/gophercloud"
)

// LoadBalancer implementation for LBaaS v2
type LbaasV2 struct {
	LoadBalancer
}

func (lbaas *LbaasV2) createLoadBalancer(service *v1.Service, name string) (*loadbalancers.LoadBalancer, error) {
	createOpts := loadbalancers.CreateOpts{
		Name:        name,
		Description: fmt.Sprintf("Kubernetes external service %s", name),
		VipSubnetID: lbaas.opts.SubnetId,
	}

	loadBalancerIP := service.Spec.LoadBalancerIP
	if loadBalancerIP != "" {
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
	status.Ingress = []v1.LoadBalancerIngress{{IP: loadbalancer.VipAddress}}

	return status, true, err
}

// TODO: This code currently ignores 'region' and always creates a
// loadbalancer in only the current OpenStack region.  We should take
// a list of regions (from config) and query/create loadbalancers in
// each region.

func (lbaas *LbaasV2) EnsureLoadBalancer(clusterName string, apiService *v1.Service, nodes []*v1.Node) (*v1.LoadBalancerStatus, error) {
	glog.V(4).Infof("EnsureLoadBalancer(%v, %v, %v, %v, %v, %v, %v)", clusterName, apiService.Namespace, apiService.Name, apiService.Spec.LoadBalancerIP, apiService.Spec.Ports, nodes, apiService.Annotations)

	ports := apiService.Spec.Ports
	if len(ports) == 0 {
		return nil, fmt.Errorf("no ports provided to openstack load balancer")
	}

	// Check for TCP protocol on each port
	// TODO: Convert all error messages to use an event recorder
	for _, port := range ports {
		if port.Protocol != v1.ProtocolTCP {
			return nil, fmt.Errorf("Only TCP LoadBalancer is supported for openstack load balancers")
		}
	}

	sourceRanges, err := service.GetLoadBalancerSourceRanges(apiService)
	if err != nil {
		return nil, err
	}

	if !service.IsAllowAll(sourceRanges) && !lbaas.opts.ManageSecurityGroups {
		return nil, fmt.Errorf("Source range restrictions are not supported for openstack load balancers without managing security groups")
	}

	affinity := api.ServiceAffinityNone
	var persistence *pools.SessionPersistence
	switch affinity {
	case api.ServiceAffinityNone:
		persistence = nil
	case api.ServiceAffinityClientIP:
		persistence = &pools.SessionPersistence{Type: "SOURCE_IP"}
	default:
		return nil, fmt.Errorf("unsupported load balancer affinity: %v", affinity)
	}

	name := cloudprovider.GetLoadBalancerName(apiService)
	loadbalancer, err := getLoadbalancerByName(lbaas.network, name)
	var loadbalancerExists = false
	if err != nil {
		if err != ErrNotFound {
			return nil, fmt.Errorf("Error getting loadbalancer %s: %v", name, err)
		}
		glog.V(2).Infof("Creating loadbalancer %s", name)
		loadbalancer, err = lbaas.createLoadBalancer(apiService, name)
		if err != nil {
			// Unknown error, retry later
			return nil, fmt.Errorf("Error creating loadbalancer %s: %v", name, err)
		}
	} else {
		loadbalancerExists = true
		glog.V(2).Infof("LoadBalancer %s already exists", name)
	}

	waitLoadbalancerActiveProvisioningStatus(lbaas.network, loadbalancer.ID)

	lbmethod := pools.LBMethod(lbaas.opts.LBMethod)
	if lbmethod == "" {
		lbmethod = pools.LBMethodRoundRobin
	}

	oldListeners, err := getListenersByLoadBalancerID(lbaas.network, loadbalancer.ID)
	if err != nil {
		return nil, fmt.Errorf("Error getting LB %s listeners: %v", name, err)
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
				return nil, fmt.Errorf("Error creating LB listener: %v", err)
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
			return nil, fmt.Errorf("Error getting pool for listener %s: %v", listener.ID, err)
		}
		if pool == nil {
			glog.V(4).Infof("Creating pool for listener %s", listener.ID)
			pool, err = pools.Create(lbaas.network, pools.CreateOpts{
				Name:        fmt.Sprintf("pool_%s_%d", name, portIndex),
				Protocol:    pools.Protocol(port.Protocol),
				LBMethod:    lbmethod,
				ListenerID:  listener.ID,
				Persistence: persistence,
			}).Extract()
			if err != nil {
				// Unknown error, retry later
				return nil, fmt.Errorf("Error creating pool for listener %s: %v", listener.ID, err)
			}
			waitLoadbalancerActiveProvisioningStatus(lbaas.network, loadbalancer.ID)
		}

		glog.V(4).Infof("Pool for listener %s: %s", listener.ID, pool.ID)
		members, err := getMembersByPoolID(lbaas.network, pool.ID)
		if err != nil && !isNotFound(err) {
			return nil, fmt.Errorf("Error getting pool members %s: %v", pool.ID, err)
		}
		for _, node := range nodes {
			addr, err := nodeAddressForLB(node)
			if err != nil {
				if err == ErrNotFound {
					// Node failure, do not create member
					glog.Warningf("Failed to create LB pool member for node %s: %v", node.Name, err)
					continue
				} else {
					return nil, fmt.Errorf("Error getting address for node %s: %v", node.Name, err)
				}
			}

			if !memberExists(members, addr, int(port.NodePort)) {
				glog.V(4).Infof("Creating member for pool %s", pool.ID)
				_, err := pools.CreateAssociateMember(lbaas.network, pool.ID, pools.MemberCreateOpts{
					ProtocolPort: int(port.NodePort),
					Address:      addr,
					SubnetID:     lbaas.opts.SubnetId,
				}).Extract()
				if err != nil {
					return nil, fmt.Errorf("Error creating LB pool member for node: %s, %v", node.Name, err)
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
			err := pools.DeleteMember(lbaas.network, pool.ID, member.ID).ExtractErr()
			if err != nil && !isNotFound(err) {
				return nil, fmt.Errorf("Error deleting obsolete member %s for pool %s address %s: %v", member.ID, pool.ID, member.Address, err)
			}
			waitLoadbalancerActiveProvisioningStatus(lbaas.network, loadbalancer.ID)
		}

		monitorID := pool.MonitorID
		if monitorID == "" && lbaas.opts.CreateMonitor {
			glog.V(4).Infof("Creating monitor for pool %s", pool.ID)
			monitor, err := monitors.Create(lbaas.network, monitors.CreateOpts{
				PoolID:     pool.ID,
				Type:       string(port.Protocol),
				Delay:      int(lbaas.opts.MonitorDelay.Duration.Seconds()),
				Timeout:    int(lbaas.opts.MonitorTimeout.Duration.Seconds()),
				MaxRetries: int(lbaas.opts.MonitorMaxRetries),
			}).Extract()
			if err != nil {
				return nil, fmt.Errorf("Error creating LB pool healthmonitor: %v", err)
			}
			waitLoadbalancerActiveProvisioningStatus(lbaas.network, loadbalancer.ID)
			monitorID = monitor.ID
		}

		glog.V(4).Infof("Monitor for pool %s: %s", pool.ID, monitorID)
	}

	// All remaining listeners are obsolete, delete
	for _, listener := range oldListeners {
		glog.V(4).Infof("Deleting obsolete listener %s:", listener.ID)
		// get pool for listener
		pool, err := getPoolByListenerID(lbaas.network, loadbalancer.ID, listener.ID)
		if err != nil && err != ErrNotFound {
			return nil, fmt.Errorf("Error getting pool for obsolete listener %s: %v", listener.ID, err)
		}
		if pool != nil {
			// get and delete monitor
			monitorID := pool.MonitorID
			if monitorID != "" {
				glog.V(4).Infof("Deleting obsolete monitor %s for pool %s", monitorID, pool.ID)
				err = monitors.Delete(lbaas.network, monitorID).ExtractErr()
				if err != nil && !isNotFound(err) {
					return nil, fmt.Errorf("Error deleting obsolete monitor %s for pool %s: %v", monitorID, pool.ID, err)
				}
				waitLoadbalancerActiveProvisioningStatus(lbaas.network, loadbalancer.ID)
			}
			// get and delete pool members
			members, err := getMembersByPoolID(lbaas.network, pool.ID)
			if err != nil && !isNotFound(err) {
				return nil, fmt.Errorf("Error getting members for pool %s: %v", pool.ID, err)
			}
			if members != nil {
				for _, member := range members {
					glog.V(4).Infof("Deleting obsolete member %s for pool %s address %s", member.ID, pool.ID, member.Address)
					err := pools.DeleteMember(lbaas.network, pool.ID, member.ID).ExtractErr()
					if err != nil && !isNotFound(err) {
						return nil, fmt.Errorf("Error deleting obsolete member %s for pool %s address %s: %v", member.ID, pool.ID, member.Address, err)
					}
					waitLoadbalancerActiveProvisioningStatus(lbaas.network, loadbalancer.ID)
				}
			}
			glog.V(4).Infof("Deleting obsolete pool %s for listener %s", pool.ID, listener.ID)
			// delete pool
			err = pools.Delete(lbaas.network, pool.ID).ExtractErr()
			if err != nil && !isNotFound(err) {
				return nil, fmt.Errorf("Error deleting obsolete pool %s for listener %s: %v", pool.ID, listener.ID, err)
			}
			waitLoadbalancerActiveProvisioningStatus(lbaas.network, loadbalancer.ID)
		}
		// delete listener
		err = listeners.Delete(lbaas.network, listener.ID).ExtractErr()
		if err != nil && !isNotFound(err) {
			return nil, fmt.Errorf("Error deleteting obsolete listener: %v", err)
		}
		waitLoadbalancerActiveProvisioningStatus(lbaas.network, loadbalancer.ID)
		glog.V(2).Infof("Deleted obsolete listener: %s", listener.ID)
	}

	status := &v1.LoadBalancerStatus{}

	status.Ingress = []v1.LoadBalancerIngress{{IP: loadbalancer.VipAddress}}

	port, err := getPortByIP(lbaas.network, loadbalancer.VipAddress)
	if err != nil {
		return nil, fmt.Errorf("Error getting port for LB vip %s: %v", loadbalancer.VipAddress, err)
	}
	floatIP, err := getFloatingIPByPortID(lbaas.network, port.ID)
	if err != nil && err != ErrNotFound {
		return nil, fmt.Errorf("Error getting floating ip for port %s: %v", port.ID, err)
	}
	if floatIP == nil && lbaas.opts.FloatingNetworkId != "" {
		glog.V(4).Infof("Creating floating ip for loadbalancer %s port %s", loadbalancer.ID, port.ID)
		floatIPOpts := floatingips.CreateOpts{
			FloatingNetworkID: lbaas.opts.FloatingNetworkId,
			PortID:            port.ID,
		}
		floatIP, err = floatingips.Create(lbaas.network, floatIPOpts).Extract()
		if err != nil {
			return nil, fmt.Errorf("Error creating LB floatingip %+v: %v", floatIPOpts, err)
		}
	}
	if floatIP != nil {
		status.Ingress = append(status.Ingress, v1.LoadBalancerIngress{IP: floatIP.FloatingIP})
	}

	err = lbaas.EnsureLoadBalancerSecurityGroups(clusterName, apiService, nodes, loadbalancerExists)

	if err != nil {
		return nil, fmt.Errorf("Error ensuring security groups: %v", err)
	}

	return status, nil
}

func (lbaas *LbaasV2) UpdateLoadBalancer(clusterName string, service *v1.Service, nodes []*v1.Node) error {
	loadBalancerName := cloudprovider.GetLoadBalancerName(service)
	glog.V(4).Infof("UpdateLoadBalancer(%v, %v, %v)", clusterName, loadBalancerName, nodes)

	ports := service.Spec.Ports
	if len(ports) == 0 {
		return fmt.Errorf("no ports provided to openstack load balancer")
	}

	loadbalancer, err := getLoadbalancerByName(lbaas.network, loadBalancerName)
	if err != nil {
		return err
	}
	if loadbalancer == nil {
		return fmt.Errorf("Loadbalancer %s does not exist", loadBalancerName)
	}

	// Get all listeners for this loadbalancer, by "port key".
	type portKey struct {
		Protocol string
		Port     int
	}

	lbListeners := make(map[portKey]listeners.Listener)
	err = listeners.List(lbaas.network, listeners.ListOpts{LoadbalancerID: loadbalancer.ID}).EachPage(func(page pagination.Page) (bool, error) {
		listenersList, err := listeners.ExtractListeners(page)
		if err != nil {
			return false, err
		}
		for _, l := range listenersList {
			for _, lb := range l.Loadbalancers {
				// Double check this Listener belongs to the LB we're updating. Neutron's API filtering
				// can't be counted on in older releases (i.e Liberty).
				if loadbalancer.ID == lb.ID {
					key := portKey{Protocol: l.Protocol, Port: l.ProtocolPort}
					lbListeners[key] = l
					break
				}
			}
		}
		return true, nil
	})
	if err != nil {
		return err
	}

	// Get all pools for this loadbalancer, by listener ID.
	lbPools := make(map[string]pools.Pool)
	err = pools.List(lbaas.network, pools.ListOpts{LoadbalancerID: loadbalancer.ID}).EachPage(func(page pagination.Page) (bool, error) {
		poolsList, err := pools.ExtractPools(page)
		if err != nil {
			return false, err
		}
		for _, p := range poolsList {
			for _, l := range p.Listeners {
				// Double check this Pool belongs to the LB we're deleting. Neutron's API filtering
				// can't be counted on in older releases (i.e Liberty).
				for _, val := range lbListeners {
					if val.ID == l.ID {
						lbPools[l.ID] = p
						break
					}
				}
			}
		}
		return true, nil
	})
	if err != nil {
		return err
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
			Protocol: string(port.Protocol),
			Port:     int(port.Port),
		}]
		if !ok {
			return fmt.Errorf("Loadbalancer %s does not contain required listener for port %d and protocol %s", loadBalancerName, port.Port, port.Protocol)
		}

		// Get pool associated with this listener
		pool, ok := lbPools[listener.ID]
		if !ok {
			return fmt.Errorf("Loadbalancer %s does not contain required pool for listener %s", loadBalancerName, listener.ID)
		}

		// Find existing pool members (by address) for this port
		members := make(map[string]pools.Member)
		err := pools.ListAssociateMembers(lbaas.network, pool.ID, pools.MemberListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
			membersList, err := pools.ExtractMembers(page)
			if err != nil {
				return false, err
			}
			for _, member := range membersList {
				members[member.Address] = member
			}
			return true, nil
		})
		if err != nil {
			return err
		}

		// Add any new members for this port
		for addr := range addrs {
			if _, ok := members[addr]; ok && members[addr].ProtocolPort == int(port.NodePort) {
				// Already exists, do not create member
				continue
			}
			_, err := pools.CreateAssociateMember(lbaas.network, pool.ID, pools.MemberCreateOpts{
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
			err = pools.DeleteMember(lbaas.network, pool.ID, member.ID).ExtractErr()
			if err != nil && !isNotFound(err) {
				return err
			}
			waitLoadbalancerActiveProvisioningStatus(lbaas.network, loadbalancer.ID)
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

	if lbaas.opts.FloatingNetworkId != "" && loadbalancer != nil {
		portID, err := getPortIDByIP(lbaas.network, loadbalancer.VipAddress)
		if err != nil {
			return err
		}

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
	var listenerIDs []string
	err = listeners.List(lbaas.network, listeners.ListOpts{LoadbalancerID: loadbalancer.ID}).EachPage(func(page pagination.Page) (bool, error) {
		listenerList, err := listeners.ExtractListeners(page)
		if err != nil {
			return false, err
		}

		for _, listener := range listenerList {
			listenerIDs = append(listenerIDs, listener.ID)
		}

		return true, nil
	})
	if err != nil {
		return err
	}

	// get all pools (and health monitors) associated with this loadbalancer
	var poolIDs []string
	var monitorIDs []string
	err = pools.List(lbaas.network, pools.ListOpts{LoadbalancerID: loadbalancer.ID}).EachPage(func(page pagination.Page) (bool, error) {
		poolsList, err := pools.ExtractPools(page)
		if err != nil {
			return false, err
		}

		for _, pool := range poolsList {
			poolIDs = append(poolIDs, pool.ID)
			monitorIDs = append(monitorIDs, pool.MonitorID)
		}

		return true, nil
	})
	if err != nil {
		return err
	}

	// get all members associated with each poolIDs
	var memberIDs []string
	for _, poolID := range poolIDs {
		err := pools.ListAssociateMembers(lbaas.network, poolID, pools.MemberListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
			membersList, err := pools.ExtractMembers(page)
			if err != nil {
				return false, err
			}

			for _, member := range membersList {
				memberIDs = append(memberIDs, member.ID)
			}

			return true, nil
		})
		if err != nil {
			return err
		}
	}

	// delete all monitors
	for _, monitorID := range monitorIDs {
		err := monitors.Delete(lbaas.network, monitorID).ExtractErr()
		if err != nil && !isNotFound(err) {
			return err
		}
		waitLoadbalancerActiveProvisioningStatus(lbaas.network, loadbalancer.ID)
	}

	// delete all members and pools
	for _, poolID := range poolIDs {
		// delete all members for this pool
		for _, memberID := range memberIDs {
			err := pools.DeleteMember(lbaas.network, poolID, memberID).ExtractErr()
			if err != nil && !isNotFound(err) {
				return err
			}
			waitLoadbalancerActiveProvisioningStatus(lbaas.network, loadbalancer.ID)
		}

		// delete pool
		err := pools.Delete(lbaas.network, poolID).ExtractErr()
		if err != nil && !isNotFound(err) {
			return err
		}
		waitLoadbalancerActiveProvisioningStatus(lbaas.network, loadbalancer.ID)
	}

	// delete all listeners
	for _, listenerID := range listenerIDs {
		err := listeners.Delete(lbaas.network, listenerID).ExtractErr()
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

	err = lbaas.EnsureLoadBalancerSecurityGroupsDeleted(clusterName, service, true)

	if err != nil {
		glog.V(1).Infof("Error occurred deleting securoty groups for: %s: %v", service.Name, err)
		return err
	}

	return nil
}


// Utility / Helper Methods

func getLoadbalancerByName(client *gophercloud.ServiceClient, name string) (*loadbalancers.LoadBalancer, error) {
	opts := loadbalancers.ListOpts{
		Name: name,
	}
	pager := loadbalancers.List(client, opts)

	loadbalancerList := make([]loadbalancers.LoadBalancer, 0, 1)

	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		v, err := loadbalancers.ExtractLoadbalancers(page)
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
		if l.Protocol == string(port.Protocol) && l.ProtocolPort == int(port.Port) {
			return &l
		}
	}

	return nil
}

// Get pool for a listener. A listener always has exactly one pool.
func getPoolByListenerID(client *gophercloud.ServiceClient, loadbalancerID string, listenerID string) (*pools.Pool, error) {
	listenerPools := make([]pools.Pool, 0, 1)
	err := pools.List(client, pools.ListOpts{LoadbalancerID: loadbalancerID}).EachPage(func(page pagination.Page) (bool, error) {
		poolsList, err := pools.ExtractPools(page)
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

func getMembersByPoolID(client *gophercloud.ServiceClient, id string) ([]pools.Member, error) {
	var members []pools.Member
	err := pools.ListAssociateMembers(client, id, pools.MemberListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		membersList, err := pools.ExtractMembers(page)
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
func memberExists(members []pools.Member, addr string, port int) bool {
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

func popMember(members []pools.Member, addr string, port int) []pools.Member {
	for i, member := range members {
		if member.Address == addr && member.ProtocolPort == port {
			members[i] = members[len(members)-1]
			members = members[:len(members)-1]
		}
	}

	return members
}

func waitLoadbalancerActiveProvisioningStatus(client *gophercloud.ServiceClient, loadbalancerID string) (string, error) {
	start := time.Now().Second()
	for {
		loadbalancer, err := loadbalancers.Get(client, loadbalancerID).Extract()
		if err != nil {
			return "", err
		}
		if loadbalancer.ProvisioningStatus == "ACTIVE" {
			return "ACTIVE", nil
		} else if loadbalancer.ProvisioningStatus == "ERROR" {
			return "ERROR", fmt.Errorf("Loadbalancer has gone into ERROR state")
		}

		time.Sleep(1 * time.Second)

		if time.Now().Second()-start >= loadbalancerActiveTimeoutSeconds {
			return loadbalancer.ProvisioningStatus, fmt.Errorf("Loadbalancer failed to go into ACTIVE provisioning status within alloted time")
		}
	}
}

func waitLoadbalancerDeleted(client *gophercloud.ServiceClient, loadbalancerID string) error {
	start := time.Now().Second()
	for {
		_, err := loadbalancers.Get(client, loadbalancerID).Extract()
		if err != nil {
			if err == ErrNotFound {
				return nil
			} else {
				return err
			}
		}

		time.Sleep(1 * time.Second)

		if time.Now().Second()-start >= loadbalancerDeleteTimeoutSeconds {
			return fmt.Errorf("Loadbalancer failed to delete within the alloted time")
		}

	}
}
