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
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions"
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions/layer3/floatingips"
	"github.com/rackspace/gophercloud/openstack/networking/v2/extensions/lbaas_v2/monitors"
	"github.com/rackspace/gophercloud/pagination"
	"k8s.io/kubernetes/pkg/api/v1"
)

// Note: when creating a new Loadbalancer (VM), it can take some time before it is ready for use,
// this timeout is used for waiting until the Loadbalancer provisioning status goes to ACTIVE state.
const loadbalancerActiveTimeoutSeconds = 120
const loadbalancerDeleteTimeoutSeconds = 30

// LoadBalancer implementation for LBaaS v1
type LbaasV1 struct {
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


func getPortIDByIP(client *gophercloud.ServiceClient, ipAddress string) (string, error) {
	targetPort, err := getPortByIP(client, ipAddress)
	if err != nil {
		return targetPort.ID, err
	}
	return targetPort.ID, nil
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

// Each pool has exactly one or zero monitors. ListOpts does not seem to filter anything.
func getMonitorByPoolID(client *gophercloud.ServiceClient, id string) (*monitors.Monitor, error) {
	var monitorList []monitors.Monitor
	err := monitors.List(client, monitors.ListOpts{PoolID: id}).EachPage(func(page pagination.Page) (bool, error) {
		monitorsList, err := monitors.ExtractMonitors(page)
		if err != nil {
			return false, err
		}

		for _, monitor := range monitorsList {
			// bugfix, filter by poolid
			for _, pool := range monitor.Pools {
				if pool.ID == id {
					monitorList = append(monitorList, monitor)
				}
			}
		}
		if len(monitorList) > 1 {
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

	if len(monitorList) == 0 {
		return nil, ErrNotFound
	} else if len(monitorList) > 1 {
		return nil, ErrMultipleResults
	}

	return &monitorList[0], nil
}
