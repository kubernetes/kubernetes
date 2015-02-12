/*
Copyright 2014 Google Inc. All rights reserved.

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

package rackspace

import (
	"errors"
	"fmt"
	"io"
	"net"
	"regexp"
	"time"

	"code.google.com/p/gcfg"
	"github.com/rackspace/gophercloud"
	os_servers "github.com/rackspace/gophercloud/openstack/compute/v2/servers"
	"github.com/rackspace/gophercloud/pagination"
	"github.com/rackspace/gophercloud/rackspace"
	"github.com/rackspace/gophercloud/rackspace/compute/v2/flavors"
	"github.com/rackspace/gophercloud/rackspace/compute/v2/servers"
	"github.com/rackspace/gophercloud/rackspace/lb/v1/lbs"
	"github.com/rackspace/gophercloud/rackspace/lb/v1/nodes"
	"github.com/rackspace/gophercloud/rackspace/lb/v1/vips"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/golang/glog"
)

var ErrNotFound = errors.New("Failed to find object")
var ErrMultipleResults = errors.New("Multiple results where only one expected")
var ErrNoAddressFound = errors.New("No address found for host")
var ErrAttrNotFound = errors.New("Expected attribute not found")

// encoding.TextUnmarshaler interface for time.Duration
type MyDuration struct {
	time.Duration
}

func (d *MyDuration) UnmarshalText(text []byte) error {
	res, err := time.ParseDuration(string(text))
	if err != nil {
		return err
	}
	d.Duration = res
	return nil
}

type LoadBalancerOpts struct {
	SubnetId          string     `gcfg:"subnet-id"` // required
	CreateMonitor     bool       `gcfg:"create-monitor"`
	MonitorDelay      MyDuration `gcfg:"monitor-delay"`
	MonitorTimeout    MyDuration `gcfg:"monitor-timeout"`
	MonitorMaxRetries uint       `gcfg:"monitor-max-retries"`
}

// Rackspace is an implementation of cloud provider Interface for Rackspace.
type Rackspace struct {
	provider *gophercloud.ProviderClient
	region   string
	lbOpts   LoadBalancerOpts
}

type Config struct {
	Global struct {
		AuthUrl    string `gcfg:"auth-url"`
		Username   string
		UserId     string `gcfg:"user-id"`
		Password   string
		ApiKey     string `gcfg:"api-key"`
		TenantId   string `gcfg:"tenant-id"`
		TenantName string `gcfg:"tenant-name"`
		DomainId   string `gcfg:"domain-id"`
		DomainName string `gcfg:"domain-name"`
		Region     string
	}
	LoadBalancer LoadBalancerOpts
}

func init() {
	cloudprovider.RegisterCloudProvider("rackspace", func(config io.Reader) (cloudprovider.Interface, error) {
		cfg, err := readConfig(config)
		if err != nil {
			return nil, err
		}
		return newRackspace(cfg)
	})
}

func (cfg Config) toAuthOptions() gophercloud.AuthOptions {
	return gophercloud.AuthOptions{
		IdentityEndpoint: cfg.Global.AuthUrl,
		Username:         cfg.Global.Username,
		UserID:           cfg.Global.UserId,
		Password:         cfg.Global.Password,
		APIKey:           cfg.Global.ApiKey,
		TenantID:         cfg.Global.TenantId,
		TenantName:       cfg.Global.TenantName,

		// Persistent service, so we need to be able to renew tokens
		AllowReauth: true,
	}
}

func readConfig(config io.Reader) (Config, error) {
	if config == nil {
		err := fmt.Errorf("no Rackspace cloud provider config file given")
		return Config{}, err
	}

	var cfg Config
	err := gcfg.ReadInto(&cfg, config)
	return cfg, err
}

func newRackspace(cfg Config) (*Rackspace, error) {
	provider, err := rackspace.AuthenticatedClient(cfg.toAuthOptions())
	if err != nil {
		return nil, err
	}

	os := Rackspace{
		provider: provider,
		region:   cfg.Global.Region,
		lbOpts:   cfg.LoadBalancer,
	}
	return &os, nil
}

type Instances struct {
	compute            *gophercloud.ServiceClient
	flavor_to_resource map[string]*api.NodeResources // keyed by flavor id
}

// Instances returns an implementation of Instances for Rackspace.
func (os *Rackspace) Instances() (cloudprovider.Instances, bool) {
	glog.V(2).Info("rackspace.Instances() called")

	compute, err := rackspace.NewComputeV2(os.provider, gophercloud.EndpointOpts{
		Region: os.region,
	})
	if err != nil {
		glog.Warningf("Failed to find compute endpoint: %v", err)
		return nil, false
	}

	pager := flavors.ListDetail(compute, nil)

	flavor_to_resource := make(map[string]*api.NodeResources)
	err = pager.EachPage(func(page pagination.Page) (bool, error) {
		flavorList, err := flavors.ExtractFlavors(page)
		if err != nil {
			return false, err
		}
		for _, flavor := range flavorList {
			rsrc := api.NodeResources{
				Capacity: api.ResourceList{
					api.ResourceCPU:            *resource.NewMilliQuantity(int64(flavor.VCPUs*1000), resource.DecimalSI),
					api.ResourceMemory:         resource.MustParse(fmt.Sprintf("%dMi", flavor.RAM)),
					"openstack.org/disk":       resource.MustParse(fmt.Sprintf("%dG", flavor.Disk)),
					"openstack.org/rxTxFactor": *resource.NewQuantity(int64(flavor.RxTxFactor*1000), resource.DecimalSI),
					"openstack.org/swap":       resource.MustParse(fmt.Sprintf("%dMi", flavor.Swap)),
				},
			}
			flavor_to_resource[flavor.ID] = &rsrc
		}
		return true, nil
	})
	if err != nil {
		glog.Warningf("Failed to find compute flavors: %v", err)
		return nil, false
	}

	glog.V(2).Infof("Found %v compute flavors", len(flavor_to_resource))
	glog.V(1).Info("Claiming to support Instances")

	return &Instances{compute, flavor_to_resource}, true
}

func (i *Instances) List(name_filter string) ([]string, error) {
	glog.V(2).Infof("rackspace List(%v) called", name_filter)

	opts := os_servers.ListOpts{
		Name:   name_filter,
		Status: "ACTIVE",
	}
	pager := servers.List(i.compute, opts)

	ret := make([]string, 0)
	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		sList, err := servers.ExtractServers(page)
		if err != nil {
			return false, err
		}
		for _, server := range sList {
			ret = append(ret, server.Name)
		}
		return true, nil
	})
	if err != nil {
		return nil, err
	}

	glog.V(2).Infof("Found %v entries: %v", len(ret), ret)

	return ret, nil
}

func serverHasAddress(srv os_servers.Server, ip string) bool {
	if ip == firstAddr(srv.Addresses["private"]) {
		return true
	}
	if ip == firstAddr(srv.Addresses["public"]) {
		return true
	}
	if ip == srv.AccessIPv4 {
		return true
	}
	if ip == srv.AccessIPv6 {
		return true
	}
	return false
}

func getServerByAddress(client *gophercloud.ServiceClient, name string) (*os_servers.Server, error) {
	pager := servers.List(client, nil)

	serverList := make([]os_servers.Server, 0, 1)

	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		s, err := servers.ExtractServers(page)
		if err != nil {
			return false, err
		}
		for _, v := range s {
			if serverHasAddress(v, name) {
				serverList = append(serverList, v)
			}
		}
		if len(serverList) > 1 {
			return false, ErrMultipleResults
		}
		return true, nil
	})
	if err != nil {
		return nil, err
	}

	if len(serverList) == 0 {
		return nil, ErrNotFound
	} else if len(serverList) > 1 {
		return nil, ErrMultipleResults
	}

	return &serverList[0], nil
}

func getServerByName(client *gophercloud.ServiceClient, name string) (*os_servers.Server, error) {
	if net.ParseIP(name) != nil {
		// we're an IP, so we'll have to walk the full list of servers to
		// figure out which one we are.
		return getServerByAddress(client, name)
	}
	opts := os_servers.ListOpts{
		Name:   fmt.Sprintf("^%s$", regexp.QuoteMeta(name)),
		Status: "ACTIVE",
	}
	pager := servers.List(client, opts)

	serverList := make([]os_servers.Server, 0, 1)

	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		s, err := servers.ExtractServers(page)
		if err != nil {
			return false, err
		}
		serverList = append(serverList, s...)
		if len(serverList) > 1 {
			return false, ErrMultipleResults
		}
		return true, nil
	})
	if err != nil {
		return nil, err
	}

	if len(serverList) == 0 {
		return nil, ErrNotFound
	} else if len(serverList) > 1 {
		return nil, ErrMultipleResults
	}

	return &serverList[0], nil
}

func firstAddr(netblob interface{}) string {
	// Run-time types for the win :(
	list, ok := netblob.([]interface{})
	if !ok || len(list) < 1 {
		return ""
	}
	props, ok := list[0].(map[string]interface{})
	if !ok {
		return ""
	}
	tmp, ok := props["addr"]
	if !ok {
		return ""
	}
	addr, ok := tmp.(string)
	if !ok {
		return ""
	}
	return addr
}

func getAddressByName(api *gophercloud.ServiceClient, name string) (string, error) {
	srv, err := getServerByName(api, name)
	if err != nil {
		return "", err
	}

	var s string
	if s == "" {
		s = firstAddr(srv.Addresses["private"])
	}
	if s == "" {
		s = firstAddr(srv.Addresses["public"])
	}
	if s == "" {
		s = srv.AccessIPv4
	}
	if s == "" {
		s = srv.AccessIPv6
	}
	if s == "" {
		return "", ErrNoAddressFound
	}
	return s, nil
}

func (i *Instances) IPAddress(name string) (net.IP, error) {
	glog.V(2).Infof("IPAddress(%v) called", name)

	ip, err := getAddressByName(i.compute, name)
	if err != nil {
		return nil, err
	}

	glog.V(2).Infof("IPAddress(%v) => %v", name, ip)

	return net.ParseIP(ip), err
}

func (i *Instances) GetNodeResources(name string) (*api.NodeResources, error) {
	glog.V(2).Infof("GetNodeResources(%v) called", name)

	srv, err := getServerByName(i.compute, name)
	if err != nil {
		return nil, err
	}

	s, ok := srv.Flavor["id"]
	if !ok {
		return nil, ErrAttrNotFound
	}
	flavId, ok := s.(string)
	if !ok {
		return nil, ErrAttrNotFound
	}
	rsrc, ok := i.flavor_to_resource[flavId]
	if !ok {
		return nil, ErrNotFound
	}

	glog.V(2).Infof("GetNodeResources(%v) => %v", name, rsrc)

	return rsrc, nil
}

func (os *Rackspace) Clusters() (cloudprovider.Clusters, bool) {
	return nil, false
}

type LoadBalancer struct {
	network *gophercloud.ServiceClient
	compute *gophercloud.ServiceClient
	opts    LoadBalancerOpts
}

func (os *Rackspace) TCPLoadBalancer() (cloudprovider.TCPLoadBalancer, bool) {
	// TODO: Search for and support Rackspace loadbalancer API, and others.
	network, err := rackspace.NewLBV1(os.provider, gophercloud.EndpointOpts{
		Region: os.region,
	})
	if err != nil {
		glog.Warningf("Failed to find LB endpoint: %v", err)
		return nil, false
	}

	compute, err := rackspace.NewComputeV2(os.provider, gophercloud.EndpointOpts{
		Region: os.region,
	})
	if err != nil {
		glog.Warningf("Failed to find compute endpoint: %v", err)
		return nil, false
	}

	glog.V(1).Info("Claiming to support TCPLoadBalancer")

	return &LoadBalancer{network, compute, os.lbOpts}, true
}

func getVipByName(client *gophercloud.ServiceClient, name string) (*lbs.LoadBalancer, error) {
	pager := lbs.List(client, nil)

	vipList := make([]lbs.LoadBalancer, 0)

	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		list, err := lbs.ExtractLBs(page)
		if err != nil {
			return false, err
		}
		for _, v := range list {
			if v.Name == name {
				vipList = append(vipList, v)
			}
		}

		if len(vipList) > 1 {
			return false, ErrMultipleResults
		}
		return true, nil
	})
	if err != nil {
		return nil, err
	}

	if len(vipList) == 0 {
		return nil, ErrNotFound
	} else if len(vipList) > 1 {
		return nil, ErrMultipleResults
	}

	return &vipList[0], nil
}

func (lb *LoadBalancer) TCPLoadBalancerExists(name, region string) (bool, error) {
	vip, err := getVipByName(lb.network, name)
	if err == ErrNotFound {
		return false, nil
	}
	return vip != nil, err
}

// TODO: This code currently ignores 'region' and always creates a
// loadbalancer in only the current Rackspace region.  We should take
// a list of regions (from config) and query/create loadbalancers in
// each region.

func (lb *LoadBalancer) CreateTCPLoadBalancer(name, region string, externalIP net.IP, port int, hosts []string, affinity api.AffinityType) (*api.LoadBalancerInfo, error) {
	glog.V(2).Infof("CreateTCPLoadBalancer(%v, %v, %v, %v, %v)", name, region, externalIP, port, hosts)
	if affinity != api.AffinityTypeNone {
		return nil, fmt.Errorf("unsupported load balancer affinity: %v", affinity)
	}

	vipList := []vips.VIP{
		{
			Type:    vips.PUBLIC,
			Version: vips.IPV4,
		},
	}

	pool, err := lbs.Create(lb.network, lbs.CreateOpts{
		Name:     name,
		Protocol: "TCP_CLIENT_FIRST",
		VIPs:     vipList,
		Port:     port,
	}).Extract()
	if err != nil {
		return nil, err
	}

	nodeList := make(nodes.CreateOpts, 0, 1)
	for _, host := range hosts {
		addr, err := getAddressByName(lb.compute, host)
		if err != nil {
			return nil, err
		}

		nodeList = append(nodeList, nodes.CreateOpt{
			Port:      port,
			Address:   addr,
			Condition: nodes.ENABLED,
			Type:      nodes.PRIMARY,
		})
	}

	glog.V(2).Infof("Waiting for LB to be ready")
	poll, _ := lbs.Get(lb.network, pool.ID).Extract()
	for poll.Status != lbs.ACTIVE {
		time.Sleep(time.Second * 5)
		poll, _ = lbs.Get(lb.network, pool.ID).Extract()
	}

	glog.V(2).Infof("Adding nodes to LB %v with ID %v", name, pool.ID)
	_, err = nodes.Create(lb.network, pool.ID, nodeList).ExtractNodes()
	if err != nil {
		lbs.Delete(lb.network, pool.ID)
		return nil, err
	}

	glog.V(2).Infof("LB Source Address is %+v", pool.SourceAddrs)
	loadBalancerInfo := &api.LoadBalancerInfo{Rewrite: true}
	loadBalancerInfo.DestIP = pool.VIPs[0].Address
	loadBalancerInfo.SourceIP = pool.SourceAddrs.IPv4Private
	return loadBalancerInfo, nil
}

func (lb *LoadBalancer) UpdateTCPLoadBalancer(name, region string, hosts []string) error {
	glog.V(2).Infof("UpdateTCPLoadBalancer(%v, %v, %v)", name, region, hosts)

	vip, err := getVipByName(lb.network, name)
	if err != nil {
		return err
	}

	// Set of member (addresses) that _should_ exist
	addrs := map[string]bool{}
	for _, host := range hosts {
		addr, err := getAddressByName(lb.compute, host)
		if err != nil {
			return err
		}

		addrs[addr] = true
	}

	// Iterate over members that _do_ exist
	pager := nodes.List(lb.network, vip.ID, nil)
	err = pager.EachPage(func(page pagination.Page) (bool, error) {
		memList, err := nodes.ExtractNodes(page)
		if err != nil {
			return false, err
		}

		for _, member := range memList {
			if _, found := addrs[member.Address]; found {
				// Member already exists
				delete(addrs, member.Address)
			} else {
				// Member needs to be deleted
				err = nodes.Delete(lb.network, vip.ID, member.ID).ExtractErr()
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
	nodeList := make(nodes.CreateOpts, 0, 1)
	for addr := range addrs {
		nodeList = append(nodeList, nodes.CreateOpt{
			Address: addr,
			Port:    vip.Port,
		})
	}
	_, err = nodes.Create(lb.network, vip.ID, nodeList).ExtractNodes()
	if err != nil {
		return err
	}

	return nil
}

func (lb *LoadBalancer) DeleteTCPLoadBalancer(name, region string) error {
	glog.V(2).Infof("DeleteTCPLoadBalancer(%v, %v)", name, region)

	vip, err := getVipByName(lb.network, name)
	if err != nil {
		return err
	}

	pool, err := lbs.Get(lb.network, vip.ID).Extract()
	if err != nil {
		return err
	}

	// Ignore errors for everything following here

	lbs.Delete(lb.network, pool.ID)

	return nil
}

func (os *Rackspace) Zones() (cloudprovider.Zones, bool) {
	glog.V(1).Info("Claiming to support Zones")

	return os, true
}
func (os *Rackspace) GetZone() (cloudprovider.Zone, error) {
	glog.V(1).Infof("Current zone is %v", os.region)

	return cloudprovider.Zone{Region: os.region}, nil
}
