/*
Copyright 2014 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/openstack"
	"github.com/rackspace/gophercloud/openstack/compute/v2/servers"
	"github.com/rackspace/gophercloud/openstack/identity/v3/extensions/trust"
	token3 "github.com/rackspace/gophercloud/openstack/identity/v3/tokens"
	"github.com/rackspace/gophercloud/pagination"
	"gopkg.in/gcfg.v1"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/types"
)

const ProviderName = "openstack"

var ErrNotFound = errors.New("Failed to find object")
var ErrMultipleResults = errors.New("Multiple results where only one expected")
var ErrNoAddressFound = errors.New("No address found for host")
var ErrAttrNotFound = errors.New("Expected attribute not found")

const (
	MiB = 1024 * 1024
	GB  = 1000 * 1000 * 1000
)

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

type LoadBalancer struct {
	network *gophercloud.ServiceClient
	compute *gophercloud.ServiceClient
	opts    LoadBalancerOpts
}

type LoadBalancerOpts struct {
	LBVersion            string     `gcfg:"lb-version"` // overrides autodetection. v1 or v2
	SubnetId             string     `gcfg:"subnet-id"`  // required
	FloatingNetworkId    string     `gcfg:"floating-network-id"`
	LBMethod             string     `gcfg:"lb-method"`
	CreateMonitor        bool       `gcfg:"create-monitor"`
	MonitorDelay         MyDuration `gcfg:"monitor-delay"`
	MonitorTimeout       MyDuration `gcfg:"monitor-timeout"`
	MonitorMaxRetries    uint       `gcfg:"monitor-max-retries"`
	ManageSecurityGroups bool       `gcfg:"manage-security-groups"`
	NodeSecurityGroupID  string     `gcfg:"node-security-group"`
}

type BlockStorageOpts struct {
	TrustDevicePath bool `gcfg:"trust-device-path"` // See Issue #33128
}

type RouterOpts struct {
	RouterId string `gcfg:"router-id"` // required
}

// OpenStack is an implementation of cloud provider Interface for OpenStack.
type OpenStack struct {
	provider  *gophercloud.ProviderClient
	region    string
	lbOpts    LoadBalancerOpts
	bsOpts    BlockStorageOpts
	routeOpts RouterOpts
	// InstanceID of the server where this OpenStack object is instantiated.
	localInstanceID string
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
		TrustId    string `gcfg:"trust-id"`
		DomainId   string `gcfg:"domain-id"`
		DomainName string `gcfg:"domain-name"`
		Region     string
	}
	LoadBalancer LoadBalancerOpts
	BlockStorage BlockStorageOpts
	Route        RouterOpts
}

func init() {
	cloudprovider.RegisterCloudProvider(ProviderName, func(config io.Reader) (cloudprovider.Interface, error) {
		cfg, err := readConfig(config)
		if err != nil {
			return nil, err
		}
		return newOpenStack(cfg)
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
		DomainID:         cfg.Global.DomainId,
		DomainName:       cfg.Global.DomainName,

		// Persistent service, so we need to be able to renew tokens.
		AllowReauth: true,
	}
}

func readConfig(config io.Reader) (Config, error) {
	if config == nil {
		err := fmt.Errorf("no OpenStack cloud provider config file given")
		return Config{}, err
	}

	var cfg Config

	// Set default values for config params
	cfg.BlockStorage.TrustDevicePath = false

	err := gcfg.ReadInto(&cfg, config)
	return cfg, err
}

// Tiny helper for conditional unwind logic
type Caller bool

func NewCaller() Caller   { return Caller(true) }
func (c *Caller) Disarm() { *c = false }

func (c *Caller) Call(f func()) {
	if *c {
		f()
	}
}

func readInstanceID() (string, error) {
	// Try to find instance ID on the local filesystem (created by cloud-init)
	const instanceIDFile = "/var/lib/cloud/data/instance-id"
	idBytes, err := ioutil.ReadFile(instanceIDFile)
	if err == nil {
		instanceID := string(idBytes)
		instanceID = strings.TrimSpace(instanceID)
		glog.V(3).Infof("Got instance id from %s: %s", instanceIDFile, instanceID)
		if instanceID != "" {
			return instanceID, nil
		}
		// Fall through to metadata server lookup
	}

	md, err := getMetadata()
	if err != nil {
		return "", err
	}

	return md.Uuid, nil
}

func newOpenStack(cfg Config) (*OpenStack, error) {
	provider, err := openstack.NewClient(cfg.Global.AuthUrl)
	if err != nil {
		return nil, err
	}
	if cfg.Global.TrustId != "" {
		authOptionsExt := trust.AuthOptionsExt{
			TrustID:     cfg.Global.TrustId,
			AuthOptions: token3.AuthOptions{AuthOptions: cfg.toAuthOptions()},
		}
		err = trust.AuthenticateV3Trust(provider, authOptionsExt)
	} else {
		err = openstack.Authenticate(provider, cfg.toAuthOptions())
	}

	if err != nil {
		return nil, err
	}

	id, err := readInstanceID()
	if err != nil {
		return nil, err
	}

	os := OpenStack{
		provider:        provider,
		region:          cfg.Global.Region,
		lbOpts:          cfg.LoadBalancer,
		bsOpts:          cfg.BlockStorage,
		routeOpts:       cfg.Route,
		localInstanceID: id,
	}

	return &os, nil
}

// mapNodeNameToServerName maps a k8s NodeName to an OpenStack Server Name
// This is a simple string cast.
func mapNodeNameToServerName(nodeName types.NodeName) string {
	return string(nodeName)
}

// mapServerToNodeName maps an OpenStack Server to a k8s NodeName
func mapServerToNodeName(server *servers.Server) types.NodeName {
	// Node names are always lowercase, and (at least)
	// routecontroller does case-sensitive string comparisons
	// assuming this
	return types.NodeName(strings.ToLower(server.Name))
}

func foreachServer(client *gophercloud.ServiceClient, opts servers.ListOptsBuilder, handler func(*servers.Server) (bool, error)) error {
	pager := servers.List(client, opts)

	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		s, err := servers.ExtractServers(page)
		if err != nil {
			return false, err
		}
		for _, server := range s {
			ok, err := handler(&server)
			if !ok || err != nil {
				return false, err
			}
		}
		return true, nil
	})
	return err
}

func getServerByName(client *gophercloud.ServiceClient, name types.NodeName) (*servers.Server, error) {
	opts := servers.ListOpts{
		Name:   fmt.Sprintf("^%s$", regexp.QuoteMeta(mapNodeNameToServerName(name))),
		Status: "ACTIVE",
	}
	pager := servers.List(client, opts)

	serverList := make([]servers.Server, 0, 1)

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

func nodeAddresses(srv *servers.Server) ([]v1.NodeAddress, error) {
	addrs := []v1.NodeAddress{}

	type Address struct {
		IpType string `mapstructure:"OS-EXT-IPS:type"`
		Addr   string
	}

	var addresses map[string][]Address
	err := mapstructure.Decode(srv.Addresses, &addresses)
	if err != nil {
		return nil, err
	}

	for network, addrlist := range addresses {
		for _, props := range addrlist {
			var addressType v1.NodeAddressType
			if props.IpType == "floating" || network == "public" {
				addressType = v1.NodeExternalIP
			} else {
				addressType = v1.NodeInternalIP
			}

			v1.AddToNodeAddresses(&addrs,
				v1.NodeAddress{
					Type:    addressType,
					Address: props.Addr,
				},
			)
		}
	}

	// AccessIPs are usually duplicates of "public" addresses.
	if srv.AccessIPv4 != "" {
		v1.AddToNodeAddresses(&addrs,
			v1.NodeAddress{
				Type:    v1.NodeExternalIP,
				Address: srv.AccessIPv4,
			},
		)
	}

	if srv.AccessIPv6 != "" {
		v1.AddToNodeAddresses(&addrs,
			v1.NodeAddress{
				Type:    v1.NodeExternalIP,
				Address: srv.AccessIPv6,
			},
		)
	}

	return addrs, nil
}

func getAddressesByName(client *gophercloud.ServiceClient, name types.NodeName) ([]v1.NodeAddress, error) {
	srv, err := getServerByName(client, name)
	if err != nil {
		return nil, err
	}

	return nodeAddresses(srv)
}

func getAddressByName(client *gophercloud.ServiceClient, name types.NodeName) (string, error) {
	addrs, err := getAddressesByName(client, name)
	if err != nil {
		return "", err
	} else if len(addrs) == 0 {
		return "", ErrNoAddressFound
	}

	for _, addr := range addrs {
		if addr.Type == v1.NodeInternalIP {
			return addr.Address, nil
		}
	}

	return addrs[0].Address, nil
}

func (os *OpenStack) Clusters() (cloudprovider.Clusters, bool) {
	return nil, false
}

// ProviderName returns the cloud provider ID.
func (os *OpenStack) ProviderName() string {
	return ProviderName
}

// ScrubDNS filters DNS settings for pods.
func (os *OpenStack) ScrubDNS(nameservers, searches []string) (nsOut, srchOut []string) {
	return nameservers, searches
}

func (os *OpenStack) LoadBalancer() (cloudprovider.LoadBalancer, bool) {
	glog.V(4).Info("openstack.LoadBalancer() called")

	// TODO: Search for and support Rackspace loadbalancer API, and others.
	network, err := openstack.NewNetworkV2(os.provider, gophercloud.EndpointOpts{
		Region: os.region,
	})
	if err != nil {
		glog.Warningf("Failed to find network endpoint: %v", err)
		return nil, false
	}

	compute, err := openstack.NewComputeV2(os.provider, gophercloud.EndpointOpts{
		Region: os.region,
	})
	if err != nil {
		glog.Warningf("Failed to find compute endpoint: %v", err)
		return nil, false
	}

	lbversion := os.lbOpts.LBVersion
	if lbversion == "" {
		// No version specified, try newest supported by server
		netExts, err := networkExtensions(network)
		if err != nil {
			glog.Warningf("Failed to list neutron extensions: %v", err)
			return nil, false
		}

		if netExts["lbaasv2"] {
			lbversion = "v2"
		} else if netExts["lbaas"] {
			lbversion = "v1"
		} else {
			glog.Warningf("Failed to find neutron LBaaS extension (v1 or v2)")
			return nil, false
		}
		glog.V(3).Infof("Using LBaaS extension %v", lbversion)
	}

	glog.V(1).Info("Claiming to support LoadBalancer")

	if lbversion == "v2" {
		return &LbaasV2{LoadBalancer{network, compute, os.lbOpts}}, true
	} else if lbversion == "v1" {
		return &LbaasV1{LoadBalancer{network, compute, os.lbOpts}}, true
	} else {
		glog.Warningf("Config error: unrecognised lb-version \"%v\"", lbversion)
		return nil, false
	}
}

func isNotFound(err error) bool {
	e, ok := err.(*gophercloud.UnexpectedResponseCodeError)
	return ok && e.Actual == http.StatusNotFound
}

func (os *OpenStack) Zones() (cloudprovider.Zones, bool) {
	glog.V(1).Info("Claiming to support Zones")

	return os, true
}
func (os *OpenStack) GetZone() (cloudprovider.Zone, error) {
	md, err := getMetadata()
	if err != nil {
		return cloudprovider.Zone{}, err
	}

	zone := cloudprovider.Zone{
		FailureDomain: md.AvailabilityZone,
		Region:        os.region,
	}
	glog.V(1).Infof("Current zone is %v", zone)

	return zone, nil
}

func (os *OpenStack) Routes() (cloudprovider.Routes, bool) {
	glog.V(4).Info("openstack.Routes() called")

	network, err := openstack.NewNetworkV2(os.provider, gophercloud.EndpointOpts{
		Region: os.region,
	})
	if err != nil {
		glog.Warningf("Failed to find network endpoint: %v", err)
		return nil, false
	}

	netExts, err := networkExtensions(network)
	if err != nil {
		glog.Warningf("Failed to list neutron extensions: %v", err)
		return nil, false
	}

	if !netExts["extraroute"] {
		glog.V(3).Infof("Neutron extraroute extension not found, required for Routes support")
		return nil, false
	}

	compute, err := openstack.NewComputeV2(os.provider, gophercloud.EndpointOpts{
		Region: os.region,
	})
	if err != nil {
		glog.Warningf("Failed to find compute endpoint: %v", err)
		return nil, false
	}

	r, err := NewRoutes(compute, network, os.routeOpts)
	if err != nil {
		glog.Warningf("Error initialising Routes support: %v", err)
		return nil, false
	}

	glog.V(1).Info("Claiming to support Routes")

	return r, true
}
