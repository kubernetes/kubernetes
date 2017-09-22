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
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack"
	apiversions_v1 "github.com/gophercloud/gophercloud/openstack/blockstorage/v1/apiversions"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/attachinterfaces"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/servers"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/extensions/trusts"
	tokens3 "github.com/gophercloud/gophercloud/openstack/identity/v3/tokens"
	"github.com/gophercloud/gophercloud/pagination"
	"github.com/mitchellh/mapstructure"
	"gopkg.in/gcfg.v1"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	netutil "k8s.io/apimachinery/pkg/util/net"
	certutil "k8s.io/client-go/util/cert"
	v1helper "k8s.io/kubernetes/pkg/api/v1/helper"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller"
)

const (
	ProviderName     = "openstack"
	AvailabilityZone = "availability_zone"
)

var ErrNotFound = errors.New("Failed to find object")
var ErrMultipleResults = errors.New("Multiple results where only one expected")
var ErrNoAddressFound = errors.New("No address found for host")

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
	LBVersion            string     `gcfg:"lb-version"`          // overrides autodetection. v1 or v2
	SubnetId             string     `gcfg:"subnet-id"`           // overrides autodetection.
	FloatingNetworkId    string     `gcfg:"floating-network-id"` // If specified, will create floating ip for loadbalancer, or do not create floating ip.
	LBMethod             string     `gcfg:"lb-method"`           // default to ROUND_ROBIN.
	CreateMonitor        bool       `gcfg:"create-monitor"`
	MonitorDelay         MyDuration `gcfg:"monitor-delay"`
	MonitorTimeout       MyDuration `gcfg:"monitor-timeout"`
	MonitorMaxRetries    uint       `gcfg:"monitor-max-retries"`
	ManageSecurityGroups bool       `gcfg:"manage-security-groups"`
	NodeSecurityGroupID  string     `gcfg:"node-security-group"`
}

type BlockStorageOpts struct {
	BSVersion       string `gcfg:"bs-version"`        // overrides autodetection. v1 or v2. Defaults to auto
	TrustDevicePath bool   `gcfg:"trust-device-path"` // See Issue #33128
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
		TenantId   string `gcfg:"tenant-id"`
		TenantName string `gcfg:"tenant-name"`
		TrustId    string `gcfg:"trust-id"`
		DomainId   string `gcfg:"domain-id"`
		DomainName string `gcfg:"domain-name"`
		Region     string
		CAFile     string `gcfg:"ca-file"`
	}
	LoadBalancer LoadBalancerOpts
	BlockStorage BlockStorageOpts
	Route        RouterOpts
}

func init() {
	RegisterMetrics()

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
		TenantID:         cfg.Global.TenantId,
		TenantName:       cfg.Global.TenantName,
		DomainID:         cfg.Global.DomainId,
		DomainName:       cfg.Global.DomainName,

		// Persistent service, so we need to be able to renew tokens.
		AllowReauth: true,
	}
}

func (cfg Config) toAuth3Options() tokens3.AuthOptions {
	return tokens3.AuthOptions{
		IdentityEndpoint: cfg.Global.AuthUrl,
		Username:         cfg.Global.Username,
		UserID:           cfg.Global.UserId,
		Password:         cfg.Global.Password,
		DomainID:         cfg.Global.DomainId,
		DomainName:       cfg.Global.DomainName,
		AllowReauth:      true,
	}
}

func readConfig(config io.Reader) (Config, error) {
	if config == nil {
		err := fmt.Errorf("no OpenStack cloud provider config file given")
		return Config{}, err
	}

	var cfg Config

	// Set default values for config params
	cfg.BlockStorage.BSVersion = "auto"
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

// check opts for OpenStack
func checkOpenStackOpts(openstackOpts *OpenStack) error {
	lbOpts := openstackOpts.lbOpts

	// if need to create health monitor for Neutron LB,
	// monitor-delay, monitor-timeout and monitor-max-retries should be set.
	emptyDuration := MyDuration{}
	if lbOpts.CreateMonitor {
		if lbOpts.MonitorDelay == emptyDuration {
			return fmt.Errorf("monitor-delay not set in cloud provider config")
		}
		if lbOpts.MonitorTimeout == emptyDuration {
			return fmt.Errorf("monitor-timeout not set in cloud provider config")
		}
		if lbOpts.MonitorMaxRetries == uint(0) {
			return fmt.Errorf("monitor-max-retries not set in cloud provider config")
		}
	}

	// if enable ManageSecurityGroups, node-security-group should be set.
	if lbOpts.ManageSecurityGroups {
		if len(lbOpts.NodeSecurityGroupID) == 0 {
			return fmt.Errorf("node-security-group not set in cloud provider config")
		}
	}

	return nil
}

func newOpenStack(cfg Config) (*OpenStack, error) {
	provider, err := openstack.NewClient(cfg.Global.AuthUrl)
	if err != nil {
		return nil, err
	}
	if cfg.Global.CAFile != "" {
		roots, err := certutil.NewPool(cfg.Global.CAFile)
		if err != nil {
			return nil, err
		}
		config := &tls.Config{}
		config.RootCAs = roots
		provider.HTTPClient.Transport = netutil.SetOldTransportDefaults(&http.Transport{TLSClientConfig: config})

	}
	if cfg.Global.TrustId != "" {
		opts := cfg.toAuth3Options()
		authOptsExt := trusts.AuthOptsExt{
			TrustID:            cfg.Global.TrustId,
			AuthOptionsBuilder: &opts,
		}
		err = openstack.AuthenticateV3(provider, authOptsExt, gophercloud.EndpointOpts{})
	} else {
		err = openstack.Authenticate(provider, cfg.toAuthOptions())
	}

	if err != nil {
		return nil, err
	}

	os := OpenStack{
		provider:  provider,
		region:    cfg.Global.Region,
		lbOpts:    cfg.LoadBalancer,
		bsOpts:    cfg.BlockStorage,
		routeOpts: cfg.Route,
	}

	err = checkOpenStackOpts(&os)
	if err != nil {
		return nil, err
	}

	return &os, nil
}

// Initialize passes a Kubernetes clientBuilder interface to the cloud provider
func (os *OpenStack) Initialize(clientBuilder controller.ControllerClientBuilder) {}

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

	for network, addrList := range addresses {
		for _, props := range addrList {
			var addressType v1.NodeAddressType
			if props.IpType == "floating" || network == "public" {
				addressType = v1.NodeExternalIP
			} else {
				addressType = v1.NodeInternalIP
			}

			v1helper.AddToNodeAddresses(&addrs,
				v1.NodeAddress{
					Type:    addressType,
					Address: props.Addr,
				},
			)
		}
	}

	// AccessIPs are usually duplicates of "public" addresses.
	if srv.AccessIPv4 != "" {
		v1helper.AddToNodeAddresses(&addrs,
			v1.NodeAddress{
				Type:    v1.NodeExternalIP,
				Address: srv.AccessIPv4,
			},
		)
	}

	if srv.AccessIPv6 != "" {
		v1helper.AddToNodeAddresses(&addrs,
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

// getAttachedInterfacesByID returns the node interfaces of the specified instance.
func getAttachedInterfacesByID(client *gophercloud.ServiceClient, serviceID string) ([]attachinterfaces.Interface, error) {
	var interfaces []attachinterfaces.Interface

	pager := attachinterfaces.List(client, serviceID)
	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		s, err := attachinterfaces.ExtractInterfaces(page)
		if err != nil {
			return false, err
		}
		interfaces = append(interfaces, s...)
		return true, nil
	})
	if err != nil {
		return interfaces, err
	}

	return interfaces, nil
}

func (os *OpenStack) Clusters() (cloudprovider.Clusters, bool) {
	return nil, false
}

// ProviderName returns the cloud provider ID.
func (os *OpenStack) ProviderName() string {
	return ProviderName
}

// ScrubDNS filters DNS settings for pods.
func (os *OpenStack) ScrubDNS(nameServers, searches []string) ([]string, []string) {
	return nameServers, searches
}

// HasClusterID returns true if the cluster has a clusterID
func (os *OpenStack) HasClusterID() bool {
	return true
}

func (os *OpenStack) LoadBalancer() (cloudprovider.LoadBalancer, bool) {
	glog.V(4).Info("openstack.LoadBalancer() called")

	// TODO: Search for and support Rackspace loadbalancer API, and others.
	network, err := os.NewNetworkV2()
	if err != nil {
		return nil, false
	}

	compute, err := os.NewComputeV2()
	if err != nil {
		return nil, false
	}

	lbVersion := os.lbOpts.LBVersion
	if lbVersion == "" {
		// No version specified, try newest supported by server
		netExts, err := networkExtensions(network)
		if err != nil {
			glog.Warningf("Failed to list neutron extensions: %v", err)
			return nil, false
		}

		if netExts["lbaasv2"] {
			lbVersion = "v2"
		} else if netExts["lbaas"] {
			lbVersion = "v1"
		} else {
			glog.Warningf("Failed to find neutron LBaaS extension (v1 or v2)")
			return nil, false
		}
		glog.V(3).Infof("Using LBaaS extension %v", lbVersion)
	}

	glog.V(1).Info("Claiming to support LoadBalancer")

	if lbVersion == "v2" {
		return &LbaasV2{LoadBalancer{network, compute, os.lbOpts}}, true
	} else if lbVersion == "v1" {
		// Since LBaaS v1 is deprecated in the OpenStack Liberty release, so deprecate LBaaSV1 at V1.8, then remove LBaaSV1 after V1.9.
		// Reference OpenStack doc:	https://docs.openstack.org/mitaka/networking-guide/config-lbaas.html
		glog.Warningf("The LBaaS v1 of OpenStack cloud provider has been deprecated, Please use LBaaS v2")
		return &LbaasV1{LoadBalancer{network, compute, os.lbOpts}}, true
	} else {
		glog.Warningf("Config error: unrecognised lb-version \"%v\"", lbVersion)
		return nil, false
	}
}

func isNotFound(err error) bool {
	e, ok := err.(*gophercloud.ErrUnexpectedResponseCode)
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

// GetZoneByProviderID implements Zones.GetZoneByProviderID
// This is particularly useful in external cloud providers where the kubelet
// does not initialize node data.
func (os *OpenStack) GetZoneByProviderID(providerID string) (cloudprovider.Zone, error) {
	instanceID, err := instanceIDFromProviderID(providerID)
	if err != nil {
		return cloudprovider.Zone{}, err
	}

	compute, err := os.NewComputeV2()
	if err != nil {
		return cloudprovider.Zone{}, err
	}

	srv, err := servers.Get(compute, instanceID).Extract()
	if err != nil {
		return cloudprovider.Zone{}, err
	}

	zone := cloudprovider.Zone{
		FailureDomain: srv.Metadata[AvailabilityZone],
		Region:        os.region,
	}
	glog.V(4).Infof("The instance %s in zone %v", srv.Name, zone)

	return zone, nil
}

// GetZoneByNodeName implements Zones.GetZoneByNodeName
// This is particularly useful in external cloud providers where the kubelet
// does not initialize node data.
func (os *OpenStack) GetZoneByNodeName(nodeName types.NodeName) (cloudprovider.Zone, error) {
	compute, err := os.NewComputeV2()
	if err != nil {
		return cloudprovider.Zone{}, err
	}

	srv, err := getServerByName(compute, nodeName)
	if err != nil {
		if err == ErrNotFound {
			return cloudprovider.Zone{}, cloudprovider.InstanceNotFound
		}
		return cloudprovider.Zone{}, err
	}

	zone := cloudprovider.Zone{
		FailureDomain: srv.Metadata[AvailabilityZone],
		Region:        os.region,
	}
	glog.V(4).Infof("The instance %s in zone %v", srv.Name, zone)

	return zone, nil
}

func (os *OpenStack) Routes() (cloudprovider.Routes, bool) {
	glog.V(4).Info("openstack.Routes() called")

	network, err := os.NewNetworkV2()
	if err != nil {
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

	compute, err := os.NewComputeV2()
	if err != nil {
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

// Implementation of sort interface for blockstorage version probing
type APIVersionsByID []apiversions_v1.APIVersion

func (apiVersions APIVersionsByID) Len() int {
	return len(apiVersions)
}

func (apiVersions APIVersionsByID) Swap(i, j int) {
	apiVersions[i], apiVersions[j] = apiVersions[j], apiVersions[i]
}

func (apiVersions APIVersionsByID) Less(i, j int) bool {
	return apiVersions[i].ID > apiVersions[j].ID
}

func autoVersionSelector(apiVersion *apiversions_v1.APIVersion) string {
	switch strings.ToLower(apiVersion.ID) {
	case "v2.0":
		return "v2"
	case "v1.0":
		return "v1"
	default:
		return ""
	}
}

func doBsApiVersionAutodetect(availableApiVersions []apiversions_v1.APIVersion) string {
	sort.Sort(APIVersionsByID(availableApiVersions))
	for _, status := range []string{"CURRENT", "SUPPORTED"} {
		for _, version := range availableApiVersions {
			if strings.ToUpper(version.Status) == status {
				if detectedApiVersion := autoVersionSelector(&version); detectedApiVersion != "" {
					glog.V(3).Infof("Blockstorage API version probing has found a suitable %s api version: %s", status, detectedApiVersion)
					return detectedApiVersion
				}
			}
		}
	}

	return ""

}

func (os *OpenStack) volumeService(forceVersion string) (volumeService, error) {
	bsVersion := ""
	if forceVersion == "" {
		bsVersion = os.bsOpts.BSVersion
	} else {
		bsVersion = forceVersion
	}

	switch bsVersion {
	case "v1":
		sClient, err := os.NewBlockStorageV1()
		if err != nil {
			return nil, err
		}
		return &VolumesV1{sClient, os.bsOpts}, nil
	case "v2":
		sClient, err := os.NewBlockStorageV2()
		if err != nil {
			return nil, err
		}
		return &VolumesV2{sClient, os.bsOpts}, nil
	case "auto":
		sClient, err := os.NewBlockStorageV1()
		if err != nil {
			return nil, err
		}
		availableApiVersions := []apiversions_v1.APIVersion{}
		err = apiversions_v1.List(sClient).EachPage(func(page pagination.Page) (bool, error) {
			// returning false from this handler stops page iteration, error is propagated to the upper function
			apiversions, err := apiversions_v1.ExtractAPIVersions(page)
			if err != nil {
				glog.Errorf("Unable to extract api versions from page: %v", err)
				return false, err
			}
			availableApiVersions = append(availableApiVersions, apiversions...)
			return true, nil
		})

		if err != nil {
			glog.Errorf("Error when retrieving list of supported blockstorage api versions: %v", err)
			return nil, err
		}
		if autodetectedVersion := doBsApiVersionAutodetect(availableApiVersions); autodetectedVersion != "" {
			return os.volumeService(autodetectedVersion)
		} else {
			// Nothing suitable found, failed autodetection, just exit with appropriate message
			err_txt := "BlockStorage API version autodetection failed. " +
				"Please set it explicitly in cloud.conf in section [BlockStorage] with key `bs-version`"
			return nil, errors.New(err_txt)
		}

	default:
		err_txt := fmt.Sprintf("Config error: unrecognised bs-version \"%v\"", os.bsOpts.BSVersion)
		glog.Warningf(err_txt)
		return nil, errors.New(err_txt)
	}
}
