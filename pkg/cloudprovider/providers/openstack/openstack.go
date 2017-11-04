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
	"strings"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack"
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
	defaultTimeOut   = 60 * time.Second
)

var ErrNotFound = errors.New("failed to find object")
var ErrMultipleResults = errors.New("multiple results where only one expected")
var ErrNoAddressFound = errors.New("no address found for host")

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
	LBVersion            string     `gcfg:"lb-version"`          // overrides autodetection. Only support v2.
	SubnetId             string     `gcfg:"subnet-id"`           // overrides autodetection.
	FloatingNetworkId    string     `gcfg:"floating-network-id"` // If specified, will create floating ip for loadbalancer, or do not create floating ip.
	LBMethod             string     `gcfg:"lb-method"`           // default to ROUND_ROBIN.
	LBProvider           string     `gcfg:"lb-provider"`
	CreateMonitor        bool       `gcfg:"create-monitor"`
	MonitorDelay         MyDuration `gcfg:"monitor-delay"`
	MonitorTimeout       MyDuration `gcfg:"monitor-timeout"`
	MonitorMaxRetries    uint       `gcfg:"monitor-max-retries"`
	ManageSecurityGroups bool       `gcfg:"manage-security-groups"`
	NodeSecurityGroupIDs []string   // Do not specify, get it automatically when enable manage-security-groups. TODO(FengyunPan): move it into cache
}

type BlockStorageOpts struct {
	BSVersion       string `gcfg:"bs-version"`        // overrides autodetection. v1 or v2. Defaults to auto
	TrustDevicePath bool   `gcfg:"trust-device-path"` // See Issue #33128
	IgnoreVolumeAZ  bool   `gcfg:"ignore-volume-az"`
}

type RouterOpts struct {
	RouterId string `gcfg:"router-id"` // required
}

type MetadataOpts struct {
	SearchOrder    string     `gcfg:"search-order"`
	RequestTimeout MyDuration `gcfg:"request-timeout"`
}

// OpenStack is an implementation of cloud provider Interface for OpenStack.
type OpenStack struct {
	provider     *gophercloud.ProviderClient
	region       string
	lbOpts       LoadBalancerOpts
	bsOpts       BlockStorageOpts
	routeOpts    RouterOpts
	metadataOpts MetadataOpts
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
	Metadata     MetadataOpts
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
		return Config{}, fmt.Errorf("no OpenStack cloud provider config file given")
	}

	var cfg Config

	// Set default values for config params
	cfg.BlockStorage.BSVersion = "auto"
	cfg.BlockStorage.TrustDevicePath = false
	cfg.BlockStorage.IgnoreVolumeAZ = false
	cfg.Metadata.SearchOrder = fmt.Sprintf("%s,%s", configDriveID, metadataID)

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

func readInstanceID(searchOrder string) (string, error) {
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

	md, err := getMetadata(searchOrder)
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

	if err := checkMetadataSearchOrder(openstackOpts.metadataOpts.SearchOrder); err != nil {
		return err
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

	emptyDuration := MyDuration{}
	if cfg.Metadata.RequestTimeout == emptyDuration {
		cfg.Metadata.RequestTimeout.Duration = time.Duration(defaultTimeOut)
	}
	provider.HTTPClient.Timeout = cfg.Metadata.RequestTimeout.Duration

	os := OpenStack{
		provider:     provider,
		region:       cfg.Global.Region,
		lbOpts:       cfg.LoadBalancer,
		bsOpts:       cfg.BlockStorage,
		routeOpts:    cfg.Route,
		metadataOpts: cfg.Metadata,
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

	network, err := os.NewNetworkV2()
	if err != nil {
		return nil, false
	}

	compute, err := os.NewComputeV2()
	if err != nil {
		return nil, false
	}

	// LBaaS v1 is deprecated in the OpenStack Liberty release.
	// Currently kubernetes OpenStack cloud provider just support LBaaS v2.
	lbVersion := os.lbOpts.LBVersion
	if lbVersion != "" && lbVersion != "v2" {
		glog.Warningf("Config error: currently only support LBaaS v2, unrecognised lb-version \"%v\"", lbVersion)
		return nil, false
	}

	glog.V(1).Info("Claiming to support LoadBalancer")

	return &LbaasV2{LoadBalancer{network, compute, os.lbOpts}}, true
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
	md, err := getMetadata(os.metadataOpts.SearchOrder)
	if err != nil {
		return cloudprovider.Zone{}, err
	}

	zone := cloudprovider.Zone{
		FailureDomain: md.AvailabilityZone,
		Region:        os.region,
	}
	glog.V(4).Infof("Current zone is %v", zone)
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
		glog.V(3).Infof("Using Blockstorage API V1")
		return &VolumesV1{sClient, os.bsOpts}, nil
	case "v2":
		sClient, err := os.NewBlockStorageV2()
		if err != nil {
			return nil, err
		}
		glog.V(3).Infof("Using Blockstorage API V2")
		return &VolumesV2{sClient, os.bsOpts}, nil
	case "auto":
		// Currently kubernetes just support Cinder v1 and Cinder v2.
		// Choose Cinder v2 firstly, if kubernetes can't initialize cinder v2 client, try to initialize cinder v1 client.
		// Return appropriate message when kubernetes can't initialize them.
		// TODO(FengyunPan): revisit 'auto' after supporting Cinder v3.
		sClient, err := os.NewBlockStorageV2()
		if err != nil {
			sClient, err = os.NewBlockStorageV1()
			if err != nil {
				// Nothing suitable found, failed autodetection, just exit with appropriate message
				err_txt := "BlockStorage API version autodetection failed. " +
					"Please set it explicitly in cloud.conf in section [BlockStorage] with key `bs-version`"
				return nil, errors.New(err_txt)
			} else {
				glog.V(3).Infof("Using Blockstorage API V1")
				return &VolumesV1{sClient, os.bsOpts}, nil
			}
		} else {
			glog.V(3).Infof("Using Blockstorage API V2")
			return &VolumesV2{sClient, os.bsOpts}, nil
		}
	default:
		err_txt := fmt.Sprintf("Config error: unrecognised bs-version \"%v\"", os.bsOpts.BSVersion)
		return nil, errors.New(err_txt)
	}
}

func checkMetadataSearchOrder(order string) error {
	if order == "" {
		return errors.New("invalid value in section [Metadata] with key `search-order`. Value cannot be empty")
	}

	elements := strings.Split(order, ",")
	if len(elements) > 2 {
		return errors.New("invalid value in section [Metadata] with key `search-order`. Value cannot contain more than 2 elements")
	}

	for _, id := range elements {
		id = strings.TrimSpace(id)
		switch id {
		case configDriveID:
		case metadataID:
		default:
			return fmt.Errorf("invalid element %q found in section [Metadata] with key `search-order`."+
				"Supported elements include %q and %q", id, configDriveID, metadataID)
		}
	}

	return nil
}
