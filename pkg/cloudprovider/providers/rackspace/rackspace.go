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

package rackspace

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"os"
	"regexp"
	"time"

	"gopkg.in/gcfg.v1"

	"github.com/golang/glog"
	"github.com/rackspace/gophercloud"
	osvolumeattach "github.com/rackspace/gophercloud/openstack/compute/v2/extensions/volumeattach"
	osservers "github.com/rackspace/gophercloud/openstack/compute/v2/servers"
	"github.com/rackspace/gophercloud/pagination"
	"github.com/rackspace/gophercloud/rackspace"
	"github.com/rackspace/gophercloud/rackspace/blockstorage/v1/volumes"
	"github.com/rackspace/gophercloud/rackspace/compute/v2/servers"
	"github.com/rackspace/gophercloud/rackspace/compute/v2/volumeattach"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller"
)

const (
	ProviderName          = "rackspace"
	MetaDataPath          = "/media/configdrive/openstack/latest/meta_data.json"
	VolumeAvailableStatus = "available"
	VolumeInUseStatus     = "in-use"
	VolumeErrorStatus     = "error"
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

type MetaData struct {
	UUID string `json:"uuid"`
	Name string `json:"name"`
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

func probeNodeAddress(compute *gophercloud.ServiceClient, name string) (string, error) {
	id, err := readInstanceID()
	if err == nil {
		srv, err := servers.Get(compute, id).Extract()
		if err != nil {
			return "", err
		}
		return getAddressByServer(srv)
	}

	ip, err := getAddressByName(compute, name)
	if err != nil {
		return "", err
	}

	return ip, nil
}

func probeInstanceID(client *gophercloud.ServiceClient, name string) (string, error) {
	// Attempt to read id from config drive.
	id, err := readInstanceID()
	if err == nil {
		return id, nil
	}

	// Attempt to get the server by the name from the API
	server, err := getServerByName(client, name)
	if err != nil {
		return "", err
	}

	return server.ID, nil
}

func parseMetaData(file io.Reader) (string, error) {
	metaDataBytes, err := ioutil.ReadAll(file)
	if err != nil {
		return "", fmt.Errorf("Cannot read %s: %v", file, err)
	}

	metaData := MetaData{}
	err = json.Unmarshal(metaDataBytes, &metaData)
	if err != nil {
		return "", fmt.Errorf("Cannot parse %s: %v", MetaDataPath, err)
	}

	return metaData.UUID, nil
}

func readInstanceID() (string, error) {
	file, err := os.Open(MetaDataPath)
	if err != nil {
		return "", fmt.Errorf("Cannot open %s: %v", MetaDataPath, err)
	}
	defer file.Close()

	return parseMetaData(file)
}

func init() {
	cloudprovider.RegisterCloudProvider(ProviderName, func(config io.Reader) (cloudprovider.Interface, error) {
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

// Initialize passes a Kubernetes clientBuilder interface to the cloud provider
func (os *Rackspace) Initialize(clientBuilder controller.ControllerClientBuilder) {}

type Instances struct {
	compute *gophercloud.ServiceClient
}

// Instances returns an implementation of Instances for Rackspace.
func (os *Rackspace) Instances() (cloudprovider.Instances, bool) {
	glog.V(2).Info("rackspace.Instances() called")

	compute, err := os.getComputeClient()
	if err != nil {
		glog.Warningf("Failed to find compute endpoint: %v", err)
		return nil, false
	}
	glog.V(1).Info("Claiming to support Instances")

	return &Instances{compute}, true
}

func serverHasAddress(srv osservers.Server, ip string) bool {
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

func getServerByAddress(client *gophercloud.ServiceClient, name string) (*osservers.Server, error) {
	pager := servers.List(client, nil)

	serverList := make([]osservers.Server, 0, 1)

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

func getServerByName(client *gophercloud.ServiceClient, name string) (*osservers.Server, error) {
	if net.ParseIP(name) != nil {
		// we're an IP, so we'll have to walk the full list of servers to
		// figure out which one we are.
		return getServerByAddress(client, name)
	}
	opts := osservers.ListOpts{
		Name:   fmt.Sprintf("^%s$", regexp.QuoteMeta(name)),
		Status: "ACTIVE",
	}
	pager := servers.List(client, opts)

	serverList := make([]osservers.Server, 0, 1)

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

func getAddressByServer(srv *osservers.Server) (string, error) {
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

func getAddressByName(api *gophercloud.ServiceClient, name string) (string, error) {
	srv, err := getServerByName(api, name)
	if err != nil {
		return "", err
	}

	return getAddressByServer(srv)
}

func (i *Instances) NodeAddresses(nodeName types.NodeName) ([]v1.NodeAddress, error) {
	glog.V(2).Infof("NodeAddresses(%v) called", nodeName)
	serverName := mapNodeNameToServerName(nodeName)
	ip, err := probeNodeAddress(i.compute, serverName)
	if err != nil {
		return nil, err
	}

	glog.V(2).Infof("NodeAddresses(%v) => %v", serverName, ip)

	// net.ParseIP().String() is to maintain compatibility with the old code
	parsedIP := net.ParseIP(ip).String()
	return []v1.NodeAddress{
		{Type: v1.NodeInternalIP, Address: parsedIP},
		{Type: v1.NodeExternalIP, Address: parsedIP},
	}, nil
}

// NodeAddressesByProviderID returns the node addresses of an instances with the specified unique providerID
// This method will not be called from the node that is requesting this ID. i.e. metadata service
// and other local methods cannot be used here
func (i *Instances) NodeAddressesByProviderID(providerID string) ([]v1.NodeAddress, error) {
	instanceID, err := instanceIDFromProviderID(providerID)

	if err != nil {
		return []v1.NodeAddress{}, err
	}

	server, err := servers.Get(i.compute, instanceID).Extract()

	if err != nil {
		return []v1.NodeAddress{}, err
	}

	addresses, err := i.NodeAddresses(mapServerToNodeName(server))

	if err != nil {
		return []v1.NodeAddress{}, err
	}

	return addresses, nil
}

// mapNodeNameToServerName maps from a k8s NodeName to a rackspace Server Name
// This is a simple string cast.
func mapNodeNameToServerName(nodeName types.NodeName) string {
	return string(nodeName)
}

// mapServerToNodeName maps a rackspace Server to an k8s NodeName
func mapServerToNodeName(s *osservers.Server) types.NodeName {
	return types.NodeName(s.Name)
}

// ExternalID returns the cloud provider ID of the node with the specified Name (deprecated).
func (i *Instances) ExternalID(nodeName types.NodeName) (string, error) {
	serverName := mapNodeNameToServerName(nodeName)
	return probeInstanceID(i.compute, serverName)
}

// InstanceID returns the cloud provider ID of the kubelet's instance.
func (rs *Rackspace) InstanceID() (string, error) {
	return readInstanceID()
}

// InstanceID returns the cloud provider ID of the node with the specified Name.
func (i *Instances) InstanceID(nodeName types.NodeName) (string, error) {
	serverName := mapNodeNameToServerName(nodeName)
	return probeInstanceID(i.compute, serverName)
}

// InstanceType returns the type of the specified instance.
func (i *Instances) InstanceType(name types.NodeName) (string, error) {
	serverName := mapNodeNameToServerName(name)

	srv, err := getServerByName(i.compute, serverName)
	if err != nil {
		return "", err
	}

	return srvInstanceType(srv)
}

func srvInstanceType(srv *osservers.Server) (string, error) {
	val, ok := srv.Flavor["name"]

	if !ok {
		return "", fmt.Errorf("flavor name not present in server info")
	}

	flavor, ok := val.(string)

	if !ok {
		return "", fmt.Errorf("flavor name is not a string")
	}

	return flavor, nil
}

func instanceIDFromProviderID(providerID string) (instanceID string, err error) {
	var providerIDRegexp = regexp.MustCompile(`^rackspace://([^/]+)$`)
	matches := providerIDRegexp.FindStringSubmatch(providerID)
	if len(matches) != 2 {
		return "", fmt.Errorf("ProviderID \"%s\" didn't match expected format \"rackspace://InstanceID\"", providerID)
	}

	return matches[1], nil
}

// InstanceTypeByProviderID returns the cloudprovider instance type of the node with the specified unique providerID
// This method will not be called from the node that is requesting this ID. i.e. metadata service
// and other local methods cannot be used here
func (i *Instances) InstanceTypeByProviderID(providerID string) (string, error) {
	instanceID, err := instanceIDFromProviderID(providerID)

	if err != nil {
		return "", err
	}

	server, err := servers.Get(i.compute, instanceID).Extract()

	if err != nil {
		return "", err
	}

	return srvInstanceType(server)
}

func (i *Instances) AddSSHKeyToAllInstances(user string, keyData []byte) error {
	return errors.New("unimplemented")
}

// Implementation of Instances.CurrentNodeName
func (i *Instances) CurrentNodeName(hostname string) (types.NodeName, error) {
	// Beware when changing this, nodename == hostname assumption is crucial to
	// apiserver => kubelet communication.
	return types.NodeName(hostname), nil
}

func (os *Rackspace) Clusters() (cloudprovider.Clusters, bool) {
	return nil, false
}

// ProviderName returns the cloud provider ID.
func (os *Rackspace) ProviderName() string {
	return ProviderName
}

// ScrubDNS filters DNS settings for pods.
func (os *Rackspace) ScrubDNS(nameservers, searches []string) (nsOut, srchOut []string) {
	return nameservers, searches
}

// HasClusterID returns true if the cluster has a clusterID
func (os *Rackspace) HasClusterID() bool {
	return true
}

func (os *Rackspace) LoadBalancer() (cloudprovider.LoadBalancer, bool) {
	return nil, false
}

func (os *Rackspace) Zones() (cloudprovider.Zones, bool) {
	glog.V(1).Info("Claiming to support Zones")

	return os, true
}

func (os *Rackspace) Routes() (cloudprovider.Routes, bool) {
	return nil, false
}

func (os *Rackspace) GetZone() (cloudprovider.Zone, error) {
	glog.V(1).Infof("Current zone is %v", os.region)

	return cloudprovider.Zone{Region: os.region}, nil
}

// Create a volume of given size (in GiB)
func (rs *Rackspace) CreateVolume(name string, size int, vtype, availability string, tags *map[string]string) (string, string, error) {
	return "", "", errors.New("unimplemented")
}

func (rs *Rackspace) DeleteVolume(volumeID string) error {
	return errors.New("unimplemented")
}

func (rs *Rackspace) OperationPending(diskName string) (bool, string, error) {
	disk, err := rs.getVolume(diskName)
	if err != nil {
		return false, "", err
	}
	volumeStatus := disk.Status
	if volumeStatus == VolumeErrorStatus {
		glog.Errorf("status of volume %s is %s", diskName, volumeStatus)
		return false, volumeStatus, nil
	}
	if volumeStatus == VolumeAvailableStatus || volumeStatus == VolumeInUseStatus {
		return false, disk.Status, nil
	}
	return true, volumeStatus, nil
}

// Attaches given cinder volume to the compute running kubelet
func (rs *Rackspace) AttachDisk(instanceID, volumeID string) (string, error) {
	volume, err := rs.getVolume(volumeID)
	if err != nil {
		return "", err
	}

	if volume.Status != VolumeAvailableStatus {
		errmsg := fmt.Sprintf("volume %s status is %s, not %s, can not be attached to instance %s.", volume.Name, volume.Status, VolumeAvailableStatus, instanceID)
		glog.Errorf(errmsg)
		return "", errors.New(errmsg)
	}

	compute, err := rs.getComputeClient()
	if err != nil {
		return "", err
	}

	if len(volume.Attachments) > 0 {
		if instanceID == volume.Attachments[0]["server_id"] {
			glog.V(4).Infof("Volume: %q is already attached to compute: %q", volumeID, instanceID)
			return volume.ID, nil
		}

		errMsg := fmt.Sprintf("Volume %q is attached to a different compute: %q, should be detached before proceeding", volumeID, volume.Attachments[0]["server_id"])
		glog.Errorf(errMsg)
		return "", errors.New(errMsg)
	}

	_, err = volumeattach.Create(compute, instanceID, &osvolumeattach.CreateOpts{
		VolumeID: volume.ID,
	}).Extract()
	if err != nil {
		glog.Errorf("Failed to attach %s volume to %s compute", volumeID, instanceID)
		return "", err
	}
	glog.V(2).Infof("Successfully attached %s volume to %s compute", volumeID, instanceID)
	return volume.ID, nil
}

// GetDevicePath returns the path of an attached block storage volume, specified by its id.
func (rs *Rackspace) GetDevicePath(volumeID string) string {
	volume, err := rs.getVolume(volumeID)
	if err != nil {
		return ""
	}
	attachments := volume.Attachments
	if len(attachments) != 1 {
		glog.Warningf("Unexpected number of volume attachments on %s: %d", volumeID, len(attachments))
		return ""
	}
	return attachments[0]["device"].(string)
}

// Takes a partial/full disk id or volumeName
func (rs *Rackspace) getVolume(volumeID string) (*volumes.Volume, error) {
	client, err := rackspace.NewBlockStorageV1(rs.provider, gophercloud.EndpointOpts{
		Region: rs.region,
	})

	volume, err := volumes.Get(client, volumeID).Extract()
	if err != nil {
		glog.Errorf("Error occurred getting volume by ID: %s", volumeID)
		return &volumes.Volume{}, err
	}
	return volume, nil
}

func (rs *Rackspace) getComputeClient() (*gophercloud.ServiceClient, error) {
	client, err := rackspace.NewComputeV2(rs.provider, gophercloud.EndpointOpts{
		Region: rs.region,
	})
	if err != nil || client == nil {
		glog.Errorf("Unable to initialize nova client for region: %s", rs.region)
	}
	return client, nil
}

// Detaches given cinder volume from the compute running kubelet
func (rs *Rackspace) DetachDisk(instanceID, volumeID string) error {
	volume, err := rs.getVolume(volumeID)
	if err != nil {
		return err
	}

	if volume.Status != VolumeInUseStatus {
		errmsg := fmt.Sprintf("can not detach volume %s, its status is %s.", volume.Name, volume.Status)
		glog.Errorf(errmsg)
		return errors.New(errmsg)
	}

	compute, err := rs.getComputeClient()
	if err != nil {
		return err
	}

	if len(volume.Attachments) > 1 {
		// Rackspace does not support "multiattach", this is a sanity check.
		errmsg := fmt.Sprintf("Volume %s is attached to multiple instances, which is not supported by this provider.", volume.ID)
		return errors.New(errmsg)
	}

	if len(volume.Attachments) > 0 && instanceID == volume.Attachments[0]["server_id"] {
		// This is a blocking call and effects kubelet's performance directly.
		// We should consider kicking it out into a separate routine, if it is bad.
		err = volumeattach.Delete(compute, instanceID, volume.ID).ExtractErr()
		if err != nil {
			glog.Errorf("Failed to delete volume %s from compute %s attached %v", volume.ID, instanceID, err)
			return err
		}
		glog.V(2).Infof("Successfully detached volume: %s from compute: %s", volume.ID, instanceID)
	} else {
		errMsg := fmt.Sprintf("Disk: %s has no attachments or is not attached to compute: %s", volume.Name, instanceID)
		glog.Errorf(errMsg)
		return errors.New(errMsg)
	}

	return nil
}

// Get device path of attached volume to the compute running kubelet, as known by cinder
func (rs *Rackspace) GetAttachmentDiskPath(instanceID, volumeID string) (string, error) {
	// See issue #33128 - Cinder does not always tell you the right device path, as such
	// we must only use this value as a last resort.
	volume, err := rs.getVolume(volumeID)
	if err != nil {
		return "", err
	}

	if volume.Status != VolumeInUseStatus {
		errmsg := fmt.Sprintf("can not get device path of volume %s, its status is %s.", volume.Name, volume.Status)
		glog.Errorf(errmsg)
		return "", errors.New(errmsg)
	}
	if len(volume.Attachments) > 0 && volume.Attachments[0]["server_id"] != nil {
		if instanceID == volume.Attachments[0]["server_id"] {
			// Attachment[0]["device"] points to the device path
			// see http://developer.openstack.org/api-ref-blockstorage-v1.html
			return volume.Attachments[0]["device"].(string), nil
		} else {
			errMsg := fmt.Sprintf("Disk %q is attached to a different compute: %q, should be detached before proceeding", volumeID, volume.Attachments[0]["server_id"])
			glog.Errorf(errMsg)
			return "", errors.New(errMsg)
		}
	}
	return "", fmt.Errorf("volume %s is not attached to %s", volumeID, instanceID)
}

// query if a volume is attached to a compute instance
func (rs *Rackspace) DiskIsAttached(instanceID, volumeID string) (bool, error) {
	volume, err := rs.getVolume(volumeID)
	if err != nil {
		return false, err
	}
	if len(volume.Attachments) > 0 && volume.Attachments[0]["server_id"] != nil && instanceID == volume.Attachments[0]["server_id"] {
		return true, nil
	}
	return false, nil
}

// query if a list volumes are attached to a compute instance
func (rs *Rackspace) DisksAreAttached(instanceID string, volumeIDs []string) (map[string]bool, error) {
	attached := make(map[string]bool)
	for _, volumeID := range volumeIDs {
		attached[volumeID] = false
	}
	var returnedErr error
	for _, volumeID := range volumeIDs {
		result, err := rs.DiskIsAttached(instanceID, volumeID)
		if err != nil {
			returnedErr = fmt.Errorf("Error in checking disk %q attached: %v \n %v", volumeID, err, returnedErr)
			continue
		}
		if result {
			attached[volumeID] = true
		}

	}
	return attached, returnedErr
}

// query if we should trust the cinder provide deviceName, See issue #33128
func (rs *Rackspace) ShouldTrustDevicePath() bool {
	return true
}
