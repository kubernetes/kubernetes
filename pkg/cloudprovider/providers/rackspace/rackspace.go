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
	"strings"
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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/types"
)

const ProviderName = "rackspace"
const metaDataPath = "/media/configdrive/openstack/latest/meta_data.json"

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
		return "", fmt.Errorf("Cannot parse %s: %v", metaDataPath, err)
	}

	return metaData.UUID, nil
}

func readInstanceID() (string, error) {
	file, err := os.Open(metaDataPath)
	if err != nil {
		return "", fmt.Errorf("Cannot open %s: %v", metaDataPath, err)
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

func (i *Instances) List(name_filter string) ([]types.NodeName, error) {
	glog.V(2).Infof("rackspace List(%v) called", name_filter)

	opts := osservers.ListOpts{
		Name:   name_filter,
		Status: "ACTIVE",
	}
	pager := servers.List(i.compute, opts)

	ret := make([]types.NodeName, 0)
	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		sList, err := servers.ExtractServers(page)
		if err != nil {
			return false, err
		}
		for i := range sList {
			ret = append(ret, mapServerToNodeName(&sList[i]))
		}
		return true, nil
	})
	if err != nil {
		return nil, err
	}

	glog.V(2).Infof("Found %v entries: %v", len(ret), ret)

	return ret, nil
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

func (i *Instances) NodeAddresses(nodeName types.NodeName) ([]api.NodeAddress, error) {
	glog.V(2).Infof("NodeAddresses(%v) called", nodeName)
	serverName := mapNodeNameToServerName(nodeName)
	ip, err := probeNodeAddress(i.compute, serverName)
	if err != nil {
		return nil, err
	}

	glog.V(2).Infof("NodeAddresses(%v) => %v", serverName, ip)

	// net.ParseIP().String() is to maintain compatibility with the old code
	parsedIP := net.ParseIP(ip).String()
	return []api.NodeAddress{
		{Type: api.NodeLegacyHostIP, Address: parsedIP},
		{Type: api.NodeInternalIP, Address: parsedIP},
		{Type: api.NodeExternalIP, Address: parsedIP},
	}, nil
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
	return "", nil
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
func (rs *Rackspace) CreateVolume(name string, size int, vtype, availability string, tags *map[string]string) (volumeName string, err error) {
	return "", errors.New("unimplemented")
}

func (rs *Rackspace) DeleteVolume(volumeName string) error {
	return errors.New("unimplemented")
}

// Attaches given cinder volume to the compute running kubelet
func (rs *Rackspace) AttachDisk(instanceID string, diskName string) (string, error) {
	disk, err := rs.getVolume(diskName)
	if err != nil {
		return "", err
	}

	compute, err := rs.getComputeClient()
	if err != nil {
		return "", err
	}

	if len(disk.Attachments) > 0 {
		if instanceID == disk.Attachments[0]["server_id"] {
			glog.V(4).Infof("Disk: %q is already attached to compute: %q", diskName, instanceID)
			return disk.ID, nil
		}

		errMsg := fmt.Sprintf("Disk %q is attached to a different compute: %q, should be detached before proceeding", diskName, disk.Attachments[0]["server_id"])
		glog.Errorf(errMsg)
		return "", errors.New(errMsg)
	}

	_, err = volumeattach.Create(compute, instanceID, &osvolumeattach.CreateOpts{
		VolumeID: disk.ID,
	}).Extract()
	if err != nil {
		glog.Errorf("Failed to attach %s volume to %s compute", diskName, instanceID)
		return "", err
	}
	glog.V(2).Infof("Successfully attached %s volume to %s compute", diskName, instanceID)
	return disk.ID, nil
}

// GetDevicePath returns the path of an attached block storage volume, specified by its id.
func (rs *Rackspace) GetDevicePath(diskId string) string {
	volume, err := rs.getVolume(diskId)
	if err != nil {
		return ""
	}
	attachments := volume.Attachments
	if len(attachments) != 1 {
		glog.Warningf("Unexpected number of volume attachments on %s: %d", diskId, len(attachments))
		return ""
	}
	return attachments[0]["device"].(string)
}

// Takes a partial/full disk id or diskname
func (rs *Rackspace) getVolume(diskName string) (volumes.Volume, error) {
	sClient, err := rackspace.NewBlockStorageV1(rs.provider, gophercloud.EndpointOpts{
		Region: rs.region,
	})

	var volume volumes.Volume
	if err != nil || sClient == nil {
		glog.Errorf("Unable to initialize cinder client for region: %s", rs.region)
		return volume, err
	}

	err = volumes.List(sClient).EachPage(func(page pagination.Page) (bool, error) {
		vols, err := volumes.ExtractVolumes(page)
		if err != nil {
			glog.Errorf("Failed to extract volumes: %v", err)
			return false, err
		}

		for _, v := range vols {
			glog.V(4).Infof("%s %s %v", v.ID, v.Name, v.Attachments)
			if v.Name == diskName || strings.Contains(v.ID, diskName) {
				volume = v
				return true, nil
			}
		}

		// if it reached here then no disk with the given name was found.
		errmsg := fmt.Sprintf("Unable to find disk: %s in region %s", diskName, rs.region)
		return false, errors.New(errmsg)
	})
	if err != nil {
		glog.Errorf("Error occurred getting volume: %s", diskName)
	}
	return volume, err
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
func (rs *Rackspace) DetachDisk(instanceID string, partialDiskId string) error {
	disk, err := rs.getVolume(partialDiskId)
	if err != nil {
		return err
	}

	compute, err := rs.getComputeClient()
	if err != nil {
		return err
	}

	if len(disk.Attachments) > 1 {
		// Rackspace does not support "multiattach", this is a sanity check.
		errmsg := fmt.Sprintf("Volume %s is attached to multiple instances, which is not supported by this provider.", disk.ID)
		return errors.New(errmsg)
	}

	if len(disk.Attachments) > 0 && instanceID == disk.Attachments[0]["server_id"] {
		// This is a blocking call and effects kubelet's performance directly.
		// We should consider kicking it out into a separate routine, if it is bad.
		err = volumeattach.Delete(compute, instanceID, disk.ID).ExtractErr()
		if err != nil {
			glog.Errorf("Failed to delete volume %s from compute %s attached %v", disk.ID, instanceID, err)
			return err
		}
		glog.V(2).Infof("Successfully detached volume: %s from compute: %s", disk.ID, instanceID)
	} else {
		errMsg := fmt.Sprintf("Disk: %s has no attachments or is not attached to compute: %s", disk.Name, instanceID)
		glog.Errorf(errMsg)
		return errors.New(errMsg)
	}

	return nil
}

// Get device path of attached volume to the compute running kubelet, as known by cinder
func (rs *Rackspace) GetAttachmentDiskPath(instanceID string, diskName string) (string, error) {
	// See issue #33128 - Cinder does not always tell you the right device path, as such
	// we must only use this value as a last resort.
	disk, err := rs.getVolume(diskName)
	if err != nil {
		return "", err
	}
	if len(disk.Attachments) > 0 && disk.Attachments[0]["server_id"] != nil {
		if instanceID == disk.Attachments[0]["server_id"] {
			// Attachment[0]["device"] points to the device path
			// see http://developer.openstack.org/api-ref-blockstorage-v1.html
			return disk.Attachments[0]["device"].(string), nil
		} else {
			errMsg := fmt.Sprintf("Disk %q is attached to a different compute: %q, should be detached before proceeding", diskName, disk.Attachments[0]["server_id"])
			glog.Errorf(errMsg)
			return "", errors.New(errMsg)
		}
	}
	return "", fmt.Errorf("volume %s is not attached to %s", diskName, instanceID)
}

// query if a volume is attached to a compute instance
func (rs *Rackspace) DiskIsAttached(diskName, instanceID string) (bool, error) {
	disk, err := rs.getVolume(diskName)
	if err != nil {
		return false, err
	}
	if len(disk.Attachments) > 0 && disk.Attachments[0]["server_id"] != nil && instanceID == disk.Attachments[0]["server_id"] {
		return true, nil
	}
	return false, nil
}

// query if a list volumes are attached to a compute instance
func (rs *Rackspace) DisksAreAttached(diskNames []string, instanceID string) (map[string]bool, error) {
	attached := make(map[string]bool)
	for _, diskName := range diskNames {
		attached[diskName] = false
	}
	var returnedErr error
	for _, diskName := range diskNames {
		result, err := rs.DiskIsAttached(diskName, instanceID)
		if err != nil {
			returnedErr = fmt.Errorf("Error in checking disk %q attached: %v \n %v", diskName, err, returnedErr)
			continue
		}
		if result {
			attached[diskName] = true
		}

	}
	return attached, returnedErr
}

// query if we should trust the cinder provide deviceName, See issue #33128
func (rs *Rackspace) ShouldTrustDevicePath() bool {
	return true
}
