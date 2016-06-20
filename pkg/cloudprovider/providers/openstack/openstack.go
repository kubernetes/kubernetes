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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"path"
	"regexp"
	"strings"
	"time"

	"gopkg.in/gcfg.v1"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/openstack"
	"github.com/rackspace/gophercloud/openstack/blockstorage/v1/volumes"
	"github.com/rackspace/gophercloud/openstack/compute/v2/extensions/volumeattach"
	"github.com/rackspace/gophercloud/openstack/compute/v2/flavors"
	"github.com/rackspace/gophercloud/openstack/compute/v2/servers"
	"github.com/rackspace/gophercloud/pagination"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

const ProviderName = "openstack"

// metadataUrl is URL to OpenStack metadata server. It's hadrcoded IPv4
// link-local address as documented in "OpenStack Cloud Administrator Guide",
// chapter Compute - Networking with nova-network.
// http://docs.openstack.org/admin-guide-cloud/compute-networking-nova.html#metadata-service
const metadataUrl = "http://169.254.169.254/openstack/2012-08-10/meta_data.json"

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
	LBVersion         string     `gcfg:"lb-version"` // v1 or v2
	SubnetId          string     `gcfg:"subnet-id"`  // required
	FloatingNetworkId string     `gcfg:"floating-network-id"`
	LBMethod          string     `gcfg:"lb-method"`
	CreateMonitor     bool       `gcfg:"create-monitor"`
	MonitorDelay      MyDuration `gcfg:"monitor-delay"`
	MonitorTimeout    MyDuration `gcfg:"monitor-timeout"`
	MonitorMaxRetries uint       `gcfg:"monitor-max-retries"`
}

// OpenStack is an implementation of cloud provider Interface for OpenStack.
type OpenStack struct {
	provider *gophercloud.ProviderClient
	region   string
	lbOpts   LoadBalancerOpts
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
		DomainId   string `gcfg:"domain-id"`
		DomainName string `gcfg:"domain-name"`
		Region     string
	}
	LoadBalancer LoadBalancerOpts
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
	err := gcfg.ReadInto(&cfg, config)
	return cfg, err
}

// parseMetadataUUID reads JSON from OpenStack metadata server and parses
// instance ID out of it.
func parseMetadataUUID(jsonData []byte) (string, error) {
	// We should receive an object with { 'uuid': '<uuid>' } and couple of other
	// properties (which we ignore).

	obj := struct{ UUID string }{}
	err := json.Unmarshal(jsonData, &obj)
	if err != nil {
		return "", err
	}

	uuid := obj.UUID
	if uuid == "" {
		err = fmt.Errorf("cannot parse OpenStack metadata, got empty uuid")
		return "", err
	}

	return uuid, nil
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
		// Fall through with empty instanceID and try metadata server.
	}
	glog.V(5).Infof("Cannot read %s: '%v', trying metadata server", instanceIDFile, err)

	// Try to get JSON from metdata server.
	resp, err := http.Get(metadataUrl)
	if err != nil {
		glog.V(3).Infof("Cannot read %s: %v", metadataUrl, err)
		return "", err
	}

	if resp.StatusCode != 200 {
		err = fmt.Errorf("got unexpected status code when reading metadata from %s: %s", metadataUrl, resp.Status)
		glog.V(3).Infof("%v", err)
		return "", err
	}

	defer resp.Body.Close()
	bodyBytes, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		glog.V(3).Infof("Cannot get HTTP response body from %s: %v", metadataUrl, err)
		return "", err
	}
	instanceID, err := parseMetadataUUID(bodyBytes)
	if err != nil {
		glog.V(3).Infof("Cannot parse instance ID from metadata from %s: %v", metadataUrl, err)
		return "", err
	}

	glog.V(3).Infof("Got instance id from %s: %s", metadataUrl, instanceID)
	return instanceID, nil
}

func newOpenStack(cfg Config) (*OpenStack, error) {
	provider, err := openstack.AuthenticatedClient(cfg.toAuthOptions())
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
		localInstanceID: id,
	}

	return &os, nil
}

type Instances struct {
	compute            *gophercloud.ServiceClient
	flavor_to_resource map[string]*api.NodeResources // keyed by flavor id
}

// Instances returns an implementation of Instances for OpenStack.
func (os *OpenStack) Instances() (cloudprovider.Instances, bool) {
	glog.V(4).Info("openstack.Instances() called")

	compute, err := openstack.NewComputeV2(os.provider, gophercloud.EndpointOpts{
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
					api.ResourceCPU:            *resource.NewQuantity(int64(flavor.VCPUs), resource.DecimalSI),
					api.ResourceMemory:         *resource.NewQuantity(int64(flavor.RAM)*MiB, resource.BinarySI),
					"openstack.org/disk":       *resource.NewQuantity(int64(flavor.Disk)*GB, resource.DecimalSI),
					"openstack.org/rxTxFactor": *resource.NewMilliQuantity(int64(flavor.RxTxFactor)*1000, resource.DecimalSI),
					"openstack.org/swap":       *resource.NewQuantity(int64(flavor.Swap)*MiB, resource.BinarySI),
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

	glog.V(3).Infof("Found %v compute flavors", len(flavor_to_resource))
	glog.V(1).Info("Claiming to support Instances")

	return &Instances{compute, flavor_to_resource}, true
}

func (i *Instances) List(name_filter string) ([]string, error) {
	glog.V(4).Infof("openstack List(%v) called", name_filter)

	opts := servers.ListOpts{
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

	glog.V(3).Infof("Found %v instances matching %v: %v",
		len(ret), name_filter, ret)

	return ret, nil
}

func getServerByName(client *gophercloud.ServiceClient, name string) (*servers.Server, error) {
	opts := servers.ListOpts{
		Name:   fmt.Sprintf("^%s$", regexp.QuoteMeta(name)),
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

func getAddressesByName(client *gophercloud.ServiceClient, name string) ([]api.NodeAddress, error) {
	srv, err := getServerByName(client, name)
	if err != nil {
		return nil, err
	}

	addrs := []api.NodeAddress{}

	for network, netblob := range srv.Addresses {
		list, ok := netblob.([]interface{})
		if !ok {
			continue
		}

		for _, item := range list {
			var addressType api.NodeAddressType

			props, ok := item.(map[string]interface{})
			if !ok {
				continue
			}

			extIPType, ok := props["OS-EXT-IPS:type"]
			if (ok && extIPType == "floating") || (!ok && network == "public") {
				addressType = api.NodeExternalIP
			} else {
				addressType = api.NodeInternalIP
			}

			tmp, ok := props["addr"]
			if !ok {
				continue
			}
			addr, ok := tmp.(string)
			if !ok {
				continue
			}

			api.AddToNodeAddresses(&addrs,
				api.NodeAddress{
					Type:    addressType,
					Address: addr,
				},
			)
		}
	}

	// AccessIPs are usually duplicates of "public" addresses.
	if srv.AccessIPv4 != "" {
		api.AddToNodeAddresses(&addrs,
			api.NodeAddress{
				Type:    api.NodeExternalIP,
				Address: srv.AccessIPv4,
			},
		)
	}

	if srv.AccessIPv6 != "" {
		api.AddToNodeAddresses(&addrs,
			api.NodeAddress{
				Type:    api.NodeExternalIP,
				Address: srv.AccessIPv6,
			},
		)
	}

	return addrs, nil
}

func getAddressByName(client *gophercloud.ServiceClient, name string) (string, error) {
	addrs, err := getAddressesByName(client, name)
	if err != nil {
		return "", err
	} else if len(addrs) == 0 {
		return "", ErrNoAddressFound
	}

	for _, addr := range addrs {
		if addr.Type == api.NodeInternalIP {
			return addr.Address, nil
		}
	}

	return addrs[0].Address, nil
}

// Implementation of Instances.CurrentNodeName
func (i *Instances) CurrentNodeName(hostname string) (string, error) {
	return hostname, nil
}

func (i *Instances) AddSSHKeyToAllInstances(user string, keyData []byte) error {
	return errors.New("unimplemented")
}

func (i *Instances) NodeAddresses(name string) ([]api.NodeAddress, error) {
	glog.V(4).Infof("NodeAddresses(%v) called", name)

	addrs, err := getAddressesByName(i.compute, name)
	if err != nil {
		return nil, err
	}

	glog.V(4).Infof("NodeAddresses(%v) => %v", name, addrs)
	return addrs, nil
}

// ExternalID returns the cloud provider ID of the specified instance (deprecated).
func (i *Instances) ExternalID(name string) (string, error) {
	srv, err := getServerByName(i.compute, name)
	if err != nil {
		return "", err
	}
	return srv.ID, nil
}

// InstanceID returns the kubelet's cloud provider ID.
func (os *OpenStack) InstanceID() (string, error) {
	return os.localInstanceID, nil
}

// InstanceID returns the cloud provider ID of the specified instance.
func (i *Instances) InstanceID(name string) (string, error) {
	srv, err := getServerByName(i.compute, name)
	if err != nil {
		return "", err
	}
	// In the future it is possible to also return an endpoint as:
	// <endpoint>/<instanceid>
	return "/" + srv.ID, nil
}

// InstanceType returns the type of the specified instance.
func (i *Instances) InstanceType(name string) (string, error) {
	return "", nil
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
		glog.Warningf("Failed to find neutron endpoint: %v", err)
		return nil, false
	}

	compute, err := openstack.NewComputeV2(os.provider, gophercloud.EndpointOpts{
		Region: os.region,
	})
	if err != nil {
		glog.Warningf("Failed to find compute endpoint: %v", err)
		return nil, false
	}

	glog.V(1).Info("Claiming to support LoadBalancer")

	if os.lbOpts.LBVersion == "v2" {
		return &LbaasV2{LoadBalancer{network, compute, os.lbOpts}}, true
	} else {

		return &LbaasV1{LoadBalancer{network, compute, os.lbOpts}}, true
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
	glog.V(1).Infof("Current zone is %v", os.region)

	return cloudprovider.Zone{Region: os.region}, nil
}

func (os *OpenStack) Routes() (cloudprovider.Routes, bool) {
	return nil, false
}

// Attaches given cinder volume to the compute running kubelet
func (os *OpenStack) AttachDisk(instanceID string, diskName string) (string, error) {
	disk, err := os.getVolume(diskName)
	if err != nil {
		return "", err
	}
	cClient, err := openstack.NewComputeV2(os.provider, gophercloud.EndpointOpts{
		Region: os.region,
	})
	if err != nil || cClient == nil {
		glog.Errorf("Unable to initialize nova client for region: %s", os.region)
		return "", err
	}

	if len(disk.Attachments) > 0 && disk.Attachments[0]["server_id"] != nil {
		if instanceID == disk.Attachments[0]["server_id"] {
			glog.V(4).Infof("Disk: %q is already attached to compute: %q", diskName, instanceID)
			return disk.ID, nil
		} else {
			errMsg := fmt.Sprintf("Disk %q is attached to a different compute: %q, should be detached before proceeding", diskName, disk.Attachments[0]["server_id"])
			glog.Errorf(errMsg)
			return "", errors.New(errMsg)
		}
	}
	// add read only flag here if possible spothanis
	_, err = volumeattach.Create(cClient, instanceID, &volumeattach.CreateOpts{
		VolumeID: disk.ID,
	}).Extract()
	if err != nil {
		glog.Errorf("Failed to attach %s volume to %s compute", diskName, instanceID)
		return "", err
	}
	glog.V(2).Infof("Successfully attached %s volume to %s compute", diskName, instanceID)
	return disk.ID, nil
}

// Detaches given cinder volume from the compute running kubelet
func (os *OpenStack) DetachDisk(instanceID string, partialDiskId string) error {
	disk, err := os.getVolume(partialDiskId)
	if err != nil {
		return err
	}
	cClient, err := openstack.NewComputeV2(os.provider, gophercloud.EndpointOpts{
		Region: os.region,
	})
	if err != nil || cClient == nil {
		glog.Errorf("Unable to initialize nova client for region: %s", os.region)
		return err
	}
	if len(disk.Attachments) > 0 && disk.Attachments[0]["server_id"] != nil && instanceID == disk.Attachments[0]["server_id"] {
		// This is a blocking call and effects kubelet's performance directly.
		// We should consider kicking it out into a separate routine, if it is bad.
		err = volumeattach.Delete(cClient, instanceID, disk.ID).ExtractErr()
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

// Takes a partial/full disk id or diskname
func (os *OpenStack) getVolume(diskName string) (volumes.Volume, error) {
	sClient, err := openstack.NewBlockStorageV1(os.provider, gophercloud.EndpointOpts{
		Region: os.region,
	})

	var volume volumes.Volume
	if err != nil || sClient == nil {
		glog.Errorf("Unable to initialize cinder client for region: %s", os.region)
		return volume, err
	}

	err = volumes.List(sClient, nil).EachPage(func(page pagination.Page) (bool, error) {
		vols, err := volumes.ExtractVolumes(page)
		if err != nil {
			glog.Errorf("Failed to extract volumes: %v", err)
			return false, err
		} else {
			for _, v := range vols {
				glog.V(4).Infof("%s %s %v", v.ID, v.Name, v.Attachments)
				if v.Name == diskName || strings.Contains(v.ID, diskName) {
					volume = v
					return true, nil
				}
			}
		}
		// if it reached here then no disk with the given name was found.
		errmsg := fmt.Sprintf("Unable to find disk: %s in region %s", diskName, os.region)
		return false, errors.New(errmsg)
	})
	if err != nil {
		glog.Errorf("Error occured getting volume: %s", diskName)
		return volume, err
	}
	return volume, err
}

// Create a volume of given size (in GiB)
func (os *OpenStack) CreateVolume(name string, size int, tags *map[string]string) (volumeName string, err error) {

	sClient, err := openstack.NewBlockStorageV1(os.provider, gophercloud.EndpointOpts{
		Region: os.region,
	})

	if err != nil || sClient == nil {
		glog.Errorf("Unable to initialize cinder client for region: %s", os.region)
		return "", err
	}

	opts := volumes.CreateOpts{
		Name: name,
		Size: size,
	}
	if tags != nil {
		opts.Metadata = *tags
	}
	vol, err := volumes.Create(sClient, opts).Extract()
	if err != nil {
		glog.Errorf("Failed to create a %d GB volume: %v", size, err)
		return "", err
	}
	glog.Infof("Created volume %v", vol.ID)
	return vol.ID, err
}

// GetDevicePath returns the path of an attached block storage volume, specified by its id.
func (os *OpenStack) GetDevicePath(diskId string) string {
	files, _ := ioutil.ReadDir("/dev/disk/by-id/")
	for _, f := range files {
		if strings.Contains(f.Name(), "virtio-") {
			devid_prefix := f.Name()[len("virtio-"):len(f.Name())]
			if strings.Contains(diskId, devid_prefix) {
				glog.V(4).Infof("Found disk attached as %q; full devicepath: %s\n", f.Name(), path.Join("/dev/disk/by-id/", f.Name()))
				return path.Join("/dev/disk/by-id/", f.Name())
			}
		}
	}
	glog.Warningf("Failed to find device for the diskid: %q\n", diskId)
	return ""
}

func (os *OpenStack) DeleteVolume(volumeName string) error {
	sClient, err := openstack.NewBlockStorageV1(os.provider, gophercloud.EndpointOpts{
		Region: os.region,
	})

	if err != nil || sClient == nil {
		glog.Errorf("Unable to initialize cinder client for region: %s", os.region)
		return err
	}
	err = volumes.Delete(sClient, volumeName).ExtractErr()
	if err != nil {
		glog.Errorf("Cannot delete volume %s: %v", volumeName, err)
	}
	return err
}

// Get device path of attached volume to the compute running kubelet
func (os *OpenStack) GetAttachmentDiskPath(instanceID string, diskName string) (string, error) {
	disk, err := os.getVolume(diskName)
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
func (os *OpenStack) DiskIsAttached(diskName, instanceID string) (bool, error) {
	disk, err := os.getVolume(diskName)
	if err != nil {
		return false, err
	}
	if len(disk.Attachments) > 0 && disk.Attachments[0]["server_id"] != nil && instanceID == disk.Attachments[0]["server_id"] {
		return true, nil
	}
	return false, nil
}
