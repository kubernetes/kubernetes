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

package aws_cloud

import (
	"errors"
	"fmt"
	"io"
	"net"
	"net/url"
	"regexp"
	"strings"
	"sync"
	"time"

	"code.google.com/p/gcfg"
	"github.com/mitchellh/goamz/aws"
	"github.com/mitchellh/goamz/ec2"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"

	"github.com/golang/glog"
)

// Abstraction over EC2, to allow mocking/other implementations
type EC2 interface {
	// Query EC2 for instances matching the filter
	Instances(instIds []string, filter *ec2InstanceFilter) (resp *ec2.InstancesResp, err error)

	// Attach a volume to an instance
	AttachVolume(volumeID string, instanceId string, mountDevice string) (resp *ec2.AttachVolumeResp, err error)
	// Detach a volume from whatever instance it is attached to
	// TODO: We should specify the InstanceID and the Device, for safety
	DetachVolume(volumeID string) (resp *ec2.SimpleResp, err error)
	// Lists volumes
	Volumes(volumeIDs []string, filter *ec2.Filter) (resp *ec2.VolumesResp, err error)
	// Create an EBS volume
	CreateVolume(request *ec2.CreateVolume) (resp *ec2.CreateVolumeResp, err error)
	// Delete an EBS volume
	DeleteVolume(volumeID string) (resp *ec2.SimpleResp, err error)
}

// Abstraction over the AWS metadata service
type AWSMetadata interface {
	// Query the EC2 metadata service (used to discover instance-id etc)
	GetMetaData(key string) ([]byte, error)
}

type VolumeOptions struct {
	CapacityMB int
}

// Volumes is an interface for managing cloud-provisioned volumes
type Volumes interface {
	// Attach the disk to the specified instance
	// instanceName can be empty to mean "the instance on which we are running"
	// Returns the device (e.g. /dev/xvdf) where we attached the volume
	AttachDisk(instanceName string, volumeName string, readOnly bool) (string, error)
	// Detach the disk from the specified instance
	// instanceName can be empty to mean "the instance on which we are running"
	DetachDisk(instanceName string, volumeName string) error

	// Create a volume with the specified options
	CreateVolume(volumeOptions *VolumeOptions) (volumeName string, err error)
	DeleteVolume(volumeName string) error
}

// AWSCloud is an implementation of Interface, TCPLoadBalancer and Instances for Amazon Web Services.
type AWSCloud struct {
	ec2              EC2
	metadata         AWSMetadata
	cfg              *AWSCloudConfig
	availabilityZone string
	region           aws.Region

	// The AWS instance that we are running on
	selfAWSInstance *awsInstance

	mutex sync.Mutex
}

type AWSCloudConfig struct {
	Global struct {
		// TODO: Is there any use for this?  We can get it from the instance metadata service
		Zone string
	}
}

// Similar to ec2.Filter, but the filter values can be read from tests
// (ec2.Filter only has private members)
type ec2InstanceFilter struct {
	PrivateDNSName string
}

// True if the passed instance matches the filter
func (f *ec2InstanceFilter) Matches(instance ec2.Instance) bool {
	if f.PrivateDNSName != "" && instance.PrivateDNSName != f.PrivateDNSName {
		return false
	}
	return true
}

// goamzEC2 is an implementation of the EC2 interface, backed by goamz
type goamzEC2 struct {
	ec2 *ec2.EC2
}

// Implementation of EC2.Instances
func (self *goamzEC2) Instances(instanceIds []string, filter *ec2InstanceFilter) (resp *ec2.InstancesResp, err error) {
	var goamzFilter *ec2.Filter
	if filter != nil {
		goamzFilter = ec2.NewFilter()
		if filter.PrivateDNSName != "" {
			goamzFilter.Add("private-dns-name", filter.PrivateDNSName)
		}
	}
	return self.ec2.Instances(instanceIds, goamzFilter)
}

type goamzMetadata struct {
}

// Implements AWSMetadata.GetMetaData
func (self *goamzMetadata) GetMetaData(key string) ([]byte, error) {
	v, err := aws.GetMetaData(key)
	if err != nil {
		return nil, fmt.Errorf("Error querying AWS metadata for key %s: %v", key, err)
	}
	return v, nil
}

type AuthFunc func() (auth aws.Auth, err error)

func (s *goamzEC2) AttachVolume(volumeID string, instanceId string, device string) (resp *ec2.AttachVolumeResp, err error) {
	return s.ec2.AttachVolume(volumeID, instanceId, device)
}

func (s *goamzEC2) DetachVolume(volumeID string) (resp *ec2.SimpleResp, err error) {
	return s.ec2.DetachVolume(volumeID)
}

func (s *goamzEC2) Volumes(volumeIDs []string, filter *ec2.Filter) (resp *ec2.VolumesResp, err error) {
	return s.ec2.Volumes(volumeIDs, filter)
}

func (s *goamzEC2) CreateVolume(request *ec2.CreateVolume) (resp *ec2.CreateVolumeResp, err error) {
	return s.ec2.CreateVolume(request)
}

func (s *goamzEC2) DeleteVolume(volumeID string) (resp *ec2.SimpleResp, err error) {
	return s.ec2.DeleteVolume(volumeID)
}

func init() {
	cloudprovider.RegisterCloudProvider("aws", func(config io.Reader) (cloudprovider.Interface, error) {
		metadata := &goamzMetadata{}
		return newAWSCloud(config, getAuth, metadata)
	})
}

func getAuth() (auth aws.Auth, err error) {
	return aws.GetAuth("", "")
}

// readAWSCloudConfig reads an instance of AWSCloudConfig from config reader.
func readAWSCloudConfig(config io.Reader, metadata AWSMetadata) (*AWSCloudConfig, error) {
	var cfg AWSCloudConfig
	var err error

	if config != nil {
		err = gcfg.ReadInto(&cfg, config)
		if err != nil {
			return nil, err
		}
	}

	if cfg.Global.Zone == "" {
		if metadata != nil {
			glog.Info("Zone not specified in configuration file; querying AWS metadata service")
			cfg.Global.Zone, err = getAvailabilityZone(metadata)
			if err != nil {
				return nil, err
			}
		}
		if cfg.Global.Zone == "" {
			return nil, fmt.Errorf("no zone specified in configuration file")
		}
	}

	return &cfg, nil
}

func getAvailabilityZone(metadata AWSMetadata) (string, error) {
	availabilityZoneBytes, err := metadata.GetMetaData("placement/availability-zone")
	if err != nil {
		return "", err
	}
	if availabilityZoneBytes == nil || len(availabilityZoneBytes) == 0 {
		return "", fmt.Errorf("Unable to determine availability-zone from instance metadata")
	}
	return string(availabilityZoneBytes), nil
}

// newAWSCloud creates a new instance of AWSCloud.
// authFunc and instanceId are primarily for tests
func newAWSCloud(config io.Reader, authFunc AuthFunc, metadata AWSMetadata) (*AWSCloud, error) {
	cfg, err := readAWSCloudConfig(config, metadata)
	if err != nil {
		return nil, fmt.Errorf("unable to read AWS cloud provider config file: %v", err)
	}

	auth, err := authFunc()
	if err != nil {
		return nil, err
	}

	zone := cfg.Global.Zone
	if len(zone) <= 1 {
		return nil, fmt.Errorf("invalid AWS zone in config file: %s", zone)
	}
	regionName := zone[:len(zone)-1]

	region, ok := aws.Regions[regionName]
	if !ok {
		return nil, fmt.Errorf("not a valid AWS zone (unknown region): %s", zone)
	}

	ec2 := &goamzEC2{ec2: ec2.New(auth, region)}

	awsCloud := &AWSCloud{
		ec2:              ec2,
		cfg:              cfg,
		region:           region,
		availabilityZone: zone,
		metadata:         metadata,
	}

	return awsCloud, nil
}

func (aws *AWSCloud) Clusters() (cloudprovider.Clusters, bool) {
	return nil, false
}

// TCPLoadBalancer returns an implementation of TCPLoadBalancer for Amazon Web Services.
func (aws *AWSCloud) TCPLoadBalancer() (cloudprovider.TCPLoadBalancer, bool) {
	return nil, false
}

// Instances returns an implementation of Instances for Amazon Web Services.
func (aws *AWSCloud) Instances() (cloudprovider.Instances, bool) {
	return aws, true
}

// Zones returns an implementation of Zones for Amazon Web Services.
func (aws *AWSCloud) Zones() (cloudprovider.Zones, bool) {
	return aws, true
}

// NodeAddresses is an implementation of Instances.NodeAddresses.
func (aws *AWSCloud) NodeAddresses(name string) ([]api.NodeAddress, error) {
	instance, err := aws.getInstancesByDnsName(name)
	if err != nil {
		return nil, err
	}

	addresses := []api.NodeAddress{}

	if instance.PrivateIpAddress != "" {
		ipAddress := instance.PrivateIpAddress
		ip := net.ParseIP(ipAddress)
		if ip == nil {
			return nil, fmt.Errorf("EC2 instance had invalid private address: %s (%s)", instance.InstanceId, ipAddress)
		}
		addresses = append(addresses, api.NodeAddress{Type: api.NodeInternalIP, Address: ip.String()})

		// Legacy compatibility: the private ip was the legacy host ip
		addresses = append(addresses, api.NodeAddress{Type: api.NodeLegacyHostIP, Address: ip.String()})
	}

	// TODO: Other IP addresses (multiple ips)?
	if instance.PublicIpAddress != "" {
		ipAddress := instance.PublicIpAddress
		ip := net.ParseIP(ipAddress)
		if ip == nil {
			return nil, fmt.Errorf("EC2 instance had invalid public address: %s (%s)", instance.InstanceId, ipAddress)
		}
		addresses = append(addresses, api.NodeAddress{Type: api.NodeExternalIP, Address: ip.String()})
	}

	return addresses, nil
}

// ExternalID returns the cloud provider ID of the specified instance.
func (aws *AWSCloud) ExternalID(name string) (string, error) {
	inst, err := aws.getInstancesByDnsName(name)
	if err != nil {
		return "", err
	}
	return inst.InstanceId, nil
}

// Return the instances matching the relevant private dns name.
func (aws *AWSCloud) getInstancesByDnsName(name string) (*ec2.Instance, error) {
	f := &ec2InstanceFilter{}
	f.PrivateDNSName = name

	resp, err := aws.ec2.Instances(nil, f)
	if err != nil {
		return nil, err
	}

	instances := []*ec2.Instance{}
	for _, reservation := range resp.Reservations {
		for _, instance := range reservation.Instances {
			// TODO: Push running logic down into filter?
			if !isAlive(&instance) {
				continue
			}

			if instance.PrivateDNSName != name {
				// TODO: Should we warn here? - the filter should have caught this
				// (this will happen in the tests if they don't fully mock the EC2 API)
				continue
			}

			instances = append(instances, &instance)
		}
	}

	if len(instances) == 0 {
		return nil, fmt.Errorf("no instances found for host: %s", name)
	}
	if len(instances) > 1 {
		return nil, fmt.Errorf("multiple instances found for host: %s", name)
	}
	return instances[0], nil
}

// Check if the instance is alive (running or pending)
// We typically ignore instances that are not alive
func isAlive(instance *ec2.Instance) bool {
	switch instance.State.Name {
	case "shutting-down", "terminated", "stopping", "stopped":
		return false
	case "pending", "running":
		return true
	default:
		glog.Errorf("unknown EC2 instance state: %s", instance.State)
		return false
	}
}

// Return a list of instances matching regex string.
func (aws *AWSCloud) getInstancesByRegex(regex string) ([]string, error) {
	resp, err := aws.ec2.Instances(nil, nil)
	if err != nil {
		return []string{}, err
	}
	if resp == nil {
		return []string{}, fmt.Errorf("no InstanceResp returned")
	}

	if strings.HasPrefix(regex, "'") && strings.HasSuffix(regex, "'") {
		glog.Infof("Stripping quotes around regex (%s)", regex)
		regex = regex[1 : len(regex)-1]
	}

	re, err := regexp.Compile(regex)
	if err != nil {
		return []string{}, err
	}

	instances := []string{}
	for _, reservation := range resp.Reservations {
		for _, instance := range reservation.Instances {
			// TODO: Push filtering down into EC2 API filter?
			if !isAlive(&instance) {
				glog.V(2).Infof("skipping EC2 instance (%s): %s",
					instance.State.Name, instance.InstanceId)
				continue
			}

			// Only return fully-ready instances when listing instances
			// (vs a query by name, where we will return it if we find it)
			if instance.State.Name == "pending" {
				glog.V(2).Infof("skipping EC2 instance (pending): %s", instance.InstanceId)
				continue
			}
			if instance.PrivateDNSName == "" {
				glog.V(2).Infof("skipping EC2 instance (no PrivateDNSName): %s",
					instance.InstanceId)
				continue
			}

			for _, tag := range instance.Tags {
				if tag.Key == "Name" && re.MatchString(tag.Value) {
					instances = append(instances, instance.PrivateDNSName)
					break
				}
			}
		}
	}
	glog.V(2).Infof("Matched EC2 instances: %s", instances)
	return instances, nil
}

// List is an implementation of Instances.List.
func (aws *AWSCloud) List(filter string) ([]string, error) {
	// TODO: Should really use tag query. No need to go regexp.
	return aws.getInstancesByRegex(filter)
}

// GetNodeResources implements Instances.GetNodeResources
func (aws *AWSCloud) GetNodeResources(name string) (*api.NodeResources, error) {
	instance, err := aws.getInstancesByDnsName(name)
	if err != nil {
		return nil, err
	}

	resources, err := getResourcesByInstanceType(instance.InstanceType)
	if err != nil {
		return nil, err
	}

	return resources, nil
}

// Builds an api.NodeResources
// cpu is in ecus, memory is in GiB
// We pass the family in so that we could provide more info (e.g. GPU or not)
func makeNodeResources(family string, cpu float64, memory float64) (*api.NodeResources, error) {
	return &api.NodeResources{
		Capacity: api.ResourceList{
			api.ResourceCPU:    *resource.NewMilliQuantity(int64(cpu*1000), resource.DecimalSI),
			api.ResourceMemory: *resource.NewQuantity(int64(memory*1024*1024*1024), resource.BinarySI),
		},
	}, nil
}

// Maps an EC2 instance type to k8s resource information
func getResourcesByInstanceType(instanceType string) (*api.NodeResources, error) {
	// There is no API for this (that I know of)
	switch instanceType {
	// t2: Burstable
	// TODO: The ECUs are fake values (because they are burstable), so this is just a guess...
	case "t1.micro":
		return makeNodeResources("t1", 0.125, 0.615)

		// t2: Burstable
		// TODO: The ECUs are fake values (because they are burstable), so this is just a guess...
	case "t2.micro":
		return makeNodeResources("t2", 0.25, 1)
	case "t2.small":
		return makeNodeResources("t2", 0.5, 2)
	case "t2.medium":
		return makeNodeResources("t2", 1, 4)

		// c1: Compute optimized
	case "c1.medium":
		return makeNodeResources("c1", 5, 1.7)
	case "c1.xlarge":
		return makeNodeResources("c1", 20, 7)

		// cc2: Compute optimized
	case "cc2.8xlarge":
		return makeNodeResources("cc2", 88, 60.5)

		// cg1: GPU instances
	case "cg1.4xlarge":
		return makeNodeResources("cg1", 33.5, 22.5)

		// cr1: Memory optimized
	case "cr1.8xlarge":
		return makeNodeResources("cr1", 88, 244)

		// c3: Compute optimized
	case "c3.large":
		return makeNodeResources("c3", 7, 3.75)
	case "c3.xlarge":
		return makeNodeResources("c3", 14, 7.5)
	case "c3.2xlarge":
		return makeNodeResources("c3", 28, 15)
	case "c3.4xlarge":
		return makeNodeResources("c3", 55, 30)
	case "c3.8xlarge":
		return makeNodeResources("c3", 108, 60)

		// c4: Compute optimized
	case "c4.large":
		return makeNodeResources("c4", 8, 3.75)
	case "c4.xlarge":
		return makeNodeResources("c4", 16, 7.5)
	case "c4.2xlarge":
		return makeNodeResources("c4", 31, 15)
	case "c4.4xlarge":
		return makeNodeResources("c4", 62, 30)
	case "c4.8xlarge":
		return makeNodeResources("c4", 132, 60)

		// g2: GPU instances
	case "g2.2xlarge":
		return makeNodeResources("g2", 26, 15)

		// hi1: Storage optimized (SSD)
	case "hi1.4xlarge":
		return makeNodeResources("hs1", 35, 60.5)

		// hs1: Storage optimized (HDD)
	case "hs1.8xlarge":
		return makeNodeResources("hs1", 35, 117)

		// m1: General purpose
	case "m1.small":
		return makeNodeResources("m1", 1, 1.7)
	case "m1.medium":
		return makeNodeResources("m1", 2, 3.75)
	case "m1.large":
		return makeNodeResources("m1", 4, 7.5)
	case "m1.xlarge":
		return makeNodeResources("m1", 8, 15)

		// m2: Memory optimized
	case "m2.xlarge":
		return makeNodeResources("m2", 6.5, 17.1)
	case "m2.2xlarge":
		return makeNodeResources("m2", 13, 34.2)
	case "m2.4xlarge":
		return makeNodeResources("m2", 26, 68.4)

		// m3: General purpose
	case "m3.medium":
		return makeNodeResources("m3", 3, 3.75)
	case "m3.large":
		return makeNodeResources("m3", 6.5, 7.5)
	case "m3.xlarge":
		return makeNodeResources("m3", 13, 15)
	case "m3.2xlarge":
		return makeNodeResources("m3", 26, 30)

		// i2: Storage optimized (SSD)
	case "i2.xlarge":
		return makeNodeResources("i2", 14, 30.5)
	case "i2.2xlarge":
		return makeNodeResources("i2", 27, 61)
	case "i2.4xlarge":
		return makeNodeResources("i2", 53, 122)
	case "i2.8xlarge":
		return makeNodeResources("i2", 104, 244)

		// r3: Memory optimized
	case "r3.large":
		return makeNodeResources("r3", 6.5, 15)
	case "r3.xlarge":
		return makeNodeResources("r3", 13, 30.5)
	case "r3.2xlarge":
		return makeNodeResources("r3", 26, 61)
	case "r3.4xlarge":
		return makeNodeResources("r3", 52, 122)
	case "r3.8xlarge":
		return makeNodeResources("r3", 104, 244)

	default:
		glog.Errorf("unknown instanceType: %s", instanceType)
		return nil, nil
	}
}

// GetZone implements Zones.GetZone
func (self *AWSCloud) GetZone() (cloudprovider.Zone, error) {
	if self.availabilityZone == "" {
		// Should be unreachable
		panic("availabilityZone not set")
	}
	return cloudprovider.Zone{
		FailureDomain: self.availabilityZone,
		Region:        self.region.Name,
	}, nil
}

// Abstraction around AWS Instance Types
// There isn't an API to get information for a particular instance type (that I know of)
type awsInstanceType struct {
}

// TODO: Also return number of mounts allowed?
func (self *awsInstanceType) getEBSMountDevices() []string {
	// See: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/block-device-mapping-concepts.html
	devices := []string{}
	for c := 'f'; c <= 'p'; c++ {
		devices = append(devices, fmt.Sprintf("/dev/sd%c", c))
	}
	return devices
}

type awsInstance struct {
	ec2 EC2

	// id in AWS
	awsID string

	mutex sync.Mutex

	// We must cache because otherwise there is a race condition,
	// where we assign a device mapping and then get a second request before we attach the volume
	deviceMappings map[string]string
}

func newAWSInstance(ec2 EC2, awsID string) *awsInstance {
	self := &awsInstance{ec2: ec2, awsID: awsID}

	// We lazy-init deviceMappings
	self.deviceMappings = nil

	return self
}

// Gets the awsInstanceType that models the instance type of this instance
func (self *awsInstance) getInstanceType() *awsInstanceType {
	// TODO: Make this real
	awsInstanceType := &awsInstanceType{}
	return awsInstanceType
}

// Gets the full information about this instance from the EC2 API
func (self *awsInstance) getInfo() (*ec2.Instance, error) {
	resp, err := self.ec2.Instances([]string{self.awsID}, nil)
	if err != nil {
		return nil, fmt.Errorf("error querying ec2 for instance info: %v", err)
	}
	if len(resp.Reservations) == 0 {
		return nil, fmt.Errorf("no reservations found for instance: %s", self.awsID)
	}
	if len(resp.Reservations) > 1 {
		return nil, fmt.Errorf("multiple reservations found for instance: %s", self.awsID)
	}
	if len(resp.Reservations[0].Instances) == 0 {
		return nil, fmt.Errorf("no instances found for instance: %s", self.awsID)
	}
	if len(resp.Reservations[0].Instances) > 1 {
		return nil, fmt.Errorf("multiple instances found for instance: %s", self.awsID)
	}
	return &resp.Reservations[0].Instances[0], nil
}

// Assigns an unused mount device for the specified volume.
// If the volume is already assigned, this will return the existing mount device and true
func (self *awsInstance) assignMountDevice(volumeID string) (mountDevice string, alreadyAttached bool, err error) {
	instanceType := self.getInstanceType()
	if instanceType == nil {
		return "", false, fmt.Errorf("could not get instance type for instance: %s", self.awsID)
	}

	// We lock to prevent concurrent mounts from conflicting
	// We may still conflict if someone calls the API concurrently,
	// but the AWS API will then fail one of the two attach operations
	self.mutex.Lock()
	defer self.mutex.Unlock()

	// We cache both for efficiency and correctness
	if self.deviceMappings == nil {
		info, err := self.getInfo()
		if err != nil {
			return "", false, err
		}
		deviceMappings := map[string]string{}
		for _, blockDevice := range info.BlockDevices {
			deviceMappings[blockDevice.DeviceName] = blockDevice.VolumeId
		}
		self.deviceMappings = deviceMappings
	}

	// Check to see if this volume is already assigned a device on this machine
	for deviceName, mappingVolumeID := range self.deviceMappings {
		if volumeID == mappingVolumeID {
			glog.Warningf("Got assignment call for already-assigned volume: %s@%s", deviceName, mappingVolumeID)
			return deviceName, true, nil
		}
	}

	// Check all the valid mountpoints to see if any of them are free
	valid := instanceType.getEBSMountDevices()
	chosen := ""
	for _, device := range valid {
		_, found := self.deviceMappings[device]
		if !found {
			chosen = device
			break
		}
	}

	if chosen == "" {
		glog.Warningf("Could not assign a mount device (all in use?).  mappings=%v, valid=%v", self.deviceMappings, valid)
		return "", false, nil
	}

	self.deviceMappings[chosen] = volumeID
	glog.V(2).Infof("Assigned mount device %s -> volume %s", chosen, volumeID)

	return chosen, false, nil
}

func (self *awsInstance) releaseMountDevice(volumeID string, mountDevice string) {
	self.mutex.Lock()
	defer self.mutex.Unlock()

	existingVolumeID, found := self.deviceMappings[mountDevice]
	if !found {
		glog.Errorf("releaseMountDevice on non-allocated device")
		return
	}
	if volumeID != existingVolumeID {
		glog.Errorf("releaseMountDevice on device assigned to different volume")
		return
	}
	glog.V(2).Infof("Releasing mount device mapping: %s -> volume %s", mountDevice, volumeID)
	delete(self.deviceMappings, mountDevice)
}

type awsDisk struct {
	ec2 EC2

	// Name in k8s
	name string
	// id in AWS
	awsID string
	// az which holds the volume
	az string
}

func newAWSDisk(ec2 EC2, name string) (*awsDisk, error) {
	// name looks like aws://availability-zone/id
	url, err := url.Parse(name)
	if err != nil {
		// TODO: Maybe we should pass a URL into the Volume functions
		return nil, fmt.Errorf("Invalid disk name (%s): %v", name, err)
	}
	if url.Scheme != "aws" {
		return nil, fmt.Errorf("Invalid scheme for AWS volume (%s)", name)
	}

	awsID := url.Path
	if len(awsID) > 1 && awsID[0] == '/' {
		awsID = awsID[1:]
	}

	// TODO: Regex match?
	if strings.Contains(awsID, "/") || !strings.HasPrefix(awsID, "vol-") {
		return nil, fmt.Errorf("Invalid format for AWS volume (%s)", name)
	}
	az := url.Host
	// TODO: Better validation?
	// TODO: Default to our AZ?  Look it up?
	// TODO: Should this be a region or an AZ?
	if az == "" {
		return nil, fmt.Errorf("Invalid format for AWS volume (%s)", name)
	}
	disk := &awsDisk{ec2: ec2, name: name, awsID: awsID, az: az}
	return disk, nil
}

// Gets the full information about this volume from the EC2 API
func (self *awsDisk) getInfo() (*ec2.Volume, error) {
	resp, err := self.ec2.Volumes([]string{self.awsID}, nil)
	if err != nil {
		return nil, fmt.Errorf("error querying ec2 for volume info: %v", err)
	}
	if len(resp.Volumes) == 0 {
		return nil, fmt.Errorf("no volumes found for volume: %s", self.awsID)
	}
	if len(resp.Volumes) > 1 {
		return nil, fmt.Errorf("multiple volumes found for volume: %s", self.awsID)
	}
	return &resp.Volumes[0], nil
}

func (self *awsDisk) waitForAttachmentStatus(status string) error {
	// TODO: There may be a faster way to get this when we're attaching locally
	attempt := 0
	maxAttempts := 60

	for {
		info, err := self.getInfo()
		if err != nil {
			return err
		}
		if len(info.Attachments) > 1 {
			glog.Warningf("Found multiple attachments for volume: %v", info)
		}
		attachmentStatus := ""
		for _, attachment := range info.Attachments {
			if attachmentStatus != "" {
				glog.Warning("Found multiple attachments: ", info)
			}
			attachmentStatus = attachment.Status
		}
		if attachmentStatus == "" {
			attachmentStatus = "detached"
		}
		if attachmentStatus == status {
			return nil
		}

		glog.V(2).Infof("Waiting for volume state: actual=%s, desired=%s", attachmentStatus, status)

		attempt++
		if attempt > maxAttempts {
			glog.Warningf("Timeout waiting for volume state: actual=%s, desired=%s", attachmentStatus, status)
			return errors.New("Timeout waiting for volume state")
		}

		time.Sleep(1 * time.Second)
	}
}

// Deletes the EBS disk
func (self *awsDisk) delete() error {
	_, err := self.ec2.DeleteVolume(self.awsID)
	if err != nil {
		return fmt.Errorf("error delete EBS volumes: %v", err)
	}
	return nil
}

// Gets the awsInstance for the EC2 instance on which we are running
// may return nil in case of error
func (aws *AWSCloud) getSelfAWSInstance() (*awsInstance, error) {
	// Note that we cache some state in awsInstance (mountpoints), so we must preserve the instance

	aws.mutex.Lock()
	defer aws.mutex.Unlock()

	i := aws.selfAWSInstance
	if i == nil {
		instanceIdBytes, err := aws.metadata.GetMetaData("instance-id")
		if err != nil {
			return nil, fmt.Errorf("error fetching instance-id from ec2 metadata service: %v", err)
		}
		i = newAWSInstance(aws.ec2, string(instanceIdBytes))
		aws.selfAWSInstance = i
	}

	return i, nil
}

// Implements Volumes.AttachDisk
func (aws *AWSCloud) AttachDisk(instanceName string, diskName string, readOnly bool) (string, error) {
	disk, err := newAWSDisk(aws.ec2, diskName)
	if err != nil {
		return "", err
	}

	var awsInstance *awsInstance
	if instanceName == "" {
		awsInstance, err = aws.getSelfAWSInstance()
		if err != nil {
			return "", fmt.Errorf("Error getting self-instance: %v", err)
		}
	} else {
		instance, err := aws.getInstancesByDnsName(instanceName)
		if err != nil {
			return "", fmt.Errorf("Error finding instance: %v", err)
		}

		awsInstance = newAWSInstance(aws.ec2, instance.InstanceId)
	}

	if readOnly {
		// TODO: We could enforce this when we mount the volume (?)
		// TODO: We could also snapshot the volume and attach copies of it
		return "", errors.New("AWS volumes cannot be mounted read-only")
	}

	mountDevice, alreadyAttached, err := awsInstance.assignMountDevice(disk.awsID)
	if err != nil {
		return "", err
	}

	attached := false
	defer func() {
		if !attached {
			awsInstance.releaseMountDevice(disk.awsID, mountDevice)
		}
	}()

	if !alreadyAttached {
		attachResponse, err := aws.ec2.AttachVolume(disk.awsID, awsInstance.awsID, mountDevice)
		if err != nil {
			// TODO: Check if the volume was concurrently attached?
			return "", fmt.Errorf("Error attaching EBS volume: %v", err)
		}

		glog.V(2).Info("AttachVolume request returned %v", attachResponse)
	}

	err = disk.waitForAttachmentStatus("attached")
	if err != nil {
		return "", err
	}

	attached = true

	hostDevice := mountDevice
	if strings.HasPrefix(hostDevice, "/dev/sd") {
		// Inside the instance, the mountpoint /dev/sdf looks like /dev/xvdf
		hostDevice = "/dev/xvd" + hostDevice[7:]
	}
	return hostDevice, nil
}

// Implements Volumes.DetachDisk
func (aws *AWSCloud) DetachDisk(instanceName string, diskName string) error {
	disk, err := newAWSDisk(aws.ec2, diskName)
	if err != nil {
		return err
	}

	// TODO: We should specify the InstanceID and the Device, for safety
	response, err := aws.ec2.DetachVolume(disk.awsID)
	if err != nil {
		return fmt.Errorf("error detaching EBS volume: %v", err)
	}
	if response == nil {
		return errors.New("no response from DetachVolume")
	}
	err = disk.waitForAttachmentStatus("detached")
	if err != nil {
		return err
	}

	return err
}

// Implements Volumes.CreateVolume
func (aws *AWSCloud) CreateVolume(volumeOptions *VolumeOptions) (string, error) {
	request := &ec2.CreateVolume{}
	request.AvailZone = aws.availabilityZone
	request.Size = (int64(volumeOptions.CapacityMB) + 1023) / 1024
	response, err := aws.ec2.CreateVolume(request)
	if err != nil {
		return "", err
	}

	az := response.AvailZone
	awsID := response.VolumeId

	volumeName := "aws://" + az + "/" + awsID

	return volumeName, nil
}

// Implements Volumes.DeleteVolume
func (aws *AWSCloud) DeleteVolume(volumeName string) error {
	awsDisk, err := newAWSDisk(aws.ec2, volumeName)
	if err != nil {
		return err
	}
	return awsDisk.delete()
}

func (v *AWSCloud) Configure(name string, spec *api.NodeSpec) error {
	return nil
}

func (v *AWSCloud) Release(name string) error {
	return nil
}
