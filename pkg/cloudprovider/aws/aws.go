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
	"github.com/mitchellh/goamz/elb"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"github.com/golang/glog"
)

const LOADBALANCER_NAME_PREFIX = "k8s-"
const LOADBALANCER_TAG_NAME = "k8s:name"
const LOADBALANCER_NAME_MAXLEN = 32 // AWS limits load balancer names to 32 characters

// TODO: Should we rename this to AWS (EBS & ELB are not technically part of EC2)
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

	// TODO: It is weird that these take a region.  I suspect it won't work cross-region anwyay.
	// TODO: Refactor to use a load balancer object?
	// List load balancers; returns a map of the kubernetes name to the load balancer
	// Note that the k8s name is not the AWS name, because the AWS name is limited to 32 chars,
	// and the k8s name is not (the k8s name may also allow more characters).
	DescribeLoadBalancers(region string, name string) (map[string]elb.LoadBalancer, error)
	// Create load balancer
	CreateLoadBalancer(region string, request *elb.CreateLoadBalancer) (string, error)
	// Add backends to load balancer
	RegisterInstancesWithLoadBalancer(region string, request *elb.RegisterInstancesWithLoadBalancer) ([]elb.Instance, error)
	// Remove backends from load balancer
	DeregisterInstancesFromLoadBalancer(region string, request *elb.DeregisterInstancesFromLoadBalancer) ([]elb.Instance, error)
	// Delete load balancer
	DeleteLoadBalancer(region string, name string) error

	// List subnets
	DescribeSubnets(subnetIds []string, filterVPCId string) ([]ec2.Subnet, error)

	// List security groups
	DescribeSecurityGroups(groupIds []string, filterName string, filterVPCId string) ([]ec2.SecurityGroupInfo, error)
	// Create security group and return the id
	CreateSecurityGroup(vpcId string, name string, description string) (string, error)
	// Authorize security group ingress
	AuthorizeSecurityGroupIngress(securityGroupId string, perms []ec2.IPPerm) (*ec2.SimpleResp, error)

	// List VPCs
	ListVPCs(filterName string) ([]ec2.VPC, error)
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
	auth aws.Auth
	ec2  *ec2.EC2

	mutex      sync.Mutex
	elbClients map[string]*elb.ELB
}

func newGoamzEC2(auth aws.Auth, region aws.Region) (*goamzEC2, error) {
	self := &goamzEC2{}
	self.ec2 = ec2.New(auth, region)
	self.auth = auth
	self.elbClients = make(map[string]*elb.ELB)
	return self, nil
}

// Find the VPC with the given name
func (self *goamzEC2) ListVPCs(filterName string) ([]ec2.VPC, error) {
	client := self.ec2

	// TODO: How do we want to identify our VPC?  Issue #6006
	filter := ec2.NewFilter()
	filter.Add("tag:Name", filterName)

	ids := []string{}
	response, err := client.DescribeVpcs(ids, filter)
	if err != nil {
		glog.Error("error listing VPCs", err)
		return nil, err
	}

	vpcs := response.VPCs
	return vpcs, nil
}

// Builds an ELB client for the specified region
func (self *goamzEC2) getELBClient(regionName string) (*elb.ELB, error) {
	self.mutex.Lock()
	defer self.mutex.Unlock()

	region, ok := aws.Regions[regionName]
	if !ok {
		return nil, fmt.Errorf("not a valid AWS region: %s", regionName)
	}
	elbClient, found := self.elbClients[region.Name]
	if !found {
		elbClient = elb.New(self.auth, region)
		self.elbClients[region.Name] = elbClient
	}
	return elbClient, nil
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

// Implements EC2.DescribeLoadBalancers
func (self *goamzEC2) DescribeLoadBalancers(region string, findName string) (map[string]elb.LoadBalancer, error) {
	client, err := self.getELBClient(region)
	if err != nil {
		return nil, err
	}

	request := &elb.DescribeLoadBalancer{}
	// Names are limited to 32 characters, so we must use tags (instead of filtering by name)
	response, err := client.DescribeLoadBalancers(request)
	if err != nil {
		elbError, ok := err.(*elb.Error)
		if ok && elbError.Code == "LoadBalancerNotFound" {
			// Not found
			return nil, nil
		}
		glog.Error("error describing load balancers: ", err)
		return nil, err
	}

	loadBalancersByAwsId := map[string]elb.LoadBalancer{}
	for _, loadBalancer := range response.LoadBalancers {
		awsId := loadBalancer.LoadBalancerName
		if !strings.HasPrefix(awsId, LOADBALANCER_NAME_PREFIX) {
			continue
		}

		// TODO: Cache the name -> tag mapping (it should never change)
		loadBalancersByAwsId[awsId] = loadBalancer
	}

	loadBalancersByName := map[string]elb.LoadBalancer{}
	if len(loadBalancersByAwsId) == 0 {
		return loadBalancersByName, nil
	}

	describeTagsRequest := &elb.DescribeTags{}
	describeTagsRequest.LoadBalancerNames = []string{}
	for awsId := range loadBalancersByAwsId {
		describeTagsRequest.LoadBalancerNames = append(describeTagsRequest.LoadBalancerNames, awsId)
	}
	describeTagsResponse, err := client.DescribeTags(describeTagsRequest)
	if err != nil {
		glog.Error("error describing tags for load balancers: ", err)
		return nil, err
	}

	if describeTagsResponse.NextToken != "" {
		// TODO: Implement this
		err := fmt.Errorf("error describing tags for load balancers - pagination not implemented")
		return nil, err
	}

	for _, loadBalancerTag := range describeTagsResponse.LoadBalancerTags {
		awsId := loadBalancerTag.LoadBalancerName
		name := ""
		for _, tag := range loadBalancerTag.Tags {
			if tag.Key == LOADBALANCER_TAG_NAME {
				name = tag.Value
			}
		}
		if name == "" {
			glog.Warning("Ignoring load balancer with no k8s name tag: ", awsId)
			continue
		}

		if findName != "" && name != findName {
			continue
		}

		loadBalancer, ok := loadBalancersByAwsId[awsId]
		if !ok {
			// This might almost be panic-worthy!
			glog.Error("unexpected internal error - did not find load balancer")
			continue
		}
		loadBalancersByName[name] = loadBalancer
	}
	return loadBalancersByName, nil
}

// Implements EC2.CreateLoadBalancer
func (self *goamzEC2) CreateLoadBalancer(region string, request *elb.CreateLoadBalancer) (string, error) {
	client, err := self.getELBClient(region)
	if err != nil {
		return "", err
	}

	response, err := client.CreateLoadBalancer(request)
	if err != nil {
		glog.Error("error creating load balancer: ", err)
		return "", err
	}
	return response.DNSName, nil
}

// Implements EC2.DeleteLoadBalancer
func (self *goamzEC2) DeleteLoadBalancer(region string, name string) error {
	client, err := self.getELBClient(region)
	if err != nil {
		return err
	}

	request := &elb.DeleteLoadBalancer{}
	request.LoadBalancerName = name

	_, err = client.DeleteLoadBalancer(request)
	if err != nil {
		// TODO: Check if error was because load balancer was concurrently deleted
		glog.Error("error deleting load balancer: ", err)
		return err
	}
	return nil
}

// Implements EC2.RegisterInstancesWithLoadBalancer
func (self *goamzEC2) RegisterInstancesWithLoadBalancer(region string, request *elb.RegisterInstancesWithLoadBalancer) ([]elb.Instance, error) {
	client, err := self.getELBClient(region)
	if err != nil {
		return nil, err
	}

	response, err := client.RegisterInstancesWithLoadBalancer(request)
	if err != nil {
		glog.Error("error registering instances with load balancer: ", err)
		return nil, err
	}
	return response.Instances, nil
}

// Implements EC2.DeregisterInstancesFromLoadBalancer
func (self *goamzEC2) DeregisterInstancesFromLoadBalancer(region string, request *elb.DeregisterInstancesFromLoadBalancer) ([]elb.Instance, error) {
	client, err := self.getELBClient(region)
	if err != nil {
		return nil, err
	}

	response, err := client.DeregisterInstancesFromLoadBalancer(request)
	if err != nil {
		glog.Error("error deregistering instances from load balancer: ", err)
		return nil, err
	}
	return response.Instances, nil
}

// Implements EC2.DescribeSubnets
func (self *goamzEC2) DescribeSubnets(subnetIds []string, filterVPCId string) ([]ec2.Subnet, error) {
	filter := ec2.NewFilter()
	if filterVPCId != "" {
		filter.Add("vpc-id", filterVPCId)
	}
	response, err := self.ec2.DescribeSubnets(subnetIds, filter)
	if err != nil {
		glog.Error("error describing subnets: ", err)
		return nil, err
	}
	return response.Subnets, nil
}

// Implements EC2.DescribeSecurityGroups
func (self *goamzEC2) DescribeSecurityGroups(securityGroupIds []string, filterName string, filterVPCId string) ([]ec2.SecurityGroupInfo, error) {
	filter := ec2.NewFilter()
	if filterName != "" {
		filter.Add("group-name", filterName)
	}
	if filterVPCId != "" {
		filter.Add("vpc-id", filterVPCId)
	}
	var findGroups []ec2.SecurityGroup
	if securityGroupIds != nil {
		findGroups = []ec2.SecurityGroup{}
		for _, securityGroupId := range securityGroupIds {
			findGroup := ec2.SecurityGroup{Id: securityGroupId}
			findGroups = append(findGroups, findGroup)
		}
	}

	response, err := self.ec2.SecurityGroups(findGroups, filter)
	if err != nil {
		glog.Error("error describing groups: ", err)
		return nil, err
	}
	return response.Groups, nil
}

// Implements EC2.CreateSecurityGroup
func (self *goamzEC2) CreateSecurityGroup(vpcId string, name string, description string) (string, error) {
	request := ec2.SecurityGroup{}
	request.VpcId = vpcId
	request.Name = name
	request.Description = description
	response, err := self.ec2.CreateSecurityGroup(request)
	if err != nil {
		glog.Error("error creating security group: ", err)
		return "", err
	}

	return response.Id, nil
}

// Implements EC2.AuthorizeSecurityGroupIngess
func (self *goamzEC2) AuthorizeSecurityGroupIngress(securityGroupId string, perms []ec2.IPPerm) (*ec2.SimpleResp, error) {
	groupSpec := ec2.SecurityGroup{Id: securityGroupId}

	response, err := self.ec2.AuthorizeSecurityGroup(groupSpec, perms)
	if err != nil {
		glog.Error("error creating security group: ", err)
		return nil, err
	}

	return response, nil
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

	ec2, err := newGoamzEC2(auth, region)
	if err != nil {
		return nil, err
	}

	awsCloud := &AWSCloud{
		ec2:              ec2,
		cfg:              cfg,
		region:           region,
		availabilityZone: zone,
		metadata:         metadata,
	}

	return awsCloud, nil
}

func (self *AWSCloud) Clusters() (cloudprovider.Clusters, bool) {
	return nil, false
}

// TCPLoadBalancer returns an implementation of TCPLoadBalancer for Amazon Web Services.
func (self *AWSCloud) TCPLoadBalancer() (cloudprovider.TCPLoadBalancer, bool) {
	return self, true
}

// Instances returns an implementation of Instances for Amazon Web Services.
func (self *AWSCloud) Instances() (cloudprovider.Instances, bool) {
	return self, true
}

// Zones returns an implementation of Zones for Amazon Web Services.
func (self *AWSCloud) Zones() (cloudprovider.Zones, bool) {
	return self, true
}

// NodeAddresses is an implementation of Instances.NodeAddresses.
func (self *AWSCloud) NodeAddresses(name string) ([]api.NodeAddress, error) {
	inst, err := self.getInstanceByDnsName(name)
	if err != nil {
		return nil, err
	}
	ip := net.ParseIP(inst.PrivateIpAddress)
	if ip == nil {
		return nil, fmt.Errorf("invalid network IP: %s", inst.PrivateIpAddress)
	}

	return []api.NodeAddress{{Type: api.NodeLegacyHostIP, Address: ip.String()}}, nil
}

// ExternalID returns the cloud provider ID of the specified instance.
func (self *AWSCloud) ExternalID(name string) (string, error) {
	inst, err := self.getInstanceByDnsName(name)
	if err != nil {
		return "", err
	}
	return inst.InstanceId, nil
}

// Return the instances matching the relevant private dns name.
func (self *AWSCloud) getInstanceByDnsName(name string) (*ec2.Instance, error) {
	f := &ec2InstanceFilter{}
	f.PrivateDNSName = name

	resp, err := self.ec2.Instances(nil, f)
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

// TODO: Make efficient
func (self *AWSCloud) getInstancesByDnsNames(names []string) ([]*ec2.Instance, error) {
	instances := []*ec2.Instance{}
	for _, name := range names {
		instance, err := self.getInstanceByDnsName(name)
		if err != nil {
			return nil, err
		}
		if instance == nil {
			return nil, fmt.Errorf("unable to find instance " + name)
		}
		instances = append(instances, instance)
	}
	return instances, nil
}

// Return a list of instances matching regex string.
func (self *AWSCloud) getInstancesByRegex(regex string) ([]string, error) {
	resp, err := self.ec2.Instances(nil, nil)
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
func (self *AWSCloud) List(filter string) ([]string, error) {
	// TODO: Should really use tag query. No need to go regexp.
	return self.getInstancesByRegex(filter)
}

// GetNodeResources implements Instances.GetNodeResources
func (self *AWSCloud) GetNodeResources(name string) (*api.NodeResources, error) {
	instance, err := self.getInstanceByDnsName(name)
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

// Gets the current load balancer state
func (self *AWSCloud) describeLoadBalancer(region, name string) (*elb.LoadBalancer, error) {
	loadBalancers, err := self.ec2.DescribeLoadBalancers(region, name)
	if err != nil {
		return nil, err
	}

	var ret *elb.LoadBalancer
	for _, loadBalancer := range loadBalancers {
		if ret != nil {
			glog.Errorf("Found multiple load balancers with name: %s", name)
		}
		ret = &loadBalancer
	}
	return ret, nil
}

// TCPLoadBalancerExists implements TCPLoadBalancer.TCPLoadBalancerExists.
func (self *AWSCloud) TCPLoadBalancerExists(name, region string) (bool, error) {
	lb, err := self.describeLoadBalancer(name, region)
	if err != nil {
		return false, err
	}

	if lb != nil {
		return true, nil
	}
	return false, nil
}

// Find the kubernetes VPC
func (self *AWSCloud) findVPC() (*ec2.VPC, error) {
	name := "kubernetes-vpc"
	vpcs, err := self.ec2.ListVPCs(name)
	if err != nil {
		return nil, err
	}
	if len(vpcs) == 0 {
		return nil, nil
	}
	if len(vpcs) == 1 {
		return &vpcs[0], nil
	}
	return nil, fmt.Errorf("Found multiple matching VPCs for name: %s", name)
}

func mapToInstanceIds(instances []*ec2.Instance) []string {
	ids := make([]string, 0, len(instances))
	for _, instance := range instances {
		ids = append(ids, instance.InstanceId)
	}
	return ids
}

// Makes sure the security group allows ingress on the specified ports (with sourceIp & protocol)
// Returns true iff changes were made
func (self *AWSCloud) ensureSecurityGroupIngess(securityGroupId string, sourceIp string, protocol string, ports []int) (bool, error) {
	groups, err := self.ec2.DescribeSecurityGroups([]string{securityGroupId}, "", "")
	if err != nil {
		glog.Warning("error retrieving security group", err)
		return false, err
	}

	if len(groups) == 0 {
		return false, fmt.Errorf("security group not found")
	}
	if len(groups) != 1 {
		return false, fmt.Errorf("multiple security groups found with same id")
	}
	group := groups[0]

	newPermissions := []ec2.IPPerm{}

	for _, port := range ports {
		found := false
		for _, permission := range group.IPPerms {
			if permission.FromPort != port {
				continue
			}
			if permission.ToPort != port {
				continue
			}
			if permission.Protocol != protocol {
				continue
			}
			if len(permission.SourceIPs) != 1 {
				continue
			}
			if permission.SourceIPs[0] != sourceIp {
				continue
			}
			found = true
			break
		}

		if !found {
			newPermission := ec2.IPPerm{}
			newPermission.FromPort = port
			newPermission.ToPort = port
			newPermission.SourceIPs = []string{sourceIp}
			newPermission.Protocol = protocol

			newPermissions = append(newPermissions, newPermission)
		}
	}

	if len(newPermissions) == 0 {
		return false, nil
	}

	_, err = self.ec2.AuthorizeSecurityGroupIngress(securityGroupId, newPermissions)
	if err != nil {
		glog.Warning("error authorizing security group ingress", err)
		return false, err
	}

	return true, nil
}

// CreateTCPLoadBalancer implements TCPLoadBalancer.CreateTCPLoadBalancer
func (self *AWSCloud) CreateTCPLoadBalancer(name, region string, externalIP net.IP, ports []int, hosts []string, affinity api.AffinityType) (string, error) {
	glog.V(2).Infof("CreateTCPLoadBalancer(%v, %v, %v, %v, %v)", name, region, externalIP, ports, hosts)

	if affinity != api.AffinityTypeNone {
		// ELB supports sticky sessions, but only when configured for HTTP/HTTPS
		return "", fmt.Errorf("unsupported load balancer affinity: %v", affinity)
	}

	if len(externalIP) > 0 {
		return "", fmt.Errorf("External IP cannot be specified for AWS ELB")
	}

	instances, err := self.getInstancesByDnsNames(hosts)
	if err != nil {
		return "", err
	}

	vpc, err := self.findVPC()
	if err != nil {
		glog.Error("error finding VPC", err)
		return "", err
	}
	if vpc == nil {
		return "", fmt.Errorf("Unable to find VPC")
	}

	// Construct list of configured subnets
	subnetIds := []string{}
	{
		subnets, err := self.ec2.DescribeSubnets(nil, vpc.VpcId)
		if err != nil {
			return "", err
		}

		//	zones := []string{}
		for _, subnet := range subnets {
			subnetIds = append(subnetIds, subnet.SubnetId)
			if !strings.HasPrefix(subnet.AvailabilityZone, region) {
				glog.Error("found AZ that did not match region", subnet.AvailabilityZone, " vs ", region)
				return "", fmt.Errorf("invalid AZ for region")
			}
			//		zones = append(zones, subnet.AvailabilityZone)
		}
	}

	// Build the load balancer itself
	var loadBalancerName, dnsName string
	{
		loadBalancer, err := self.describeLoadBalancer(region, name)
		if err != nil {
			return "", err
		}

		if loadBalancer == nil {
			createRequest := &elb.CreateLoadBalancer{}
			// TODO: Is there a k8s UUID that it would make sense to use?
			uuid := strings.Replace(string(util.NewUUID()), "-", "", -1)
			awsId := LOADBALANCER_NAME_PREFIX + uuid
			if len(awsId) > LOADBALANCER_NAME_MAXLEN {
				awsId = awsId[:LOADBALANCER_NAME_MAXLEN]
			}
			createRequest.LoadBalancerName = awsId

			listeners := []elb.Listener{}
			for _, port := range ports {
				listener := elb.Listener{}
				listener.InstancePort = int64(port)
				listener.LoadBalancerPort = int64(port)
				listener.Protocol = "tcp"
				listener.InstanceProtocol = "tcp"

				listeners = append(listeners, listener)
			}

			createRequest.Listeners = listeners

			// TODO: Should we use a better identifier (the kubernetes uuid?)
			//	nameTag := &elb.Tag{ Key: "Name", Value: name}
			//	createRequest.Tags = []Tag { nameTag }

			//	zones := []string{"us-east-1a"}
			//	createRequest.AvailZone = removeDuplicates(zones)

			// We are supposed to specify one subnet per AZ.
			// TODO: What happens if we have more than one subnet per AZ?
			createRequest.Subnets = subnetIds

			sgName := "k8s-elb-" + name
			sgDescription := "Security group for Kubernetes ELB " + name

			{
				// TODO: Should we do something more reliable ?? .Where("tag:kubernetes-id", kubernetesId)
				securityGroups, err := self.ec2.DescribeSecurityGroups(nil, sgName, vpc.VpcId)
				if err != nil {
					return "", err
				}
				var securityGroupId string
				for _, securityGroup := range securityGroups {
					securityGroupId = securityGroup.Id
				}
				if securityGroupId == "" {
					securityGroupId, err = self.ec2.CreateSecurityGroup(vpc.VpcId, sgName, sgDescription)
					if err != nil {
						return "", err
					}
				}
				_, err = self.ensureSecurityGroupIngess(securityGroupId, "0.0.0.0/0", "tcp", ports)
				if err != nil {
					return "", err
				}
				createRequest.SecurityGroups = []string{securityGroupId}
			}

			if len(externalIP) > 0 {
				return "", fmt.Errorf("External IP cannot be specified for AWS ELB")
			}

			glog.Info("Creating load balancer with name: ", createRequest.LoadBalancerName)
			createdDnsName, err := self.ec2.CreateLoadBalancer(region, createRequest)
			if err != nil {
				return "", err
			}
			dnsName = createdDnsName
			loadBalancerName = createRequest.LoadBalancerName
		} else {
			// TODO: Verify that load balancer configuration matches?
			dnsName = loadBalancer.DNSName
			loadBalancerName = loadBalancer.LoadBalancerName
		}
	}

	registerRequest := &elb.RegisterInstancesWithLoadBalancer{}
	registerRequest.LoadBalancerName = loadBalancerName
	registerRequest.Instances = mapToInstanceIds(instances)

	attachedInstances, err := self.ec2.RegisterInstancesWithLoadBalancer(region, registerRequest)
	if err != nil {
		return err
	}

	glog.V(1).Info("Updated instances registered with load-balancer", name, attachedInstances)
	glog.V(1).Info("Loadbalancer %s has DNS name %s", name, dnsName)

	// TODO: Wait for creation?

	return dnsName, nil
}

// DeleteTCPLoadBalancer implements TCPLoadBalancer.DeleteTCPLoadBalancer.
func (self *AWSCloud) DeleteTCPLoadBalancer(name, region string) error {
	// TODO: Delete security group

	lb, err := self.describeLoadBalancer(region, name)
	if err != nil {
		return err
	}

	if lb == nil {
		glog.Info("Load balancer already deleted: ", name)
		return nil
	}

	err = self.ec2.DeleteLoadBalancer(region, lb.LoadBalancerName)
	if err != nil {
		return err
	}
	return nil
}

// UpdateTCPLoadBalancer implements TCPLoadBalancer.UpdateTCPLoadBalancer
func (self *AWSCloud) UpdateTCPLoadBalancer(name, region string, hosts []string) error {
	instances, err := self.getInstancesByDnsNames(hosts)
	if err != nil {
		return err
	}

	lb, err := self.describeLoadBalancer(region, name)
	if err != nil {
		return err
	}

	if lb == nil {
		return fmt.Errorf("Load balancer not found")
	}

	existingInstances := map[string]*elb.Instance{}
	for _, instance := range lb.Instances {
		existingInstances[instance.InstanceId] = &instance
	}

	wantInstances := map[string]*ec2.Instance{}
	for _, instance := range instances {
		wantInstances[instance.InstanceId] = instance
	}

	addInstances := []string{}
	for key := range wantInstances {
		_, found := existingInstances[key]
		if !found {
			addInstances = append(addInstances, key)
		}
	}

	removeInstances := []string{}
	for key := range existingInstances {
		_, found := wantInstances[key]
		if !found {
			removeInstances = append(removeInstances, key)
		}
	}

	if len(addInstances) > 0 {
		registerRequest := &elb.RegisterInstancesWithLoadBalancer{}
		registerRequest.Instances = addInstances
		registerRequest.LoadBalancerName = lb.LoadBalancerName
		_, err = self.ec2.RegisterInstancesWithLoadBalancer(region, registerRequest)
		if err != nil {
			return err
		}
	}

	if len(removeInstances) > 0 {
		deregisterRequest := &elb.DeregisterInstancesFromLoadBalancer{}
		deregisterRequest.Instances = removeInstances
		deregisterRequest.LoadBalancerName = lb.LoadBalancerName
		_, err = self.ec2.DeregisterInstancesFromLoadBalancer(region, deregisterRequest)
		if err != nil {
			return err
		}
	}

	return nil
}
