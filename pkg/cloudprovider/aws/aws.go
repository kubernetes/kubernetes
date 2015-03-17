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
	"fmt"
	"io"
	"net"
	"regexp"
	"strings"

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

	// Query the EC2 metadata service (used to discover instance-id etc)
	GetMetaData(key string) ([]byte, error)
}

// AWSCloud is an implementation of Interface, TCPLoadBalancer and Instances for Amazon Web Services.
type AWSCloud struct {
	ec2              EC2
	cfg              *AWSCloudConfig
	availabilityZone string
	region           aws.Region
}

type AWSCloudConfig struct {
	Global struct {
		// TODO: Is there any use for this?  We can get it from the instance metadata service
		Region string
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
type GoamzEC2 struct {
	ec2 *ec2.EC2
}

// Implementation of EC2.Instances
func (self *GoamzEC2) Instances(instanceIds []string, filter *ec2InstanceFilter) (resp *ec2.InstancesResp, err error) {
	var goamzFilter *ec2.Filter
	if filter != nil {
		goamzFilter = ec2.NewFilter()
		if filter.PrivateDNSName != "" {
			goamzFilter.Add("private-dns-name", filter.PrivateDNSName)
		}
	}
	return self.ec2.Instances(instanceIds, goamzFilter)
}

func (self *GoamzEC2) GetMetaData(key string) ([]byte, error) {
	v, err := aws.GetMetaData(key)
	if err != nil {
		return nil, fmt.Errorf("Error querying AWS metadata for key %s: %v", key, err)
	}
	return v, nil
}

type AuthFunc func() (auth aws.Auth, err error)

func init() {
	cloudprovider.RegisterCloudProvider("aws", func(config io.Reader) (cloudprovider.Interface, error) {
		return newAWSCloud(config, getAuth)
	})
}

func getAuth() (auth aws.Auth, err error) {
	return aws.GetAuth("", "")
}

// readAWSCloudConfig reads an instance of AWSCloudConfig from config reader.
func readAWSCloudConfig(config io.Reader) (*AWSCloudConfig, error) {
	if config == nil {
		return nil, fmt.Errorf("no AWS cloud provider config file given")
	}

	var cfg AWSCloudConfig
	err := gcfg.ReadInto(&cfg, config)
	if err != nil {
		return nil, err
	}

	if cfg.Global.Region == "" {
		return nil, fmt.Errorf("no region specified in configuration file")
	}

	return &cfg, nil
}

// newAWSCloud creates a new instance of AWSCloud.
func newAWSCloud(config io.Reader, authFunc AuthFunc) (*AWSCloud, error) {
	cfg, err := readAWSCloudConfig(config)
	if err != nil {
		return nil, fmt.Errorf("unable to read AWS cloud provider config file: %v", err)
	}

	auth, err := authFunc()
	if err != nil {
		return nil, err
	}

	// TODO: We can get the region very easily from the instance-metadata service
	region, ok := aws.Regions[cfg.Global.Region]
	if !ok {
		return nil, fmt.Errorf("not a valid AWS region: %s", cfg.Global.Region)
	}

	return &AWSCloud{
		ec2:    &GoamzEC2{ec2: ec2.New(auth, region)},
		cfg:    cfg,
		region: region,
	}, nil
}

func (self *AWSCloud) getAvailabilityZone() (string, error) {
	// TODO: Do we need sync.Mutex here?
	availabilityZone := self.availabilityZone
	if self.availabilityZone == "" {
		availabilityZoneBytes, err := self.ec2.GetMetaData("placement/availability-zone")
		if err != nil {
			return "", err
		}
		if availabilityZoneBytes == nil || len(availabilityZoneBytes) == 0 {
			return "", fmt.Errorf("Unable to determine availability-zone from instance metadata")
		}
		availabilityZone = string(availabilityZoneBytes)
		self.availabilityZone = availabilityZone
	}
	return availabilityZone, nil
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
	inst, err := aws.getInstancesByDnsName(name)
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
				glog.V(2).Infof("skipping EC2 instance (not alive): %s", instance.InstanceId)
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
	availabilityZone, err := self.getAvailabilityZone()
	if err != nil {
		return cloudprovider.Zone{}, err
	}
	return cloudprovider.Zone{
		FailureDomain: availabilityZone,
		Region:        self.region.Name,
	}, nil
}
