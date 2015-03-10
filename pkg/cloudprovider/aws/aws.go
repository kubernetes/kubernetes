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

// IPAddress is an implementation of Instances.IPAddress.
func (aws *AWSCloud) IPAddress(name string) (net.IP, error) {
	inst, err := aws.getInstancesByDnsName(name)
	if err != nil {
		return nil, err
	}
	ip := net.ParseIP(inst.PrivateIpAddress)
	if ip == nil {
		return nil, fmt.Errorf("invalid network IP: %s", inst.PrivateIpAddress)
	}
	return ip, nil
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

func (v *AWSCloud) GetNodeResources(name string) (*api.NodeResources, error) {
	return nil, nil
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
