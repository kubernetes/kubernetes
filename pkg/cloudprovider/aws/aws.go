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

	"code.google.com/p/gcfg"
	"github.com/mitchellh/goamz/aws"
	"github.com/mitchellh/goamz/ec2"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
)

// Abstraction over EC2, to allow mocking/other implementations
type EC2 interface {
	// Query EC2 for instances matching the filter
	Instances(instIds []string, filter *ec2InstanceFilter) (resp *ec2.InstancesResp, err error)
}

// AWSCloud is an implementation of Interface, TCPLoadBalancer and Instances for Amazon Web Services.
type AWSCloud struct {
	ec2 EC2
	cfg *AWSCloudConfig
}

type AWSCloudConfig struct {
	Global struct {
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

	region, ok := aws.Regions[cfg.Global.Region]
	if !ok {
		return nil, fmt.Errorf("not a valid AWS region: %s", cfg.Global.Region)
	}

	ec2 := ec2.New(auth, region)
	return &AWSCloud{
		ec2: &GoamzEC2{ec2: ec2},
		cfg: cfg,
	}, nil
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
	return nil, false
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
	if len(resp.Reservations) == 0 {
		return nil, fmt.Errorf("no reservations found for host: %s", name)
	}
	if len(resp.Reservations) > 1 {
		return nil, fmt.Errorf("multiple reservations found for host: %s", name)
	}
	if len(resp.Reservations[0].Instances) == 0 {
		return nil, fmt.Errorf("no instances found for host: %s", name)
	}
	if len(resp.Reservations[0].Instances) > 1 {
		return nil, fmt.Errorf("multiple instances found for host: %s", name)
	}

	return &resp.Reservations[0].Instances[0], nil
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

	re, err := regexp.Compile(regex)
	if err != nil {
		return []string{}, err
	}

	instances := []string{}
	for _, reservation := range resp.Reservations {
		for _, instance := range reservation.Instances {
			for _, tag := range instance.Tags {
				if tag.Key == "Name" && re.MatchString(tag.Value) {
					instances = append(instances, instance.PrivateDNSName)
					break
				}
			}
		}
	}
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
