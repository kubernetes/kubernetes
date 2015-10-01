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
	"fmt"
	"io"
	"reflect"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/ec2"
	"github.com/aws/aws-sdk-go/service/elb"

	"github.com/aws/aws-sdk-go/service/autoscaling"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
)

const TestClusterId = "clusterid.test"

func TestReadAWSCloudConfig(t *testing.T) {
	tests := []struct {
		name string

		reader io.Reader
		aws    AWSServices

		expectError bool
		zone        string
	}{
		{
			"No config reader",
			nil, nil,
			true, "",
		},
		{
			"Empty config, no metadata",
			strings.NewReader(""), nil,
			true, "",
		},
		{
			"No zone in config, no metadata",
			strings.NewReader("[global]\n"), nil,
			true, "",
		},
		{
			"Zone in config, no metadata",
			strings.NewReader("[global]\nzone = eu-west-1a"), nil,
			false, "eu-west-1a",
		},
		{
			"No zone in config, metadata does not have zone",
			strings.NewReader("[global]\n"), NewFakeAWSServices().withAz(""),
			true, "",
		},
		{
			"No zone in config, metadata has zone",
			strings.NewReader("[global]\n"), NewFakeAWSServices(),
			false, "us-east-1a",
		},
		{
			"Zone in config should take precedence over metadata",
			strings.NewReader("[global]\nzone = eu-west-1a"), NewFakeAWSServices(),
			false, "eu-west-1a",
		},
	}

	for _, test := range tests {
		t.Logf("Running test case %s", test.name)
		var metadata EC2Metadata
		if test.aws != nil {
			metadata, _ = test.aws.Metadata()
		}
		cfg, err := readAWSCloudConfig(test.reader, metadata)
		if test.expectError {
			if err == nil {
				t.Errorf("Should error for case %s (cfg=%v)", test.name, cfg)
			}
		} else {
			if err != nil {
				t.Errorf("Should succeed for case: %s", test.name)
			}
			if cfg.Global.Zone != test.zone {
				t.Errorf("Incorrect zone value (%s vs %s) for case: %s",
					cfg.Global.Zone, test.zone, test.name)
			}
		}
	}
}

type FakeAWSServices struct {
	availabilityZone        string
	instances               []*ec2.Instance
	instanceId              string
	privateDnsName          string
	networkInterfacesMacs   []string
	networkInterfacesVpcIDs []string

	ec2      *FakeEC2
	elb      *FakeELB
	asg      *FakeASG
	metadata *FakeMetadata
}

func NewFakeAWSServices() *FakeAWSServices {
	s := &FakeAWSServices{}
	s.availabilityZone = "us-east-1a"
	s.ec2 = &FakeEC2{aws: s}
	s.elb = &FakeELB{aws: s}
	s.asg = &FakeASG{aws: s}
	s.metadata = &FakeMetadata{aws: s}

	s.networkInterfacesMacs = []string{"aa:bb:cc:dd:ee:00", "aa:bb:cc:dd:ee:01"}
	s.networkInterfacesVpcIDs = []string{"vpc-mac0", "vpc-mac1"}

	s.instanceId = "i-self"
	s.privateDnsName = "ip-172-20-0-100.ec2.internal"
	var selfInstance ec2.Instance
	selfInstance.InstanceId = &s.instanceId
	selfInstance.PrivateDnsName = &s.privateDnsName
	s.instances = []*ec2.Instance{&selfInstance}

	var tag ec2.Tag
	tag.Key = aws.String(TagNameKubernetesCluster)
	tag.Value = aws.String(TestClusterId)
	selfInstance.Tags = []*ec2.Tag{&tag}

	return s
}

func (s *FakeAWSServices) withAz(az string) *FakeAWSServices {
	s.availabilityZone = az
	return s
}

func (s *FakeAWSServices) withInstances(instances []*ec2.Instance) *FakeAWSServices {
	s.instances = instances
	return s
}

func (s *FakeAWSServices) Compute(region string) (EC2, error) {
	return s.ec2, nil
}

func (s *FakeAWSServices) LoadBalancing(region string) (ELB, error) {
	return s.elb, nil
}

func (s *FakeAWSServices) Autoscaling(region string) (ASG, error) {
	return s.asg, nil
}

func (s *FakeAWSServices) Metadata() (EC2Metadata, error) {
	return s.metadata, nil
}

func TestFilterTags(t *testing.T) {
	awsServices := NewFakeAWSServices()
	c, err := newAWSCloud(strings.NewReader("[global]"), awsServices)
	if err != nil {
		t.Errorf("Error building aws cloud: %v", err)
		return
	}

	if len(c.filterTags) != 1 {
		t.Errorf("unexpected filter tags: %v", c.filterTags)
		return
	}

	if c.filterTags[TagNameKubernetesCluster] != TestClusterId {
		t.Errorf("unexpected filter tags: %v", c.filterTags)
	}
}

func TestNewAWSCloud(t *testing.T) {
	tests := []struct {
		name string

		reader      io.Reader
		awsServices AWSServices

		expectError bool
		zone        string
	}{
		{
			"No config reader",
			nil, NewFakeAWSServices().withAz(""),
			true, "",
		},
		{
			"Config specified invalid zone",
			strings.NewReader("[global]\nzone = blahonga"), NewFakeAWSServices(),
			true, "",
		},
		{
			"Config specifies valid zone",
			strings.NewReader("[global]\nzone = eu-west-1a"), NewFakeAWSServices(),
			false, "eu-west-1a",
		},
		{
			"Gets zone from metadata when not in config",

			strings.NewReader("[global]\n"),
			NewFakeAWSServices(),
			false, "us-east-1a",
		},
		{
			"No zone in config or metadata",
			strings.NewReader("[global]\n"),
			NewFakeAWSServices().withAz(""),
			true, "",
		},
	}

	for _, test := range tests {
		t.Logf("Running test case %s", test.name)
		c, err := newAWSCloud(test.reader, test.awsServices)
		if test.expectError {
			if err == nil {
				t.Errorf("Should error for case %s", test.name)
			}
		} else {
			if err != nil {
				t.Errorf("Should succeed for case: %s, got %v", test.name, err)
			} else if c.availabilityZone != test.zone {
				t.Errorf("Incorrect zone value (%s vs %s) for case: %s",
					c.availabilityZone, test.zone, test.name)
			}
		}
	}
}

type FakeEC2 struct {
	aws                  *FakeAWSServices
	Subnets              []*ec2.Subnet
	DescribeSubnetsInput *ec2.DescribeSubnetsInput
}

func contains(haystack []*string, needle string) bool {
	for _, s := range haystack {
		// (deliberately panic if s == nil)
		if needle == *s {
			return true
		}
	}
	return false
}

func instanceMatchesFilter(instance *ec2.Instance, filter *ec2.Filter) bool {
	name := *filter.Name
	if name == "private-dns-name" {
		if instance.PrivateDnsName == nil {
			return false
		}
		return contains(filter.Values, *instance.PrivateDnsName)
	}
	panic("Unknown filter name: " + name)
}

func (self *FakeEC2) DescribeInstances(request *ec2.DescribeInstancesInput) ([]*ec2.Instance, error) {
	matches := []*ec2.Instance{}
	for _, instance := range self.aws.instances {
		if request.InstanceIds != nil {
			if instance.InstanceId == nil {
				glog.Warning("Instance with no instance id: ", instance)
				continue
			}

			found := false
			for _, instanceId := range request.InstanceIds {
				if *instanceId == *instance.InstanceId {
					found = true
					break
				}
			}
			if !found {
				continue
			}
		}
		if request.Filters != nil {
			allMatch := true
			for _, filter := range request.Filters {
				if !instanceMatchesFilter(instance, filter) {
					allMatch = false
					break
				}
			}
			if !allMatch {
				continue
			}
		}
		matches = append(matches, instance)
	}

	return matches, nil
}

type FakeMetadata struct {
	aws *FakeAWSServices
}

func (self *FakeMetadata) GetMetadata(key string) (string, error) {
	networkInterfacesPrefix := "network/interfaces/macs/"
	if key == "placement/availability-zone" {
		return self.aws.availabilityZone, nil
	} else if key == "instance-id" {
		return self.aws.instanceId, nil
	} else if key == "local-hostname" {
		return self.aws.privateDnsName, nil
	} else if strings.HasPrefix(key, networkInterfacesPrefix) {
		if key == networkInterfacesPrefix {
			return strings.Join(self.aws.networkInterfacesMacs, "/\n") + "/\n", nil
		} else {
			keySplit := strings.Split(key, "/")
			macParam := keySplit[3]
			if len(keySplit) == 5 && keySplit[4] == "vpc-id" {
				for i, macElem := range self.aws.networkInterfacesMacs {
					if macParam == macElem {
						return self.aws.networkInterfacesVpcIDs[i], nil
					}
				}
			}
			return "", nil
		}
	} else {
		return "", nil
	}
}

func (ec2 *FakeEC2) AttachVolume(volumeID, instanceId, mountDevice string) (resp *ec2.VolumeAttachment, err error) {
	panic("Not implemented")
}

func (ec2 *FakeEC2) DetachVolume(request *ec2.DetachVolumeInput) (resp *ec2.VolumeAttachment, err error) {
	panic("Not implemented")
}

func (ec2 *FakeEC2) DescribeVolumes(request *ec2.DescribeVolumesInput) ([]*ec2.Volume, error) {
	panic("Not implemented")
}

func (ec2 *FakeEC2) CreateVolume(request *ec2.CreateVolumeInput) (resp *ec2.Volume, err error) {
	panic("Not implemented")
}

func (ec2 *FakeEC2) DeleteVolume(volumeID string) (resp *ec2.DeleteVolumeOutput, err error) {
	panic("Not implemented")
}

func (ec2 *FakeEC2) DescribeSecurityGroups(request *ec2.DescribeSecurityGroupsInput) ([]*ec2.SecurityGroup, error) {
	panic("Not implemented")
}

func (ec2 *FakeEC2) CreateSecurityGroup(*ec2.CreateSecurityGroupInput) (*ec2.CreateSecurityGroupOutput, error) {
	panic("Not implemented")
}

func (ec2 *FakeEC2) DeleteSecurityGroup(*ec2.DeleteSecurityGroupInput) (*ec2.DeleteSecurityGroupOutput, error) {
	panic("Not implemented")
}

func (ec2 *FakeEC2) AuthorizeSecurityGroupIngress(*ec2.AuthorizeSecurityGroupIngressInput) (*ec2.AuthorizeSecurityGroupIngressOutput, error) {
	panic("Not implemented")
}

func (ec2 *FakeEC2) RevokeSecurityGroupIngress(*ec2.RevokeSecurityGroupIngressInput) (*ec2.RevokeSecurityGroupIngressOutput, error) {
	panic("Not implemented")
}

func (ec2 *FakeEC2) DescribeVPCs(*ec2.DescribeVpcsInput) ([]*ec2.Vpc, error) {
	panic("Not implemented")
}

func (ec2 *FakeEC2) DescribeSubnets(request *ec2.DescribeSubnetsInput) ([]*ec2.Subnet, error) {
	ec2.DescribeSubnetsInput = request
	return ec2.Subnets, nil
}

func (ec2 *FakeEC2) CreateTags(*ec2.CreateTagsInput) (*ec2.CreateTagsOutput, error) {
	panic("Not implemented")
}

func (s *FakeEC2) DescribeRouteTables(request *ec2.DescribeRouteTablesInput) ([]*ec2.RouteTable, error) {
	panic("Not implemented")
}

func (s *FakeEC2) CreateRoute(request *ec2.CreateRouteInput) (*ec2.CreateRouteOutput, error) {
	panic("Not implemented")
}

func (s *FakeEC2) DeleteRoute(request *ec2.DeleteRouteInput) (*ec2.DeleteRouteOutput, error) {
	panic("Not implemented")
}

func (s *FakeEC2) ModifyInstanceAttribute(request *ec2.ModifyInstanceAttributeInput) (*ec2.ModifyInstanceAttributeOutput, error) {
	panic("Not implemented")
}

type FakeELB struct {
	aws *FakeAWSServices
}

func (ec2 *FakeELB) CreateLoadBalancer(*elb.CreateLoadBalancerInput) (*elb.CreateLoadBalancerOutput, error) {
	panic("Not implemented")
}

func (ec2 *FakeELB) DeleteLoadBalancer(*elb.DeleteLoadBalancerInput) (*elb.DeleteLoadBalancerOutput, error) {
	panic("Not implemented")
}

func (ec2 *FakeELB) DescribeLoadBalancers(*elb.DescribeLoadBalancersInput) (*elb.DescribeLoadBalancersOutput, error) {
	panic("Not implemented")
}
func (ec2 *FakeELB) RegisterInstancesWithLoadBalancer(*elb.RegisterInstancesWithLoadBalancerInput) (*elb.RegisterInstancesWithLoadBalancerOutput, error) {
	panic("Not implemented")
}

func (ec2 *FakeELB) DeregisterInstancesFromLoadBalancer(*elb.DeregisterInstancesFromLoadBalancerInput) (*elb.DeregisterInstancesFromLoadBalancerOutput, error) {
	panic("Not implemented")
}

func (ec2 *FakeELB) DetachLoadBalancerFromSubnets(*elb.DetachLoadBalancerFromSubnetsInput) (*elb.DetachLoadBalancerFromSubnetsOutput, error) {
	panic("Not implemented")
}

func (ec2 *FakeELB) AttachLoadBalancerToSubnets(*elb.AttachLoadBalancerToSubnetsInput) (*elb.AttachLoadBalancerToSubnetsOutput, error) {
	panic("Not implemented")
}

func (ec2 *FakeELB) CreateLoadBalancerListeners(*elb.CreateLoadBalancerListenersInput) (*elb.CreateLoadBalancerListenersOutput, error) {
	panic("Not implemented")
}

func (ec2 *FakeELB) DeleteLoadBalancerListeners(*elb.DeleteLoadBalancerListenersInput) (*elb.DeleteLoadBalancerListenersOutput, error) {
	panic("Not implemented")
}

func (ec2 *FakeELB) ApplySecurityGroupsToLoadBalancer(*elb.ApplySecurityGroupsToLoadBalancerInput) (*elb.ApplySecurityGroupsToLoadBalancerOutput, error) {
	panic("Not implemented")
}

func (elb *FakeELB) ConfigureHealthCheck(*elb.ConfigureHealthCheckInput) (*elb.ConfigureHealthCheckOutput, error) {
	panic("Not implemented")
}

type FakeASG struct {
	aws *FakeAWSServices
}

func (a *FakeASG) UpdateAutoScalingGroup(*autoscaling.UpdateAutoScalingGroupInput) (*autoscaling.UpdateAutoScalingGroupOutput, error) {
	panic("Not implemented")
}

func (a *FakeASG) DescribeAutoScalingGroups(*autoscaling.DescribeAutoScalingGroupsInput) (*autoscaling.DescribeAutoScalingGroupsOutput, error) {
	panic("Not implemented")
}

func mockInstancesResp(instances []*ec2.Instance) *AWSCloud {
	awsServices := NewFakeAWSServices().withInstances(instances)
	return &AWSCloud{
		awsServices:      awsServices,
		ec2:              awsServices.ec2,
		availabilityZone: awsServices.availabilityZone,
	}
}

func mockAvailabilityZone(region string, availabilityZone string) *AWSCloud {
	awsServices := NewFakeAWSServices().withAz(availabilityZone)
	return &AWSCloud{
		awsServices:      awsServices,
		ec2:              awsServices.ec2,
		availabilityZone: awsServices.availabilityZone,
		region:           region,
	}
}

func TestList(t *testing.T) {
	// TODO this setup is not very clean and could probably be improved
	var instance0 ec2.Instance
	var instance1 ec2.Instance
	var instance2 ec2.Instance
	var instance3 ec2.Instance

	//0
	tag0 := ec2.Tag{
		Key:   aws.String("Name"),
		Value: aws.String("foo"),
	}
	instance0.Tags = []*ec2.Tag{&tag0}
	instance0.InstanceId = aws.String("instance0")
	instance0.PrivateDnsName = aws.String("instance0.ec2.internal")
	state0 := ec2.InstanceState{
		Name: aws.String("running"),
	}
	instance0.State = &state0

	//1
	tag1 := ec2.Tag{
		Key:   aws.String("Name"),
		Value: aws.String("bar"),
	}
	instance1.Tags = []*ec2.Tag{&tag1}
	instance1.InstanceId = aws.String("instance1")
	instance1.PrivateDnsName = aws.String("instance1.ec2.internal")
	state1 := ec2.InstanceState{
		Name: aws.String("running"),
	}
	instance1.State = &state1

	//2
	tag2 := ec2.Tag{
		Key:   aws.String("Name"),
		Value: aws.String("baz"),
	}
	instance2.Tags = []*ec2.Tag{&tag2}
	instance2.InstanceId = aws.String("instance2")
	instance2.PrivateDnsName = aws.String("instance2.ec2.internal")
	state2 := ec2.InstanceState{
		Name: aws.String("running"),
	}
	instance2.State = &state2

	//3
	tag3 := ec2.Tag{
		Key:   aws.String("Name"),
		Value: aws.String("quux"),
	}
	instance3.Tags = []*ec2.Tag{&tag3}
	instance3.InstanceId = aws.String("instance3")
	instance3.PrivateDnsName = aws.String("instance3.ec2.internal")
	state3 := ec2.InstanceState{
		Name: aws.String("running"),
	}
	instance3.State = &state3

	instances := []*ec2.Instance{&instance0, &instance1, &instance2, &instance3}
	aws := mockInstancesResp(instances)

	table := []struct {
		input  string
		expect []string
	}{
		{"blahonga", []string{}},
		{"quux", []string{"instance3.ec2.internal"}},
		{"a", []string{"instance1.ec2.internal", "instance2.ec2.internal"}},
	}

	for _, item := range table {
		result, err := aws.List(item.input)
		if err != nil {
			t.Errorf("Expected call with %v to succeed, failed with %s", item.input, err)
		}
		if e, a := item.expect, result; !reflect.DeepEqual(e, a) {
			t.Errorf("Expected %v, got %v", e, a)
		}
	}
}

func testHasNodeAddress(t *testing.T, addrs []api.NodeAddress, addressType api.NodeAddressType, address string) {
	for _, addr := range addrs {
		if addr.Type == addressType && addr.Address == address {
			return
		}
	}
	t.Errorf("Did not find expected address: %s:%s in %v", addressType, address, addrs)
}

func TestNodeAddresses(t *testing.T) {
	// Note these instances have the same name
	// (we test that this produces an error)
	var instance0 ec2.Instance
	var instance1 ec2.Instance

	//0
	instance0.InstanceId = aws.String("instance-same")
	instance0.PrivateDnsName = aws.String("instance-same.ec2.internal")
	instance0.PrivateIpAddress = aws.String("192.168.0.1")
	instance0.PublicIpAddress = aws.String("1.2.3.4")
	instance0.InstanceType = aws.String("c3.large")
	state0 := ec2.InstanceState{
		Name: aws.String("running"),
	}
	instance0.State = &state0

	//1
	instance1.InstanceId = aws.String("instance-same")
	instance1.PrivateDnsName = aws.String("instance-same.ec2.internal")
	instance1.PrivateIpAddress = aws.String("192.168.0.2")
	instance1.InstanceType = aws.String("c3.large")
	state1 := ec2.InstanceState{
		Name: aws.String("running"),
	}
	instance1.State = &state1

	instances := []*ec2.Instance{&instance0, &instance1}

	aws1 := mockInstancesResp([]*ec2.Instance{})
	_, err1 := aws1.NodeAddresses("instance-mismatch.ec2.internal")
	if err1 == nil {
		t.Errorf("Should error when no instance found")
	}

	aws2 := mockInstancesResp(instances)
	_, err2 := aws2.NodeAddresses("instance-same.ec2.internal")
	if err2 == nil {
		t.Errorf("Should error when multiple instances found")
	}

	aws3 := mockInstancesResp(instances[0:1])
	addrs3, err3 := aws3.NodeAddresses("instance-same.ec2.internal")
	if err3 != nil {
		t.Errorf("Should not error when instance found")
	}
	if len(addrs3) != 3 {
		t.Errorf("Should return exactly 3 NodeAddresses")
	}
	testHasNodeAddress(t, addrs3, api.NodeInternalIP, "192.168.0.1")
	testHasNodeAddress(t, addrs3, api.NodeLegacyHostIP, "192.168.0.1")
	testHasNodeAddress(t, addrs3, api.NodeExternalIP, "1.2.3.4")
}

func TestGetRegion(t *testing.T) {
	aws := mockAvailabilityZone("us-west-2", "us-west-2e")
	zones, ok := aws.Zones()
	if !ok {
		t.Fatalf("Unexpected missing zones impl")
	}
	zone, err := zones.GetZone()
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	if zone.Region != "us-west-2" {
		t.Errorf("Unexpected region: %s", zone.Region)
	}
	if zone.FailureDomain != "us-west-2e" {
		t.Errorf("Unexpected FailureDomain: %s", zone.FailureDomain)
	}
}

func TestFindVPCID(t *testing.T) {
	awsServices := NewFakeAWSServices()
	c, err := newAWSCloud(strings.NewReader("[global]"), awsServices)
	if err != nil {
		t.Errorf("Error building aws cloud: %v", err)
		return
	}
	vpcID, err := c.findVPCID()
	if err != nil {
		t.Errorf("Unexpected error:", err)
	}
	if vpcID != "vpc-mac0" {
		t.Errorf("Unexpected vpcID: %s", vpcID)
	}
}

func TestLoadBalancerMatchesClusterRegion(t *testing.T) {
	awsServices := NewFakeAWSServices()
	c, err := newAWSCloud(strings.NewReader("[global]"), awsServices)
	if err != nil {
		t.Errorf("Error building aws cloud: %v", err)
		return
	}

	badELBRegion := "bad-elb-region"
	errorMessage := fmt.Sprintf("requested load balancer region '%s' does not match cluster region '%s'", badELBRegion, c.region)

	_, _, err = c.GetTCPLoadBalancer("elb-name", badELBRegion)
	if err == nil || err.Error() != errorMessage {
		t.Errorf("Expected GetTCPLoadBalancer region mismatch error.")
	}

	_, err = c.EnsureTCPLoadBalancer("elb-name", badELBRegion, nil, nil, nil, api.ServiceAffinityNone)
	if err == nil || err.Error() != errorMessage {
		t.Errorf("Expected EnsureTCPLoadBalancer region mismatch error.")
	}

	err = c.EnsureTCPLoadBalancerDeleted("elb-name", badELBRegion)
	if err == nil || err.Error() != errorMessage {
		t.Errorf("Expected EnsureTCPLoadBalancerDeleted region mismatch error.")
	}

	err = c.UpdateTCPLoadBalancer("elb-name", badELBRegion, nil)
	if err == nil || err.Error() != errorMessage {
		t.Errorf("Expected UpdateTCPLoadBalancer region mismatch error.")
	}
}

func constructSubnets(subnetsIn map[int]map[string]string) (subnetsOut []*ec2.Subnet) {
	for i := range subnetsIn {
		subnetsOut = append(
			subnetsOut,
			constructSubnet(
				subnetsIn[i]["id"],
				subnetsIn[i]["az"],
			),
		)
	}
	return
}

func constructSubnet(id string, az string) *ec2.Subnet {
	return &ec2.Subnet{
		SubnetId:         &id,
		AvailabilityZone: &az,
	}
}

func TestSubnetIDsinVPC(t *testing.T) {
	awsServices := NewFakeAWSServices()
	c, err := newAWSCloud(strings.NewReader("[global]"), awsServices)
	if err != nil {
		t.Errorf("Error building aws cloud: %v", err)
		return
	}

	vpcID := "vpc-deadbeef"
	vpc := &ec2.Vpc{
		VpcId: &vpcID,
	}

	// test with 3 subnets from 3 different AZs
	subnets := make(map[int]map[string]string)
	subnets[0] = make(map[string]string)
	subnets[0]["id"] = "subnet-a0000001"
	subnets[0]["az"] = "af-south-1a"
	subnets[1] = make(map[string]string)
	subnets[1]["id"] = "subnet-b0000001"
	subnets[1]["az"] = "af-south-1b"
	subnets[2] = make(map[string]string)
	subnets[2]["id"] = "subnet-c0000001"
	subnets[2]["az"] = "af-south-1c"
	awsServices.ec2.Subnets = constructSubnets(subnets)

	result, err := c.listSubnetIDsinVPC(vpc)
	if err != nil {
		t.Errorf("Error listing subnets: %v", err)
		return
	}

	if len(result) != 3 {
		t.Errorf("Expected 3 subnets but got %d", len(result))
		return
	}

	result_set := make(map[string]bool)
	for _, v := range result {
		result_set[v] = true
	}

	for i := range subnets {
		if !result_set[subnets[i]["id"]] {
			t.Errorf("Expected subnet%d '%s' in result: %v", i, subnets[i]["id"], result)
			return
		}
	}

	// test with 4 subnets from 3 different AZs
	// add duplicate az subnet
	subnets[3] = make(map[string]string)
	subnets[3]["id"] = "subnet-c0000002"
	subnets[3]["az"] = "af-south-1c"
	awsServices.ec2.Subnets = constructSubnets(subnets)

	result, err = c.listSubnetIDsinVPC(vpc)
	if err != nil {
		t.Errorf("Error listing subnets: %v", err)
		return
	}

	if len(result) != 3 {
		t.Errorf("Expected 3 subnets but got %d", len(result))
		return
	}

}
