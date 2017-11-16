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

package aws

import (
	"fmt"
	"io"
	"reflect"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/ec2"
	"github.com/aws/aws-sdk-go/service/elb"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"k8s.io/apimachinery/pkg/types"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
)

const TestClusterId = "clusterid.test"
const TestClusterName = "testCluster"

type MockedFakeEC2 struct {
	*FakeEC2Impl
	mock.Mock
}

func (m *MockedFakeEC2) expectDescribeSecurityGroups(clusterId, groupName, clusterID string) {
	tags := []*ec2.Tag{
		{Key: aws.String(TagNameKubernetesClusterLegacy), Value: aws.String(clusterId)},
		{Key: aws.String(fmt.Sprintf("%s%s", TagNameKubernetesClusterPrefix, clusterId)), Value: aws.String(ResourceLifecycleOwned)},
	}

	m.On("DescribeSecurityGroups", &ec2.DescribeSecurityGroupsInput{Filters: []*ec2.Filter{
		newEc2Filter("group-name", groupName),
		newEc2Filter("vpc-id", ""),
	}}).Return([]*ec2.SecurityGroup{{Tags: tags}})
}

func (m *MockedFakeEC2) DescribeVolumes(request *ec2.DescribeVolumesInput) ([]*ec2.Volume, error) {
	args := m.Called(request)
	return args.Get(0).([]*ec2.Volume), nil
}

func (m *MockedFakeEC2) DescribeSecurityGroups(request *ec2.DescribeSecurityGroupsInput) ([]*ec2.SecurityGroup, error) {
	args := m.Called(request)
	return args.Get(0).([]*ec2.SecurityGroup), nil
}

type MockedFakeELB struct {
	*FakeELB
	mock.Mock
}

func (m *MockedFakeELB) DescribeLoadBalancers(input *elb.DescribeLoadBalancersInput) (*elb.DescribeLoadBalancersOutput, error) {
	args := m.Called(input)
	return args.Get(0).(*elb.DescribeLoadBalancersOutput), nil
}

func (m *MockedFakeELB) expectDescribeLoadBalancers(loadBalancerName string) {
	m.On("DescribeLoadBalancers", &elb.DescribeLoadBalancersInput{LoadBalancerNames: []*string{aws.String(loadBalancerName)}}).Return(&elb.DescribeLoadBalancersOutput{
		LoadBalancerDescriptions: []*elb.LoadBalancerDescription{{}},
	})
}

func TestReadAWSCloudConfig(t *testing.T) {
	tests := []struct {
		name string

		reader io.Reader
		aws    Services

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
			strings.NewReader("[global]\n"), newMockedFakeAWSServices(TestClusterId).WithAz(""),
			true, "",
		},
		{
			"No zone in config, metadata has zone",
			strings.NewReader("[global]\n"), newMockedFakeAWSServices(TestClusterId),
			false, "us-east-1a",
		},
		{
			"Zone in config should take precedence over metadata",
			strings.NewReader("[global]\nzone = eu-west-1a"), newMockedFakeAWSServices(TestClusterId),
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

func TestNewAWSCloud(t *testing.T) {
	tests := []struct {
		name string

		reader      io.Reader
		awsServices Services

		expectError bool
		region      string
	}{
		{
			"No config reader",
			nil, newMockedFakeAWSServices(TestClusterId).WithAz(""),
			true, "",
		},
		{
			"Config specifies valid zone",
			strings.NewReader("[global]\nzone = eu-west-1a"), newMockedFakeAWSServices(TestClusterId),
			false, "eu-west-1",
		},
		{
			"Gets zone from metadata when not in config",
			strings.NewReader("[global]\n"),
			newMockedFakeAWSServices(TestClusterId),
			false, "us-east-1",
		},
		{
			"No zone in config or metadata",
			strings.NewReader("[global]\n"),
			newMockedFakeAWSServices(TestClusterId).WithAz(""),
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
			} else if c.region != test.region {
				t.Errorf("Incorrect region value (%s vs %s) for case: %s",
					c.region, test.region, test.name)
			}
		}
	}
}

func mockInstancesResp(selfInstance *ec2.Instance, instances []*ec2.Instance) (*Cloud, *FakeAWSServices) {
	awsServices := newMockedFakeAWSServices(TestClusterId)
	awsServices.instances = instances
	awsServices.selfInstance = selfInstance
	awsCloud, err := newAWSCloud(nil, awsServices)
	if err != nil {
		panic(err)
	}
	return awsCloud, awsServices
}

func mockAvailabilityZone(availabilityZone string) *Cloud {
	awsServices := newMockedFakeAWSServices(TestClusterId).WithAz(availabilityZone)
	awsCloud, err := newAWSCloud(nil, awsServices)
	if err != nil {
		panic(err)
	}
	return awsCloud
}

func testHasNodeAddress(t *testing.T, addrs []v1.NodeAddress, addressType v1.NodeAddressType, address string) {
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
	var instance2 ec2.Instance

	//0
	instance0.InstanceId = aws.String("i-0")
	instance0.PrivateDnsName = aws.String("instance-same.ec2.internal")
	instance0.PrivateIpAddress = aws.String("192.168.0.1")
	instance0.PublicDnsName = aws.String("instance-same.ec2.external")
	instance0.PublicIpAddress = aws.String("1.2.3.4")
	instance0.NetworkInterfaces = []*ec2.InstanceNetworkInterface{
		{
			Status: aws.String(ec2.NetworkInterfaceStatusInUse),
			PrivateIpAddresses: []*ec2.InstancePrivateIpAddress{
				{
					PrivateIpAddress: aws.String("192.168.0.1"),
				},
			},
		},
	}
	instance0.InstanceType = aws.String("c3.large")
	instance0.Placement = &ec2.Placement{AvailabilityZone: aws.String("us-east-1a")}
	state0 := ec2.InstanceState{
		Name: aws.String("running"),
	}
	instance0.State = &state0

	//1
	instance1.InstanceId = aws.String("i-1")
	instance1.PrivateDnsName = aws.String("instance-same.ec2.internal")
	instance1.PrivateIpAddress = aws.String("192.168.0.2")
	instance1.InstanceType = aws.String("c3.large")
	instance1.Placement = &ec2.Placement{AvailabilityZone: aws.String("us-east-1a")}
	state1 := ec2.InstanceState{
		Name: aws.String("running"),
	}
	instance1.State = &state1

	//2
	instance2.InstanceId = aws.String("i-2")
	instance2.PrivateDnsName = aws.String("instance-other.ec2.internal")
	instance2.PrivateIpAddress = aws.String("192.168.0.1")
	instance2.PublicIpAddress = aws.String("1.2.3.4")
	instance2.InstanceType = aws.String("c3.large")
	instance2.Placement = &ec2.Placement{AvailabilityZone: aws.String("us-east-1a")}
	state2 := ec2.InstanceState{
		Name: aws.String("running"),
	}
	instance2.State = &state2

	instances := []*ec2.Instance{&instance0, &instance1, &instance2}

	aws1, _ := mockInstancesResp(&instance0, []*ec2.Instance{&instance0})
	_, err1 := aws1.NodeAddresses("instance-mismatch.ec2.internal")
	if err1 == nil {
		t.Errorf("Should error when no instance found")
	}

	aws2, _ := mockInstancesResp(&instance2, instances)
	_, err2 := aws2.NodeAddresses("instance-same.ec2.internal")
	if err2 == nil {
		t.Errorf("Should error when multiple instances found")
	}

	aws3, _ := mockInstancesResp(&instance0, instances[0:1])
	// change node name so it uses the instance instead of metadata
	aws3.selfAWSInstance.nodeName = "foo"
	addrs3, err3 := aws3.NodeAddresses("instance-same.ec2.internal")
	if err3 != nil {
		t.Errorf("Should not error when instance found")
	}
	if len(addrs3) != 4 {
		t.Errorf("Should return exactly 4 NodeAddresses")
	}
	testHasNodeAddress(t, addrs3, v1.NodeInternalIP, "192.168.0.1")
	testHasNodeAddress(t, addrs3, v1.NodeExternalIP, "1.2.3.4")
	testHasNodeAddress(t, addrs3, v1.NodeExternalDNS, "instance-same.ec2.external")
	testHasNodeAddress(t, addrs3, v1.NodeInternalDNS, "instance-same.ec2.internal")
}

func TestNodeAddressesWithMetadata(t *testing.T) {
	var instance ec2.Instance

	instanceName := "instance.ec2.internal"
	instance.InstanceId = aws.String("i-0")
	instance.PrivateDnsName = &instanceName
	instance.PublicIpAddress = aws.String("2.3.4.5")
	instance.InstanceType = aws.String("c3.large")
	instance.Placement = &ec2.Placement{AvailabilityZone: aws.String("us-east-1a")}
	state := ec2.InstanceState{
		Name: aws.String("running"),
	}
	instance.State = &state

	instances := []*ec2.Instance{&instance}
	awsCloud, awsServices := mockInstancesResp(&instance, instances)

	awsServices.networkInterfacesMacs = []string{"0a:26:89:f3:9c:f6", "0a:77:64:c4:6a:48"}
	awsServices.networkInterfacesPrivateIPs = [][]string{{"192.168.0.1"}, {"192.168.0.2"}}
	addrs, err := awsCloud.NodeAddresses("")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	testHasNodeAddress(t, addrs, v1.NodeInternalIP, "192.168.0.1")
	testHasNodeAddress(t, addrs, v1.NodeInternalIP, "192.168.0.2")
	testHasNodeAddress(t, addrs, v1.NodeExternalIP, "2.3.4.5")
}

func TestGetRegion(t *testing.T) {
	aws := mockAvailabilityZone("us-west-2e")
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
	awsServices := newMockedFakeAWSServices(TestClusterId)
	c, err := newAWSCloud(strings.NewReader("[global]"), awsServices)
	if err != nil {
		t.Errorf("Error building aws cloud: %v", err)
		return
	}
	vpcID, err := c.findVPCID()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if vpcID != "vpc-mac0" {
		t.Errorf("Unexpected vpcID: %s", vpcID)
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

func constructRouteTables(routeTablesIn map[string]bool) (routeTablesOut []*ec2.RouteTable) {
	routeTablesOut = append(routeTablesOut,
		&ec2.RouteTable{
			Associations: []*ec2.RouteTableAssociation{{Main: aws.Bool(true)}},
			Routes: []*ec2.Route{{
				DestinationCidrBlock: aws.String("0.0.0.0/0"),
				GatewayId:            aws.String("igw-main"),
			}},
		})

	for subnetID := range routeTablesIn {
		routeTablesOut = append(
			routeTablesOut,
			constructRouteTable(
				subnetID,
				routeTablesIn[subnetID],
			),
		)
	}
	return
}

func constructRouteTable(subnetID string, public bool) *ec2.RouteTable {
	var gatewayID string
	if public {
		gatewayID = "igw-" + subnetID[len(subnetID)-8:8]
	} else {
		gatewayID = "vgw-" + subnetID[len(subnetID)-8:8]
	}
	return &ec2.RouteTable{
		Associations: []*ec2.RouteTableAssociation{{SubnetId: aws.String(subnetID)}},
		Routes: []*ec2.Route{{
			DestinationCidrBlock: aws.String("0.0.0.0/0"),
			GatewayId:            aws.String(gatewayID),
		}},
	}
}

func TestSubnetIDsinVPC(t *testing.T) {
	awsServices := newMockedFakeAWSServices(TestClusterId)
	c, err := newAWSCloud(strings.NewReader("[global]"), awsServices)
	if err != nil {
		t.Errorf("Error building aws cloud: %v", err)
		return
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
	constructedSubnets := constructSubnets(subnets)
	awsServices.ec2.RemoveSubnets()
	for _, subnet := range constructedSubnets {
		awsServices.ec2.CreateSubnet(subnet)
	}

	routeTables := map[string]bool{
		"subnet-a0000001": true,
		"subnet-b0000001": true,
		"subnet-c0000001": true,
	}
	constructedRouteTables := constructRouteTables(routeTables)
	awsServices.ec2.RemoveRouteTables()
	for _, rt := range constructedRouteTables {
		awsServices.ec2.CreateRouteTable(rt)
	}

	result, err := c.findELBSubnets(false)
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

	// test implicit routing table - when subnets are not explicitly linked to a table they should use main
	constructedRouteTables = constructRouteTables(map[string]bool{})
	awsServices.ec2.RemoveRouteTables()
	for _, rt := range constructedRouteTables {
		awsServices.ec2.CreateRouteTable(rt)
	}

	result, err = c.findELBSubnets(false)
	if err != nil {
		t.Errorf("Error listing subnets: %v", err)
		return
	}

	if len(result) != 3 {
		t.Errorf("Expected 3 subnets but got %d", len(result))
		return
	}

	result_set = make(map[string]bool)
	for _, v := range result {
		result_set[v] = true
	}

	for i := range subnets {
		if !result_set[subnets[i]["id"]] {
			t.Errorf("Expected subnet%d '%s' in result: %v", i, subnets[i]["id"], result)
			return
		}
	}

	// Test with 5 subnets from 3 different AZs.
	// Add 2 duplicate AZ subnets lexicographically chosen one is the middle element in array to
	// check that we both choose the correct entry when it comes after and before another element
	// in the same AZ.
	subnets[3] = make(map[string]string)
	subnets[3]["id"] = "subnet-c0000000"
	subnets[3]["az"] = "af-south-1c"
	subnets[4] = make(map[string]string)
	subnets[4]["id"] = "subnet-c0000002"
	subnets[4]["az"] = "af-south-1c"
	constructedSubnets = constructSubnets(subnets)
	awsServices.ec2.RemoveSubnets()
	for _, subnet := range constructedSubnets {
		awsServices.ec2.CreateSubnet(subnet)
	}
	routeTables["subnet-c0000000"] = true
	routeTables["subnet-c0000002"] = true
	constructedRouteTables = constructRouteTables(routeTables)
	awsServices.ec2.RemoveRouteTables()
	for _, rt := range constructedRouteTables {
		awsServices.ec2.CreateRouteTable(rt)
	}

	result, err = c.findELBSubnets(false)
	if err != nil {
		t.Errorf("Error listing subnets: %v", err)
		return
	}

	if len(result) != 3 {
		t.Errorf("Expected 3 subnets but got %d", len(result))
		return
	}

	expected := []*string{aws.String("subnet-a0000001"), aws.String("subnet-b0000001"), aws.String("subnet-c0000000")}
	for _, s := range result {
		if !contains(expected, s) {
			t.Errorf("Unexpected subnet '%s' found", s)
			return
		}
	}

	delete(routeTables, "subnet-c0000002")

	// test with 6 subnets from 3 different AZs
	// with 3 private subnets
	subnets[4] = make(map[string]string)
	subnets[4]["id"] = "subnet-d0000001"
	subnets[4]["az"] = "af-south-1a"
	subnets[5] = make(map[string]string)
	subnets[5]["id"] = "subnet-d0000002"
	subnets[5]["az"] = "af-south-1b"

	constructedSubnets = constructSubnets(subnets)
	awsServices.ec2.RemoveSubnets()
	for _, subnet := range constructedSubnets {
		awsServices.ec2.CreateSubnet(subnet)
	}

	routeTables["subnet-a0000001"] = false
	routeTables["subnet-b0000001"] = false
	routeTables["subnet-c0000001"] = false
	routeTables["subnet-c0000000"] = true
	routeTables["subnet-d0000001"] = true
	routeTables["subnet-d0000002"] = true
	constructedRouteTables = constructRouteTables(routeTables)
	awsServices.ec2.RemoveRouteTables()
	for _, rt := range constructedRouteTables {
		awsServices.ec2.CreateRouteTable(rt)
	}
	result, err = c.findELBSubnets(false)
	if err != nil {
		t.Errorf("Error listing subnets: %v", err)
		return
	}

	if len(result) != 3 {
		t.Errorf("Expected 3 subnets but got %d", len(result))
		return
	}

	expected = []*string{aws.String("subnet-c0000000"), aws.String("subnet-d0000001"), aws.String("subnet-d0000002")}
	for _, s := range result {
		if !contains(expected, s) {
			t.Errorf("Unexpected subnet '%s' found", s)
			return
		}
	}
}

func TestIpPermissionExistsHandlesMultipleGroupIds(t *testing.T) {
	oldIpPermission := ec2.IpPermission{
		UserIdGroupPairs: []*ec2.UserIdGroupPair{
			{GroupId: aws.String("firstGroupId")},
			{GroupId: aws.String("secondGroupId")},
			{GroupId: aws.String("thirdGroupId")},
		},
	}

	existingIpPermission := ec2.IpPermission{
		UserIdGroupPairs: []*ec2.UserIdGroupPair{
			{GroupId: aws.String("secondGroupId")},
		},
	}

	newIpPermission := ec2.IpPermission{
		UserIdGroupPairs: []*ec2.UserIdGroupPair{
			{GroupId: aws.String("fourthGroupId")},
		},
	}

	equals := ipPermissionExists(&existingIpPermission, &oldIpPermission, false)
	if !equals {
		t.Errorf("Should have been considered equal since first is in the second array of groups")
	}

	equals = ipPermissionExists(&newIpPermission, &oldIpPermission, false)
	if equals {
		t.Errorf("Should have not been considered equal since first is not in the second array of groups")
	}

	// The first pair matches, but the second does not
	newIpPermission2 := ec2.IpPermission{
		UserIdGroupPairs: []*ec2.UserIdGroupPair{
			{GroupId: aws.String("firstGroupId")},
			{GroupId: aws.String("fourthGroupId")},
		},
	}
	equals = ipPermissionExists(&newIpPermission2, &oldIpPermission, false)
	if equals {
		t.Errorf("Should have not been considered equal since first is not in the second array of groups")
	}
}

func TestIpPermissionExistsHandlesRangeSubsets(t *testing.T) {
	// Two existing scenarios we'll test against
	emptyIpPermission := ec2.IpPermission{}

	oldIpPermission := ec2.IpPermission{
		IpRanges: []*ec2.IpRange{
			{CidrIp: aws.String("10.0.0.0/8")},
			{CidrIp: aws.String("192.168.1.0/24")},
		},
	}

	// Two already existing ranges and a new one
	existingIpPermission := ec2.IpPermission{
		IpRanges: []*ec2.IpRange{
			{CidrIp: aws.String("10.0.0.0/8")},
		},
	}
	existingIpPermission2 := ec2.IpPermission{
		IpRanges: []*ec2.IpRange{
			{CidrIp: aws.String("192.168.1.0/24")},
		},
	}

	newIpPermission := ec2.IpPermission{
		IpRanges: []*ec2.IpRange{
			{CidrIp: aws.String("172.16.0.0/16")},
		},
	}

	exists := ipPermissionExists(&emptyIpPermission, &emptyIpPermission, false)
	if !exists {
		t.Errorf("Should have been considered existing since we're comparing a range array against itself")
	}
	exists = ipPermissionExists(&oldIpPermission, &oldIpPermission, false)
	if !exists {
		t.Errorf("Should have been considered existing since we're comparing a range array against itself")
	}

	exists = ipPermissionExists(&existingIpPermission, &oldIpPermission, false)
	if !exists {
		t.Errorf("Should have been considered existing since 10.* is in oldIpPermission's array of ranges")
	}
	exists = ipPermissionExists(&existingIpPermission2, &oldIpPermission, false)
	if !exists {
		t.Errorf("Should have been considered existing since 192.* is in oldIpPermission2's array of ranges")
	}

	exists = ipPermissionExists(&newIpPermission, &emptyIpPermission, false)
	if exists {
		t.Errorf("Should have not been considered existing since we compared against a missing array of ranges")
	}
	exists = ipPermissionExists(&newIpPermission, &oldIpPermission, false)
	if exists {
		t.Errorf("Should have not been considered existing since 172.* is not in oldIpPermission's array of ranges")
	}
}

func TestIpPermissionExistsHandlesMultipleGroupIdsWithUserIds(t *testing.T) {
	oldIpPermission := ec2.IpPermission{
		UserIdGroupPairs: []*ec2.UserIdGroupPair{
			{GroupId: aws.String("firstGroupId"), UserId: aws.String("firstUserId")},
			{GroupId: aws.String("secondGroupId"), UserId: aws.String("secondUserId")},
			{GroupId: aws.String("thirdGroupId"), UserId: aws.String("thirdUserId")},
		},
	}

	existingIpPermission := ec2.IpPermission{
		UserIdGroupPairs: []*ec2.UserIdGroupPair{
			{GroupId: aws.String("secondGroupId"), UserId: aws.String("secondUserId")},
		},
	}

	newIpPermission := ec2.IpPermission{
		UserIdGroupPairs: []*ec2.UserIdGroupPair{
			{GroupId: aws.String("secondGroupId"), UserId: aws.String("anotherUserId")},
		},
	}

	equals := ipPermissionExists(&existingIpPermission, &oldIpPermission, true)
	if !equals {
		t.Errorf("Should have been considered equal since first is in the second array of groups")
	}

	equals = ipPermissionExists(&newIpPermission, &oldIpPermission, true)
	if equals {
		t.Errorf("Should have not been considered equal since first is not in the second array of groups")
	}
}

func TestFindInstanceByNodeNameExcludesTerminatedInstances(t *testing.T) {
	awsServices := newMockedFakeAWSServices(TestClusterId)

	nodeName := types.NodeName("my-dns.internal")

	var tag ec2.Tag
	tag.Key = aws.String(TagNameKubernetesClusterLegacy)
	tag.Value = aws.String(TestClusterId)
	tags := []*ec2.Tag{&tag}

	var runningInstance ec2.Instance
	runningInstance.InstanceId = aws.String("i-running")
	runningInstance.PrivateDnsName = aws.String(string(nodeName))
	runningInstance.State = &ec2.InstanceState{Code: aws.Int64(16), Name: aws.String("running")}
	runningInstance.Tags = tags

	var terminatedInstance ec2.Instance
	terminatedInstance.InstanceId = aws.String("i-terminated")
	terminatedInstance.PrivateDnsName = aws.String(string(nodeName))
	terminatedInstance.State = &ec2.InstanceState{Code: aws.Int64(48), Name: aws.String("terminated")}
	terminatedInstance.Tags = tags

	instances := []*ec2.Instance{&terminatedInstance, &runningInstance}
	awsServices.instances = append(awsServices.instances, instances...)

	c, err := newAWSCloud(strings.NewReader("[global]"), awsServices)
	if err != nil {
		t.Errorf("Error building aws cloud: %v", err)
		return
	}

	instance, err := c.findInstanceByNodeName(nodeName)

	if err != nil {
		t.Errorf("Failed to find instance: %v", err)
		return
	}

	if *instance.InstanceId != "i-running" {
		t.Errorf("Expected running instance but got %v", *instance.InstanceId)
	}
}

func TestGetInstanceByNodeNameBatching(t *testing.T) {
	awsServices := newMockedFakeAWSServices(TestClusterId)
	c, err := newAWSCloud(strings.NewReader("[global]"), awsServices)
	assert.Nil(t, err, "Error building aws cloud: %v", err)
	var tag ec2.Tag
	tag.Key = aws.String(TagNameKubernetesClusterPrefix + TestClusterId)
	tag.Value = aws.String("")
	tags := []*ec2.Tag{&tag}
	nodeNames := []string{}
	for i := 0; i < 200; i++ {
		nodeName := fmt.Sprintf("ip-171-20-42-%d.ec2.internal", i)
		nodeNames = append(nodeNames, nodeName)
		ec2Instance := &ec2.Instance{}
		instanceId := fmt.Sprintf("i-abcedf%d", i)
		ec2Instance.InstanceId = aws.String(instanceId)
		ec2Instance.PrivateDnsName = aws.String(nodeName)
		ec2Instance.State = &ec2.InstanceState{Code: aws.Int64(48), Name: aws.String("running")}
		ec2Instance.Tags = tags
		awsServices.instances = append(awsServices.instances, ec2Instance)

	}

	instances, err := c.getInstancesByNodeNames(nodeNames)
	assert.NotEmpty(t, instances)
	assert.Equal(t, 200, len(instances), "Expected 200 but got less")
}

func TestGetVolumeLabels(t *testing.T) {
	awsServices := newMockedFakeAWSServices(TestClusterId)
	c, err := newAWSCloud(strings.NewReader("[global]"), awsServices)
	assert.Nil(t, err, "Error building aws cloud: %v", err)
	volumeId := awsVolumeID("vol-VolumeId")
	expectedVolumeRequest := &ec2.DescribeVolumesInput{VolumeIds: []*string{volumeId.awsString()}}
	awsServices.ec2.(*MockedFakeEC2).On("DescribeVolumes", expectedVolumeRequest).Return([]*ec2.Volume{
		{
			VolumeId:         volumeId.awsString(),
			AvailabilityZone: aws.String("us-east-1a"),
		},
	})

	labels, err := c.GetVolumeLabels(KubernetesVolumeID("aws:///" + string(volumeId)))

	assert.Nil(t, err, "Error creating Volume %v", err)
	assert.Equal(t, map[string]string{
		kubeletapis.LabelZoneFailureDomain: "us-east-1a",
		kubeletapis.LabelZoneRegion:        "us-east-1"}, labels)
	awsServices.ec2.(*MockedFakeEC2).AssertExpectations(t)
}

func TestDescribeLoadBalancerOnDelete(t *testing.T) {
	awsServices := newMockedFakeAWSServices(TestClusterId)
	c, _ := newAWSCloud(strings.NewReader("[global]"), awsServices)
	awsServices.elb.(*MockedFakeELB).expectDescribeLoadBalancers("aid")

	c.EnsureLoadBalancerDeleted(TestClusterName, &v1.Service{ObjectMeta: metav1.ObjectMeta{Name: "myservice", UID: "id"}})
}

func TestDescribeLoadBalancerOnUpdate(t *testing.T) {
	awsServices := newMockedFakeAWSServices(TestClusterId)
	c, _ := newAWSCloud(strings.NewReader("[global]"), awsServices)
	awsServices.elb.(*MockedFakeELB).expectDescribeLoadBalancers("aid")

	c.UpdateLoadBalancer(TestClusterName, &v1.Service{ObjectMeta: metav1.ObjectMeta{Name: "myservice", UID: "id"}}, []*v1.Node{})
}

func TestDescribeLoadBalancerOnGet(t *testing.T) {
	awsServices := newMockedFakeAWSServices(TestClusterId)
	c, _ := newAWSCloud(strings.NewReader("[global]"), awsServices)
	awsServices.elb.(*MockedFakeELB).expectDescribeLoadBalancers("aid")

	c.GetLoadBalancer(TestClusterName, &v1.Service{ObjectMeta: metav1.ObjectMeta{Name: "myservice", UID: "id"}})
}

func TestDescribeLoadBalancerOnEnsure(t *testing.T) {
	awsServices := newMockedFakeAWSServices(TestClusterId)
	c, _ := newAWSCloud(strings.NewReader("[global]"), awsServices)
	awsServices.elb.(*MockedFakeELB).expectDescribeLoadBalancers("aid")

	c.EnsureLoadBalancer(TestClusterName, &v1.Service{ObjectMeta: metav1.ObjectMeta{Name: "myservice", UID: "id"}}, []*v1.Node{})
}

func TestBuildListener(t *testing.T) {
	tests := []struct {
		name string

		lbPort                    int64
		portName                  string
		instancePort              int64
		backendProtocolAnnotation string
		certAnnotation            string
		sslPortAnnotation         string

		expectError      bool
		lbProtocol       string
		instanceProtocol string
		certID           string
	}{
		{
			"No cert or BE protocol annotation, passthrough",
			80, "", 7999, "", "", "",
			false, "tcp", "tcp", "",
		},
		{
			"Cert annotation without BE protocol specified, SSL->TCP",
			80, "", 8000, "", "cert", "",
			false, "ssl", "tcp", "cert",
		},
		{
			"BE protocol without cert annotation, passthrough",
			443, "", 8001, "https", "", "",
			false, "tcp", "tcp", "",
		},
		{
			"Invalid cert annotation, bogus backend protocol",
			443, "", 8002, "bacon", "foo", "",
			true, "tcp", "tcp", "",
		},
		{
			"Invalid cert annotation, protocol followed by equal sign",
			443, "", 8003, "http=", "=", "",
			true, "tcp", "tcp", "",
		},
		{
			"HTTPS->HTTPS",
			443, "", 8004, "https", "cert", "",
			false, "https", "https", "cert",
		},
		{
			"HTTPS->HTTP",
			443, "", 8005, "http", "cert", "",
			false, "https", "http", "cert",
		},
		{
			"SSL->SSL",
			443, "", 8006, "ssl", "cert", "",
			false, "ssl", "ssl", "cert",
		},
		{
			"SSL->TCP",
			443, "", 8007, "tcp", "cert", "",
			false, "ssl", "tcp", "cert",
		},
		{
			"Port in whitelist",
			1234, "", 8008, "tcp", "cert", "1234,5678",
			false, "ssl", "tcp", "cert",
		},
		{
			"Port not in whitelist, passthrough",
			443, "", 8009, "tcp", "cert", "1234,5678",
			false, "tcp", "tcp", "",
		},
		{
			"Named port in whitelist",
			1234, "bar", 8010, "tcp", "cert", "foo,bar",
			false, "ssl", "tcp", "cert",
		},
		{
			"Named port not in whitelist, passthrough",
			443, "", 8011, "tcp", "cert", "foo,bar",
			false, "tcp", "tcp", "",
		},
		{
			"HTTP->HTTP",
			80, "", 8012, "http", "", "",
			false, "http", "http", "",
		},
	}

	for _, test := range tests {
		t.Logf("Running test case %s", test.name)
		annotations := make(map[string]string)
		if test.backendProtocolAnnotation != "" {
			annotations[ServiceAnnotationLoadBalancerBEProtocol] = test.backendProtocolAnnotation
		}
		if test.certAnnotation != "" {
			annotations[ServiceAnnotationLoadBalancerCertificate] = test.certAnnotation
		}
		ports := getPortSets(test.sslPortAnnotation)
		l, err := buildListener(v1.ServicePort{
			NodePort: int32(test.instancePort),
			Port:     int32(test.lbPort),
			Name:     test.portName,
			Protocol: v1.Protocol("tcp"),
		}, annotations, ports)
		if test.expectError {
			if err == nil {
				t.Errorf("Should error for case %s", test.name)
			}
		} else {
			if err != nil {
				t.Errorf("Should succeed for case: %s, got %v", test.name, err)
			} else {
				var cert *string
				if test.certID != "" {
					cert = &test.certID
				}
				expected := &elb.Listener{
					InstancePort:     &test.instancePort,
					InstanceProtocol: &test.instanceProtocol,
					LoadBalancerPort: &test.lbPort,
					Protocol:         &test.lbProtocol,
					SSLCertificateId: cert,
				}
				if !reflect.DeepEqual(l, expected) {
					t.Errorf("Incorrect listener (%v vs expected %v) for case: %s",
						l, expected, test.name)
				}
			}
		}
	}
}

func TestProxyProtocolEnabled(t *testing.T) {
	policies := sets.NewString(ProxyProtocolPolicyName, "FooBarFoo")
	fakeBackend := &elb.BackendServerDescription{
		InstancePort: aws.Int64(80),
		PolicyNames:  stringSetToPointers(policies),
	}
	result := proxyProtocolEnabled(fakeBackend)
	assert.True(t, result, "expected to find %s in %s", ProxyProtocolPolicyName, policies)

	policies = sets.NewString("FooBarFoo")
	fakeBackend = &elb.BackendServerDescription{
		InstancePort: aws.Int64(80),
		PolicyNames: []*string{
			aws.String("FooBarFoo"),
		},
	}
	result = proxyProtocolEnabled(fakeBackend)
	assert.False(t, result, "did not expect to find %s in %s", ProxyProtocolPolicyName, policies)

	policies = sets.NewString()
	fakeBackend = &elb.BackendServerDescription{
		InstancePort: aws.Int64(80),
	}
	result = proxyProtocolEnabled(fakeBackend)
	assert.False(t, result, "did not expect to find %s in %s", ProxyProtocolPolicyName, policies)
}

func TestGetLoadBalancerAdditionalTags(t *testing.T) {
	tagTests := []struct {
		Annotations map[string]string
		Tags        map[string]string
	}{
		{
			Annotations: map[string]string{
				ServiceAnnotationLoadBalancerAdditionalTags: "Key=Val",
			},
			Tags: map[string]string{
				"Key": "Val",
			},
		},
		{
			Annotations: map[string]string{
				ServiceAnnotationLoadBalancerAdditionalTags: "Key1=Val1, Key2=Val2",
			},
			Tags: map[string]string{
				"Key1": "Val1",
				"Key2": "Val2",
			},
		},
		{
			Annotations: map[string]string{
				ServiceAnnotationLoadBalancerAdditionalTags: "Key1=, Key2=Val2",
				"anotherKey":                                "anotherValue",
			},
			Tags: map[string]string{
				"Key1": "",
				"Key2": "Val2",
			},
		},
		{
			Annotations: map[string]string{
				"Nothing": "Key1=, Key2=Val2, Key3",
			},
			Tags: map[string]string{},
		},
		{
			Annotations: map[string]string{
				ServiceAnnotationLoadBalancerAdditionalTags: "K=V K1=V2,Key1========, =====, ======Val, =Val, , 234,",
			},
			Tags: map[string]string{
				"K":    "V K1",
				"Key1": "",
				"234":  "",
			},
		},
	}

	for _, tagTest := range tagTests {
		result := getLoadBalancerAdditionalTags(tagTest.Annotations)
		for k, v := range result {
			if len(result) != len(tagTest.Tags) {
				t.Errorf("incorrect expected length: %v != %v", result, tagTest.Tags)
				continue
			}
			if tagTest.Tags[k] != v {
				t.Errorf("%s != %s", tagTest.Tags[k], v)
				continue
			}
		}
	}
}

func TestLBExtraSecurityGroupsAnnotation(t *testing.T) {
	awsServices := newMockedFakeAWSServices(TestClusterId)
	c, _ := newAWSCloud(strings.NewReader("[global]"), awsServices)

	sg1 := "sg-000001"
	sg2 := "sg-000002"

	tests := []struct {
		name string

		extraSGsAnnotation string
		expectedSGs        []string
	}{
		{"No extra SG annotation", "", []string{}},
		{"Empty extra SGs specified", ", ,,", []string{}},
		{"SG specified", sg1, []string{sg1}},
		{"Multiple SGs specified", fmt.Sprintf("%s, %s", sg1, sg2), []string{sg1, sg2}},
	}

	awsServices.ec2.(*MockedFakeEC2).expectDescribeSecurityGroups(TestClusterId, "k8s-elb-aid", "cluster.test")

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			serviceName := types.NamespacedName{Namespace: "default", Name: "myservice"}

			sgList, err := c.buildELBSecurityGroupList(serviceName, "aid", test.extraSGsAnnotation)
			assert.NoError(t, err, "buildELBSecurityGroupList failed")
			extraSGs := sgList[1:]
			assert.True(t, sets.NewString(test.expectedSGs...).Equal(sets.NewString(extraSGs...)),
				"Security Groups expected=%q , returned=%q", test.expectedSGs, extraSGs)
		})
	}
}

func newMockedFakeAWSServices(id string) *FakeAWSServices {
	s := NewFakeAWSServices(id)
	s.ec2 = &MockedFakeEC2{FakeEC2Impl: s.ec2.(*FakeEC2Impl)}
	s.elb = &MockedFakeELB{FakeELB: s.elb.(*FakeELB)}
	return s
}
