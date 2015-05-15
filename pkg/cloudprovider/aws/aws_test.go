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
	"io"
	"reflect"
	"strings"
	"testing"

	"github.com/awslabs/aws-sdk-go/aws"
	"github.com/awslabs/aws-sdk-go/aws/credentials"
	"github.com/awslabs/aws-sdk-go/service/ec2"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
)

func TestReadAWSCloudConfig(t *testing.T) {
	tests := []struct {
		name string

		reader   io.Reader
		metadata AWSMetadata

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
			strings.NewReader("[global]\n"), &FakeMetadata{},
			true, "",
		},
		{
			"No zone in config, metadata has zone",
			strings.NewReader("[global]\n"), &FakeMetadata{availabilityZone: "eu-west-1a"},
			false, "eu-west-1a",
		},
		{
			"Zone in config should take precedence over metadata",
			strings.NewReader("[global]\nzone = us-east-1a"), &FakeMetadata{availabilityZone: "eu-west-1a"},
			false, "us-east-1a",
		},
	}

	for _, test := range tests {
		t.Logf("Running test case %s", test.name)
		cfg, err := readAWSCloudConfig(test.reader, test.metadata)
		if test.expectError {
			if err == nil {
				t.Errorf("Should error for case %s", test.name)
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
	fakeAuthFunc := func() (creds *credentials.Credentials) {
		return credentials.NewStaticCredentials("", "", "")
	}

	tests := []struct {
		name string

		reader   io.Reader
		authFunc AuthFunc
		metadata AWSMetadata

		expectError bool
		zone        string
	}{
		{
			"No config reader",
			nil, fakeAuthFunc, &FakeMetadata{},
			true, "",
		},
		{
			"Config specified invalid zone",
			strings.NewReader("[global]\nzone = blahonga"), fakeAuthFunc, &FakeMetadata{},
			true, "",
		},
		{
			"Config specifies valid zone",
			strings.NewReader("[global]\nzone = eu-west-1a"), fakeAuthFunc, &FakeMetadata{},
			false, "eu-west-1a",
		},
		{
			"Gets zone from metadata when not in config",

			strings.NewReader("[global]\n"),
			fakeAuthFunc,
			&FakeMetadata{availabilityZone: "us-east-1a"},

			false, "us-east-1a",
		},
		{
			"No zone in config or metadata",
			strings.NewReader("[global]\n"), fakeAuthFunc, &FakeMetadata{},
			true, "",
		},
	}

	for _, test := range tests {
		t.Logf("Running test case %s", test.name)
		c, err := newAWSCloud(test.reader, test.authFunc, test.metadata)
		if test.expectError {
			if err == nil {
				t.Errorf("Should error for case %s", test.name)
			}
		} else {
			if err != nil {
				t.Errorf("Should succeed for case: %s", test.name)
			}
			if c.availabilityZone != test.zone {
				t.Errorf("Incorrect zone value (%s vs %s) for case: %s",
					c.availabilityZone, test.zone, test.name)
			}
		}
	}
}

type FakeEC2 struct {
	instances []*ec2.Instance
}

func contains(haystack []string, needle string) bool {
	for _, s := range haystack {
		if needle == s {
			return true
		}
	}
	return false
}

func (self *FakeEC2) Instances(instanceIds []string, filter *ec2InstanceFilter) (instances []*ec2.Instance, err error) {
	matches := []*ec2.Instance{}
	for _, instance := range self.instances {
		if filter != nil && !filter.Matches(instance) {
			continue
		}
		if instanceIds != nil && !contains(instanceIds, *instance.InstanceID) {
			continue
		}
		matches = append(matches, instance)
	}

	return matches, nil
}

type FakeMetadata struct {
	availabilityZone string
	instanceId       string
}

func (self *FakeMetadata) GetMetaData(key string) ([]byte, error) {
	if key == "placement/availability-zone" {
		return []byte(self.availabilityZone), nil
	} else if key == "instance-id" {
		return []byte(self.instanceId), nil
	} else {
		return nil, nil
	}
}

func (ec2 *FakeEC2) AttachVolume(volumeID, instanceId, mountDevice string) (resp *ec2.VolumeAttachment, err error) {
	panic("Not implemented")
}

func (ec2 *FakeEC2) DetachVolume(volumeID, instanceId, mountDevice string) (resp *ec2.VolumeAttachment, err error) {
	panic("Not implemented")
}

func (ec2 *FakeEC2) Volumes(volumeIDs []string, filter *ec2.Filter) (resp *ec2.DescribeVolumesOutput, err error) {
	panic("Not implemented")
}

func (ec2 *FakeEC2) CreateVolume(request *ec2.CreateVolumeInput) (resp *ec2.Volume, err error) {
	panic("Not implemented")
}

func (ec2 *FakeEC2) DeleteVolume(volumeID string) (resp *ec2.DeleteVolumeOutput, err error) {
	panic("Not implemented")
}

func mockInstancesResp(instances []*ec2.Instance) (aws *AWSCloud) {
	availabilityZone := "us-west-2d"
	return &AWSCloud{
		ec2: &FakeEC2{
			instances: instances,
		},
		availabilityZone: availabilityZone,
	}
}

func mockAvailabilityZone(region string, availabilityZone string) *AWSCloud {
	return &AWSCloud{
		ec2:              &FakeEC2{},
		availabilityZone: availabilityZone,
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
	instance0.PrivateDNSName = aws.String("instance1")
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
	instance1.PrivateDNSName = aws.String("instance2")
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
	instance2.PrivateDNSName = aws.String("instance3")
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
	instance3.PrivateDNSName = aws.String("instance4")
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
		{"quux", []string{"instance4"}},
		{"a", []string{"instance2", "instance3"}},
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
	instance0.PrivateDNSName = aws.String("instance1")
	instance0.PrivateIPAddress = aws.String("192.168.0.1")
	instance0.PublicIPAddress = aws.String("1.2.3.4")
	instance0.InstanceType = aws.String("c3.large")
	state0 := ec2.InstanceState{
		Name: aws.String("running"),
	}
	instance0.State = &state0

	//1
	instance1.PrivateDNSName = aws.String("instance1")
	instance1.PrivateIPAddress = aws.String("192.168.0.2")
	instance1.InstanceType = aws.String("c3.large")
	state1 := ec2.InstanceState{
		Name: aws.String("running"),
	}
	instance1.State = &state1

	instances := []*ec2.Instance{&instance0, &instance1}

	aws1 := mockInstancesResp([]*ec2.Instance{})
	_, err1 := aws1.NodeAddresses("instance")
	if err1 == nil {
		t.Errorf("Should error when no instance found")
	}

	aws2 := mockInstancesResp(instances)
	_, err2 := aws2.NodeAddresses("instance1")
	if err2 == nil {
		t.Errorf("Should error when multiple instances found")
	}

	aws3 := mockInstancesResp(instances[0:1])
	addrs3, err3 := aws3.NodeAddresses("instance1")
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

func TestGetResources(t *testing.T) {
	var instance0 ec2.Instance
	var instance1 ec2.Instance
	var instance2 ec2.Instance

	//0
	instance0.PrivateDNSName = aws.String("m3.medium")
	instance0.InstanceType = aws.String("m3.medium")
	state0 := ec2.InstanceState{
		Name: aws.String("running"),
	}
	instance0.State = &state0

	//1
	instance1.PrivateDNSName = aws.String("r3.8xlarge")
	instance1.InstanceType = aws.String("r3.8xlarge")
	state1 := ec2.InstanceState{
		Name: aws.String("running"),
	}
	instance1.State = &state1

	//2
	instance2.PrivateDNSName = aws.String("unknown.type")
	instance2.InstanceType = aws.String("unknown.type")
	state2 := ec2.InstanceState{
		Name: aws.String("running"),
	}
	instance2.State = &state2

	instances := []*ec2.Instance{&instance0, &instance1, &instance2}

	aws1 := mockInstancesResp(instances)

	res1, err1 := aws1.GetNodeResources("m3.medium")
	if err1 != nil {
		t.Errorf("Should not error when instance type found: %v", err1)
	}
	e1 := &api.NodeResources{
		Capacity: api.ResourceList{
			api.ResourceCPU:    *resource.NewMilliQuantity(int64(3.0*1000), resource.DecimalSI),
			api.ResourceMemory: *resource.NewQuantity(int64(3.75*1024*1024*1024), resource.BinarySI),
		},
	}
	if !reflect.DeepEqual(e1, res1) {
		t.Errorf("Expected %v, got %v", e1, res1)
	}

	res2, err2 := aws1.GetNodeResources("r3.8xlarge")
	if err2 != nil {
		t.Errorf("Should not error when instance type found: %v", err2)
	}
	e2 := &api.NodeResources{
		Capacity: api.ResourceList{
			api.ResourceCPU:    *resource.NewMilliQuantity(int64(104.0*1000), resource.DecimalSI),
			api.ResourceMemory: *resource.NewQuantity(int64(244.0*1024*1024*1024), resource.BinarySI),
		},
	}
	if !reflect.DeepEqual(e2, res2) {
		t.Errorf("Expected %v, got %v", e2, res2)
	}

	res3, err3 := aws1.GetNodeResources("unknown.type")
	if err3 != nil {
		t.Errorf("Should not error when unknown instance type")
	}
	if res3 != nil {
		t.Errorf("Should return nil resources when unknown instance type")
	}
}
