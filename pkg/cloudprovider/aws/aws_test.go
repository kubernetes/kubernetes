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

	"github.com/mitchellh/goamz/aws"
	"github.com/mitchellh/goamz/ec2"

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
	fakeAuthFunc := func() (auth aws.Auth, err error) {
		return aws.Auth{"", "", ""}, nil
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
	instances []ec2.Instance
}

func contains(haystack []string, needle string) bool {
	for _, s := range haystack {
		if needle == s {
			return true
		}
	}
	return false
}

func (self *FakeEC2) Instances(instanceIds []string, filter *ec2InstanceFilter) (resp *ec2.InstancesResp, err error) {
	matches := []ec2.Instance{}
	for _, instance := range self.instances {
		if filter != nil && !filter.Matches(instance) {
			continue
		}
		if instanceIds != nil && !contains(instanceIds, instance.InstanceId) {
			continue
		}
		matches = append(matches, instance)
	}
	return &ec2.InstancesResp{"",
		[]ec2.Reservation{
			{"", "", "", nil, matches}}}, nil
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

func (ec2 *FakeEC2) AttachVolume(volumeID string, instanceId string, mountDevice string) (resp *ec2.AttachVolumeResp, err error) {
	panic("Not implemented")
}

func (ec2 *FakeEC2) DetachVolume(volumeID string) (resp *ec2.SimpleResp, err error) {
	panic("Not implemented")
}

func (ec2 *FakeEC2) Volumes(volumeIDs []string, filter *ec2.Filter) (resp *ec2.VolumesResp, err error) {
	panic("Not implemented")
}

func (ec2 *FakeEC2) CreateVolume(request *ec2.CreateVolume) (resp *ec2.CreateVolumeResp, err error) {
	panic("Not implemented")
}

func (ec2 *FakeEC2) DeleteVolume(volumeID string) (resp *ec2.SimpleResp, err error) {
	panic("Not implemented")
}

func mockInstancesResp(instances []ec2.Instance) (aws *AWSCloud) {
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
		region:           aws.Regions[region],
	}

}

func TestList(t *testing.T) {
	instances := make([]ec2.Instance, 4)
	instances[0].Tags = []ec2.Tag{{"Name", "foo"}}
	instances[0].PrivateDNSName = "instance1"
	instances[0].State.Name = "running"
	instances[1].Tags = []ec2.Tag{{"Name", "bar"}}
	instances[1].PrivateDNSName = "instance2"
	instances[1].State.Name = "running"
	instances[2].Tags = []ec2.Tag{{"Name", "baz"}}
	instances[2].PrivateDNSName = "instance3"
	instances[2].State.Name = "running"
	instances[3].Tags = []ec2.Tag{{"Name", "quux"}}
	instances[3].PrivateDNSName = "instance4"
	instances[3].State.Name = "running"

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
	instances := make([]ec2.Instance, 2)
	instances[0].PrivateDNSName = "instance1"
	instances[0].PrivateIpAddress = "192.168.0.1"
	instances[0].PublicIpAddress = "1.2.3.4"
	instances[0].State.Name = "running"
	instances[1].PrivateDNSName = "instance1"
	instances[1].PrivateIpAddress = "192.168.0.2"
	instances[1].State.Name = "running"

	aws1 := mockInstancesResp([]ec2.Instance{})
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
	instances := make([]ec2.Instance, 3)
	instances[0].PrivateDNSName = "m3.medium"
	instances[0].InstanceType = "m3.medium"
	instances[0].State.Name = "running"
	instances[1].PrivateDNSName = "r3.8xlarge"
	instances[1].InstanceType = "r3.8xlarge"
	instances[1].State.Name = "running"
	instances[2].PrivateDNSName = "unknown.type"
	instances[2].InstanceType = "unknown.type"
	instances[2].State.Name = "running"

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
