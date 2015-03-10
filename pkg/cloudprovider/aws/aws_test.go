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
	"reflect"
	"strings"
	"testing"

	"github.com/mitchellh/goamz/aws"
	"github.com/mitchellh/goamz/ec2"
)

func TestReadAWSCloudConfig(t *testing.T) {
	_, err1 := readAWSCloudConfig(nil)
	if err1 == nil {
		t.Errorf("Should error when no config reader is given")
	}

	_, err2 := readAWSCloudConfig(strings.NewReader(""))
	if err2 == nil {
		t.Errorf("Should error when config is empty")
	}

	_, err3 := readAWSCloudConfig(strings.NewReader("[global]\n"))
	if err3 == nil {
		t.Errorf("Should error when no region is specified")
	}

	cfg, err4 := readAWSCloudConfig(strings.NewReader("[global]\nregion = eu-west-1"))
	if err4 != nil {
		t.Errorf("Should succeed when a region is specified: %s", err4)
	}
	if cfg.Global.Region != "eu-west-1" {
		t.Errorf("Should read region from config")
	}
}

func TestNewAWSCloud(t *testing.T) {
	fakeAuthFunc := func() (auth aws.Auth, err error) {
		return aws.Auth{"", "", ""}, nil
	}

	_, err1 := newAWSCloud(nil, fakeAuthFunc)
	if err1 == nil {
		t.Errorf("Should error when no config reader is given")
	}

	_, err2 := newAWSCloud(strings.NewReader(
		"[global]\nregion = blahonga"),
		fakeAuthFunc)
	if err2 == nil {
		t.Errorf("Should error when config specifies invalid region")
	}

	_, err3 := newAWSCloud(
		strings.NewReader("[global]\nregion = eu-west-1"),
		fakeAuthFunc)
	if err3 != nil {
		t.Errorf("Should succeed when a valid region is specified: %s", err3)
	}
}

type FakeEC2 struct {
	instances        []ec2.Instance
	availabilityZone string
}

func (self *FakeEC2) Instances(instanceIds []string, filter *ec2InstanceFilter) (resp *ec2.InstancesResp, err error) {
	matches := []ec2.Instance{}
	for _, instance := range self.instances {
		if filter == nil || filter.Matches(instance) {
			matches = append(matches, instance)
		}
	}
	return &ec2.InstancesResp{"",
		[]ec2.Reservation{
			{"", "", "", nil, matches}}}, nil
}

func (self *FakeEC2) GetMetaData(key string) ([]byte, error) {
	if key == "placement/availability-zone" {
		return []byte(self.availabilityZone), nil
	} else {
		return nil, nil
	}
}

func mockInstancesResp(instances []ec2.Instance) (aws *AWSCloud) {
	availabilityZone := "us-west-2d"
	return &AWSCloud{
		ec2: &FakeEC2{
			instances:        instances,
			availabilityZone: availabilityZone,
		},
	}
}

func mockAvailabilityZone(region string, availabilityZone string) *AWSCloud {
	return &AWSCloud{
		ec2: &FakeEC2{
			availabilityZone: availabilityZone,
		},
		region: aws.Regions[region],
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

func TestIPAddress(t *testing.T) {
	// Note these instances have the same name
	// (we test that this produces an error)
	instances := make([]ec2.Instance, 2)
	instances[0].PrivateDNSName = "instance1"
	instances[0].PrivateIpAddress = "192.168.0.1"
	instances[0].State.Name = "running"
	instances[1].PrivateDNSName = "instance1"
	instances[1].PrivateIpAddress = "192.168.0.2"
	instances[1].State.Name = "running"

	aws1 := mockInstancesResp([]ec2.Instance{})
	_, err1 := aws1.IPAddress("instance")
	if err1 == nil {
		t.Errorf("Should error when no instance found")
	}

	aws2 := mockInstancesResp(instances)
	_, err2 := aws2.IPAddress("instance1")
	if err2 == nil {
		t.Errorf("Should error when multiple instances found")
	}

	aws3 := mockInstancesResp(instances[0:1])
	ip3, err3 := aws3.IPAddress("instance1")
	if err3 != nil {
		t.Errorf("Should not error when instance found")
	}
	if e, a := instances[0].PrivateIpAddress, ip3.String(); e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}
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
