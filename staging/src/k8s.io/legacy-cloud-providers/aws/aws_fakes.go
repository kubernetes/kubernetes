// +build !providerless

/*
Copyright 2017 The Kubernetes Authors.

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
	"sort"
	"strings"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/autoscaling"
	"github.com/aws/aws-sdk-go/service/ec2"
	"github.com/aws/aws-sdk-go/service/elb"
	"github.com/aws/aws-sdk-go/service/elbv2"
	"github.com/aws/aws-sdk-go/service/kms"
	"k8s.io/klog/v2"
)

// FakeAWSServices is an fake AWS session used for testing
type FakeAWSServices struct {
	region                      string
	instances                   []*ec2.Instance
	selfInstance                *ec2.Instance
	networkInterfacesMacs       []string
	networkInterfacesPrivateIPs [][]string
	networkInterfacesVpcIDs     []string

	ec2      FakeEC2
	elb      ELB
	elbv2    ELBV2
	asg      *FakeASG
	metadata *FakeMetadata
	kms      *FakeKMS
}

// NewFakeAWSServices creates a new FakeAWSServices
func NewFakeAWSServices(clusterID string) *FakeAWSServices {
	s := &FakeAWSServices{}
	s.region = "us-east-1"
	s.ec2 = &FakeEC2Impl{aws: s}
	s.elb = &FakeELB{aws: s}
	s.elbv2 = &FakeELBV2{aws: s}
	s.asg = &FakeASG{aws: s}
	s.metadata = &FakeMetadata{aws: s}
	s.kms = &FakeKMS{aws: s}

	s.networkInterfacesMacs = []string{"aa:bb:cc:dd:ee:00", "aa:bb:cc:dd:ee:01"}
	s.networkInterfacesVpcIDs = []string{"vpc-mac0", "vpc-mac1"}

	selfInstance := &ec2.Instance{}
	selfInstance.InstanceId = aws.String("i-self")
	selfInstance.Placement = &ec2.Placement{
		AvailabilityZone: aws.String("us-east-1a"),
	}
	selfInstance.PrivateDnsName = aws.String("ip-172-20-0-100.ec2.internal")
	selfInstance.PrivateIpAddress = aws.String("192.168.0.1")
	selfInstance.PublicIpAddress = aws.String("1.2.3.4")
	s.selfInstance = selfInstance
	s.instances = []*ec2.Instance{selfInstance}

	var tag ec2.Tag
	tag.Key = aws.String(TagNameKubernetesClusterLegacy)
	tag.Value = aws.String(clusterID)
	selfInstance.Tags = []*ec2.Tag{&tag}

	return s
}

// WithAz sets the ec2 placement availability zone
func (s *FakeAWSServices) WithAz(az string) *FakeAWSServices {
	if s.selfInstance.Placement == nil {
		s.selfInstance.Placement = &ec2.Placement{}
	}
	s.selfInstance.Placement.AvailabilityZone = aws.String(az)
	return s
}

// Compute returns a fake EC2 client
func (s *FakeAWSServices) Compute(region string) (EC2, error) {
	return s.ec2, nil
}

// LoadBalancing returns a fake ELB client
func (s *FakeAWSServices) LoadBalancing(region string) (ELB, error) {
	return s.elb, nil
}

// LoadBalancingV2 returns a fake ELBV2 client
func (s *FakeAWSServices) LoadBalancingV2(region string) (ELBV2, error) {
	return s.elbv2, nil
}

// Autoscaling returns a fake ASG client
func (s *FakeAWSServices) Autoscaling(region string) (ASG, error) {
	return s.asg, nil
}

// Metadata returns a fake EC2Metadata client
func (s *FakeAWSServices) Metadata() (EC2Metadata, error) {
	return s.metadata, nil
}

// KeyManagement returns a fake KMS client
func (s *FakeAWSServices) KeyManagement(region string) (KMS, error) {
	return s.kms, nil
}

// FakeEC2 is a fake EC2 client used for testing
type FakeEC2 interface {
	EC2
	CreateSubnet(*ec2.Subnet) (*ec2.CreateSubnetOutput, error)
	RemoveSubnets()
	CreateRouteTable(*ec2.RouteTable) (*ec2.CreateRouteTableOutput, error)
	RemoveRouteTables()
}

// FakeEC2Impl is an implementation of the FakeEC2 interface used for testing
type FakeEC2Impl struct {
	aws                      *FakeAWSServices
	Subnets                  []*ec2.Subnet
	DescribeSubnetsInput     *ec2.DescribeSubnetsInput
	RouteTables              []*ec2.RouteTable
	DescribeRouteTablesInput *ec2.DescribeRouteTablesInput
}

// DescribeInstances returns fake instance descriptions
func (ec2i *FakeEC2Impl) DescribeInstances(request *ec2.DescribeInstancesInput) ([]*ec2.Instance, error) {
	matches := []*ec2.Instance{}
	for _, instance := range ec2i.aws.instances {
		if request.InstanceIds != nil {
			if instance.InstanceId == nil {
				klog.Warning("Instance with no instance id: ", instance)
				continue
			}

			found := false
			for _, instanceID := range request.InstanceIds {
				if *instanceID == *instance.InstanceId {
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

// AttachVolume is not implemented but is required for interface conformance
func (ec2i *FakeEC2Impl) AttachVolume(request *ec2.AttachVolumeInput) (resp *ec2.VolumeAttachment, err error) {
	panic("Not implemented")
}

// DetachVolume is not implemented but is required for interface conformance
func (ec2i *FakeEC2Impl) DetachVolume(request *ec2.DetachVolumeInput) (resp *ec2.VolumeAttachment, err error) {
	panic("Not implemented")
}

// DescribeVolumes is not implemented but is required for interface conformance
func (ec2i *FakeEC2Impl) DescribeVolumes(request *ec2.DescribeVolumesInput) ([]*ec2.Volume, error) {
	panic("Not implemented")
}

// CreateVolume is not implemented but is required for interface conformance
func (ec2i *FakeEC2Impl) CreateVolume(request *ec2.CreateVolumeInput) (resp *ec2.Volume, err error) {
	panic("Not implemented")
}

// DeleteVolume is not implemented but is required for interface conformance
func (ec2i *FakeEC2Impl) DeleteVolume(request *ec2.DeleteVolumeInput) (resp *ec2.DeleteVolumeOutput, err error) {
	panic("Not implemented")
}

// DescribeSecurityGroups is not implemented but is required for interface
// conformance
func (ec2i *FakeEC2Impl) DescribeSecurityGroups(request *ec2.DescribeSecurityGroupsInput) ([]*ec2.SecurityGroup, error) {
	panic("Not implemented")
}

// CreateSecurityGroup is not implemented but is required for interface
// conformance
func (ec2i *FakeEC2Impl) CreateSecurityGroup(*ec2.CreateSecurityGroupInput) (*ec2.CreateSecurityGroupOutput, error) {
	panic("Not implemented")
}

// DeleteSecurityGroup is not implemented but is required for interface
// conformance
func (ec2i *FakeEC2Impl) DeleteSecurityGroup(*ec2.DeleteSecurityGroupInput) (*ec2.DeleteSecurityGroupOutput, error) {
	panic("Not implemented")
}

// AuthorizeSecurityGroupIngress is not implemented but is required for
// interface conformance
func (ec2i *FakeEC2Impl) AuthorizeSecurityGroupIngress(*ec2.AuthorizeSecurityGroupIngressInput) (*ec2.AuthorizeSecurityGroupIngressOutput, error) {
	panic("Not implemented")
}

// RevokeSecurityGroupIngress is not implemented but is required for interface
// conformance
func (ec2i *FakeEC2Impl) RevokeSecurityGroupIngress(*ec2.RevokeSecurityGroupIngressInput) (*ec2.RevokeSecurityGroupIngressOutput, error) {
	panic("Not implemented")
}

// DescribeVolumeModifications is not implemented but is required for interface
// conformance
func (ec2i *FakeEC2Impl) DescribeVolumeModifications(*ec2.DescribeVolumesModificationsInput) ([]*ec2.VolumeModification, error) {
	panic("Not implemented")
}

// ModifyVolume is not implemented but is required for interface conformance
func (ec2i *FakeEC2Impl) ModifyVolume(*ec2.ModifyVolumeInput) (*ec2.ModifyVolumeOutput, error) {
	panic("Not implemented")
}

// CreateSubnet creates fake subnets
func (ec2i *FakeEC2Impl) CreateSubnet(request *ec2.Subnet) (*ec2.CreateSubnetOutput, error) {
	ec2i.Subnets = append(ec2i.Subnets, request)
	response := &ec2.CreateSubnetOutput{
		Subnet: request,
	}
	return response, nil
}

// DescribeSubnets returns fake subnet descriptions
func (ec2i *FakeEC2Impl) DescribeSubnets(request *ec2.DescribeSubnetsInput) ([]*ec2.Subnet, error) {
	ec2i.DescribeSubnetsInput = request
	return ec2i.Subnets, nil
}

// RemoveSubnets clears subnets on client
func (ec2i *FakeEC2Impl) RemoveSubnets() {
	ec2i.Subnets = ec2i.Subnets[:0]
}

// CreateTags is not implemented but is required for interface conformance
func (ec2i *FakeEC2Impl) CreateTags(*ec2.CreateTagsInput) (*ec2.CreateTagsOutput, error) {
	panic("Not implemented")
}

// DescribeRouteTables returns fake route table descriptions
func (ec2i *FakeEC2Impl) DescribeRouteTables(request *ec2.DescribeRouteTablesInput) ([]*ec2.RouteTable, error) {
	ec2i.DescribeRouteTablesInput = request
	return ec2i.RouteTables, nil
}

// CreateRouteTable creates fake route tables
func (ec2i *FakeEC2Impl) CreateRouteTable(request *ec2.RouteTable) (*ec2.CreateRouteTableOutput, error) {
	ec2i.RouteTables = append(ec2i.RouteTables, request)
	response := &ec2.CreateRouteTableOutput{
		RouteTable: request,
	}
	return response, nil
}

// RemoveRouteTables clears route tables on client
func (ec2i *FakeEC2Impl) RemoveRouteTables() {
	ec2i.RouteTables = ec2i.RouteTables[:0]
}

// CreateRoute is not implemented but is required for interface conformance
func (ec2i *FakeEC2Impl) CreateRoute(request *ec2.CreateRouteInput) (*ec2.CreateRouteOutput, error) {
	panic("Not implemented")
}

// DeleteRoute is not implemented but is required for interface conformance
func (ec2i *FakeEC2Impl) DeleteRoute(request *ec2.DeleteRouteInput) (*ec2.DeleteRouteOutput, error) {
	panic("Not implemented")
}

// ModifyInstanceAttribute is not implemented but is required for interface
// conformance
func (ec2i *FakeEC2Impl) ModifyInstanceAttribute(request *ec2.ModifyInstanceAttributeInput) (*ec2.ModifyInstanceAttributeOutput, error) {
	panic("Not implemented")
}

// DescribeVpcs returns fake VPC descriptions
func (ec2i *FakeEC2Impl) DescribeVpcs(request *ec2.DescribeVpcsInput) (*ec2.DescribeVpcsOutput, error) {
	return &ec2.DescribeVpcsOutput{Vpcs: []*ec2.Vpc{{CidrBlock: aws.String("172.20.0.0/16")}}}, nil
}

// FakeMetadata is a fake EC2 metadata service client used for testing
type FakeMetadata struct {
	aws *FakeAWSServices
}

// GetMetadata returns fake EC2 metadata for testing
func (m *FakeMetadata) GetMetadata(key string) (string, error) {
	networkInterfacesPrefix := "network/interfaces/macs/"
	i := m.aws.selfInstance
	if key == "placement/availability-zone" {
		az := ""
		if i.Placement != nil {
			az = aws.StringValue(i.Placement.AvailabilityZone)
		}
		return az, nil
	} else if key == "instance-id" {
		return aws.StringValue(i.InstanceId), nil
	} else if key == "local-hostname" {
		return aws.StringValue(i.PrivateDnsName), nil
	} else if key == "public-hostname" {
		return aws.StringValue(i.PublicDnsName), nil
	} else if key == "local-ipv4" {
		return aws.StringValue(i.PrivateIpAddress), nil
	} else if key == "public-ipv4" {
		return aws.StringValue(i.PublicIpAddress), nil
	} else if strings.HasPrefix(key, networkInterfacesPrefix) {
		if key == networkInterfacesPrefix {
			// Return the MACs sorted lexically rather than in device-number
			// order; this matches AWS's observed behavior and lets us test
			// that we fix up the ordering correctly in NodeAddresses().
			macs := make([]string, len(m.aws.networkInterfacesMacs))
			copy(macs, m.aws.networkInterfacesMacs)
			sort.Strings(macs)
			return strings.Join(macs, "/\n") + "/\n", nil
		}

		keySplit := strings.Split(key, "/")
		macParam := keySplit[3]
		if len(keySplit) == 5 && keySplit[4] == "vpc-id" {
			for i, macElem := range m.aws.networkInterfacesMacs {
				if macParam == macElem {
					return m.aws.networkInterfacesVpcIDs[i], nil
				}
			}
		}
		if len(keySplit) == 5 && keySplit[4] == "device-number" {
			for i, macElem := range m.aws.networkInterfacesMacs {
				if macParam == macElem {
					return fmt.Sprintf("%d\n", i), nil
				}
			}
		}
		if len(keySplit) == 5 && keySplit[4] == "local-ipv4s" {
			for i, macElem := range m.aws.networkInterfacesMacs {
				if macParam == macElem {
					return strings.Join(m.aws.networkInterfacesPrivateIPs[i], "/\n"), nil
				}
			}
		}

		return "", nil
	}

	return "", nil
}

// FakeELB is a fake ELB client used for testing
type FakeELB struct {
	aws *FakeAWSServices
}

// CreateLoadBalancer is not implemented but is required for interface
// conformance
func (elb *FakeELB) CreateLoadBalancer(*elb.CreateLoadBalancerInput) (*elb.CreateLoadBalancerOutput, error) {
	panic("Not implemented")
}

// DeleteLoadBalancer is not implemented but is required for interface
// conformance
func (elb *FakeELB) DeleteLoadBalancer(input *elb.DeleteLoadBalancerInput) (*elb.DeleteLoadBalancerOutput, error) {
	panic("Not implemented")
}

// DescribeLoadBalancers is not implemented but is required for interface
// conformance
func (elb *FakeELB) DescribeLoadBalancers(input *elb.DescribeLoadBalancersInput) (*elb.DescribeLoadBalancersOutput, error) {
	panic("Not implemented")
}

// AddTags is not implemented but is required for interface conformance
func (elb *FakeELB) AddTags(input *elb.AddTagsInput) (*elb.AddTagsOutput, error) {
	panic("Not implemented")
}

// RegisterInstancesWithLoadBalancer is not implemented but is required for
// interface conformance
func (elb *FakeELB) RegisterInstancesWithLoadBalancer(*elb.RegisterInstancesWithLoadBalancerInput) (*elb.RegisterInstancesWithLoadBalancerOutput, error) {
	panic("Not implemented")
}

// DeregisterInstancesFromLoadBalancer is not implemented but is required for
// interface conformance
func (elb *FakeELB) DeregisterInstancesFromLoadBalancer(*elb.DeregisterInstancesFromLoadBalancerInput) (*elb.DeregisterInstancesFromLoadBalancerOutput, error) {
	panic("Not implemented")
}

// DetachLoadBalancerFromSubnets is not implemented but is required for
// interface conformance
func (elb *FakeELB) DetachLoadBalancerFromSubnets(*elb.DetachLoadBalancerFromSubnetsInput) (*elb.DetachLoadBalancerFromSubnetsOutput, error) {
	panic("Not implemented")
}

// AttachLoadBalancerToSubnets is not implemented but is required for interface
// conformance
func (elb *FakeELB) AttachLoadBalancerToSubnets(*elb.AttachLoadBalancerToSubnetsInput) (*elb.AttachLoadBalancerToSubnetsOutput, error) {
	panic("Not implemented")
}

// CreateLoadBalancerListeners is not implemented but is required for interface
// conformance
func (elb *FakeELB) CreateLoadBalancerListeners(*elb.CreateLoadBalancerListenersInput) (*elb.CreateLoadBalancerListenersOutput, error) {
	panic("Not implemented")
}

// DeleteLoadBalancerListeners is not implemented but is required for interface
// conformance
func (elb *FakeELB) DeleteLoadBalancerListeners(*elb.DeleteLoadBalancerListenersInput) (*elb.DeleteLoadBalancerListenersOutput, error) {
	panic("Not implemented")
}

// ApplySecurityGroupsToLoadBalancer is not implemented but is required for
// interface conformance
func (elb *FakeELB) ApplySecurityGroupsToLoadBalancer(*elb.ApplySecurityGroupsToLoadBalancerInput) (*elb.ApplySecurityGroupsToLoadBalancerOutput, error) {
	panic("Not implemented")
}

// ConfigureHealthCheck is not implemented but is required for interface
// conformance
func (elb *FakeELB) ConfigureHealthCheck(*elb.ConfigureHealthCheckInput) (*elb.ConfigureHealthCheckOutput, error) {
	panic("Not implemented")
}

// CreateLoadBalancerPolicy is not implemented but is required for interface
// conformance
func (elb *FakeELB) CreateLoadBalancerPolicy(*elb.CreateLoadBalancerPolicyInput) (*elb.CreateLoadBalancerPolicyOutput, error) {
	panic("Not implemented")
}

// SetLoadBalancerPoliciesForBackendServer is not implemented but is required
// for interface conformance
func (elb *FakeELB) SetLoadBalancerPoliciesForBackendServer(*elb.SetLoadBalancerPoliciesForBackendServerInput) (*elb.SetLoadBalancerPoliciesForBackendServerOutput, error) {
	panic("Not implemented")
}

// SetLoadBalancerPoliciesOfListener is not implemented but is required for
// interface conformance
func (elb *FakeELB) SetLoadBalancerPoliciesOfListener(input *elb.SetLoadBalancerPoliciesOfListenerInput) (*elb.SetLoadBalancerPoliciesOfListenerOutput, error) {
	panic("Not implemented")
}

// DescribeLoadBalancerPolicies is not implemented but is required for
// interface conformance
func (elb *FakeELB) DescribeLoadBalancerPolicies(input *elb.DescribeLoadBalancerPoliciesInput) (*elb.DescribeLoadBalancerPoliciesOutput, error) {
	panic("Not implemented")
}

// DescribeLoadBalancerAttributes is not implemented but is required for
// interface conformance
func (elb *FakeELB) DescribeLoadBalancerAttributes(*elb.DescribeLoadBalancerAttributesInput) (*elb.DescribeLoadBalancerAttributesOutput, error) {
	panic("Not implemented")
}

// ModifyLoadBalancerAttributes is not implemented but is required for
// interface conformance
func (elb *FakeELB) ModifyLoadBalancerAttributes(*elb.ModifyLoadBalancerAttributesInput) (*elb.ModifyLoadBalancerAttributesOutput, error) {
	panic("Not implemented")
}

// FakeELBV2 is a fake ELBV2 client used for testing
type FakeELBV2 struct {
	aws *FakeAWSServices
}

// AddTags is not implemented but is required for interface conformance
func (elb *FakeELBV2) AddTags(input *elbv2.AddTagsInput) (*elbv2.AddTagsOutput, error) {
	panic("Not implemented")
}

// CreateLoadBalancer is not implemented but is required for interface
// conformance
func (elb *FakeELBV2) CreateLoadBalancer(*elbv2.CreateLoadBalancerInput) (*elbv2.CreateLoadBalancerOutput, error) {
	panic("Not implemented")
}

// DescribeLoadBalancers is not implemented but is required for interface
// conformance
func (elb *FakeELBV2) DescribeLoadBalancers(*elbv2.DescribeLoadBalancersInput) (*elbv2.DescribeLoadBalancersOutput, error) {
	panic("Not implemented")
}

// DeleteLoadBalancer is not implemented but is required for interface
// conformance
func (elb *FakeELBV2) DeleteLoadBalancer(*elbv2.DeleteLoadBalancerInput) (*elbv2.DeleteLoadBalancerOutput, error) {
	panic("Not implemented")
}

// ModifyLoadBalancerAttributes is not implemented but is required for
// interface conformance
func (elb *FakeELBV2) ModifyLoadBalancerAttributes(*elbv2.ModifyLoadBalancerAttributesInput) (*elbv2.ModifyLoadBalancerAttributesOutput, error) {
	panic("Not implemented")
}

// DescribeLoadBalancerAttributes is not implemented but is required for
// interface conformance
func (elb *FakeELBV2) DescribeLoadBalancerAttributes(*elbv2.DescribeLoadBalancerAttributesInput) (*elbv2.DescribeLoadBalancerAttributesOutput, error) {
	panic("Not implemented")
}

// CreateTargetGroup is not implemented but is required for interface
// conformance
func (elb *FakeELBV2) CreateTargetGroup(*elbv2.CreateTargetGroupInput) (*elbv2.CreateTargetGroupOutput, error) {
	panic("Not implemented")
}

// DescribeTargetGroups is not implemented but is required for interface
// conformance
func (elb *FakeELBV2) DescribeTargetGroups(*elbv2.DescribeTargetGroupsInput) (*elbv2.DescribeTargetGroupsOutput, error) {
	panic("Not implemented")
}

// ModifyTargetGroup is not implemented but is required for interface
// conformance
func (elb *FakeELBV2) ModifyTargetGroup(*elbv2.ModifyTargetGroupInput) (*elbv2.ModifyTargetGroupOutput, error) {
	panic("Not implemented")
}

// DeleteTargetGroup is not implemented but is required for interface
// conformance
func (elb *FakeELBV2) DeleteTargetGroup(*elbv2.DeleteTargetGroupInput) (*elbv2.DeleteTargetGroupOutput, error) {
	panic("Not implemented")
}

// DescribeTargetHealth is not implemented but is required for interface
// conformance
func (elb *FakeELBV2) DescribeTargetHealth(input *elbv2.DescribeTargetHealthInput) (*elbv2.DescribeTargetHealthOutput, error) {
	panic("Not implemented")
}

// DescribeTargetGroupAttributes is not implemented but is required for
// interface conformance
func (elb *FakeELBV2) DescribeTargetGroupAttributes(*elbv2.DescribeTargetGroupAttributesInput) (*elbv2.DescribeTargetGroupAttributesOutput, error) {
	panic("Not implemented")
}

// ModifyTargetGroupAttributes is not implemented but is required for interface
// conformance
func (elb *FakeELBV2) ModifyTargetGroupAttributes(*elbv2.ModifyTargetGroupAttributesInput) (*elbv2.ModifyTargetGroupAttributesOutput, error) {
	panic("Not implemented")
}

// RegisterTargets is not implemented but is required for interface conformance
func (elb *FakeELBV2) RegisterTargets(*elbv2.RegisterTargetsInput) (*elbv2.RegisterTargetsOutput, error) {
	panic("Not implemented")
}

// DeregisterTargets is not implemented but is required for interface
// conformance
func (elb *FakeELBV2) DeregisterTargets(*elbv2.DeregisterTargetsInput) (*elbv2.DeregisterTargetsOutput, error) {
	panic("Not implemented")
}

// CreateListener is not implemented but is required for interface conformance
func (elb *FakeELBV2) CreateListener(*elbv2.CreateListenerInput) (*elbv2.CreateListenerOutput, error) {
	panic("Not implemented")
}

// DescribeListeners is not implemented but is required for interface
// conformance
func (elb *FakeELBV2) DescribeListeners(*elbv2.DescribeListenersInput) (*elbv2.DescribeListenersOutput, error) {
	panic("Not implemented")
}

// DeleteListener is not implemented but is required for interface conformance
func (elb *FakeELBV2) DeleteListener(*elbv2.DeleteListenerInput) (*elbv2.DeleteListenerOutput, error) {
	panic("Not implemented")
}

// ModifyListener is not implemented but is required for interface conformance
func (elb *FakeELBV2) ModifyListener(*elbv2.ModifyListenerInput) (*elbv2.ModifyListenerOutput, error) {
	panic("Not implemented")
}

// WaitUntilLoadBalancersDeleted is not implemented but is required for
// interface conformance
func (elb *FakeELBV2) WaitUntilLoadBalancersDeleted(*elbv2.DescribeLoadBalancersInput) error {
	panic("Not implemented")
}

// FakeASG is a fake Autoscaling client used for testing
type FakeASG struct {
	aws *FakeAWSServices
}

// UpdateAutoScalingGroup is not implemented but is required for interface
// conformance
func (a *FakeASG) UpdateAutoScalingGroup(*autoscaling.UpdateAutoScalingGroupInput) (*autoscaling.UpdateAutoScalingGroupOutput, error) {
	panic("Not implemented")
}

// DescribeAutoScalingGroups is not implemented but is required for interface
// conformance
func (a *FakeASG) DescribeAutoScalingGroups(*autoscaling.DescribeAutoScalingGroupsInput) (*autoscaling.DescribeAutoScalingGroupsOutput, error) {
	panic("Not implemented")
}

// FakeKMS is a fake KMS client used for testing
type FakeKMS struct {
	aws *FakeAWSServices
}

// DescribeKey is not implemented but is required for interface conformance
func (kms *FakeKMS) DescribeKey(*kms.DescribeKeyInput) (*kms.DescribeKeyOutput, error) {
	panic("Not implemented")
}

func instanceMatchesFilter(instance *ec2.Instance, filter *ec2.Filter) bool {
	name := *filter.Name
	if name == "private-dns-name" {
		if instance.PrivateDnsName == nil {
			return false
		}
		return contains(filter.Values, *instance.PrivateDnsName)
	}

	if name == "instance-state-name" {
		return contains(filter.Values, *instance.State.Name)
	}

	if name == "tag-key" {
		for _, instanceTag := range instance.Tags {
			if contains(filter.Values, aws.StringValue(instanceTag.Key)) {
				return true
			}
		}
		return false
	}

	if strings.HasPrefix(name, "tag:") {
		tagName := name[4:]
		for _, instanceTag := range instance.Tags {
			if aws.StringValue(instanceTag.Key) == tagName && contains(filter.Values, aws.StringValue(instanceTag.Value)) {
				return true
			}
		}
		return false
	}

	panic("Unknown filter name: " + name)
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
