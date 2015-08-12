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
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/autoscaling"
	"github.com/golang/glog"
)

// AWSCloud implements InstanceGroups
var _ InstanceGroups = &AWSCloud{}

// Implement InstanceGroups.ResizeInstanceGroup
// Set the size to the fixed size
func (a *AWSCloud) ResizeInstanceGroup(instanceGroupName string, size int) error {
	request := &autoscaling.UpdateAutoScalingGroupInput{
		AutoScalingGroupName: aws.String(instanceGroupName),
		MinSize:              aws.Long(int64(size)),
		MaxSize:              aws.Long(int64(size)),
	}
	if _, err := a.asg.UpdateAutoScalingGroup(request); err != nil {
		return fmt.Errorf("error resizing AWS autoscaling group: %v", err)
	}
	return nil
}

// Implement InstanceGroups.DescribeInstanceGroup
// Queries the cloud provider for information about the specified instance group
func (a *AWSCloud) DescribeInstanceGroup(instanceGroupName string) (InstanceGroupInfo, error) {
	request := &autoscaling.DescribeAutoScalingGroupsInput{
		AutoScalingGroupNames: []*string{aws.String(instanceGroupName)},
	}
	response, err := a.asg.DescribeAutoScalingGroups(request)
	if err != nil {
		return nil, fmt.Errorf("error listing AWS autoscaling group (%s): %v", instanceGroupName, err)
	}

	if len(response.AutoScalingGroups) == 0 {
		return nil, nil
	}
	if len(response.AutoScalingGroups) > 1 {
		glog.Warning("AWS returned multiple autoscaling groups with name ", instanceGroupName)
	}
	group := response.AutoScalingGroups[0]
	return &awsInstanceGroup{group: group}, nil
}

// awsInstanceGroup implements InstanceGroupInfo
var _ InstanceGroupInfo = &awsInstanceGroup{}

type awsInstanceGroup struct {
	group *autoscaling.Group
}

// Implement InstanceGroupInfo.CurrentSize
// The number of instances currently running under control of this group
func (g *awsInstanceGroup) CurrentSize() (int, error) {
	return len(g.group.Instances), nil
}
