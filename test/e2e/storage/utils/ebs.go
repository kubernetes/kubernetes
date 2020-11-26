/*
Copyright 2020 The Kubernetes Authors.

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

package utils

import (
	"fmt"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/ec2"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2"
)

const (
	volumeAttachmentStatusPollDelay = 2 * time.Second
	volumeAttachmentStatusFactor    = 2
	volumeAttachmentStatusSteps     = 6

	// represents expected attachment status of a volume after attach
	volumeAttachedStatus = "attached"

	// represents expected attachment status of a volume after detach
	volumeDetachedStatus = "detached"
)

// EBSUtil provides functions to interact with EBS volumes
type EBSUtil struct {
	client       *ec2.EC2
	validDevices []string
}

// NewEBSUtil returns an instance of EBSUtil which can be used to
// to interact with EBS volumes
func NewEBSUtil(client *ec2.EC2) *EBSUtil {
	ebsUtil := &EBSUtil{client: client}
	validDevices := []string{}
	for _, firstChar := range []rune{'b', 'c'} {
		for i := 'a'; i <= 'z'; i++ {
			dev := string([]rune{firstChar, i})
			validDevices = append(validDevices, dev)
		}
	}
	ebsUtil.validDevices = validDevices
	return ebsUtil
}

// AttachDisk attaches an EBS volume to a node.
func (ebs *EBSUtil) AttachDisk(volumeID string, nodeName string) error {
	instance, err := findInstanceByNodeName(nodeName, ebs.client)
	if err != nil {
		return fmt.Errorf("error finding node %s: %v", nodeName, err)
	}
	err = ebs.waitForAvailable(volumeID)
	if err != nil {
		return fmt.Errorf("error waiting volume %s to be available: %v", volumeID, err)
	}

	device, err := ebs.findFreeDevice(instance)
	if err != nil {
		return fmt.Errorf("error finding free device on node %s: %v", nodeName, err)
	}
	hostDevice := "/dev/xvd" + string(device)
	attachInput := &ec2.AttachVolumeInput{
		VolumeId:   &volumeID,
		InstanceId: instance.InstanceId,
		Device:     &hostDevice,
	}
	_, err = ebs.client.AttachVolume(attachInput)
	if err != nil {
		return fmt.Errorf("error attaching volume %s to node %s: %v", volumeID, nodeName, err)
	}
	return ebs.waitForAttach(volumeID)
}

func (ebs *EBSUtil) findFreeDevice(instance *ec2.Instance) (string, error) {
	deviceMappings := map[string]string{}

	for _, blockDevice := range instance.BlockDeviceMappings {
		name := aws.StringValue(blockDevice.DeviceName)
		name = strings.TrimPrefix(name, "/dev/sd")
		name = strings.TrimPrefix(name, "/dev/xvd")
		if len(name) < 1 || len(name) > 2 {
			klog.Warningf("Unexpected EBS DeviceName: %q", aws.StringValue(blockDevice.DeviceName))
		}

		deviceMappings[name] = aws.StringValue(blockDevice.Ebs.VolumeId)
	}

	for _, device := range ebs.validDevices {
		if _, found := deviceMappings[device]; !found {
			return device, nil
		}
	}
	return "", fmt.Errorf("no available device")
}

func (ebs *EBSUtil) waitForAttach(volumeID string) error {
	backoff := wait.Backoff{
		Duration: volumeAttachmentStatusPollDelay,
		Factor:   volumeAttachmentStatusFactor,
		Steps:    volumeAttachmentStatusSteps,
	}
	time.Sleep(volumeAttachmentStatusPollDelay)
	err := wait.ExponentialBackoff(backoff, func() (bool, error) {
		info, err := ebs.describeVolume(volumeID)
		if err != nil {
			return false, err
		}

		if len(info.Attachments) > 1 {
			// Shouldn't happen; log so we know if it is
			klog.Warningf("Found multiple attachments for volume %q: %v", volumeID, info)
		}
		attachmentStatus := ""
		for _, a := range info.Attachments {
			if attachmentStatus != "" {
				// Shouldn't happen; log so we know if it is
				klog.Warningf("Found multiple attachments for volume %q: %v", volumeID, info)
			}
			if a.State != nil {
				attachmentStatus = *a.State
			} else {
				// Shouldn't happen; log so we know if it is
				klog.Warningf("Ignoring nil attachment state for volume %q: %v", volumeID, a)
			}
		}
		if attachmentStatus == "" {
			attachmentStatus = volumeDetachedStatus
		}
		if attachmentStatus == volumeAttachedStatus {
			// Attachment is in requested state, finish waiting
			return true, nil
		}
		return false, nil
	})
	return err
}

func (ebs *EBSUtil) waitForAvailable(volumeID string) error {
	backoff := wait.Backoff{
		Duration: volumeAttachmentStatusPollDelay,
		Factor:   volumeAttachmentStatusFactor,
		Steps:    volumeAttachmentStatusSteps,
	}
	time.Sleep(volumeAttachmentStatusPollDelay)
	err := wait.ExponentialBackoff(backoff, func() (bool, error) {
		info, err := ebs.describeVolume(volumeID)
		if err != nil {
			return false, err
		}
		volumeState := aws.StringValue(info.State)
		if volumeState != ec2.VolumeStateAvailable {
			return false, nil
		}
		return true, nil
	})
	return err
}

// Gets the full information about this volume from the EC2 API
func (ebs *EBSUtil) describeVolume(volumeID string) (*ec2.Volume, error) {
	request := &ec2.DescribeVolumesInput{
		VolumeIds: []*string{&volumeID},
	}

	results := []*ec2.Volume{}
	var nextToken *string
	for {
		response, err := ebs.client.DescribeVolumes(request)
		if err != nil {
			return nil, err
		}

		results = append(results, response.Volumes...)

		nextToken = response.NextToken
		if aws.StringValue(nextToken) == "" {
			break
		}
		request.NextToken = nextToken
	}

	if len(results) == 0 {
		return nil, fmt.Errorf("no volumes found")
	}
	if len(results) > 1 {
		return nil, fmt.Errorf("multiple volumes found")
	}
	return results[0], nil
}

func newEc2Filter(name string, value string) *ec2.Filter {
	filter := &ec2.Filter{
		Name: aws.String(name),
		Values: []*string{
			aws.String(value),
		},
	}
	return filter
}

func findInstanceByNodeName(nodeName string, cloud *ec2.EC2) (*ec2.Instance, error) {
	filters := []*ec2.Filter{
		newEc2Filter("private-dns-name", nodeName),
	}

	request := &ec2.DescribeInstancesInput{
		Filters: filters,
	}

	instances, err := describeInstances(request, cloud)
	if err != nil {
		return nil, err
	}
	if len(instances) == 0 {
		return nil, nil
	}
	if len(instances) > 1 {
		return nil, fmt.Errorf("multiple instances found for name: %s", nodeName)
	}
	return instances[0], nil
}

func describeInstances(request *ec2.DescribeInstancesInput, cloud *ec2.EC2) ([]*ec2.Instance, error) {
	// Instances are paged
	results := []*ec2.Instance{}
	var nextToken *string

	for {
		response, err := cloud.DescribeInstances(request)
		if err != nil {
			return nil, fmt.Errorf("error listing AWS instances: %v", err)
		}

		for _, reservation := range response.Reservations {
			results = append(results, reservation.Instances...)
		}

		nextToken = response.NextToken
		if nextToken == nil || len(*nextToken) == 0 {
			break
		}
		request.NextToken = nextToken
	}

	return results, nil
}
