/*
Copyright 2016 The Kubernetes Authors.

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
	"net/url"
	"regexp"
	"strings"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/ec2"
	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/types"
)

// awsVolumeRegMatch represents Regex Match for AWS volume.
var awsVolumeRegMatch = regexp.MustCompile("^vol-[^/]*$")

// awsVolumeID represents the ID of the volume in the AWS API, e.g. vol-12345678
// The "traditional" format is "vol-12345678"
// A new longer format is also being introduced: "vol-12345678abcdef01"
// We should not assume anything about the length or format, though it seems
// reasonable to assume that volumes will continue to start with "vol-".
type awsVolumeID string

func (i awsVolumeID) awsString() *string {
	return aws.String(string(i))
}

// KubernetesVolumeID represents the id for a volume in the kubernetes API;
// a few forms are recognized:
//  * aws://<zone>/<awsVolumeId>
//  * aws:///<awsVolumeId>
//  * <awsVolumeId>
type KubernetesVolumeID string

// DiskInfo returns aws disk information in easy to use manner
type diskInfo struct {
	ec2Instance     *ec2.Instance
	nodeName        types.NodeName
	volumeState     string
	attachmentState string
	hasAttachment   bool
	disk            *awsDisk
}

// MapToAWSVolumeID extracts the awsVolumeID from the KubernetesVolumeID
func (name KubernetesVolumeID) MapToAWSVolumeID() (awsVolumeID, error) {
	// name looks like aws://availability-zone/awsVolumeId

	// The original idea of the URL-style name was to put the AZ into the
	// host, so we could find the AZ immediately from the name without
	// querying the API.  But it turns out we don't actually need it for
	// multi-AZ clusters, as we put the AZ into the labels on the PV instead.
	// However, if in future we want to support multi-AZ cluster
	// volume-awareness without using PersistentVolumes, we likely will
	// want the AZ in the host.

	s := string(name)

	if !strings.HasPrefix(s, "aws://") {
		// Assume a bare aws volume id (vol-1234...)
		// Build a URL with an empty host (AZ)
		s = "aws://" + "" + "/" + s
	}
	url, err := url.Parse(s)
	if err != nil {
		// TODO: Maybe we should pass a URL into the Volume functions
		return "", fmt.Errorf("Invalid disk name (%s): %v", name, err)
	}
	if url.Scheme != "aws" {
		return "", fmt.Errorf("Invalid scheme for AWS volume (%s)", name)
	}

	awsID := url.Path
	awsID = strings.Trim(awsID, "/")

	// We sanity check the resulting volume; the two known formats are
	// vol-12345678 and vol-12345678abcdef01
	if !awsVolumeRegMatch.MatchString(awsID) {
		return "", fmt.Errorf("Invalid format for AWS volume (%s)", name)
	}

	return awsVolumeID(awsID), nil
}

func GetAWSVolumeID(kubeVolumeID string) (string, error) {
	kid := KubernetesVolumeID(kubeVolumeID)
	awsID, err := kid.MapToAWSVolumeID()
	return string(awsID), err
}

func (c *Cloud) checkIfAttachedToNode(diskName KubernetesVolumeID, nodeName types.NodeName) (*diskInfo, bool, error) {
	disk, err := newAWSDisk(c, diskName)

	if err != nil {
		return nil, true, err
	}

	awsDiskInfo := &diskInfo{
		disk: disk,
	}

	info, err := disk.describeVolume()

	if err != nil {
		glog.Warningf("Error describing volume %s with %v", diskName, err)
		awsDiskInfo.volumeState = "unknown"
		return awsDiskInfo, false, err
	}

	awsDiskInfo.volumeState = aws.StringValue(info.State)

	if len(info.Attachments) > 0 {
		attachment := info.Attachments[0]
		awsDiskInfo.attachmentState = aws.StringValue(attachment.State)
		instanceID := aws.StringValue(attachment.InstanceId)
		instanceInfo, err := c.getInstanceByID(instanceID)

		// This should never happen but if it does it could mean there was a race and instance
		// has been deleted
		if err != nil {
			fetchErr := fmt.Errorf("Error fetching instance %s for volume %s", instanceID, diskName)
			glog.Warning(fetchErr)
			return awsDiskInfo, false, fetchErr
		}

		awsDiskInfo.ec2Instance = instanceInfo
		awsDiskInfo.nodeName = mapInstanceToNodeName(instanceInfo)
		awsDiskInfo.hasAttachment = true
		if awsDiskInfo.nodeName == nodeName {
			return awsDiskInfo, true, nil
		}
	}
	return awsDiskInfo, false, nil
}
