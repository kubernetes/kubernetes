/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package e2e

import (
	"fmt"

	"github.com/golang/glog"
	. "github.com/onsi/ginkgo"
	//. "github.com/onsi/gomega"
	awscloud "k8s.io/kubernetes/pkg/cloudprovider/providers/aws"
)

// These tests check functionality of AWS cloud provider w.r.t. disk
// provisioning.

func createAWSVolume(volumeType string, iops int64) error {

	volumes, ok := testContext.CloudConfig.Provider.(awscloud.Volumes)
	if !ok {
		return fmt.Errorf("Provider does not support volumes")
	}

	opts := awscloud.VolumeOptions{
		CapacityGB: 10,
		VolumeType: volumeType,
		IOPS:       iops,
	}

	volName, err := volumes.CreateDisk(&opts)
	if err != nil {
		return fmt.Errorf("Failed to create AWS volume: %v", err)
	}
	glog.Infof("Created disk %s", volName)
	defer func() {
		volumes.DeleteDisk(volName)
		glog.Infof("Deleted disk %s", volName)
	}()

	opts, err = volumes.GetDiskProperties(volName)
	if err != nil {
		return fmt.Errorf("Failed to get disk properties: %v", err)
	}
	if opts.CapacityGB != 10 {
		return fmt.Errorf("Expected disk size 10 GiB, got %d", opts.CapacityGB)
	}
	if opts.VolumeType != volumeType {
		return fmt.Errorf("Expected disk type %s, got %s", volumeType, opts.VolumeType)
	}
	if iops != 0 && opts.IOPS != iops {
		return fmt.Errorf("Expected IOPS %d, got %d", iops, opts.IOPS)
	}

	return nil
}

var _ = Describe("AWS cloud provider", func() {
	It("should provison SSD disk", func() {
		SkipUnlessProviderIs("aws")
		err := createAWSVolume(awscloud.EBSVolumeTypeGP2, 0)
		expectNoError(err, "Error creating AWS SSD disk")
	})

	It("should provison standard disk", func() {
		SkipUnlessProviderIs("aws")
		err := createAWSVolume(awscloud.EBSVolumeTypeStandard, 0)
		expectNoError(err, "Error creating AWS standard disk")
	})
	It("should provison IOPS disk", func() {
		SkipUnlessProviderIs("aws")
		err := createAWSVolume(awscloud.EBSVolumeTypeIO1, 300)
		expectNoError(err, "Error creating AWS IOPS disk")
	})
})
