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
	gcecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
	"k8s.io/kubernetes/pkg/util"
)

// These tests check functionality of GCE cloud provider w.r.t. disk
// provisioning.

func createGCEDisk(diskType string) error {

	gceCloud, err := getGCECloud()
	if err != nil {
		return fmt.Errorf("Error getting GCE provisioner: %v", err)
	}

	pdName := fmt.Sprintf("%s-%s", testContext.prefix, string(util.NewUUID()))
	err = gceCloud.CreateDisk(pdName, testContext.CloudConfig.Zone, 10 /*GiB*/, diskType, nil)
	if err != nil {
		return fmt.Errorf("Failed to create GCE disk: %v", err)
	}
	glog.Infof("Created disk %s", pdName)
	defer func() {
		gceCloud.DeleteDisk(pdName)
		glog.Infof("Deleted disk %s", pdName)
	}()

	size, createdDiskType, err := gceCloud.GetDiskProperties(pdName, testContext.CloudConfig.Zone)
	if err != nil {
		return fmt.Errorf("Failed to get disk properties: %v", err)
	}
	if size != 10 {
		return fmt.Errorf("Expected disk size 10 GiB, got %d", size)
	}
	if createdDiskType != diskType {
		return fmt.Errorf("Expected disk type %s, got %s", diskType, createdDiskType)
	}
	return nil
}

var _ = Describe("GCE PD cloud provider", func() {
	It("should provison SSD disk", func() {
		SkipUnlessProviderIs("gce", "gke")
		err := createGCEDisk(gcecloud.DiskTypeSSD)
		expectNoError(err, "Error creating GCE SSD disk")
	})

	It("should provison standard disk", func() {
		SkipUnlessProviderIs("gce", "gke")
		err := createGCEDisk(gcecloud.DiskTypeStandard)
		expectNoError(err, "Error creating GCE standard disk")
	})
})
