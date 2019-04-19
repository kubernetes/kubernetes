/*
Copyright 2019 The Kubernetes Authors.

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

package azure

import (
	"fmt"
	"testing"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2018-10-01/compute"
)

func TestAttachDisk(t *testing.T) {
	c := getTestCloud()

	common := &controllerCommon{
		location:              c.Location,
		storageEndpointSuffix: c.Environment.StorageEndpointSuffix,
		resourceGroup:         c.ResourceGroup,
		subscriptionID:        c.SubscriptionID,
		cloud:                 c,
	}

	diskURI := fmt.Sprintf("/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Compute/disks/disk-name", c.SubscriptionID, c.ResourceGroup)

	err := common.AttachDisk(true, "", diskURI, "node1", compute.CachingTypesReadOnly)
	if err != nil {
		fmt.Printf("TestAttachDisk return expected error: %v", err)
	} else {
		t.Errorf("TestAttachDisk unexpected nil err")
	}
}

func TestDetachDisk(t *testing.T) {
	c := getTestCloud()

	common := &controllerCommon{
		location:              c.Location,
		storageEndpointSuffix: c.Environment.StorageEndpointSuffix,
		resourceGroup:         c.ResourceGroup,
		subscriptionID:        c.SubscriptionID,
		cloud:                 c,
	}

	diskURI := fmt.Sprintf("/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Compute/disks/disk-name", c.SubscriptionID, c.ResourceGroup)

	err := common.DetachDisk("", diskURI, "node1")
	if err != nil {
		fmt.Printf("TestAttachDisk return expected error: %v", err)
	} else {
		t.Errorf("TestAttachDisk unexpected nil err")
	}
}
