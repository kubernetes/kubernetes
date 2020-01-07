// +build !providerless

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

package azure

import (
	"context"
	"testing"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-07-01/compute"
	"github.com/Azure/go-autorest/autorest/to"
	"github.com/stretchr/testify/assert"
)

func TestAttachDisk(t *testing.T) {
	tests := []struct {
		desc                string
		isManagedDisk       bool
		diskName            string
		diskURI             string
		diskEncryptionSetID string
		expectedVMSSVM      compute.VirtualMachineScaleSetVM
	}{
		{
			desc:                "AttachDisk should attach managed disk to VMSS VM",
			isManagedDisk:       true,
			diskName:            "disk-1",
			diskURI:             "uri",
			diskEncryptionSetID: "id",
			expectedVMSSVM: compute.VirtualMachineScaleSetVM{
				Location: to.StringPtr("westus"),
				VirtualMachineScaleSetVMProperties: &compute.VirtualMachineScaleSetVMProperties{
					StorageProfile: &compute.StorageProfile{
						DataDisks: &[]compute.DataDisk{
							{
								Lun:          to.Int32Ptr(2),
								Name:         to.StringPtr("disk-2"),
								Caching:      "None",
								CreateOption: "attach",
								Vhd: &compute.VirtualHardDisk{
									URI: to.StringPtr("uri"),
								},
							},
							{
								Lun:          to.Int32Ptr(3),
								Name:         to.StringPtr("disk-3"),
								Caching:      "None",
								CreateOption: "attach",
								ManagedDisk: &compute.ManagedDiskParameters{
									ID:                to.StringPtr("uri"),
									DiskEncryptionSet: &compute.DiskEncryptionSetParameters{ID: to.StringPtr("id")},
								},
							},
							{
								Lun:          to.Int32Ptr(0),
								Name:         to.StringPtr("disk-1"),
								Caching:      "None",
								CreateOption: "attach",
								ManagedDisk: &compute.ManagedDiskParameters{
									ID:                to.StringPtr("uri"),
									DiskEncryptionSet: &compute.DiskEncryptionSetParameters{ID: to.StringPtr("id")},
								},
							},
						},
					},
				},
			},
		},
		{
			desc:                "AttachDisk should attach non-managed disk to VMSS VM",
			isManagedDisk:       false,
			diskName:            "disk-1",
			diskURI:             "uri",
			diskEncryptionSetID: "id",
			expectedVMSSVM: compute.VirtualMachineScaleSetVM{
				Location: to.StringPtr("westus"),
				VirtualMachineScaleSetVMProperties: &compute.VirtualMachineScaleSetVMProperties{
					StorageProfile: &compute.StorageProfile{
						DataDisks: &[]compute.DataDisk{
							{
								Lun:          to.Int32Ptr(2),
								Name:         to.StringPtr("disk-2"),
								Caching:      "None",
								CreateOption: "attach",
								Vhd: &compute.VirtualHardDisk{
									URI: to.StringPtr("uri"),
								},
							},
							{
								Lun:          to.Int32Ptr(3),
								Name:         to.StringPtr("disk-3"),
								Caching:      "None",
								CreateOption: "attach",
								ManagedDisk: &compute.ManagedDiskParameters{
									ID:                to.StringPtr("uri"),
									DiskEncryptionSet: &compute.DiskEncryptionSetParameters{ID: to.StringPtr("id")},
								},
							},
							{
								Lun:          to.Int32Ptr(0),
								Name:         to.StringPtr("disk-1"),
								Caching:      "None",
								CreateOption: "attach",
								Vhd: &compute.VirtualHardDisk{
									URI: to.StringPtr("uri"),
								},
							},
						},
					},
				},
			},
		},
	}

	for _, test := range tests {
		scaleSet, err := newTestScaleSet("testScaleSet", "zone", 1, []string{"vmss-00000000"})
		assert.NoError(t, err)

		fakeVMSSVMClient := scaleSet.cloud.VirtualMachineScaleSetVMsClient.(*fakeVirtualMachineScaleSetVMsClient)

		_ = scaleSet.AttachDisk(test.isManagedDisk, test.diskName, test.diskURI, "vmss-00000000", 0, compute.CachingTypesNone, test.diskEncryptionSetID)
		vms, _ := fakeVMSSVMClient.List(context.Background(), scaleSet.cloud.ResourceGroup, "testScaleSet", "", "", "")
		assert.Equal(t, test.expectedVMSSVM, vms[0], test.desc)
	}
}

func TestDetachDisk(t *testing.T) {
	tests := []struct {
		desc           string
		diskName       string
		diskURI        string
		expectedVMSSVM compute.VirtualMachineScaleSetVM
	}{
		{
			desc:     "Detach disk should detach disk by disk name",
			diskName: "disk-2",
			expectedVMSSVM: compute.VirtualMachineScaleSetVM{
				Location: to.StringPtr("westus"),
				VirtualMachineScaleSetVMProperties: &compute.VirtualMachineScaleSetVMProperties{
					StorageProfile: &compute.StorageProfile{
						DataDisks: &[]compute.DataDisk{
							{
								Lun:          to.Int32Ptr(3),
								Name:         to.StringPtr("disk-3"),
								Caching:      "None",
								CreateOption: "attach",
								ManagedDisk: &compute.ManagedDiskParameters{
									ID:                to.StringPtr("uri"),
									DiskEncryptionSet: &compute.DiskEncryptionSetParameters{ID: to.StringPtr("id")},
								},
							},
						},
					},
				},
			},
		},
		{
			desc:    "Detach disk should detach disk by the URI of the VHD or the ID of the managed disk",
			diskURI: "uri",
			expectedVMSSVM: compute.VirtualMachineScaleSetVM{
				Location: to.StringPtr("westus"),
				VirtualMachineScaleSetVMProperties: &compute.VirtualMachineScaleSetVMProperties{
					StorageProfile: &compute.StorageProfile{
						DataDisks: &[]compute.DataDisk{
							{
								Lun:          to.Int32Ptr(3),
								Name:         to.StringPtr("disk-3"),
								Caching:      "None",
								CreateOption: "attach",
								ManagedDisk: &compute.ManagedDiskParameters{
									ID:                to.StringPtr("uri"),
									DiskEncryptionSet: &compute.DiskEncryptionSetParameters{ID: to.StringPtr("id")},
								},
							},
						},
					},
				},
			},
		},
	}

	for _, test := range tests {
		scaleSet, err := newTestScaleSet("testScaleSet", "zone", 1, []string{"vmss-00000000"})
		assert.NoError(t, err)

		fakeVMSSVMClient := scaleSet.cloud.VirtualMachineScaleSetVMsClient.(*fakeVirtualMachineScaleSetVMsClient)

		_ = scaleSet.DetachDisk(test.diskName, test.diskURI, "vmss-00000000")
		vms, _ := fakeVMSSVMClient.List(context.Background(), scaleSet.cloud.ResourceGroup, "testScaleSet", "", "", "")
		assert.Equal(t, test.expectedVMSSVM, vms[0], test.desc)
	}
}
