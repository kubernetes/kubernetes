// +build !providerless

/*
Copyright 2018 The Kubernetes Authors.

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
	"sync"
	"testing"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-07-01/compute"
	"github.com/Azure/go-autorest/autorest/to"
	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"
	cloudprovider "k8s.io/cloud-provider"
	azcache "k8s.io/legacy-cloud-providers/azure/cache"
)

func TestExtractVmssVMName(t *testing.T) {
	cases := []struct {
		description        string
		vmName             string
		expectError        bool
		expectedScaleSet   string
		expectedInstanceID string
	}{
		{
			description: "wrong vmss VM name should report error",
			vmName:      "vm1234",
			expectError: true,
		},
		{
			description: "wrong VM name separator should report error",
			vmName:      "vm-1234",
			expectError: true,
		},
		{
			description:        "correct vmss VM name should return correct scaleSet and instanceID",
			vmName:             "vm_1234",
			expectedScaleSet:   "vm",
			expectedInstanceID: "1234",
		},
		{
			description:        "correct vmss VM name with Extra Separator should return correct scaleSet and instanceID",
			vmName:             "vm_test_1234",
			expectedScaleSet:   "vm_test",
			expectedInstanceID: "1234",
		},
	}

	for _, c := range cases {
		ssName, instanceID, err := extractVmssVMName(c.vmName)
		if c.expectError {
			assert.Error(t, err, c.description)
			continue
		}

		assert.Equal(t, c.expectedScaleSet, ssName, c.description)
		assert.Equal(t, c.expectedInstanceID, instanceID, c.description)
	}
}

func TestVMSSVMCache(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	vmssName := "vmss"
	vmList := []string{"vmssee6c2000000", "vmssee6c2000001", "vmssee6c2000002"}
	ss, err := newTestScaleSet(ctrl, vmssName, "", 0, vmList)
	assert.NoError(t, err)

	// validate getting VMSS VM via cache.
	virtualMachines, rerr := ss.VirtualMachineScaleSetVMsClient.List(
		context.Background(), "rg", "vmss", "")
	assert.Nil(t, rerr)
	assert.Equal(t, 3, len(virtualMachines))
	for i := range virtualMachines {
		vm := virtualMachines[i]
		vmName := to.String(vm.OsProfile.ComputerName)
		ssName, instanceID, realVM, err := ss.getVmssVM(vmName, azcache.CacheReadTypeDefault)
		assert.Nil(t, err)
		assert.Equal(t, "vmss", ssName)
		assert.Equal(t, to.String(vm.InstanceID), instanceID)
		assert.Equal(t, &vm, realVM)
	}

	// validate deleteCacheForNode().
	vm := virtualMachines[0]
	vmName := to.String(vm.OsProfile.ComputerName)
	err = ss.deleteCacheForNode(vmName)
	assert.NoError(t, err)

	// the VM should be removed from cache after deleteCacheForNode().
	cacheKey, cache, err := ss.getVMSSVMCache("rg", vmssName)
	assert.NoError(t, err)
	cached, err := cache.Get(cacheKey, azcache.CacheReadTypeDefault)
	assert.NoError(t, err)
	cachedVirtualMachines := cached.(*sync.Map)
	_, ok := cachedVirtualMachines.Load(vmName)
	assert.Equal(t, false, ok)

	// the VM should be get back after another cache refresh.
	ssName, instanceID, realVM, err := ss.getVmssVM(vmName, azcache.CacheReadTypeDefault)
	assert.NoError(t, err)
	assert.Equal(t, "vmss", ssName)
	assert.Equal(t, to.String(vm.InstanceID), instanceID)
	assert.Equal(t, &vm, realVM)
}

func TestVMSSVMCacheWithDeletingNodes(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	vmssName := "vmss"
	vmList := []string{"vmssee6c2000000", "vmssee6c2000001", "vmssee6c2000002"}
	ss, err := newTestScaleSetWithState(ctrl, vmssName, "", 0, vmList, "Deleting")
	assert.NoError(t, err)

	virtualMachines, rerr := ss.VirtualMachineScaleSetVMsClient.List(
		context.Background(), "rg", "vmss", "")
	assert.Nil(t, rerr)
	assert.Equal(t, 3, len(virtualMachines))
	for i := range virtualMachines {
		vm := virtualMachines[i]
		vmName := to.String(vm.OsProfile.ComputerName)
		assert.Equal(t, vm.ProvisioningState, to.StringPtr(string(compute.ProvisioningStateDeleting)))

		ssName, instanceID, realVM, err := ss.getVmssVM(vmName, azcache.CacheReadTypeDefault)
		assert.Nil(t, realVM)
		assert.Equal(t, "", ssName)
		assert.Equal(t, instanceID, ssName)
		assert.Equal(t, cloudprovider.InstanceNotFound, err)
	}
}
