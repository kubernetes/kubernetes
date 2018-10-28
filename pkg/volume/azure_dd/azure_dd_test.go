/*
Copyright 2015 The Kubernetes Authors.

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

package azure_dd

import (
	"os"
	"testing"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2018-10-01/compute"
	"github.com/Azure/go-autorest/autorest/to"
	"github.com/stretchr/testify/assert"

	"k8s.io/api/core/v1"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

func TestCanSupport(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("azure_dd")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil /* prober */, volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName(azureDataDiskPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	if plug.GetPluginName() != azureDataDiskPluginName {
		t.Errorf("Wrong name: %s", plug.GetPluginName())
	}
	if !plug.CanSupport(&volume.Spec{Volume: &v1.Volume{VolumeSource: v1.VolumeSource{AzureDisk: &v1.AzureDiskVolumeSource{}}}}) {
		t.Errorf("Expected true")
	}

	if !plug.CanSupport(&volume.Spec{PersistentVolume: &v1.PersistentVolume{Spec: v1.PersistentVolumeSpec{PersistentVolumeSource: v1.PersistentVolumeSource{AzureDisk: &v1.AzureDiskVolumeSource{}}}}}) {
		t.Errorf("Expected true")
	}
}

// fakeAzureProvider type was removed because all functions were not used
// Testing mounting will require path calculation which depends on the cloud provider, which is faked in the above test.

func TestGetMaxDataDiskCount(t *testing.T) {
	tests := []struct {
		instanceType string
		sizeList     *[]compute.VirtualMachineSize
		expectResult int64
	}{
		{
			instanceType: "standard_d2_v2",
			sizeList: &[]compute.VirtualMachineSize{
				{Name: to.StringPtr("Standard_D2_V2"), MaxDataDiskCount: to.Int32Ptr(8)},
				{Name: to.StringPtr("Standard_D3_V2"), MaxDataDiskCount: to.Int32Ptr(16)},
			},
			expectResult: 8,
		},
		{
			instanceType: "NOT_EXISTING",
			sizeList: &[]compute.VirtualMachineSize{
				{Name: to.StringPtr("Standard_D2_V2"), MaxDataDiskCount: to.Int32Ptr(8)},
			},
			expectResult: defaultAzureVolumeLimit,
		},
		{
			instanceType: "",
			sizeList:     &[]compute.VirtualMachineSize{},
			expectResult: defaultAzureVolumeLimit,
		},
	}

	for _, test := range tests {
		result := getMaxDataDiskCount(test.instanceType, test.sizeList)
		assert.Equal(t, test.expectResult, result)
	}
}
