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

package azure_dd

import (
	"fmt"
	"io/ioutil"
	"os"
	"path"
	libstrings "strings"

	storage "github.com/Azure/azure-sdk-for-go/arm/storage"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/azure"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
)

const (
	defaultFSType             = "ext4"
	defaultStorageAccountType = storage.StandardLRS
	defaultAzureDiskKind      = v1.AzureSharedBlobDisk
)

type dataDisk struct {
	volume.MetricsProvider
	volumeName string
	diskName   string
	podUID     types.UID
}

var (
	supportedCachingModes = sets.NewString(
		string(api.AzureDataDiskCachingNone),
		string(api.AzureDataDiskCachingReadOnly),
		string(api.AzureDataDiskCachingReadWrite))

	supportedDiskKinds = sets.NewString(
		string(api.AzureSharedBlobDisk),
		string(api.AzureDedicatedBlobDisk),
		string(api.AzureManagedDisk))

	supportedStorageAccountTypes = sets.NewString("Premium_LRS", "Standard_LRS")
)

func getPath(uid types.UID, volName string, host volume.VolumeHost) string {
	return host.GetPodVolumeDir(uid, strings.EscapeQualifiedNameForDisk(azureDataDiskPluginName), volName)
}

// creates a unique path for disks (even if they share the same *.vhd name)
func makeGlobalPDPath(host volume.VolumeHost, diskUri string, isManaged bool) (string, error) {
	diskUri = libstrings.ToLower(diskUri) // always lower uri because users may enter it in caps.
	uniqueDiskNameTemplate := "%s%s"
	hashedDiskUri := azure.MakeCRC32(diskUri)
	prefix := "b"
	if isManaged {
		prefix = "m"
	}
	// "{m for managed b for blob}{hashed diskUri or DiskId depending on disk kind }"
	diskName := fmt.Sprintf(uniqueDiskNameTemplate, prefix, hashedDiskUri)
	pdPath := path.Join(host.GetPluginDir(azureDataDiskPluginName), mount.MountsInGlobalPDPath, diskName)

	return pdPath, nil
}

func makeDataDisk(volumeName string, podUID types.UID, diskName string, host volume.VolumeHost) *dataDisk {
	var metricProvider volume.MetricsProvider
	if podUID != "" {
		metricProvider = volume.NewMetricsStatFS(getPath(podUID, volumeName, host))
	}

	return &dataDisk{
		MetricsProvider: metricProvider,
		volumeName:      volumeName,
		diskName:        diskName,
		podUID:          podUID,
	}
}

func getVolumeSource(spec *volume.Spec) (*v1.AzureDiskVolumeSource, error) {
	if spec.Volume != nil && spec.Volume.AzureDisk != nil {
		return spec.Volume.AzureDisk, nil
	}

	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.AzureDisk != nil {
		return spec.PersistentVolume.Spec.AzureDisk, nil
	}

	return nil, fmt.Errorf("azureDisk - Spec does not reference an Azure disk volume type")
}

func normalizeFsType(fsType string) string {
	if fsType == "" {
		return defaultFSType
	}

	return fsType
}

func normalizeKind(kind string) (v1.AzureDataDiskKind, error) {
	if kind == "" {
		return defaultAzureDiskKind, nil
	}

	if !supportedDiskKinds.Has(kind) {
		return "", fmt.Errorf("azureDisk - %s is not supported disk kind. Supported values are %s", kind, supportedDiskKinds.List())
	}

	return v1.AzureDataDiskKind(kind), nil
}

func normalizeStorageAccountType(storageAccountType string) (storage.SkuName, error) {
	if storageAccountType == "" {
		return defaultStorageAccountType, nil
	}

	if !supportedStorageAccountTypes.Has(storageAccountType) {
		return "", fmt.Errorf("azureDisk - %s is not supported sku/storageaccounttype. Supported values are %s", storageAccountType, supportedStorageAccountTypes.List())
	}

	return storage.SkuName(storageAccountType), nil
}

func normalizeCachingMode(cachingMode v1.AzureDataDiskCachingMode) (v1.AzureDataDiskCachingMode, error) {
	if cachingMode == "" {
		return v1.AzureDataDiskCachingReadWrite, nil
	}

	if !supportedCachingModes.Has(string(cachingMode)) {
		return "", fmt.Errorf("azureDisk - %s is not supported cachingmode. Supported values are %s", cachingMode, supportedCachingModes.List())
	}

	return cachingMode, nil
}

type ioHandler interface {
	ReadDir(dirname string) ([]os.FileInfo, error)
	WriteFile(filename string, data []byte, perm os.FileMode) error
	Readlink(name string) (string, error)
	ReadFile(filename string) ([]byte, error)
}

//TODO: check if priming the iscsi interface is actually needed

type osIOHandler struct{}

func (handler *osIOHandler) ReadDir(dirname string) ([]os.FileInfo, error) {
	return ioutil.ReadDir(dirname)
}

func (handler *osIOHandler) WriteFile(filename string, data []byte, perm os.FileMode) error {
	return ioutil.WriteFile(filename, data, perm)
}

func (handler *osIOHandler) Readlink(name string) (string, error) {
	return os.Readlink(name)
}

func (handler *osIOHandler) ReadFile(filename string) ([]byte, error) {
	return ioutil.ReadFile(filename)
}

func getDiskController(host volume.VolumeHost) (DiskController, error) {
	cloudProvider := host.GetCloudProvider()
	az, ok := cloudProvider.(*azure.Cloud)

	if !ok || az == nil {
		return nil, fmt.Errorf("AzureDisk -  failed to get Azure Cloud Provider. GetCloudProvider returned %v instead", cloudProvider)
	}
	return az, nil
}

func getCloud(host volume.VolumeHost) (*azure.Cloud, error) {
	cloudProvider := host.GetCloudProvider()
	az, ok := cloudProvider.(*azure.Cloud)

	if !ok || az == nil {
		return nil, fmt.Errorf("AzureDisk -  failed to get Azure Cloud Provider. GetCloudProvider returned %v instead", cloudProvider)
	}
	return az, nil
}

func strFirstLetterToUpper(str string) string {
	if len(str) < 2 {
		return str
	}
	return libstrings.ToUpper(string(str[0])) + str[1:]
}
