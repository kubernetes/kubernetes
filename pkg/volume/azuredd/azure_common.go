// +build !providerless

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

package azuredd

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	libstrings "strings"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-12-01/compute"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/legacy-cloud-providers/azure"
	utilstrings "k8s.io/utils/strings"
)

const (
	defaultStorageAccountType       = compute.StandardSSDLRS
	defaultAzureDiskKind            = v1.AzureManagedDisk
	defaultAzureDataDiskCachingMode = v1.AzureDataDiskCachingReadOnly
)

type dataDisk struct {
	volume.MetricsProvider
	volumeName string
	diskName   string
	podUID     types.UID
	plugin     *azureDataDiskPlugin
}

var (
	supportedCachingModes = sets.NewString(
		string(api.AzureDataDiskCachingNone),
		string(api.AzureDataDiskCachingReadOnly),
		string(api.AzureDataDiskCachingReadWrite))

	supportedDiskKinds = sets.NewString(
		string(api.AzureSharedBlobDisk),    // deprecated
		string(api.AzureDedicatedBlobDisk), // deprecated
		string(api.AzureManagedDisk))

	// only for Windows node
	winDiskNumRE = regexp.MustCompile(`/dev/disk(.+)`)
)

func getPath(uid types.UID, volName string, host volume.VolumeHost) string {
	return host.GetPodVolumeDir(uid, utilstrings.EscapeQualifiedName(azureDataDiskPluginName), volName)
}

// creates a unique path for disks (even if they share the same *.vhd name)
func makeGlobalPDPath(host volume.VolumeHost, diskURI string, isManaged bool) (string, error) {
	diskURI = libstrings.ToLower(diskURI) // always lower uri because users may enter it in caps.
	uniqueDiskNameTemplate := "%s%s"
	hashedDiskURI := azure.MakeCRC32(diskURI)
	prefix := "b"
	if isManaged {
		prefix = "m"
	}
	// "{m for managed b for blob}{hashed diskURI or DiskId depending on disk kind }"
	diskName := fmt.Sprintf(uniqueDiskNameTemplate, prefix, hashedDiskURI)
	pdPath := filepath.Join(host.GetPluginDir(azureDataDiskPluginName), util.MountsInGlobalPDPath, diskName)

	return pdPath, nil
}

func makeDataDisk(volumeName string, podUID types.UID, diskName string, host volume.VolumeHost, plugin *azureDataDiskPlugin) *dataDisk {
	var metricProvider volume.MetricsProvider
	if podUID != "" {
		metricProvider = volume.NewMetricsStatFS(getPath(podUID, volumeName, host))
	}

	return &dataDisk{
		MetricsProvider: metricProvider,
		volumeName:      volumeName,
		diskName:        diskName,
		podUID:          podUID,
		plugin:          plugin,
	}
}

func getVolumeSource(spec *volume.Spec) (volumeSource *v1.AzureDiskVolumeSource, readOnly bool, err error) {
	if spec.Volume != nil && spec.Volume.AzureDisk != nil {
		return spec.Volume.AzureDisk, spec.Volume.AzureDisk.ReadOnly != nil && *spec.Volume.AzureDisk.ReadOnly, nil
	}

	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.AzureDisk != nil {
		return spec.PersistentVolume.Spec.AzureDisk, spec.ReadOnly, nil
	}

	return nil, false, fmt.Errorf("azureDisk - Spec does not reference an Azure disk volume type")
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

func normalizeStorageAccountType(storageAccountType string) (compute.DiskStorageAccountTypes, error) {
	if storageAccountType == "" {
		return defaultStorageAccountType, nil
	}

	sku := compute.DiskStorageAccountTypes(storageAccountType)
	supportedSkuNames := compute.PossibleDiskStorageAccountTypesValues()
	for _, s := range supportedSkuNames {
		if sku == s {
			return sku, nil
		}
	}

	return "", fmt.Errorf("azureDisk - %s is not supported sku/storageaccounttype. Supported values are %s", storageAccountType, supportedSkuNames)
}

func normalizeCachingMode(cachingMode v1.AzureDataDiskCachingMode) (v1.AzureDataDiskCachingMode, error) {
	if cachingMode == "" {
		return defaultAzureDataDiskCachingMode, nil
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

// getDiskNum : extract the disk num from a device path,
// deviceInfo format could be like this: e.g. /dev/disk2
func getDiskNum(deviceInfo string) (string, error) {
	matches := winDiskNumRE.FindStringSubmatch(deviceInfo)
	if len(matches) == 2 {
		return matches[1], nil
	}
	return "", fmt.Errorf("cannot parse deviceInfo: %s, correct format: /dev/disk?", deviceInfo)
}
