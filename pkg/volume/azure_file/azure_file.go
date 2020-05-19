// +build !providerless

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

package azure_file

import (
	"fmt"
	"io/ioutil"
	"os"
	"runtime"
	"time"

	"k8s.io/klog/v2"
	"k8s.io/utils/mount"
	utilstrings "k8s.io/utils/strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	cloudprovider "k8s.io/cloud-provider"
	volumehelpers "k8s.io/cloud-provider/volume/helpers"
	"k8s.io/kubernetes/pkg/volume"
	volutil "k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/legacy-cloud-providers/azure"
)

// ProbeVolumePlugins is the primary endpoint for volume plugins
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&azureFilePlugin{nil}}
}

type azureFilePlugin struct {
	host volume.VolumeHost
}

var _ volume.VolumePlugin = &azureFilePlugin{}
var _ volume.PersistentVolumePlugin = &azureFilePlugin{}
var _ volume.ExpandableVolumePlugin = &azureFilePlugin{}

const (
	azureFilePluginName = "kubernetes.io/azure-file"
)

func getPath(uid types.UID, volName string, host volume.VolumeHost) string {
	return host.GetPodVolumeDir(uid, utilstrings.EscapeQualifiedName(azureFilePluginName), volName)
}

func (plugin *azureFilePlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

func (plugin *azureFilePlugin) GetPluginName() string {
	return azureFilePluginName
}

func (plugin *azureFilePlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	share, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	return share, nil
}

func (plugin *azureFilePlugin) CanSupport(spec *volume.Spec) bool {
	//TODO: check if mount.cifs is there
	return (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.AzureFile != nil) ||
		(spec.Volume != nil && spec.Volume.AzureFile != nil)
}

func (plugin *azureFilePlugin) RequiresRemount() bool {
	return false
}

func (plugin *azureFilePlugin) SupportsMountOption() bool {
	return true
}

func (plugin *azureFilePlugin) SupportsBulkVolumeVerification() bool {
	return false
}

func (plugin *azureFilePlugin) GetAccessModes() []v1.PersistentVolumeAccessMode {
	return []v1.PersistentVolumeAccessMode{
		v1.ReadWriteOnce,
		v1.ReadOnlyMany,
		v1.ReadWriteMany,
	}
}

func (plugin *azureFilePlugin) NewMounter(spec *volume.Spec, pod *v1.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
	return plugin.newMounterInternal(spec, pod, &azureSvc{}, plugin.host.GetMounter(plugin.GetPluginName()))
}

func (plugin *azureFilePlugin) newMounterInternal(spec *volume.Spec, pod *v1.Pod, util azureUtil, mounter mount.Interface) (volume.Mounter, error) {
	share, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}
	secretName, secretNamespace, err := getSecretNameAndNamespace(spec, pod.Namespace)
	if err != nil {
		// Log-and-continue instead of returning an error for now
		// due to unspecified backwards compatibility concerns (a subject to revise)
		klog.Errorf("get secret name and namespace from pod(%s) return with error: %v", pod.Name, err)
	}
	return &azureFileMounter{
		azureFile: &azureFile{
			volName:         spec.Name(),
			mounter:         mounter,
			pod:             pod,
			plugin:          plugin,
			MetricsProvider: volume.NewMetricsStatFS(getPath(pod.UID, spec.Name(), plugin.host)),
		},
		util:            util,
		secretNamespace: secretNamespace,
		secretName:      secretName,
		shareName:       share,
		readOnly:        readOnly,
		mountOptions:    volutil.MountOptionFromSpec(spec),
	}, nil
}

func (plugin *azureFilePlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return plugin.newUnmounterInternal(volName, podUID, plugin.host.GetMounter(plugin.GetPluginName()))
}

func (plugin *azureFilePlugin) newUnmounterInternal(volName string, podUID types.UID, mounter mount.Interface) (volume.Unmounter, error) {
	return &azureFileUnmounter{&azureFile{
		volName:         volName,
		mounter:         mounter,
		pod:             &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: podUID}},
		plugin:          plugin,
		MetricsProvider: volume.NewMetricsStatFS(getPath(podUID, volName, plugin.host)),
	}}, nil
}

func (plugin *azureFilePlugin) RequiresFSResize() bool {
	return false
}

func (plugin *azureFilePlugin) ExpandVolumeDevice(
	spec *volume.Spec,
	newSize resource.Quantity,
	oldSize resource.Quantity) (resource.Quantity, error) {

	if spec.PersistentVolume == nil || spec.PersistentVolume.Spec.AzureFile == nil {
		return oldSize, fmt.Errorf("invalid PV spec")
	}
	shareName := spec.PersistentVolume.Spec.AzureFile.ShareName
	azure, resourceGroup, err := getAzureCloudProvider(plugin.host.GetCloudProvider())
	if err != nil {
		return oldSize, err
	}
	if spec.PersistentVolume.ObjectMeta.Annotations[resourceGroupAnnotation] != "" {
		resourceGroup = spec.PersistentVolume.ObjectMeta.Annotations[resourceGroupAnnotation]
	}

	secretName, secretNamespace, err := getSecretNameAndNamespace(spec, spec.PersistentVolume.Spec.ClaimRef.Namespace)
	if err != nil {
		return oldSize, err
	}

	accountName, _, err := (&azureSvc{}).GetAzureCredentials(plugin.host, secretNamespace, secretName)
	if err != nil {
		return oldSize, err
	}

	requestGiB, err := volumehelpers.RoundUpToGiBInt(newSize)

	if err != nil {
		return oldSize, err
	}

	if err := azure.ResizeFileShare(resourceGroup, accountName, shareName, requestGiB); err != nil {
		return oldSize, err
	}

	return newSize, nil
}

func (plugin *azureFilePlugin) ConstructVolumeSpec(volName, mountPath string) (*volume.Spec, error) {
	azureVolume := &v1.Volume{
		Name: volName,
		VolumeSource: v1.VolumeSource{
			AzureFile: &v1.AzureFileVolumeSource{
				SecretName: volName,
				ShareName:  volName,
			},
		},
	}
	return volume.NewSpecFromVolume(azureVolume), nil
}

// azureFile volumes represent mount of an AzureFile share.
type azureFile struct {
	volName string
	podUID  types.UID
	pod     *v1.Pod
	mounter mount.Interface
	plugin  *azureFilePlugin
	volume.MetricsProvider
}

func (azureFileVolume *azureFile) GetPath() string {
	return getPath(azureFileVolume.pod.UID, azureFileVolume.volName, azureFileVolume.plugin.host)
}

type azureFileMounter struct {
	*azureFile
	util            azureUtil
	secretName      string
	secretNamespace string
	shareName       string
	readOnly        bool
	mountOptions    []string
}

var _ volume.Mounter = &azureFileMounter{}

func (b *azureFileMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        b.readOnly,
		Managed:         !b.readOnly,
		SupportsSELinux: false,
	}
}

// Checks prior to mount operations to verify that the required components (binaries, etc.)
// to mount the volume are available on the underlying node.
// If not, it returns an error
func (b *azureFileMounter) CanMount() error {
	return nil
}

// SetUp attaches the disk and bind mounts to the volume path.
func (b *azureFileMounter) SetUp(mounterArgs volume.MounterArgs) error {
	return b.SetUpAt(b.GetPath(), mounterArgs)
}

func (b *azureFileMounter) SetUpAt(dir string, mounterArgs volume.MounterArgs) error {
	notMnt, err := b.mounter.IsLikelyNotMountPoint(dir)
	klog.V(4).Infof("AzureFile mount set up: %s %v %v", dir, !notMnt, err)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	if !notMnt {
		// testing original mount point, make sure the mount link is valid
		if _, err := ioutil.ReadDir(dir); err == nil {
			klog.V(4).Infof("azureFile - already mounted to target %s", dir)
			return nil
		}
		// mount link is invalid, now unmount and remount later
		klog.Warningf("azureFile - ReadDir %s failed with %v, unmount this directory", dir, err)
		if err := b.mounter.Unmount(dir); err != nil {
			klog.Errorf("azureFile - Unmount directory %s failed with %v", dir, err)
			return err
		}
	}

	var accountKey, accountName string
	if accountName, accountKey, err = b.util.GetAzureCredentials(b.plugin.host, b.secretNamespace, b.secretName); err != nil {
		return err
	}

	var mountOptions []string
	var sensitiveMountOptions []string
	source := ""
	osSeparator := string(os.PathSeparator)
	source = fmt.Sprintf("%s%s%s.file.%s%s%s", osSeparator, osSeparator, accountName, getStorageEndpointSuffix(b.plugin.host.GetCloudProvider()), osSeparator, b.shareName)

	if runtime.GOOS == "windows" {
		sensitiveMountOptions = []string{fmt.Sprintf("AZURE\\%s", accountName), accountKey}
	} else {
		if err := os.MkdirAll(dir, 0700); err != nil {
			return err
		}
		// parameters suggested by https://azure.microsoft.com/en-us/documentation/articles/storage-how-to-use-files-linux/
		options := []string{}
		sensitiveMountOptions = []string{fmt.Sprintf("username=%s,password=%s", accountName, accountKey)}
		if b.readOnly {
			options = append(options, "ro")
		}
		mountOptions = volutil.JoinMountOptions(b.mountOptions, options)
		mountOptions = appendDefaultMountOptions(mountOptions, mounterArgs.FsGroup)
	}

	mountComplete := false
	err = wait.Poll(5*time.Second, 10*time.Minute, func() (bool, error) {
		err := b.mounter.MountSensitive(source, dir, "cifs", mountOptions, sensitiveMountOptions)
		mountComplete = true
		return true, err
	})
	if !mountComplete {
		return fmt.Errorf("volume(%s) mount on %s timeout(10m)", source, dir)
	}
	if err != nil {
		notMnt, mntErr := b.mounter.IsLikelyNotMountPoint(dir)
		if mntErr != nil {
			klog.Errorf("IsLikelyNotMountPoint check failed: %v", mntErr)
			return err
		}
		if !notMnt {
			if mntErr = b.mounter.Unmount(dir); mntErr != nil {
				klog.Errorf("Failed to unmount: %v", mntErr)
				return err
			}
			notMnt, mntErr := b.mounter.IsLikelyNotMountPoint(dir)
			if mntErr != nil {
				klog.Errorf("IsLikelyNotMountPoint check failed: %v", mntErr)
				return err
			}
			if !notMnt {
				// This is very odd, we don't expect it.  We'll try again next sync loop.
				klog.Errorf("%s is still mounted, despite call to unmount().  Will try again next sync loop.", dir)
				return err
			}
		}
		os.Remove(dir)
		return err
	}
	return nil
}

var _ volume.Unmounter = &azureFileUnmounter{}

type azureFileUnmounter struct {
	*azureFile
}

func (c *azureFileUnmounter) TearDown() error {
	return c.TearDownAt(c.GetPath())
}

func (c *azureFileUnmounter) TearDownAt(dir string) error {
	return mount.CleanupMountPoint(dir, c.mounter, false)
}

func getVolumeSource(spec *volume.Spec) (string, bool, error) {
	if spec.Volume != nil && spec.Volume.AzureFile != nil {
		share := spec.Volume.AzureFile.ShareName
		readOnly := spec.Volume.AzureFile.ReadOnly
		return share, readOnly, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.AzureFile != nil {
		share := spec.PersistentVolume.Spec.AzureFile.ShareName
		readOnly := spec.ReadOnly
		return share, readOnly, nil
	}
	return "", false, fmt.Errorf("Spec does not reference an AzureFile volume type")
}

func getSecretNameAndNamespace(spec *volume.Spec, defaultNamespace string) (string, string, error) {
	secretName := ""
	secretNamespace := ""
	if spec.Volume != nil && spec.Volume.AzureFile != nil {
		secretName = spec.Volume.AzureFile.SecretName
		secretNamespace = defaultNamespace

	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.AzureFile != nil {
		secretNamespace = defaultNamespace
		if spec.PersistentVolume.Spec.AzureFile.SecretNamespace != nil {
			secretNamespace = *spec.PersistentVolume.Spec.AzureFile.SecretNamespace
		}
		secretName = spec.PersistentVolume.Spec.AzureFile.SecretName
	} else {
		return "", "", fmt.Errorf("Spec does not reference an AzureFile volume type")
	}

	if len(secretNamespace) == 0 {
		return "", "", fmt.Errorf("invalid Azure volume: nil namespace")
	}
	return secretName, secretNamespace, nil

}

func getAzureCloud(cloudProvider cloudprovider.Interface) (*azure.Cloud, error) {
	azure, ok := cloudProvider.(*azure.Cloud)
	if !ok || azure == nil {
		return nil, fmt.Errorf("Failed to get Azure Cloud Provider. GetCloudProvider returned %v instead", cloudProvider)
	}

	return azure, nil
}

func getStorageEndpointSuffix(cloudprovider cloudprovider.Interface) string {
	const publicCloudStorageEndpointSuffix = "core.windows.net"
	azure, err := getAzureCloud(cloudprovider)
	if err != nil {
		klog.Warningf("No Azure cloud provider found. Using the Azure public cloud endpoint: %s", publicCloudStorageEndpointSuffix)
		return publicCloudStorageEndpointSuffix
	}
	return azure.Environment.StorageEndpointSuffix
}
