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
	"os"

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	kstrings "k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"

	"github.com/golang/glog"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&azureFilePlugin{nil}}
}

type azureFilePlugin struct {
	host volume.VolumeHost
}

var _ volume.VolumePlugin = &azureFilePlugin{}
var _ volume.PersistentVolumePlugin = &azureFilePlugin{}

const (
	azureFilePluginName = "kubernetes.io/azure-file"
)

func getPath(uid types.UID, volName string, host volume.VolumeHost) string {
	return host.GetPodVolumeDir(uid, kstrings.EscapeQualifiedNameForDisk(azureFilePluginName), volName)
}

func (plugin *azureFilePlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

func (plugin *azureFilePlugin) GetPluginName() string {
	return azureFilePluginName
}

func (plugin *azureFilePlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	return volumeSource.ShareName, nil
}

func (plugin *azureFilePlugin) CanSupport(spec *volume.Spec) bool {
	//TODO: check if mount.cifs is there
	return (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.AzureFile != nil) ||
		(spec.Volume != nil && spec.Volume.AzureFile != nil)
}

func (plugin *azureFilePlugin) RequiresRemount() bool {
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
	return plugin.newMounterInternal(spec, pod, &azureSvc{}, plugin.host.GetMounter())
}

func (plugin *azureFilePlugin) newMounterInternal(spec *volume.Spec, pod *v1.Pod, util azureUtil, mounter mount.Interface) (volume.Mounter, error) {
	source, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}

	return &azureFileMounter{
		azureFile: &azureFile{
			volName:         spec.Name(),
			mounter:         mounter,
			pod:             pod,
			plugin:          plugin,
			MetricsProvider: volume.NewMetricsStatFS(getPath(pod.UID, spec.Name(), plugin.host)),
		},
		util:       util,
		secretName: source.SecretName,
		shareName:  source.ShareName,
		readOnly:   readOnly,
	}, nil
}

func (plugin *azureFilePlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return plugin.newUnmounterInternal(volName, podUID, plugin.host.GetMounter())
}

func (plugin *azureFilePlugin) newUnmounterInternal(volName string, podUID types.UID, mounter mount.Interface) (volume.Unmounter, error) {
	return &azureFileUnmounter{&azureFile{
		volName:         volName,
		mounter:         mounter,
		pod:             &v1.Pod{ObjectMeta: v1.ObjectMeta{UID: podUID}},
		plugin:          plugin,
		MetricsProvider: volume.NewMetricsStatFS(getPath(podUID, volName, plugin.host)),
	}}, nil
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
	util       azureUtil
	secretName string
	shareName  string
	readOnly   bool
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
func (b *azureFileMounter) SetUp(fsGroup *int64) error {
	return b.SetUpAt(b.GetPath(), fsGroup)
}

func (b *azureFileMounter) SetUpAt(dir string, fsGroup *int64) error {
	notMnt, err := b.mounter.IsLikelyNotMountPoint(dir)
	glog.V(4).Infof("AzureFile mount set up: %s %v %v", dir, !notMnt, err)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	if !notMnt {
		return nil
	}
	var accountKey, accountName string
	if accountName, accountKey, err = b.util.GetAzureCredentials(b.plugin.host, b.pod.Namespace, b.secretName, b.shareName); err != nil {
		return err
	}
	os.MkdirAll(dir, 0750)
	source := fmt.Sprintf("//%s.file.core.windows.net/%s", accountName, b.shareName)
	// parameters suggested by https://azure.microsoft.com/en-us/documentation/articles/storage-how-to-use-files-linux/
	options := []string{fmt.Sprintf("vers=3.0,username=%s,password=%s,dir_mode=0777,file_mode=0777", accountName, accountKey)}
	if b.readOnly {
		options = append(options, "ro")
	}
	err = b.mounter.Mount(source, dir, "cifs", options)
	if err != nil {
		notMnt, mntErr := b.mounter.IsLikelyNotMountPoint(dir)
		if mntErr != nil {
			glog.Errorf("IsLikelyNotMountPoint check failed: %v", mntErr)
			return err
		}
		if !notMnt {
			if mntErr = b.mounter.Unmount(dir); mntErr != nil {
				glog.Errorf("Failed to unmount: %v", mntErr)
				return err
			}
			notMnt, mntErr := b.mounter.IsLikelyNotMountPoint(dir)
			if mntErr != nil {
				glog.Errorf("IsLikelyNotMountPoint check failed: %v", mntErr)
				return err
			}
			if !notMnt {
				// This is very odd, we don't expect it.  We'll try again next sync loop.
				glog.Errorf("%s is still mounted, despite call to unmount().  Will try again next sync loop.", dir)
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
	notMnt, err := c.mounter.IsLikelyNotMountPoint(dir)
	if err != nil {
		glog.Errorf("Error checking IsLikelyNotMountPoint: %v", err)
		return err
	}
	if notMnt {
		return os.Remove(dir)
	}

	if err := c.mounter.Unmount(dir); err != nil {
		glog.Errorf("Unmounting failed: %v", err)
		return err
	}
	notMnt, mntErr := c.mounter.IsLikelyNotMountPoint(dir)
	if mntErr != nil {
		glog.Errorf("IsLikelyNotMountPoint check failed: %v", mntErr)
		return mntErr
	}
	if notMnt {
		if err := os.Remove(dir); err != nil {
			return err
		}
	}

	return nil
}

func getVolumeSource(
	spec *volume.Spec) (*v1.AzureFileVolumeSource, bool, error) {
	if spec.Volume != nil && spec.Volume.AzureFile != nil {
		return spec.Volume.AzureFile, spec.Volume.AzureFile.ReadOnly, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.AzureFile != nil {
		return spec.PersistentVolume.Spec.AzureFile, spec.ReadOnly, nil
	}

	return nil, false, fmt.Errorf("Spec does not reference an AzureFile volume type")
}
