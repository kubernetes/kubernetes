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

package azure_file

import (
	"fmt"
	"os"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/strings"
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

func (plugin *azureFilePlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

func (plugin *azureFilePlugin) Name() string {
	return azureFilePluginName
}

func (plugin *azureFilePlugin) CanSupport(spec *volume.Spec) bool {
	//TODO: check if mount.cifs is there
	return (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.AzureFile != nil) ||
		(spec.Volume != nil && spec.Volume.AzureFile != nil)
}

func (plugin *azureFilePlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
		api.ReadOnlyMany,
		api.ReadWriteMany,
	}
}

func (plugin *azureFilePlugin) NewBuilder(spec *volume.Spec, pod *api.Pod, _ volume.VolumeOptions) (volume.Builder, error) {
	return plugin.newBuilderInternal(spec, pod, &azureSvc{}, plugin.host.GetMounter())
}

func (plugin *azureFilePlugin) newBuilderInternal(spec *volume.Spec, pod *api.Pod, util azureUtil, mounter mount.Interface) (volume.Builder, error) {
	var source *api.AzureFileVolumeSource
	var readOnly bool
	if spec.Volume != nil && spec.Volume.AzureFile != nil {
		source = spec.Volume.AzureFile
		readOnly = spec.Volume.AzureFile.ReadOnly
	} else {
		source = spec.PersistentVolume.Spec.AzureFile
		readOnly = spec.ReadOnly
	}
	return &azureFileBuilder{
		azureFile: &azureFile{
			volName: spec.Name(),
			mounter: mounter,
			pod:     pod,
			plugin:  plugin,
		},
		util:       util,
		secretName: source.SecretName,
		shareName:  source.ShareName,
		readOnly:   readOnly,
	}, nil
}

func (plugin *azureFilePlugin) NewCleaner(volName string, podUID types.UID) (volume.Cleaner, error) {
	return plugin.newCleanerInternal(volName, podUID, plugin.host.GetMounter())
}

func (plugin *azureFilePlugin) newCleanerInternal(volName string, podUID types.UID, mounter mount.Interface) (volume.Cleaner, error) {
	return &azureFileCleaner{&azureFile{
		volName: volName,
		mounter: mounter,
		pod:     &api.Pod{ObjectMeta: api.ObjectMeta{UID: podUID}},
		plugin:  plugin,
	}}, nil
}

// azureFile volumes represent mount of an AzureFile share.
type azureFile struct {
	volName string
	pod     *api.Pod
	mounter mount.Interface
	plugin  *azureFilePlugin
	volume.MetricsNil
}

func (azureFileVolume *azureFile) GetPath() string {
	name := azureFilePluginName
	return azureFileVolume.plugin.host.GetPodVolumeDir(azureFileVolume.pod.UID, strings.EscapeQualifiedNameForDisk(name), azureFileVolume.volName)
}

type azureFileBuilder struct {
	*azureFile
	util       azureUtil
	secretName string
	shareName  string
	readOnly   bool
}

var _ volume.Builder = &azureFileBuilder{}

func (b *azureFileBuilder) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        b.readOnly,
		Managed:         !b.readOnly,
		SupportsSELinux: false,
	}
}

// SetUp attaches the disk and bind mounts to the volume path.
func (b *azureFileBuilder) SetUp(fsGroup *int64) error {
	return b.SetUpAt(b.GetPath(), fsGroup)
}

func (b *azureFileBuilder) SetUpAt(dir string, fsGroup *int64) error {
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

var _ volume.Cleaner = &azureFileCleaner{}

type azureFileCleaner struct {
	*azureFile
}

func (c *azureFileCleaner) TearDown() error {
	return c.TearDownAt(c.GetPath())
}

func (c *azureFileCleaner) TearDownAt(dir string) error {
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
