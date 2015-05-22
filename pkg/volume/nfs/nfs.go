/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package nfs

import (
	"fmt"
	"os"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
	"github.com/golang/glog"
)

// This is the primary entrypoint for volume plugins.
// Tests covering recycling should not use this func but instead
// use their own array of plugins w/ a custom recyclerFunc as appropriate
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&nfsPlugin{nil, newRecycler}}
}

type nfsPlugin struct {
	host volume.VolumeHost
	// decouple creating recyclers by deferring to a function.  Allows for easier testing.
	newRecyclerFunc func(spec *volume.Spec, host volume.VolumeHost) (volume.Recycler, error)
}

var _ volume.VolumePlugin = &nfsPlugin{}
var _ volume.PersistentVolumePlugin = &nfsPlugin{}
var _ volume.RecyclableVolumePlugin = &nfsPlugin{}

const (
	nfsPluginName = "kubernetes.io/nfs"
)

func (plugin *nfsPlugin) Init(host volume.VolumeHost) {
	plugin.host = host
}

func (plugin *nfsPlugin) Name() string {
	return nfsPluginName
}

func (plugin *nfsPlugin) CanSupport(spec *volume.Spec) bool {
	return spec.VolumeSource.NFS != nil || spec.PersistentVolumeSource.NFS != nil
}

func (plugin *nfsPlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
		api.ReadOnlyMany,
		api.ReadWriteMany,
	}
}

func (plugin *nfsPlugin) NewBuilder(spec *volume.Spec, pod *api.Pod, _ volume.VolumeOptions, mounter mount.Interface) (volume.Builder, error) {
	return plugin.newBuilderInternal(spec, pod, mounter)
}

func (plugin *nfsPlugin) newBuilderInternal(spec *volume.Spec, pod *api.Pod, mounter mount.Interface) (volume.Builder, error) {
	return &nfs{
		volName:    spec.Name,
		server:     spec.VolumeSource.NFS.Server,
		exportPath: spec.VolumeSource.NFS.Path,
		readOnly:   spec.VolumeSource.NFS.ReadOnly,
		mounter:    mounter,
		pod:        pod,
		plugin:     plugin,
	}, nil
}

func (plugin *nfsPlugin) NewCleaner(volName string, podUID types.UID, mounter mount.Interface) (volume.Cleaner, error) {
	return plugin.newCleanerInternal(volName, podUID, mounter)
}

func (plugin *nfsPlugin) newCleanerInternal(volName string, podUID types.UID, mounter mount.Interface) (volume.Cleaner, error) {
	return &nfs{
		volName:    volName,
		server:     "",
		exportPath: "",
		readOnly:   false,
		mounter:    mounter,
		pod:        &api.Pod{ObjectMeta: api.ObjectMeta{UID: podUID}},
		plugin:     plugin,
	}, nil
}

func (plugin *nfsPlugin) NewRecycler(spec *volume.Spec) (volume.Recycler, error) {
	return plugin.newRecyclerFunc(spec, plugin.host)
}

func newRecycler(spec *volume.Spec, host volume.VolumeHost) (volume.Recycler, error) {
	if spec.VolumeSource.HostPath != nil {
		return &nfsRecycler{
			volName:    spec.Name,
			server:     spec.VolumeSource.NFS.Server,
			exportPath: spec.VolumeSource.NFS.Path,
		}, nil
	} else {
		return &nfsRecycler{
			volName:    spec.Name,
			server:     spec.PersistentVolumeSource.NFS.Server,
			exportPath: spec.PersistentVolumeSource.NFS.Path,
		}, nil
	}
}

// NFS volumes represent a bare host file or directory mount of an NFS export.
type nfs struct {
	volName    string
	pod        *api.Pod
	server     string
	exportPath string
	readOnly   bool
	mounter    mount.Interface
	plugin     *nfsPlugin
	// decouple creating recyclers by deferring to a function.  Allows for easier testing.
	newRecyclerFunc func(spec *volume.Spec, host volume.VolumeHost) (volume.Recycler, error)
}

// SetUp attaches the disk and bind mounts to the volume path.
func (nfsVolume *nfs) SetUp() error {
	return nfsVolume.SetUpAt(nfsVolume.GetPath())
}

func (nfsVolume *nfs) SetUpAt(dir string) error {
	mountpoint, err := nfsVolume.mounter.IsMountPoint(dir)
	glog.V(4).Infof("NFS mount set up: %s %v %v", dir, mountpoint, err)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	if mountpoint {
		return nil
	}
	os.MkdirAll(dir, 0750)
	source := fmt.Sprintf("%s:%s", nfsVolume.server, nfsVolume.exportPath)
	options := []string{}
	if nfsVolume.readOnly {
		options = append(options, "ro")
	}
	err = nfsVolume.mounter.Mount(source, dir, "nfs", options)
	if err != nil {
		mountpoint, mntErr := nfsVolume.mounter.IsMountPoint(dir)
		if mntErr != nil {
			glog.Errorf("IsMountpoint check failed: %v", mntErr)
			return err
		}
		if mountpoint {
			if mntErr = nfsVolume.mounter.Unmount(dir); mntErr != nil {
				glog.Errorf("Failed to unmount: %v", mntErr)
				return err
			}
			mountpoint, mntErr := nfsVolume.mounter.IsMountPoint(dir)
			if mntErr != nil {
				glog.Errorf("IsMountpoint check failed: %v", mntErr)
				return err
			}
			if mountpoint {
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

func (nfsVolume *nfs) GetPath() string {
	name := nfsPluginName
	return nfsVolume.plugin.host.GetPodVolumeDir(nfsVolume.pod.UID, util.EscapeQualifiedNameForDisk(name), nfsVolume.volName)
}

func (nfsVolume *nfs) TearDown() error {
	return nfsVolume.TearDownAt(nfsVolume.GetPath())
}

func (nfsVolume *nfs) TearDownAt(dir string) error {
	mountpoint, err := nfsVolume.mounter.IsMountPoint(dir)
	if err != nil {
		glog.Errorf("Error checking IsMountPoint: %v", err)
		return err
	}
	if !mountpoint {
		return os.Remove(dir)
	}

	if err := nfsVolume.mounter.Unmount(dir); err != nil {
		glog.Errorf("Unmounting failed: %v", err)
		return err
	}
	mountpoint, mntErr := nfsVolume.mounter.IsMountPoint(dir)
	if mntErr != nil {
		glog.Errorf("IsMountpoint check failed: %v", mntErr)
		return mntErr
	}
	if !mountpoint {
		if err := os.Remove(dir); err != nil {
			return err
		}
	}

	return nil
}

// nfsRecycler scrubs an NFS volume by running "rm -rf" on the volume in a pod.
type nfsRecycler struct {
	volName    string
	server     string
	exportPath string
	host       volume.VolumeHost
}

func (r *nfsRecycler) GetPath() string {
	return r.exportPath
}

// Recycler provides methods to reclaim the volume resource.
func (r *nfsRecycler) Recycle() error {

	// TODO implement "basic scrub" recycler -- busybox w/ "rm -rf" for volume

	/*
		1. make a pod that uses a volume like me
		2. use host.client to save to API and watch it
		3. if pod errors or times out, return error
		   if pod exits, return nil for success
	*/

	return nil
}
