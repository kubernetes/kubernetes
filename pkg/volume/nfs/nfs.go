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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"

	"github.com/golang/glog"
)

// This is the primary entrypoint for volume plugins.
// The volumeConfig arg provides the ability to configure recycler behavior.  It is implemented as a pointer to allow nils.
// The nfsPlugin is used to store the volumeConfig and give it, when needed, to the func that creates NFS Recyclers.
// Tests that exercise recycling should not use this func but instead use ProbeRecyclablePlugins() to override default behavior.
func ProbeVolumePlugins(volumeConfig volume.VolumeConfig) []volume.VolumePlugin {
	return []volume.VolumePlugin{
		&nfsPlugin{
			host:            nil,
			newRecyclerFunc: newRecycler,
			config:          volumeConfig,
		},
	}
}

type nfsPlugin struct {
	host volume.VolumeHost
	// decouple creating recyclers by deferring to a function.  Allows for easier testing.
	newRecyclerFunc func(spec *volume.Spec, host volume.VolumeHost, volumeConfig volume.VolumeConfig) (volume.Recycler, error)
	config          volume.VolumeConfig
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
	return (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.NFS != nil) ||
		(spec.Volume != nil && spec.Volume.NFS != nil)
}

func (plugin *nfsPlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
		api.ReadOnlyMany,
		api.ReadWriteMany,
	}
}

func (plugin *nfsPlugin) NewBuilder(spec *volume.Spec, pod *api.Pod, _ volume.VolumeOptions) (volume.Builder, error) {
	return plugin.newBuilderInternal(spec, pod, plugin.host.GetMounter())
}

func (plugin *nfsPlugin) newBuilderInternal(spec *volume.Spec, pod *api.Pod, mounter mount.Interface) (volume.Builder, error) {
	var source *api.NFSVolumeSource
	var readOnly bool
	if spec.Volume != nil && spec.Volume.NFS != nil {
		source = spec.Volume.NFS
		readOnly = spec.Volume.NFS.ReadOnly
	} else {
		source = spec.PersistentVolume.Spec.NFS
		readOnly = spec.ReadOnly
	}
	return &nfsBuilder{
		nfs: &nfs{
			volName: spec.Name(),
			mounter: mounter,
			pod:     pod,
			plugin:  plugin,
		},
		server:     source.Server,
		exportPath: source.Path,
		readOnly:   readOnly,
	}, nil
}

func (plugin *nfsPlugin) NewCleaner(volName string, podUID types.UID) (volume.Cleaner, error) {
	return plugin.newCleanerInternal(volName, podUID, plugin.host.GetMounter())
}

func (plugin *nfsPlugin) newCleanerInternal(volName string, podUID types.UID, mounter mount.Interface) (volume.Cleaner, error) {
	return &nfsCleaner{&nfs{
		volName: volName,
		mounter: mounter,
		pod:     &api.Pod{ObjectMeta: api.ObjectMeta{UID: podUID}},
		plugin:  plugin,
	}}, nil
}

func (plugin *nfsPlugin) NewRecycler(spec *volume.Spec) (volume.Recycler, error) {
	return plugin.newRecyclerFunc(spec, plugin.host, plugin.config)
}

// NFS volumes represent a bare host file or directory mount of an NFS export.
type nfs struct {
	volName string
	pod     *api.Pod
	mounter mount.Interface
	plugin  *nfsPlugin
	// decouple creating recyclers by deferring to a function.  Allows for easier testing.
	newRecyclerFunc func(spec *volume.Spec, host volume.VolumeHost, volumeConfig volume.VolumeConfig) (volume.Recycler, error)
}

func (nfsVolume *nfs) GetPath() string {
	name := nfsPluginName
	return nfsVolume.plugin.host.GetPodVolumeDir(nfsVolume.pod.UID, util.EscapeQualifiedNameForDisk(name), nfsVolume.volName)
}

type nfsBuilder struct {
	*nfs
	server     string
	exportPath string
	readOnly   bool
}

var _ volume.Builder = &nfsBuilder{}

func (_ *nfsBuilder) SupportsOwnershipManagement() bool {
	return false
}

// SetUp attaches the disk and bind mounts to the volume path.
func (b *nfsBuilder) SetUp() error {
	return b.SetUpAt(b.GetPath())
}

func (b *nfsBuilder) SetUpAt(dir string) error {
	notMnt, err := b.mounter.IsLikelyNotMountPoint(dir)
	glog.V(4).Infof("NFS mount set up: %s %v %v", dir, !notMnt, err)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	if !notMnt {
		return nil
	}
	os.MkdirAll(dir, 0750)
	source := fmt.Sprintf("%s:%s", b.server, b.exportPath)
	options := []string{}
	if b.readOnly {
		options = append(options, "ro")
	}
	err = b.mounter.Mount(source, dir, "nfs", options)
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

func (b *nfsBuilder) IsReadOnly() bool {
	return b.readOnly
}

func (b *nfsBuilder) SupportsSELinux() bool {
	return false
}

//
//func (c *nfsCleaner) GetPath() string {
//	name := nfsPluginName
//	return c.plugin.host.GetPodVolumeDir(c.pod.UID, util.EscapeQualifiedNameForDisk(name), c.volName)
//}

var _ volume.Cleaner = &nfsCleaner{}

type nfsCleaner struct {
	*nfs
}

func (c *nfsCleaner) TearDown() error {
	return c.TearDownAt(c.GetPath())
}

func (c *nfsCleaner) TearDownAt(dir string) error {
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

func newRecycler(spec *volume.Spec, host volume.VolumeHost, volumeConfig volume.VolumeConfig) (volume.Recycler, error) {
	if spec.PersistentVolume == nil || spec.PersistentVolume.Spec.NFS == nil {
		return nil, fmt.Errorf("spec.PersistentVolumeSource.NFS is nil")
	}
	return &nfsRecycler{
		name:    spec.Name(),
		server:  spec.PersistentVolume.Spec.NFS.Server,
		path:    spec.PersistentVolume.Spec.NFS.Path,
		host:    host,
		config:  volumeConfig,
		timeout: volume.CalculateTimeoutForVolume(volumeConfig.RecyclerMinimumTimeout, volumeConfig.RecyclerTimeoutIncrement, spec.PersistentVolume),
	}, nil
}

// nfsRecycler scrubs an NFS volume by running "rm -rf" on the volume in a pod.
type nfsRecycler struct {
	name    string
	server  string
	path    string
	host    volume.VolumeHost
	config  volume.VolumeConfig
	timeout int64
}

func (r *nfsRecycler) GetPath() string {
	return r.path
}

// Recycle recycles/scrubs clean an NFS volume.
// Recycle blocks until the pod has completed or any error occurs.
func (r *nfsRecycler) Recycle() error {
	pod := r.config.RecyclerPodTemplate
	// overrides
	pod.Spec.ActiveDeadlineSeconds = &r.timeout
	pod.GenerateName = "pv-recycler-nfs-"
	pod.Spec.Volumes[0].VolumeSource = api.VolumeSource{
		NFS: &api.NFSVolumeSource{
			Server: r.server,
			Path:   r.path,
		},
	}
	return volume.RecycleVolumeByWatchingPodUntilCompletion(pod, r.host.GetKubeClient())
}
