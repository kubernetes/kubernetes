/*
Copyright 2014 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/strings"
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
	newRecyclerFunc func(pvName string, spec *volume.Spec, host volume.VolumeHost, volumeConfig volume.VolumeConfig) (volume.Recycler, error)
	config          volume.VolumeConfig
}

var _ volume.VolumePlugin = &nfsPlugin{}
var _ volume.PersistentVolumePlugin = &nfsPlugin{}
var _ volume.RecyclableVolumePlugin = &nfsPlugin{}

const (
	nfsPluginName = "kubernetes.io/nfs"
)

func (plugin *nfsPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

func (plugin *nfsPlugin) GetPluginName() string {
	return nfsPluginName
}

func (plugin *nfsPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	return fmt.Sprintf(
		"%v/%v",
		volumeSource.Server,
		volumeSource.Path), nil
}

func (plugin *nfsPlugin) CanSupport(spec *volume.Spec) bool {
	return (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.NFS != nil) ||
		(spec.Volume != nil && spec.Volume.NFS != nil)
}

func (plugin *nfsPlugin) RequiresRemount() bool {
	return false
}

func (plugin *nfsPlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
		api.ReadOnlyMany,
		api.ReadWriteMany,
	}
}

func (plugin *nfsPlugin) NewMounter(spec *volume.Spec, pod *api.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
	return plugin.newMounterInternal(spec, pod, plugin.host.GetMounter())
}

func (plugin *nfsPlugin) newMounterInternal(spec *volume.Spec, pod *api.Pod, mounter mount.Interface) (volume.Mounter, error) {
	source, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}

	return &nfsMounter{
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

func (plugin *nfsPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return plugin.newUnmounterInternal(volName, podUID, plugin.host.GetMounter())
}

func (plugin *nfsPlugin) newUnmounterInternal(volName string, podUID types.UID, mounter mount.Interface) (volume.Unmounter, error) {
	return &nfsUnmounter{&nfs{
		volName: volName,
		mounter: mounter,
		pod:     &api.Pod{ObjectMeta: api.ObjectMeta{UID: podUID}},
		plugin:  plugin,
	}}, nil
}

func (plugin *nfsPlugin) NewRecycler(pvName string, spec *volume.Spec) (volume.Recycler, error) {
	return plugin.newRecyclerFunc(pvName, spec, plugin.host, plugin.config)
}

func (plugin *nfsPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	nfsVolume := &api.Volume{
		Name: volumeName,
		VolumeSource: api.VolumeSource{
			NFS: &api.NFSVolumeSource{
				Path: volumeName,
			},
		},
	}
	return volume.NewSpecFromVolume(nfsVolume), nil
}

// NFS volumes represent a bare host file or directory mount of an NFS export.
type nfs struct {
	volName string
	pod     *api.Pod
	mounter mount.Interface
	plugin  *nfsPlugin
	// decouple creating recyclers by deferring to a function.  Allows for easier testing.
	newRecyclerFunc func(spec *volume.Spec, host volume.VolumeHost, volumeConfig volume.VolumeConfig) (volume.Recycler, error)
	volume.MetricsNil
}

func (nfsVolume *nfs) GetPath() string {
	name := nfsPluginName
	return nfsVolume.plugin.host.GetPodVolumeDir(nfsVolume.pod.UID, strings.EscapeQualifiedNameForDisk(name), nfsVolume.volName)
}

type nfsMounter struct {
	*nfs
	server     string
	exportPath string
	readOnly   bool
}

var _ volume.Mounter = &nfsMounter{}

func (b *nfsMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        b.readOnly,
		Managed:         false,
		SupportsSELinux: false,
	}
}

// SetUp attaches the disk and bind mounts to the volume path.
func (b *nfsMounter) SetUp(fsGroup *int64) error {
	return b.SetUpAt(b.GetPath(), fsGroup)
}

func (b *nfsMounter) SetUpAt(dir string, fsGroup *int64) error {
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

//
//func (c *nfsUnmounter) GetPath() string {
//	name := nfsPluginName
//	return c.plugin.host.GetPodVolumeDir(c.pod.UID, strings.EscapeQualifiedNameForDisk(name), c.volName)
//}

var _ volume.Unmounter = &nfsUnmounter{}

type nfsUnmounter struct {
	*nfs
}

func (c *nfsUnmounter) TearDown() error {
	return c.TearDownAt(c.GetPath())
}

func (c *nfsUnmounter) TearDownAt(dir string) error {
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

func newRecycler(pvName string, spec *volume.Spec, host volume.VolumeHost, volumeConfig volume.VolumeConfig) (volume.Recycler, error) {
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
		pvName:  pvName,
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
	volume.MetricsNil
	pvName string
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
	return volume.RecycleVolumeByWatchingPodUntilCompletion(r.pvName, pod, r.host.GetKubeClient())
}

func getVolumeSource(spec *volume.Spec) (*api.NFSVolumeSource, bool, error) {
	if spec.Volume != nil && spec.Volume.NFS != nil {
		return spec.Volume.NFS, spec.Volume.NFS.ReadOnly, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.NFS != nil {
		return spec.PersistentVolume.Spec.NFS, spec.ReadOnly, nil
	}

	return nil, false, fmt.Errorf("Spec does not reference a NFS volume type")
}
