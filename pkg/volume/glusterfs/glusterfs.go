/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package glusterfs

import (
	"math/rand"
	"os"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/exec"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
	"github.com/golang/glog"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&glusterfsPlugin{nil}}
}

type glusterfsPlugin struct {
	host volume.VolumeHost
}

var _ volume.VolumePlugin = &glusterfsPlugin{}

const (
	glusterfsPluginName = "kubernetes.io/glusterfs"
)

func (plugin *glusterfsPlugin) Init(host volume.VolumeHost) {
	plugin.host = host
}

func (plugin *glusterfsPlugin) Name() string {
	return glusterfsPluginName
}

func (plugin *glusterfsPlugin) CanSupport(spec *volume.Spec) bool {
	return spec.VolumeSource.Glusterfs != nil || spec.PersistentVolumeSource.Glusterfs != nil
}

func (plugin *glusterfsPlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
		api.ReadOnlyMany,
		api.ReadWriteMany,
	}
}

func (plugin *glusterfsPlugin) NewBuilder(spec *volume.Spec, pod *api.Pod, _ volume.VolumeOptions, mounter mount.Interface) (volume.Builder, error) {
	source := plugin.getGlusterVolumeSource(spec)
	ep_name := source.EndpointsName
	ns := pod.Namespace
	ep, err := plugin.host.GetKubeClient().Endpoints(ns).Get(ep_name)
	if err != nil {
		glog.Errorf("Glusterfs: failed to get endpoints %s[%v]", ep_name, err)
		return nil, err
	}
	glog.V(1).Infof("Glusterfs: endpoints %v", ep)
	return plugin.newBuilderInternal(spec, ep, pod, mounter, exec.New())
}

func (plugin *glusterfsPlugin) getGlusterVolumeSource(spec *volume.Spec) *api.GlusterfsVolumeSource {
	if spec.VolumeSource.Glusterfs != nil {
		return spec.VolumeSource.Glusterfs
	} else {
		return spec.PersistentVolumeSource.Glusterfs
	}
}

func (plugin *glusterfsPlugin) newBuilderInternal(spec *volume.Spec, ep *api.Endpoints, pod *api.Pod, mounter mount.Interface, exe exec.Interface) (volume.Builder, error) {
	source := plugin.getGlusterVolumeSource(spec)
	return &glusterfs{
		volName:  spec.Name,
		hosts:    ep,
		path:     source.Path,
		readonly: source.ReadOnly,
		mounter:  mounter,
		exe:      exe,
		pod:      pod,
		plugin:   plugin,
	}, nil
}

func (plugin *glusterfsPlugin) NewCleaner(volName string, podUID types.UID, mounter mount.Interface) (volume.Cleaner, error) {
	return plugin.newCleanerInternal(volName, podUID, mounter)
}

func (plugin *glusterfsPlugin) newCleanerInternal(volName string, podUID types.UID, mounter mount.Interface) (volume.Cleaner, error) {
	return &glusterfs{
		volName: volName,
		mounter: mounter,
		pod:     &api.Pod{ObjectMeta: api.ObjectMeta{UID: podUID}},
		plugin:  plugin,
	}, nil
}

// Glusterfs volumes represent a bare host file or directory mount of an Glusterfs export.
type glusterfs struct {
	volName  string
	pod      *api.Pod
	hosts    *api.Endpoints
	path     string
	readonly bool
	mounter  mount.Interface
	exe      exec.Interface
	plugin   *glusterfsPlugin
}

// SetUp attaches the disk and bind mounts to the volume path.
func (glusterfsVolume *glusterfs) SetUp() error {
	return glusterfsVolume.SetUpAt(glusterfsVolume.GetPath())
}

func (glusterfsVolume *glusterfs) SetUpAt(dir string) error {
	mountpoint, err := glusterfsVolume.mounter.IsMountPoint(dir)
	glog.V(4).Infof("Glusterfs: mount set up: %s %v %v", dir, mountpoint, err)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	if mountpoint {
		return nil
	}

	os.MkdirAll(dir, 0750)
	err = glusterfsVolume.setUpAtInternal(dir)
	if err == nil {
		return nil
	}

	// Cleanup upon failure.
	glusterfsVolume.cleanup(dir)
	return err
}

func (glusterfsVolume *glusterfs) GetPath() string {
	name := glusterfsPluginName
	return glusterfsVolume.plugin.host.GetPodVolumeDir(glusterfsVolume.pod.UID, util.EscapeQualifiedNameForDisk(name), glusterfsVolume.volName)
}

func (glusterfsVolume *glusterfs) TearDown() error {
	return glusterfsVolume.TearDownAt(glusterfsVolume.GetPath())
}

func (glusterfsVolume *glusterfs) TearDownAt(dir string) error {
	return glusterfsVolume.cleanup(dir)
}

func (glusterfsVolume *glusterfs) cleanup(dir string) error {
	mountpoint, err := glusterfsVolume.mounter.IsMountPoint(dir)
	if err != nil {
		glog.Errorf("Glusterfs: Error checking IsMountPoint: %v", err)
		return err
	}
	if !mountpoint {
		return os.RemoveAll(dir)
	}

	if err := glusterfsVolume.mounter.Unmount(dir); err != nil {
		glog.Errorf("Glusterfs: Unmounting failed: %v", err)
		return err
	}
	mountpoint, mntErr := glusterfsVolume.mounter.IsMountPoint(dir)
	if mntErr != nil {
		glog.Errorf("Glusterfs: IsMountpoint check failed: %v", mntErr)
		return mntErr
	}
	if !mountpoint {
		if err := os.RemoveAll(dir); err != nil {
			return err
		}
	}

	return nil
}

func (glusterfsVolume *glusterfs) setUpAtInternal(dir string) error {
	var errs error

	options := []string{}
	if glusterfsVolume.readonly {
		options = append(options, "ro")
	}

	l := len(glusterfsVolume.hosts.Subsets)
	// Avoid mount storm, pick a host randomly.
	start := rand.Int() % l
	// Iterate all hosts until mount succeeds.
	for i := start; i < start+l; i++ {
		hostIP := glusterfsVolume.hosts.Subsets[i%l].Addresses[0].IP
		errs = glusterfsVolume.mounter.Mount(hostIP+":"+glusterfsVolume.path, dir, "glusterfs", options)
		if errs == nil {
			return nil
		}
	}
	glog.Errorf("Glusterfs: mount failed: %v", errs)
	return errs
}
