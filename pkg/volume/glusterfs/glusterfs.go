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

package glusterfs

import (
	"fmt"
	"os"
	"path"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&glusterfsPlugin{nil, exec.New()}}
}

type glusterfsPlugin struct {
	host volume.VolumeHost
	exe  exec.Interface
}

var _ volume.VolumePlugin = &glusterfsPlugin{}
var _ volume.PersistentVolumePlugin = &glusterfsPlugin{}

const (
	glusterfsPluginName = "kubernetes.io/glusterfs"
)

func (plugin *glusterfsPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

func (plugin *glusterfsPlugin) GetPluginName() string {
	return glusterfsPluginName
}

func (plugin *glusterfsPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	return fmt.Sprintf(
		"%v:%v",
		volumeSource.EndpointsName,
		volumeSource.Path), nil
}

func (plugin *glusterfsPlugin) CanSupport(spec *volume.Spec) bool {
	if (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.Glusterfs == nil) ||
		(spec.Volume != nil && spec.Volume.Glusterfs == nil) {
		return false
	}

	return true
}

func (plugin *glusterfsPlugin) RequiresRemount() bool {
	return false
}

func (plugin *glusterfsPlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
		api.ReadOnlyMany,
		api.ReadWriteMany,
	}
}

func (plugin *glusterfsPlugin) NewMounter(spec *volume.Spec, pod *api.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
	source, _ := plugin.getGlusterVolumeSource(spec)
	ep_name := source.EndpointsName
	ns := pod.Namespace
	ep, err := plugin.host.GetKubeClient().Core().Endpoints(ns).Get(ep_name)
	if err != nil {
		glog.Errorf("glusterfs: failed to get endpoints %s[%v]", ep_name, err)
		return nil, err
	}
	glog.V(1).Infof("glusterfs: endpoints %v", ep)
	return plugin.newMounterInternal(spec, ep, pod, plugin.host.GetMounter(), exec.New())
}

func (plugin *glusterfsPlugin) getGlusterVolumeSource(spec *volume.Spec) (*api.GlusterfsVolumeSource, bool) {
	// Glusterfs volumes used directly in a pod have a ReadOnly flag set by the pod author.
	// Glusterfs volumes used as a PersistentVolume gets the ReadOnly flag indirectly through the persistent-claim volume used to mount the PV
	if spec.Volume != nil && spec.Volume.Glusterfs != nil {
		return spec.Volume.Glusterfs, spec.Volume.Glusterfs.ReadOnly
	} else {
		return spec.PersistentVolume.Spec.Glusterfs, spec.ReadOnly
	}
}

func (plugin *glusterfsPlugin) newMounterInternal(spec *volume.Spec, ep *api.Endpoints, pod *api.Pod, mounter mount.Interface, exe exec.Interface) (volume.Mounter, error) {
	source, readOnly := plugin.getGlusterVolumeSource(spec)
	return &glusterfsMounter{
		glusterfs: &glusterfs{
			volName: spec.Name(),
			mounter: mounter,
			pod:     pod,
			plugin:  plugin,
		},
		hosts:    ep,
		path:     source.Path,
		readOnly: readOnly,
		exe:      exe}, nil
}

func (plugin *glusterfsPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return plugin.newUnmounterInternal(volName, podUID, plugin.host.GetMounter())
}

func (plugin *glusterfsPlugin) newUnmounterInternal(volName string, podUID types.UID, mounter mount.Interface) (volume.Unmounter, error) {
	return &glusterfsUnmounter{&glusterfs{
		volName: volName,
		mounter: mounter,
		pod:     &api.Pod{ObjectMeta: api.ObjectMeta{UID: podUID}},
		plugin:  plugin,
	}}, nil
}

func (plugin *glusterfsPlugin) execCommand(command string, args []string) ([]byte, error) {
	cmd := plugin.exe.Command(command, args...)
	return cmd.CombinedOutput()
}

func (plugin *glusterfsPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	glusterfsVolume := &api.Volume{
		Name: volumeName,
		VolumeSource: api.VolumeSource{
			Glusterfs: &api.GlusterfsVolumeSource{
				EndpointsName: volumeName,
				Path:          volumeName,
			},
		},
	}
	return volume.NewSpecFromVolume(glusterfsVolume), nil
}

// Glusterfs volumes represent a bare host file or directory mount of an Glusterfs export.
type glusterfs struct {
	volName string
	pod     *api.Pod
	mounter mount.Interface
	plugin  *glusterfsPlugin
	volume.MetricsNil
}

type glusterfsMounter struct {
	*glusterfs
	hosts    *api.Endpoints
	path     string
	readOnly bool
	exe      exec.Interface
}

var _ volume.Mounter = &glusterfsMounter{}

func (b *glusterfsMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        b.readOnly,
		Managed:         false,
		SupportsSELinux: false,
	}
}

// SetUp attaches the disk and bind mounts to the volume path.
func (b *glusterfsMounter) SetUp(fsGroup *int64) error {
	return b.SetUpAt(b.GetPath(), fsGroup)
}

func (b *glusterfsMounter) SetUpAt(dir string, fsGroup *int64) error {
	notMnt, err := b.mounter.IsLikelyNotMountPoint(dir)
	glog.V(4).Infof("glusterfs: mount set up: %s %v %v", dir, !notMnt, err)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	if !notMnt {
		return nil
	}

	os.MkdirAll(dir, 0750)
	err = b.setUpAtInternal(dir)
	if err == nil {
		return nil
	}

	// Cleanup upon failure.
	c := &glusterfsUnmounter{b.glusterfs}
	c.cleanup(dir)
	return err
}

func (glusterfsVolume *glusterfs) GetPath() string {
	name := glusterfsPluginName
	return glusterfsVolume.plugin.host.GetPodVolumeDir(glusterfsVolume.pod.UID, strings.EscapeQualifiedNameForDisk(name), glusterfsVolume.volName)
}

type glusterfsUnmounter struct {
	*glusterfs
}

var _ volume.Unmounter = &glusterfsUnmounter{}

func (c *glusterfsUnmounter) TearDown() error {
	return c.TearDownAt(c.GetPath())
}

func (c *glusterfsUnmounter) TearDownAt(dir string) error {
	return c.cleanup(dir)
}

func (c *glusterfsUnmounter) cleanup(dir string) error {
	notMnt, err := c.mounter.IsLikelyNotMountPoint(dir)
	if err != nil {
		return fmt.Errorf("glusterfs: Error checking IsLikelyNotMountPoint: %v", err)
	}
	if notMnt {
		return os.RemoveAll(dir)
	}

	if err := c.mounter.Unmount(dir); err != nil {
		return fmt.Errorf("glusterfs: Unmounting failed: %v", err)
	}
	notMnt, mntErr := c.mounter.IsLikelyNotMountPoint(dir)
	if mntErr != nil {
		return fmt.Errorf("glusterfs: IsLikelyNotMountPoint check failed: %v", mntErr)
	}
	if notMnt {
		if err := os.RemoveAll(dir); err != nil {
			return fmt.Errorf("glusterfs: RemoveAll failed: %v", err)
		}
	}

	return nil
}

func (b *glusterfsMounter) setUpAtInternal(dir string) error {
	var errs error

	options := []string{}
	if b.readOnly {
		options = append(options, "ro")
	}

	p := path.Join(b.glusterfs.plugin.host.GetPluginDir(glusterfsPluginName), b.glusterfs.volName)
	if err := os.MkdirAll(p, 0750); err != nil {
		return fmt.Errorf("glusterfs: mkdir failed: %v", err)
	}

	// adding log-level ERROR to remove noise
	// and more specific log path so each pod has
	// it's own log based on PV + Pod
	log := path.Join(p, b.pod.Name+"-glusterfs.log")
	options = append(options, "log-level=ERROR")
	options = append(options, "log-file="+log)

	addr := make(map[string]struct{})
	for _, s := range b.hosts.Subsets {
		for _, a := range s.Addresses {
			addr[a.IP] = struct{}{}
		}
	}

	// Avoid mount storm, pick a host randomly.
	// Iterate all hosts until mount succeeds.
	for hostIP := range addr {
		errs = b.mounter.Mount(hostIP+":"+b.path, dir, "glusterfs", options)
		if errs == nil {
			glog.Infof("glusterfs: successfully mounted %s", dir)
			return nil
		}
	}

	// Failed mount scenario.
	// Since gluster does not return eror text
	// it all goes in a log file, we will read the log file
	logerror := readGlusterLog(log, b.pod.Name)
	if logerror != nil {
		// return fmt.Errorf("glusterfs: mount failed: %v", logerror)
		return fmt.Errorf("glusterfs: mount failed: %v the following error information was pulled from the glusterfs log to help diagnose this issue: %v", errs, logerror)
	}
	return fmt.Errorf("glusterfs: mount failed: %v", errs)
}

func getVolumeSource(
	spec *volume.Spec) (*api.GlusterfsVolumeSource, bool, error) {
	if spec.Volume != nil && spec.Volume.Glusterfs != nil {
		return spec.Volume.Glusterfs, spec.Volume.Glusterfs.ReadOnly, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.Glusterfs != nil {
		return spec.PersistentVolume.Spec.Glusterfs, spec.ReadOnly, nil
	}

	return nil, false, fmt.Errorf("Spec does not reference a Gluster volume type")
}
