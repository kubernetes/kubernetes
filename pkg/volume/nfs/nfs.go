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
	"time"

	netutil "k8s.io/utils/net"

	"k8s.io/klog/v2"
	"k8s.io/mount-utils"
	utilstrings "k8s.io/utils/strings"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/pkg/volume/util/recyclerclient"
)

func getPath(uid types.UID, volName string, host volume.VolumeHost) string {
	return host.GetPodVolumeDir(uid, utilstrings.EscapeQualifiedName(nfsPluginName), volName)
}

// ProbeVolumePlugins is the primary entrypoint for volume plugins.
// This is the primary entrypoint for volume plugins.
// The volumeConfig arg provides the ability to configure recycler behavior.  It is implemented as a pointer to allow nils.
// The nfsPlugin is used to store the volumeConfig and give it, when needed, to the func that creates NFS Recyclers.
// Tests that exercise recycling should not use this func but instead use ProbeRecyclablePlugins() to override default behavior.
func ProbeVolumePlugins(volumeConfig volume.VolumeConfig) []volume.VolumePlugin {
	return []volume.VolumePlugin{
		&nfsPlugin{
			host:   nil,
			config: volumeConfig,
		},
	}
}

type nfsPlugin struct {
	host   volume.VolumeHost
	config volume.VolumeConfig
}

var _ volume.VolumePlugin = &nfsPlugin{}
var _ volume.PersistentVolumePlugin = &nfsPlugin{}
var _ volume.RecyclableVolumePlugin = &nfsPlugin{}

const (
	nfsPluginName  = "kubernetes.io/nfs"
	unMountTimeout = time.Minute
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

func (plugin *nfsPlugin) RequiresRemount(spec *volume.Spec) bool {
	return false
}

func (plugin *nfsPlugin) SupportsMountOption() bool {
	return true
}

func (plugin *nfsPlugin) SupportsSELinuxContextMount(spec *volume.Spec) (bool, error) {
	return false, nil
}

func (plugin *nfsPlugin) GetAccessModes() []v1.PersistentVolumeAccessMode {
	return []v1.PersistentVolumeAccessMode{
		v1.ReadWriteOnce,
		v1.ReadOnlyMany,
		v1.ReadWriteMany,
	}
}

func (plugin *nfsPlugin) NewMounter(spec *volume.Spec, pod *v1.Pod) (volume.Mounter, error) {
	return plugin.newMounterInternal(spec, pod, plugin.host.GetMounter())
}

func (plugin *nfsPlugin) newMounterInternal(spec *volume.Spec, pod *v1.Pod, mounter mount.Interface) (volume.Mounter, error) {
	source, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}
	return &nfsMounter{
		nfs: &nfs{
			volName:         spec.Name(),
			mounter:         mounter,
			pod:             pod,
			plugin:          plugin,
			MetricsProvider: volume.NewMetricsStatFS(getPath(pod.UID, spec.Name(), plugin.host)),
		},
		server:       getServerFromSource(source),
		exportPath:   source.Path,
		readOnly:     readOnly,
		mountOptions: util.MountOptionFromSpec(spec),
	}, nil
}

func (plugin *nfsPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return plugin.newUnmounterInternal(volName, podUID, plugin.host.GetMounter())
}

func (plugin *nfsPlugin) newUnmounterInternal(volName string, podUID types.UID, mounter mount.Interface) (volume.Unmounter, error) {
	return &nfsUnmounter{&nfs{
		volName:         volName,
		mounter:         mounter,
		pod:             &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: podUID}},
		plugin:          plugin,
		MetricsProvider: volume.NewMetricsStatFS(getPath(podUID, volName, plugin.host)),
	}}, nil
}

// Recycle recycles/scrubs clean an NFS volume.
// Recycle blocks until the pod has completed or any error occurs.
func (plugin *nfsPlugin) Recycle(pvName string, spec *volume.Spec, eventRecorder recyclerclient.RecycleEventRecorder) error {
	if spec.PersistentVolume == nil || spec.PersistentVolume.Spec.NFS == nil {
		return fmt.Errorf("spec.PersistentVolumeSource.NFS is nil")
	}

	pod := plugin.config.RecyclerPodTemplate
	timeout := util.CalculateTimeoutForVolume(plugin.config.RecyclerMinimumTimeout, plugin.config.RecyclerTimeoutIncrement, spec.PersistentVolume)
	// overrides
	pod.Spec.ActiveDeadlineSeconds = &timeout
	pod.GenerateName = "pv-recycler-nfs-"
	pod.Spec.Volumes[0].VolumeSource = v1.VolumeSource{
		NFS: &v1.NFSVolumeSource{
			Server: spec.PersistentVolume.Spec.NFS.Server,
			Path:   spec.PersistentVolume.Spec.NFS.Path,
		},
	}
	return recyclerclient.RecycleVolumeByWatchingPodUntilCompletion(pvName, pod, plugin.host.GetKubeClient(), eventRecorder)
}

func (plugin *nfsPlugin) ConstructVolumeSpec(volumeName, mountPath string) (volume.ReconstructedVolume, error) {
	nfsVolume := &v1.Volume{
		Name: volumeName,
		VolumeSource: v1.VolumeSource{
			NFS: &v1.NFSVolumeSource{
				Path: volumeName,
			},
		},
	}
	return volume.ReconstructedVolume{
		Spec: volume.NewSpecFromVolume(nfsVolume),
	}, nil
}

// NFS volumes represent a bare host file or directory mount of an NFS export.
type nfs struct {
	volName string
	pod     *v1.Pod
	mounter mount.Interface
	plugin  *nfsPlugin
	volume.MetricsProvider
}

func (nfsVolume *nfs) GetPath() string {
	name := nfsPluginName
	return nfsVolume.plugin.host.GetPodVolumeDir(nfsVolume.pod.UID, utilstrings.EscapeQualifiedName(name), nfsVolume.volName)
}

type nfsMounter struct {
	*nfs
	server       string
	exportPath   string
	readOnly     bool
	mountOptions []string
}

var _ volume.Mounter = &nfsMounter{}

func (nfsMounter *nfsMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:       nfsMounter.readOnly,
		Managed:        false,
		SELinuxRelabel: false,
	}
}

// SetUp attaches the disk and bind mounts to the volume path.
func (nfsMounter *nfsMounter) SetUp(mounterArgs volume.MounterArgs) error {
	return nfsMounter.SetUpAt(nfsMounter.GetPath(), mounterArgs)
}

func (nfsMounter *nfsMounter) SetUpAt(dir string, mounterArgs volume.MounterArgs) error {
	notMnt, err := mount.IsNotMountPoint(nfsMounter.mounter, dir)
	klog.V(4).Infof("NFS mount set up: %s %v %v", dir, !notMnt, err)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	if !notMnt {
		return nil
	}
	if err := os.MkdirAll(dir, 0750); err != nil {
		return err
	}
	source := fmt.Sprintf("%s:%s", nfsMounter.server, nfsMounter.exportPath)
	options := []string{}
	if nfsMounter.readOnly {
		options = append(options, "ro")
	}
	mountOptions := util.JoinMountOptions(nfsMounter.mountOptions, options)
	err = nfsMounter.mounter.MountSensitiveWithoutSystemd(source, dir, "nfs", mountOptions, nil)
	if err != nil {
		notMnt, mntErr := mount.IsNotMountPoint(nfsMounter.mounter, dir)
		if mntErr != nil {
			klog.Errorf("IsNotMountPoint check failed: %v", mntErr)
			return err
		}
		if !notMnt {
			if mntErr = nfsMounter.mounter.Unmount(dir); mntErr != nil {
				klog.Errorf("Failed to unmount: %v", mntErr)
				return err
			}
			notMnt, mntErr := mount.IsNotMountPoint(nfsMounter.mounter, dir)
			if mntErr != nil {
				klog.Errorf("IsNotMountPoint check failed: %v", mntErr)
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

var _ volume.Unmounter = &nfsUnmounter{}

type nfsUnmounter struct {
	*nfs
}

func (c *nfsUnmounter) TearDown() error {
	return c.TearDownAt(c.GetPath())
}

func (c *nfsUnmounter) TearDownAt(dir string) error {
	// Use extensiveMountPointCheck to consult /proc/mounts. We can't use faster
	// IsLikelyNotMountPoint (lstat()), since there may be root_squash on the
	// NFS server and kubelet may not be able to do lstat/stat() there.
	forceUnmounter, ok := c.mounter.(mount.MounterForceUnmounter)
	if ok {
		klog.V(4).Infof("Using force unmounter interface")
		return mount.CleanupMountWithForce(dir, forceUnmounter, true /* extensiveMountPointCheck */, unMountTimeout)
	}
	return mount.CleanupMountPoint(dir, c.mounter, true /* extensiveMountPointCheck */)
}

func getVolumeSource(spec *volume.Spec) (*v1.NFSVolumeSource, bool, error) {
	if spec.Volume != nil && spec.Volume.NFS != nil {
		return spec.Volume.NFS, spec.Volume.NFS.ReadOnly, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.NFS != nil {
		return spec.PersistentVolume.Spec.NFS, spec.ReadOnly, nil
	}

	return nil, false, fmt.Errorf("Spec does not reference a NFS volume type")
}

func getServerFromSource(source *v1.NFSVolumeSource) string {
	if netutil.IsIPv6String(source.Server) {
		return fmt.Sprintf("[%s]", source.Server)
	}
	return source.Server
}
