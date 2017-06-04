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

package rook

import (
	"fmt"
	"os"
	"path"
	rstring "strings"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&rookPlugin{nil}}
}

type rookPlugin struct {
	host volume.VolumeHost
}

var _ volume.PersistentVolumePlugin = &rookPlugin{}
var _ volume.AttachableVolumePlugin = &rookPlugin{}

const (
	rookPluginName = "kubernetes.io/rook"
)

type rookVolume struct {
	pvName      string
	podUID      types.UID
	volumeID    string
	volumeGroup string
	cluster     string
	mounter     mount.Interface
	plugin      *rookPlugin
	volume.MetricsProvider
}

type rookMounter struct {
	*rookVolume
	// Specifies whether the disk will be mounted as read-only.
	readOnly bool
}

var _ volume.Mounter = &rookMounter{}

func getPath(uid types.UID, pvName string, host volume.VolumeHost) string {
	return host.GetPodVolumeDir(uid, strings.EscapeQualifiedNameForDisk(rookPluginName), pvName)
}

func makeGlobalPDName(host volume.VolumeHost, devName string) string {
	return path.Join(host.GetPluginDir(rookPluginName), mount.MountsInGlobalPDPath, devName)
}

func (plugin *rookPlugin) NewAttacher() (volume.Attacher, error) {
	return &rookAttacher{
		host: plugin.host,
	}, nil
}

func (plugin *rookPlugin) NewDetacher() (volume.Detacher, error) {
	return &rookDetacher{
		host: plugin.host,
	}, nil
}

func (plugin *rookPlugin) GetDeviceMountRefs(deviceMountPath string) ([]string, error) {
	mounter := plugin.host.GetMounter()
	return mount.GetMountRefs(mounter, deviceMountPath)
}

func (plugin *rookPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

func (plugin *rookPlugin) GetPluginName() string {
	return rookPluginName
}

func (plugin *rookPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}
	return generateVolumeName(volumeSource.Cluster, volumeSource.VolumeGroup, volumeSource.VolumeID), nil
}

func (plugin *rookPlugin) CanSupport(spec *volume.Spec) bool {
	if (spec.Volume != nil && spec.Volume.Rook == nil) || (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.Rook == nil) {
		return false
	}
	return true
}

func (plugin *rookPlugin) NewMounter(spec *volume.Spec, pod *v1.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
	// Inject real implementations here, test through the internal function.
	return plugin.newMounterInternal(spec, pod.UID)
}

func (plugin *rookPlugin) newMounterInternal(spec *volume.Spec, podUID types.UID) (volume.Mounter, error) {
	volumeSource, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}

	return &rookMounter{
		rookVolume: &rookVolume{
			podUID:          podUID,
			pvName:          spec.Name(),
			volumeID:        volumeSource.VolumeID,
			volumeGroup:     volumeSource.VolumeGroup,
			cluster:         volumeSource.Cluster,
			mounter:         plugin.host.GetMounter(),
			plugin:          plugin,
			MetricsProvider: volume.NewMetricsStatFS(getPath(podUID, spec.Name(), plugin.host)),
		},
		readOnly: readOnly}, nil
}

func (b *rookMounter) SetUp(fsGroup *types.UnixGroupID) error {
	return b.SetUpAt(b.GetPath(), fsGroup)
}

func (b *rookMounter) SetUpAt(dir string, fsGroup *types.UnixGroupID) error {
	notMnt, err := b.mounter.IsLikelyNotMountPoint(dir)
	glog.V(4).Infof("Rook: Setup dir (%s) Volume ID (%s) Volume Group (%s) Mounted (%t) Error (%v), ReadOnly (%t)", dir, b.volumeID, b.volumeGroup, !notMnt, err, b.readOnly)
	if err != nil && !os.IsNotExist(err) {
		glog.Errorf("cannot validate mount point: %s %v", dir, err)
		return err
	}
	if !notMnt {
		return nil
	}

	if err := os.MkdirAll(dir, 0750); err != nil {
		glog.Errorf("mkdir failed on disk %s (%v)", dir, err)
		return err
	}

	// Perform a bind mount to the full path to allow duplicate mounts of the same PD.
	options := []string{"bind"}
	if b.readOnly {
		options = append(options, "ro")
	}

	devName := generateVolumeName(b.cluster, b.volumeGroup, b.volumeID)
	globalPDPath := makeGlobalPDName(b.plugin.host, devName)
	glog.V(4).Infof("attempting to mount %s to %s ", globalPDPath, dir)

	err = b.mounter.Mount(globalPDPath, dir, "", options)
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
		glog.Errorf("Mount of disk %s failed: %v", dir, err)
		return err
	}

	if !b.readOnly {
		volume.SetVolumeOwnership(b, fsGroup)
	}

	glog.V(4).Infof("Successfully mounted %s", dir)
	return nil
}

type rookUnmounter struct {
	*rookVolume
}

var _ volume.Unmounter = &rookUnmounter{}

func (plugin *rookPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return plugin.newUnmounterInternal(volName, podUID)
}

func (plugin *rookPlugin) newUnmounterInternal(volName string, podUID types.UID) (volume.Unmounter, error) {
	return &rookUnmounter{&rookVolume{
		podUID:          podUID,
		pvName:          volName,
		plugin:          plugin,
		mounter:         plugin.host.GetMounter(),
		MetricsProvider: volume.NewMetricsStatFS(getPath(podUID, volName, plugin.host)),
	}}, nil
}

func (c *rookUnmounter) GetPath() string {
	return getPath(c.podUID, c.pvName, c.plugin.host)
}

func (c *rookUnmounter) TearDown() error {
	return c.TearDownAt(c.GetPath())
}

// TearDownAt unmounts the bind mount
func (c *rookUnmounter) TearDownAt(dir string) error {
	return util.UnmountPath(dir, c.mounter)
}

func (plugin *rookPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	mounter := plugin.host.GetMounter()
	pluginDir := plugin.host.GetPluginDir(plugin.GetPluginName())
	sourceName, err := mounter.GetDeviceNameFromMount(mountPath, pluginDir)
	if err != nil {
		return nil, err
	}

	// sourceName is in the form of xxx/cluster-group-id
	devName := path.Base(sourceName)
	volumeSource := rstring.Split(devName, "-")
	rookVolume := &v1.Volume{
		Name: volumeName,
		VolumeSource: v1.VolumeSource{
			Rook: &v1.RookVolumeSource{
				Cluster:     volumeSource[0],
				VolumeGroup: volumeSource[1],
				VolumeID:    volumeSource[2],
			},
		},
	}
	return volume.NewSpecFromVolume(rookVolume), nil
}

func (plugin *rookPlugin) RequiresRemount() bool {
	return false
}

func (plugin *rookPlugin) SupportsMountOption() bool {
	return true
}

func (plugin *rookPlugin) SupportsBulkVolumeVerification() bool {
	return false
}

func (plugin *rookPlugin) GetAccessModes() []v1.PersistentVolumeAccessMode {
	return []v1.PersistentVolumeAccessMode{
		v1.ReadWriteOnce,
		v1.ReadOnlyMany,
		v1.ReadWriteMany,
	}
}

func getVolumeSource(spec *volume.Spec) (*v1.RookVolumeSource, bool, error) {
	if spec.Volume != nil && spec.Volume.Rook != nil {
		return spec.Volume.Rook, spec.Volume.Rook.ReadOnly, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.Rook != nil {
		return spec.PersistentVolume.Spec.Rook, spec.ReadOnly, nil
	}
	return nil, false, fmt.Errorf("Spec does not reference a Rook volume type")
}

func (b *rookMounter) CanMount() error {
	return nil
}

func (b *rookMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        b.readOnly,
		Managed:         !b.readOnly,
		SupportsSELinux: true,
	}
}

func (b *rookMounter) GetPath() string {
	return getPath(b.podUID, b.pvName, b.plugin.host)
}

func generateVolumeName(cluster, group, id string) string {
	return fmt.Sprintf("%s-%s-%s", cluster, group, id)
}
