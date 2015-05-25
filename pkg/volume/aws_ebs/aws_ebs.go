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

package aws_ebs

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/aws"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/exec"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
	"github.com/golang/glog"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&awsElasticBlockStorePlugin{nil}}
}

type awsElasticBlockStorePlugin struct {
	host volume.VolumeHost
}

var _ volume.VolumePlugin = &awsElasticBlockStorePlugin{}

const (
	awsElasticBlockStorePluginName = "kubernetes.io/aws-ebs"
)

func (plugin *awsElasticBlockStorePlugin) Init(host volume.VolumeHost) {
	plugin.host = host
}

func (plugin *awsElasticBlockStorePlugin) Name() string {
	return awsElasticBlockStorePluginName
}

func (plugin *awsElasticBlockStorePlugin) CanSupport(spec *volume.Spec) bool {
	return spec.PersistentVolumeSource.AWSElasticBlockStore != nil || spec.VolumeSource.AWSElasticBlockStore != nil
}

func (plugin *awsElasticBlockStorePlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
	}
}

func (plugin *awsElasticBlockStorePlugin) NewBuilder(spec *volume.Spec, pod *api.Pod, _ volume.VolumeOptions, mounter mount.Interface) (volume.Builder, error) {
	// Inject real implementations here, test through the internal function.
	return plugin.newBuilderInternal(spec, pod.UID, &AWSDiskUtil{}, mounter)
}

func (plugin *awsElasticBlockStorePlugin) newBuilderInternal(spec *volume.Spec, podUID types.UID, manager pdManager, mounter mount.Interface) (volume.Builder, error) {
	var ebs *api.AWSElasticBlockStoreVolumeSource
	if spec.VolumeSource.AWSElasticBlockStore != nil {
		ebs = spec.VolumeSource.AWSElasticBlockStore
	} else {
		ebs = spec.PersistentVolumeSource.AWSElasticBlockStore
	}

	volumeID := ebs.VolumeID
	fsType := ebs.FSType
	partition := ""
	if ebs.Partition != 0 {
		partition = strconv.Itoa(ebs.Partition)
	}
	readOnly := ebs.ReadOnly

	return &awsElasticBlockStore{
		podUID:      podUID,
		volName:     spec.Name,
		volumeID:    volumeID,
		fsType:      fsType,
		partition:   partition,
		readOnly:    readOnly,
		manager:     manager,
		mounter:     mounter,
		diskMounter: &awsSafeFormatAndMount{mounter, exec.New()},
		plugin:      plugin,
	}, nil
}

func (plugin *awsElasticBlockStorePlugin) NewCleaner(volName string, podUID types.UID, mounter mount.Interface) (volume.Cleaner, error) {
	// Inject real implementations here, test through the internal function.
	return plugin.newCleanerInternal(volName, podUID, &AWSDiskUtil{}, mounter)
}

func (plugin *awsElasticBlockStorePlugin) newCleanerInternal(volName string, podUID types.UID, manager pdManager, mounter mount.Interface) (volume.Cleaner, error) {
	return &awsElasticBlockStore{
		podUID:      podUID,
		volName:     volName,
		manager:     manager,
		mounter:     mounter,
		diskMounter: &awsSafeFormatAndMount{mounter, exec.New()},
		plugin:      plugin,
	}, nil
}

// Abstract interface to PD operations.
type pdManager interface {
	// Attaches the disk to the kubelet's host machine.
	AttachAndMountDisk(pd *awsElasticBlockStore, globalPDPath string) error
	// Detaches the disk from the kubelet's host machine.
	DetachDisk(pd *awsElasticBlockStore) error
}

// awsElasticBlockStore volumes are disk resources provided by Google Compute Engine
// that are attached to the kubelet's host machine and exposed to the pod.
type awsElasticBlockStore struct {
	volName string
	podUID  types.UID
	// Unique id of the PD, used to find the disk resource in the provider.
	volumeID string
	// Filesystem type, optional.
	fsType string
	// Specifies the partition to mount
	partition string
	// Specifies whether the disk will be attached as read-only.
	readOnly bool
	// Utility interface that provides API calls to the provider to attach/detach disks.
	manager pdManager
	// Mounter interface that provides system calls to mount the global path to the pod local path.
	mounter mount.Interface
	// diskMounter provides the interface that is used to mount the actual block device.
	diskMounter mount.Interface
	plugin      *awsElasticBlockStorePlugin
}

func detachDiskLogError(pd *awsElasticBlockStore) {
	err := pd.manager.DetachDisk(pd)
	if err != nil {
		glog.Warningf("Failed to detach disk: %v (%v)", pd, err)
	}
}

// getVolumeProvider returns the AWS Volumes interface
func (pd *awsElasticBlockStore) getVolumeProvider() (aws_cloud.Volumes, error) {
	name := "aws"
	cloud, err := cloudprovider.GetCloudProvider(name, nil)
	if err != nil {
		return nil, err
	}
	volumes, ok := cloud.(aws_cloud.Volumes)
	if !ok {
		return nil, fmt.Errorf("Cloud provider does not support volumes")
	}
	return volumes, nil
}

// SetUp attaches the disk and bind mounts to the volume path.
func (pd *awsElasticBlockStore) SetUp() error {
	return pd.SetUpAt(pd.GetPath())
}

// SetUpAt attaches the disk and bind mounts to the volume path.
func (pd *awsElasticBlockStore) SetUpAt(dir string) error {
	// TODO: handle failed mounts here.
	mountpoint, err := pd.mounter.IsMountPoint(dir)
	glog.V(4).Infof("PersistentDisk set up: %s %v %v", dir, mountpoint, err)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	if mountpoint {
		return nil
	}

	globalPDPath := makeGlobalPDPath(pd.plugin.host, pd.volumeID)
	if err := pd.manager.AttachAndMountDisk(pd, globalPDPath); err != nil {
		return err
	}

	if err := os.MkdirAll(dir, 0750); err != nil {
		// TODO: we should really eject the attach/detach out into its own control loop.
		detachDiskLogError(pd)
		return err
	}

	// Perform a bind mount to the full path to allow duplicate mounts of the same PD.
	options := []string{"bind"}
	if pd.readOnly {
		options = append(options, "ro")
	}
	err = pd.mounter.Mount(globalPDPath, dir, "", options)
	if err != nil {
		mountpoint, mntErr := pd.mounter.IsMountPoint(dir)
		if mntErr != nil {
			glog.Errorf("isMountpoint check failed: %v", mntErr)
			return err
		}
		if mountpoint {
			if mntErr = pd.mounter.Unmount(dir); mntErr != nil {
				glog.Errorf("Failed to unmount: %v", mntErr)
				return err
			}
			mountpoint, mntErr := pd.mounter.IsMountPoint(dir)
			if mntErr != nil {
				glog.Errorf("isMountpoint check failed: %v", mntErr)
				return err
			}
			if mountpoint {
				// This is very odd, we don't expect it.  We'll try again next sync loop.
				glog.Errorf("%s is still mounted, despite call to unmount().  Will try again next sync loop.", dir)
				return err
			}
		}
		os.Remove(dir)
		// TODO: we should really eject the attach/detach out into its own control loop.
		detachDiskLogError(pd)
		return err
	}

	return nil
}

func makeGlobalPDPath(host volume.VolumeHost, volumeID string) string {
	// Clean up the URI to be more fs-friendly
	name := volumeID
	name = strings.Replace(name, "://", "/", -1)
	return path.Join(host.GetPluginDir(awsElasticBlockStorePluginName), "mounts", name)
}

func getVolumeIDFromGlobalMount(host volume.VolumeHost, globalPath string) (string, error) {
	basePath := path.Join(host.GetPluginDir(awsElasticBlockStorePluginName), "mounts")
	rel, err := filepath.Rel(basePath, globalPath)
	if err != nil {
		return "", err
	}
	if strings.Contains(rel, "../") {
		return "", fmt.Errorf("Unexpected mount path: " + globalPath)
	}
	// Reverse the :// replacement done in makeGlobalPDPath
	volumeID := rel
	if strings.HasPrefix(volumeID, "aws/") {
		volumeID = strings.Replace(volumeID, "aws/", "aws://", 1)
	}
	glog.V(2).Info("Mapping mount dir ", globalPath, " to volumeID ", volumeID)
	return volumeID, nil
}

func (pd *awsElasticBlockStore) GetPath() string {
	name := awsElasticBlockStorePluginName
	return pd.plugin.host.GetPodVolumeDir(pd.podUID, util.EscapeQualifiedNameForDisk(name), pd.volName)
}

// Unmounts the bind mount, and detaches the disk only if the PD
// resource was the last reference to that disk on the kubelet.
func (pd *awsElasticBlockStore) TearDown() error {
	return pd.TearDownAt(pd.GetPath())
}

// Unmounts the bind mount, and detaches the disk only if the PD
// resource was the last reference to that disk on the kubelet.
func (pd *awsElasticBlockStore) TearDownAt(dir string) error {
	mountpoint, err := pd.mounter.IsMountPoint(dir)
	if err != nil {
		glog.V(2).Info("Error checking if mountpoint ", dir, ": ", err)
		return err
	}
	if !mountpoint {
		glog.V(2).Info("Not mountpoint, deleting")
		return os.Remove(dir)
	}

	refs, err := mount.GetMountRefs(pd.mounter, dir)
	if err != nil {
		glog.V(2).Info("Error getting mountrefs for ", dir, ": ", err)
		return err
	}
	if len(refs) == 0 {
		glog.Warning("Did not find pod-mount for ", dir, " during tear-down")
	}
	// Unmount the bind-mount inside this pod
	if err := pd.mounter.Unmount(dir); err != nil {
		glog.V(2).Info("Error unmounting dir ", dir, ": ", err)
		return err
	}
	// If len(refs) is 1, then all bind mounts have been removed, and the
	// remaining reference is the global mount. It is safe to detach.
	if len(refs) == 1 {
		// pd.volumeID is not initially set for volume-cleaners, so set it here.
		pd.volumeID, err = getVolumeIDFromGlobalMount(pd.plugin.host, refs[0])
		if err != nil {
			glog.V(2).Info("Could not determine volumeID from mountpoint ", refs[0], ": ", err)
			return err
		}
		if err := pd.manager.DetachDisk(pd); err != nil {
			glog.V(2).Info("Error detaching disk ", pd.volumeID, ": ", err)
			return err
		}
	} else {
		glog.V(2).Infof("Found multiple refs; won't detach EBS volume: %v", refs)
	}
	mountpoint, mntErr := pd.mounter.IsMountPoint(dir)
	if mntErr != nil {
		glog.Errorf("isMountpoint check failed: %v", mntErr)
		return err
	}
	if !mountpoint {
		if err := os.Remove(dir); err != nil {
			glog.V(2).Info("Error removing mountpoint ", dir, ": ", err)
			return err
		}
	}
	return nil
}
