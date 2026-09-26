/*
Copyright The Kubernetes Authors.

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

package csi

import (
	"fmt"
	"os"
	"path/filepath"

	api "k8s.io/api/core/v1"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/volume"
)

const (
	// blockStagingDirName and blockPublishDirName are siblings of the
	// per-volume directories under volumeDevices rather than volumes, so the
	// block walk has to step over them by name. They are the literals
	// csiBlockMapper.GetStagingPath and getPublishDir build their paths from.
	blockStagingDirName = "staging"
	blockPublishDirName = "publish"
	// blockVolumeDataDirName holds a block volume's vol_data.json, the way
	// getVolumeDeviceDataDir lays it out. The listing needs it before it knows
	// which volume the directory belongs to, so it cannot use that helper.
	blockVolumeDataDirName = "data"
)

var _ volume.GlobalVolumeListerPlugin = &csiPlugin{}

// ListGlobalVolumes reports every CSI volume staged on this node, whether or
// not a pod directory still refers to it. Reconstruction cannot find those
// volumes on its own, because it walks pod directories, and a global mount
// outlives its pod directory whenever kubelet stops between NodeUnpublishVolume
// and NodeUnstageVolume.
//
// Filesystem and raw block volumes are staged under layouts of their own, so
// this is two walks rather than one. The filesystem layout is the one
// MountDevice writes:
//
//	<pluginDir>/<driver>/<sha256(volumeHandle)>/globalmount   the staged volume
//	<pluginDir>/<driver>/<sha256(volumeHandle)>/vol_data.json its identity
//
// A directory that cannot be described is skipped rather than reported as an
// error, so that one volume staged by an older kubelet, or one caught
// mid-MountDevice, does not hide the rest.
//
// The walk is os.ReadDir at a fixed depth rather than filepath.WalkDir, which
// would descend into globalmount, that is, into the volume's own contents.
func (p *csiPlugin) ListGlobalVolumes() ([]volume.GlobalVolume, error) {
	pluginDir := p.host.GetPluginDir(CSIPluginName)

	// volumeDevices is a sibling of the per-driver directories rather than a
	// driver, and holds raw block volumes, which are walked separately below.
	// Compare the whole path, since a bare basename would silently stop
	// excluding anything if the host returned "".
	blockDir := p.host.GetVolumeDevicePluginDir(CSIPluginName)

	drivers, err := os.ReadDir(pluginDir)
	if err != nil {
		if os.IsNotExist(err) {
			// No CSI volume has ever been staged on this node.
			return nil, nil
		}
		return nil, fmt.Errorf("failed to read the CSI plugin directory %q: %w", pluginDir, err)
	}

	var found []volume.GlobalVolume
	for _, driver := range drivers {
		driverDir := filepath.Join(pluginDir, driver.Name())
		if !driver.IsDir() || driverDir == blockDir {
			continue
		}
		volumes, err := os.ReadDir(driverDir)
		if err != nil {
			klog.V(4).Info(log("skipping driver directory %s: %v", driverDir, err))
			continue
		}
		for _, vol := range volumes {
			if !vol.IsDir() {
				continue
			}
			if gv, ok := p.describeGlobalVolume(filepath.Join(driverDir, vol.Name())); ok {
				found = append(found, gv)
			}
		}
	}
	return append(found, p.listGlobalBlockVolumes(blockDir)...), nil
}

// describeGlobalVolume turns one staged volume directory into a GlobalVolume,
// reporting whether it could be described at all.
func (p *csiPlugin) describeGlobalVolume(volDir string) (volume.GlobalVolume, bool) {
	deviceMountPath := filepath.Join(volDir, globalMountInGlobalPath)
	if _, err := os.Stat(deviceMountPath); err != nil {
		klog.V(4).Info(log("skipping %s, no staged volume: %v", volDir, err))
		return volume.GlobalVolume{}, false
	}

	data, err := loadVolumeData(volDir, volDataFileName)
	if err != nil {
		klog.V(4).Info(log("skipping %s with no readable volume data: %v", volDir, err))
		return volume.GlobalVolume{}, false
	}
	if data[volDataKey.driverName] == "" || data[volDataKey.volHandle] == "" {
		klog.V(4).Info(log("skipping %s, volume data names no driver or handle", volDir))
		return volume.GlobalVolume{}, false
	}

	// The spec has to carry the real volume handle: GetVolumeName derives the
	// unique volume name from it, and GenerateUnmountDeviceFunc recomputes the
	// device mount path from it, so a spec naming a different handle would
	// unstage a different directory.
	//
	// specVolID only names the volume for a human reading a log, and a kubelet
	// older than this feature never wrote one. Those are precisely the volumes
	// staged on the node at the moment the feature is turned on, so fall back
	// to the volume handle rather than passing them over.
	specVolID := data[volDataKey.specVolID]
	if specVolID == "" {
		specVolID = data[volDataKey.volHandle]
	}

	spec := p.constructPVSourceSpec(specVolID, data[volDataKey.driverName], data[volDataKey.volHandle])

	// MountDevice stages every volume under sha256(volumeHandle), and
	// GenerateUnmountDeviceFunc recomputes that path from the spec rather than
	// using the one reported here. A directory whose volume data names another
	// volume would therefore be unstaged at a path that is not this one, which
	// reports success while leaving this mount in place.
	expected, err := makeDeviceMountPath(p, spec)
	if err != nil || expected != deviceMountPath {
		klog.V(4).Info(log("skipping %s, volume data belongs to another volume: %v", volDir, err))
		return volume.GlobalVolume{}, false
	}

	return volume.GlobalVolume{
		ReconstructedVolume: volume.ReconstructedVolume{
			Spec:                spec,
			SELinuxMountContext: data[volDataKey.seLinuxMountContext],
		},
		DeviceMountPath: deviceMountPath,
		VolumeMode:      api.PersistentVolumeFilesystem,
	}, true
}

// listGlobalBlockVolumes reports every raw block volume still staged under the
// volumeDevices subtree, whose layout is its own:
//
//	<blockDir>/<specVolID>/data/vol_data.json  its identity
//	<blockDir>/<specVolID>/dev                 the global map path
//	<blockDir>/staging/<specVolID>             the staged volume
//
// A subtree that cannot be read is skipped rather than reported as an error,
// for the same reason one unreadable directory does not hide the others: a node
// that has never staged a block volume has no subtree at all.
func (p *csiPlugin) listGlobalBlockVolumes(blockDir string) []volume.GlobalVolume {
	entries, err := os.ReadDir(blockDir)
	if err != nil {
		if !os.IsNotExist(err) {
			klog.V(4).Info(log("skipping the raw block subtree %s: %v", blockDir, err))
		}
		return nil
	}

	var found []volume.GlobalVolume
	for _, entry := range entries {
		name := entry.Name()
		// staging and publish sit beside the volume directories and are not
		// volumes, so they are left out by name.
		if !entry.IsDir() || name == blockStagingDirName || name == blockPublishDirName {
			continue
		}
		if gv, ok := p.describeGlobalBlockVolume(blockDir, name); ok {
			found = append(found, gv)
		}
	}
	return found
}

// describeGlobalBlockVolume turns one raw block volume directory into a
// GlobalVolume, reporting whether it could be described at all.
func (p *csiPlugin) describeGlobalBlockVolume(blockDir, name string) (volume.GlobalVolume, bool) {
	volDir := filepath.Join(blockDir, name)

	// The staging directory is what says the volume is still staged, and it is
	// the only thing that does. stageVolumeForBlock creates it before
	// NodeStageVolume and TearDownDevice removes it after NodeUnstageVolume,
	// whereas the volume directory outlives both, and the global map path is
	// not created until the volume is mapped into a pod. Looking for that path
	// instead would pass over a volume staged for a pod that never ran, which
	// leaks exactly like the rest.
	stagingPath := filepath.Join(blockDir, blockStagingDirName, name)
	if _, err := os.Stat(stagingPath); err != nil {
		klog.V(4).Info(log("skipping %s, no staged volume: %v", volDir, err))
		return volume.GlobalVolume{}, false
	}

	data, err := loadVolumeData(filepath.Join(volDir, blockVolumeDataDirName), volDataFileName)
	if err != nil {
		klog.V(4).Info(log("skipping %s with no readable volume data: %v", volDir, err))
		return volume.GlobalVolume{}, false
	}
	if data[volDataKey.driverName] == "" || data[volDataKey.volHandle] == "" {
		klog.V(4).Info(log("skipping %s, volume data names no driver or handle", volDir))
		return volume.GlobalVolume{}, false
	}

	// Every path a block teardown touches is built from specVolID, so the spec
	// has to carry the one this directory is named after. There is no hash to
	// recompute here as there is for a filesystem volume, which leaves one
	// check: the name the file carries has to be the name of the directory
	// holding it. A volume whose data names another volume is passed over
	// rather than torn down under a name that is not its own.
	//
	// Unlike a filesystem volume, this cannot fall back to the volume handle
	// when the file carries no specVolID, because the name is what the paths
	// are built from and taking it from the directory would be checking the
	// directory against itself. NewBlockVolumeMapper has always written it.
	specVolID := data[volDataKey.specVolID]
	if specVolID == "" || getVolumePluginDir(specVolID, p.host) != volDir {
		klog.V(4).Info(log("skipping %s, volume data belongs to another volume", volDir))
		return volume.GlobalVolume{}, false
	}

	blockMode := api.PersistentVolumeBlock
	spec := volume.NewSpecFromPersistentVolume(&api.PersistentVolume{
		ObjectMeta: meta.ObjectMeta{Name: specVolID},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeSource: api.PersistentVolumeSource{
				CSI: &api.CSIPersistentVolumeSource{
					Driver:       data[volDataKey.driverName],
					VolumeHandle: data[volDataKey.volHandle],
				},
			},
			VolumeMode: &blockMode,
		},
	}, false)

	// GenerateUnmapDeviceFunc takes the reported path as the global map path,
	// which for a block volume is the dev directory rather than the staging
	// one. It reads it for pod references and removes it, and tolerates its
	// absence as a SetUpDevice that did not finish.
	return volume.GlobalVolume{
		ReconstructedVolume: volume.ReconstructedVolume{
			// A raw block volume is not mounted, so no mount context applies.
			Spec: spec,
		},
		DeviceMountPath: getVolumeDevicePluginDir(specVolID, p.host),
		VolumeMode:      api.PersistentVolumeBlock,
	}, true
}
