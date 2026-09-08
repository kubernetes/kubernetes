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
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/volume"
)

var _ volume.GlobalVolumeListerPlugin = &csiPlugin{}

// ListGlobalVolumes reports every CSI volume staged on this node, whether or
// not a pod directory still refers to it. Reconstruction cannot find those
// volumes on its own, because it walks pod directories, and a global mount
// outlives its pod directory whenever kubelet stops between NodeUnpublishVolume
// and NodeUnstageVolume.
//
// The layout walked here is the one MountDevice writes:
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
//
// TODO(#121937): raw block volumes stage under volumeDevices with a layout of
// their own and are skipped here. They leak the same way and are tracked for
// beta.
func (p *csiPlugin) ListGlobalVolumes() ([]volume.GlobalVolume, error) {
	pluginDir := p.host.GetPluginDir(CSIPluginName)

	// volumeDevices is a sibling of the per-driver directories rather than a
	// driver, and holds raw block volumes. Compare the whole path, since a bare
	// basename would silently stop excluding anything if the host returned "".
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
	return found, nil
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
