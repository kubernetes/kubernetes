/*
Copyright 2026 The Kubernetes Authors.

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

	"k8s.io/klog/v2"
	"k8s.io/mount-utils"
)

// CleanupUnmountedVolumeArtifacts removes residual CSI volume directory content
// when the volume is not mounted.
//
// Orphaned-pod volume cleanup uses rmdir-only traversal of the volumes tree
// (see removeall.RemoveDirsOneFilesystem). After a hard reboot, CSI mounts are
// typically gone while vol_data.json remains under
// volumes/kubernetes.io~csi/<volume>/, which blocks cleanup.
//
// Callers must only invoke this after verifying the pod has no remaining
// mounted volumes (kubelet podVolumesExist == false). That gate preserves CSI
// metadata needed for reconstruction / NodeUnstage.
//
// Only the well-known CSI metadata file (vol_data.json) and empty directories under
// the pod volume path are removed. Arbitrary files are left untouched.
// Filesystem CSI pod volume dirs only; staging/publish and block volumeDevices
// paths are out of scope.
//
// Returns cleaned=true only when vol_data.json was successfully removed.
func CleanupUnmountedVolumeArtifacts(mounter mount.Interface, volumeDir string) (cleaned bool, err error) {
	if mounter == nil {
		return false, fmt.Errorf("mounter is required")
	}

	exists, err := mount.PathExists(volumeDir)
	if err != nil {
		return false, err
	}
	if !exists {
		return false, nil
	}

	mountPath := GetCSIMounterPath(volumeDir)
	mountExists, err := mount.PathExists(mountPath)
	if err != nil {
		return false, err
	}
	if mountExists {
		// IsMountPoint detects bind mounts; safer than IsLikelyNotMountPoint for CSI.
		isMnt, err := mounter.IsMountPoint(mountPath)
		if err != nil {
			if !os.IsNotExist(err) {
				return false, fmt.Errorf("failed to check mount point %q: %w", mountPath, err)
			}
			isMnt = false
		}
		if isMnt {
			return false, nil
		}
		if err := os.Remove(mountPath); err != nil && !os.IsNotExist(err) {
			// Non-empty mount dir may still hold data; leave vol_data.json alone.
			return false, fmt.Errorf("failed to remove unmounted CSI mount path %q: %w", mountPath, err)
		}
	}

	dataFile := filepath.Join(volumeDir, volDataFileName)
	if err := os.Remove(dataFile); err != nil {
		if !os.IsNotExist(err) {
			return false, fmt.Errorf("failed to remove CSI volume data file %q: %w", dataFile, err)
		}
	} else {
		klog.V(4).InfoS("Removed residual CSI volume metadata", "path", dataFile)
		cleaned = true
	}

	// Best-effort: remove the volume directory if it is now empty.
	if err := os.Remove(volumeDir); err != nil && !os.IsNotExist(err) {
		klog.V(4).InfoS("CSI volume directory not empty after metadata cleanup", "path", volumeDir, "err", err)
	}
	return cleaned, nil
}
