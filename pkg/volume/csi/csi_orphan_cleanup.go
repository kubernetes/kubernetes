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

// CleanupUnmountedVolumeArtifacts removes residual CSI volume directory content when
// the volume is not mounted.
//
// After a hard node reboot, CSI mount points are typically gone while the per-volume
// metadata file (vol_data.json) remains under:
//
//	.../volumes/kubernetes.io~csi/<volume>/vol_data.json
//
// Orphaned-pod cleanup intentionally uses rmdir-only traversal on the volumes tree
// (see kubelet removeOrphanedPodVolumeDirs / pkg/util/removeall.RemoveDirsOneFilesystem)
// so that real mount contents are never recursively deleted. Residual vol_data.json
// therefore blocks cleanup and produces perpetual "not a directory" errors.
//
// Safety:
//   - If the CSI mount path is still a mount point, this is a no-op.
//   - Only the well-known CSI metadata file (vol_data.json) and empty directories are removed.
//   - Arbitrary files under the volume path are left untouched.
func CleanupUnmountedVolumeArtifacts(mounter mount.Interface, volumeDir string) error {
	if mounter == nil {
		return fmt.Errorf("mounter is required")
	}

	exists, err := mount.PathExists(volumeDir)
	if err != nil {
		return err
	}
	if !exists {
		return nil
	}

	mountPath := GetCSIMounterPath(volumeDir)
	mountExists, err := mount.PathExists(mountPath)
	if err != nil {
		return err
	}
	if mountExists {
		notMnt, err := mounter.IsLikelyNotMountPoint(mountPath)
		if err != nil {
			// Path may race; treat missing path as unmounted.
			if !os.IsNotExist(err) {
				return fmt.Errorf("failed to check mount point %q: %w", mountPath, err)
			}
			notMnt = true
		}
		if !notMnt {
			// Still mounted: never touch metadata or the volume directory.
			return nil
		}
		// Unmounted local directory left after reboot / successful unmount.
		if err := os.Remove(mountPath); err != nil && !os.IsNotExist(err) {
			// Non-empty mount dir may still hold data; do not delete vol_data.json yet.
			return fmt.Errorf("failed to remove unmounted CSI mount path %q: %w", mountPath, err)
		}
	}

	dataFile := filepath.Join(volumeDir, volDataFileName)
	if err := os.Remove(dataFile); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to remove CSI volume data file %q: %w", dataFile, err)
	}
	klog.V(4).InfoS("Removed residual CSI volume metadata", "path", dataFile)

	// Best-effort: remove the volume directory if it is now empty.
	if err := os.Remove(volumeDir); err != nil && !os.IsNotExist(err) {
		// Directory still has residual content that is not safe for this helper to delete.
		klog.V(4).InfoS("CSI volume directory not empty after metadata cleanup", "path", volumeDir, "err", err)
	}
	return nil
}
