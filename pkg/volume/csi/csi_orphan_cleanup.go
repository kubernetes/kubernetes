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
// Prerequisites / safety (review notes for #102576 / reconstruction):
//   - Callers must only invoke this after determining the pod has no remaining
//     mounted/uncertain volumes (kubelet podVolumesExist == false). That gate
//     is what prevents deleting CSI metadata while volume manager still needs it
//     for NodeUnstage / reconstruction.
//   - This uses mounter.IsMountPoint (not only IsLikelyNotMountPoint) so bind
//     mounts are detected on Linux when the mount table is available.
//   - If the CSI mount path is still a mount point, this is a no-op.
//   - Only the well-known CSI metadata file (vol_data.json) and empty dirs under
//     the pod volume path are removed. Arbitrary files are left untouched.
//   - Scope: filesystem CSI pod volume dirs under volumes/kubernetes.io~csi/.
//     Global plugin staging/publish paths and block volumeDevices layouts are
//     intentionally out of scope for this helper.
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
		// Prefer IsMountPoint: detects bind mounts; more expensive but safer for CSI.
		isMnt, err := mounter.IsMountPoint(mountPath)
		if err != nil {
			// Path may race; treat missing path as unmounted.
			if !os.IsNotExist(err) {
				return false, fmt.Errorf("failed to check mount point %q: %w", mountPath, err)
			}
			isMnt = false
		}
		if isMnt {
			// Still mounted: never touch metadata or the volume directory.
			// Preserves vol_data.json for any remaining unstage/unpublish work.
			return false, nil
		}
		// Unmounted local directory left after reboot / successful unmount.
		if err := os.Remove(mountPath); err != nil && !os.IsNotExist(err) {
			// Non-empty mount dir may still hold data; do not delete vol_data.json yet.
			return false, fmt.Errorf("failed to remove unmounted CSI mount path %q: %w", mountPath, err)
		}
	}

	dataFile := filepath.Join(volumeDir, volDataFileName)
	if err := os.Remove(dataFile); err != nil {
		if os.IsNotExist(err) {
			// Nothing to clean; try empty volumeDir removal below.
			if remErr := os.Remove(volumeDir); remErr != nil && !os.IsNotExist(remErr) {
				klog.V(4).InfoS("CSI volume directory not empty after metadata cleanup", "path", volumeDir, "err", remErr)
			}
			return false, nil
		}
		return false, fmt.Errorf("failed to remove CSI volume data file %q: %w", dataFile, err)
	}
	klog.V(4).InfoS("Removed residual CSI volume metadata", "path", dataFile)
	cleaned = true

	// Best-effort: remove the volume directory if it is now empty.
	if err := os.Remove(volumeDir); err != nil && !os.IsNotExist(err) {
		// Directory still has residual content that is not safe for this helper to delete.
		klog.V(4).InfoS("CSI volume directory not empty after metadata cleanup", "path", volumeDir, "err", err)
	}
	return cleaned, nil
}
