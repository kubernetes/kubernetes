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

package util

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/klog/v2"
	"k8s.io/mount-utils"
)

// UnmountSubmounts unmounts every mount point that is strictly below dir.
//
// Mounts can end up below a pod volume directory without the kubelet having
// created them: when a container mounts a volume with Bidirectional mount
// propagation, any mount the runtime or the workload places under that path
// inside the container (for example the bind mount of a terminationMessagePath
// that points into the volume) propagates back to the host copy of the volume
// directory and outlives the container. Such leaked mounts make the volume
// unmount or removal fail with EBUSY forever.
//
// dir is resolved through symlinks before it is compared with the mount table,
// because the mount table always lists canonical paths while the kubelet root
// directory may be a symlink. A dir that does not exist is treated as having
// no submounts. Submounts are unmounted deepest first so that nested mounts are
// handled correctly. All unmount failures are collected and returned together.
//
// The helper does not distinguish mounts created by the kubelet, the runtime,
// or the workload itself: everything below dir is unmounted. On a shared
// (Bidirectional) volume, the unmount propagates to peers.
func UnmountSubmounts(mounter mount.Interface, dir string) error {
	resolvedDir, err := filepath.EvalSymlinks(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("failed to resolve %s: %w", dir, err)
	}

	mountPoints, err := mounter.List()
	if err != nil {
		return fmt.Errorf("failed to list mount points: %w", err)
	}

	var submounts []string
	for _, mp := range mountPoints {
		if mp.Path != resolvedDir && mount.PathWithinBase(mp.Path, resolvedDir) {
			submounts = append(submounts, mp.Path)
		}
	}
	if len(submounts) == 0 {
		return nil
	}

	// A parent mount point is always a strict prefix of its children, so
	// reverse lexicographic order unmounts the deepest mounts first.
	slices.Sort(submounts)
	slices.Reverse(submounts)

	var errs []error
	for _, mp := range submounts {
		klog.V(4).InfoS("Unmounting submount found under volume directory", "path", mp, "volumeDir", dir)
		if err := mounter.Unmount(mp); err != nil {
			errs = append(errs, fmt.Errorf("failed to unmount %s: %w", mp, err))
		}
	}
	return utilerrors.NewAggregate(errs)
}
