// +build linux

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

package empty_dir

import (
	"fmt"
	"syscall"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/util/mount"
)

// Defined by Linux - the type number for tmpfs mounts.
const linuxTmpfsMagic = 0x01021994

// realMountDetector implements mountDetector in terms of syscalls.
type realMountDetector struct {
	mounter mount.Interface
}

func (m *realMountDetector) GetMountMedium(path string) (storageMedium, bool, error) {
	glog.V(5).Infof("Determining mount medium of %v", path)
	notMnt, err := m.mounter.IsLikelyNotMountPoint(path)
	if err != nil {
		return 0, false, fmt.Errorf("IsLikelyNotMountPoint(%q): %v", path, err)
	}
	buf := syscall.Statfs_t{}
	if err := syscall.Statfs(path, &buf); err != nil {
		return 0, false, fmt.Errorf("statfs(%q): %v", path, err)
	}

	glog.V(5).Infof("Statfs_t of %v: %+v", path, buf)
	if buf.Type == linuxTmpfsMagic {
		return mediumMemory, !notMnt, nil
	}
	return mediumUnknown, !notMnt, nil
}
