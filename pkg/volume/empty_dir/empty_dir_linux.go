// +build linux

/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"path"
	"syscall"

	"github.com/golang/glog"
	"github.com/opencontainers/runc/libcontainer/selinux"
	"k8s.io/kubernetes/pkg/util/mount"
)

// Defined by Linux - the type number for tmpfs mounts.
const linuxTmpfsMagic = 0x01021994

// realMountDetector implements mountDetector in terms of syscalls.
type realMountDetector struct {
	mounter mount.Interface
}

func (m *realMountDetector) GetMountMedium(p string) (storageMedium, bool, error) {
	glog.V(5).Infof("Determining mount medium of %v", p)
	notMnt, err := m.mounter.IsLikelyNotMountPoint(p)
	if err != nil {
		return 0, false, fmt.Errorf("IsLikelyNotMountPoint(%q): %v", p, err)
	}

	// Workaround buggy filesystems like vmhgfs which fail on statfs with long
	// file names, walk up the tree stopping just in front of the filesystem border
	stat := syscall.Stat_t{}
	if err = syscall.Stat(p, &stat); err != nil {
		return 0, false, fmt.Errorf("stat(%q): %v", p, err)
	}
	dev := stat.Dev
	dotdot := p
	for dotdot != "/" && err == nil && dev == stat.Dev {
		p, dotdot = dotdot, path.Dir(p)
		dev = stat.Dev
		err = syscall.Stat(dotdot, &stat)
	}

	buf := syscall.Statfs_t{}
	if err := syscall.Statfs(p, &buf); err != nil {
		return 0, false, fmt.Errorf("statfs(%q): %v", p, err)
	}

	glog.V(5).Info("Statfs_t of %v: %+v", p, buf)
	if buf.Type == linuxTmpfsMagic {
		return mediumMemory, !notMnt, nil
	}
	return mediumUnknown, !notMnt, nil
}

// selinuxEnabled determines whether SELinux is enabled.
func selinuxEnabled() bool {
	return selinux.SelinuxEnabled()
}
