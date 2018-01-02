// Copyright 2015 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//+build linux

package cgroup

import (
	"errors"
	"path/filepath"
	"syscall"

	"github.com/coreos/rkt/common/cgroup/v1"
	"github.com/coreos/rkt/common/cgroup/v2"
	"github.com/hashicorp/errwrap"
)

const (
	// The following const comes from
	// #define CGROUP2_SUPER_MAGIC  0x63677270
	// https://github.com/torvalds/linux/blob/v4.6/include/uapi/linux/magic.h#L58
	Cgroup2fsMagicNumber = 0x63677270
)

// IsIsolatorSupported returns whether an isolator is supported in the kernel
func IsIsolatorSupported(isolator string) (bool, error) {
	isUnified, err := IsCgroupUnified("/")
	if err != nil {
		return false, errwrap.Wrap(errors.New("error determining cgroup version"), err)
	}

	if isUnified {
		controllers, err := v2.GetEnabledControllers()
		if err != nil {
			return false, errwrap.Wrap(errors.New("error determining enabled controllers"), err)
		}
		for _, c := range controllers {
			if c == isolator {
				return true, nil
			}
		}
		return false, nil
	}
	return v1.IsControllerMounted(isolator)
}

// IsCgroupUnified checks if cgroup mounted at /sys/fs/cgroup is
// the new unified version (cgroup v2)
func IsCgroupUnified(root string) (bool, error) {
	cgroupFsPath := filepath.Join(root, "/sys/fs/cgroup")
	var statfs syscall.Statfs_t
	if err := syscall.Statfs(cgroupFsPath, &statfs); err != nil {
		return false, err
	}

	return statfs.Type == Cgroup2fsMagicNumber, nil
}
