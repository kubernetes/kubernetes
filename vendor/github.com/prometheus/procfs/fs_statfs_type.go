// Copyright 2018 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build !netbsd && !openbsd && !solaris && !windows
// +build !netbsd,!openbsd,!solaris,!windows

package procfs

import (
	"syscall"
)

// isRealProc determines whether supplied mountpoint is really a proc filesystem.
func isRealProc(mountPoint string) (bool, error) {
	stat := syscall.Statfs_t{}
	err := syscall.Statfs(mountPoint, &stat)
	if err != nil {
		return false, err
	}

	// 0x9fa0 is PROC_SUPER_MAGIC: https://elixir.bootlin.com/linux/v6.1/source/include/uapi/linux/magic.h#L87
	return stat.Type == 0x9fa0, nil
}
