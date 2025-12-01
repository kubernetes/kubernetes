//go:build darwin

/*
Copyright 2018 The Kubernetes Authors.

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
	"syscall"
	"time"
	"unsafe"

	"golang.org/x/sys/unix"
)

// GetBootTime returns the time at which the machine was started, truncated to the nearest second
func GetBootTime() (time.Time, error) {
	output, err := unix.SysctlRaw("kern.boottime")
	if err != nil {
		return time.Time{}, err
	}
	var timeval syscall.Timeval
	if len(output) != int(unsafe.Sizeof(timeval)) {
		return time.Time{}, fmt.Errorf("unexpected output when calling syscall kern.bootime.  Expected len(output) to be %v, but got %v",
			int(unsafe.Sizeof(timeval)), len(output))
	}
	timeval = *(*syscall.Timeval)(unsafe.Pointer(&output[0]))
	sec, nsec := timeval.Unix()
	return time.Unix(sec, nsec).Truncate(time.Second), nil
}
