//go:build linux
// +build linux

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
	"time"

	"golang.org/x/sys/unix"
)

// GetBootTime returns the time at which the machine was started, truncated to the nearest second.
// It makes multiple attempts to eliminate inconsistency due to the lack of precision on uptime.
func GetBootTime() (time.Time, error) {
	var info unix.Sysinfo_t
	if err := unix.Sysinfo(&info); err != nil {
		return time.Time{}, fmt.Errorf("error getting system uptime: %s", err)
	}
	uptime_p := info.Uptime
	for i:=0;i<2;i++{
		time.Sleep(200 * time.Millisecond)
		if err := unix.Sysinfo(&info); err != nil {
			return time.Time{}, fmt.Errorf("error getting system uptime: %s", err)
		}
		if info.Uptime == uptime_p {
			return time.Now().Add(-time.Duration(info.Uptime) * time.Second).Truncate(time.Second), nil
		}
		uptime_p = info.Uptime
	}
	return time.Time{}, fmt.Errorf("error getting consistent boot time.")
}
