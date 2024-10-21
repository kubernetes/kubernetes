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
	"os"
	"strconv"
	"strings"
	"time"

	"golang.org/x/sys/unix"
	"k8s.io/klog/v2"
)

// GetBootTime returns the time at which the machine was started, truncated to the nearest second.
// It uses /proc/uptime first, which is more accurate, and falls back to the less accurate
// unix.Sysinfo if /proc/uptime failed.
func GetBootTime() (time.Time, error) {
	bootTime, err := getBootTimeWithProcUptime()
	if err != nil {
		klog.ErrorS(err, "Failed to get boot time from /proc/uptime. Will retry with unix.Sysinfo")
		return getBootTimeWithSysinfo()
	}
	return bootTime, nil
}

func getBootTimeWithProcUptime() (time.Time, error) {
	currentTime := time.Now()
	raw, err := os.ReadFile("/proc/uptime")
	if err != nil {
		return time.Time{}, fmt.Errorf("error getting system uptime: %w", err)
	}
	uptimeStr, _, found := strings.Cut(string(raw), " ")
	if !found {
		return time.Time{}, fmt.Errorf("unexpected value from /proc/uptime: %s", raw)
	}
	uptime, err := strconv.ParseFloat(uptimeStr, 64)
	if err != nil {
		return time.Time{}, fmt.Errorf("error parsing uptime %s: %w", uptimeStr, err)
	}
	return currentTime.Add(-time.Duration(uptime * float64(time.Second))).Truncate(time.Second), nil
}

func getBootTimeWithSysinfo() (time.Time, error) {
	currentTime := time.Now()
	var info unix.Sysinfo_t
	if err := unix.Sysinfo(&info); err != nil {
		return time.Time{}, fmt.Errorf("error getting system uptime: %w", err)
	}
	return currentTime.Add(-time.Duration(info.Uptime) * time.Second).Truncate(time.Second), nil
}
