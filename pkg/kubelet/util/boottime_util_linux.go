//go:build linux

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
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"golang.org/x/sys/unix"
	"k8s.io/klog/v2"
)

// GetBootTime returns the time at which the machine was started, truncated to the nearest second.
// It uses /proc/stat first, which is more accurate, and falls back to the less accurate
// unix.Sysinfo if /proc/stat failed.
func GetBootTime() (time.Time, error) {
	bootTime, err := getBootTimeWithProcStat()
	if err != nil {
		// TODO: it needs to be replaced by a proper context in the future
		ctx := context.TODO()
		logger := klog.FromContext(ctx)
		logger.Info("Failed to get boot time from /proc/uptime. Will retry with unix.Sysinfo.", "error", err)
		return getBootTimeWithSysinfo()
	}
	return bootTime, nil
}

func getBootTimeWithProcStat() (time.Time, error) {
	raw, err := os.ReadFile("/proc/stat")
	if err != nil {
		return time.Time{}, fmt.Errorf("error getting boot time: %w", err)
	}
	rawFields := strings.Fields(string(raw))
	for i, v := range rawFields {
		if v == "btime" {
			if len(rawFields) > i+1 {
				sec, err := strconv.ParseInt(rawFields[i+1], 10, 64)
				if err != nil {
					return time.Time{}, fmt.Errorf("error parsing boot time %s: %w", rawFields[i+1], err)
				}
				return time.Unix(sec, 0), nil
			}
			break
		}
	}

	return time.Time{}, fmt.Errorf("can not find btime from /proc/stat: %s", raw)
}

func getBootTimeWithSysinfo() (time.Time, error) {
	currentTime := time.Now()
	var info unix.Sysinfo_t
	if err := unix.Sysinfo(&info); err != nil {
		return time.Time{}, fmt.Errorf("error getting system uptime: %w", err)
	}
	return currentTime.Add(-time.Duration(info.Uptime) * time.Second).Truncate(time.Second), nil
}
