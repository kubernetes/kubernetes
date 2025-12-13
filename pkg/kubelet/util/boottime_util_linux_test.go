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
	"testing"
	"time"
)

func TestGetBootTime(t *testing.T) {
	boottime, err := GetBootTime()

	if err != nil {
		t.Errorf("Unable to get system uptime")
	}

	if !boottime.After(time.Time{}) {
		t.Errorf("Invalid system uptime")
	}
}

func TestVariationBetweenGetBootTimeMethods(t *testing.T) {
	boottime1, err := getBootTimeWithProcStat()
	if err != nil {
		t.Errorf("Unable to get boot time from /proc/uptime")
	}
	boottime2, err := getBootTimeWithSysinfo()
	if err != nil {
		t.Errorf("Unable to get boot time from unix.Sysinfo")
	}
	diff := boottime1.Sub(boottime2)
	if diff > time.Second || diff < -time.Second {
		t.Errorf("boot time produced by 2 methods should not vary more than a second")
	}
}
