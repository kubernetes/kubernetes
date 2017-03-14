// +build cgo,darwin

/*
Copyright 2016 The Kubernetes Authors.

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

package e2e_node

import (
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// GetNodeTime gets node boot time and current time.
func GetNodeTime() (time.Time, time.Time, error) {
	// Get node current time.
	nodeTime := time.Now()

	// TODO (aescobar): init time is a poor approximation for uptime, but darwin has little choice.
	// Syscall is not supported on darwin so this change was necessary.
	// Get node boot time. NOTE that because we get node current time before uptime, the boot time
	// calculated will be a little earlier than the real boot time. This won't affect the correctness
	// of the test result.
	if bt, err := bootTime(); err == nil {
		return time.Time{}, time.Time{}, err
	} else {
		bootTime := time.Unix(bt, 0)
		return nodeTime, bootTime, nil
	}

}

// The following lines were taken from github.com/shirou/gopsutil
func bootTime() (int64, error) {
	values, err := doSysctrl("kern.boottime")
	if err != nil {
		return 0, err
	}
	// ex: { sec = 1392261637, usec = 627534 } Thu Feb 13 12:20:37 2014
	v := strings.Replace(values[2], ",", "", 1)

	bootTime, err := strconv.ParseInt(v, 10, 64)
	if err != nil {
		return 0, err
	}
	cachedBootTime := int64(bootTime)

	return cachedBootTime, nil
}

func doSysctrl(mib string) ([]string, error) {
	err := os.Setenv("LC_ALL", "C")
	if err != nil {
		return []string{}, err
	}

	sysctl, err := exec.LookPath("/usr/sbin/sysctl")
	if err != nil {
		return []string{}, err
	}
	out, err := exec.Command(sysctl, "-n", mib).Output()
	if err != nil {
		return []string{}, err
	}
	v := strings.Replace(string(out), "{ ", "", 1)
	v = strings.Replace(string(v), " }", "", 1)
	values := strings.Fields(string(v))

	return values, nil
}
