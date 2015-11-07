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

package sysctl

import (
	"io/ioutil"
	"path"
	"strconv"
	"strings"
)

const (
	sysctlBase         = "/proc/sys"
	VmOvercommitMemory = "vm/overcommit_memory"
	VmPanicOnOOM       = "vm/panic_on_oom"

	VmOvercommitMemoryAlways    = 1 // kernel performs no memory over-commit handling
	VmPanicOnOOMInvokeOOMKiller = 0 // kernel calls the oom_killer function when OOM occurs
)

// GetSysctl returns the value for the specified sysctl setting
func GetSysctl(sysctl string) (int, error) {
	data, err := ioutil.ReadFile(path.Join(sysctlBase, sysctl))
	if err != nil {
		return -1, err
	}
	val, err := strconv.Atoi(strings.Trim(string(data), " \n"))
	if err != nil {
		return -1, err
	}
	return val, nil
}

// SetSysctl modifies the specified sysctl flag to the new value
func SetSysctl(sysctl string, newVal int) error {
	return ioutil.WriteFile(path.Join(sysctlBase, sysctl), []byte(strconv.Itoa(newVal)), 0640)
}
