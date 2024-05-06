/*
Copyright 2015 The Kubernetes Authors.

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
	"os"
	"path"
	"strconv"
	"strings"
)

const (
	sysctlBase = "/proc/sys"
	// VMOvercommitMemory refers to the sysctl variable responsible for defining
	// the memory over-commit policy used by kernel.
	VMOvercommitMemory = "vm/overcommit_memory"
	// VMPanicOnOOM refers to the sysctl variable responsible for defining
	// the OOM behavior used by kernel.
	VMPanicOnOOM = "vm/panic_on_oom"
	// KernelPanic refers to the sysctl variable responsible for defining
	// the timeout after a panic for the kernel to reboot.
	KernelPanic = "kernel/panic"
	// KernelPanicOnOops refers to the sysctl variable responsible for defining
	// the kernel behavior when an oops or BUG is encountered.
	KernelPanicOnOops = "kernel/panic_on_oops"
	// RootMaxKeys refers to the sysctl variable responsible for defining
	// the maximum number of keys that the root user (UID 0 in the root user namespace) may own.
	RootMaxKeys = "kernel/keys/root_maxkeys"
	// RootMaxBytes refers to the sysctl variable responsible for defining
	// the maximum number of bytes of data that the root user (UID 0 in the root user namespace)
	// can hold in the payloads of the keys owned by root.
	RootMaxBytes = "kernel/keys/root_maxbytes"

	// VMOvercommitMemoryAlways represents that kernel performs no memory over-commit handling.
	VMOvercommitMemoryAlways = 1
	// VMPanicOnOOMInvokeOOMKiller represents that kernel calls the oom_killer function when OOM occurs.
	VMPanicOnOOMInvokeOOMKiller = 0

	// KernelPanicOnOopsAlways represents that kernel panics on kernel oops.
	KernelPanicOnOopsAlways = 1
	// KernelPanicRebootTimeout is the timeout seconds after a panic for the kernel to reboot.
	KernelPanicRebootTimeout = 10

	// RootMaxKeysSetting is the maximum number of keys that the root user (UID 0 in the root user namespace) may own.
	// Needed since docker creates a new key per container.
	RootMaxKeysSetting = 1000000
	// RootMaxBytesSetting is the maximum number of bytes of data that the root user (UID 0 in the root user namespace)
	// can hold in the payloads of the keys owned by root.
	// Allocate 25 bytes per key * number of MaxKeys.
	RootMaxBytesSetting = RootMaxKeysSetting * 25
)

// Interface is an injectable interface for running sysctl commands.
type Interface interface {
	// GetSysctl returns the value for the specified sysctl setting
	GetSysctl(sysctl string) (int, error)
	// SetSysctl modifies the specified sysctl flag to the new value
	SetSysctl(sysctl string, newVal int) error
}

// New returns a new Interface for accessing sysctl
func New() Interface {
	return &procSysctl{}
}

// procSysctl implements Interface by reading and writing files under /proc/sys
type procSysctl struct {
}

// GetSysctl returns the value for the specified sysctl setting
func (*procSysctl) GetSysctl(sysctl string) (int, error) {
	data, err := os.ReadFile(path.Join(sysctlBase, sysctl))
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
func (*procSysctl) SetSysctl(sysctl string, newVal int) error {
	return os.WriteFile(path.Join(sysctlBase, sysctl), []byte(strconv.Itoa(newVal)), 0640)
}

// NormalizeName can return sysctl variables in dots separator format.
// The '/' separator is also accepted in place of a '.'.
// Convert the sysctl variables to dots separator format for validation.
// More info:
//
//	https://man7.org/linux/man-pages/man8/sysctl.8.html
//	https://man7.org/linux/man-pages/man5/sysctl.d.5.html
func NormalizeName(val string) string {
	if val == "" {
		return val
	}
	firstSepIndex := strings.IndexAny(val, "./")
	// if the first found is `.` like `net.ipv4.conf.eno2/100.rp_filter`
	if firstSepIndex == -1 || val[firstSepIndex] == '.' {
		return val
	}

	// for `net/ipv4/conf/eno2.100/rp_filter`, swap the use of `.` and `/`
	// to `net.ipv4.conf.eno2/100.rp_filter`
	f := func(r rune) rune {
		switch r {
		case '.':
			return '/'
		case '/':
			return '.'
		}
		return r
	}
	return strings.Map(f, val)
}
