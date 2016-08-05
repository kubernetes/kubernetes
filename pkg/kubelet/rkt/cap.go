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

package rkt

// TODO(yifan): Export this to higher level package.
const (
	CAP_CHOWN = iota
	CAP_DAC_OVERRIDE
	CAP_DAC_READ_SEARCH
	CAP_FOWNER
	CAP_FSETID
	CAP_KILL
	CAP_SETGID
	CAP_SETUID
	CAP_SETPCAP
	CAP_LINUX_IMMUTABLE
	CAP_NET_BIND_SERVICE
	CAP_NET_BROADCAST
	CAP_NET_ADMIN
	CAP_NET_RAW
	CAP_IPC_LOCK
	CAP_IPC_OWNER
	CAP_SYS_MODULE
	CAP_SYS_RAWIO
	CAP_SYS_CHROOT
	CAP_SYS_PTRACE
	CAP_SYS_PACCT
	CAP_SYS_ADMIN
	CAP_SYS_BOOT
	CAP_SYS_NICE
	CAP_SYS_RESOURCE
	CAP_SYS_TIME
	CAP_SYS_TTY_CONFIG
	CAP_MKNOD
	CAP_LEASE
	CAP_AUDIT_WRITE
	CAP_AUDIT_CONTROL
	CAP_SETFCAP
	CAP_MAC_OVERRIDE
	CAP_MAC_ADMIN
	CAP_SYSLOG
	CAP_WAKE_ALARM
	CAP_BLOCK_SUSPEND
	CAP_AUDIT_READ
)

// TODO(yifan): Export this to higher level package.
var capabilityList = map[int]string{
	CAP_CHOWN:            "CAP_CHOWN",
	CAP_DAC_OVERRIDE:     "CAP_DAC_OVERRIDE",
	CAP_DAC_READ_SEARCH:  "CAP_DAC_READ_SEARCH",
	CAP_FOWNER:           "CAP_FOWNER",
	CAP_FSETID:           "CAP_FSETID",
	CAP_KILL:             "CAP_KILL",
	CAP_SETGID:           "CAP_SETGID",
	CAP_SETUID:           "CAP_SETUID",
	CAP_SETPCAP:          "CAP_SETPCAP",
	CAP_LINUX_IMMUTABLE:  "CAP_LINUX_IMMUTABLE",
	CAP_NET_BIND_SERVICE: "CAP_NET_BIND_SERVICE",
	CAP_NET_BROADCAST:    "CAP_NET_BROADCAST",
	CAP_NET_ADMIN:        "CAP_NET_ADMIN",
	CAP_NET_RAW:          "CAP_NET_RAW",
	CAP_IPC_LOCK:         "CAP_IPC_LOCK",
	CAP_IPC_OWNER:        "CAP_IPC_OWNER",
	CAP_SYS_MODULE:       "CAP_SYS_MODULE",
	CAP_SYS_RAWIO:        "CAP_SYS_RAWIO",
	CAP_SYS_CHROOT:       "CAP_SYS_CHROOT",
	CAP_SYS_PTRACE:       "CAP_SYS_PTRACE",
	CAP_SYS_PACCT:        "CAP_SYS_PACCT",
	CAP_SYS_ADMIN:        "CAP_SYS_ADMIN",
	CAP_SYS_BOOT:         "CAP_SYS_BOOT",
	CAP_SYS_NICE:         "CAP_SYS_NICE",
	CAP_SYS_RESOURCE:     "CAP_SYS_RESOURCE",
	CAP_SYS_TIME:         "CAP_SYS_TIME",
	CAP_SYS_TTY_CONFIG:   "CAP_SYS_TTY_CONFIG",
	CAP_MKNOD:            "CAP_MKNOD",
	CAP_LEASE:            "CAP_LEASE",
	CAP_AUDIT_WRITE:      "CAP_AUDIT_WRITE",
	CAP_AUDIT_CONTROL:    "CAP_AUDIT_CONTROL",
	CAP_SETFCAP:          "CAP_SETFCAP",
	CAP_MAC_OVERRIDE:     "CAP_MAC_OVERRIDE",
	CAP_MAC_ADMIN:        "CAP_MAC_ADMIN",
	CAP_SYSLOG:           "CAP_SYSLOG",
	CAP_WAKE_ALARM:       "CAP_WAKE_ALARM",
	CAP_BLOCK_SUSPEND:    "CAP_BLOCK_SUSPEND",
	CAP_AUDIT_READ:       "CAP_AUDIT_READ",
}

// allCapabilities returns the capability list with all capabilities.
func allCapabilities() []string {
	var capabilities []string
	for _, cap := range capabilityList {
		capabilities = append(capabilities, cap)
	}
	return capabilities
}
