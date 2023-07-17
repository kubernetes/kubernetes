/*
Copyright 2023 The Kubernetes Authors.

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
	"strings"
)

// Namespace represents a kernel namespace name.
type Namespace string

const (
	// refer to https://man7.org/linux/man-pages/man7/ipc_namespaces.7.html
	// the Linux IPC namespace
	IPCNamespace = Namespace("IPC")

	// refer to https://man7.org/linux/man-pages/man7/network_namespaces.7.html
	// the network namespace
	NetNamespace = Namespace("Net")

	// the zero value if no namespace is known
	UnknownNamespace = Namespace("")
)

var nameToNamespace = map[string]Namespace{
	// kernel semaphore parameters: SEMMSL, SEMMNS, SEMOPM, and SEMMNI.
	"kernel.sem": IPCNamespace,
	// kernel shared memory limits include shmall, shmmax, shmmni, and shm_rmid_forced.
	"kernel.shmall":          IPCNamespace,
	"kernel.shmmax":          IPCNamespace,
	"kernel.shmmni":          IPCNamespace,
	"kernel.shm_rmid_forced": IPCNamespace,
	// make backward compatibility  to know the namespace of kernel.shm*
	"kernel.shm": IPCNamespace,
	// kernel messages include msgmni, msgmax and msgmnb.
	"kernel.msgmax": IPCNamespace,
	"kernel.msgmnb": IPCNamespace,
	"kernel.msgmni": IPCNamespace,
	// make backward compatibility to know the namespace of kernel.msg*
	"kernel.msg": IPCNamespace,
}

var prefixToNamespace = map[string]Namespace{
	"net": NetNamespace,
	// mqueue filesystem provides the necessary kernel features to enable the creation
	// of a user space library that implements the POSIX message queues API.
	"fs.mqueue": IPCNamespace,
}

// namespaceOf returns the namespace of the Linux kernel for a sysctl, or
// unknownNamespace if the sysctl is not known to be namespaced.
// The second return is prefixed bool.
// It returns true if the key is prefixed with a key in the prefix map
func namespaceOf(val string) Namespace {
	if ns, found := nameToNamespace[val]; found {
		return ns
	}
	for p, ns := range prefixToNamespace {
		if strings.HasPrefix(val, p+".") {
			return ns
		}
	}
	return UnknownNamespace
}

// GetNamespace extracts information from a sysctl string. It returns:
//  1. The sysctl namespace, which can be one of the following: IPC, Net, or unknown.
//  2. sysctlOrPrefix: the prefix of the sysctl parameter until the first '*'.
//     If there is no '*', it will be the original string.
//  3. 'prefixed' is set to true if the sysctl parameter contains '*' or it is in the prefixToNamespace key list, in most cases, it is a suffix *.
//
// For example, if the input sysctl is 'net.ipv6.neigh.*', GetNamespace will return:
// - The Net namespace
// - The sysctlOrPrefix as 'net.ipv6.neigh'
// - 'prefixed' set to true
//
// For the input sysctl 'net.ipv6.conf.all.disable_ipv6', GetNamespace will return:
// - The Net namespace
// - The sysctlOrPrefix as 'net.ipv6.conf.all.disable_ipv6'
// - 'prefixed' set to false.
func GetNamespace(sysctl string) (ns Namespace, sysctlOrPrefix string, prefixed bool) {
	sysctlOrPrefix = NormalizeName(sysctl)
	firstIndex := strings.IndexAny(sysctlOrPrefix, "*")
	if firstIndex != -1 {
		sysctlOrPrefix = sysctlOrPrefix[:firstIndex]
		prefixed = true
	}
	ns = namespaceOf(sysctlOrPrefix)
	return
}
