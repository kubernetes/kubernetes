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

package sysctl

import (
	"strings"
)

// Namespace represents a kernel namespace name.
type Namespace string

const (
	// the Linux IPC namespace
	ipcNamespace = Namespace("ipc")

	// the network namespace
	netNamespace = Namespace("net")

	// the zero value if no namespace is known
	unknownNamespace = Namespace("")
)

var namespaces = map[string]Namespace{
	"kernel.sem": ipcNamespace,
}

var prefixNamespaces = map[string]Namespace{
	"kernel.shm": ipcNamespace,
	"kernel.msg": ipcNamespace,
	"fs.mqueue.": ipcNamespace,
	"net.":       netNamespace,
}

// NamespacedBy returns the namespace of the Linux kernel for a sysctl, or
// unknownNamespace if the sysctl is not known to be namespaced.
func NamespacedBy(val string) Namespace {
	if ns, found := namespaces[val]; found {
		return ns
	}
	for p, ns := range prefixNamespaces {
		if strings.HasPrefix(val, p) {
			return ns
		}
	}
	return unknownNamespace
}
