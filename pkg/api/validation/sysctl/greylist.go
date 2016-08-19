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
	"strings"
)

type Namespace string

const (
	IpcNamespace     = Namespace("ipc")
	NetNamespace     = Namespace("net")
	UnknownNamespace = Namespace("")
)

var greylist = map[string]Namespace{
	"kernel.sem": IpcNamespace,
}

var greylistPrefixes = map[string]Namespace{
	"kernel.shm": IpcNamespace,
	"kernel.msg": IpcNamespace,
	"net.":       NetNamespace,
	"fs.mqueue.": IpcNamespace,
}

// NamespacedBy checks that a sysctl is greylisted because it is known
// to be namespaced by the Linux kernel. The namespace is returned or UnknownNamespace
// if it is not known to be namespaced. Note that being greylisted is required
// to be validated, but not sufficient: the kubelet has a node-level whitelist
// and the container runtime might have a stricter check and refuse to launch a pod.
func NamespacedBy(val string) Namespace {
	if ns, found := greylist[val]; found {
		return ns
	}
	for p, ns := range greylistPrefixes {
		if strings.HasPrefix(val, p) {
			return ns
		}
	}
	return ""
}
