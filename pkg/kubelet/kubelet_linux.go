//go:build linux
// +build linux

/*
Copyright 2022 The Kubernetes Authors.

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

package kubelet

import (
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/security/apparmor"
)

// The path in containers' filesystems where the hosts file is mounted.
var linuxEtcHostsPath = "/etc/hosts"

func getContainerEtcHostsPath() string {
	return linuxEtcHostsPath
}

// AppArmor is a Linux kernel security module and it does not support other operating systems.
func (kl *Kubelet) setAppArmorIfExists() {
	kl.appArmorValidator = apparmor.NewValidator()
	kl.softAdmitHandlers.AddPodAdmitHandler(lifecycle.NewAppArmorAdmitHandler(kl.appArmorValidator))
}
