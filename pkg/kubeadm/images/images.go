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

package images

import (
	"fmt"
	"runtime"
)

const (
	KubeEtcdImage = "etcd"

	KubeApiServerImage         = "apiserver"
	KubeControllerManagerImage = "controller-manager"
	KubeSchedulerImage         = "scheduler"
	KubeProxyImage             = "proxy"

	KubeDnsImage         = "kube-dns"
	KubeDnsmasqImage     = "dnsmasq"
	KubeExechealthzImage = "exechealthz"

	gcrPrefix   = "gcr.io/google_containers"
	etcdVersion = "2.2.5"

	kubeDnsVersion     = "1.7"
	dnsmasqVersion     = "1.3"
	exechealthzVersion = "1.1"
)

// TODO(phase1): Make this configurable + default to a v1.4 value fetched from: https://storage.googleapis.com/kubernetes-release/release/stable.txt
var DefaultKubeVersion = "v1.4.0-beta.6"

func GetCoreImage(image string, overrideImage string) string {
	if overrideImage != "" {
		return overrideImage
	}

	return map[string]string{
		KubeEtcdImage:              fmt.Sprintf("%s/%s-%s:%s", gcrPrefix, "etcd", runtime.GOARCH, etcdVersion),
		KubeApiServerImage:         fmt.Sprintf("%s/%s-%s:%s", gcrPrefix, "kube-apiserver", runtime.GOARCH, DefaultKubeVersion),
		KubeControllerManagerImage: fmt.Sprintf("%s/%s-%s:%s", gcrPrefix, "kube-controller-manager", runtime.GOARCH, DefaultKubeVersion),
		KubeSchedulerImage:         fmt.Sprintf("%s/%s-%s:%s", gcrPrefix, "kube-scheduler", runtime.GOARCH, DefaultKubeVersion),
		KubeProxyImage:             fmt.Sprintf("%s/%s-%s:%s", gcrPrefix, "kube-proxy", runtime.GOARCH, DefaultKubeVersion),
	}[image]
}

func GetAddonImage(image string) string {
	return map[string]string{
		KubeDnsImage:         fmt.Sprintf("%s/%s-%s:%s", gcrPrefix, "kubedns", runtime.GOARCH, kubeDnsVersion),
		KubeDnsmasqImage:     fmt.Sprintf("%s/%s-%s:%s", gcrPrefix, "kube-dnsmasq", runtime.GOARCH, dnsmasqVersion),
		KubeExechealthzImage: fmt.Sprintf("%s/%s-%s:%s", gcrPrefix, "exechealthz", runtime.GOARCH, exechealthzVersion),
	}[image]
}
