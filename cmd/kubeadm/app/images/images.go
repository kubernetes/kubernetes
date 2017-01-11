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

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

const (
	KubeEtcdImage = "etcd"

	KubeAPIServerImage         = "apiserver"
	KubeControllerManagerImage = "controller-manager"
	KubeSchedulerImage         = "scheduler"
	KubeProxyImage             = "proxy"

	KubeDNSImage        = "k8s-dns-kube-dns"
	KubeDNSmasqImage    = "k8s-dns-dnsmasq"
	KubeDNSSidecarImage = "k8s-dns-sidecar"
	Pause               = "pause"

	gcrPrefix   = "gcr.io/google_containers"
	etcdVersion = "3.0.14-kubeadm"

	kubeDNSVersion        = "1.10.1"
	dnsmasqVersion        = "1.10.1"
	kubeDNSSidecarVersion = "1.10.1"
	pauseVersion          = "3.0"
)

func GetCoreImage(image string, cfg *kubeadmapi.MasterConfiguration, overrideImage string) string {
	if overrideImage != "" {
		return overrideImage
	}
	repoPrefix := kubeadmapi.GlobalEnvParams.RepositoryPrefix
	return map[string]string{
		KubeEtcdImage:              fmt.Sprintf("%s/%s-%s:%s", repoPrefix, "etcd", runtime.GOARCH, etcdVersion),
		KubeAPIServerImage:         fmt.Sprintf("%s/%s-%s:%s", repoPrefix, "kube-apiserver", runtime.GOARCH, cfg.KubernetesVersion),
		KubeControllerManagerImage: fmt.Sprintf("%s/%s-%s:%s", repoPrefix, "kube-controller-manager", runtime.GOARCH, cfg.KubernetesVersion),
		KubeSchedulerImage:         fmt.Sprintf("%s/%s-%s:%s", repoPrefix, "kube-scheduler", runtime.GOARCH, cfg.KubernetesVersion),
		KubeProxyImage:             fmt.Sprintf("%s/%s-%s:%s", repoPrefix, "kube-proxy", runtime.GOARCH, cfg.KubernetesVersion),
	}[image]
}

func GetAddonImage(image string) string {
	repoPrefix := kubeadmapi.GlobalEnvParams.RepositoryPrefix
	return map[string]string{
		KubeDNSImage:        fmt.Sprintf("%s/%s-%s:%s", repoPrefix, KubeDNSImage, runtime.GOARCH, kubeDNSVersion),
		KubeDNSmasqImage:    fmt.Sprintf("%s/%s-%s:%s", repoPrefix, KubeDNSmasqImage, runtime.GOARCH, dnsmasqVersion),
		KubeDNSSidecarImage: fmt.Sprintf("%s/%s-%s:%s", repoPrefix, KubeDNSSidecarImage, runtime.GOARCH, kubeDNSSidecarVersion),
		Pause:               fmt.Sprintf("%s/%s-%s:%s", repoPrefix, Pause, runtime.GOARCH, pauseVersion),
	}[image]
}
