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
	KubeEtcdImage              = "etcd"
	KubeAPIServerImage         = "apiserver"
	KubeControllerManagerImage = "controller-manager"
	KubeSchedulerImage         = "scheduler"
	KubeProxyImage             = "proxy"
	EtcdOperatorImage          = "etcd-operator"
	PodCheckpointerImage       = "pod-checkpointer"
	NetworkCheckpointerImage   = "network-checkpointer"

	etcdOperatorVersion    = "v0.3.3"
	podCheckpointerVersion = "4e7a7dab10bc4d895b66c21656291c6e0b017248"
	kencVersion            = "8f6e2e885f790030fbbb0496ea2a2d8830e58b8f"
)

func GetCoreImage(image string, cfg *kubeadmapi.MasterConfiguration, overrideImage string) string {
	if overrideImage != "" {
		return overrideImage
	}
	repoPrefix := kubeadmapi.GlobalEnvParams.RepositoryPrefix
	return map[string]string{
		KubeAPIServerImage:         fmt.Sprintf("%s/%s-%s:%s", repoPrefix, "kube-apiserver", runtime.GOARCH, cfg.KubernetesVersion),
		KubeControllerManagerImage: fmt.Sprintf("%s/%s-%s:%s", repoPrefix, "kube-controller-manager", runtime.GOARCH, cfg.KubernetesVersion),
		KubeSchedulerImage:         fmt.Sprintf("%s/%s-%s:%s", repoPrefix, "kube-scheduler", runtime.GOARCH, cfg.KubernetesVersion),
		KubeProxyImage:             fmt.Sprintf("%s/%s-%s:%s", repoPrefix, "kube-proxy", runtime.GOARCH, cfg.KubernetesVersion),
		KubeEtcdImage:              fmt.Sprintf("quay.io/coreos/etcd:%s", cfg.Etcd.SelfHosted.Version),
		EtcdOperatorImage:          fmt.Sprintf("quay.io/coreos/etcd-operator:%s", etcdOperatorVersion),
		PodCheckpointerImage:       fmt.Sprintf("quay.io/coreos/pod-checkpointer:%s", podCheckpointerVersion),
		NetworkCheckpointerImage:   fmt.Sprintf("quay.io/coreos/kenc:%s", kencVersion),
	}[image]
}
