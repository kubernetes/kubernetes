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
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/dns"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

// GetCoreImage generates and returns the image for the core Kubernetes components or returns overrideImage if specified
func GetCoreImage(image, repoPrefix, k8sVersion, overrideImage string) string {
	if overrideImage != "" {
		return overrideImage
	}
	kubernetesImageTag := kubeadmutil.KubernetesVersionToImageTag(k8sVersion)
	etcdImageTag := constants.DefaultEtcdVersion
	etcdImageVersion, err := constants.EtcdSupportedVersion(k8sVersion)
	if err == nil {
		etcdImageTag = etcdImageVersion.String()
	}
	return map[string]string{
		constants.Etcd:                  fmt.Sprintf("%s/%s-%s:%s", repoPrefix, "etcd", runtime.GOARCH, etcdImageTag),
		constants.KubeAPIServer:         fmt.Sprintf("%s/%s-%s:%s", repoPrefix, "kube-apiserver", runtime.GOARCH, kubernetesImageTag),
		constants.KubeControllerManager: fmt.Sprintf("%s/%s-%s:%s", repoPrefix, "kube-controller-manager", runtime.GOARCH, kubernetesImageTag),
		constants.KubeScheduler:         fmt.Sprintf("%s/%s-%s:%s", repoPrefix, "kube-scheduler", runtime.GOARCH, kubernetesImageTag),
	}[image]
}

// GetAllImages returns a list of container images kubeadm expects to use on a control plane node
func GetAllImages(cfg *kubeadmapi.MasterConfiguration) []string {
	imgs := []string{}
	imgs = append(imgs, GetCoreImage(constants.KubeAPIServer, cfg.ImageRepository, cfg.KubernetesVersion, cfg.UnifiedControlPlaneImage))
	imgs = append(imgs, GetCoreImage(constants.KubeControllerManager, cfg.ImageRepository, cfg.KubernetesVersion, cfg.UnifiedControlPlaneImage))
	imgs = append(imgs, GetCoreImage(constants.KubeScheduler, cfg.ImageRepository, cfg.KubernetesVersion, cfg.UnifiedControlPlaneImage))
	imgs = append(imgs, fmt.Sprintf("%v/%v-%v:%v", cfg.ImageRepository, constants.KubeProxy, runtime.GOARCH, kubeadmutil.KubernetesVersionToImageTag(cfg.KubernetesVersion)))
	imgs = append(imgs, fmt.Sprintf("%v/pause-%v:%v", cfg.ImageRepository, runtime.GOARCH, "3.1"))

	// if etcd is not external then add the image as it will be required
	if cfg.Etcd.Local != nil {
		imgs = append(imgs, GetCoreImage(constants.Etcd, cfg.ImageRepository, cfg.KubernetesVersion, cfg.Etcd.Local.Image))
	}

	dnsImage := fmt.Sprintf("%v/k8s-dns-kube-dns-%v:%v", cfg.ImageRepository, runtime.GOARCH, dns.GetDNSVersion(nil, constants.KubeDNS))
	if features.Enabled(cfg.FeatureGates, features.CoreDNS) {
		dnsImage = fmt.Sprintf("coredns/coredns:%v", dns.GetDNSVersion(nil, constants.CoreDNS))
	}
	imgs = append(imgs, dnsImage)
	return imgs
}
