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
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

// GetGenericImage generates and returns a platform agnostic image (backed by manifest list)
func GetGenericImage(prefix, image, tag string) string {
	return fmt.Sprintf("%s/%s:%s", prefix, image, tag)
}

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
	repoPrefix := cfg.GetControlPlaneImageRepository()
	imgs := []string{}
	imgs = append(imgs, GetCoreImage(constants.KubeAPIServer, repoPrefix, cfg.KubernetesVersion, cfg.UnifiedControlPlaneImage))
	imgs = append(imgs, GetCoreImage(constants.KubeControllerManager, repoPrefix, cfg.KubernetesVersion, cfg.UnifiedControlPlaneImage))
	imgs = append(imgs, GetCoreImage(constants.KubeScheduler, repoPrefix, cfg.KubernetesVersion, cfg.UnifiedControlPlaneImage))
	imgs = append(imgs, fmt.Sprintf("%v/%v-%v:%v", repoPrefix, constants.KubeProxy, runtime.GOARCH, kubeadmutil.KubernetesVersionToImageTag(cfg.KubernetesVersion)))

	// pause, etcd and kube-dns are not available on the ci image repository so use the default image repository.
	imgs = append(imgs, GetGenericImage(cfg.ImageRepository, "pause", "3.1"))

	// if etcd is not external then add the image as it will be required
	if cfg.Etcd.Local != nil {
		imgs = append(imgs, GetCoreImage(constants.Etcd, cfg.ImageRepository, cfg.KubernetesVersion, cfg.Etcd.Local.Image))
	}

	dnsImage := fmt.Sprintf("%v/k8s-dns-kube-dns-%v:%v", cfg.ImageRepository, runtime.GOARCH, constants.KubeDNSVersion)
	if features.Enabled(cfg.FeatureGates, features.CoreDNS) {
		dnsImage = fmt.Sprintf("%v/coredns:%v", cfg.ImageRepository, constants.CoreDNSVersion)
	}
	imgs = append(imgs, dnsImage)
	return imgs
}
