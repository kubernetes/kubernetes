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

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

// GetGenericImage generates and returns a platform agnostic image (backed by manifest list)
func GetGenericImage(prefix, image, tag string) string {
	return fmt.Sprintf("%s/%s:%s", prefix, image, tag)
}

// GetKubernetesImage generates and returns the image for the components managed in the Kubernetes main repository,
// including the control-plane components and kube-proxy. If specified, the HyperKube image will be used.
func GetKubernetesImage(image string, cfg *kubeadmapi.ClusterConfiguration) string {
	if cfg.UseHyperKubeImage {
		image = constants.HyperKube
	}
	repoPrefix := cfg.GetControlPlaneImageRepository()
	kubernetesImageTag := kubeadmutil.KubernetesVersionToImageTag(cfg.KubernetesVersion)
	return GetGenericImage(repoPrefix, image, kubernetesImageTag)
}

// GetAddOnImage generates and returns an image for an addon
func GetAddOnImage(cfg *kubeadmapi.ClusterConfiguration, addon kubeadmapi.AddOn, image, tag string) string {
	// use the default image repository by default
	repository := cfg.ImageRepository

	// unless an override is specified
	if addon.ImageRepository != "" {
		repository = addon.ImageRepository
	}

	// the default tag is overwritten if an addon tag is specified
	if addon.ImageTag != "" {
		tag = addon.ImageTag
	}

	return GetGenericImage(repository, image, tag)
}

// GetEtcdImage generates and returns the image for etcd
func GetEtcdImage(cfg *kubeadmapi.ClusterConfiguration) string {
	// Etcd uses default image repository by default
	etcdImageRepository := cfg.ImageRepository
	// unless an override is specified
	if cfg.Etcd.Local != nil && cfg.Etcd.Local.ImageRepository != "" {
		etcdImageRepository = cfg.Etcd.Local.ImageRepository
	}
	// Etcd uses an imageTag that corresponds to the etcd version matching the Kubernetes version
	etcdImageTag := constants.DefaultEtcdVersion
	etcdVersion, err := constants.EtcdSupportedVersion(cfg.KubernetesVersion)
	if err == nil {
		etcdImageTag = etcdVersion.String()
	}
	// unless an override is specified
	if cfg.Etcd.Local != nil && cfg.Etcd.Local.ImageTag != "" {
		etcdImageTag = cfg.Etcd.Local.ImageTag
	}
	return GetGenericImage(etcdImageRepository, constants.Etcd, etcdImageTag)
}

// GetPauseImage returns the image for the "pause" container
func GetPauseImage(cfg *kubeadmapi.ClusterConfiguration) string {
	return GetGenericImage(cfg.ImageRepository, "pause", constants.PauseVersion)
}

// GetControlPlaneImages returns a list of container images kubeadm expects to use on a control plane node
func GetControlPlaneImages(cfg *kubeadmapi.ClusterConfiguration) []string {
	imgs := []string{}

	// start with core kubernetes images
	if cfg.UseHyperKubeImage {
		imgs = append(imgs, GetKubernetesImage(constants.HyperKube, cfg))
	} else {
		imgs = append(imgs, GetKubernetesImage(constants.KubeAPIServer, cfg))
		imgs = append(imgs, GetKubernetesImage(constants.KubeControllerManager, cfg))
		imgs = append(imgs, GetKubernetesImage(constants.KubeScheduler, cfg))
		if _, ok := cfg.AddOns[constants.KubeProxy]; ok {
			imgs = append(imgs, GetKubernetesImage(constants.KubeProxy, cfg))
		}
	}

	// pause is not available on the ci image repository so use the default image repository.
	imgs = append(imgs, GetPauseImage(cfg))

	// if etcd is not external then add the image as it will be required
	if cfg.Etcd.Local != nil {
		imgs = append(imgs, GetEtcdImage(cfg))
	}

	// CoreDNS images (if used)
	if addon, ok := cfg.AddOns[constants.CoreDNS]; ok {
		imgs = append(imgs, GetAddOnImage(cfg, addon, constants.CoreDNSImageName, constants.CoreDNSVersion))
	}

	// kube-dns images (if used)
	if addon, ok := cfg.AddOns[constants.KubeDNS]; ok {
		imgs = append(imgs, GetAddOnImage(cfg, addon, constants.KubeDNSKubeDNSImageName, constants.KubeDNSVersion))
		imgs = append(imgs, GetAddOnImage(cfg, addon, constants.KubeDNSSidecarImageName, constants.KubeDNSVersion))
		imgs = append(imgs, GetAddOnImage(cfg, addon, constants.KubeDNSDnsMasqNannyImageName, constants.KubeDNSVersion))
	}

	return imgs
}
