// +build windows

/*
Copyright 2020 The Kubernetes Authors.

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
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1beta2 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta2"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

// GetPauseImage returns the image for the "pause" container
func GetPauseImage(cfg *kubeadmapi.ClusterConfiguration) string {
	pauseImageRepo := cfg.ImageRepository
	pauseImageTag := constants.PauseVersion
	if cfg.PauseImage != nil {
		if cfg.PauseImage.ImageRepository != "" {
			pauseImageRepo = cfg.PauseImage.ImageRepository
		}
		if cfg.PauseImage.ImageTag != "" {
			pauseImageTag = cfg.PauseImage.ImageTag
		}
	}
	if cfg.ImageRepository == kubeadmapiv1beta2.DefaultImageRepository {
		pauseImageRepo = "mcr.microsoft.com/oss/kubernetes"
	}
	//If user has configured the cluster to use a different image repository, use that for the Windows pause image.
	if cfg.UseArchImage {
		return GetGenericArchImage(pauseImageRepo, "pause", pauseImageTag)
	}
	return GetGenericImage(pauseImageRepo, "pause", pauseImageTag)
}
