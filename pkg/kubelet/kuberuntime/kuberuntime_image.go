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

package kuberuntime

import (
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/credentialprovider"
	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/parsers"
)

// PullImage pulls an image from the network to local storage using the supplied
// secrets if necessary.
func (m *kubeGenericRuntimeManager) PullImage(image kubecontainer.ImageSpec, pullSecrets []api.Secret) error {
	img := image.Image
	repoToPull, _, _, err := parsers.ParseImageName(img)
	if err != nil {
		return err
	}

	keyring, err := credentialprovider.MakeDockerKeyring(pullSecrets, m.keyring)
	if err != nil {
		return err
	}

	imgSpec := &runtimeApi.ImageSpec{Image: &img}
	creds, withCredentials := keyring.Lookup(repoToPull)
	if !withCredentials {
		glog.V(3).Infof("Pulling image %q without credentials", img)

		err = m.imageService.PullImage(imgSpec, nil)
		if err != nil {
			glog.Errorf("Pull image %q failed: %v", img, err)
			return err
		}

		return nil
	}

	var pullErrs []error
	for _, currentCreds := range creds {
		authConfig := credentialprovider.LazyProvide(currentCreds)
		auth := &runtimeApi.AuthConfig{
			Username:      &authConfig.Username,
			Password:      &authConfig.Password,
			Auth:          &authConfig.Auth,
			ServerAddress: &authConfig.ServerAddress,
			IdentityToken: &authConfig.IdentityToken,
			RegistryToken: &authConfig.RegistryToken,
		}

		err = m.imageService.PullImage(imgSpec, auth)
		// If there was no error, return success
		if err == nil {
			return nil
		}

		pullErrs = append(pullErrs, err)
	}

	return utilerrors.NewAggregate(pullErrs)
}

// IsImagePresent checks whether the container image is already in the local storage.
func (m *kubeGenericRuntimeManager) IsImagePresent(image kubecontainer.ImageSpec) (bool, error) {
	status, err := m.imageService.ImageStatus(&runtimeApi.ImageSpec{Image: &image.Image})
	if err != nil {
		glog.Errorf("ImageStatus for image %q failed: %v", image, err)
		return false, err
	}
	return status != nil, nil
}

// ListImages gets all images currently on the machine.
func (m *kubeGenericRuntimeManager) ListImages() ([]kubecontainer.Image, error) {
	var images []kubecontainer.Image

	allImages, err := m.imageService.ListImages(nil)
	if err != nil {
		glog.Errorf("ListImages failed: %v", err)
		return nil, err
	}

	for _, img := range allImages {
		images = append(images, kubecontainer.Image{
			ID:          img.GetId(),
			Size:        int64(img.GetSize_()),
			RepoTags:    img.RepoTags,
			RepoDigests: img.RepoDigests,
		})
	}

	return images, nil
}

// RemoveImage removes the specified image.
func (m *kubeGenericRuntimeManager) RemoveImage(image kubecontainer.ImageSpec) error {
	err := m.imageService.RemoveImage(&runtimeApi.ImageSpec{Image: &image.Image})
	if err != nil {
		glog.Errorf("Remove image %q failed: %v", image.Image, err)
		return err
	}

	return nil
}

// ImageStats returns the statistics of the image.
// Notice that current logic doesn't really work for images which share layers (e.g. docker image),
// this is a known issue, and we'll address this by getting imagefs stats directly from CRI.
// TODO: Get imagefs stats directly from CRI.
func (m *kubeGenericRuntimeManager) ImageStats() (*kubecontainer.ImageStats, error) {
	allImages, err := m.imageService.ListImages(nil)
	if err != nil {
		glog.Errorf("ListImages failed: %v", err)
		return nil, err
	}
	stats := &kubecontainer.ImageStats{}
	for _, img := range allImages {
		stats.TotalStorageBytes += img.GetSize_()
	}
	return stats, nil
}
