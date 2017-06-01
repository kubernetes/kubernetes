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
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/credentialprovider"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/util/parsers"
)

// PullImage pulls an image from the network to local storage using the supplied
// secrets if necessary.
func (m *kubeGenericRuntimeManager) PullImage(image kubecontainer.ImageSpec, pullSecrets []v1.Secret) (string, error) {
	img := image.Image
	repoToPull, _, _, err := parsers.ParseImageName(img)
	if err != nil {
		return "", err
	}

	keyring, err := credentialprovider.MakeDockerKeyring(pullSecrets, m.keyring)
	if err != nil {
		return "", err
	}

	imgSpec := &runtimeapi.ImageSpec{Image: img}
	creds, withCredentials := keyring.Lookup(repoToPull)
	if !withCredentials {
		glog.V(3).Infof("Pulling image %q without credentials", img)

		imageRef, err := m.imageService.PullImage(imgSpec, nil)
		if err != nil {
			glog.Errorf("Pull image %q failed: %v", img, err)
			return "", err
		}

		return imageRef, nil
	}

	var pullErrs []error
	for _, currentCreds := range creds {
		authConfig := credentialprovider.LazyProvide(currentCreds)
		auth := &runtimeapi.AuthConfig{
			Username:      authConfig.Username,
			Password:      authConfig.Password,
			Auth:          authConfig.Auth,
			ServerAddress: authConfig.ServerAddress,
			IdentityToken: authConfig.IdentityToken,
			RegistryToken: authConfig.RegistryToken,
		}

		imageRef, err := m.imageService.PullImage(imgSpec, auth)
		// If there was no error, return success
		if err == nil {
			return imageRef, nil
		}

		pullErrs = append(pullErrs, err)
	}

	return "", utilerrors.NewAggregate(pullErrs)
}

// GetImageRef gets the reference (digest or ID) of the image which has already been in
// the local storage. It returns ("", nil) if the image isn't in the local storage.
func (m *kubeGenericRuntimeManager) GetImageRef(image kubecontainer.ImageSpec) (string, error) {
	status, err := m.imageService.ImageStatus(&runtimeapi.ImageSpec{Image: image.Image})
	if err != nil {
		glog.Errorf("ImageStatus for image %q failed: %v", image, err)
		return "", err
	}
	if status == nil {
		return "", nil
	}

	imageRef := status.Id
	if len(status.RepoDigests) > 0 {
		imageRef = status.RepoDigests[0]
	}
	return imageRef, nil
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
			ID:          img.Id,
			Size:        int64(img.Size_),
			RepoTags:    img.RepoTags,
			RepoDigests: img.RepoDigests,
		})
	}

	return images, nil
}

// RemoveImage removes the specified image.
func (m *kubeGenericRuntimeManager) RemoveImage(image kubecontainer.ImageSpec) error {
	err := m.imageService.RemoveImage(&runtimeapi.ImageSpec{Image: image.Image})
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
		stats.TotalStorageBytes += img.Size_
	}
	return stats, nil
}
