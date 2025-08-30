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
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/google/go-containerregistry/pkg/authn"
	"github.com/google/go-containerregistry/pkg/name"
	"github.com/google/go-containerregistry/pkg/v1/remote"

	v1 "k8s.io/api/core/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
	crededentialprovider "k8s.io/kubernetes/pkg/credentialprovider"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// PullImage pulls an image from the network to local storage using the supplied
// secrets if necessary.
func (m *kubeGenericRuntimeManager) PullImage(ctx context.Context, image kubecontainer.ImageSpec, credentials []crededentialprovider.TrackedAuthConfig, podSandboxConfig *runtimeapi.PodSandboxConfig) (string, *crededentialprovider.TrackedAuthConfig, error) {
	logger := klog.FromContext(ctx)
	img := image.Image
	imgSpec := toRuntimeAPIImageSpec(image)

	if len(credentials) == 0 {
		logger.V(3).Info("Pulling image without credentials", "image", img)

		imageRef, err := m.imageService.PullImage(ctx, imgSpec, nil, podSandboxConfig)
		if err != nil {
			logger.Error(err, "Failed to pull image", "image", img)
			return "", nil, err
		}

		return imageRef, nil, nil
	}

	var pullErrs []error
	for _, currentCreds := range credentials {
		auth := &runtimeapi.AuthConfig{
			Username:      currentCreds.Username,
			Password:      currentCreds.Password,
			Auth:          currentCreds.Auth,
			ServerAddress: currentCreds.ServerAddress,
			IdentityToken: currentCreds.IdentityToken,
			RegistryToken: currentCreds.RegistryToken,
		}

		imageRef, err := m.imageService.PullImage(ctx, imgSpec, auth, podSandboxConfig)
		// If there was no error, return success
		if err == nil {
			return imageRef, &currentCreds, nil
		}

		pullErrs = append(pullErrs, err)
	}

	return "", nil, utilerrors.NewAggregate(pullErrs)
}

// createKeychainFromSecrets creates an authn.Keychain from Kubernetes pull secrets.
// It falls back to authn.DefaultKeychain if no valid secrets are found.
func createKeychainFromSecrets(pullSecrets []v1.Secret) authn.Keychain {
	if len(pullSecrets) == 0 {
		klog.V(4).Info("No pull secrets provided, using default keychain")
		return authn.DefaultKeychain
	}

	return &secretKeychain{secrets: pullSecrets}
}

// secretKeychain implements authn.Keychain for Kubernetes secrets
type secretKeychain struct {
	secrets []v1.Secret
}

// Resolve returns the appropriate authenticator for the given registry.
func (k *secretKeychain) Resolve(target authn.Resource) (authn.Authenticator, error) {
	registry := target.RegistryStr()
	klog.V(4).Infof("Resolving auth for registry: %s", registry)

	for _, secret := range k.secrets {
		authConfig, err := getAuthenticatorFromSecret(&secret)
		if err != nil {
			klog.V(4).Infof("Failed to get auth from secret: %v", err)
			continue
		}
		if authConfig != nil {
			klog.V(4).Infof("Found credentials in secret for registry %s", registry)
			return authConfig, nil
		}
	}

	klog.V(4).Infof("No credentials found in secrets for registry %s, falling back to default keychain", registry)
	return authn.DefaultKeychain.Resolve(target)
}

// getAuthenticatorFromSecret extracts an authn.Authenticator from a Kubernetes pull secret
func getAuthenticatorFromSecret(secret *v1.Secret) (authn.Authenticator, error) {
	configData, ok := secret.Data[".dockerconfigjson"]
	if !ok {
		return nil, fmt.Errorf("no .dockerconfigjson in secret %s", secret.Name)
	}

	var dockerConfig struct {
		Auths map[string]struct {
			Auth string `json:"auth"`
		} `json:"auths"`
	}

	if err := json.Unmarshal(configData, &dockerConfig); err != nil {
		return nil, fmt.Errorf("failed to unmarshal docker config in secret %s: %w", secret.Name, err)
	}

	for _, authData := range dockerConfig.Auths {
		if authData.Auth == "" {
			continue
		}

		decoded, err := base64.StdEncoding.DecodeString(authData.Auth)
		if err != nil {
			continue
		}

		parts := strings.SplitN(string(decoded), ":", 2)
		if len(parts) != 2 {
			continue
		}

		return &authn.Basic{
			Username: parts[0],
			Password: parts[1],
		}, nil
	}

	return nil, nil
}

// GetRemoteImageRef gets the reference (digest or ID) of the image from the registry.
// It supports private registries via pull secrets and falls back to the original imageRef on error.
func (m *kubeGenericRuntimeManager) GetRemoteImageRef(
	ctx context.Context,
	imageRef string,
	pullSecrets []v1.Secret,
) (string, error) {
	logger := klog.FromContext(ctx)
	logger.V(3).Info("Fetching remote image ref", "imageRef", imageRef)

	// Parse the image reference
	ref, err := name.ParseReference(imageRef)
	if err != nil {
		logger.Error(err, "Failed to parse image reference", "imageRef", imageRef)
		return imageRef, err
	}

	// Create keychain from pull secrets
	keychain := createKeychainFromSecrets(pullSecrets)

	// Try fetching the remote image digest with retries (15 sec total)
	remoteImage, err := remote.Image(ref, remote.WithContext(ctx), remote.WithAuthFromKeychain(keychain))
	if err != nil {
		for i := 0; i < 5; i++ {
			remoteImage, err = remote.Image(ref, remote.WithContext(ctx), remote.WithAuthFromKeychain(keychain))
			if err == nil {
				break
			}
			logger.V(4).Info("Failed x%d to fetch remote image: %s, retrying after %d seconds", i+1, imageRef, i+1)
			time.Sleep(time.Second * time.Duration(i+1))
		}
	}

	if err != nil {
		logger.V(3).Info("Failed to fetch remote image digest, fallback to imageRef: %s, err: %v", imageRef, err)
		return imageRef, err
	}

	// Get the image digest
	digest, err := remoteImage.ConfigName()
	if err != nil {
		logger.Error(err, "Failed to get remote image digest, fallback to imageRef", "imageRef", imageRef)
		return imageRef, err
	}

	logger.V(4).Info("Successfully fetched remote digest: %s", digest.String())
	return digest.String(), nil
}

// GetImageRef gets the ID of the image which has already been in
// the local storage. It returns ("", nil) if the image isn't in the local storage.
func (m *kubeGenericRuntimeManager) GetImageRef(ctx context.Context, image kubecontainer.ImageSpec) (string, error) {
	logger := klog.FromContext(ctx)
	resp, err := m.imageService.ImageStatus(ctx, toRuntimeAPIImageSpec(image), false)
	if err != nil {
		logger.Error(err, "Failed to get image status", "image", image.Image)
		return "", err
	}
	if resp.Image == nil {
		return "", nil
	}
	return resp.Image.Id, nil
}

func (m *kubeGenericRuntimeManager) GetImageSize(ctx context.Context, image kubecontainer.ImageSpec) (uint64, error) {
	logger := klog.FromContext(ctx)
	resp, err := m.imageService.ImageStatus(ctx, toRuntimeAPIImageSpec(image), false)
	if err != nil {
		logger.Error(err, "Failed to get image status", "image", image.Image)
		return 0, err
	}
	if resp.Image == nil {
		return 0, nil
	}
	return resp.Image.Size, nil
}

// ListImages gets all images currently on the machine.
func (m *kubeGenericRuntimeManager) ListImages(ctx context.Context) ([]kubecontainer.Image, error) {
	logger := klog.FromContext(ctx)
	var images []kubecontainer.Image

	allImages, err := m.imageService.ListImages(ctx, nil)
	if err != nil {
		logger.Error(err, "Failed to list images")
		return nil, err
	}

	for _, img := range allImages {
		// Container runtimes may choose not to implement changes needed for KEP 4216. If
		// the changes are not implemented by a container runtime, the exisiting behavior
		// of not populating the runtimeHandler CRI field in ImageSpec struct is preserved.
		// Therefore, when RuntimeClassInImageCriAPI feature gate is set, check to see if this
		// field is empty and log a warning message.
		if utilfeature.DefaultFeatureGate.Enabled(features.RuntimeClassInImageCriAPI) {
			if img.Spec == nil || (img.Spec != nil && img.Spec.RuntimeHandler == "") {
				logger.V(2).Info("WARNING: RuntimeHandler is empty", "ImageID", img.Id)
			}
		}

		images = append(images, kubecontainer.Image{
			ID:          img.Id,
			Size:        int64(img.Size),
			RepoTags:    img.RepoTags,
			RepoDigests: img.RepoDigests,
			Spec:        toKubeContainerImageSpec(img),
			Pinned:      img.Pinned,
		})
	}

	return images, nil
}

// RemoveImage removes the specified image.
func (m *kubeGenericRuntimeManager) RemoveImage(ctx context.Context, image kubecontainer.ImageSpec) error {
	logger := klog.FromContext(ctx)
	err := m.imageService.RemoveImage(ctx, &runtimeapi.ImageSpec{Image: image.Image})
	if err != nil {
		logger.Error(err, "Failed to remove image", "image", image.Image)
		return err
	}

	return nil
}

// ImageStats returns the statistics of the image.
// Notice that current logic doesn't really work for images which share layers (e.g. docker image),
// this is a known issue, and we'll address this by getting imagefs stats directly from CRI.
// TODO: Get imagefs stats directly from CRI.
func (m *kubeGenericRuntimeManager) ImageStats(ctx context.Context) (*kubecontainer.ImageStats, error) {
	logger := klog.FromContext(ctx)
	allImages, err := m.imageService.ListImages(ctx, nil)
	if err != nil {
		logger.Error(err, "Failed to list images")
		return nil, err
	}
	stats := &kubecontainer.ImageStats{}
	for _, img := range allImages {
		stats.TotalStorageBytes += img.Size
	}
	return stats, nil
}

func (m *kubeGenericRuntimeManager) ImageFsInfo(ctx context.Context) (*runtimeapi.ImageFsInfoResponse, error) {
	logger := klog.FromContext(ctx)
	allImages, err := m.imageService.ImageFsInfo(ctx)
	if err != nil {
		logger.Error(err, "Failed to get image filesystem")
		return nil, err
	}
	return allImages, nil
}
