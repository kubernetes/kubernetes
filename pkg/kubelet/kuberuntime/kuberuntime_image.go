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

// createKeychainFromSecrets creates an authn.Keychain from Kubernetes pull secrets
func createKeychainFromSecrets(pullSecrets []v1.Secret) authn.Keychain {
	if len(pullSecrets) == 0 {
		klog.V(4).Info("No pull secrets provided for fetching remote image digest, using default keychain")
		return authn.DefaultKeychain
	}
	klog.V(4).Info("Using pull secrets for keychain", "count", len(pullSecrets))
	return &secretKeychain{secrets: pullSecrets}
}

// GetRemoteImageRef fetches the remote image reference (digest) from the registry.
// Falls back to the original imageRef string if fetching fails.
func (m *kubeGenericRuntimeManager) GetRemoteImageRef(
	ctx context.Context,
	imageName string,
	pullSecrets []v1.Secret,
) (string, error) {
	logger := klog.FromContext(ctx)
	logger.V(3).Info("Fetching remote image", "imageName", imageName)

	// Parse the image reference
	imageRef, err := name.ParseReference(imageName)
	if err != nil {
		logger.Error(err, "Failed to parse image reference", "imageName", imageName)
		return "", fmt.Errorf("invalid image reference %q: %w", imageName, err)
	}

	// Create keychain from pull secrets
	keychain := createKeychainFromSecrets(pullSecrets)

	// Retry fetching remote image digest
	remoteImage, err := remote.Image(imageRef, remote.WithContext(ctx), remote.WithAuthFromKeychain(keychain))

	for i := 0; i < 5; i++ {

		//Checking if fetched and then retries if not.
		if err == nil {
			break
		}
		logger.V(4).Error(err, "Retry fetching remote image digest", "imageRef", imageRef, "attempt", i+1)
		time.Sleep(time.Second * time.Duration(i+1))
		remoteImage, err = remote.Image(imageRef, remote.WithContext(ctx), remote.WithAuthFromKeychain(keychain))
	}

	if err != nil {
		logger.V(3).Error(err, "Failed to fetch remote image digest, fallback to original imageRef", "imageRef", imageRef)
		return imageRef.String(), nil
	}

	// Get the digest reference
	remoteImageRef, err := remoteImage.ConfigName()
	if err != nil {
		logger.V(3).Error(err, "Failed to get remote image digest, fallback to original imageRef", "imageRef", imageRef)
		return imageRef.String(), nil
	}

	logger.V(3).Info("Successfully fetched remote image digest", "digest", remoteImageRef.String())
	return remoteImageRef.String(), nil
}

// secretKeychain implements authn.Keychain for Kubernetes secrets
type secretKeychain struct {
	secrets []v1.Secret
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

	// Loop through auth entries and return the first valid one
	for registry, authData := range dockerConfig.Auths {
		if authData.Auth == "" {
			continue
		}

		decoded, err := base64.StdEncoding.DecodeString(authData.Auth)
		if err != nil {
			klog.V(4).Infof("Failed to decode auth token for registry %s in secret %s: %v", registry, secret.Name, err)
			continue
		}

		parts := strings.SplitN(string(decoded), ":", 2)
		if len(parts) != 2 {
			klog.V(4).Infof("Invalid auth token format for registry %s in secret %s", registry, secret.Name)
			continue
		}

		klog.V(4).Infof("Using credentials from secret %s for registry %s", secret.Name, registry)
		return &authn.Basic{
			Username: parts[0],
			Password: parts[1],
		}, nil
	}

	klog.V(5).Infof("No valid auth entries found in secret %s", secret.Name)
	return nil, nil
}

func (k *secretKeychain) Resolve(target authn.Resource) (authn.Authenticator, error) {
	registry := target.RegistryStr()
	klog.V(4).Infof("Resolving auth for registry: %s", registry)

	for _, secret := range k.secrets {
		authConfig, err := getAuthenticatorFromSecret(&secret)
		if err != nil {
			klog.V(4).Infof("Failed to get auth from secret %s: %v", secret.Name, err)
			continue
		}
		if authConfig != nil {
			klog.V(4).Infof("Found credentials in secret %s for registry %s", secret.Name, registry)
			return authConfig, nil
		}
	}

	klog.V(4).Infof("No credentials found in secrets for registry %s, falling back to default keychain", registry)
	return authn.DefaultKeychain.Resolve(target)
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
