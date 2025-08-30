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
	"k8s.io/client-go/util/flowcontrol"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/credentialprovider"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// throttleImagePulling wraps kubecontainer.ImageService to throttle image
// pulling based on the given QPS and burst limits. If QPS is zero, defaults
// to no throttling.
func throttleImagePulling(imageService kubecontainer.ImageService, qps float32, burst int) kubecontainer.ImageService {
	if qps == 0.0 {
		return imageService
	}
	return &throttledImageService{
		ImageService: imageService,
		limiter:      flowcontrol.NewTokenBucketRateLimiter(qps, burst),
	}
}

type throttledImageService struct {
	kubecontainer.ImageService
	limiter flowcontrol.RateLimiter
}

func (ts throttledImageService) PullImage(ctx context.Context, image kubecontainer.ImageSpec, credentials []credentialprovider.TrackedAuthConfig, podSandboxConfig *runtimeapi.PodSandboxConfig) (string, *credentialprovider.TrackedAuthConfig, error) {
	if ts.limiter.TryAccept() {
		return ts.ImageService.PullImage(ctx, image, credentials, podSandboxConfig)
	}
	return "", nil, fmt.Errorf("pull QPS exceeded")
}

// GetRemoteImageDigestWithoutPull fetches the digest of a remote image
// without pulling its layers. It supports public and private registries via authn.DefaultKeychain.
func GetRemoteImageDigestWithoutPull(ctx context.Context, imageName string, pullSecrets []v1.Secret) (string, error) {
	// Parse the image reference
	imageRef, err := name.ParseReference(imageName)
	if err != nil {
		klog.Errorf("Failed to parse image reference %q: %v", imageName, err)
		return "", err
	}

	// Create keychain from pull secrets
	keychain := createKeychainFromSecrets(pullSecrets)
    
	// Fetch the remote image with authentication
    remoteImage, err := remote.Image(imageRef, remote.WithContext(ctx), remote.WithAuthFromKeychain(keychain))
	// Retries for 15 sec
	for i := 0; i < 5; i++ {
		if err == nil {
			break
		}
	    klog.V(4).Infof("Failed x%d to fetch remote image: %s, retrying after %d", i+1, imageName, time.Second * time.Duration(i+1))
		time.Sleep(time.Second * time.Duration(i+1))
		remoteImage, err = remote.Image(imageRef, remote.WithContext(ctx), remote.WithAuthFromKeychain(keychain))
	}
    
	if err != nil {
		return imageRef.String(), fmt.Errorf("Failed to fetch image after retries: %w.", err)
	}
    
	// Get the image ID (config digest)
	remoteImageRef, err := remoteImage.ConfigName()
	if err != nil {
		klog.Errorf("Failed to get remote image id (digest) for %q: %v.", imageName, err)
		return imageRef.String(), err
	}

	klog.V(4).Infof("Successfully fetched remote digest: %s.", remoteImageRef.String())
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
		return nil, fmt.Errorf("Failed to unmarshal docker config in secret %s: %w", secret.Name, err)
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

// createKeychainFromSecrets creates an authn.Keychain from Kubernetes pull secrets
func createKeychainFromSecrets(pullSecrets []v1.Secret) authn.Keychain {
	if len(pullSecrets) == 0 {
		klog.V(4).Info("No pull secrets provided, using default keychain")
		return authn.DefaultKeychain
	}
	return &secretKeychain{secrets: pullSecrets}
}
