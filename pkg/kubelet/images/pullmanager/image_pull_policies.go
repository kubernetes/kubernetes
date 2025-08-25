/*
Copyright 2025 The Kubernetes Authors.

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

package pullmanager

import (
	"fmt"
	"strings"

	dockerref "github.com/distribution/reference"

	"k8s.io/apimachinery/pkg/util/sets"
	kubeletconfiginternal "k8s.io/kubernetes/pkg/kubelet/apis/config"
)

// ImagePullPolicyEnforcer defines a class of functions implementing a credential
// verification policies for image pulls. These function determines whether the
// implemented policy requires credential verification based on image name, local
// image presence and existence of records about previous image pulls.
//
// `image` is an image name from a Pod's container "image" field.
// `imagePresent` informs whether the `image` is present on the node.
// `imagePulledByKubelet` marks that ImagePulledRecord or ImagePullingIntent records
// for the `image` exist on the node, meaning it was pulled by the kubelet somewhere
// in the past.
type ImagePullPolicyEnforcer interface {
	RequireCredentialVerificationForImage(image string, imagePulledByKubelet bool) bool
}

// ImagePullPolicyEnforcerFunc is a function type that implements the ImagePullPolicyEnforcer interface
type ImagePullPolicyEnforcerFunc func(image string, imagePulledByKubelet bool) bool

func (e ImagePullPolicyEnforcerFunc) RequireCredentialVerificationForImage(image string, imagePulledByKubelet bool) bool {
	return e(image, imagePulledByKubelet)
}

func NewImagePullCredentialVerificationPolicy(policy kubeletconfiginternal.ImagePullCredentialsVerificationPolicy, imageAllowList []string) (ImagePullPolicyEnforcer, error) {
	switch policy {
	case kubeletconfiginternal.NeverVerify:
		return NeverVerifyImagePullPolicy(), nil
	case kubeletconfiginternal.NeverVerifyPreloadedImages:
		return NeverVerifyPreloadedPullPolicy(), nil
	case kubeletconfiginternal.NeverVerifyAllowlistedImages:
		return NewNeverVerifyAllowListedPullPolicy(imageAllowList)
	case kubeletconfiginternal.AlwaysVerify:
		return AlwaysVerifyImagePullPolicy(), nil
	default:
		return nil, fmt.Errorf("unknown image pull credential verification policy: %v", policy)
	}
}

func NeverVerifyImagePullPolicy() ImagePullPolicyEnforcerFunc {
	return func(image string, imagePulledByKubelet bool) bool {
		return false
	}
}

func NeverVerifyPreloadedPullPolicy() ImagePullPolicyEnforcerFunc {
	return func(image string, imagePulledByKubelet bool) bool {
		return imagePulledByKubelet
	}
}

func AlwaysVerifyImagePullPolicy() ImagePullPolicyEnforcerFunc {
	return func(image string, imagePulledByKubelet bool) bool {
		return true
	}
}

type NeverVerifyAllowlistedImages struct {
	absoluteURLs sets.Set[string]
	prefixes     []string
}

func NewNeverVerifyAllowListedPullPolicy(allowList []string) (*NeverVerifyAllowlistedImages, error) {
	policy := &NeverVerifyAllowlistedImages{
		absoluteURLs: sets.New[string](),
	}
	for _, pattern := range allowList {
		normalizedPattern, isWildcard, err := getAllowlistImagePattern(pattern)
		if err != nil {
			return nil, err
		}

		if isWildcard {
			policy.prefixes = append(policy.prefixes, normalizedPattern)
		} else {
			policy.absoluteURLs.Insert(normalizedPattern)
		}
	}

	return policy, nil
}

func (p *NeverVerifyAllowlistedImages) RequireCredentialVerificationForImage(image string, imagePulledByKubelet bool) bool {
	return !p.imageMatches(image)
}

func (p *NeverVerifyAllowlistedImages) imageMatches(image string) bool {
	if p.absoluteURLs.Has(image) {
		return true
	}
	for _, prefix := range p.prefixes {
		if strings.HasPrefix(image, prefix) {
			return true
		}
	}
	return false
}

func ValidateAllowlistImagesPatterns(patterns []string) error {
	for _, p := range patterns {
		if _, _, err := getAllowlistImagePattern(p); err != nil {
			return err
		}
	}
	return nil
}

func getAllowlistImagePattern(pattern string) (string, bool, error) {
	if pattern != strings.TrimSpace(pattern) {
		return "", false, fmt.Errorf("leading/trailing spaces are not allowed: %s", pattern)
	}

	trimmedPattern := pattern
	isWildcard := false
	if strings.HasSuffix(pattern, "/*") {
		isWildcard = true
		trimmedPattern = strings.TrimSuffix(trimmedPattern, "*")
	}

	if len(trimmedPattern) == 0 {
		return "", false, fmt.Errorf("the supplied pattern is too short: %s", pattern)
	}

	if strings.ContainsRune(trimmedPattern, '*') {
		return "", false, fmt.Errorf("not a valid wildcard pattern, only patterns ending with '/*' are allowed: %s", pattern)
	}

	if isWildcard {
		if len(trimmedPattern) == 1 {
			return "", false, fmt.Errorf("at least registry hostname is required")
		}
	} else { // not a wildcard
		image, err := dockerref.ParseNormalizedNamed(trimmedPattern)
		if err != nil {
			return "", false, fmt.Errorf("failed to parse as an image name: %w", err)
		}

		if trimmedPattern != image.Name() { // image.Name() returns the image name without tag/digest
			return "", false, fmt.Errorf("neither tag nor digest is accepted in an image reference: %s", pattern)
		}

		return trimmedPattern, false, nil
	}

	return trimmedPattern, true, nil
}
