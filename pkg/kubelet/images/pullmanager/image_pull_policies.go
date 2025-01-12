/*
Copyright 2024 The Kubernetes Authors.

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
type ImagePullPolicyEnforcer func(image string, imagePresent, imagePulledByKubelet bool) bool

func NewImagePullCredentialVerificationPolicy(policy kubeletconfiginternal.ImagePullCredentialsVerificationPolicy, imageAllowList []string) (ImagePullPolicyEnforcer, error) {
	switch policy {
	case kubeletconfiginternal.NeverVerify:
		return ImagePullPolicyEnforcer(NeverVerifyImagePullPolicy), nil
	case "", kubeletconfiginternal.NeverVerifyPreloadedImages:
		return ImagePullPolicyEnforcer(NeverVerifyPreloadedPullPolicy), nil
	case kubeletconfiginternal.NeverVerifyAllowlistedImages:
		return NewNeverVerifyAllowListedPullPolicy(imageAllowList)
	case kubeletconfiginternal.AlwaysVerify:
		return ImagePullPolicyEnforcer(AlwaysVerifyImagePullPolicy), nil
	default:
		return nil, fmt.Errorf("unknown image pull credential verification policy: %v", policy)
	}
}

func NeverVerifyImagePullPolicy(image string, imagePresent, imagePulledByKubelet bool) bool {
	return false
}

func NeverVerifyPreloadedPullPolicy(image string, imagePresent, imagePulledByKubelet bool) bool {
	if imagePresent && !imagePulledByKubelet {
		return false
	}
	return true
}

func AlwaysVerifyImagePullPolicy(image string, imagePresent, imagePulledByKubelet bool) bool {
	return true
}

type NeverVerifyAllowlistedImages struct {
	absoluteURLs sets.Set[string]
	prefixes     []string
}

func (p *NeverVerifyAllowlistedImages) RequiresVerification(image string, imagePresent, imagePulledByKubelet bool) bool {
	if imagePresent && p.imageMatches(image) {
		return false
	}
	return true
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

func (p *NeverVerifyAllowlistedImages) AllowPattern(pattern string) error {
	pattern = strings.TrimSpace(pattern)

	if err := validateAllowlistImagePattern(pattern); err != nil {
		return err
	}

	if strings.HasSuffix(pattern, "/*") {
		p.prefixes = append(p.prefixes, pattern[:len(pattern)-2])
	} else {
		p.absoluteURLs.Insert(pattern)
	}
	return nil
}

func NewNeverVerifyAllowListedPullPolicy(allowList []string) (ImagePullPolicyEnforcer, error) {
	policy := &NeverVerifyAllowlistedImages{
		absoluteURLs: sets.New[string](),
	}
	for _, pattern := range allowList {
		if err := policy.AllowPattern(pattern); err != nil {
			return nil, err
		}
	}

	return policy.RequiresVerification, nil
}

func ValidateAllowlistImagesPatterns(patterns []string) error {
	for _, p := range patterns {
		if err := validateAllowlistImagePattern(p); err != nil {
			return err
		}
	}
	return nil
}

func validateAllowlistImagePattern(pattern string) error {
	if strings.ContainsRune(pattern[:len(pattern)-1], '*') {
		return fmt.Errorf("only a trailing full path segment wildcard '/*' is allowed: %s", pattern)
	}

	if !strings.HasSuffix(pattern, "*") {
		if strings.HasSuffix(pattern, "/") {
			return fmt.Errorf("image pattern cannot end with '/': %s", pattern)
		}
		return nil
	}

	if !strings.HasSuffix(pattern, "/*") {
		return fmt.Errorf("only a full path segment wildcard '/*' is allowed: %s", pattern)
	}

	prefix := pattern[:len(pattern)-2]
	if len(prefix) == 0 {
		return fmt.Errorf("at least registry hostname is required: %q", pattern)
	}
	return nil
}
