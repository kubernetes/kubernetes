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

package images

import (
	"fmt"

	"k8s.io/apimachinery/pkg/util/sets"
	kubeletconfig1betav1 "k8s.io/kubelet/config/v1beta1"
)

type ImagePullPolicyEnforcer func(image string, imagePresent, imageRecordsExist bool) bool

func NewImagePullCredentialVerificationPolicy(policy kubeletconfig1betav1.ImagePullCredentialsVerificationPolicy, imageAllowList []string) (ImagePullPolicyEnforcer, error) {
	switch policy {
	case kubeletconfig1betav1.NeverVerify:
		return ImagePullPolicyEnforcer(NeverVerifyImagePullPolicy), nil
	case kubeletconfig1betav1.NeverVerifyPreloadedImages:
		return ImagePullPolicyEnforcer(NeverVerifyPreloadedPullPolicy), nil
	case kubeletconfig1betav1.NeverVerifyAllowlistedImages:
		return NewNeverVerifyAllowListedPullPolicy(imageAllowList), nil
	case kubeletconfig1betav1.AlwaysVerify:
		return ImagePullPolicyEnforcer(AlwaysVerifyImagePullPolicy), nil
	default:
		return nil, fmt.Errorf("unknown image pull credential verification policy: %v", policy)
	}
}

func NeverVerifyImagePullPolicy(image string, imagePresent, imageRecordsExist bool) bool {
	return false
}

func NeverVerifyPreloadedPullPolicy(image string, imagePresent, imageRecordsExist bool) bool {
	if imagePresent && !imageRecordsExist {
		return false
	}
	return true
}

func NewNeverVerifyAllowListedPullPolicy(allowList []string) ImagePullPolicyEnforcer {
	allowListSet := sets.New(allowList...)
	return func(image string, imagePresent, imageRecordsExist bool) bool {
		if imagePresent && allowListSet.Has(image) {
			return false
		}
		return true
	}
}

func AlwaysVerifyImagePullPolicy(image string, imagePresent, imageRecordsExist bool) bool {
	return true
}
