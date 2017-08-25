/*
Copyright 2017 The Kubernetes Authors.

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

package upgrade

import (
	"k8s.io/kubernetes/pkg/util/version"
)

// VersionSkewPolicyErrors describes version skew errors that might be seen during the validation process in EnforceVersionPolicies
type VersionSkewPolicyErrors struct {
	Mandatory []error
	Skippable []error
}

// EnforceVersionPolicies enforces that the proposed new version is compatible with all the different version skew policies
func EnforceVersionPolicies(_ VersionGetter, _ string, _ *version.Version, _, _ bool) *VersionSkewPolicyErrors {
	// No-op now and return no skew errors
	return nil
}
