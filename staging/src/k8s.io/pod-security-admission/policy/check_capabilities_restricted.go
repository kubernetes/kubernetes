/*
Copyright 2021 The Kubernetes Authors.

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

package policy

import (
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/pod-security-admission/api"
)

const (
	capabilityAll            = "ALL"
	capabilityNetBindService = "NET_BIND_SERVICE"
)

/*
Containers must drop ALL, and may only add NET_BIND_SERVICE.

**Restricted Fields:**
spec.containers[*].securityContext.capabilities.drop
spec.initContainers[*].securityContext.capabilities.drop

**Allowed Values:**
Must include "ALL"

**Restricted Fields:**
spec.containers[*].securityContext.capabilities.add
spec.initContainers[*].securityContext.capabilities.add

**Allowed Values:**
undefined / empty
"NET_BIND_SERVICE"
*/

func init() {
	addCheck(CheckCapabilitiesRestricted)
}

// CheckCapabilitiesRestricted returns a restricted level check
// that ensures ALL capabilities are dropped in 1.22+
func CheckCapabilitiesRestricted() Check {
	return Check{
		ID:    "capabilities_restricted",
		Level: api.LevelRestricted,
		Versions: []VersionedCheck{
			{
				MinimumVersion: api.MajorMinorVersion(1, 22),
				CheckPod:       capabilitiesRestricted_1_22,
			},
		},
	}
}

func capabilitiesRestricted_1_22(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec) CheckResult {
	var (
		containersMissingDropAll  []string
		containersAddingForbidden []string
		forbiddenCapabilities     = sets.NewString()
	)

	visitContainers(podSpec, func(container *corev1.Container) {
		if container.SecurityContext == nil || container.SecurityContext.Capabilities == nil {
			containersMissingDropAll = append(containersMissingDropAll, container.Name)
			return
		}

		droppedAll := false
		for _, c := range container.SecurityContext.Capabilities.Drop {
			if c == capabilityAll {
				droppedAll = true
				break
			}
		}
		if !droppedAll {
			containersMissingDropAll = append(containersMissingDropAll, container.Name)
		}

		addedForbidden := false
		for _, c := range container.SecurityContext.Capabilities.Add {
			if c != capabilityNetBindService {
				addedForbidden = true
				forbiddenCapabilities.Insert(string(c))
			}
		}
		if addedForbidden {
			containersAddingForbidden = append(containersAddingForbidden, container.Name)
		}
	})
	var forbiddenDetails []string
	if len(containersMissingDropAll) > 0 {
		forbiddenDetails = append(forbiddenDetails, fmt.Sprintf(
			`%s %s must set securityContext.capabilities.drop=["ALL"]`,
			pluralize("container", "containers", len(containersMissingDropAll)),
			joinQuote(containersMissingDropAll)))
	}
	if len(containersAddingForbidden) > 0 {
		forbiddenDetails = append(forbiddenDetails, fmt.Sprintf(
			`%s %s must not include %s in securityContext.capabilities.add`,
			pluralize("container", "containers", len(containersAddingForbidden)),
			joinQuote(containersAddingForbidden),
			joinQuote(forbiddenCapabilities.List())))
	}
	if len(forbiddenDetails) > 0 {
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: "unrestricted capabilities",
			ForbiddenDetail: strings.Join(forbiddenDetails, "; "),
		}
	}
	return CheckResult{Allowed: true}
}
