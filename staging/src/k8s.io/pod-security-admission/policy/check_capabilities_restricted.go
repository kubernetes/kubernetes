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
	"k8s.io/apimachinery/pkg/util/validation/field"
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
				MinimumVersion:   api.MajorMinorVersion(1, 22),
				CheckPod:         withOptions(capabilitiesRestrictedV1Dot22),
				OverrideCheckIDs: []CheckID{checkCapabilitiesBaselineID},
			},
			// Starting 1.25, windows pods would be exempted from this check using pod.spec.os field when set to windows.
			{
				MinimumVersion:   api.MajorMinorVersion(1, 25),
				CheckPod:         withOptions(capabilitiesRestrictedV1Dot25),
				OverrideCheckIDs: []CheckID{checkCapabilitiesBaselineID},
			},
		},
	}
}

func capabilitiesRestrictedV1Dot22(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec, opts options) CheckResult {
	forbiddenCapabilities := sets.NewString()
	containersMissingDropAll := NewViolations(opts.withFieldErrors)
	containersAddingForbidden := NewViolations(opts.withFieldErrors)

	visitContainers(podSpec, opts, func(container *corev1.Container, path *field.Path) {
		if container.SecurityContext == nil || container.SecurityContext.Capabilities == nil {
			containersMissingDropAll.Add(container.Name, required(path.Child("securityContext", "capabilities", "drop")))
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
			if opts.withFieldErrors {
				length := len(container.SecurityContext.Capabilities.Drop)
				if length > 0 {
					strSlice := make([]string, len(container.SecurityContext.Capabilities.Drop))
					for i, v := range container.SecurityContext.Capabilities.Drop {
						strSlice[i] = string(v)
					}
					forbiddenValues := sets.NewString(strSlice...)
					containersMissingDropAll.Add(container.Name, withBadValue(forbidden(path.Child("securityContext", "capabilities", "drop")), forbiddenValues.List()))
				} else if length == 0 {
					containersMissingDropAll.Add(container.Name, required(path.Child("securityContext", "capabilities", "drop")))
				}
			} else {
				containersMissingDropAll.Add(container.Name)
			}
		}

		addedForbidden := false
		if opts.withFieldErrors {
			forbiddenValues := sets.NewString()
			for _, c := range container.SecurityContext.Capabilities.Add {
				if c != capabilityNetBindService {
					addedForbidden = true
					forbiddenCapabilities.Insert(string(c))
					forbiddenValues.Insert(string(c))
				}
			}
			if addedForbidden {
				containersAddingForbidden.Add(container.Name, withBadValue(forbidden(path.Child("securityContext", "capabilities", "add")), forbiddenValues.List()))
			}
		} else {
			for _, c := range container.SecurityContext.Capabilities.Add {
				if c != capabilityNetBindService {
					addedForbidden = true
					forbiddenCapabilities.Insert(string(c))
				}
			}
			if addedForbidden {
				containersAddingForbidden.Add(container.Name)
			}
		}
	})

	var forbiddenDetails []string
	var errList *field.ErrorList
	if opts.withFieldErrors {
		errs := append(*containersMissingDropAll.Errs(), *containersAddingForbidden.Errs()...)
		errList = &errs
	}
	if !containersMissingDropAll.Empty() {
		forbiddenDetails = append(forbiddenDetails, fmt.Sprintf(
			`%s %s must set securityContext.capabilities.drop=["ALL"]`,
			pluralize("container", "containers", containersMissingDropAll.Len()),
			joinQuote(containersMissingDropAll.Data())))
	}
	if !containersAddingForbidden.Empty() {
		forbiddenDetails = append(forbiddenDetails, fmt.Sprintf(
			`%s %s must not include %s in securityContext.capabilities.add`,
			pluralize("container", "containers", containersAddingForbidden.Len()),
			joinQuote(containersAddingForbidden.Data()),
			joinQuote(forbiddenCapabilities.List())))
	}
	if len(forbiddenDetails) > 0 {
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: "unrestricted capabilities",
			ForbiddenDetail: strings.Join(forbiddenDetails, "; "),
			ErrList:         errList,
		}
	}
	return CheckResult{Allowed: true}
}

func capabilitiesRestrictedV1Dot25(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec, opts options) CheckResult {
	// Pod API validation would have failed if podOS == Windows and if capabilities have been set.
	// We can admit the Windows pod even if capabilities has not been set.
	if podSpec.OS != nil && podSpec.OS.Name == corev1.Windows {
		return CheckResult{Allowed: true}
	}
	return capabilitiesRestrictedV1Dot22(podMetadata, podSpec, opts)
}
