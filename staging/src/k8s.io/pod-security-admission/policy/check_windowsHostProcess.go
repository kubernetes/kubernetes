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
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/pod-security-admission/api"
)

/*
Pod and containers must not set securityContext.windowsOptions.hostProcess to true.

**Restricted Fields:**

spec.securityContext.windowsOptions.hostProcess
spec.containers[*].securityContext.windowsOptions.hostProcess
spec.initContainers[*].securityContext.windowsOptions.hostProcess

**Allowed Values:** undefined / false
*/

func init() {
	addCheck(CheckWindowsHostProcess)
}

// CheckWindowsHostProcess returns a baseline level check
// that forbids hostProcess=true in 1.0+
func CheckWindowsHostProcess() Check {
	return Check{
		ID:    "windowsHostProcess",
		Level: api.LevelBaseline,
		Versions: []VersionedCheck{
			{
				MinimumVersion: api.MajorMinorVersion(1, 0),
				CheckPod:       withOptions(windowsHostProcessV1Dot0),
			},
		},
	}
}

func windowsHostProcessV1Dot0(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec, opts options) CheckResult {
	var badContainers []string
	var errs field.ErrorList
	visitContainers(podSpec, opts, func(container *corev1.Container, path *field.Path) {
		if container.SecurityContext != nil &&
			container.SecurityContext.WindowsOptions != nil &&
			container.SecurityContext.WindowsOptions.HostProcess != nil &&
			*container.SecurityContext.WindowsOptions.HostProcess {
			badContainers = append(badContainers, container.Name)
			if opts.withFieldErrors {
				errs = append(errs, withBadValue(forbidden(path.Child("securityContext", "windowsOptions", "hostProcess")), true))
			}
		}
	})

	podSpecForbidden := false
	if podSpec.SecurityContext != nil &&
		podSpec.SecurityContext.WindowsOptions != nil &&
		podSpec.SecurityContext.WindowsOptions.HostProcess != nil &&
		*podSpec.SecurityContext.WindowsOptions.HostProcess {
		podSpecForbidden = true
	}

	// pod or containers explicitly set hostProcess=true
	forbiddenSetters := NewViolations(opts.withFieldErrors)
	if podSpecForbidden {
		if opts.withFieldErrors {
			forbiddenSetters.Add("pod", withBadValue(forbidden(hostProcessPath), true))
		} else {
			forbiddenSetters.Add("pod")
		}

	}
	if len(badContainers) > 0 {
		forbiddenSetters.Add(
			fmt.Sprintf(
				"%s %s",
				pluralize("container", "containers", len(badContainers)),
				joinQuote(badContainers),
			),
			errs...,
		)
	}
	if !forbiddenSetters.Empty() {
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: "hostProcess",
			ForbiddenDetail: fmt.Sprintf("%s must not set securityContext.windowsOptions.hostProcess=true", strings.Join(forbiddenSetters.Data(), " and ")),
			ErrList:         forbiddenSetters.Errs(),
		}
	}

	return CheckResult{Allowed: true}
}
