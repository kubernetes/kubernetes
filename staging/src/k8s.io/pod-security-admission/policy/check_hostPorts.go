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
	"strconv"
	"strings"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/pod-security-admission/api"
)

/*
HostPort ports must be forbidden.

**Restricted Fields:**

spec.containers[*].ports[*].hostPort
spec.initContainers[*].ports[*].hostPort

**Allowed Values:** undefined/0
*/

func init() {
	addCheck(CheckHostPorts)
}

// CheckHostPorts returns a baseline level check
// that forbids any host ports in 1.0+
func CheckHostPorts() Check {
	return Check{
		ID:    "hostPorts",
		Level: api.LevelBaseline,
		Versions: []VersionedCheck{
			{
				MinimumVersion: api.MajorMinorVersion(1, 0),
				CheckPod:       withOptions(hostPortsV1Dot0),
			},
		},
	}
}

func hostPortsV1Dot0(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec, opts options) CheckResult {
	badContainers := NewViolations(opts.withFieldErrors)
	forbiddenHostPorts := sets.NewString()
	visitContainers(podSpec, opts, func(container *corev1.Container, path *field.Path) {
		valid := true
		var errs field.ErrorList
		for i, c := range container.Ports {
			if c.HostPort != 0 {
				valid = false
				forbiddenHostPorts.Insert(strconv.Itoa(int(c.HostPort)))
				if opts.withFieldErrors {
					errs = append(errs, withBadValue(forbidden(path.Child("ports").Index(i).Child("hostPort")), int(c.HostPort)))
				}
			}
		}
		if !valid {
			if opts.withFieldErrors {
				badContainers.Add(container.Name, errs...)
			} else {
				badContainers.Add(container.Name)
			}
		}
	})

	if !badContainers.Empty() {
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: "hostPort",
			ForbiddenDetail: fmt.Sprintf(
				"%s %s %s %s %s",
				pluralize("container", "containers", badContainers.Len()),
				joinQuote(badContainers.Data()),
				pluralize("uses", "use", badContainers.Len()),
				pluralize("hostPort", "hostPorts", len(forbiddenHostPorts)),
				strings.Join(forbiddenHostPorts.List(), ", "),
			),
			ErrList: badContainers.Errs(),
		}
	}
	return CheckResult{Allowed: true}
}
