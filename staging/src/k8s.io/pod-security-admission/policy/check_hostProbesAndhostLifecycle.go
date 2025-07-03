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

package policy

import (
	"fmt"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/pod-security-admission/api"
)

/*
Host field must be forbidden in the probes and lifecycle handlers.

**Restricted Fields:**

* spec.containers[*].livenessProbe.httpGet.host
* spec.containers[*].readinessProbe.httpGet.host
* spec.containers[*].startupProbe.httpGet.host
* spec.containers[*].livenessProbe.tcpSocket.host
* spec.containers[*].readinessProbe.tcpSocket.host
* spec.containers[*].startupProbe.tcpSocket.host
* spec.containers[*].lifecycle.postStart.tcpSocket.host // Deprecated. TCPSocket is NOT supported as a LifecycleHandler and kept for backward compatibility.
* spec.containers[*].lifecycle.preStop.tcpSocket.host // Deprecated. TCPSocket is NOT supported as a LifecycleHandler and kept for backward compatibility.
* spec.containers[*].lifecycle.postStart.httpGet.host
* spec.containers[*].lifecycle.preStop.httpGet.host
* spec.initContainers[*].livenessProbe.httpGet.host
* spec.initContainers[*].readinessProbe.httpGet.host
* spec.initContainers[*].startupProbe.httpGet.host
* spec.initContainers[*].livenessProbe.tcpSocket.host
* spec.initContainers[*].readinessProbe.tcpSocket.host
* spec.initContainers[*].startupProbe.tcpSocket.host
* spec.initContainers[*].lifecycle.postStart.tcpSocket.host // Deprecated. TCPSocket is NOT supported as a LifecycleHandler and kept for backward compatibility.
* spec.initContainers[*].lifecycle.preStop.tcpSocket.host // Deprecated. TCPSocket is NOT supported as a LifecycleHandler and kept for backward compatibility.
* spec.initContainers[*].lifecycle.postStart.httpGet.host
* spec.initContainers[*].lifecycle.preStop.httpGet.host


**Allowed Values:** "127.0.0.1", "::1"
*/

const (
	allowedLocalHostIPv4 = "127.0.0.1"
	allowedLocalHostIPv6 = "::1"
)

func init() {
	addCheck(CheckHostProbesAndHostLifecycle)
}

// CheckHostProbesAndHostLifecycle returns a baseline level check
// that forbids setting host field in probes and lifecycle handlers in 1.34+
// the only allowed values are `127.0.0.1` and `::1`
func CheckHostProbesAndHostLifecycle() Check {
	return Check{
		ID:    "hostProbesAndHostLifecycle",
		Level: api.LevelBaseline,
		Versions: []VersionedCheck{
			{
				MinimumVersion: api.MajorMinorVersion(1, 34),
				CheckPod:       hostProbesAndHostLifecycleV1Dot34,
			},
		},
	}
}

func hostProbesAndHostLifecycleV1Dot34(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec) CheckResult {
	var badContainers []string
	forbidden := sets.New[string]()
	allowed := sets.New[string](allowedLocalHostIPv4, allowedLocalHostIPv6)
	visitContainers(podSpec, func(container *corev1.Container) {
		isValid := true

		// Check probes
		for _, probe := range []*corev1.Probe{container.LivenessProbe, container.ReadinessProbe, container.StartupProbe} {
			if probe == nil {
				continue
			}
			if probe.HTTPGet != nil {
				if !checkHost(probe.HTTPGet.Host, allowed, forbidden) {
					isValid = false
				}
			}
			if probe.TCPSocket != nil {
				if !checkHost(probe.TCPSocket.Host, allowed, forbidden) {
					isValid = false
				}
			}
		}

		// Check lifecycle handlers
		if container.Lifecycle != nil {
			for _, handler := range []*corev1.LifecycleHandler{container.Lifecycle.PostStart, container.Lifecycle.PreStop} {
				if handler == nil {
					continue
				}
				if handler.HTTPGet != nil {
					if !checkHost(handler.HTTPGet.Host, allowed, forbidden) {
						isValid = false
					}
				}
				if handler.TCPSocket != nil {
					if !checkHost(handler.TCPSocket.Host, allowed, forbidden) {
						isValid = false
					}
				}
			}
		}

		if !isValid {
			badContainers = append(badContainers, container.Name)
		}

	})

	if len(badContainers) > 0 {
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: "hostProbesOrHostLifecycle",
			ForbiddenDetail: fmt.Sprintf(
				"%s %s %s %s %s",
				pluralize("container", "containers", len(badContainers)),
				joinQuote(badContainers),
				pluralize("uses", "use", len(badContainers)),
				pluralize("hostProbeOrHostLifecycle", "hostProbesOrHostLifecycles", len(forbidden)),
				joinQuote(sets.List(forbidden)),
			),
		}
	}
	return CheckResult{Allowed: true}
}

func checkHost(host string, allowed, forbidden sets.Set[string]) bool {
	if host != "" && !allowed.Has(host) {
		forbidden.Insert(host)
		return false
	}
	return true
}
