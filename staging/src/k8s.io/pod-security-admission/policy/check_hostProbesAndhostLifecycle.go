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
Host field is restricted in the probes and lifecycle handlers.

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

Allowed Values: "", undefined/null
*/

func init() {
	addCheck(CheckHostProbesAndHostLifecycle)
}

// CheckHostProbesAndHostLifecycle returns a baseline level check
// that forbids setting host field in probes and lifecycle handlers in 1.34+
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
	badContainers := sets.New[string]()
	forbidden := sets.New[string]()
	visitContainers(podSpec, func(container *corev1.Container) {
		// Check probes
		if badHosts := getForbiddenHostProbes(container.LivenessProbe); len(badHosts) > 0 {
			badContainers.Insert(container.Name)
			forbidden.Insert(badHosts...)
		}
		if badHosts := getForbiddenHostProbes(container.ReadinessProbe); len(badHosts) > 0 {
			badContainers.Insert(container.Name)
			forbidden.Insert(badHosts...)
		}
		if badHosts := getForbiddenHostProbes(container.StartupProbe); len(badHosts) > 0 {
			badContainers.Insert(container.Name)
			forbidden.Insert(badHosts...)
		}

		// Check lifecycle handlers
		if container.Lifecycle != nil {
			if badHosts := getForbiddenHostLifecycle(container.Lifecycle.PostStart); len(badHosts) > 0 {
				badContainers.Insert(container.Name)
				forbidden.Insert(badHosts...)
			}
			if badHosts := getForbiddenHostLifecycle(container.Lifecycle.PreStop); len(badHosts) > 0 {
				badContainers.Insert(container.Name)
				forbidden.Insert(badHosts...)
			}
		}
	})

	if len(badContainers) > 0 {
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: "probe or lifecycle host",
			ForbiddenDetail: fmt.Sprintf(
				"%s %s %s %s %s",
				pluralize("container", "containers", len(badContainers)),
				joinQuote(sets.List(badContainers)),
				pluralize("uses", "use", len(badContainers)),
				pluralize("probe or lifecycle host", "probe or lifecycle hosts", len(forbidden)),
				joinQuote(sets.List(forbidden)),
			),
		}
	}
	return CheckResult{Allowed: true}
}

func getForbiddenHostProbes(probe *corev1.Probe) []string {
	var badHosts []string
	if probe == nil {
		return nil
	}
	if probe.HTTPGet != nil && probe.HTTPGet.Host != "" {
		badHosts = append(badHosts, probe.HTTPGet.Host)
	}
	if probe.TCPSocket != nil && probe.TCPSocket.Host != "" {
		badHosts = append(badHosts, probe.TCPSocket.Host)
	}
	return badHosts
}

func getForbiddenHostLifecycle(handler *corev1.LifecycleHandler) []string {
	var badHosts []string
	if handler == nil {
		return nil
	}
	if handler.HTTPGet != nil && handler.HTTPGet.Host != "" {
		badHosts = append(badHosts, handler.HTTPGet.Host)
	}
	if handler.TCPSocket != nil && handler.TCPSocket.Host != "" {
		badHosts = append(badHosts, handler.TCPSocket.Host)
	}
	return badHosts
}
