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
	"strings"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/pod-security-admission/api"
)

/*
Sharing the host network, PID, and IPC namespaces must be disallowed.

**Restricted Fields:**

spec.hostNetwork
spec.hostPID
spec.hostIPC

**Allowed Values:** undefined, false
*/

func init() {
	addCheck(CheckHostNamespaces)
}

// CheckHostNamespaces returns a baseline level check
// that prohibits host namespaces in 1.0+
func CheckHostNamespaces() Check {
	return Check{
		ID:    "hostNamespaces",
		Level: api.LevelBaseline,
		Versions: []VersionedCheck{
			{
				MinimumVersion: api.MajorMinorVersion(1, 0),
				CheckPod:       withOptions(hostNamespacesV1Dot0),
			},
		},
	}
}

func hostNamespacesV1Dot0(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec, opts options) CheckResult {
	hostNamespaces := NewViolations(opts.withFieldErrors)

	if podSpec.HostNetwork {
		if opts.withFieldErrors {
			hostNamespaces.Add("hostNetwork=true", withBadValue(forbidden(hostNetworkPath), true))
		} else {
			hostNamespaces.Add("hostNetwork=true")
		}

	}

	if podSpec.HostPID {
		if opts.withFieldErrors {
			hostNamespaces.Add("hostPID=true", withBadValue(forbidden(hostPIDPath), true))
		} else {
			hostNamespaces.Add("hostPID=true")
		}
	}

	if podSpec.HostIPC {
		if opts.withFieldErrors {
			hostNamespaces.Add("hostIPC=true", withBadValue(forbidden(hostIPCPath), true))
		} else {
			hostNamespaces.Add("hostIPC=true")
		}
	}

	if !hostNamespaces.Empty() {
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: "host namespaces",
			ForbiddenDetail: strings.Join(hostNamespaces.Data(), ", "),
			ErrList:         hostNamespaces.Errs(),
		}
	}

	return CheckResult{Allowed: true}
}
