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

package util

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/util/filesystem"
)

// FromApiserverCache modifies <opts> so that the GET request will
// be served from apiserver cache instead of from etcd.
func FromApiserverCache(opts *metav1.GetOptions) {
	opts.ResourceVersion = "0"
}

var IsUnixDomainSocket = filesystem.IsUnixDomainSocket

// GetNodenameForKernel gets hostname value to set in the hostname field (the nodename field of struct utsname) of the pod.
func GetNodenameForKernel(hostname string, hostDomainName string, setHostnameAsFQDN *bool) (string, error) {
	kernelHostname := hostname
	// FQDN has to be 64 chars to fit in the Linux nodename kernel field (specification 64 chars and the null terminating char).
	const fqdnMaxLen = 64
	if len(hostDomainName) > 0 && setHostnameAsFQDN != nil && *setHostnameAsFQDN {
		fqdn := fmt.Sprintf("%s.%s", hostname, hostDomainName)
		// FQDN has to be shorter than hostnameMaxLen characters.
		if len(fqdn) > fqdnMaxLen {
			return "", fmt.Errorf("failed to construct FQDN from pod hostname and cluster domain, FQDN %s is too long (%d characters is the max, %d characters requested)", fqdn, fqdnMaxLen, len(fqdn))
		}
		kernelHostname = fqdn
	}
	return kernelHostname, nil
}

// GetContainerByIndex validates and extracts the container at index "idx" from
// "containers" with respect to "statuses".
// It returns true if the container is valid, else returns false.
func GetContainerByIndex(containers []v1.Container, statuses []v1.ContainerStatus, idx int) (v1.Container, bool) {
	if idx < 0 || idx >= len(containers) || idx >= len(statuses) {
		return v1.Container{}, false
	}
	if statuses[idx].Name != containers[idx].Name {
		return v1.Container{}, false
	}
	return containers[idx], true
}
