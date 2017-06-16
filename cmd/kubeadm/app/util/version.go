/*
Copyright 2016 The Kubernetes Authors.

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
	"io/ioutil"
	"net/http"
	"regexp"
	"strings"

	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/pkg/util/version"
)

var (
	kubeReleaseBucketURL  = "https://storage.googleapis.com/kubernetes-release/release"
	kubeReleaseRegex      = regexp.MustCompile(`^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)([-0-9a-zA-Z_\.+]*)?$`)
	kubeReleaseLabelRegex = regexp.MustCompile(`^[[:lower:]]+(-[-\w_\.]+)?$`)
)

// KubernetesReleaseVersion is helper function that can fetch
// available version information from release servers based on
// label names, like "stable" or "latest".
//
// If argument is already semantic version string, it
// will return same string.
//
// In case of labels, it tries to fetch from release
// servers and then return actual semantic version.
//
// Available names on release servers:
//  stable      (latest stable release)
//  stable-1    (latest stable release in 1.x)
//  stable-1.0  (and similarly 1.1, 1.2, 1.3, ...)
//  latest      (latest release, including alpha/beta)
//  latest-1    (latest release in 1.x, including alpha/beta)
//  latest-1.0  (and similarly 1.1, 1.2, 1.3, ...)
func KubernetesReleaseVersion(version string) (string, error) {
	if kubeReleaseRegex.MatchString(version) {
		return version, nil
	} else if kubeReleaseLabelRegex.MatchString(version) {
		url := fmt.Sprintf("%s/%s.txt", kubeReleaseBucketURL, version)
		resp, err := http.Get(url)
		if err != nil {
			return "", fmt.Errorf("unable to get URL %q: %s", url, err.Error())
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return "", fmt.Errorf("unable to fetch release information. URL: %q Status: %v", url, resp.Status)
		}
		body, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			return "", fmt.Errorf("unable to read content of URL %q: %s", url, err.Error())
		}
		// Re-validate received version and return.
		return KubernetesReleaseVersion(strings.Trim(string(body), " \t\n"))
	}
	return "", fmt.Errorf("version %q doesn't match patterns for neither semantic version nor labels (stable, latest, ...)", version)
}

// IsNodeAuthorizerSupported returns true if the provided version of kubernetes is able to use the Node Authorizer feature.
// There is a really nasty problem with the branching here and the timing of this feature implementation. When the release-1.7 branch was
// cut, two new tags were made: v1.7.0-beta.0 and v1.8.0-alpha.0. The Node Authorizer feature merged _after those cuts_. This means the minimum
// version we have to use is v1.7.0-beta.1. BUT since v1.8.0-alpha.0 sorts higher than v1.7.0-beta.1 (the actual version gate), we have to manually
// exclude v1.8.0-alpha.0 from this condition. v1.8.0-alpha.1 will indeed contain the patch.
func IsNodeAuthorizerSupported(k8sVersion *version.Version) bool {
	return k8sVersion.AtLeast(kubeadmconstants.MinimumNodeAuthorizerVersion) && k8sVersion.String() != "1.8.0-alpha.0"
}
