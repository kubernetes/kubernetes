/*
Copyright 2014 The Kubernetes Authors.

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
)

const (
	kubeReleaseBucketURL    = "https://storage.googleapis.com/kubernetes-release"
	kubeDevReleaseBucketURL = "https://storage.googleapis.com/kubernetes-release-dev"
)

var (
	kubeReleaseRegex        = regexp.MustCompile(`^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)([-0-9a-zA-Z_\.+]*)?$`)
	kubeReleaseLabelRegex   = regexp.MustCompile(`^release/[-\w_\.]+`)
	kubeCiReleaseLabelRegex = regexp.MustCompile(`^ci/[-\w_\.]+`)
)

// KubernetesReleaseVersion is helper function that can fetch
// from release servers based on label names, like "release/stable"
// or "release/latest".
// If argument is already well formatted version string, it
// will return same string.
// In case of symbolic names, it tries to fetch from release
// servers and then return actual build ID
//
// Available names on release servers:
//  release/stable      (latest stable release)
//  release/stable-1    (latest stable release in 1.x)
//  release/stable-1.0  (and similarly 1.1, 1.2, 1.3, ...)
//  release/latest      (latest release, including alpha/beta)
//  release/latest-1    (latest release in 1.x, including alpha/beta)
//  release/latest-1.0  (and similarly 1.1, 1.2, 1.3, ...)
//  ci/latest
//  ci/latest-green
//  ci/latest-1
//  ci/latest-1.0       (and similarly 1.1, 1.2, 1.3, ...)
func KubernetesReleaseVersion(version string) (string, error) {
	var url string

	if kubeReleaseRegex.MatchString(version) {
		return version, nil
	} else if kubeReleaseLabelRegex.MatchString(version) {
		url = fmt.Sprintf("%s/%s.txt", kubeReleaseBucketURL, version)
	} else if kubeCiReleaseLabelRegex.MatchString(version) {
		url = fmt.Sprintf("%s/%s.txt", kubeDevReleaseBucketURL, version)
	} else {
		return "", fmt.Errorf("Error in version string %q", version)
	}
	ver, err := fetchVersion(url)
	if err != nil {
		return "", err
	}
	return KubernetesReleaseVersion(ver)
}

// fetchVersion simple helper to fech content
// of URL and remove leading and trailing whitespaces
func fetchVersion(url string) (string, error) {
	resp, err := http.Get(url)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	return strings.Trim(string(body), " \t\n"), nil
}
