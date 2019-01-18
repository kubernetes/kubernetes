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
	"net/url"
	"regexp"
	"strings"
	"time"

	"github.com/pkg/errors"

	netutil "k8s.io/apimachinery/pkg/util/net"
	versionutil "k8s.io/apimachinery/pkg/util/version"
	"k8s.io/klog"
	pkgversion "k8s.io/kubernetes/pkg/version"
)

const (
	getReleaseVersionTimeout = time.Duration(10 * time.Second)
	// Max number of attempts allowed to resolve recursive version labels
	// like "stable" -> "stable-1" -> parsable version.
	maxVersionLabelResolveAttempts = 5
)

var (
	kubeReleaseBucketURL  = "https://dl.k8s.io"
	kubeReleaseRegex      = regexp.MustCompile(`^v?(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)([-0-9a-zA-Z_\.+]*)?$`)
	kubeReleaseLabelRegex = regexp.MustCompile(`^[[:lower:]]+(-[-\w_\.]+)?$`)
	kubeBucketPrefixes    = regexp.MustCompile(`^((release|ci|ci-cross)/)?([-\w_\.+]+)$`)
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
//
// Also the names and versions can be prefixed with a bucket; valid formats include
// ci/1.13.0 or ci/v1.13.0 or ci/stable. Valid prefixes are release, ci, ci-cross.
func KubernetesReleaseVersion(version string) (string, error) {
	bucketURL, versionLabel, err := splitVersion(version)
	if err != nil {
		return "", err
	}

	// if versionLabel is a semantic version, use it; valid formats include
	//  - 1.13.0 or v1.13.0 (version with or without initial v)
	if ver, ok := normalizeBuildVersion(versionLabel); ok {
		return ver, nil
	}

	// kubeReleaseLabelRegex matches labels such as: latest, latest-1, latest-1.10
	if !kubeReleaseLabelRegex.MatchString(versionLabel) {
		return "", errors.Errorf("version %q doesn't match patterns for neither semantic version nor labels (stable, latest, ...)", version)
	}

	// Try to obtain a client version.
	// pkgversion.Get().String() should always return a correct version added by the golang
	// linker and the build system. The version can still be missing when doing unit tests
	// on individual packages.
	clientVersion, clientVersionErr := kubeadmVersion(pkgversion.Get().String())
	// Fetch version from the internet.
	remoteVersion, err := resolveVersionLabel(bucketURL, versionLabel)
	if err != nil {
		if urlErr, ok := errors.Cause(err).(*url.Error); !(ok && urlErr.Timeout()) {
			return "", errors.Wrapf(err, "failed to resolve version label while having good connectivity")
		}

		// Handle air-gapped environments by falling back to the client version.
		klog.Warningf("could not fetch a Kubernetes version from the internet: %v", err)
		klog.Warningf("falling back to the local client version: %s", clientVersion)
		return clientVersion, nil
	}

	if clientVersionErr != nil {
		klog.Warningf("could not obtain client version; using remote version: %s", remoteVersion)
		return remoteVersion, nil
	}

	// both the client and the remote version are obtained; validate them and pick a stable version
	ver, err := validateStableVersion(remoteVersion, clientVersion)
	if err != nil {
		return "", err
	}

	if kubeReleaseLabelRegex.MatchString(ver) {
		return resolveVersionLabel(bucketURL, ver)
	}

	return ver, nil
}

// KubernetesVersionToImageTag is helper function that replaces all
// non-allowed symbols in tag strings with underscores.
// Image tag can only contain lowercase and uppercase letters, digits,
// underscores, periods and dashes.
// Current usage is for CI images where all of symbols except '+' are valid,
// but function is for generic usage where input can't be always pre-validated.
func KubernetesVersionToImageTag(version string) string {
	allowed := regexp.MustCompile(`[^-a-zA-Z0-9_\.]`)
	return allowed.ReplaceAllString(version, "_")
}

// KubernetesIsCIVersion checks if user requested CI version
func KubernetesIsCIVersion(version string) bool {
	subs := kubeBucketPrefixes.FindAllStringSubmatch(version, 1)
	if len(subs) == 1 && len(subs[0]) == 4 && strings.HasPrefix(subs[0][2], "ci") {
		return true
	}
	return false
}

// normalizeBuildVersion adds the "v" prefix to version string if it's missing
// If input doesn't match known version pattern returns empty string and false.
func normalizeBuildVersion(version string) (string, bool) {
	if !kubeReleaseRegex.MatchString(version) {
		return "", false
	}
	if strings.HasPrefix(version, "v") {
		return version, true
	}
	return "v" + version, true
}

// Internal helper: split version parts,
// Return base URL and cleaned-up version
func splitVersion(version string) (string, string, error) {
	var urlSuffix string
	subs := kubeBucketPrefixes.FindAllStringSubmatch(version, 1)
	if len(subs) != 1 || len(subs[0]) != 4 {
		return "", "", errors.Errorf("invalid version %q", version)
	}

	switch {
	case strings.HasPrefix(subs[0][2], "ci"):
		// Just use whichever the user specified
		urlSuffix = subs[0][2]
	default:
		urlSuffix = "release"
	}
	baseURL := fmt.Sprintf("%s/%s", kubeReleaseBucketURL, urlSuffix)
	return baseURL, subs[0][3], nil
}

// resolveVersionLabel resolves version label into normalized version string
func resolveVersionLabel(bucketURL, ver string) (string, error) {
	client := &http.Client{Timeout: getReleaseVersionTimeout, Transport: netutil.SetOldTransportDefaults(&http.Transport{})}
	resolutionSequence := make([]string, 0, maxVersionLabelResolveAttempts)

	// try to resolve recursive labels like "stable" -> "stable-1" -> parsable version
	for attempts := 0; attempts < maxVersionLabelResolveAttempts; attempts++ {
		resolutionSequence = append(resolutionSequence, ver)
		versionURL := fmt.Sprintf("%s/%s.txt", bucketURL, ver)
		klog.V(2).Infof("fetching Kubernetes version from URL: %s", versionURL)
		resp, err := client.Get(versionURL)
		if err != nil {
			return "", errors.Wrapf(err, "unable to get URL %q", versionURL)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			return "", errors.Errorf("unable to fetch file. URL: %q, status: %v", versionURL, resp.Status)
		}

		body, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			return "", errors.Wrapf(err, "unable to read content of URL %q", versionURL)
		}
		ver = strings.TrimSpace(string(body))

		if remoteVer, ok := normalizeBuildVersion(ver); ok {
			return remoteVer, nil
		}

		if !kubeReleaseLabelRegex.MatchString(ver) {
			return "", errors.Errorf("server returned unrecognizable string: %s", ver)
		}
	}

	return "", errors.Errorf("exhausted all attempts to resolve recursive remote version. Resolution sequence was %s", strings.Join(resolutionSequence, " -> "))
}

// kubeadmVersion returns the version of the client without metadata.
func kubeadmVersion(info string) (string, error) {
	v, err := versionutil.ParseSemantic(info)
	if err != nil {
		return "", errors.Wrap(err, "kubeadm version error")
	}
	// There is no utility in versionutil to get the version without the metadata,
	// so this needs some manual formatting.
	// Discard offsets after a release label and keep the labels down to e.g. `alpha.0` instead of
	// including the offset e.g. `alpha.0.206`. This is done to comply with GCR image tags.
	pre := v.PreRelease()
	patch := v.Patch()
	if len(pre) > 0 {
		if patch > 0 {
			// If the patch version is more than zero, decrement it and remove the label.
			// this is done to comply with the latest stable patch release.
			patch = patch - 1
			pre = ""
		} else {
			split := strings.Split(pre, ".")
			if len(split) > 2 {
				pre = split[0] + "." + split[1] // Exclude the third element
			} else if len(split) < 2 {
				pre = split[0] + ".0" // Append .0 to a partial label
			}
			pre = "-" + pre
		}
	}
	vStr := fmt.Sprintf("v%d.%d.%d%s", v.Major(), v.Minor(), patch, pre)
	return vStr, nil
}

// Validate if the remote version is one Minor release newer than the client version.
// This is done to conform with "stable-X" and only allow remote versions from
// the same Patch level release.
func validateStableVersion(remoteVersion, clientVersion string) (string, error) {
	verRemote, err := versionutil.ParseGeneric(remoteVersion)
	if err != nil {
		return "", errors.Wrap(err, "remote version error")
	}
	verClient, err := versionutil.ParseGeneric(clientVersion)
	if err != nil {
		return "", errors.Wrap(err, "client version error")
	}
	// If the remote Major version is bigger or if the Major versions are the same,
	// but the remote Minor is bigger use the client version release. This handles Major bumps too.
	if verClient.Major() < verRemote.Major() ||
		(verClient.Major() == verRemote.Major()) && verClient.Minor() < verRemote.Minor() {
		estimatedRelease := fmt.Sprintf("stable-%d.%d", verClient.Major(), verClient.Minor())
		klog.Infof("remote version is much newer: %s; falling back to: %s", remoteVersion, estimatedRelease)
		return estimatedRelease, nil
	}
	return remoteVersion, nil
}
