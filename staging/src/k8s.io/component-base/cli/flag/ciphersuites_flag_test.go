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

package flag

import (
	"crypto/tls"
	"go/importer"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

func TestStrToUInt16(t *testing.T) {
	tests := []struct {
		flag           []string
		expected       []uint16
		expected_error bool
	}{
		{
			// Happy case
			flag:           []string{"TLS_RSA_WITH_RC4_128_SHA", "TLS_RSA_WITH_AES_128_CBC_SHA", "TLS_ECDHE_RSA_WITH_RC4_128_SHA", "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA"},
			expected:       []uint16{tls.TLS_RSA_WITH_RC4_128_SHA, tls.TLS_RSA_WITH_AES_128_CBC_SHA, tls.TLS_ECDHE_RSA_WITH_RC4_128_SHA, tls.TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA},
			expected_error: false,
		},
		{
			// One flag only
			flag:           []string{"TLS_RSA_WITH_RC4_128_SHA"},
			expected:       []uint16{tls.TLS_RSA_WITH_RC4_128_SHA},
			expected_error: false,
		},
		{
			// Empty flag
			flag:           []string{},
			expected:       nil,
			expected_error: false,
		},
		{
			// Duplicated flag
			flag:           []string{"TLS_RSA_WITH_RC4_128_SHA", "TLS_RSA_WITH_AES_128_CBC_SHA", "TLS_ECDHE_RSA_WITH_RC4_128_SHA", "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA", "TLS_RSA_WITH_RC4_128_SHA"},
			expected:       []uint16{tls.TLS_RSA_WITH_RC4_128_SHA, tls.TLS_RSA_WITH_AES_128_CBC_SHA, tls.TLS_ECDHE_RSA_WITH_RC4_128_SHA, tls.TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA, tls.TLS_RSA_WITH_RC4_128_SHA},
			expected_error: false,
		},
		{
			// Invalid flag
			flag:           []string{"foo"},
			expected:       nil,
			expected_error: true,
		},
	}

	for i, test := range tests {
		uIntFlags, err := TLSCipherSuites(test.flag)
		if reflect.DeepEqual(uIntFlags, test.expected) == false {
			t.Errorf("%d: expected %+v, got %+v", i, test.expected, uIntFlags)
		}
		if test.expected_error && err == nil {
			t.Errorf("%d: expecting error, got %+v", i, err)
		}
	}
}

func TestConstantMaps(t *testing.T) {
	pkg, err := importer.Default().Import("crypto/tls")
	if err != nil {
		t.Logf("Cannot parse go source to verify TLS constants (this is expected in bazel environments): %v", err)
		// if we can't parse the go source (like in bazel envs),
		// at least verify we're on a go version we've manually tested
		manuallyVerifiedVersion := "go1.12"
		currentVersion := runtime.Version()
		if strings.HasPrefix(currentVersion, "go1.") {
			currentMajorMinor := strings.Join(strings.Split(currentVersion, ".")[0:2], ".")
			if currentMajorMinor != manuallyVerifiedVersion {
				t.Errorf("Manually verified version (%q) does not match current environment (%q)", manuallyVerifiedVersion, currentMajorMinor)
				t.Errorf("Run `go test ./vendor/k8s.io/component-base/cli/flag` with %s, fix any issues found, then update manuallyVerifiedVersion to %s", currentMajorMinor, currentMajorMinor)
			} else {
				t.Logf("current version of go (%q) matches manually verified version (%q)", currentMajorMinor, manuallyVerifiedVersion)
			}
		} else {
			t.Logf("current version of go (%q) does not contain major/minor version", currentVersion)
		}
		t.SkipNow()
	}
	discoveredVersions := map[string]bool{}
	discoveredCiphers := map[string]bool{}
	for _, declName := range pkg.Scope().Names() {
		if strings.HasPrefix(declName, "VersionTLS") {
			discoveredVersions[declName] = true
		}
		if strings.HasPrefix(declName, "TLS_RSA_") || strings.HasPrefix(declName, "TLS_ECDHE_") {
			discoveredCiphers[declName] = true
		}
	}

	for k := range discoveredCiphers {
		if _, ok := ciphers[k]; !ok {
			t.Errorf("discovered cipher tls.%s not in ciphers map", k)
		}
	}
	for k := range ciphers {
		if _, ok := discoveredCiphers[k]; !ok {
			t.Errorf("ciphers map has %s not in tls package", k)
		}
	}
	for k := range discoveredVersions {
		if _, ok := versions[k]; !ok {
			t.Errorf("discovered version tls.%s not in version map", k)
		}
	}
	for k := range versions {
		if _, ok := discoveredVersions[k]; !ok {
			t.Errorf("versions map has %s not in tls package", k)
		}
	}
}
