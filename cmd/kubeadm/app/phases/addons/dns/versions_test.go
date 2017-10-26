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

package dns

import (
	"testing"

	"k8s.io/kubernetes/pkg/util/version"
)

func TestGetKubeDNSVersion(t *testing.T) {
	var tests = []struct {
		k8sVersion, expected string
	}{
		{
			k8sVersion: "v1.7.0",
			expected:   "1.14.5",
		},
		{
			k8sVersion: "v1.7.1",
			expected:   "1.14.5",
		},
		{
			k8sVersion: "v1.7.2",
			expected:   "1.14.5",
		},
		{
			k8sVersion: "v1.7.3",
			expected:   "1.14.5",
		},
		{
			k8sVersion: "v1.8.0-alpha.2",
			expected:   "1.14.5",
		},
		{
			k8sVersion: "v1.8.0",
			expected:   "1.14.5",
		},
		{
			k8sVersion: "v1.9.0",
			expected:   "1.14.7",
		},
	}
	for _, rt := range tests {

		k8sVersion, err := version.ParseSemantic(rt.k8sVersion)
		if err != nil {
			t.Fatalf("couldn't parse kubernetes version %q: %v", rt.k8sVersion, err)
		}

		actualDNSVersion := GetKubeDNSVersion(k8sVersion)
		if actualDNSVersion != rt.expected {
			t.Errorf(
				"failed GetKubeDNSVersion:\n\texpected: %s\n\t  actual: %s",
				rt.expected,
				actualDNSVersion,
			)
		}
	}
}

func TestGetKubeDNSProbeType(t *testing.T) {
	var tests = []struct {
		k8sVersion, expected string
	}{
		{
			k8sVersion: "v1.7.0",
			expected:   "A",
		},
		{
			k8sVersion: "v1.7.1",
			expected:   "A",
		},
		{
			k8sVersion: "v1.7.2",
			expected:   "A",
		},
		{
			k8sVersion: "v1.7.3",
			expected:   "A",
		},
		{
			k8sVersion: "v1.8.0-alpha.2",
			expected:   "A",
		},
		{
			k8sVersion: "v1.8.0",
			expected:   "A",
		},
		{
			k8sVersion: "v1.9.0",
			expected:   "SRV",
		},
	}
	for _, rt := range tests {

		k8sVersion, err := version.ParseSemantic(rt.k8sVersion)
		if err != nil {
			t.Fatalf("couldn't parse kubernetes version %q: %v", rt.k8sVersion, err)
		}

		actualDNSProbeType := GetKubeDNSProbeType(k8sVersion)
		if actualDNSProbeType != rt.expected {
			t.Errorf(
				"failed GetKubeDNSProbeType:\n\texpected: %s\n\t  actual: %s",
				rt.expected,
				actualDNSProbeType,
			)
		}
	}
}
