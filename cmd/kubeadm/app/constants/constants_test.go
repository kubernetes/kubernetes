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

package constants

import (
	"testing"
)

func TestGetStaticPodDirectory(t *testing.T) {
	expected := "/etc/kubernetes/manifests"
	actual := GetStaticPodDirectory()

	if actual != expected {
		t.Errorf(
			"failed GetStaticPodDirectory:\n\texpected: %s\n\t  actual: %s",
			expected,
			actual,
		)
	}
}

func TestGetAdminKubeConfigPath(t *testing.T) {
	expected := "/etc/kubernetes/admin.conf"
	actual := GetAdminKubeConfigPath()

	if actual != expected {
		t.Errorf(
			"failed GetAdminKubeConfigPath:\n\texpected: %s\n\t  actual: %s",
			expected,
			actual,
		)
	}
}

func TestGetStaticPodFilepath(t *testing.T) {
	var tests = []struct {
		componentName, manifestsDir, expected string
	}{
		{
			componentName: "kube-apiserver",
			manifestsDir:  "/etc/kubernetes/manifests",
			expected:      "/etc/kubernetes/manifests/kube-apiserver.yaml",
		},
		{
			componentName: "kube-controller-manager",
			manifestsDir:  "/etc/kubernetes/manifests/",
			expected:      "/etc/kubernetes/manifests/kube-controller-manager.yaml",
		},
		{
			componentName: "foo",
			manifestsDir:  "/etc/bar/",
			expected:      "/etc/bar/foo.yaml",
		},
	}
	for _, rt := range tests {
		actual := GetStaticPodFilepath(rt.componentName, rt.manifestsDir)
		if actual != rt.expected {
			t.Errorf(
				"failed GetStaticPodFilepath:\n\texpected: %s\n\t  actual: %s",
				rt.expected,
				actual,
			)
		}
	}
}

func TestAddSelfHostedPrefix(t *testing.T) {
	var tests = []struct {
		componentName, expected string
	}{
		{
			componentName: "kube-apiserver",
			expected:      "self-hosted-kube-apiserver",
		},
		{
			componentName: "kube-controller-manager",
			expected:      "self-hosted-kube-controller-manager",
		},
		{
			componentName: "kube-scheduler",
			expected:      "self-hosted-kube-scheduler",
		},
		{
			componentName: "foo",
			expected:      "self-hosted-foo",
		},
	}
	for _, rt := range tests {
		actual := AddSelfHostedPrefix(rt.componentName)
		if actual != rt.expected {
			t.Errorf(
				"failed AddSelfHostedPrefix:\n\texpected: %s\n\t  actual: %s",
				rt.expected,
				actual,
			)
		}
	}
}

func TestGetDefaultEtcdVersion(t *testing.T) {
	var tests = []struct {
		k8sversion, expected string
	}{
		{
			// unknown version. Should return current default
			k8sversion: "",
			expected:   DefaultEtcdVersion,
		},
		{
			// non-semantic version. Should return current default
			k8sversion: "foo",
			expected:   DefaultEtcdVersion,
		},
		{
			// same major version, but way higher minor version. Should return current default
			k8sversion: "v1.99999.0",
			expected:   DefaultEtcdVersion,
		},
		{
			// version way higher than current. Should return current default
			k8sversion: "v10.2.3",
			expected:   DefaultEtcdVersion,
		},
		{
			// version lower than current. Should return etcd version for previous releases
			k8sversion: "v1.0.0",
			expected:   DefaultEtcdVersionForPreviousKubernetesRelease,
		},
	}
	for _, rt := range tests {
		actual := GetDefaultEtcdVersion(rt.k8sversion)
		if actual != rt.expected {
			t.Errorf(
				"failed GetDefaultEtcdVersion:\n\tk8sversion: %s\n\texpected: %s\n\t  actual: %s",
				rt.k8sversion,
				rt.expected,
				actual,
			)
		}
	}
}
