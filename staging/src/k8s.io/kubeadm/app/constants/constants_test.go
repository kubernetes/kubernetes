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

	"k8s.io/kubernetes/pkg/util/version"
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

func TestGetNodeBootstrapTokenAuthGroup(t *testing.T) {
	var tests = []struct {
		k8sVersion, expected string
	}{
		{
			k8sVersion: "v1.7.0",
			expected:   "system:bootstrappers",
		},
		{
			k8sVersion: "v1.7.8",
			expected:   "system:bootstrappers",
		},
		{
			k8sVersion: "v1.8.0-alpha.3",
			expected:   "system:bootstrappers",
		},
		{
			k8sVersion: "v1.8.0-beta.0",
			expected:   "system:bootstrappers:kubeadm:default-node-token",
		},
		{
			k8sVersion: "v1.8.0-rc.1",
			expected:   "system:bootstrappers:kubeadm:default-node-token",
		},
		{
			k8sVersion: "v1.8.0",
			expected:   "system:bootstrappers:kubeadm:default-node-token",
		},
		{
			k8sVersion: "v1.8.9",
			expected:   "system:bootstrappers:kubeadm:default-node-token",
		},
	}
	for _, rt := range tests {
		actual := GetNodeBootstrapTokenAuthGroup(version.MustParseSemantic(rt.k8sVersion))
		if actual != rt.expected {
			t.Errorf(
				"failed GetNodeBootstrapTokenAuthGroup:\n\texpected: %s\n\t  actual: %s",
				rt.expected,
				actual,
			)
		}
	}
}
