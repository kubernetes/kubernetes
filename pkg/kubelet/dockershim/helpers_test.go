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

package dockershim

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"k8s.io/kubernetes/pkg/api"
	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	"k8s.io/kubernetes/pkg/security/apparmor"
)

func TestLabelsAndAnnotationsRoundTrip(t *testing.T) {
	expectedLabels := map[string]string{"foo.123.abc": "baz", "bar.456.xyz": "qwe"}
	expectedAnnotations := map[string]string{"uio.ert": "dfs", "jkl": "asd"}
	// Merge labels and annotations into docker labels.
	dockerLabels := makeLabels(expectedLabels, expectedAnnotations)
	// Extract labels and annotations from docker labels.
	actualLabels, actualAnnotations := extractLabels(dockerLabels)
	assert.Equal(t, expectedLabels, actualLabels)
	assert.Equal(t, expectedAnnotations, actualAnnotations)
}

// TestGetContainerSecurityOpts tests the logic of generating container security options from sandbox annotations.
// The actual profile loading logic is tested in dockertools.
// TODO: Migrate the corresponding test to dockershim.
func TestGetContainerSecurityOpts(t *testing.T) {
	containerName := "bar"
	makeConfig := func(annotations map[string]string) *runtimeApi.PodSandboxConfig {
		return makeSandboxConfigWithLabelsAndAnnotations("pod", "ns", "1234", 1, nil, annotations)
	}

	tests := []struct {
		msg          string
		config       *runtimeApi.PodSandboxConfig
		expectedOpts []string
	}{{
		msg:          "No security annotations",
		config:       makeConfig(nil),
		expectedOpts: []string{"seccomp=unconfined"},
	}, {
		msg: "Seccomp unconfined",
		config: makeConfig(map[string]string{
			api.SeccompContainerAnnotationKeyPrefix + containerName: "unconfined",
		}),
		expectedOpts: []string{"seccomp=unconfined"},
	}, {
		msg: "Seccomp default",
		config: makeConfig(map[string]string{
			api.SeccompContainerAnnotationKeyPrefix + containerName: "docker/default",
		}),
		expectedOpts: nil,
	}, {
		msg: "Seccomp pod default",
		config: makeConfig(map[string]string{
			api.SeccompPodAnnotationKey: "docker/default",
		}),
		expectedOpts: nil,
	}, {
		msg: "AppArmor runtime/default",
		config: makeConfig(map[string]string{
			apparmor.ContainerAnnotationKeyPrefix + containerName: apparmor.ProfileRuntimeDefault,
		}),
		expectedOpts: []string{"seccomp=unconfined"},
	}, {
		msg: "AppArmor local profile",
		config: makeConfig(map[string]string{
			apparmor.ContainerAnnotationKeyPrefix + containerName: apparmor.ProfileNamePrefix + "foo",
		}),
		expectedOpts: []string{"seccomp=unconfined", "apparmor=foo"},
	}, {
		msg: "AppArmor and seccomp profile",
		config: makeConfig(map[string]string{
			api.SeccompContainerAnnotationKeyPrefix + containerName: "docker/default",
			apparmor.ContainerAnnotationKeyPrefix + containerName:   apparmor.ProfileNamePrefix + "foo",
		}),
		expectedOpts: []string{"apparmor=foo"},
	}}

	for i, test := range tests {
		opts, err := getContainerSecurityOpts(containerName, test.config, "test/seccomp/profile/root")
		assert.NoError(t, err, "TestCase[%d]: %s", i, test.msg)
		assert.Len(t, opts, len(test.expectedOpts), "TestCase[%d]: %s", i, test.msg)
		for _, opt := range test.expectedOpts {
			assert.Contains(t, opts, opt, "TestCase[%d]: %s", i, test.msg)
		}
	}
}

// TestGetSandboxSecurityOpts tests the logic of generating sandbox security options from sandbox annotations.
func TestGetSandboxSecurityOpts(t *testing.T) {
	makeConfig := func(annotations map[string]string) *runtimeApi.PodSandboxConfig {
		return makeSandboxConfigWithLabelsAndAnnotations("pod", "ns", "1234", 1, nil, annotations)
	}

	tests := []struct {
		msg          string
		config       *runtimeApi.PodSandboxConfig
		expectedOpts []string
	}{{
		msg:          "No security annotations",
		config:       makeConfig(nil),
		expectedOpts: []string{"seccomp=unconfined"},
	}, {
		msg: "Seccomp default",
		config: makeConfig(map[string]string{
			api.SeccompPodAnnotationKey: "docker/default",
		}),
		expectedOpts: nil,
	}, {
		msg: "Seccomp unconfined",
		config: makeConfig(map[string]string{
			api.SeccompPodAnnotationKey: "unconfined",
		}),
		expectedOpts: []string{"seccomp=unconfined"},
	}, {
		msg: "Seccomp pod and container profile",
		config: makeConfig(map[string]string{
			api.SeccompContainerAnnotationKeyPrefix + "test-container": "unconfined",
			api.SeccompPodAnnotationKey:                                "docker/default",
		}),
		expectedOpts: nil,
	}}

	for i, test := range tests {
		opts, err := getSandboxSecurityOpts(test.config, "test/seccomp/profile/root")
		assert.NoError(t, err, "TestCase[%d]: %s", i, test.msg)
		assert.Len(t, opts, len(test.expectedOpts), "TestCase[%d]: %s", i, test.msg)
		for _, opt := range test.expectedOpts {
			assert.Contains(t, opts, opt, "TestCase[%d]: %s", i, test.msg)
		}
	}
}

// TestGetSystclsFromAnnotations tests the logic of getting sysctls from annotations.
func TestGetSystclsFromAnnotations(t *testing.T) {
	tests := []struct {
		annotations     map[string]string
		expectedSysctls map[string]string
	}{{
		annotations: map[string]string{
			api.SysctlsPodAnnotationKey:       "kernel.shmmni=32768,kernel.shmmax=1000000000",
			api.UnsafeSysctlsPodAnnotationKey: "knet.ipv4.route.min_pmtu=1000",
		},
		expectedSysctls: map[string]string{
			"kernel.shmmni":            "32768",
			"kernel.shmmax":            "1000000000",
			"knet.ipv4.route.min_pmtu": "1000",
		},
	}, {
		annotations: map[string]string{
			api.SysctlsPodAnnotationKey: "kernel.shmmni=32768,kernel.shmmax=1000000000",
		},
		expectedSysctls: map[string]string{
			"kernel.shmmni": "32768",
			"kernel.shmmax": "1000000000",
		},
	}, {
		annotations: map[string]string{
			api.UnsafeSysctlsPodAnnotationKey: "knet.ipv4.route.min_pmtu=1000",
		},
		expectedSysctls: map[string]string{
			"knet.ipv4.route.min_pmtu": "1000",
		},
	}}

	for i, test := range tests {
		actual, err := getSysctlsFromAnnotations(test.annotations)
		assert.NoError(t, err, "TestCase[%d]", i)
		assert.Len(t, actual, len(test.expectedSysctls), "TestCase[%d]", i)
		assert.Equal(t, test.expectedSysctls, actual, "TestCase[%d]", i)
	}
}
