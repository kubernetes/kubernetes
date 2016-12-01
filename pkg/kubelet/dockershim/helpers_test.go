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
	"github.com/stretchr/testify/require"

	"k8s.io/kubernetes/pkg/api/v1"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
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
	makeConfig := func(annotations map[string]string) *runtimeapi.PodSandboxConfig {
		return makeSandboxConfigWithLabelsAndAnnotations("pod", "ns", "1234", 1, nil, annotations)
	}

	tests := []struct {
		msg          string
		config       *runtimeapi.PodSandboxConfig
		expectedOpts []string
	}{{
		msg:          "No security annotations",
		config:       makeConfig(nil),
		expectedOpts: []string{"seccomp=unconfined"},
	}, {
		msg: "Seccomp unconfined",
		config: makeConfig(map[string]string{
			v1.SeccompContainerAnnotationKeyPrefix + containerName: "unconfined",
		}),
		expectedOpts: []string{"seccomp=unconfined"},
	}, {
		msg: "Seccomp default",
		config: makeConfig(map[string]string{
			v1.SeccompContainerAnnotationKeyPrefix + containerName: "docker/default",
		}),
		expectedOpts: nil,
	}, {
		msg: "Seccomp pod default",
		config: makeConfig(map[string]string{
			v1.SeccompPodAnnotationKey: "docker/default",
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
			v1.SeccompContainerAnnotationKeyPrefix + containerName: "docker/default",
			apparmor.ContainerAnnotationKeyPrefix + containerName:  apparmor.ProfileNamePrefix + "foo",
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
	makeConfig := func(annotations map[string]string) *runtimeapi.PodSandboxConfig {
		return makeSandboxConfigWithLabelsAndAnnotations("pod", "ns", "1234", 1, nil, annotations)
	}

	tests := []struct {
		msg          string
		config       *runtimeapi.PodSandboxConfig
		expectedOpts []string
	}{{
		msg:          "No security annotations",
		config:       makeConfig(nil),
		expectedOpts: []string{"seccomp=unconfined"},
	}, {
		msg: "Seccomp default",
		config: makeConfig(map[string]string{
			v1.SeccompPodAnnotationKey: "docker/default",
		}),
		expectedOpts: nil,
	}, {
		msg: "Seccomp unconfined",
		config: makeConfig(map[string]string{
			v1.SeccompPodAnnotationKey: "unconfined",
		}),
		expectedOpts: []string{"seccomp=unconfined"},
	}, {
		msg: "Seccomp pod and container profile",
		config: makeConfig(map[string]string{
			v1.SeccompContainerAnnotationKeyPrefix + "test-container": "unconfined",
			v1.SeccompPodAnnotationKey:                                "docker/default",
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
			v1.SysctlsPodAnnotationKey:       "kernel.shmmni=32768,kernel.shmmax=1000000000",
			v1.UnsafeSysctlsPodAnnotationKey: "knet.ipv4.route.min_pmtu=1000",
		},
		expectedSysctls: map[string]string{
			"kernel.shmmni":            "32768",
			"kernel.shmmax":            "1000000000",
			"knet.ipv4.route.min_pmtu": "1000",
		},
	}, {
		annotations: map[string]string{
			v1.SysctlsPodAnnotationKey: "kernel.shmmni=32768,kernel.shmmax=1000000000",
		},
		expectedSysctls: map[string]string{
			"kernel.shmmni": "32768",
			"kernel.shmmax": "1000000000",
		},
	}, {
		annotations: map[string]string{
			v1.UnsafeSysctlsPodAnnotationKey: "knet.ipv4.route.min_pmtu=1000",
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

// TestGetUserFromImageUser tests the logic of getting image uid or user name of image user.
func TestGetUserFromImageUser(t *testing.T) {
	newI64 := func(i int64) *int64 { return &i }
	newStr := func(s string) *string { return &s }
	for c, test := range map[string]struct {
		user string
		uid  *int64
		name *string
	}{
		"no gid": {
			user: "0",
			uid:  newI64(0),
		},
		"uid/gid": {
			user: "0:1",
			uid:  newI64(0),
		},
		"empty user": {
			user: "",
		},
		"multiple spearators": {
			user: "1:2:3",
			uid:  newI64(1),
		},
		"root username": {
			user: "root:root",
			name: newStr("root"),
		},
		"username": {
			user: "test:test",
			name: newStr("test"),
		},
	} {
		t.Logf("TestCase - %q", c)
		actualUID, actualName := getUserFromImageUser(test.user)
		assert.Equal(t, test.uid, actualUID)
		assert.Equal(t, test.name, actualName)
	}
}

func TestParsingCreationConflictError(t *testing.T) {
	// Expected error message from docker.
	msg := "Conflict. The name \"/k8s_POD_pfpod_e2e-tests-port-forwarding-dlxt2_81a3469e-99e1-11e6-89f2-42010af00002_0\" is already in use by container 24666ab8c814d16f986449e504ea0159468ddf8da01897144a770f66dce0e14e. You have to remove (or rename) that container to be able to reuse that name."

	matches := conflictRE.FindStringSubmatch(msg)
	require.Len(t, matches, 2)
	require.Equal(t, matches[1], "24666ab8c814d16f986449e504ea0159468ddf8da01897144a770f66dce0e14e")
}
