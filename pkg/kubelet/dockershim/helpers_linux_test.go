// +build linux

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

package dockershim

import (
	"fmt"
	"path"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/api/core/v1"

	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/v1alpha1/runtime"
)

func TestGetSeccompSecurityOpts(t *testing.T) {
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
	}}

	for i, test := range tests {
		opts, err := getSeccompSecurityOpts(containerName, test.config, "test/seccomp/profile/root", '=')
		assert.NoError(t, err, "TestCase[%d]: %s", i, test.msg)
		assert.Len(t, opts, len(test.expectedOpts), "TestCase[%d]: %s", i, test.msg)
		for _, opt := range test.expectedOpts {
			assert.Contains(t, opts, opt, "TestCase[%d]: %s", i, test.msg)
		}
	}
}

func TestLoadSeccompLocalhostProfiles(t *testing.T) {
	containerName := "bar"
	makeConfig := func(annotations map[string]string) *runtimeapi.PodSandboxConfig {
		return makeSandboxConfigWithLabelsAndAnnotations("pod", "ns", "1234", 1, nil, annotations)
	}

	tests := []struct {
		msg          string
		config       *runtimeapi.PodSandboxConfig
		expectedOpts []string
		expectErr    bool
	}{{
		msg: "Seccomp localhost/test profile",
		config: makeConfig(map[string]string{
			v1.SeccompPodAnnotationKey: "localhost/test",
		}),
		expectedOpts: []string{`seccomp={"foo":"bar"}`},
		expectErr:    false,
	}, {
		msg: "Seccomp localhost/sub/subtest profile",
		config: makeConfig(map[string]string{
			v1.SeccompPodAnnotationKey: "localhost/sub/subtest",
		}),
		expectedOpts: []string{`seccomp={"abc":"def"}`},
		expectErr:    false,
	}, {
		msg: "Seccomp non-existent",
		config: makeConfig(map[string]string{
			v1.SeccompPodAnnotationKey: "localhost/non-existent",
		}),
		expectedOpts: nil,
		expectErr:    true,
	}}

	profileRoot := path.Join("fixtures", "seccomp")
	for i, test := range tests {
		opts, err := getSeccompSecurityOpts(containerName, test.config, profileRoot, '=')
		if test.expectErr {
			assert.Error(t, err, fmt.Sprintf("TestCase[%d]: %s", i, test.msg))
			continue
		}
		assert.NoError(t, err, "TestCase[%d]: %s", i, test.msg)
		assert.Len(t, opts, len(test.expectedOpts), "TestCase[%d]: %s", i, test.msg)
		for _, opt := range test.expectedOpts {
			assert.Contains(t, opts, opt, "TestCase[%d]: %s", i, test.msg)
		}
	}
}
