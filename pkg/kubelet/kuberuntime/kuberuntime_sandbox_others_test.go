//go:build !windows
// +build !windows

/*
Copyright 2024 The Kubernetes Authors.

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

package kuberuntime

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
)

func TestAddLinuxSecurityContext(t *testing.T) {
	lc := &runtimeapi.LinuxPodSandboxConfig{
		SecurityContext: &runtimeapi.LinuxSandboxSecurityContext{},
	}

	var uid, gid, fsgroup int64 = 1000, 1001, 2001
	pod := newTestPod()
	pod.Spec.SecurityContext = &v1.PodSecurityContext{
		RunAsUser:  &uid,
		RunAsGroup: &gid,
		FSGroup:    &fsgroup,
		SELinuxOptions: &v1.SELinuxOptions{
			User:  "foo",
			Role:  "lish",
			Type:  "epyT",
			Level: "maximum",
		},
	}

	_, _, m, err := createTestRuntimeManager()
	require.NoError(t, err)

	err = m.addLinuxSecurityContext(lc, pod)
	require.NoError(t, err)

	expectedSc := &runtimeapi.LinuxSandboxSecurityContext{
		RunAsUser:          &runtimeapi.Int64Value{Value: uid},
		RunAsGroup:         &runtimeapi.Int64Value{Value: gid},
		SupplementalGroups: []int64{fsgroup},
		SelinuxOptions: &runtimeapi.SELinuxOption{
			User:  "foo",
			Role:  "lish",
			Type:  "epyT",
			Level: "maximum",
		},
		NamespaceOptions: &runtimeapi.NamespaceOption{
			Pid: runtimeapi.NamespaceMode_CONTAINER,
		},
	}
	assert.Equal(t, expectedSc, lc.SecurityContext)
}
