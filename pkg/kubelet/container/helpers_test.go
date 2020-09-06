/*
Copyright 2015 The Kubernetes Authors.

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

package container

import (
	"reflect"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
)

func TestEnvVarsToMap(t *testing.T) {
	vars := []EnvVar{
		{
			Name:  "foo",
			Value: "bar",
		},
		{
			Name:  "zoo",
			Value: "baz",
		},
	}

	varMap := envVarsToMap(vars)

	if e, a := len(vars), len(varMap); e != a {
		t.Errorf("Unexpected map length; expected: %d, got %d", e, a)
	}

	if a := varMap["foo"]; a != "bar" {
		t.Errorf("Unexpected value of key 'foo': %v", a)
	}

	if a := varMap["zoo"]; a != "baz" {
		t.Errorf("Unexpected value of key 'zoo': %v", a)
	}
}

func TestExpandCommandAndArgs(t *testing.T) {
	cases := []struct {
		name            string
		container       *v1.Container
		envs            []EnvVar
		expectedCommand []string
		expectedArgs    []string
	}{
		{
			name:      "none",
			container: &v1.Container{},
		},
		{
			name: "command expanded",
			container: &v1.Container{
				Command: []string{"foo", "$(VAR_TEST)", "$(VAR_TEST2)"},
			},
			envs: []EnvVar{
				{
					Name:  "VAR_TEST",
					Value: "zoo",
				},
				{
					Name:  "VAR_TEST2",
					Value: "boo",
				},
			},
			expectedCommand: []string{"foo", "zoo", "boo"},
		},
		{
			name: "args expanded",
			container: &v1.Container{
				Args: []string{"zap", "$(VAR_TEST)", "$(VAR_TEST2)"},
			},
			envs: []EnvVar{
				{
					Name:  "VAR_TEST",
					Value: "hap",
				},
				{
					Name:  "VAR_TEST2",
					Value: "trap",
				},
			},
			expectedArgs: []string{"zap", "hap", "trap"},
		},
		{
			name: "both expanded",
			container: &v1.Container{
				Command: []string{"$(VAR_TEST2)--$(VAR_TEST)", "foo", "$(VAR_TEST3)"},
				Args:    []string{"foo", "$(VAR_TEST)", "$(VAR_TEST2)"},
			},
			envs: []EnvVar{
				{
					Name:  "VAR_TEST",
					Value: "zoo",
				},
				{
					Name:  "VAR_TEST2",
					Value: "boo",
				},
				{
					Name:  "VAR_TEST3",
					Value: "roo",
				},
			},
			expectedCommand: []string{"boo--zoo", "foo", "roo"},
			expectedArgs:    []string{"foo", "zoo", "boo"},
		},
	}

	for _, tc := range cases {
		actualCommand, actualArgs := ExpandContainerCommandAndArgs(tc.container, tc.envs)

		if e, a := tc.expectedCommand, actualCommand; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: unexpected command; expected %v, got %v", tc.name, e, a)
		}

		if e, a := tc.expectedArgs, actualArgs; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: unexpected args; expected %v, got %v", tc.name, e, a)
		}

	}
}

func TestExpandVolumeMountsWithSubpath(t *testing.T) {
	cases := []struct {
		name              string
		container         *v1.Container
		envs              []EnvVar
		expectedSubPath   string
		expectedMountPath string
		expectedOk        bool
	}{
		{
			name: "subpath with no expansion",
			container: &v1.Container{
				VolumeMounts: []v1.VolumeMount{{SubPathExpr: "foo"}},
			},
			expectedSubPath:   "foo",
			expectedMountPath: "",
			expectedOk:        true,
		},
		{
			name: "volumes with expanded subpath",
			container: &v1.Container{
				VolumeMounts: []v1.VolumeMount{{SubPathExpr: "foo/$(POD_NAME)"}},
			},
			envs: []EnvVar{
				{
					Name:  "POD_NAME",
					Value: "bar",
				},
			},
			expectedSubPath:   "foo/bar",
			expectedMountPath: "",
			expectedOk:        true,
		},
		{
			name: "volumes expanded with empty subpath",
			container: &v1.Container{
				VolumeMounts: []v1.VolumeMount{{SubPathExpr: ""}},
			},
			envs: []EnvVar{
				{
					Name:  "POD_NAME",
					Value: "bar",
				},
			},
			expectedSubPath:   "",
			expectedMountPath: "",
			expectedOk:        true,
		},
		{
			name: "volumes expanded with no envs subpath",
			container: &v1.Container{
				VolumeMounts: []v1.VolumeMount{{SubPathExpr: "/foo/$(POD_NAME)"}},
			},
			expectedSubPath:   "/foo/$(POD_NAME)",
			expectedMountPath: "",
			expectedOk:        false,
		},
		{
			name: "volumes expanded with leading environment variable",
			container: &v1.Container{
				VolumeMounts: []v1.VolumeMount{{SubPathExpr: "$(POD_NAME)/bar"}},
			},
			envs: []EnvVar{
				{
					Name:  "POD_NAME",
					Value: "foo",
				},
			},
			expectedSubPath:   "foo/bar",
			expectedMountPath: "",
			expectedOk:        true,
		},
		{
			name: "volumes with volume and subpath",
			container: &v1.Container{
				VolumeMounts: []v1.VolumeMount{{MountPath: "/foo", SubPathExpr: "$(POD_NAME)/bar"}},
			},
			envs: []EnvVar{
				{
					Name:  "POD_NAME",
					Value: "foo",
				},
			},
			expectedSubPath:   "foo/bar",
			expectedMountPath: "/foo",
			expectedOk:        true,
		},
		{
			name: "volumes with volume and no subpath",
			container: &v1.Container{
				VolumeMounts: []v1.VolumeMount{{MountPath: "/foo"}},
			},
			envs: []EnvVar{
				{
					Name:  "POD_NAME",
					Value: "foo",
				},
			},
			expectedSubPath:   "",
			expectedMountPath: "/foo",
			expectedOk:        true,
		},
		{
			name: "subpaths with empty environment variable",
			container: &v1.Container{
				VolumeMounts: []v1.VolumeMount{{SubPathExpr: "foo/$(POD_NAME)/$(ANNOTATION)"}},
			},
			envs: []EnvVar{
				{
					Name:  "ANNOTATION",
					Value: "",
				},
			},
			expectedSubPath:   "foo/$(POD_NAME)/$(ANNOTATION)",
			expectedMountPath: "",
			expectedOk:        false,
		},
		{
			name: "subpaths with missing env variables",
			container: &v1.Container{
				VolumeMounts: []v1.VolumeMount{{SubPathExpr: "foo/$(ODD_NAME)/$(POD_NAME)"}},
			},
			envs: []EnvVar{
				{
					Name:  "ODD_NAME",
					Value: "bar",
				},
			},
			expectedSubPath:   "foo/$(ODD_NAME)/$(POD_NAME)",
			expectedMountPath: "",
			expectedOk:        false,
		},
		{
			name: "subpaths with empty expansion",
			container: &v1.Container{
				VolumeMounts: []v1.VolumeMount{{SubPathExpr: "$()"}},
			},
			expectedSubPath:   "$()",
			expectedMountPath: "",
			expectedOk:        false,
		},
		{
			name: "subpaths with nested expandable envs",
			container: &v1.Container{
				VolumeMounts: []v1.VolumeMount{{SubPathExpr: "$(POD_NAME$(ANNOTATION))"}},
			},
			envs: []EnvVar{
				{
					Name:  "POD_NAME",
					Value: "foo",
				},
				{
					Name:  "ANNOTATION",
					Value: "bar",
				},
			},
			expectedSubPath:   "$(POD_NAME$(ANNOTATION))",
			expectedMountPath: "",
			expectedOk:        false,
		},
	}

	for _, tc := range cases {
		actualSubPath, err := ExpandContainerVolumeMounts(tc.container.VolumeMounts[0], tc.envs)
		ok := err == nil
		if e, a := tc.expectedOk, ok; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: unexpected validation failure of subpath; expected %v, got %v", tc.name, e, a)
		}
		if !ok {
			// if ExpandContainerVolumeMounts returns an error, we don't care what the actualSubPath value is
			continue
		}
		if e, a := tc.expectedSubPath, actualSubPath; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: unexpected subpath; expected %v, got %v", tc.name, e, a)
		}
		if e, a := tc.expectedMountPath, tc.container.VolumeMounts[0].MountPath; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: unexpected mountpath; expected %v, got %v", tc.name, e, a)
		}
	}

}

func TestGetContainerSpec(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.EphemeralContainers, true)()
	for _, tc := range []struct {
		name          string
		havePod       *v1.Pod
		haveName      string
		wantContainer *v1.Container
	}{
		{
			name: "regular container",
			havePod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{Name: "plain-ole-container"},
					},
					InitContainers: []v1.Container{
						{Name: "init-container"},
					},
				},
			},
			haveName:      "plain-ole-container",
			wantContainer: &v1.Container{Name: "plain-ole-container"},
		},
		{
			name: "init container",
			havePod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{Name: "plain-ole-container"},
					},
					InitContainers: []v1.Container{
						{Name: "init-container"},
					},
				},
			},
			haveName:      "init-container",
			wantContainer: &v1.Container{Name: "init-container"},
		},
		{
			name: "ephemeral container",
			havePod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{Name: "plain-ole-container"},
					},
					InitContainers: []v1.Container{
						{Name: "init-container"},
					},
					EphemeralContainers: []v1.EphemeralContainer{
						{EphemeralContainerCommon: v1.EphemeralContainerCommon{
							Name: "debug-container",
						}},
					},
				},
			},
			haveName:      "debug-container",
			wantContainer: &v1.Container{Name: "debug-container"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gotContainer := GetContainerSpec(tc.havePod, tc.haveName)
			if diff := cmp.Diff(tc.wantContainer, gotContainer); diff != "" {
				t.Fatalf("GetContainerSpec for %q returned diff (-want +got):%v", tc.name, diff)
			}
		})
	}
}

func TestShouldContainerBeRestarted(t *testing.T) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{Name: "no-history"},
				{Name: "alive"},
				{Name: "succeed"},
				{Name: "failed"},
				{Name: "unknown"},
			},
		},
	}
	podStatus := &PodStatus{
		ID:        pod.UID,
		Name:      pod.Name,
		Namespace: pod.Namespace,
		ContainerStatuses: []*Status{
			{
				Name:  "alive",
				State: ContainerStateRunning,
			},
			{
				Name:     "succeed",
				State:    ContainerStateExited,
				ExitCode: 0,
			},
			{
				Name:     "failed",
				State:    ContainerStateExited,
				ExitCode: 1,
			},
			{
				Name:     "alive",
				State:    ContainerStateExited,
				ExitCode: 2,
			},
			{
				Name:  "unknown",
				State: ContainerStateUnknown,
			},
			{
				Name:     "failed",
				State:    ContainerStateExited,
				ExitCode: 3,
			},
		},
	}
	policies := []v1.RestartPolicy{
		v1.RestartPolicyNever,
		v1.RestartPolicyOnFailure,
		v1.RestartPolicyAlways,
	}

	// test policies
	expected := map[string][]bool{
		"no-history": {true, true, true},
		"alive":      {false, false, false},
		"succeed":    {false, false, true},
		"failed":     {false, true, true},
		"unknown":    {true, true, true},
	}
	for _, c := range pod.Spec.Containers {
		for i, policy := range policies {
			pod.Spec.RestartPolicy = policy
			e := expected[c.Name][i]
			r := ShouldContainerBeRestarted(&c, pod, podStatus)
			if r != e {
				t.Errorf("Restart for container %q with restart policy %q expected %t, got %t",
					c.Name, policy, e, r)
			}
		}
	}

	// test deleted pod
	pod.DeletionTimestamp = &metav1.Time{Time: time.Now()}
	expected = map[string][]bool{
		"no-history": {false, false, false},
		"alive":      {false, false, false},
		"succeed":    {false, false, false},
		"failed":     {false, false, false},
		"unknown":    {false, false, false},
	}
	for _, c := range pod.Spec.Containers {
		for i, policy := range policies {
			pod.Spec.RestartPolicy = policy
			e := expected[c.Name][i]
			r := ShouldContainerBeRestarted(&c, pod, podStatus)
			if r != e {
				t.Errorf("Restart for container %q with restart policy %q expected %t, got %t",
					c.Name, policy, e, r)
			}
		}
	}
}

func TestHasPrivilegedContainer(t *testing.T) {
	newBoolPtr := func(b bool) *bool {
		return &b
	}
	tests := map[string]struct {
		securityContext *v1.SecurityContext
		expected        bool
	}{
		"nil security context": {
			securityContext: nil,
			expected:        false,
		},
		"nil privileged": {
			securityContext: &v1.SecurityContext{},
			expected:        false,
		},
		"false privileged": {
			securityContext: &v1.SecurityContext{Privileged: newBoolPtr(false)},
			expected:        false,
		},
		"true privileged": {
			securityContext: &v1.SecurityContext{Privileged: newBoolPtr(true)},
			expected:        true,
		},
	}

	for k, v := range tests {
		pod := &v1.Pod{
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{SecurityContext: v.securityContext},
				},
			},
		}
		actual := HasPrivilegedContainer(pod)
		if actual != v.expected {
			t.Errorf("%s expected %t but got %t", k, v.expected, actual)
		}
	}
	// Test init containers as well.
	for k, v := range tests {
		pod := &v1.Pod{
			Spec: v1.PodSpec{
				InitContainers: []v1.Container{
					{SecurityContext: v.securityContext},
				},
			},
		}
		actual := HasPrivilegedContainer(pod)
		if actual != v.expected {
			t.Errorf("%s expected %t but got %t", k, v.expected, actual)
		}
	}
}

func TestMakePortMappings(t *testing.T) {
	port := func(name string, protocol v1.Protocol, containerPort, hostPort int32, ip string) v1.ContainerPort {
		return v1.ContainerPort{
			Name:          name,
			Protocol:      protocol,
			ContainerPort: containerPort,
			HostPort:      hostPort,
			HostIP:        ip,
		}
	}
	portMapping := func(protocol v1.Protocol, containerPort, hostPort int, ip string) PortMapping {
		return PortMapping{
			Protocol:      protocol,
			ContainerPort: containerPort,
			HostPort:      hostPort,
			HostIP:        ip,
		}
	}

	tests := []struct {
		container            *v1.Container
		expectedPortMappings []PortMapping
	}{
		{
			&v1.Container{
				Name: "fooContainer",
				Ports: []v1.ContainerPort{
					port("", v1.ProtocolTCP, 80, 8080, "127.0.0.1"),
					port("", v1.ProtocolTCP, 443, 4343, "192.168.0.1"),
					port("foo", v1.ProtocolUDP, 555, 5555, ""),
					// Duplicated, should be ignored.
					port("foo", v1.ProtocolUDP, 888, 8888, ""),
					// Duplicated, should be ignored.
					port("", v1.ProtocolTCP, 80, 8888, "127.0.0.1"),
					// Duplicated with different address family, shouldn't be ignored
					port("", v1.ProtocolTCP, 80, 8080, "::"),
					// No address family specified
					port("", v1.ProtocolTCP, 1234, 5678, ""),
				},
			},
			[]PortMapping{
				portMapping(v1.ProtocolTCP, 80, 8080, "127.0.0.1"),
				portMapping(v1.ProtocolTCP, 443, 4343, "192.168.0.1"),
				portMapping(v1.ProtocolUDP, 555, 5555, ""),
				portMapping(v1.ProtocolTCP, 80, 8080, "::"),
				portMapping(v1.ProtocolTCP, 1234, 5678, ""),
			},
		},
	}

	for i, tt := range tests {
		actual := MakePortMappings(tt.container)
		assert.Equal(t, tt.expectedPortMappings, actual, "[%d]", i)
	}
}

func TestHashContainer(t *testing.T) {
	testCases := []struct {
		name          string
		image         string
		args          []string
		containerPort int32
		expectedHash  uint64
	}{
		{
			name:  "test_container",
			image: "foo/image:v1",
			args: []string{
				"/bin/sh",
				"-c",
				"echo abc",
			},
			containerPort: int32(8001),
			expectedHash:  uint64(0x3c42280f),
		},
	}

	for _, tc := range testCases {
		container := v1.Container{
			Name:  tc.name,
			Image: tc.image,
			Args:  tc.args,
			Ports: []v1.ContainerPort{{ContainerPort: tc.containerPort}},
		}

		hashVal := HashContainer(&container)
		assert.Equal(t, tc.expectedHash, hashVal, "the hash value here should not be changed.")
	}
}
