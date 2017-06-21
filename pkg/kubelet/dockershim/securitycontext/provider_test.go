/*
Copyright 2014 The Kubernetes Authors.

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

package securitycontext

import (
	"fmt"
	"reflect"
	"strconv"
	"testing"

	dockercontainer "github.com/docker/engine-api/types/container"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/api/v1"
)

func TestModifyContainerConfig(t *testing.T) {
	userID := int64(123)
	overrideUserID := int64(321)

	cases := []struct {
		name     string
		podSc    *v1.PodSecurityContext
		sc       *v1.SecurityContext
		expected *dockercontainer.Config
	}{
		{
			name: "container.SecurityContext.RunAsUser set",
			sc: &v1.SecurityContext{
				RunAsUser: &userID,
			},
			expected: &dockercontainer.Config{
				User: strconv.FormatInt(int64(userID), 10),
			},
		},
		{
			name:     "no RunAsUser value set",
			sc:       &v1.SecurityContext{},
			expected: &dockercontainer.Config{},
		},
		{
			name: "pod.Spec.SecurityContext.RunAsUser set",
			podSc: &v1.PodSecurityContext{
				RunAsUser: &userID,
			},
			expected: &dockercontainer.Config{
				User: strconv.FormatInt(int64(userID), 10),
			},
		},
		{
			name: "container.SecurityContext.RunAsUser overrides pod.Spec.SecurityContext.RunAsUser",
			podSc: &v1.PodSecurityContext{
				RunAsUser: &userID,
			},
			sc: &v1.SecurityContext{
				RunAsUser: &overrideUserID,
			},
			expected: &dockercontainer.Config{
				User: strconv.FormatInt(int64(overrideUserID), 10),
			},
		},
	}

	provider := NewSimpleSecurityContextProvider('=')
	dummyContainer := &v1.Container{}
	for _, tc := range cases {
		pod := &v1.Pod{Spec: v1.PodSpec{SecurityContext: tc.podSc}}
		dummyContainer.SecurityContext = tc.sc
		dockerCfg := &dockercontainer.Config{}

		provider.ModifyContainerConfig(pod, dummyContainer, dockerCfg)

		if e, a := tc.expected, dockerCfg; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: unexpected modification of docker config\nExpected:\n\n%#v\n\nGot:\n\n%#v", tc.name, e, a)
		}
	}
}

func TestModifyHostConfig(t *testing.T) {
	priv := true
	setPrivSC := &v1.SecurityContext{}
	setPrivSC.Privileged = &priv
	setPrivHC := &dockercontainer.HostConfig{
		Privileged: true,
	}

	setCapsHC := &dockercontainer.HostConfig{
		CapAdd:  []string{"addCapA", "addCapB"},
		CapDrop: []string{"dropCapA", "dropCapB"},
	}

	setSELinuxHC := &dockercontainer.HostConfig{}
	setSELinuxHC.SecurityOpt = []string{
		fmt.Sprintf("%s:%s", DockerLabelUser(':'), "user"),
		fmt.Sprintf("%s:%s", DockerLabelRole(':'), "role"),
		fmt.Sprintf("%s:%s", DockerLabelType(':'), "type"),
		fmt.Sprintf("%s:%s", DockerLabelLevel(':'), "level"),
	}

	// seLinuxLabelsSC := fullValidSecurityContext()
	// seLinuxLabelsHC := fullValidHostConfig()

	cases := []struct {
		name     string
		podSc    *v1.PodSecurityContext
		sc       *v1.SecurityContext
		expected *dockercontainer.HostConfig
	}{
		{
			name:     "fully set container.SecurityContext",
			sc:       fullValidSecurityContext(),
			expected: fullValidHostConfig(),
		},
		{
			name:     "container.SecurityContext.Privileged",
			sc:       setPrivSC,
			expected: setPrivHC,
		},
		{
			name: "container.SecurityContext.Capabilities",
			sc: &v1.SecurityContext{
				Capabilities: inputCapabilities(),
			},
			expected: setCapsHC,
		},
		{
			name: "container.SecurityContext.SELinuxOptions",
			sc: &v1.SecurityContext{
				SELinuxOptions: inputSELinuxOptions(),
			},
			expected: setSELinuxHC,
		},
		{
			name: "pod.Spec.SecurityContext.SELinuxOptions",
			podSc: &v1.PodSecurityContext{
				SELinuxOptions: inputSELinuxOptions(),
			},
			expected: setSELinuxHC,
		},
		{
			name:     "container.SecurityContext overrides pod.Spec.SecurityContext",
			podSc:    overridePodSecurityContext(),
			sc:       fullValidSecurityContext(),
			expected: fullValidHostConfig(),
		},
	}

	provider := NewSimpleSecurityContextProvider(':')
	dummyContainer := &v1.Container{}

	for _, tc := range cases {
		pod := &v1.Pod{Spec: v1.PodSpec{SecurityContext: tc.podSc}}
		dummyContainer.SecurityContext = tc.sc
		dockerCfg := &dockercontainer.HostConfig{}

		provider.ModifyHostConfig(pod, dummyContainer, dockerCfg, nil)

		if e, a := tc.expected, dockerCfg; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: unexpected modification of host config\nExpected:\n\n%#v\n\nGot:\n\n%#v", tc.name, e, a)
		}
	}
}

func TestModifyHostConfigPodSecurityContext(t *testing.T) {
	supplementalGroupsSC := &v1.PodSecurityContext{}
	supplementalGroupsSC.SupplementalGroups = []int64{2222}
	supplementalGroupHC := fullValidHostConfig()
	supplementalGroupHC.GroupAdd = []string{"2222"}
	fsGroupHC := fullValidHostConfig()
	fsGroupHC.GroupAdd = []string{"1234"}
	extraSupplementalGroupHC := fullValidHostConfig()
	extraSupplementalGroupHC.GroupAdd = []string{"1234"}
	bothHC := fullValidHostConfig()
	bothHC.GroupAdd = []string{"2222", "1234"}
	fsGroup := int64(1234)
	extraSupplementalGroup := []int64{1234}

	testCases := map[string]struct {
		securityContext         *v1.PodSecurityContext
		expected                *dockercontainer.HostConfig
		extraSupplementalGroups []int64
	}{
		"nil": {
			securityContext:         nil,
			expected:                fullValidHostConfig(),
			extraSupplementalGroups: nil,
		},
		"SupplementalGroup": {
			securityContext:         supplementalGroupsSC,
			expected:                supplementalGroupHC,
			extraSupplementalGroups: nil,
		},
		"FSGroup": {
			securityContext:         &v1.PodSecurityContext{FSGroup: &fsGroup},
			expected:                fsGroupHC,
			extraSupplementalGroups: nil,
		},
		"FSGroup + SupplementalGroups": {
			securityContext: &v1.PodSecurityContext{
				SupplementalGroups: []int64{2222},
				FSGroup:            &fsGroup,
			},
			expected:                bothHC,
			extraSupplementalGroups: nil,
		},
		"ExtraSupplementalGroup": {
			securityContext:         nil,
			expected:                extraSupplementalGroupHC,
			extraSupplementalGroups: extraSupplementalGroup,
		},
		"ExtraSupplementalGroup + SupplementalGroups": {
			securityContext:         supplementalGroupsSC,
			expected:                bothHC,
			extraSupplementalGroups: extraSupplementalGroup,
		},
	}

	provider := NewSimpleSecurityContextProvider(':')
	dummyContainer := &v1.Container{}
	dummyContainer.SecurityContext = fullValidSecurityContext()
	dummyPod := &v1.Pod{
		Spec: apitesting.V1DeepEqualSafePodSpec(),
	}

	for k, v := range testCases {
		dummyPod.Spec.SecurityContext = v.securityContext
		dockerCfg := &dockercontainer.HostConfig{}
		provider.ModifyHostConfig(dummyPod, dummyContainer, dockerCfg, v.extraSupplementalGroups)
		if !reflect.DeepEqual(v.expected, dockerCfg) {
			t.Errorf("unexpected modification of host config for %s.  Expected: %#v Got: %#v", k, v.expected, dockerCfg)
		}
	}
}

func TestModifySecurityOption(t *testing.T) {
	testCases := []struct {
		name     string
		config   []string
		optName  string
		optVal   string
		expected []string
	}{
		{
			name:     "Empty val",
			config:   []string{"a:b", "c:d"},
			optName:  "optA",
			optVal:   "",
			expected: []string{"a:b", "c:d"},
		},
		{
			name:     "Valid",
			config:   []string{"a:b", "c:d"},
			optName:  "e",
			optVal:   "f",
			expected: []string{"a:b", "c:d", "e:f"},
		},
	}

	for _, tc := range testCases {
		actual := modifySecurityOption(tc.config, tc.optName, tc.optVal)
		if !reflect.DeepEqual(tc.expected, actual) {
			t.Errorf("Failed to apply options correctly for tc: %s.  Expected: %v but got %v", tc.name, tc.expected, actual)
		}
	}
}

func overridePodSecurityContext() *v1.PodSecurityContext {
	return &v1.PodSecurityContext{
		SELinuxOptions: &v1.SELinuxOptions{
			User:  "user2",
			Role:  "role2",
			Type:  "type2",
			Level: "level2",
		},
	}
}

func fullValidPodSecurityContext() *v1.PodSecurityContext {
	return &v1.PodSecurityContext{
		SELinuxOptions: inputSELinuxOptions(),
	}
}

func fullValidSecurityContext() *v1.SecurityContext {
	priv := true
	return &v1.SecurityContext{
		Privileged:     &priv,
		Capabilities:   inputCapabilities(),
		SELinuxOptions: inputSELinuxOptions(),
	}
}

func inputCapabilities() *v1.Capabilities {
	return &v1.Capabilities{
		Add:  []v1.Capability{"addCapA", "addCapB"},
		Drop: []v1.Capability{"dropCapA", "dropCapB"},
	}
}

func inputSELinuxOptions() *v1.SELinuxOptions {
	return &v1.SELinuxOptions{
		User:  "user",
		Role:  "role",
		Type:  "type",
		Level: "level",
	}
}

func fullValidHostConfig() *dockercontainer.HostConfig {
	return &dockercontainer.HostConfig{
		Privileged: true,
		CapAdd:     []string{"addCapA", "addCapB"},
		CapDrop:    []string{"dropCapA", "dropCapB"},
		SecurityOpt: []string{
			fmt.Sprintf("%s:%s", DockerLabelUser(':'), "user"),
			fmt.Sprintf("%s:%s", DockerLabelRole(':'), "role"),
			fmt.Sprintf("%s:%s", DockerLabelType(':'), "type"),
			fmt.Sprintf("%s:%s", DockerLabelLevel(':'), "level"),
		},
	}
}
