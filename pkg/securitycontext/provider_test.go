/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

	docker "github.com/fsouza/go-dockerclient"
	"k8s.io/kubernetes/pkg/api"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
)

func TestModifyContainerConfig(t *testing.T) {
	var uid int64 = 123
	var overrideUid int64 = 321

	cases := []struct {
		name     string
		podSc    *api.PodSecurityContext
		sc       *api.SecurityContext
		expected *docker.Config
	}{
		{
			name: "container.SecurityContext.RunAsUser set",
			sc: &api.SecurityContext{
				RunAsUser: &uid,
			},
			expected: &docker.Config{
				User: strconv.FormatInt(uid, 10),
			},
		},
		{
			name:     "no RunAsUser value set",
			sc:       &api.SecurityContext{},
			expected: &docker.Config{},
		},
		{
			name: "pod.Spec.SecurityContext.RunAsUser set",
			podSc: &api.PodSecurityContext{
				RunAsUser: &uid,
			},
			expected: &docker.Config{
				User: strconv.FormatInt(uid, 10),
			},
		},
		{
			name: "container.SecurityContext.RunAsUser overrides pod.Spec.SecurityContext.RunAsUser",
			podSc: &api.PodSecurityContext{
				RunAsUser: &uid,
			},
			sc: &api.SecurityContext{
				RunAsUser: &overrideUid,
			},
			expected: &docker.Config{
				User: strconv.FormatInt(overrideUid, 10),
			},
		},
	}

	provider := NewSimpleSecurityContextProvider()
	dummyContainer := &api.Container{}
	for _, tc := range cases {
		pod := &api.Pod{Spec: api.PodSpec{SecurityContext: tc.podSc}}
		dummyContainer.SecurityContext = tc.sc
		dockerCfg := &docker.Config{}

		provider.ModifyContainerConfig(pod, dummyContainer, dockerCfg)

		if e, a := tc.expected, dockerCfg; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: unexpected modification of docker config\nExpected:\n\n%#v\n\nGot:\n\n%#v", tc.name, e, a)
		}
	}
}

func TestModifyHostConfig(t *testing.T) {
	priv := true
	setPrivSC := &api.SecurityContext{}
	setPrivSC.Privileged = &priv
	setPrivHC := &docker.HostConfig{
		Privileged: true,
	}

	setCapsHC := &docker.HostConfig{
		CapAdd:  []string{"addCapA", "addCapB"},
		CapDrop: []string{"dropCapA", "dropCapB"},
	}

	setSELinuxHC := &docker.HostConfig{}
	setSELinuxHC.SecurityOpt = []string{
		fmt.Sprintf("%s:%s", dockerLabelUser, "user"),
		fmt.Sprintf("%s:%s", dockerLabelRole, "role"),
		fmt.Sprintf("%s:%s", dockerLabelType, "type"),
		fmt.Sprintf("%s:%s", dockerLabelLevel, "level"),
	}

	// seLinuxLabelsSC := fullValidSecurityContext()
	// seLinuxLabelsHC := fullValidHostConfig()

	cases := []struct {
		name     string
		podSc    *api.PodSecurityContext
		sc       *api.SecurityContext
		expected *docker.HostConfig
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
			sc: &api.SecurityContext{
				Capabilities: inputCapabilities(),
			},
			expected: setCapsHC,
		},
		{
			name: "container.SecurityContext.SELinuxOptions",
			sc: &api.SecurityContext{
				SELinuxOptions: inputSELinuxOptions(),
			},
			expected: setSELinuxHC,
		},
		{
			name: "pod.Spec.SecurityContext.SELinuxOptions",
			podSc: &api.PodSecurityContext{
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

	provider := NewSimpleSecurityContextProvider()
	dummyContainer := &api.Container{}

	for _, tc := range cases {
		pod := &api.Pod{Spec: api.PodSpec{SecurityContext: tc.podSc}}
		dummyContainer.SecurityContext = tc.sc
		dockerCfg := &docker.HostConfig{}

		provider.ModifyHostConfig(pod, dummyContainer, dockerCfg)

		if e, a := tc.expected, dockerCfg; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: unexpected modification of host config\nExpected:\n\n%#v\n\nGot:\n\n%#v", tc.name, e, a)
		}
	}
}

func TestModifyHostConfigPodSecurityContext(t *testing.T) {
	supplementalGroupsSC := &api.PodSecurityContext{}
	supplementalGroupsSC.SupplementalGroups = []int64{2222}
	supplementalGroupHC := fullValidHostConfig()
	supplementalGroupHC.GroupAdd = []string{"2222"}

	testCases := map[string]struct {
		securityContext *api.PodSecurityContext
		expected        *docker.HostConfig
	}{
		"nil Security Context": {
			securityContext: nil,
			expected:        fullValidHostConfig(),
		},
		"Security Context with SupplementalGroup": {
			securityContext: supplementalGroupsSC,
			expected:        supplementalGroupHC,
		},
	}

	provider := NewSimpleSecurityContextProvider()
	dummyContainer := &api.Container{}
	dummyContainer.SecurityContext = fullValidSecurityContext()
	dummyPod := &api.Pod{
		Spec: apitesting.DeepEqualSafePodSpec(),
	}

	for k, v := range testCases {
		dummyPod.Spec.SecurityContext = v.securityContext
		dockerCfg := &docker.HostConfig{}
		provider.ModifyHostConfig(dummyPod, dummyContainer, dockerCfg)
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

func overridePodSecurityContext() *api.PodSecurityContext {
	return &api.PodSecurityContext{
		SELinuxOptions: &api.SELinuxOptions{
			User:  "user2",
			Role:  "role2",
			Type:  "type2",
			Level: "level2",
		},
	}
}

func fullValidPodSecurityContext() *api.PodSecurityContext {
	return &api.PodSecurityContext{
		SELinuxOptions: inputSELinuxOptions(),
	}
}

func fullValidSecurityContext() *api.SecurityContext {
	priv := true
	return &api.SecurityContext{
		Privileged:     &priv,
		Capabilities:   inputCapabilities(),
		SELinuxOptions: inputSELinuxOptions(),
	}
}

func inputCapabilities() *api.Capabilities {
	return &api.Capabilities{
		Add:  []api.Capability{"addCapA", "addCapB"},
		Drop: []api.Capability{"dropCapA", "dropCapB"},
	}
}

func inputSELinuxOptions() *api.SELinuxOptions {
	return &api.SELinuxOptions{
		User:  "user",
		Role:  "role",
		Type:  "type",
		Level: "level",
	}
}

func fullValidHostConfig() *docker.HostConfig {
	return &docker.HostConfig{
		Privileged: true,
		CapAdd:     []string{"addCapA", "addCapB"},
		CapDrop:    []string{"dropCapA", "dropCapB"},
		SecurityOpt: []string{
			fmt.Sprintf("%s:%s", dockerLabelUser, "user"),
			fmt.Sprintf("%s:%s", dockerLabelRole, "role"),
			fmt.Sprintf("%s:%s", dockerLabelType, "type"),
			fmt.Sprintf("%s:%s", dockerLabelLevel, "level"),
		},
	}
}
