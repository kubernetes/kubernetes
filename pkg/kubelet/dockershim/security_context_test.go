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
	"fmt"
	"reflect"
	"strconv"
	"testing"

	dockercontainer "github.com/docker/engine-api/types/container"
	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
)

func TestModifyContainerConfig(t *testing.T) {
	var uid int64 = 123
	var overrideUid int64 = 321

	cases := []struct {
		name     string
		podSc    *runtimeApi.PodSecurityContext
		sc       *runtimeApi.SecurityContext
		expected *dockercontainer.Config
	}{
		{
			name: "container.SecurityContext.RunAsUser set",
			sc: &runtimeApi.SecurityContext{
				RunAsUser: &uid,
			},
			expected: &dockercontainer.Config{
				User: strconv.FormatInt(uid, 10),
			},
		},
		{
			name:     "no RunAsUser value set",
			sc:       &runtimeApi.SecurityContext{},
			expected: &dockercontainer.Config{},
		},
		{
			name: "pod.Spec.SecurityContext.RunAsUser set",
			podSc: &runtimeApi.PodSecurityContext{
				RunAsUser: &uid,
			},
			expected: &dockercontainer.Config{
				User: strconv.FormatInt(uid, 10),
			},
		},
		{
			name: "container.SecurityContext.RunAsUser overrides pod.Spec.SecurityContext.RunAsUser",
			podSc: &runtimeApi.PodSecurityContext{
				RunAsUser: &uid,
			},
			sc: &runtimeApi.SecurityContext{
				RunAsUser: &overrideUid,
			},
			expected: &dockercontainer.Config{
				User: strconv.FormatInt(overrideUid, 10),
			},
		},
	}

	for _, tc := range cases {
		dockerCfg := &dockercontainer.Config{}
		modifyContainerConfig(tc.sc, tc.podSc, dockerCfg)
		if e, a := tc.expected, dockerCfg; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: unexpected modification of docker config\nExpected:\n\n%#v\n\nGot:\n\n%#v", tc.name, e, a)
		}
	}
}

func TestModifyHostConfig(t *testing.T) {
	priv := true
	setPrivSC := &runtimeApi.SecurityContext{}
	setPrivSC.Privileged = &priv
	setPrivHC := &dockercontainer.HostConfig{
		Privileged: true,
	}
	setCapsHC := &dockercontainer.HostConfig{
		CapAdd:  []string{"addCapA", "addCapB"},
		CapDrop: []string{"dropCapA", "dropCapB"},
	}
	setSELinuxHC := &dockercontainer.HostConfig{
		SecurityOpt: []string{
			fmt.Sprintf("%s:%s", dockerLabelUser, "user"),
			fmt.Sprintf("%s:%s", dockerLabelRole, "role"),
			fmt.Sprintf("%s:%s", dockerLabelType, "type"),
			fmt.Sprintf("%s:%s", dockerLabelLevel, "level"),
		},
	}

	cases := []struct {
		name     string
		podSc    *runtimeApi.PodSecurityContext
		sc       *runtimeApi.SecurityContext
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
			sc: &runtimeApi.SecurityContext{
				Capabilities: inputCapabilities(),
			},
			expected: setCapsHC,
		},
		{
			name: "container.SecurityContext.SELinuxOptions",
			sc: &runtimeApi.SecurityContext{
				SelinuxOptions: inputSELinuxOptions(),
			},
			expected: setSELinuxHC,
		},
		{
			name: "pod.Spec.SecurityContext.SELinuxOptions",
			podSc: &runtimeApi.PodSecurityContext{
				SelinuxOptions: inputSELinuxOptions(),
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

	for _, tc := range cases {
		dockerCfg := &dockercontainer.HostConfig{}
		modifyHostConfig(tc.sc, tc.podSc, dockerCfg, false)

		if e, a := tc.expected, dockerCfg; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: unexpected modification of host config\nExpected:\n\n%#v\n\nGot:\n\n%#v", tc.name, e, a)
		}
	}
}

func TestModifyHostConfigPodSecurityContext(t *testing.T) {
	supplementalGroupsSC := &runtimeApi.PodSecurityContext{}
	supplementalGroupsSC.SupplementGroups = []int64{2222}
	supplementalGroupHC := &dockercontainer.HostConfig{}
	supplementalGroupHC.GroupAdd = []string{"2222"}
	fsGroupHC := &dockercontainer.HostConfig{}
	fsGroupHC.GroupAdd = []string{"1234"}
	bothHC := &dockercontainer.HostConfig{}
	bothHC.GroupAdd = []string{"2222", "1234"}
	fsGroup := int64(1234)

	testCases := map[string]struct {
		securityContext *runtimeApi.PodSecurityContext
		expected        *dockercontainer.HostConfig
	}{
		"nil": {
			securityContext: nil,
			expected:        &dockercontainer.HostConfig{},
		},
		"SupplementalGroup": {
			securityContext: supplementalGroupsSC,
			expected:        supplementalGroupHC,
		},
		"FSGroup": {
			securityContext: &runtimeApi.PodSecurityContext{FsGroup: &fsGroup},
			expected:        fsGroupHC,
		},
		"FSGroup + SupplementalGroups": {
			securityContext: &runtimeApi.PodSecurityContext{
				SupplementGroups: []int64{2222},
				FsGroup:          &fsGroup,
			},
			expected: bothHC,
		},
	}

	for k, v := range testCases {
		dockerCfg := &dockercontainer.HostConfig{}
		modifyHostConfig(nil, v.securityContext, dockerCfg, false)
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

func overridePodSecurityContext() *runtimeApi.PodSecurityContext {
	user := "user2"
	role := "role2"
	stype := "type2"
	level := "level2"
	return &runtimeApi.PodSecurityContext{
		SelinuxOptions: &runtimeApi.SELinuxOption{
			User:  &user,
			Role:  &role,
			Type:  &stype,
			Level: &level,
		},
	}
}

func fullValidSecurityContext() *runtimeApi.SecurityContext {
	priv := true
	return &runtimeApi.SecurityContext{
		Privileged:     &priv,
		Capabilities:   inputCapabilities(),
		SelinuxOptions: inputSELinuxOptions(),
	}
}

func inputCapabilities() *runtimeApi.Capability {
	return &runtimeApi.Capability{
		AddCapabilities:  []string{"addCapA", "addCapB"},
		DropCapabilities: []string{"dropCapA", "dropCapB"},
	}
}

func inputSELinuxOptions() *runtimeApi.SELinuxOption {
	user := "user"
	role := "role"
	stype := "type"
	level := "level"

	return &runtimeApi.SELinuxOption{
		User:  &user,
		Role:  &role,
		Type:  &stype,
		Level: &level,
	}
}

func fullValidHostConfig() *dockercontainer.HostConfig {
	return &dockercontainer.HostConfig{
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
