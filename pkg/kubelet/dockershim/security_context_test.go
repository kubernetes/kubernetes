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
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
)

func TestModifyContainerConfig(t *testing.T) {
	var uid int64 = 123

	cases := []struct {
		name     string
		sc       *runtimeapi.SecurityContext
		expected *dockercontainer.Config
	}{
		{
			name: "container.SecurityContext.RunAsUser set",
			sc: &runtimeapi.SecurityContext{
				RunAsUser: &uid,
			},
			expected: &dockercontainer.Config{
				User: strconv.FormatInt(uid, 10),
			},
		},
		{
			name:     "no RunAsUser value set",
			sc:       &runtimeapi.SecurityContext{},
			expected: &dockercontainer.Config{},
		},
	}

	for _, tc := range cases {
		dockerCfg := &dockercontainer.Config{}
		modifyContainerConfig(tc.sc, dockerCfg)
		if e, a := tc.expected, dockerCfg; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: unexpected modification of docker config\nExpected:\n\n%#v\n\nGot:\n\n%#v", tc.name, e, a)
		}
	}
}

func TestModifyHostConfig(t *testing.T) {
	priv := true
	setPrivSC := &runtimeapi.SecurityContext{}
	setPrivSC.Privileged = &priv
	setPrivHC := &dockercontainer.HostConfig{
		Privileged:  true,
		NetworkMode: "none",
	}
	setCapsHC := &dockercontainer.HostConfig{
		NetworkMode: "none",
		CapAdd:      []string{"addCapA", "addCapB"},
		CapDrop:     []string{"dropCapA", "dropCapB"},
	}
	setSELinuxHC := &dockercontainer.HostConfig{
		NetworkMode: "none",
		SecurityOpt: []string{
			fmt.Sprintf("%s:%s", dockerLabelUser, "user"),
			fmt.Sprintf("%s:%s", dockerLabelRole, "role"),
			fmt.Sprintf("%s:%s", dockerLabelType, "type"),
			fmt.Sprintf("%s:%s", dockerLabelLevel, "level"),
		},
	}

	cases := []struct {
		name     string
		sc       *runtimeapi.SecurityContext
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
			sc: &runtimeapi.SecurityContext{
				Capabilities: inputCapabilities(),
			},
			expected: setCapsHC,
		},
		{
			name: "container.SecurityContext.SELinuxOptions",
			sc: &runtimeapi.SecurityContext{
				SelinuxOptions: inputSELinuxOptions(),
			},
			expected: setSELinuxHC,
		},
	}

	for _, tc := range cases {
		dockerCfg := &dockercontainer.HostConfig{}
		modifyHostConfig(tc.sc, "", dockerCfg)

		if e, a := tc.expected, dockerCfg; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: unexpected modification of host config\nExpected:\n\n%#v\n\nGot:\n\n%#v", tc.name, e, a)
		}
	}
}

func TestModifyHostConfigWithGroups(t *testing.T) {
	supplementalGroupsSC := &runtimeapi.SecurityContext{}
	supplementalGroupsSC.SupplementalGroups = []int64{2222}
	supplementalGroupHC := &dockercontainer.HostConfig{NetworkMode: "none"}
	supplementalGroupHC.GroupAdd = []string{"2222"}
	fsGroupHC := &dockercontainer.HostConfig{NetworkMode: "none"}
	fsGroupHC.GroupAdd = []string{"1234"}
	bothHC := &dockercontainer.HostConfig{NetworkMode: "none"}
	bothHC.GroupAdd = []string{"2222", "1234"}
	fsGroup := int64(1234)

	testCases := map[string]struct {
		securityContext *runtimeapi.SecurityContext
		expected        *dockercontainer.HostConfig
	}{
		"nil": {
			securityContext: nil,
			expected:        &dockercontainer.HostConfig{NetworkMode: "none"},
		},
		"SupplementalGroup": {
			securityContext: supplementalGroupsSC,
			expected:        supplementalGroupHC,
		},
		"FSGroup": {
			securityContext: &runtimeapi.SecurityContext{FsGroup: &fsGroup},
			expected:        fsGroupHC,
		},
		"FSGroup + SupplementalGroups": {
			securityContext: &runtimeapi.SecurityContext{
				SupplementalGroups: []int64{2222},
				FsGroup:            &fsGroup,
			},
			expected: bothHC,
		},
	}

	for k, v := range testCases {
		dockerCfg := &dockercontainer.HostConfig{}
		modifyHostConfig(v.securityContext, "", dockerCfg)
		if !reflect.DeepEqual(v.expected, dockerCfg) {
			t.Errorf("unexpected modification of host config for %s.  Expected: %#v Got: %#v", k, v.expected, dockerCfg)
		}
	}
}

func TestModifyHostConfigWithSandboxID(t *testing.T) {
	priv := true
	sandboxID := "sandbox"
	sandboxNSMode := fmt.Sprintf("container:%v", sandboxID)
	setPrivSC := &runtimeapi.SecurityContext{}
	setPrivSC.Privileged = &priv
	setPrivHC := &dockercontainer.HostConfig{
		Privileged:  true,
		IpcMode:     dockercontainer.IpcMode(sandboxNSMode),
		NetworkMode: dockercontainer.NetworkMode(sandboxNSMode),
	}
	setCapsHC := &dockercontainer.HostConfig{
		CapAdd:      []string{"addCapA", "addCapB"},
		CapDrop:     []string{"dropCapA", "dropCapB"},
		IpcMode:     dockercontainer.IpcMode(sandboxNSMode),
		NetworkMode: dockercontainer.NetworkMode(sandboxNSMode),
	}
	setSELinuxHC := &dockercontainer.HostConfig{
		SecurityOpt: []string{
			fmt.Sprintf("%s:%s", dockerLabelUser, "user"),
			fmt.Sprintf("%s:%s", dockerLabelRole, "role"),
			fmt.Sprintf("%s:%s", dockerLabelType, "type"),
			fmt.Sprintf("%s:%s", dockerLabelLevel, "level"),
		},
		IpcMode:     dockercontainer.IpcMode(sandboxNSMode),
		NetworkMode: dockercontainer.NetworkMode(sandboxNSMode),
	}

	cases := []struct {
		name     string
		sc       *runtimeapi.SecurityContext
		expected *dockercontainer.HostConfig
	}{
		{
			name:     "container.SecurityContext.Privileged",
			sc:       setPrivSC,
			expected: setPrivHC,
		},
		{
			name: "container.SecurityContext.Capabilities",
			sc: &runtimeapi.SecurityContext{
				Capabilities: inputCapabilities(),
			},
			expected: setCapsHC,
		},
		{
			name: "container.SecurityContext.SELinuxOptions",
			sc: &runtimeapi.SecurityContext{
				SelinuxOptions: inputSELinuxOptions(),
			},
			expected: setSELinuxHC,
		},
	}

	for _, tc := range cases {
		dockerCfg := &dockercontainer.HostConfig{}
		modifyHostConfig(tc.sc, sandboxID, dockerCfg)

		if e, a := tc.expected, dockerCfg; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: unexpected modification of host config\nExpected:\n\n%#v\n\nGot:\n\n%#v", tc.name, e, a)
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

func fullValidSecurityContext() *runtimeapi.SecurityContext {
	priv := true
	return &runtimeapi.SecurityContext{
		Privileged:     &priv,
		Capabilities:   inputCapabilities(),
		SelinuxOptions: inputSELinuxOptions(),
	}
}

func inputCapabilities() *runtimeapi.Capability {
	return &runtimeapi.Capability{
		AddCapabilities:  []string{"addCapA", "addCapB"},
		DropCapabilities: []string{"dropCapA", "dropCapB"},
	}
}

func inputSELinuxOptions() *runtimeapi.SELinuxOption {
	user := "user"
	role := "role"
	stype := "type"
	level := "level"

	return &runtimeapi.SELinuxOption{
		User:  &user,
		Role:  &role,
		Type:  &stype,
		Level: &level,
	}
}

func fullValidHostConfig() *dockercontainer.HostConfig {
	return &dockercontainer.HostConfig{
		Privileged:  true,
		NetworkMode: "none",
		CapAdd:      []string{"addCapA", "addCapB"},
		CapDrop:     []string{"dropCapA", "dropCapB"},
		SecurityOpt: []string{
			fmt.Sprintf("%s:%s", dockerLabelUser, "user"),
			fmt.Sprintf("%s:%s", dockerLabelRole, "role"),
			fmt.Sprintf("%s:%s", dockerLabelType, "type"),
			fmt.Sprintf("%s:%s", dockerLabelLevel, "level"),
		},
	}
}
