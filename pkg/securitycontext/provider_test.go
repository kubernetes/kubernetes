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
	var uid int64 = 1
	testCases := map[string]struct {
		securityContext *api.SecurityContext
		expected        *docker.Config
	}{
		"modify config, value set for user": {
			securityContext: &api.SecurityContext{
				RunAsUser: &uid,
			},
			expected: &docker.Config{
				User: strconv.FormatInt(uid, 10),
			},
		},
		"modify config, nil user value": {
			securityContext: &api.SecurityContext{},
			expected:        &docker.Config{},
		},
	}

	provider := NewSimpleSecurityContextProvider()
	dummyContainer := &api.Container{}
	for k, v := range testCases {
		dummyContainer.SecurityContext = v.securityContext
		dockerCfg := &docker.Config{}
		provider.ModifyContainerConfig(nil, dummyContainer, dockerCfg)
		if !reflect.DeepEqual(v.expected, dockerCfg) {
			t.Errorf("unexpected modification of docker config for %s.  Expected: %#v Got: %#v", k, v.expected, dockerCfg)
		}
	}
}

func TestModifyHostConfig(t *testing.T) {
	nilPrivSC := fullValidSecurityContext()
	nilPrivSC.Privileged = nil
	nilPrivHC := fullValidHostConfig()
	nilPrivHC.Privileged = false

	nilCapsSC := fullValidSecurityContext()
	nilCapsSC.Capabilities = nil
	nilCapsHC := fullValidHostConfig()
	nilCapsHC.CapAdd = *new([]string)
	nilCapsHC.CapDrop = *new([]string)

	nilSELinuxSC := fullValidSecurityContext()
	nilSELinuxSC.SELinuxOptions = nil
	nilSELinuxHC := fullValidHostConfig()
	nilSELinuxHC.SecurityOpt = *new([]string)

	seLinuxLabelsSC := fullValidSecurityContext()
	seLinuxLabelsHC := fullValidHostConfig()

	testCases := map[string]struct {
		securityContext *api.SecurityContext
		expected        *docker.HostConfig
	}{
		"full settings": {
			securityContext: fullValidSecurityContext(),
			expected:        fullValidHostConfig(),
		},
		"nil privileged": {
			securityContext: nilPrivSC,
			expected:        nilPrivHC,
		},
		"nil capabilities": {
			securityContext: nilCapsSC,
			expected:        nilCapsHC,
		},
		"nil selinux options": {
			securityContext: nilSELinuxSC,
			expected:        nilSELinuxHC,
		},
		"selinux labels": {
			securityContext: seLinuxLabelsSC,
			expected:        seLinuxLabelsHC,
		},
	}

	provider := NewSimpleSecurityContextProvider()
	dummyContainer := &api.Container{}
	dummyPod := &api.Pod{
		Spec: apitesting.DeepEqualSafePodSpec(),
	}
	for k, v := range testCases {
		dummyContainer.SecurityContext = v.securityContext
		dockerCfg := &docker.HostConfig{}
		provider.ModifyHostConfig(dummyPod, dummyContainer, dockerCfg)
		if !reflect.DeepEqual(v.expected, dockerCfg) {
			t.Errorf("unexpected modification of host config for %s.  Expected: %#v Got: %#v", k, v.expected, dockerCfg)
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

func fullValidSecurityContext() *api.SecurityContext {
	priv := true
	return &api.SecurityContext{
		Privileged: &priv,
		Capabilities: &api.Capabilities{
			Add:  []api.Capability{"addCapA", "addCapB"},
			Drop: []api.Capability{"dropCapA", "dropCapB"},
		},
		SELinuxOptions: &api.SELinuxOptions{
			User:  "user",
			Role:  "role",
			Type:  "type",
			Level: "level",
		},
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
