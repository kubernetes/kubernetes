// +build windows,!dockerless

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
	"github.com/stretchr/testify/assert"
	"testing"

	dockercontainer "github.com/docker/docker/api/types/container"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
)

func TestModifyWindowsContainerConfig(t *testing.T) {
	var username = "testuser"
	cases := []struct {
		name     string
		sc       *runtimeapi.WindowsContainerSecurityContext
		expected *dockercontainer.Config
		isErr    bool
	}{
		{
			name: "container.SecurityContext.RunAsUser set",
			sc: &runtimeapi.WindowsContainerSecurityContext{
				RunAsUsername: username,
			},
			expected: &dockercontainer.Config{
				User: username,
			},
			isErr: false,
		},
	}
	for _, tc := range cases {
		dockerCfg := &dockercontainer.Config{}
		err := modifyContainerConfig(tc.sc, dockerCfg)
		if tc.isErr {
			assert.NotNil(t, err)
		} else {
			assert.Nil(t, err)
			assert.Equal(t, tc.expected, dockerCfg, "[Test case %q]", tc.name)
		}
	}
}
