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

package kuberuntime

import (
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
)

func TestSysctlFiltering(t *testing.T) {
	_, _, m, err := createTestRuntimeManager()
	assert.NoError(t, err)
	originalPodSysctls := PodSysctls
	defer func() { PodSysctls = originalPodSysctls }()
	PodSysctls = map[string]string{
		"net.somaxconn":     "1024",
		"kernel.msgmax":     "true",
		"fs.mqueue.msg_max": "1024",
		"kernel.domainname": "my-name",
		"kernel.hostname":   "my-name",
		"non.whitelisted":   "true",
	}

	createTestPodFunc := func(hostNetwork, hostIPC bool) *v1.Pod {
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				UID:       "12345678",
				Name:      "foo",
				Namespace: "new",
			},
			Spec: v1.PodSpec{
				HostNetwork: hostNetwork,
				HostIPC:     hostIPC,
				Containers: []v1.Container{
					{
						Name:            "foo1",
						Image:           "busybox",
						ImagePullPolicy: v1.PullIfNotPresent,
					},
				},
			},
		}
	}

	for _, test := range []struct {
		hostNetwork     bool
		hostIPC         bool
		expectedSysctls map[string]string
	}{
		{
			expectedSysctls: map[string]string{
				"net.somaxconn":     "1024",
				"kernel.msgmax":     "true",
				"fs.mqueue.msg_max": "1024",
				"kernel.domainname": "my-name",
			},
		},
		{
			hostNetwork: true,
			expectedSysctls: map[string]string{
				"kernel.msgmax":     "true",
				"fs.mqueue.msg_max": "1024",
			},
		},
		{
			hostIPC: true,
			expectedSysctls: map[string]string{
				"net.somaxconn":     "1024",
				"kernel.domainname": "my-name",
			},
		},
		{
			hostNetwork:     true,
			hostIPC:         true,
			expectedSysctls: map[string]string{},
		},
	} {
		pod := createTestPodFunc(test.hostNetwork, test.hostIPC)
		template := sandboxTemplate{pod, 1, fakeCreatedAt, runtimeapi.PodSandboxState_SANDBOX_READY, true, false}
		config, err := m.generatePodSandboxConfig(template.pod, template.attempt)
		assert.NoError(t, err)
		if !reflect.DeepEqual(test.expectedSysctls, config.Linux.Sysctls) {
			t.Errorf("Expected sysctls %v, got %v", test.expectedSysctls, config.Linux.Sysctls)
		}
	}
}
