/*
Copyright 2020 The Kubernetes Authors.

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

package kubelet

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/util/wait"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

type testPodSandboxDeleter struct {
	podSandboxDeleter
	deletedSandoxes []string
}

func newTestPodSandboxDeleter() (*testPodSandboxDeleter, chan struct{}) {
	buffer := make(chan string, 5)
	stopCh := make(chan struct{})
	testSandboxDeleter := &testPodSandboxDeleter{
		podSandboxDeleter: podSandboxDeleter{
			worker: buffer,
		},
		deletedSandoxes: []string{},
	}
	go wait.Until(func() {
		for {
			id, ok := <-buffer
			if !ok {
				close(stopCh)
				break
			}
			testSandboxDeleter.deletedSandoxes = append(testSandboxDeleter.deletedSandoxes, id)
		}
	}, 0, stopCh)

	return testSandboxDeleter, stopCh
}

func Test_podSandboxDeleter_deleteSandboxesInPod(t *testing.T) {
	type args struct {
		podStatus *kubecontainer.PodStatus
		toKeep    int
	}
	tests := []struct {
		name string
		args args
		want []string
	}{
		{
			name: "ready sandboxes shouldn't be deleted ever",
			args: args{
				podStatus: &kubecontainer.PodStatus{
					SandboxStatuses: []*runtimeapi.PodSandboxStatus{
						{
							Id:    "testsandbox",
							State: runtimeapi.PodSandboxState_SANDBOX_READY,
						},
					},
				},
				toKeep: 0,
			},
			want: []string{},
		},
		{
			name: "all unready sandboxes should be deleted if to keep is 0",
			args: args{
				podStatus: &kubecontainer.PodStatus{
					SandboxStatuses: []*runtimeapi.PodSandboxStatus{
						{
							Id:    "testsandbox",
							State: runtimeapi.PodSandboxState_SANDBOX_READY,
						},
						{
							Id:    "testsandbox1",
							State: runtimeapi.PodSandboxState_SANDBOX_NOTREADY,
						},
						{
							Id:    "testsandbox2",
							State: runtimeapi.PodSandboxState_SANDBOX_NOTREADY,
						},
					},
				},
				toKeep: 0,
			},
			want: []string{"testsandbox1", "testsandbox2"},
		},
		{
			name: "sandboxes with containers shouldn't be deleted",
			args: args{
				podStatus: &kubecontainer.PodStatus{
					ContainerStatuses: []*kubecontainer.Status{
						{
							PodSandboxID: "testsandbox1",
						},
					},
					SandboxStatuses: []*runtimeapi.PodSandboxStatus{
						{
							Id:    "testsandbox1",
							State: runtimeapi.PodSandboxState_SANDBOX_NOTREADY,
						},
						{
							Id:    "testsandbox2",
							State: runtimeapi.PodSandboxState_SANDBOX_NOTREADY,
						},
					},
				},
				toKeep: 0,
			},
			want: []string{"testsandbox2"},
		},
		{
			name: "latest unready sandboxes shouldn't be deleted if to keep is 1",
			args: args{
				podStatus: &kubecontainer.PodStatus{
					SandboxStatuses: []*runtimeapi.PodSandboxStatus{
						{
							Id:        "testsandbox1",
							State:     runtimeapi.PodSandboxState_SANDBOX_NOTREADY,
							CreatedAt: time.Now().Add(time.Second).UnixNano(),
						},
						{
							Id:        "testsandbox2",
							State:     runtimeapi.PodSandboxState_SANDBOX_NOTREADY,
							CreatedAt: time.Now().Add(2 * time.Second).UnixNano(),
						},
					},
				},
				toKeep: 1,
			},
			want: []string{"testsandbox1"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, stopCh := newTestPodSandboxDeleter()
			p.deleteSandboxesInPod(tt.args.podStatus, tt.args.toKeep)
			close(p.worker)
			<-stopCh
			assert.ElementsMatch(t, tt.want, p.deletedSandoxes, tt.name)
		})
	}
}
