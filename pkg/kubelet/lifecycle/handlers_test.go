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

package lifecycle

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertesting "k8s.io/kubernetes/pkg/kubelet/container/testing"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/utils/pointer"
)

func TestResolvePort(t *testing.T) {
	for _, testCase := range []struct {
		container  *v1.Container
		stringPort string
		expected   int
	}{
		{
			stringPort: "foo",
			container: &v1.Container{
				Ports: []v1.ContainerPort{{Name: "foo", ContainerPort: int32(80)}},
			},
			expected: 80,
		},
		{
			container:  &v1.Container{},
			stringPort: "80",
			expected:   80,
		},
		{
			container: &v1.Container{
				Ports: []v1.ContainerPort{
					{Name: "bar", ContainerPort: int32(80)},
				},
			},
			stringPort: "foo",
			expected:   -1,
		},
	} {
		port, err := resolvePort(intstr.FromString(testCase.stringPort), testCase.container)
		if testCase.expected != -1 && err != nil {
			t.Fatalf("unexpected error while resolving port: %s", err)
		}
		if testCase.expected == -1 && err == nil {
			t.Errorf("expected error when a port fails to resolve")
		}
		if testCase.expected != port {
			t.Errorf("failed to resolve port, expected %d, got %d", testCase.expected, port)
		}
	}
}

type fakeContainerCommandRunner struct {
	Cmd []string
	ID  kubecontainer.ContainerID
	Err error
	Msg string
}

func (f *fakeContainerCommandRunner) RunInContainer(id kubecontainer.ContainerID, cmd []string, timeout time.Duration) ([]byte, error) {
	f.Cmd = cmd
	f.ID = id
	return []byte(f.Msg), f.Err
}

func TestRunHandlerExec(t *testing.T) {
	fakeCommandRunner := fakeContainerCommandRunner{}
	handlerRunner := NewHandlerRunner(&fakeHTTP{}, &fakeCommandRunner, nil)

	containerID := kubecontainer.ContainerID{Type: "test", ID: "abc1234"}
	containerName := "containerFoo"

	container := v1.Container{
		Name: containerName,
		Lifecycle: &v1.Lifecycle{
			PostStart: &v1.Handler{
				Exec: &v1.ExecAction{
					Command: []string{"ls", "-a"},
				},
			},
		},
	}

	pod := v1.Pod{}
	pod.ObjectMeta.Name = "podFoo"
	pod.ObjectMeta.Namespace = "nsFoo"
	pod.Spec.Containers = []v1.Container{container}
	_, err := handlerRunner.Run(containerID, &pod, &container, container.Lifecycle.PostStart)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if fakeCommandRunner.ID != containerID ||
		!reflect.DeepEqual(container.Lifecycle.PostStart.Exec.Command, fakeCommandRunner.Cmd) {
		t.Errorf("unexpected commands: %v", fakeCommandRunner)
	}
}

type fakeHTTP struct {
	url  string
	err  error
	resp *http.Response
}

func (f *fakeHTTP) Get(url string) (*http.Response, error) {
	f.url = url
	return f.resp, f.err
}

func TestRunHandlerHttp(t *testing.T) {
	fakeHTTPGetter := fakeHTTP{}
	handlerRunner := NewHandlerRunner(&fakeHTTPGetter, &fakeContainerCommandRunner{}, nil)

	containerID := kubecontainer.ContainerID{Type: "test", ID: "abc1234"}
	containerName := "containerFoo"

	container := v1.Container{
		Name: containerName,
		Lifecycle: &v1.Lifecycle{
			PostStart: &v1.Handler{
				HTTPGet: &v1.HTTPGetAction{
					Host: "foo",
					Port: intstr.FromInt(8080),
					Path: "bar",
				},
			},
		},
	}
	pod := v1.Pod{}
	pod.ObjectMeta.Name = "podFoo"
	pod.ObjectMeta.Namespace = "nsFoo"
	pod.Spec.Containers = []v1.Container{container}
	_, err := handlerRunner.Run(containerID, &pod, &container, container.Lifecycle.PostStart)

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if fakeHTTPGetter.url != "http://foo:8080/bar" {
		t.Errorf("unexpected url: %s", fakeHTTPGetter.url)
	}
}

func TestRunHandlerNil(t *testing.T) {
	handlerRunner := NewHandlerRunner(&fakeHTTP{}, &fakeContainerCommandRunner{}, nil)
	containerID := kubecontainer.ContainerID{Type: "test", ID: "abc1234"}
	podName := "podFoo"
	podNamespace := "nsFoo"
	containerName := "containerFoo"

	container := v1.Container{
		Name: containerName,
		Lifecycle: &v1.Lifecycle{
			PostStart: &v1.Handler{},
		},
	}
	pod := v1.Pod{}
	pod.ObjectMeta.Name = podName
	pod.ObjectMeta.Namespace = podNamespace
	pod.Spec.Containers = []v1.Container{container}
	_, err := handlerRunner.Run(containerID, &pod, &container, container.Lifecycle.PostStart)
	if err == nil {
		t.Errorf("expect error, but got nil")
	}
}

func TestRunHandlerExecFailure(t *testing.T) {
	expectedErr := fmt.Errorf("invalid command")
	fakeCommandRunner := fakeContainerCommandRunner{Err: expectedErr, Msg: expectedErr.Error()}
	handlerRunner := NewHandlerRunner(&fakeHTTP{}, &fakeCommandRunner, nil)

	containerID := kubecontainer.ContainerID{Type: "test", ID: "abc1234"}
	containerName := "containerFoo"
	command := []string{"ls", "--a"}

	container := v1.Container{
		Name: containerName,
		Lifecycle: &v1.Lifecycle{
			PostStart: &v1.Handler{
				Exec: &v1.ExecAction{
					Command: command,
				},
			},
		},
	}

	pod := v1.Pod{}
	pod.ObjectMeta.Name = "podFoo"
	pod.ObjectMeta.Namespace = "nsFoo"
	pod.Spec.Containers = []v1.Container{container}
	expectedErrMsg := fmt.Sprintf("Exec lifecycle hook (%s) for Container %q in Pod %q failed - error: %v, message: %q", command, containerName, format.Pod(&pod), expectedErr, expectedErr.Error())
	msg, err := handlerRunner.Run(containerID, &pod, &container, container.Lifecycle.PostStart)
	if err == nil {
		t.Errorf("expected error: %v", expectedErr)
	}
	if msg != expectedErrMsg {
		t.Errorf("unexpected error message: %q; expected %q", msg, expectedErrMsg)
	}
}

func TestRunHandlerHttpFailure(t *testing.T) {
	expectedErr := fmt.Errorf("fake http error")
	expectedResp := http.Response{
		Body: ioutil.NopCloser(strings.NewReader(expectedErr.Error())),
	}
	fakeHTTPGetter := fakeHTTP{err: expectedErr, resp: &expectedResp}
	handlerRunner := NewHandlerRunner(&fakeHTTPGetter, &fakeContainerCommandRunner{}, nil)
	containerName := "containerFoo"
	containerID := kubecontainer.ContainerID{Type: "test", ID: "abc1234"}
	container := v1.Container{
		Name: containerName,
		Lifecycle: &v1.Lifecycle{
			PostStart: &v1.Handler{
				HTTPGet: &v1.HTTPGetAction{
					Host: "foo",
					Port: intstr.FromInt(8080),
					Path: "bar",
				},
			},
		},
	}
	pod := v1.Pod{}
	pod.ObjectMeta.Name = "podFoo"
	pod.ObjectMeta.Namespace = "nsFoo"
	pod.Spec.Containers = []v1.Container{container}
	expectedErrMsg := fmt.Sprintf("Http lifecycle hook (%s) for Container %q in Pod %q failed - error: %v, message: %q", "bar", containerName, format.Pod(&pod), expectedErr, expectedErr.Error())
	msg, err := handlerRunner.Run(containerID, &pod, &container, container.Lifecycle.PostStart)
	if err == nil {
		t.Errorf("expected error: %v", expectedErr)
	}
	if msg != expectedErrMsg {
		t.Errorf("unexpected error message: %q; expected %q", msg, expectedErrMsg)
	}
	if fakeHTTPGetter.url != "http://foo:8080/bar" {
		t.Errorf("unexpected url: %s", fakeHTTPGetter.url)
	}
}

func TestNoNewPrivsRequired(t *testing.T) {
	for _, testCase := range []struct {
		input    *v1.Pod
		expected bool
	}{
		{
			input: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							SecurityContext: &v1.SecurityContext{
								AllowPrivilegeEscalation: pointer.BoolPtr(false),
							},
						},
					},
				},
			},
			expected: true,
		},
		{
			input:    &v1.Pod{},
			expected: false,
		},
	} {
		res := noNewPrivsRequired(testCase.input)
		if res != testCase.expected {
			t.Errorf("expected admission result for pod to be %t, got %t", res, testCase.expected)
		}
	}
}

func TestNoNewPrivsAdmitHandler(t *testing.T) {
	for _, testCase := range []struct {
		name         string
		runtime      *containertesting.FakeRuntime
		podAdmitAttr *PodAdmitAttributes
		expected     PodAdmitResult
	}{
		{
			name:    "Pod with status different than 'Pending'",
			runtime: &containertesting.FakeRuntime{},
			podAdmitAttr: &PodAdmitAttributes{
				Pod: &v1.Pod{
					Status: v1.PodStatus{Phase: v1.PodRunning},
				},
			},
			expected: PodAdmitResult{Admit: true},
		},
		{
			name:    "'Pending' status and AllowPrivilegeEscalation: true",
			runtime: &containertesting.FakeRuntime{},
			podAdmitAttr: &PodAdmitAttributes{
				Pod: &v1.Pod{
					Status: v1.PodStatus{
						Phase: v1.PodPending,
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								SecurityContext: &v1.SecurityContext{
									AllowPrivilegeEscalation: pointer.BoolPtr(true),
								},
							},
						},
					},
				},
			},
			expected: PodAdmitResult{Admit: true},
		},
		{
			name: "Docker runtime version => 1.23.0",
			runtime: &containertesting.FakeRuntime{
				RuntimeType:    kubetypes.DockerContainerRuntime,
				APIVersionInfo: "1.23.0",
			},
			podAdmitAttr: &PodAdmitAttributes{
				Pod: &v1.Pod{
					Status: v1.PodStatus{
						Phase: v1.PodPending,
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								SecurityContext: &v1.SecurityContext{
									AllowPrivilegeEscalation: pointer.BoolPtr(false),
								},
							},
						},
					},
				},
			},
			expected: PodAdmitResult{Admit: true},
		},
		{
			name: "Docker runtime version < 1.23.0",
			runtime: &containertesting.FakeRuntime{
				RuntimeType:    kubetypes.DockerContainerRuntime,
				APIVersionInfo: "1.20.0",
			},
			podAdmitAttr: &PodAdmitAttributes{
				Pod: &v1.Pod{
					Status: v1.PodStatus{
						Phase: v1.PodPending,
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								SecurityContext: &v1.SecurityContext{
									AllowPrivilegeEscalation: pointer.BoolPtr(false),
								},
							},
						},
					},
				},
			},
			expected: PodAdmitResult{
				Admit:  false,
				Reason: "NoNewPrivs",
			},
		},
		{
			name: "Docker runtime with invalid api version",
			runtime: &containertesting.FakeRuntime{
				RuntimeType: kubetypes.DockerContainerRuntime,
				Err:         fmt.Errorf("failed to parse docker runtime version"),
			},
			podAdmitAttr: &PodAdmitAttributes{
				Pod: &v1.Pod{
					Status: v1.PodStatus{
						Phase: v1.PodPending,
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								SecurityContext: &v1.SecurityContext{
									AllowPrivilegeEscalation: pointer.BoolPtr(false),
								},
							},
						},
					},
				},
			},
			expected: PodAdmitResult{
				Admit:   false,
				Reason:  "NoNewPrivs",
				Message: "Cannot enforce NoNewPrivs: failed to parse docker runtime version",
			},
		},
		{
			name:    "Remote runtime",
			runtime: &containertesting.FakeRuntime{RuntimeType: kubetypes.RemoteContainerRuntime},
			podAdmitAttr: &PodAdmitAttributes{
				Pod: &v1.Pod{
					Status: v1.PodStatus{
						Phase: v1.PodPending,
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								SecurityContext: &v1.SecurityContext{
									AllowPrivilegeEscalation: pointer.BoolPtr(false),
								},
							},
						},
					},
				},
			},
			expected: PodAdmitResult{Admit: true},
		},
	} {
		noNewPrivsAdminHandler := NewNoNewPrivsAdmitHandler(testCase.runtime)
		res := noNewPrivsAdminHandler.Admit(testCase.podAdmitAttr)
		if testCase.expected.Admit != res.Admit {
			t.Errorf("[%s] expected pod admission to be %t, got %t", testCase.name, testCase.expected.Admit, res.Admit)
		}
		if testCase.expected.Reason != res.Reason {
			t.Errorf("[%s]: expected pod admission failure reason to be %s, got %s", testCase.name, testCase.expected.Reason, res.Reason)
		}
		if testCase.expected.Message != "" && testCase.expected.Message != res.Message {
			t.Errorf("[%s]: expected admission failure message to be %s, got: %s", testCase.name, testCase.expected.Message, res.Message)
		}
	}
}

type fakeAppArmorValidator struct {
	err error
}

func (f *fakeAppArmorValidator) Validate(pod *v1.Pod) error {
	if f.err != nil {
		return f.err
	}
	return nil
}

func (f *fakeAppArmorValidator) ValidateHost() error { return nil }

func TestAppArmorAdmitHandler(t *testing.T) {
	for _, testCase := range []struct {
		name          string
		podAdmitAttr  *PodAdmitAttributes
		expected      *PodAdmitResult
		fakeValidator *fakeAppArmorValidator
	}{
		{
			name: "Pod with status different than 'Pending'",
			podAdmitAttr: &PodAdmitAttributes{
				Pod: &v1.Pod{
					Status: v1.PodStatus{
						Phase: v1.PodRunning,
					},
				},
			},
			fakeValidator: &fakeAppArmorValidator{},
			expected:      &PodAdmitResult{Admit: true},
		},
		{
			name: "Pod with status 'Pending'",
			podAdmitAttr: &PodAdmitAttributes{
				Pod: &v1.Pod{
					Status: v1.PodStatus{
						Phase: v1.PodPending,
					},
				},
			},
			fakeValidator: &fakeAppArmorValidator{},
			expected:      &PodAdmitResult{Admit: true},
		},
		{
			name: "Pod that fails validation",
			podAdmitAttr: &PodAdmitAttributes{
				Pod: &v1.Pod{
					Status: v1.PodStatus{
						Phase: v1.PodPending,
					},
				},
			},
			fakeValidator: &fakeAppArmorValidator{err: fmt.Errorf("failed validation")},
			expected:      &PodAdmitResult{Admit: false},
		},
	} {
		noNewPrivsAdminHandler := NewAppArmorAdmitHandler(testCase.fakeValidator)
		res := noNewPrivsAdminHandler.Admit(testCase.podAdmitAttr)
		if testCase.expected.Admit != res.Admit {
			t.Errorf("[%s] expected pod admission to be %t but got %t", testCase.name, testCase.expected.Admit, res.Admit)
		}

	}

}

var (
	defaultProcMount  = v1.DefaultProcMount
	unmaskedProcMount = v1.UnmaskedProcMount
)

func TestProcMountAdmitHandler(t *testing.T) {
	for _, testCase := range []struct {
		name         string
		runtime      *containertesting.FakeRuntime
		podAdmitAttr *PodAdmitAttributes
		expected     PodAdmitResult
	}{
		{
			name:    "Pod with status different than 'Pending'",
			runtime: &containertesting.FakeRuntime{},
			podAdmitAttr: &PodAdmitAttributes{
				Pod: &v1.Pod{
					Status: v1.PodStatus{Phase: v1.PodRunning},
				},
			},
			expected: PodAdmitResult{Admit: true},
		},
		{
			name:    "'Pending' status and DefaultProcMount",
			runtime: &containertesting.FakeRuntime{},
			podAdmitAttr: &PodAdmitAttributes{
				Pod: &v1.Pod{
					Status: v1.PodStatus{
						Phase: v1.PodPending,
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								SecurityContext: &v1.SecurityContext{
									ProcMount: &defaultProcMount,
								},
							},
						},
					},
				},
			},
			expected: PodAdmitResult{Admit: true},
		},
		{
			name: "Docker runtime version => 1.38.0",
			runtime: &containertesting.FakeRuntime{
				RuntimeType:    kubetypes.DockerContainerRuntime,
				APIVersionInfo: "1.38.0",
			},
			podAdmitAttr: &PodAdmitAttributes{
				Pod: &v1.Pod{
					Status: v1.PodStatus{
						Phase: v1.PodPending,
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								SecurityContext: &v1.SecurityContext{
									ProcMount: &unmaskedProcMount,
								},
							},
						},
					},
				},
			},
			expected: PodAdmitResult{Admit: true},
		},
		{
			name: "Docker runtime version < 1.38.0",
			runtime: &containertesting.FakeRuntime{
				RuntimeType:    kubetypes.DockerContainerRuntime,
				APIVersionInfo: "1.20.0",
			},
			podAdmitAttr: &PodAdmitAttributes{
				Pod: &v1.Pod{
					Status: v1.PodStatus{
						Phase: v1.PodPending,
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								SecurityContext: &v1.SecurityContext{
									ProcMount: &unmaskedProcMount,
								},
							},
						},
					},
				},
			},
			expected: PodAdmitResult{
				Admit:  false,
				Reason: "ProcMount",
			},
		},
		{
			name: "Docker runtime with invalid api version",
			runtime: &containertesting.FakeRuntime{
				RuntimeType: kubetypes.DockerContainerRuntime,
				Err:         fmt.Errorf("failed to parse docker runtime version"),
			},
			podAdmitAttr: &PodAdmitAttributes{
				Pod: &v1.Pod{
					Status: v1.PodStatus{
						Phase: v1.PodPending,
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								SecurityContext: &v1.SecurityContext{
									ProcMount: &unmaskedProcMount,
								},
							},
						},
					},
				},
			},
			expected: PodAdmitResult{
				Admit:   false,
				Reason:  "ProcMount",
				Message: "Cannot enforce ProcMount: failed to parse docker runtime version",
			},
		},
		{
			name:    "Remote runtime",
			runtime: &containertesting.FakeRuntime{RuntimeType: kubetypes.RemoteContainerRuntime},
			podAdmitAttr: &PodAdmitAttributes{
				Pod: &v1.Pod{
					Status: v1.PodStatus{
						Phase: v1.PodPending,
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								SecurityContext: &v1.SecurityContext{
									ProcMount: &unmaskedProcMount,
								},
							},
						},
					},
				},
			},
			expected: PodAdmitResult{Admit: true},
		},
	} {
		procMountAdmitHandler := NewProcMountAdmitHandler(testCase.runtime)
		res := procMountAdmitHandler.Admit(testCase.podAdmitAttr)
		if testCase.expected.Admit != res.Admit {
			t.Errorf("[%s] expected pod admission to be %t, got %t", testCase.name, testCase.expected.Admit, res.Admit)
		}
		if testCase.expected.Reason != res.Reason {
			t.Errorf("[%s]: expected pod admission failure reason to be %s, got %s", testCase.name, testCase.expected.Reason, res.Reason)
		}
		if testCase.expected.Message != "" && testCase.expected.Message != res.Message {
			t.Errorf("[%s]: expected admission failure message to be %s, got: %s", testCase.name, testCase.expected.Message, res.Message)
		}
	}
}

func TestProcMountIsDefault(t *testing.T) {
	for _, testCase := range []struct {
		pod      *v1.Pod
		expected bool
	}{
		{
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							SecurityContext: &v1.SecurityContext{
								ProcMount: &defaultProcMount,
							},
						},
					},
				},
			},
			expected: true,
		},
		{
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							SecurityContext: &v1.SecurityContext{
								ProcMount: &unmaskedProcMount,
							},
						},
					},
				},
			},
			expected: false,
		},
	} {
		res := procMountIsDefault(testCase.pod)
		if testCase.expected != res {
			t.Errorf("expected result to be %t but got %t", testCase.expected, res)
		}
	}
}
