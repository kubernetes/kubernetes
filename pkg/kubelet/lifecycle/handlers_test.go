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

	"k8s.io/kubernetes/pkg/api/v1"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/util/intstr"
)

func TestResolvePortInt(t *testing.T) {
	expected := 80
	port, err := resolvePort(intstr.FromInt(expected), &v1.Container{})
	if port != expected {
		t.Errorf("expected: %d, saw: %d", expected, port)
	}
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestResolvePortString(t *testing.T) {
	expected := 80
	name := "foo"
	container := &v1.Container{
		Ports: []v1.ContainerPort{
			{Name: name, ContainerPort: int32(expected)},
		},
	}
	port, err := resolvePort(intstr.FromString(name), container)
	if port != expected {
		t.Errorf("expected: %d, saw: %d", expected, port)
	}
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestResolvePortStringUnknown(t *testing.T) {
	expected := int32(80)
	name := "foo"
	container := &v1.Container{
		Ports: []v1.ContainerPort{
			{Name: "bar", ContainerPort: expected},
		},
	}
	port, err := resolvePort(intstr.FromString(name), container)
	if port != -1 {
		t.Errorf("expected: -1, saw: %d", port)
	}
	if err == nil {
		t.Error("unexpected non-error")
	}
}

type fakeContainerCommandRunner struct {
	Cmd []string
	ID  kubecontainer.ContainerID
}

func (f *fakeContainerCommandRunner) RunInContainer(id kubecontainer.ContainerID, cmd []string, timeout time.Duration) ([]byte, error) {
	f.Cmd = cmd
	f.ID = id
	return nil, nil
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
	fakeHttp := fakeHTTP{}
	handlerRunner := NewHandlerRunner(&fakeHttp, &fakeContainerCommandRunner{}, nil)

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
	if fakeHttp.url != "http://foo:8080/bar" {
		t.Errorf("unexpected url: %s", fakeHttp.url)
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

func TestRunHandlerHttpFailure(t *testing.T) {
	expectedErr := fmt.Errorf("fake http error")
	expectedResp := http.Response{
		Body: ioutil.NopCloser(strings.NewReader(expectedErr.Error())),
	}
	fakeHttp := fakeHTTP{err: expectedErr, resp: &expectedResp}
	handlerRunner := NewHandlerRunner(&fakeHttp, &fakeContainerCommandRunner{}, nil)
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
	msg, err := handlerRunner.Run(containerID, &pod, &container, container.Lifecycle.PostStart)
	if err == nil {
		t.Errorf("expected error: %v", expectedErr)
	}
	if msg != expectedErr.Error() {
		t.Errorf("unexpected error message: %q; expected %q", msg, expectedErr)
	}
	if fakeHttp.url != "http://foo:8080/bar" {
		t.Errorf("unexpected url: %s", fakeHttp.url)
	}
}
