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

package lifecycle

import (
	"io"
	"net/http"
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/util"
)

func TestResolvePortInt(t *testing.T) {
	expected := 80
	port, err := resolvePort(util.IntOrString{Kind: util.IntstrInt, IntVal: expected}, &api.Container{})
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
	container := &api.Container{
		Ports: []api.ContainerPort{
			{Name: name, ContainerPort: expected},
		},
	}
	port, err := resolvePort(util.IntOrString{Kind: util.IntstrString, StrVal: name}, container)
	if port != expected {
		t.Errorf("expected: %d, saw: %d", expected, port)
	}
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestResolvePortStringUnknown(t *testing.T) {
	expected := 80
	name := "foo"
	container := &api.Container{
		Ports: []api.ContainerPort{
			{Name: "bar", ContainerPort: expected},
		},
	}
	port, err := resolvePort(util.IntOrString{Kind: util.IntstrString, StrVal: name}, container)
	if port != -1 {
		t.Errorf("expected: -1, saw: %d", port)
	}
	if err == nil {
		t.Error("unexpected non-error")
	}
}

type fakeContainerCommandRunner struct {
	Cmd []string
	ID  string
}

func (f *fakeContainerCommandRunner) RunInContainer(id string, cmd []string) ([]byte, error) {
	f.Cmd = cmd
	f.ID = id
	return []byte{}, nil
}

func (f *fakeContainerCommandRunner) ExecInContainer(id string, cmd []string, in io.Reader, out, err io.WriteCloser, tty bool) error {
	return nil
}

func (f *fakeContainerCommandRunner) PortForward(pod *kubecontainer.Pod, port uint16, stream io.ReadWriteCloser) error {
	return nil
}

func TestRunHandlerExec(t *testing.T) {
	fakeCommandRunner := fakeContainerCommandRunner{}
	handlerRunner := NewHandlerRunner(&fakeHTTP{}, &fakeCommandRunner, nil)

	containerID := "abc1234"
	containerName := "containerFoo"

	container := api.Container{
		Name: containerName,
		Lifecycle: &api.Lifecycle{
			PostStart: &api.Handler{
				Exec: &api.ExecAction{
					Command: []string{"ls", "-a"},
				},
			},
		},
	}

	pod := api.Pod{}
	pod.ObjectMeta.Name = "podFoo"
	pod.ObjectMeta.Namespace = "nsFoo"
	pod.Spec.Containers = []api.Container{container}
	err := handlerRunner.Run(containerID, &pod, &container, container.Lifecycle.PostStart)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if fakeCommandRunner.ID != containerID ||
		!reflect.DeepEqual(container.Lifecycle.PostStart.Exec.Command, fakeCommandRunner.Cmd) {
		t.Errorf("unexpected commands: %v", fakeCommandRunner)
	}
}

type fakeHTTP struct {
	url string
	err error
}

func (f *fakeHTTP) Get(url string) (*http.Response, error) {
	f.url = url
	return nil, f.err
}

func TestRunHandlerHttp(t *testing.T) {
	fakeHttp := fakeHTTP{}
	handlerRunner := NewHandlerRunner(&fakeHttp, &fakeContainerCommandRunner{}, nil)

	containerID := "abc1234"
	containerName := "containerFoo"

	container := api.Container{
		Name: containerName,
		Lifecycle: &api.Lifecycle{
			PostStart: &api.Handler{
				HTTPGet: &api.HTTPGetAction{
					Host: "foo",
					Port: util.IntOrString{IntVal: 8080, Kind: util.IntstrInt},
					Path: "bar",
				},
			},
		},
	}
	pod := api.Pod{}
	pod.ObjectMeta.Name = "podFoo"
	pod.ObjectMeta.Namespace = "nsFoo"
	pod.Spec.Containers = []api.Container{container}
	err := handlerRunner.Run(containerID, &pod, &container, container.Lifecycle.PostStart)

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if fakeHttp.url != "http://foo:8080/bar" {
		t.Errorf("unexpected url: %s", fakeHttp.url)
	}
}

func TestRunHandlerNil(t *testing.T) {
	handlerRunner := NewHandlerRunner(&fakeHTTP{}, &fakeContainerCommandRunner{}, nil)
	containerID := "abc1234"
	podName := "podFoo"
	podNamespace := "nsFoo"
	containerName := "containerFoo"

	container := api.Container{
		Name: containerName,
		Lifecycle: &api.Lifecycle{
			PostStart: &api.Handler{},
		},
	}
	pod := api.Pod{}
	pod.ObjectMeta.Name = podName
	pod.ObjectMeta.Namespace = podNamespace
	pod.Spec.Containers = []api.Container{container}
	err := handlerRunner.Run(containerID, &pod, &container, container.Lifecycle.PostStart)
	if err == nil {
		t.Errorf("expect error, but got nil")
	}
}
