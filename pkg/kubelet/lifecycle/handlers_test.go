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
	"net"
	"net/http"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
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
			PostStart: &v1.LifecycleHandler{
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
	url     string
	headers http.Header
	err     error
	resp    *http.Response
}

func (f *fakeHTTP) Do(req *http.Request) (*http.Response, error) {
	f.url = req.URL.String()
	f.headers = req.Header.Clone()
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
			PostStart: &v1.LifecycleHandler{
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

func TestRunHandlerHttpWithHeaders(t *testing.T) {
	fakeHTTPDoer := fakeHTTP{}
	handlerRunner := NewHandlerRunner(&fakeHTTPDoer, &fakeContainerCommandRunner{}, nil)

	containerID := kubecontainer.ContainerID{Type: "test", ID: "abc1234"}
	containerName := "containerFoo"

	container := v1.Container{
		Name: containerName,
		Lifecycle: &v1.Lifecycle{
			PostStart: &v1.Handler{
				HTTPGet: &v1.HTTPGetAction{
					Host: "foo",
					Port: intstr.FromInt(8080),
					Path: "/bar",
					HTTPHeaders: []v1.HTTPHeader{
						{Name: "Foo", Value: "bar"},
					},
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
	if fakeHTTPDoer.url != "http://foo:8080/bar" {
		t.Errorf("unexpected url: %s", fakeHTTPDoer.url)
	}
	if fakeHTTPDoer.headers["Foo"][0] != "bar" {
		t.Errorf("missing http header: %s", fakeHTTPDoer.headers)
	}
}

func TestRunHandlerHttps(t *testing.T) {

	fakeHTTPDoer := fakeHTTP{}
	handlerRunner := NewHandlerRunner(&fakeHTTPDoer, &fakeContainerCommandRunner{}, nil)

	containerID := kubecontainer.ContainerID{Type: "test", ID: "abc1234"}
	containerName := "containerFoo"

	container := v1.Container{
		Name: containerName,
		Lifecycle: &v1.Lifecycle{
			PostStart: &v1.Handler{
				HTTPGet: &v1.HTTPGetAction{
					Scheme: v1.URISchemeHTTPS,
					Host:   "foo",
					Port:   intstr.FromString(""),
					Path:   "bar",
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
	if fakeHTTPDoer.url != "https://foo:443/bar" {
		t.Errorf("unexpected url: %s", fakeHTTPDoer.url)
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
			PostStart: &v1.LifecycleHandler{},
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
			PostStart: &v1.LifecycleHandler{
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
			PostStart: &v1.LifecycleHandler{
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
	expectedErrMsg := fmt.Sprintf("HTTP lifecycle hook (%s) for Container %q in Pod %q failed - error: %v, message: %q", "bar", containerName, format.Pod(&pod), expectedErr, expectedErr.Error())
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

func TestFormatURL(t *testing.T) {
	tt := []struct {
		name   string
		scheme string
		host   string
		port   int
		path   string
	}{
		{
			name:   "https-scheme",
			scheme: "https",
			host:   "test",
			port:   123,
			path:   "test",
		},
		{
			name:   "path",
			scheme: "http",
			host:   "test",
			port:   123,
			path:   "path",
		},
		{
			name:   "path?query=/@?:%2F",
			scheme: "http",
			host:   "test",
			port:   123,
			path:   "path?query=/@?:%2F",
		},
		{
			name:   "path?query=/@?:%",
			scheme: "http",
			host:   "test",
			port:   123,
			path:   "path?query=/@?:%",
		},
		{
			name:   "path?query=%GG",
			scheme: "http",
			host:   "test",
			port:   123,
			path:   "path?query=%GG",
		},
		{
			name:   "/test",
			scheme: "http",
			host:   "test",
			port:   123,
			path:   "/test",
		},
		{
			name:   "//test/path",
			scheme: "http",
			host:   "test",
			port:   123,
			path:   "//test/path",
		},
		{
			name:   "//test:123/path?query=/@?:%2F",
			scheme: "http",
			host:   "test",
			port:   123,
			path:   "//test:123/path?query=/@?:%2F",
		},
	}

	for _, test := range tt {
		t.Run(test.name, func(t *testing.T) {
			path := test.path
			if strings.HasPrefix(path, "/") {
				path = path[1:]
			}
			expectURL := fmt.Sprintf("%s://%s/%s", test.scheme, net.JoinHostPort(test.host, strconv.Itoa(test.port)), path)
			actualURL := formatURL(test.scheme, test.host, test.port, test.path)
			if expectURL != actualURL.String() {
				t.Errorf("expected url: %s\ngot url: %s", expectURL, actualURL)
			}
		})
	}
}
