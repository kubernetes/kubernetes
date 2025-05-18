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
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/record"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
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

func (f *fakeContainerCommandRunner) RunInContainer(_ context.Context, id kubecontainer.ContainerID, cmd []string, timeout time.Duration) ([]byte, error) {
	f.Cmd = cmd
	f.ID = id
	return []byte(f.Msg), f.Err
}

func stubPodStatusProvider(podIP string) podStatusProvider {
	return podStatusProviderFunc(func(uid types.UID, name, namespace string) (*kubecontainer.PodStatus, error) {
		return &kubecontainer.PodStatus{
			ID:        uid,
			Name:      name,
			Namespace: namespace,
			IPs:       []string{podIP},
		}, nil
	})
}

type podStatusProviderFunc func(uid types.UID, name, namespace string) (*kubecontainer.PodStatus, error)

func (f podStatusProviderFunc) GetPodStatus(_ context.Context, uid types.UID, name, namespace string) (*kubecontainer.PodStatus, error) {
	return f(uid, name, namespace)
}

func TestRunHandlerExec(t *testing.T) {
	ctx := context.Background()
	fakeCommandRunner := fakeContainerCommandRunner{}
	handlerRunner := NewHandlerRunner(&fakeHTTP{}, &fakeCommandRunner, nil, nil)

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
	_, err := handlerRunner.Run(ctx, containerID, &pod, &container, container.Lifecycle.PostStart)
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
	ctx := context.Background()
	fakeHTTPGetter := fakeHTTP{}
	fakePodStatusProvider := stubPodStatusProvider("127.0.0.1")
	handlerRunner := NewHandlerRunner(&fakeHTTPGetter, &fakeContainerCommandRunner{}, fakePodStatusProvider, nil)

	containerID := kubecontainer.ContainerID{Type: "test", ID: "abc1234"}
	containerName := "containerFoo"

	container := v1.Container{
		Name: containerName,
		Lifecycle: &v1.Lifecycle{
			PostStart: &v1.LifecycleHandler{
				HTTPGet: &v1.HTTPGetAction{
					Host: "foo",
					Port: intstr.FromInt32(8080),
					Path: "bar",
				},
			},
		},
	}
	pod := v1.Pod{}
	pod.ObjectMeta.Name = "podFoo"
	pod.ObjectMeta.Namespace = "nsFoo"
	pod.ObjectMeta.UID = "foo-bar-quux"
	pod.Spec.Containers = []v1.Container{container}
	_, err := handlerRunner.Run(ctx, containerID, &pod, &container, container.Lifecycle.PostStart)

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if fakeHTTPGetter.url != "http://foo:8080/bar" {
		t.Errorf("unexpected url: %s", fakeHTTPGetter.url)
	}
}

func TestRunHandlerHttpWithHeaders(t *testing.T) {
	ctx := context.Background()
	fakeHTTPDoer := fakeHTTP{}
	fakePodStatusProvider := stubPodStatusProvider("127.0.0.1")

	handlerRunner := NewHandlerRunner(&fakeHTTPDoer, &fakeContainerCommandRunner{}, fakePodStatusProvider, nil)

	containerID := kubecontainer.ContainerID{Type: "test", ID: "abc1234"}
	containerName := "containerFoo"

	container := v1.Container{
		Name: containerName,
		Lifecycle: &v1.Lifecycle{
			PostStart: &v1.LifecycleHandler{
				HTTPGet: &v1.HTTPGetAction{
					Host: "foo",
					Port: intstr.FromInt32(8080),
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
	_, err := handlerRunner.Run(ctx, containerID, &pod, &container, container.Lifecycle.PostStart)

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
	ctx := context.Background()
	fakeHTTPDoer := fakeHTTP{}
	fakePodStatusProvider := stubPodStatusProvider("127.0.0.1")
	handlerRunner := NewHandlerRunner(&fakeHTTPDoer, &fakeContainerCommandRunner{}, fakePodStatusProvider, nil)

	containerID := kubecontainer.ContainerID{Type: "test", ID: "abc1234"}
	containerName := "containerFoo"

	container := v1.Container{
		Name: containerName,
		Lifecycle: &v1.Lifecycle{
			PostStart: &v1.LifecycleHandler{
				HTTPGet: &v1.HTTPGetAction{
					Scheme: v1.URISchemeHTTPS,
					Host:   "foo",
					Path:   "bar",
				},
			},
		},
	}
	pod := v1.Pod{}
	pod.ObjectMeta.Name = "podFoo"
	pod.ObjectMeta.Namespace = "nsFoo"
	pod.Spec.Containers = []v1.Container{container}

	t.Run("consistent", func(t *testing.T) {
		container.Lifecycle.PostStart.HTTPGet.Port = intstr.FromString("70")
		pod.Spec.Containers = []v1.Container{container}
		_, err := handlerRunner.Run(ctx, containerID, &pod, &container, container.Lifecycle.PostStart)

		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if fakeHTTPDoer.url != "https://foo:70/bar" {
			t.Errorf("unexpected url: %s", fakeHTTPDoer.url)
		}
	})
}

func TestRunHandlerHTTPPort(t *testing.T) {
	tests := []struct {
		Name        string
		Port        intstr.IntOrString
		ExpectError bool
		Expected    string
	}{
		{
			Name:     "consistent/with port",
			Port:     intstr.FromString("70"),
			Expected: "https://foo:70/bar",
		}, {
			Name:        "consistent/without port",
			Port:        intstr.FromString(""),
			ExpectError: true,
		},
	}

	fakePodStatusProvider := stubPodStatusProvider("127.0.0.1")

	containerID := kubecontainer.ContainerID{Type: "test", ID: "abc1234"}
	containerName := "containerFoo"

	container := v1.Container{
		Name: containerName,
		Lifecycle: &v1.Lifecycle{
			PostStart: &v1.LifecycleHandler{
				HTTPGet: &v1.HTTPGetAction{
					Scheme: v1.URISchemeHTTPS,
					Host:   "foo",
					Port:   intstr.FromString("unexpected"),
					Path:   "bar",
				},
			},
		},
	}
	pod := v1.Pod{}
	pod.ObjectMeta.Name = "podFoo"
	pod.ObjectMeta.Namespace = "nsFoo"
	pod.Spec.Containers = []v1.Container{container}

	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			ctx := context.Background()
			fakeHTTPDoer := fakeHTTP{}
			handlerRunner := NewHandlerRunner(&fakeHTTPDoer, &fakeContainerCommandRunner{}, fakePodStatusProvider, nil)

			container.Lifecycle.PostStart.HTTPGet.Port = tt.Port
			pod.Spec.Containers = []v1.Container{container}
			_, err := handlerRunner.Run(ctx, containerID, &pod, &container, container.Lifecycle.PostStart)

			if hasError := (err != nil); hasError != tt.ExpectError {
				t.Errorf("unexpected error: %v", err)
			}

			if fakeHTTPDoer.url != tt.Expected {
				t.Errorf("unexpected url: %s", fakeHTTPDoer.url)
			}
		})
	}
}

func TestRunHTTPHandler(t *testing.T) {
	type expected struct {
		OldURL    string
		OldHeader http.Header
		NewURL    string
		NewHeader http.Header
	}

	tests := []struct {
		Name     string
		PodIP    string
		HTTPGet  *v1.HTTPGetAction
		Expected expected
	}{
		{
			Name:  "missing pod IP",
			PodIP: "",
			HTTPGet: &v1.HTTPGetAction{
				Path:        "foo",
				Port:        intstr.FromString("42"),
				Host:        "example.test",
				Scheme:      "http",
				HTTPHeaders: []v1.HTTPHeader{},
			},
			Expected: expected{
				OldURL:    "http://example.test:42/foo",
				OldHeader: http.Header{},
				NewURL:    "http://example.test:42/foo",
				NewHeader: http.Header{
					"Accept":     {"*/*"},
					"User-Agent": {"kube-lifecycle/."},
				},
			},
		}, {
			Name:  "missing host",
			PodIP: "233.252.0.1",
			HTTPGet: &v1.HTTPGetAction{
				Path:        "foo",
				Port:        intstr.FromString("42"),
				Scheme:      "http",
				HTTPHeaders: []v1.HTTPHeader{},
			},
			Expected: expected{
				OldURL:    "http://233.252.0.1:42/foo",
				OldHeader: http.Header{},
				NewURL:    "http://233.252.0.1:42/foo",
				NewHeader: http.Header{
					"Accept":     {"*/*"},
					"User-Agent": {"kube-lifecycle/."},
				},
			},
		}, {
			Name:  "path with leading slash",
			PodIP: "233.252.0.1",
			HTTPGet: &v1.HTTPGetAction{
				Path:        "/foo",
				Port:        intstr.FromString("42"),
				Scheme:      "http",
				HTTPHeaders: []v1.HTTPHeader{},
			},
			Expected: expected{
				OldURL:    "http://233.252.0.1:42//foo",
				OldHeader: http.Header{},
				NewURL:    "http://233.252.0.1:42/foo",
				NewHeader: http.Header{
					"Accept":     {"*/*"},
					"User-Agent": {"kube-lifecycle/."},
				},
			},
		}, {
			Name:  "path without leading slash",
			PodIP: "233.252.0.1",
			HTTPGet: &v1.HTTPGetAction{
				Path:        "foo",
				Port:        intstr.FromString("42"),
				Scheme:      "http",
				HTTPHeaders: []v1.HTTPHeader{},
			},
			Expected: expected{
				OldURL:    "http://233.252.0.1:42/foo",
				OldHeader: http.Header{},
				NewURL:    "http://233.252.0.1:42/foo",
				NewHeader: http.Header{
					"Accept":     {"*/*"},
					"User-Agent": {"kube-lifecycle/."},
				},
			},
		}, {
			Name:  "port resolution",
			PodIP: "233.252.0.1",
			HTTPGet: &v1.HTTPGetAction{
				Path:        "foo",
				Port:        intstr.FromString("quux"),
				Scheme:      "http",
				HTTPHeaders: []v1.HTTPHeader{},
			},
			Expected: expected{
				OldURL:    "http://233.252.0.1:8080/foo",
				OldHeader: http.Header{},
				NewURL:    "http://233.252.0.1:8080/foo",
				NewHeader: http.Header{
					"Accept":     {"*/*"},
					"User-Agent": {"kube-lifecycle/."},
				},
			},
		}, {
			Name:  "https",
			PodIP: "233.252.0.1",
			HTTPGet: &v1.HTTPGetAction{
				Path:        "foo",
				Port:        intstr.FromString("4430"),
				Scheme:      "https",
				HTTPHeaders: []v1.HTTPHeader{},
			},
			Expected: expected{
				OldURL:    "http://233.252.0.1:4430/foo",
				OldHeader: http.Header{},
				NewURL:    "https://233.252.0.1:4430/foo",
				NewHeader: http.Header{
					"Accept":     {"*/*"},
					"User-Agent": {"kube-lifecycle/."},
				},
			},
		}, {
			Name:  "unknown scheme",
			PodIP: "233.252.0.1",
			HTTPGet: &v1.HTTPGetAction{
				Path:        "foo",
				Port:        intstr.FromString("80"),
				Scheme:      "baz",
				HTTPHeaders: []v1.HTTPHeader{},
			},
			Expected: expected{
				OldURL:    "http://233.252.0.1:80/foo",
				OldHeader: http.Header{},
				NewURL:    "baz://233.252.0.1:80/foo",
				NewHeader: http.Header{
					"Accept":     {"*/*"},
					"User-Agent": {"kube-lifecycle/."},
				},
			},
		}, {
			Name:  "query param",
			PodIP: "233.252.0.1",
			HTTPGet: &v1.HTTPGetAction{
				Path:        "foo?k=v",
				Port:        intstr.FromString("80"),
				Scheme:      "http",
				HTTPHeaders: []v1.HTTPHeader{},
			},
			Expected: expected{
				OldURL:    "http://233.252.0.1:80/foo?k=v",
				OldHeader: http.Header{},
				NewURL:    "http://233.252.0.1:80/foo?k=v",
				NewHeader: http.Header{
					"Accept":     {"*/*"},
					"User-Agent": {"kube-lifecycle/."},
				},
			},
		}, {
			Name:  "fragment",
			PodIP: "233.252.0.1",
			HTTPGet: &v1.HTTPGetAction{
				Path:        "foo#frag",
				Port:        intstr.FromString("80"),
				Scheme:      "http",
				HTTPHeaders: []v1.HTTPHeader{},
			},
			Expected: expected{
				OldURL:    "http://233.252.0.1:80/foo#frag",
				OldHeader: http.Header{},
				NewURL:    "http://233.252.0.1:80/foo#frag",
				NewHeader: http.Header{
					"Accept":     {"*/*"},
					"User-Agent": {"kube-lifecycle/."},
				},
			},
		}, {
			Name:  "headers",
			PodIP: "233.252.0.1",
			HTTPGet: &v1.HTTPGetAction{
				Path:   "foo",
				Port:   intstr.FromString("80"),
				Scheme: "http",
				HTTPHeaders: []v1.HTTPHeader{
					{
						Name:  "Foo",
						Value: "bar",
					},
				},
			},
			Expected: expected{
				OldURL:    "http://233.252.0.1:80/foo",
				OldHeader: http.Header{},
				NewURL:    "http://233.252.0.1:80/foo",
				NewHeader: http.Header{
					"Accept":     {"*/*"},
					"Foo":        {"bar"},
					"User-Agent": {"kube-lifecycle/."},
				},
			},
		}, {
			Name:  "host header",
			PodIP: "233.252.0.1",
			HTTPGet: &v1.HTTPGetAction{
				Host:   "example.test",
				Path:   "foo",
				Port:   intstr.FromString("80"),
				Scheme: "http",
				HTTPHeaders: []v1.HTTPHeader{
					{
						Name:  "Host",
						Value: "from.header",
					},
				},
			},
			Expected: expected{
				OldURL:    "http://example.test:80/foo",
				OldHeader: http.Header{},
				NewURL:    "http://example.test:80/foo",
				NewHeader: http.Header{
					"Accept":     {"*/*"},
					"User-Agent": {"kube-lifecycle/."},
					"Host":       {"from.header"},
				},
			},
		},
	}

	containerID := kubecontainer.ContainerID{Type: "test", ID: "abc1234"}
	containerName := "containerFoo"

	container := v1.Container{
		Name: containerName,
		Lifecycle: &v1.Lifecycle{
			PostStart: &v1.LifecycleHandler{},
		},
		Ports: []v1.ContainerPort{
			{
				Name:          "quux",
				ContainerPort: 8080,
			},
		},
	}

	pod := v1.Pod{}
	pod.ObjectMeta.Name = "podFoo"
	pod.ObjectMeta.Namespace = "nsFoo"
	pod.Spec.Containers = []v1.Container{container}

	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			ctx := context.Background()
			fakePodStatusProvider := stubPodStatusProvider(tt.PodIP)

			container.Lifecycle.PostStart.HTTPGet = tt.HTTPGet
			pod.Spec.Containers = []v1.Container{container}

			verify := func(t *testing.T, expectedHeader http.Header, expectedURL string) {
				fakeHTTPDoer := fakeHTTP{}
				handlerRunner := NewHandlerRunner(&fakeHTTPDoer, &fakeContainerCommandRunner{}, fakePodStatusProvider, nil)

				_, err := handlerRunner.Run(ctx, containerID, &pod, &container, container.Lifecycle.PostStart)
				if err != nil {
					t.Fatal(err)
				}

				if diff := cmp.Diff(expectedHeader, fakeHTTPDoer.headers); diff != "" {
					t.Errorf("unexpected header (-want, +got)\n:%s", diff)
				}
				if fakeHTTPDoer.url != expectedURL {
					t.Errorf("url = %v; want %v", fakeHTTPDoer.url, tt.Expected.NewURL)
				}
			}

			t.Run("consistent", func(t *testing.T) {
				verify(t, tt.Expected.NewHeader, tt.Expected.NewURL)
			})
		})
	}
}

func TestRunHandlerNil(t *testing.T) {
	ctx := context.Background()
	handlerRunner := NewHandlerRunner(&fakeHTTP{}, &fakeContainerCommandRunner{}, nil, nil)
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
	_, err := handlerRunner.Run(ctx, containerID, &pod, &container, container.Lifecycle.PostStart)
	if err == nil {
		t.Errorf("expect error, but got nil")
	}
}

func TestRunHandlerExecFailure(t *testing.T) {
	ctx := context.Background()
	expectedErr := fmt.Errorf("invalid command")
	fakeCommandRunner := fakeContainerCommandRunner{Err: expectedErr, Msg: expectedErr.Error()}
	handlerRunner := NewHandlerRunner(&fakeHTTP{}, &fakeCommandRunner, nil, nil)

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
	msg, err := handlerRunner.Run(ctx, containerID, &pod, &container, container.Lifecycle.PostStart)
	if err == nil {
		t.Errorf("expected error: %v", expectedErr)
	}
	if msg != expectedErrMsg {
		t.Errorf("unexpected error message: %q; expected %q", msg, expectedErrMsg)
	}
}

func TestRunHandlerHttpFailure(t *testing.T) {
	ctx := context.Background()
	expectedErr := fmt.Errorf("fake http error")
	expectedResp := http.Response{
		Body: io.NopCloser(strings.NewReader(expectedErr.Error())),
	}
	fakeHTTPGetter := fakeHTTP{err: expectedErr, resp: &expectedResp}

	fakePodStatusProvider := stubPodStatusProvider("127.0.0.1")

	handlerRunner := NewHandlerRunner(&fakeHTTPGetter, &fakeContainerCommandRunner{}, fakePodStatusProvider, nil)

	containerName := "containerFoo"
	containerID := kubecontainer.ContainerID{Type: "test", ID: "abc1234"}
	container := v1.Container{
		Name: containerName,
		Lifecycle: &v1.Lifecycle{
			PostStart: &v1.LifecycleHandler{
				HTTPGet: &v1.HTTPGetAction{
					Host: "foo",
					Port: intstr.FromInt32(8080),
					Path: "bar",
				},
			},
		},
	}
	pod := v1.Pod{}
	pod.ObjectMeta.Name = "podFoo"
	pod.ObjectMeta.Namespace = "nsFoo"
	pod.Spec.Containers = []v1.Container{container}
	expectedErrMsg := fmt.Sprintf("HTTP lifecycle hook (%s) for Container %q in Pod %q failed - error: %v", "bar", containerName, format.Pod(&pod), expectedErr)
	msg, err := handlerRunner.Run(ctx, containerID, &pod, &container, container.Lifecycle.PostStart)
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

func TestRunHandlerHttpsFailureFallback(t *testing.T) {
	ctx := context.Background()

	// Since prometheus' gatherer is global, other tests may have updated metrics already, so
	// we need to reset them prior running this test.
	// This also implies that we can't run this test in parallel with other tests.
	metrics.Register()
	legacyregistry.Reset()

	var actualHeaders http.Header
	srv := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		actualHeaders = r.Header.Clone()
	}))
	defer srv.Close()
	_, port, err := net.SplitHostPort(srv.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}

	recorder := &record.FakeRecorder{Events: make(chan string, 10)}

	fakePodStatusProvider := stubPodStatusProvider("127.0.0.1")

	handlerRunner := NewHandlerRunner(srv.Client(), &fakeContainerCommandRunner{}, fakePodStatusProvider, recorder).(*handlerRunner)

	containerName := "containerFoo"
	containerID := kubecontainer.ContainerID{Type: "test", ID: "abc1234"}
	container := v1.Container{
		Name: containerName,
		Lifecycle: &v1.Lifecycle{
			PostStart: &v1.LifecycleHandler{
				HTTPGet: &v1.HTTPGetAction{
					// set the scheme to https to ensure it falls back to HTTP.
					Scheme: "https",
					Host:   "127.0.0.1",
					Port:   intstr.FromString(port),
					Path:   "bar",
					HTTPHeaders: []v1.HTTPHeader{
						{
							Name:  "Authorization",
							Value: "secret",
						},
					},
				},
			},
		},
	}
	pod := v1.Pod{}
	pod.ObjectMeta.Name = "podFoo"
	pod.ObjectMeta.Namespace = "nsFoo"
	pod.Spec.Containers = []v1.Container{container}
	msg, err := handlerRunner.Run(ctx, containerID, &pod, &container, container.Lifecycle.PostStart)

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if msg != "" {
		t.Errorf("unexpected error message: %q", msg)
	}
	if actualHeaders.Get("Authorization") != "" {
		t.Error("unexpected Authorization header")
	}

	expectedMetrics := `
# HELP kubelet_lifecycle_handler_http_fallbacks_total [ALPHA] The number of times lifecycle handlers successfully fell back to http from https.
# TYPE kubelet_lifecycle_handler_http_fallbacks_total counter
kubelet_lifecycle_handler_http_fallbacks_total 1
`

	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expectedMetrics), "kubelet_lifecycle_handler_http_fallbacks_total"); err != nil {
		t.Fatal(err)
	}

	select {
	case event := <-recorder.Events:
		if !strings.Contains(event, "LifecycleHTTPFallback") {
			t.Fatalf("expected LifecycleHTTPFallback event, got %q", event)
		}
	default:
		t.Fatal("no event recorded")
	}
}

func TestIsHTTPResponseError(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {}))
	defer s.Close()
	req, err := http.NewRequest("GET", s.URL, nil)
	if err != nil {
		t.Fatal(err)
	}
	req.URL.Scheme = "https"
	_, err = http.DefaultClient.Do(req)
	if !isHTTPResponseError(err) {
		t.Errorf("unexpected http response error: %v", err)
	}
}

func TestRunSleepHandler(t *testing.T) {
	handlerRunner := NewHandlerRunner(&fakeHTTP{}, &fakeContainerCommandRunner{}, nil, nil)
	containerID := kubecontainer.ContainerID{Type: "test", ID: "abc1234"}
	containerName := "containerFoo"
	container := v1.Container{
		Name: containerName,
		Lifecycle: &v1.Lifecycle{
			PreStop: &v1.LifecycleHandler{},
		},
	}
	pod := v1.Pod{}
	pod.ObjectMeta.Name = "podFoo"
	pod.ObjectMeta.Namespace = "nsFoo"
	pod.Spec.Containers = []v1.Container{container}

	tests := []struct {
		name                          string
		sleepSeconds                  int64
		terminationGracePeriodSeconds int64
		expectErr                     bool
		expectedErr                   string
	}{
		{
			name:                          "valid seconds",
			sleepSeconds:                  5,
			terminationGracePeriodSeconds: 30,
		},
		{
			name:                          "longer than TerminationGracePeriodSeconds",
			sleepSeconds:                  3,
			terminationGracePeriodSeconds: 2,
			expectErr:                     true,
			expectedErr:                   "container terminated before sleep hook finished",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLifecycleSleepAction, true)

			pod.Spec.Containers[0].Lifecycle.PreStop.Sleep = &v1.SleepAction{Seconds: tt.sleepSeconds}
			ctx, cancel := context.WithTimeout(context.Background(), time.Duration(tt.terminationGracePeriodSeconds)*time.Second)
			defer cancel()

			_, err := handlerRunner.Run(ctx, containerID, &pod, &container, container.Lifecycle.PreStop)

			if !tt.expectErr && err != nil {
				t.Errorf("unexpected success")
			}
			if tt.expectErr && err.Error() != tt.expectedErr {
				t.Errorf("%s: expected error want %s, got %s", tt.name, tt.expectedErr, err.Error())
			}
		})
	}
}

func TestRejectStaticPodAdmissionBasedOnServiceAccount(t *testing.T) {
	tests := []struct {
		name            string
		pod             *v1.Pod
		expectRejection bool
	}{
		{
			name:            "Empty ServiceAccount in Static Pod",
			pod:             &v1.Pod{Spec: v1.PodSpec{}},
			expectRejection: false,
		},
		{
			name:            "Non empty ServiceAccount",
			pod:             &v1.Pod{Spec: v1.PodSpec{ServiceAccountName: "default"}},
			expectRejection: true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actualResult := rejectStaticPodAdmissionBasedOnServiceAccount(test.pod)
			if test.expectRejection != actualResult {
				t.Errorf("unexpected result, expected %v but got %v", test.expectRejection, actualResult)
			}
		})
	}
}

func TestRejectStaticPodAdmissionBasedOnVolume(t *testing.T) {
	tests := []struct {
		name            string
		pod             *v1.Pod
		expectRejection bool
	}{
		{
			name:            "Empty Volume list",
			pod:             &v1.Pod{Spec: v1.PodSpec{Volumes: nil}},
			expectRejection: false,
		},
		{
			name: "Non empty volume list with HostPath volume",
			pod: &v1.Pod{Spec: v1.PodSpec{Volumes: []v1.Volume{
				{Name: "test-volume-hostpath", VolumeSource: v1.VolumeSource{HostPath: &v1.HostPathVolumeSource{}}},
			}}},
			expectRejection: false,
		},
		{
			name: "Non empty volume list with EmptyDir volume",
			pod: &v1.Pod{Spec: v1.PodSpec{Volumes: []v1.Volume{
				{Name: "test-volume-emptydir", VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}}},
			}}},
			expectRejection: false,
		},
		{
			name: "Non empty volume list with Secret volume",
			pod: &v1.Pod{Spec: v1.PodSpec{Volumes: []v1.Volume{
				{Name: "test-volume-secret", VolumeSource: v1.VolumeSource{Secret: &v1.SecretVolumeSource{}}},
			}}},
			expectRejection: true,
		},
		{
			name: "Non empty volume list with ConfigMap volume",
			pod: &v1.Pod{Spec: v1.PodSpec{Volumes: []v1.Volume{
				{Name: "test-volume-configmap", VolumeSource: v1.VolumeSource{ConfigMap: &v1.ConfigMapVolumeSource{}}},
			}}},
			expectRejection: true,
		},
		{
			name: "Non empty volume list with GCEPersistentDisk volume",
			pod: &v1.Pod{Spec: v1.PodSpec{Volumes: []v1.Volume{
				{Name: "test-volume-gce", VolumeSource: v1.VolumeSource{GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{}}},
			}}},
			expectRejection: true,
		},
		{
			name: "Non empty volume list with AWSElasticBlockStore volume",
			pod: &v1.Pod{Spec: v1.PodSpec{Volumes: []v1.Volume{
				{Name: "test-volume-aws", VolumeSource: v1.VolumeSource{AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{}}},
			}}},
			expectRejection: true,
		},
		{
			name: "Non empty volume list with GitRepo volume",
			pod: &v1.Pod{Spec: v1.PodSpec{Volumes: []v1.Volume{
				{Name: "test-volume-gitrepo", VolumeSource: v1.VolumeSource{GitRepo: &v1.GitRepoVolumeSource{}}},
			}}},
			expectRejection: true,
		},
		{
			name: "Non empty volume list with NFS volume",
			pod: &v1.Pod{Spec: v1.PodSpec{Volumes: []v1.Volume{
				{Name: "test-volume-nfs", VolumeSource: v1.VolumeSource{NFS: &v1.NFSVolumeSource{}}},
			}}},
			expectRejection: true,
		},
		{
			name: "Non empty volume list with ISCSI volume",
			pod: &v1.Pod{Spec: v1.PodSpec{Volumes: []v1.Volume{
				{Name: "test-volume-iscsi", VolumeSource: v1.VolumeSource{ISCSI: &v1.ISCSIVolumeSource{}}},
			}}},
			expectRejection: true,
		},
		{
			name: "Non empty volume list with Glusterfs volume",
			pod: &v1.Pod{Spec: v1.PodSpec{Volumes: []v1.Volume{
				{Name: "test-volume-glusterfs", VolumeSource: v1.VolumeSource{Glusterfs: &v1.GlusterfsVolumeSource{}}},
			}}},
			expectRejection: true,
		},
		{
			name: "Non empty volume list with PersistentVolumeClaim",
			pod: &v1.Pod{Spec: v1.PodSpec{Volumes: []v1.Volume{
				{Name: "test-volume-pvc", VolumeSource: v1.VolumeSource{PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{}}},
			}}},
			expectRejection: true,
		},
		{
			name: "Non empty volume list with RBD volume",
			pod: &v1.Pod{Spec: v1.PodSpec{Volumes: []v1.Volume{
				{Name: "test-volume-rbd", VolumeSource: v1.VolumeSource{RBD: &v1.RBDVolumeSource{}}},
			}}},
			expectRejection: true,
		},
		{
			name: "Non empty volume list with FlexVolume volume",
			pod: &v1.Pod{Spec: v1.PodSpec{Volumes: []v1.Volume{
				{Name: "test-volume-flexvolume", VolumeSource: v1.VolumeSource{FlexVolume: &v1.FlexVolumeSource{}}},
			}}},
			expectRejection: true,
		},
		{
			name: "Non empty volume list with Cinder volume",
			pod: &v1.Pod{Spec: v1.PodSpec{Volumes: []v1.Volume{
				{Name: "test-volume-cinder", VolumeSource: v1.VolumeSource{Cinder: &v1.CinderVolumeSource{}}},
			}}},
			expectRejection: true,
		},
		{
			name: "Non empty volume list with CephFS volume",
			pod: &v1.Pod{Spec: v1.PodSpec{Volumes: []v1.Volume{
				{Name: "test-volume-cephfs", VolumeSource: v1.VolumeSource{CephFS: &v1.CephFSVolumeSource{}}},
			}}},
			expectRejection: true,
		},
		{
			name: "Non empty volume list with Flocker volume",
			pod: &v1.Pod{Spec: v1.PodSpec{Volumes: []v1.Volume{
				{Name: "test-volume-flocker", VolumeSource: v1.VolumeSource{Flocker: &v1.FlockerVolumeSource{}}},
			}}},
			expectRejection: true,
		},
		{
			name: "Non empty volume list with DownwardAPI volume",
			pod: &v1.Pod{Spec: v1.PodSpec{Volumes: []v1.Volume{
				{Name: "test-volume-downwardapi", VolumeSource: v1.VolumeSource{DownwardAPI: &v1.DownwardAPIVolumeSource{}}},
			}}},
			expectRejection: true,
		},
		{
			name: "Non empty volume list with FC volume",
			pod: &v1.Pod{Spec: v1.PodSpec{Volumes: []v1.Volume{
				{Name: "test-volume-fc", VolumeSource: v1.VolumeSource{FC: &v1.FCVolumeSource{}}},
			}}},
			expectRejection: true,
		},
		{
			name: "Non empty volume list with AzureFile volume",
			pod: &v1.Pod{Spec: v1.PodSpec{Volumes: []v1.Volume{
				{Name: "test-volume-azurefile", VolumeSource: v1.VolumeSource{AzureFile: &v1.AzureFileVolumeSource{}}},
			}}},
			expectRejection: true,
		},
		{
			name: "Non empty volume list with VsphereVolume volume",
			pod: &v1.Pod{Spec: v1.PodSpec{Volumes: []v1.Volume{
				{Name: "test-volume-vsphere", VolumeSource: v1.VolumeSource{VsphereVolume: &v1.VsphereVirtualDiskVolumeSource{}}},
			}}},
			expectRejection: true,
		},
		{
			name: "Non empty volume list with Quobyte volume",
			pod: &v1.Pod{Spec: v1.PodSpec{Volumes: []v1.Volume{
				{Name: "test-volume-quobyte", VolumeSource: v1.VolumeSource{Quobyte: &v1.QuobyteVolumeSource{}}},
			}}},
			expectRejection: true,
		},
		{
			name: "Non empty volume list with AzureDisk volume",
			pod: &v1.Pod{Spec: v1.PodSpec{Volumes: []v1.Volume{
				{Name: "test-volume-azuredisk", VolumeSource: v1.VolumeSource{AzureDisk: &v1.AzureDiskVolumeSource{}}},
			}}},
			expectRejection: true,
		},
		{
			name: "Non empty volume list with PhotonPersistentDisk volume",
			pod: &v1.Pod{Spec: v1.PodSpec{Volumes: []v1.Volume{
				{Name: "test-volume-photon", VolumeSource: v1.VolumeSource{PhotonPersistentDisk: &v1.PhotonPersistentDiskVolumeSource{}}},
			}}},
			expectRejection: true,
		},
		{
			name: "Non empty volume list with Projected volume",
			pod: &v1.Pod{Spec: v1.PodSpec{Volumes: []v1.Volume{
				{Name: "test-volume-projected", VolumeSource: v1.VolumeSource{Projected: &v1.ProjectedVolumeSource{}}},
			}}},
			expectRejection: true,
		},
		{
			name: "Non empty volume list with Portworx volume",
			pod: &v1.Pod{Spec: v1.PodSpec{Volumes: []v1.Volume{
				{Name: "test-volume-portworx", VolumeSource: v1.VolumeSource{PortworxVolume: &v1.PortworxVolumeSource{}}},
			}}},
			expectRejection: true,
		},
		{
			name: "Non empty volume list with ScaleIO volume",
			pod: &v1.Pod{Spec: v1.PodSpec{Volumes: []v1.Volume{
				{Name: "test-volume-scaleio", VolumeSource: v1.VolumeSource{ScaleIO: &v1.ScaleIOVolumeSource{}}},
			}}},
			expectRejection: true,
		},
		{
			name: "Non empty volume list with StorageOS volume",
			pod: &v1.Pod{Spec: v1.PodSpec{Volumes: []v1.Volume{
				{Name: "test-volume-storageos", VolumeSource: v1.VolumeSource{StorageOS: &v1.StorageOSVolumeSource{}}},
			}}},
			expectRejection: true,
		},
		{
			name: "Non empty volume list with CSI volume",
			pod: &v1.Pod{Spec: v1.PodSpec{Volumes: []v1.Volume{
				{Name: "test-volume-csi", VolumeSource: v1.VolumeSource{CSI: &v1.CSIVolumeSource{}}},
			}}},
			expectRejection: true,
		},
		{
			name: "Non empty volume list with Ephemeral volume",
			pod: &v1.Pod{Spec: v1.PodSpec{Volumes: []v1.Volume{
				{Name: "test-volume-ephemeral", VolumeSource: v1.VolumeSource{Ephemeral: &v1.EphemeralVolumeSource{}}},
			}}},
			expectRejection: true,
		},
		{
			name: "Non empty volume list with Image volume",
			pod: &v1.Pod{Spec: v1.PodSpec{Volumes: []v1.Volume{
				{Name: "test-volume-image", VolumeSource: v1.VolumeSource{Image: &v1.ImageVolumeSource{}}},
			}}},
			expectRejection: true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actualResult := rejectStaticPodAdmissionBasedOnVolume(test.pod)
			if test.expectRejection != actualResult {
				t.Errorf("unexpected result, expected %v but got %v", test.expectRejection, actualResult)
			}
		})
	}
}

func TestRejectStaticPodAdmissionBasedOnEnv(t *testing.T) {
	tests := []struct {
		name            string
		pod             *v1.Pod
		expectRejection bool
	}{
		{
			name:            "No envs",
			pod:             &v1.Pod{Spec: v1.PodSpec{}},
			expectRejection: false,
		},
		{
			name: "Non empty Env with value",
			pod: &v1.Pod{Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name: "test-container",
						Env: []v1.EnvVar{
							{
								Name:  "test-env",
								Value: "TEST_ENV_VAL",
							},
						},
					},
				},
			}},
			expectRejection: false,
		},
		{
			name: "Non empty EnvFrom with ConfigMap",
			pod: &v1.Pod{Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name: "test-container",
						EnvFrom: []v1.EnvFromSource{
							{ConfigMapRef: &v1.ConfigMapEnvSource{}},
						},
					},
				},
			}},
			expectRejection: true,
		},
		{
			name: "Non empty EnvFrom with Secret",
			pod: &v1.Pod{Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name: "test-container",
						EnvFrom: []v1.EnvFromSource{
							{SecretRef: &v1.SecretEnvSource{}},
						},
					},
				},
			}},
			expectRejection: true,
		},
		{
			name: "Non empty Env with ConfigMap",
			pod: &v1.Pod{Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name: "test-container",
						Env: []v1.EnvVar{
							{
								Name:      "test-env",
								ValueFrom: &v1.EnvVarSource{ConfigMapKeyRef: &v1.ConfigMapKeySelector{}},
							},
						},
					},
				},
			}},
			expectRejection: true,
		},
		{
			name: "Non empty Env with Secret",
			pod: &v1.Pod{Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name: "test-container",
						Env: []v1.EnvVar{
							{
								Name:      "test-env",
								ValueFrom: &v1.EnvVarSource{SecretKeyRef: &v1.SecretKeySelector{}},
							},
						},
					},
				},
			}},
			expectRejection: true,
		},
		{
			name: "Non empty Env with FieldRef",
			pod: &v1.Pod{Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name: "test-container",
						Env: []v1.EnvVar{
							{
								Name:      "test-env",
								ValueFrom: &v1.EnvVarSource{FieldRef: &v1.ObjectFieldSelector{}},
							},
						},
					},
				},
			}},
			expectRejection: true,
		},
		{
			name: "Non empty Env with ResourceFieldRef",
			pod: &v1.Pod{Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name: "test-container",
						Env: []v1.EnvVar{
							{
								Name:      "test-env",
								ValueFrom: &v1.EnvVarSource{ResourceFieldRef: &v1.ResourceFieldSelector{}},
							},
						},
					},
				},
			}},
			expectRejection: true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actualResult := rejectStaticPodAdmissionBasedOnEnv(test.pod)
			if test.expectRejection != actualResult {
				t.Errorf("unexpected result, expected %v but got %v", test.expectRejection, actualResult)
			}
		})
	}
}
