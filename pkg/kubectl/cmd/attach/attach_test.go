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

package attach

import (
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	"k8s.io/client-go/tools/remotecommand"
	"k8s.io/kubernetes/pkg/kubectl/cmd/exec"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	"k8s.io/kubernetes/pkg/kubectl/polymorphichelpers"
	"k8s.io/kubernetes/pkg/kubectl/scheme"
)

type fakeRemoteAttach struct {
	method string
	url    *url.URL
	err    error
}

func (f *fakeRemoteAttach) Attach(method string, url *url.URL, config *restclient.Config, stdin io.Reader, stdout, stderr io.Writer, tty bool, terminalSizeQueue remotecommand.TerminalSizeQueue) error {
	f.method = method
	f.url = url
	return f.err
}

func fakeAttachablePodFn(pod *corev1.Pod) polymorphichelpers.AttachableLogsForObjectFunc {
	return func(getter genericclioptions.RESTClientGetter, obj runtime.Object, timeout time.Duration) (*corev1.Pod, error) {
		return pod, nil
	}
}

func TestPodAndContainerAttach(t *testing.T) {
	tests := []struct {
		name                  string
		args                  []string
		options               *AttachOptions
		expectError           string
		expectedPodName       string
		expectedContainerName string
		obj                   *corev1.Pod
	}{
		{
			name:        "empty",
			options:     &AttachOptions{GetPodTimeout: 1},
			expectError: "at least 1 argument is required",
		},
		{
			name:        "too many args",
			options:     &AttachOptions{GetPodTimeout: 2},
			args:        []string{"one", "two", "three"},
			expectError: "at most 2 arguments",
		},
		{
			name:                  "no container, no flags",
			options:               &AttachOptions{GetPodTimeout: defaultPodLogsTimeout},
			args:                  []string{"foo"},
			expectedPodName:       "foo",
			expectedContainerName: "bar",
			obj:                   attachPod(),
		},
		{
			name:                  "container in flag",
			options:               &AttachOptions{StreamOptions: exec.StreamOptions{ContainerName: "bar"}, GetPodTimeout: 10000000},
			args:                  []string{"foo"},
			expectedPodName:       "foo",
			expectedContainerName: "bar",
			obj:                   attachPod(),
		},
		{
			name:                  "init container in flag",
			options:               &AttachOptions{StreamOptions: exec.StreamOptions{ContainerName: "initfoo"}, GetPodTimeout: 30},
			args:                  []string{"foo"},
			expectedPodName:       "foo",
			expectedContainerName: "initfoo",
			obj:                   attachPod(),
		},
		{
			name:            "non-existing container",
			options:         &AttachOptions{StreamOptions: exec.StreamOptions{ContainerName: "wrong"}, GetPodTimeout: 10},
			args:            []string{"foo"},
			expectedPodName: "foo",
			expectError:     "container not found",
			obj:             attachPod(),
		},
		{
			name:                  "no container, no flags, pods and name",
			options:               &AttachOptions{GetPodTimeout: 10000},
			args:                  []string{"pods", "foo"},
			expectedPodName:       "foo",
			expectedContainerName: "bar",
			obj:                   attachPod(),
		},
		{
			name:                  "invalid get pod timeout value",
			options:               &AttachOptions{GetPodTimeout: 0},
			args:                  []string{"pod/foo"},
			expectedPodName:       "foo",
			expectedContainerName: "bar",
			obj:                   attachPod(),
			expectError:           "must be higher than zero",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// setup opts to fetch our test pod
			test.options.AttachablePodFn = fakeAttachablePodFn(test.obj)
			test.options.Resources = test.args

			if err := test.options.Validate(); err != nil {
				if !strings.Contains(err.Error(), test.expectError) {
					t.Errorf("unexpected error: expected %q, got %q", test.expectError, err)
				}
				return
			}

			pod, err := test.options.findAttachablePod(&corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "test"},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "foobar",
						},
					},
				},
			})
			if err != nil {
				if !strings.Contains(err.Error(), test.expectError) {
					t.Errorf("unexpected error: expected %q, got %q", err, test.expectError)
				}
				return
			}

			if pod.Name != test.expectedPodName {
				t.Errorf("unexpected pod name: expected %q, got %q", test.expectedContainerName, pod.Name)
			}

			container, err := test.options.containerToAttachTo(attachPod())
			if err != nil {
				if !strings.Contains(err.Error(), test.expectError) {
					t.Errorf("unexpected error: expected %q, got %q", err, test.expectError)
				}
				return
			}

			if container.Name != test.expectedContainerName {
				t.Errorf("unexpected container name: expected %q, got %q", test.expectedContainerName, container.Name)
			}

			if test.options.PodName != test.expectedPodName {
				t.Errorf("%s: expected: %s, got: %s", test.name, test.expectedPodName, test.options.PodName)
			}

			if len(test.expectError) > 0 {
				t.Fatalf("expected error %q, but saw none", test.expectError)
			}
		})
	}
}

func TestAttach(t *testing.T) {
	version := "v1"
	tests := []struct {
		name, version, podPath, fetchPodPath, attachPath, container string
		pod                                                         *corev1.Pod
		remoteAttachErr                                             bool
		exepctedErr                                                 string
	}{
		{
			name:         "pod attach",
			version:      version,
			podPath:      "/api/" + version + "/namespaces/test/pods/foo",
			fetchPodPath: "/namespaces/test/pods/foo",
			attachPath:   "/api/" + version + "/namespaces/test/pods/foo/attach",
			pod:          attachPod(),
			container:    "bar",
		},
		{
			name:            "pod attach error",
			version:         version,
			podPath:         "/api/" + version + "/namespaces/test/pods/foo",
			fetchPodPath:    "/namespaces/test/pods/foo",
			attachPath:      "/api/" + version + "/namespaces/test/pods/foo/attach",
			pod:             attachPod(),
			remoteAttachErr: true,
			container:       "bar",
			exepctedErr:     "attach error",
		},
		{
			name:         "container not found error",
			version:      version,
			podPath:      "/api/" + version + "/namespaces/test/pods/foo",
			fetchPodPath: "/namespaces/test/pods/foo",
			attachPath:   "/api/" + version + "/namespaces/test/pods/foo/attach",
			pod:          attachPod(),
			container:    "foo",
			exepctedErr:  "cannot attach to the container: container not found (foo)",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
			ns := scheme.Codecs

			tf.Client = &fake.RESTClient{
				GroupVersion:         schema.GroupVersion{Group: "", Version: "v1"},
				NegotiatedSerializer: ns,
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					switch p, m := req.URL.Path, req.Method; {
					case p == test.podPath && m == "GET":
						body := cmdtesting.ObjBody(codec, test.pod)
						return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: body}, nil
					case p == test.fetchPodPath && m == "GET":
						body := cmdtesting.ObjBody(codec, test.pod)
						return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: body}, nil
					default:
						t.Errorf("%s: unexpected request: %s %#v\n%#v", p, req.Method, req.URL, req)
						return nil, fmt.Errorf("unexpected request")
					}
				}),
			}
			tf.ClientConfigVal = &restclient.Config{APIPath: "/api", ContentConfig: restclient.ContentConfig{NegotiatedSerializer: scheme.Codecs, GroupVersion: &schema.GroupVersion{Version: test.version}}}

			remoteAttach := &fakeRemoteAttach{}
			if test.remoteAttachErr {
				remoteAttach.err = fmt.Errorf("attach error")
			}
			options := &AttachOptions{
				StreamOptions: exec.StreamOptions{
					ContainerName: test.container,
					IOStreams:     genericclioptions.NewTestIOStreamsDiscard(),
				},
				Attach:        remoteAttach,
				GetPodTimeout: 1000,
			}

			options.restClientGetter = tf
			options.Namespace = "test"
			options.Resources = []string{"foo"}
			options.Builder = tf.NewBuilder
			options.AttachablePodFn = fakeAttachablePodFn(test.pod)
			options.AttachFunc = func(opts *AttachOptions, containerToAttach *corev1.Container, raw bool, sizeQueue remotecommand.TerminalSizeQueue) func() error {
				return func() error {
					u, err := url.Parse(fmt.Sprintf("%s?container=%s", test.attachPath, containerToAttach.Name))
					if err != nil {
						return err
					}

					return options.Attach.Attach("POST", u, nil, nil, nil, nil, raw, sizeQueue)
				}
			}

			err := options.Run()
			if test.exepctedErr != "" && err.Error() != test.exepctedErr {
				t.Errorf("%s: Unexpected exec error: %v", test.name, err)
				return
			}
			if test.exepctedErr == "" && err != nil {
				t.Errorf("%s: Unexpected error: %v", test.name, err)
				return
			}
			if test.exepctedErr != "" {
				return
			}
			if remoteAttach.url.Path != test.attachPath {
				t.Errorf("%s: Did not get expected path for exec request: %q %q", test.name, test.attachPath, remoteAttach.url.Path)
				return
			}
			if remoteAttach.method != "POST" {
				t.Errorf("%s: Did not get method for attach request: %s", test.name, remoteAttach.method)
			}
			if remoteAttach.url.Query().Get("container") != "bar" {
				t.Errorf("%s: Did not have query parameters: %s", test.name, remoteAttach.url.Query())
			}
		})
	}
}

func TestAttachWarnings(t *testing.T) {
	version := "v1"
	tests := []struct {
		name, container, version, podPath, fetchPodPath, expectedErr string
		pod                                                          *corev1.Pod
		stdin, tty                                                   bool
	}{
		{
			name:         "fallback tty if not supported",
			version:      version,
			podPath:      "/api/" + version + "/namespaces/test/pods/foo",
			fetchPodPath: "/namespaces/test/pods/foo",
			pod:          attachPod(),
			stdin:        true,
			tty:          true,
			expectedErr:  "Unable to use a TTY - container bar did not allocate one",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			streams, _, _, bufErr := genericclioptions.NewTestIOStreams()

			codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
			ns := scheme.Codecs

			tf.Client = &fake.RESTClient{
				GroupVersion:         schema.GroupVersion{Group: "", Version: "v1"},
				NegotiatedSerializer: ns,
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					switch p, m := req.URL.Path, req.Method; {
					case p == test.podPath && m == "GET":
						body := cmdtesting.ObjBody(codec, test.pod)
						return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: body}, nil
					case p == test.fetchPodPath && m == "GET":
						body := cmdtesting.ObjBody(codec, test.pod)
						return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: body}, nil
					default:
						t.Errorf("%s: unexpected request: %s %#v\n%#v", p, req.Method, req.URL, req)
						return nil, fmt.Errorf("unexpected request")
					}
				}),
			}
			tf.ClientConfigVal = &restclient.Config{APIPath: "/api", ContentConfig: restclient.ContentConfig{NegotiatedSerializer: scheme.Codecs, GroupVersion: &schema.GroupVersion{Version: test.version}}}

			options := &AttachOptions{
				StreamOptions: exec.StreamOptions{
					Stdin:         test.stdin,
					TTY:           test.tty,
					ContainerName: test.container,
					IOStreams:     streams,
				},

				Attach:        &fakeRemoteAttach{},
				GetPodTimeout: 1000,
			}

			options.restClientGetter = tf
			options.Namespace = "test"
			options.Resources = []string{"foo"}
			options.Builder = tf.NewBuilder
			options.AttachablePodFn = fakeAttachablePodFn(test.pod)
			options.AttachFunc = func(opts *AttachOptions, containerToAttach *corev1.Container, raw bool, sizeQueue remotecommand.TerminalSizeQueue) func() error {
				return func() error {
					u, err := url.Parse("http://foo.bar")
					if err != nil {
						return err
					}

					return options.Attach.Attach("POST", u, nil, nil, nil, nil, raw, sizeQueue)
				}
			}

			if err := options.Run(); err != nil {
				t.Fatal(err)
			}

			if test.stdin && test.tty {
				if !test.pod.Spec.Containers[0].TTY {
					if !strings.Contains(bufErr.String(), test.expectedErr) {
						t.Errorf("%s: Expected TTY fallback warning for attach request: %s", test.name, bufErr.String())
						return
					}
				}
			}
		})
	}
}

func attachPod() *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test", ResourceVersion: "10"},
		Spec: corev1.PodSpec{
			RestartPolicy: corev1.RestartPolicyAlways,
			DNSPolicy:     corev1.DNSClusterFirst,
			Containers: []corev1.Container{
				{
					Name: "bar",
				},
			},
			InitContainers: []corev1.Container{
				{
					Name: "initfoo",
				},
			},
		},
		Status: corev1.PodStatus{
			Phase: corev1.PodRunning,
		},
	}
}
