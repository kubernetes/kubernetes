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

package cmd

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/spf13/cobra"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	"k8s.io/kubernetes/pkg/api"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/util/term"
)

type fakeRemoteAttach struct {
	method string
	url    *url.URL
	err    error
}

func (f *fakeRemoteAttach) Attach(method string, url *url.URL, config *restclient.Config, stdin io.Reader, stdout, stderr io.Writer, tty bool, terminalSizeQueue term.TerminalSizeQueue) error {
	f.method = method
	f.url = url
	return f.err
}

func TestPodAndContainerAttach(t *testing.T) {
	tests := []struct {
		args              []string
		p                 *AttachOptions
		name              string
		expectError       bool
		expectedPod       string
		expectedContainer string
		timeout           time.Duration
		obj               runtime.Object
	}{
		{
			p:           &AttachOptions{},
			expectError: true,
			name:        "empty",
			timeout:     1,
		},
		{
			p:           &AttachOptions{},
			args:        []string{"one", "two", "three"},
			expectError: true,
			name:        "too many args",
			timeout:     2,
		},
		{
			p:           &AttachOptions{},
			args:        []string{"foo"},
			expectedPod: "foo",
			name:        "no container, no flags",
			obj:         attachPod(),
			timeout:     defaultPodLogsTimeout,
		},
		{
			p:                 &AttachOptions{StreamOptions: StreamOptions{ContainerName: "bar"}},
			args:              []string{"foo"},
			expectedPod:       "foo",
			expectedContainer: "bar",
			name:              "container in flag",
			obj:               attachPod(),
			timeout:           10000000,
		},
		{
			p:                 &AttachOptions{StreamOptions: StreamOptions{ContainerName: "initfoo"}},
			args:              []string{"foo"},
			expectedPod:       "foo",
			expectedContainer: "initfoo",
			name:              "init container in flag",
			obj:               attachPod(),
			timeout:           30,
		},
		{
			p:           &AttachOptions{StreamOptions: StreamOptions{ContainerName: "bar"}},
			args:        []string{"foo", "-c", "wrong"},
			expectError: true,
			name:        "non-existing container in flag",
			obj:         attachPod(),
			timeout:     10,
		},
		{
			p:           &AttachOptions{},
			args:        []string{"pods", "foo"},
			expectedPod: "foo",
			name:        "no container, no flags, pods and name",
			obj:         attachPod(),
			timeout:     10000,
		},
		{
			p:           &AttachOptions{},
			args:        []string{"pod/foo"},
			expectedPod: "foo",
			name:        "no container, no flags, pod/name",
			obj:         attachPod(),
			timeout:     1,
		},
		{
			p:           &AttachOptions{},
			args:        []string{"pod/foo"},
			expectedPod: "foo",
			name:        "invalid get pod timeout value",
			obj:         attachPod(),
			expectError: true,
			timeout:     0,
		},
	}

	for _, test := range tests {
		f, tf, codec, ns := cmdtesting.NewAPIFactory()
		tf.Client = &fake.RESTClient{
			APIRegistry:          api.Registry,
			NegotiatedSerializer: ns,
			Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
				if test.obj != nil {
					return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(codec, test.obj)}, nil
				}
				return nil, nil
			}),
		}
		tf.Namespace = "test"
		tf.ClientConfig = defaultClientConfig()

		cmd := &cobra.Command{}
		options := test.p
		cmdutil.AddPodRunningTimeoutFlag(cmd, test.timeout)

		err := options.Complete(f, cmd, test.args)
		if test.expectError && err == nil {
			t.Errorf("unexpected non-error (%s)", test.name)
		}
		if !test.expectError && err != nil {
			t.Errorf("unexpected error: %v (%s)", err, test.name)
		}
		if err != nil {
			continue
		}
		if options.PodName != test.expectedPod {
			t.Errorf("expected: %s, got: %s (%s)", test.expectedPod, options.PodName, test.name)
		}
		if options.ContainerName != test.expectedContainer {
			t.Errorf("expected: %s, got: %s (%s)", test.expectedContainer, options.ContainerName, test.name)
		}
	}
}

func TestAttach(t *testing.T) {
	version := api.Registry.GroupOrDie(api.GroupName).GroupVersion.Version
	tests := []struct {
		name, version, podPath, fetchPodPath, attachPath, container string
		pod                                                         *api.Pod
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
		f, tf, codec, ns := cmdtesting.NewAPIFactory()
		tf.Client = &fake.RESTClient{
			APIRegistry:          api.Registry,
			NegotiatedSerializer: ns,
			Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
				switch p, m := req.URL.Path, req.Method; {
				case p == test.podPath && m == "GET":
					body := objBody(codec, test.pod)
					return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: body}, nil
				case p == test.fetchPodPath && m == "GET":
					body := objBody(codec, test.pod)
					return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: body}, nil
				default:
					// Ensures no GET is performed when deleting by name
					t.Errorf("%s: unexpected request: %s %#v\n%#v", p, req.Method, req.URL, req)
					return nil, fmt.Errorf("unexpected request")
				}
			}),
		}
		tf.Namespace = "test"
		tf.ClientConfig = &restclient.Config{APIPath: "/api", ContentConfig: restclient.ContentConfig{NegotiatedSerializer: api.Codecs, GroupVersion: &schema.GroupVersion{Version: test.version}}}
		bufOut := bytes.NewBuffer([]byte{})
		bufErr := bytes.NewBuffer([]byte{})
		bufIn := bytes.NewBuffer([]byte{})
		remoteAttach := &fakeRemoteAttach{}
		if test.remoteAttachErr {
			remoteAttach.err = fmt.Errorf("attach error")
		}
		params := &AttachOptions{
			StreamOptions: StreamOptions{
				ContainerName: test.container,
				In:            bufIn,
				Out:           bufOut,
				Err:           bufErr,
			},
			Attach:        remoteAttach,
			GetPodTimeout: 1000,
		}
		cmd := &cobra.Command{}
		cmdutil.AddPodRunningTimeoutFlag(cmd, 1000)
		if err := params.Complete(f, cmd, []string{"foo"}); err != nil {
			t.Fatal(err)
		}
		err := params.Run()
		if test.exepctedErr != "" && err.Error() != test.exepctedErr {
			t.Errorf("%s: Unexpected exec error: %v", test.name, err)
			continue
		}
		if test.exepctedErr == "" && err != nil {
			t.Errorf("%s: Unexpected error: %v", test.name, err)
			continue
		}
		if test.exepctedErr != "" {
			continue
		}
		if remoteAttach.url.Path != test.attachPath {
			t.Errorf("%s: Did not get expected path for exec request: %q %q", test.name, test.attachPath, remoteAttach.url.Path)
			continue
		}
		if remoteAttach.method != "POST" {
			t.Errorf("%s: Did not get method for attach request: %s", test.name, remoteAttach.method)
		}
		if remoteAttach.url.Query().Get("container") != "bar" {
			t.Errorf("%s: Did not have query parameters: %s", test.name, remoteAttach.url.Query())
		}
	}
}

func TestAttachWarnings(t *testing.T) {
	version := api.Registry.GroupOrDie(api.GroupName).GroupVersion.Version
	tests := []struct {
		name, container, version, podPath, fetchPodPath, expectedErr, expectedOut string
		pod                                                                       *api.Pod
		stdin, tty                                                                bool
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
		f, tf, codec, ns := cmdtesting.NewAPIFactory()
		tf.Client = &fake.RESTClient{
			APIRegistry:          api.Registry,
			NegotiatedSerializer: ns,
			Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
				switch p, m := req.URL.Path, req.Method; {
				case p == test.podPath && m == "GET":
					body := objBody(codec, test.pod)
					return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: body}, nil
				case p == test.fetchPodPath && m == "GET":
					body := objBody(codec, test.pod)
					return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: body}, nil
				default:
					t.Errorf("%s: unexpected request: %s %#v\n%#v", test.name, req.Method, req.URL, req)
					return nil, fmt.Errorf("unexpected request")
				}
			}),
		}
		tf.Namespace = "test"
		tf.ClientConfig = &restclient.Config{APIPath: "/api", ContentConfig: restclient.ContentConfig{NegotiatedSerializer: api.Codecs, GroupVersion: &schema.GroupVersion{Version: test.version}}}
		bufOut := bytes.NewBuffer([]byte{})
		bufErr := bytes.NewBuffer([]byte{})
		bufIn := bytes.NewBuffer([]byte{})
		ex := &fakeRemoteAttach{}
		params := &AttachOptions{
			StreamOptions: StreamOptions{
				ContainerName: test.container,
				In:            bufIn,
				Out:           bufOut,
				Err:           bufErr,
				Stdin:         test.stdin,
				TTY:           test.tty,
			},
			Attach:        ex,
			GetPodTimeout: 1000,
		}
		cmd := &cobra.Command{}
		cmdutil.AddPodRunningTimeoutFlag(cmd, 1000)
		if err := params.Complete(f, cmd, []string{"foo"}); err != nil {
			t.Fatal(err)
		}
		if err := params.Run(); err != nil {
			t.Fatal(err)
		}

		if test.stdin && test.tty {
			if !test.pod.Spec.Containers[0].TTY {
				if !strings.Contains(bufErr.String(), test.expectedErr) {
					t.Errorf("%s: Expected TTY fallback warning for attach request: %s", test.name, bufErr.String())
					continue
				}
			}
		}
	}
}

func attachPod() *api.Pod {
	return &api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test", ResourceVersion: "10"},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,
			Containers: []api.Container{
				{
					Name: "bar",
				},
			},
			InitContainers: []api.Container{
				{
					Name: "initfoo",
				},
			},
		},
		Status: api.PodStatus{
			Phase: api.PodRunning,
		},
	}
}
