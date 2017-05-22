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
	"reflect"
	"testing"

	"github.com/spf13/cobra"

	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	"k8s.io/client-go/tools/remotecommand"
	"k8s.io/kubernetes/pkg/api"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
)

type fakeRemoteDebugger struct {
	method   string
	url      *url.URL
	debugErr error
}

func (f *fakeRemoteDebugger) Debug(method string, url *url.URL, config *restclient.Config, stdin io.Reader, stdout, stderr io.Writer, tty bool, terminalSizeQueue remotecommand.TerminalSizeQueue) error {
	f.method = method
	f.url = url
	return f.debugErr
}

func TestDebugPodAndContainer(t *testing.T) {
	tests := []struct {
		args              []string
		argsLenAtDash     int
		p                 *DebugOptions
		name              string
		expectError       bool
		expectedPod       string
		expectedContainer string
		expectedArgs      []string
	}{
		{
			p:             &DebugOptions{},
			argsLenAtDash: -1,
			expectError:   true,
			name:          "empty",
		},
		{
			p:             &DebugOptions{},
			args:          []string{"foo", "cmd"},
			argsLenAtDash: 0,
			expectError:   true,
			name:          "no pod, pod name is behind dash",
		},
		{
			p:                 &DebugOptions{},
			args:              []string{"foo"},
			argsLenAtDash:     -1,
			expectError:       false,
			expectedPod:       "foo",
			expectedContainer: "debug",
			expectedArgs:      []string{},
			name:              "pod w/ default image & container",
		},
		{
			p:                 &DebugOptions{StreamOptions: StreamOptions{ContainerName: "bar"}},
			args:              []string{"foo"},
			argsLenAtDash:     -1,
			expectError:       false,
			expectedPod:       "foo",
			expectedContainer: "bar",
			expectedArgs:      []string{},
			name:              "pod w/ container",
		},
		{
			p:                 &DebugOptions{ImageName: "image"},
			args:              []string{"foo"},
			argsLenAtDash:     -1,
			expectError:       false,
			expectedPod:       "foo",
			expectedContainer: "debug",
			expectedArgs:      []string{},
			name:              "pod w/ image",
		},
		{
			p:                 &DebugOptions{},
			args:              []string{"foo", "cmd"},
			argsLenAtDash:     -1,
			expectedPod:       "foo",
			expectedContainer: "debug",
			expectedArgs:      []string{"cmd"},
			name:              "cmd, w/o flags",
		},
		{
			p:                 &DebugOptions{},
			args:              []string{"foo", "cmd"},
			argsLenAtDash:     1,
			expectedPod:       "foo",
			expectedContainer: "debug",
			expectedArgs:      []string{"cmd"},
			name:              "cmd, cmd is behind dash",
		},
		{
			p:                 &DebugOptions{StreamOptions: StreamOptions{ContainerName: "bar"}},
			args:              []string{"foo", "cmd"},
			argsLenAtDash:     -1,
			expectedPod:       "foo",
			expectedContainer: "bar",
			expectedArgs:      []string{"cmd"},
			name:              "cmd, container in flag",
		},
	}
	for _, test := range tests {
		f, tf, _, ns := cmdtesting.NewAPIFactory()
		tf.Client = &fake.RESTClient{
			APIRegistry:          api.Registry,
			NegotiatedSerializer: ns,
			Client:               fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) { return nil, nil }),
		}
		tf.Namespace = "test"
		tf.ClientConfig = defaultClientConfig()

		cmd := &cobra.Command{}
		options := test.p
		err := options.Complete(f, cmd, test.args, test.argsLenAtDash)
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
		if !reflect.DeepEqual(test.expectedArgs, options.Command) {
			t.Errorf("expected: %v, got %v (%s)", test.expectedArgs, options.Command, test.name)
		}
	}
}

func TestDebug(t *testing.T) {
	version := api.Registry.GroupOrDie(api.GroupName).GroupVersion.Version
	tests := []struct {
		name, podPath, debugPath, container string
		pod                                 *api.Pod
		debugErr                            bool
	}{
		{
			name:      "pod debug",
			podPath:   "/api/" + version + "/namespaces/test/pods/foo",
			debugPath: "/api/" + version + "/namespaces/test/pods/foo/debug",
			pod:       execPod(),
		},
		{
			name:      "pod debug error",
			podPath:   "/api/" + version + "/namespaces/test/pods/foo",
			debugPath: "/api/" + version + "/namespaces/test/pods/foo/debug",
			pod:       execPod(),
			debugErr:  true,
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
				default:
					t.Errorf("%s: unexpected request: %s %#v\n%#v", test.name, req.Method, req.URL, req)
					return nil, fmt.Errorf("unexpected request")
				}
			}),
		}
		tf.Namespace = "test"
		tf.ClientConfig = defaultClientConfig()
		bufOut := bytes.NewBuffer([]byte{})
		bufErr := bytes.NewBuffer([]byte{})
		bufIn := bytes.NewBuffer([]byte{})
		db := &fakeRemoteDebugger{}
		if test.debugErr {
			db.debugErr = fmt.Errorf("debug error")
		}
		params := &DebugOptions{
			StreamOptions: StreamOptions{
				PodName:       "foo",
				ContainerName: "bar",
				In:            bufIn,
				Out:           bufOut,
				Err:           bufErr,
			},
			ImageName: "test-image",
			Debugger:  db,
		}
		cmd := &cobra.Command{}
		args := []string{"foo", "command", "arg"}
		if err := params.Complete(f, cmd, args, -1); err != nil {
			t.Fatal(err)
		}
		err := params.RunDebug()
		if test.debugErr && err != db.debugErr {
			t.Errorf("%s: Unexpected debug error: %v", test.name, err)
			continue
		}
		if !test.debugErr && err != nil {
			t.Errorf("%s: Unexpected error: %v", test.name, err)
			continue
		}
		if test.debugErr {
			continue
		}
		if db.url.Path != test.debugPath {
			t.Errorf("%s: Did not get expected path for debug request", test.name)
			continue
		}
		if db.method != "POST" {
			t.Errorf("%s: Did not get method for debug request: %s", test.name, db.method)
		}
	}
}
