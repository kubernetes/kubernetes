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
	"net/http"
	"net/url"
	"os"
	"strings"
	"testing"

	"github.com/spf13/cobra"

	"k8s.io/client-go/rest/fake"
	"k8s.io/kubernetes/pkg/api"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
)

type fakePortForwarder struct {
	method string
	url    *url.URL
	pfErr  error
}

func (f *fakePortForwarder) ForwardPorts(method string, url *url.URL, opts PortForwardOptions) error {
	f.method = method
	f.url = url
	return f.pfErr
}

func testPortForward(t *testing.T, flags map[string]string, args []string) {
	version := api.Registry.GroupOrDie(api.GroupName).GroupVersion.Version

	tests := []struct {
		name                       string
		podPath, pfPath, container string
		pod                        *api.Pod
		pfErr                      bool
	}{
		{
			name:    "pod portforward",
			podPath: "/api/" + version + "/namespaces/test/pods/foo",
			pfPath:  "/api/" + version + "/namespaces/test/pods/foo/portforward",
			pod:     execPod(),
		},
		{
			name:    "pod portforward error",
			podPath: "/api/" + version + "/namespaces/test/pods/foo",
			pfPath:  "/api/" + version + "/namespaces/test/pods/foo/portforward",
			pod:     execPod(),
			pfErr:   true,
		},
	}
	for _, test := range tests {
		var err error
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
					// Ensures no GET is performed when deleting by name
					t.Errorf("%s: unexpected request: %#v\n%#v", test.name, req.URL, req)
					return nil, nil
				}
			}),
		}
		tf.Namespace = "test"
		tf.ClientConfig = defaultClientConfig()
		ff := &fakePortForwarder{}
		if test.pfErr {
			ff.pfErr = fmt.Errorf("pf error")
		}

		opts := &PortForwardOptions{}
		cmd := NewCmdPortForward(f, os.Stdout, os.Stderr)
		cmd.Run = func(cmd *cobra.Command, args []string) {
			if err = opts.Complete(f, cmd, args); err != nil {
				return
			}
			opts.PortForwarder = ff
			if err = opts.Validate(); err != nil {
				return
			}
			err = opts.RunPortForward()
		}

		for name, value := range flags {
			cmd.Flags().Set(name, value)
		}
		cmd.Run(cmd, args)

		if test.pfErr && err != ff.pfErr {
			t.Errorf("%s: Unexpected port-forward error: %v", test.name, err)
		}
		if !test.pfErr && err != nil {
			t.Errorf("%s: Unexpected error: %v", test.name, err)
		}
		if test.pfErr {
			continue
		}

		if ff.url.Path != test.pfPath {
			t.Errorf("%s: Did not get expected path for portforward request", test.name)
		}
		if ff.method != "POST" {
			t.Errorf("%s: Did not get method for attach request: %s", test.name, ff.method)
		}
	}
}

func TestPortForward(t *testing.T) {
	testPortForward(t, nil, []string{"foo", ":5000", ":1000"})
}

func TestPortForwardWithPFlag(t *testing.T) {
	testPortForward(t, map[string]string{"pod": "foo"}, []string{":5000", ":1000"})
}

func TestValidate(t *testing.T) {
	f, tf, _, _ := cmdtesting.NewAPIFactory()
	tf.Client = &fake.RESTClient{}
	tf.ClientConfig = defaultClientConfig()

	tests := []struct {
		name     string
		args     []string
		expected string
	}{
		{
			name:     "Test for pod name is null",
			args:     []string{""},
			expected: "pod name must be specified",
		},
		{
			name:     "Test for port is null",
			args:     []string{"my-pod"},
			expected: "at least 1 PORT is required for port-forward",
		},
	}
	for _, test := range tests {
		buf := bytes.NewBuffer([]byte{})
		cmd := NewCmdPortForward(f, buf, buf)
		out := ""

		o := &PortForwardOptions{}
		cmd.Run = func(cmd *cobra.Command, args []string) {
			o.Complete(f, cmd, args)
			out = o.Validate().Error()
		}
		cmd.Run(cmd, test.args)

		if !strings.Contains(out, test.expected) {
			t.Errorf("%s: expected to find:\n\t%s\nfound:\n\t%s\n", test.name, test.expected, out)
		}
	}
}
