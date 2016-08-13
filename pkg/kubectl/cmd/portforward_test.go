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
	"fmt"
	"net/http"
	"net/url"
	"os"
	"testing"

	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/unversioned/fake"
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

func TestPortForward(t *testing.T) {
	version := testapi.Default.GroupVersion().Version

	tests := []struct {
		name, version, podPath, pfPath, container string
		pod                                       *api.Pod
		pfErr                                     bool
	}{
		{
			name:    "pod portforward",
			version: version,
			podPath: "/api/" + version + "/namespaces/test/pods/foo",
			pfPath:  "/api/" + version + "/namespaces/test/pods/foo/portforward",
			pod:     execPod(),
		},
		{
			name:    "pod portforward error",
			version: version,
			podPath: "/api/" + version + "/namespaces/test/pods/foo",
			pfPath:  "/api/" + version + "/namespaces/test/pods/foo/portforward",
			pod:     execPod(),
			pfErr:   true,
		},
	}
	for _, test := range tests {
		var err error
		f, tf, codec, ns := NewAPIFactory()
		tf.Client = &fake.RESTClient{
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
		tf.ClientConfig = &restclient.Config{ContentConfig: restclient.ContentConfig{GroupVersion: &unversioned.GroupVersion{Version: test.version}}}
		ff := &fakePortForwarder{}
		if test.pfErr {
			ff.pfErr = fmt.Errorf("pf error")
		}

		opts := &PortForwardOptions{}
		cmd := NewCmdPortForward(f, os.Stdout, os.Stderr)
		cmd.Run = func(cmd *cobra.Command, args []string) {
			if err = opts.Complete(f, cmd, args, os.Stdout, os.Stderr); err != nil {
				return
			}
			opts.PortForwarder = ff
			if err = opts.Validate(); err != nil {
				return
			}
			err = opts.RunPortForward()
		}

		cmd.Run(cmd, []string{"foo", ":5000", ":1000"})

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

func TestPortForwardWithPFlag(t *testing.T) {
	version := testapi.Default.GroupVersion().Version

	tests := []struct {
		name, version, podPath, pfPath, container string
		pod                                       *api.Pod
		pfErr                                     bool
	}{
		{
			name:    "pod portforward",
			version: version,
			podPath: "/api/" + version + "/namespaces/test/pods/foo",
			pfPath:  "/api/" + version + "/namespaces/test/pods/foo/portforward",
			pod:     execPod(),
		},
		{
			name:    "pod portforward error",
			version: version,
			podPath: "/api/" + version + "/namespaces/test/pods/foo",
			pfPath:  "/api/" + version + "/namespaces/test/pods/foo/portforward",
			pod:     execPod(),
			pfErr:   true,
		},
	}
	for _, test := range tests {
		var err error
		f, tf, codec, ns := NewAPIFactory()
		tf.Client = &fake.RESTClient{
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
		tf.ClientConfig = &restclient.Config{ContentConfig: restclient.ContentConfig{GroupVersion: &unversioned.GroupVersion{Version: test.version}}}
		ff := &fakePortForwarder{}
		if test.pfErr {
			ff.pfErr = fmt.Errorf("pf error")
		}

		opts := &PortForwardOptions{}
		cmd := NewCmdPortForward(f, os.Stdout, os.Stderr)
		cmd.Run = func(cmd *cobra.Command, args []string) {
			if err = opts.Complete(f, cmd, args, os.Stdout, os.Stderr); err != nil {
				return
			}
			opts.PortForwarder = ff
			if err = opts.Validate(); err != nil {
				return
			}
			err = opts.RunPortForward()
		}
		cmd.Flags().Set("pod", "foo")

		cmd.Run(cmd, []string{":5000", ":1000"})

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
