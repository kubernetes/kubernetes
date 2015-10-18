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

package cmd

import (
	"bytes"
	"io/ioutil"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/fake"
)

func TestLog(t *testing.T) {
	tests := []struct {
		name, version, podPath, logPath, container string
		pod                                        *api.Pod
	}{
		{
			name:    "v1 - pod log",
			version: "v1",
			podPath: "/namespaces/test/pods/foo",
			logPath: "/api/v1/namespaces/test/pods/foo/log",
			pod:     testPod(),
		},
	}
	for _, test := range tests {
		logContent := "test log content"
		f, tf, codec := NewAPIFactory()
		tf.Client = &fake.RESTClient{
			Codec: codec,
			Client: fake.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
				switch p, m := req.URL.Path, req.Method; {
				case p == test.podPath && m == "GET":
					body := objBody(codec, test.pod)
					return &http.Response{StatusCode: 200, Body: body}, nil
				case p == test.logPath && m == "GET":
					body := ioutil.NopCloser(bytes.NewBufferString(logContent))
					return &http.Response{StatusCode: 200, Body: body}, nil
				default:
					// Ensures no GET is performed when deleting by name
					t.Errorf("%s: unexpected request: %#v\n%#v", test.name, req.URL, req)
					return nil, nil
				}
			}),
		}
		tf.Namespace = "test"
		tf.ClientConfig = &client.Config{Version: test.version}
		buf := bytes.NewBuffer([]byte{})

		cmd := NewCmdLog(f, buf)
		cmd.Flags().Set("namespace", "test")
		cmd.Run(cmd, []string{"foo"})

		if buf.String() != logContent {
			t.Errorf("%s: did not get expected log content. Got: %s", test.name, buf.String())
		}
	}
}

func testPod() *api.Pod {
	return &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "test", ResourceVersion: "10"},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,
			Containers: []api.Container{
				{
					Name: "bar",
				},
			},
		},
	}
}

func TestValidateLogFlags(t *testing.T) {
	f, _, _ := NewAPIFactory()

	tests := []struct {
		name     string
		flags    map[string]string
		expected string
	}{
		{
			name:     "since & since-time",
			flags:    map[string]string{"since": "1h", "since-time": "2006-01-02T15:04:05Z"},
			expected: "only one of sinceTime or sinceSeconds can be provided",
		},
		{
			name:     "negative limit-bytes",
			flags:    map[string]string{"limit-bytes": "-100"},
			expected: "limitBytes must be a positive integer or nil",
		},
		{
			name:     "negative tail",
			flags:    map[string]string{"tail": "-100"},
			expected: "tailLines must be a non-negative integer or nil",
		},
	}
	for _, test := range tests {
		cmd := NewCmdLog(f, bytes.NewBuffer([]byte{}))
		out := ""
		for flag, value := range test.flags {
			cmd.Flags().Set(flag, value)
		}
		// checkErr breaks tests in case of errors, plus we just
		// need to check errors returned by the command validation
		o := &LogsOptions{}
		cmd.Run = func(cmd *cobra.Command, args []string) {
			o.Complete(f, os.Stdout, cmd, args)
			out = o.Validate().Error()
			o.RunLog()
		}
		cmd.Run(cmd, []string{"foo"})

		if !strings.Contains(out, test.expected) {
			t.Errorf("%s: expected to find:\n\t%s\nfound:\n\t%s\n", test.name, test.expected, out)
		}
	}
}
