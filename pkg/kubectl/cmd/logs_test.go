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
	"io/ioutil"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/spf13/cobra"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/rest/fake"
	"k8s.io/kubernetes/pkg/api"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
)

func TestLog(t *testing.T) {
	tests := []struct {
		name, version, podNamePath, podResourcePath, logPath, container string
	}{
		{
			name:            "v1 - pod log",
			version:         "v1",
			podNamePath:     "/namespaces/test/pods/foo",
			podResourcePath: "/namespaces/test/pods",
			logPath:         "/api/v1/namespaces/test/pods/foo/log",
		},
	}
	for _, test := range tests {
		logContent := "test log content"
		f, tf, codec, ns := cmdtesting.NewAPIFactory()
		tf.Client = &fake.RESTClient{
			APIRegistry:          api.Registry,
			NegotiatedSerializer: ns,
			Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
				switch p, m := req.URL.Path, req.Method; {
				case p == test.podNamePath && m == "GET":
					body := objBody(codec, testPod())
					return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: body}, nil
				case p == test.podResourcePath && m == "GET":
					body := objBody(codec, testPodList())
					return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: body}, nil
				case p == test.logPath && m == "GET":
					body := ioutil.NopCloser(bytes.NewBufferString(logContent))
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
		buf := bytes.NewBuffer([]byte{})

		cmd := NewCmdLogs(f, buf)
		cmd.Flags().Set("namespace", "test")
		cmd.Run(cmd, []string{"foo"})

		if buf.String() != logContent {
			t.Errorf("%s: did not get expected log content. Got: %s", test.name, buf.String())
		}
	}
}

func testPod() *api.Pod {
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
		},
	}
}

func testPodList() *api.PodList {
	return &api.PodList{
		ListMeta: metav1.ListMeta{
			ResourceVersion: "10",
		},
		Items: []api.Pod{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test", ResourceVersion: "10"},
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicyAlways,
					DNSPolicy:     api.DNSClusterFirst,
					Containers: []api.Container{
						{
							Name: "bar",
						},
					},
				},
			},
		},
	}
}

func TestValidateLogFlags(t *testing.T) {

	tests := []struct {
		name            string
		flags           map[string]string
		expected        string
		podResourcePath string
		podNamePath     string
	}{
		{
			name:            "since & since-time",
			flags:           map[string]string{"since": "1h", "since-time": "2006-01-02T15:04:05Z"},
			expected:        "at most one of `sinceTime` or `sinceSeconds` may be specified",
			podResourcePath: "/namespaces/test/pods",
			podNamePath:     "/namespaces/test/pods/foo",
		},
		{
			name:            "negative limit-bytes",
			flags:           map[string]string{"limit-bytes": "-100"},
			expected:        "must be greater than 0",
			podResourcePath: "/namespaces/test/pods",
			podNamePath:     "/namespaces/test/pods/foo",
		},
		{
			name:            "negative tail",
			flags:           map[string]string{"tail": "-100"},
			expected:        "must be greater than or equal to 0",
			podResourcePath: "/namespaces/test/pods",
			podNamePath:     "/namespaces/test/pods/foo",
		},
	}
	for _, test := range tests {
		f, tf, codec, ns := cmdtesting.NewAPIFactory()
		tf.Client = &fake.RESTClient{
			APIRegistry:          api.Registry,
			NegotiatedSerializer: ns,
			Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
				switch p, m := req.URL.Path, req.Method; {
				case p == test.podNamePath && m == "GET":
					body := objBody(codec, testPod())
					return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: body}, nil
				case p == test.podResourcePath && m == "GET":
					body := objBody(codec, testPodList())
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
		cmd := NewCmdLogs(f, bytes.NewBuffer([]byte{}))
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
		}
		cmd.Run(cmd, []string{"foo"})

		if !strings.Contains(out, test.expected) {
			t.Errorf("%s: expected to find:\n\t%s\nfound:\n\t%s\n", test.name, test.expected, out)
		}
	}
}
