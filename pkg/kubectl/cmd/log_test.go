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
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
)

func TestSelectContainer(t *testing.T) {
	tests := []struct {
		input             string
		pod               api.Pod
		expectedContainer string
	}{
		{
			input: "1\n",
			pod: api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name: "foo",
						},
					},
				},
			},
			expectedContainer: "foo",
		},
		{
			input: "foo\n",
			pod: api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name: "foo",
						},
					},
				},
			},
			expectedContainer: "foo",
		},
		{
			input: "foo\n",
			pod: api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name: "bar",
						},
						{
							Name: "foo",
						},
					},
				},
			},
			expectedContainer: "foo",
		},
		{
			input: "2\n",
			pod: api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name: "bar",
						},
						{
							Name: "foo",
						},
					},
				},
			},
			expectedContainer: "foo",
		},
		{
			input: "-1\n2\n",
			pod: api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name: "bar",
						},
						{
							Name: "foo",
						},
					},
				},
			},
			expectedContainer: "foo",
		},
		{
			input: "3\n2\n",
			pod: api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name: "bar",
						},
						{
							Name: "foo",
						},
					},
				},
			},
			expectedContainer: "foo",
		},
		{
			input: "baz\n2\n",
			pod: api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name: "bar",
						},
						{
							Name: "foo",
						},
					},
				},
			},
			expectedContainer: "foo",
		},
	}

	for _, test := range tests {
		var buff bytes.Buffer
		container := selectContainer(&test.pod, bytes.NewBufferString(test.input), &buff)
		if container != test.expectedContainer {
			t.Errorf("unexpected output: %s for input: %s", container, test.input)
		}
	}
}

func TestLog(t *testing.T) {

	tests := []struct {
		name, version, podPath, logPath, container string
		nsInQuery                                  bool
		pod                                        *api.Pod
	}{
		{
			name:      "v1beta1 - pod log",
			version:   "v1beta1",
			podPath:   "/api/v1beta1/pods/foo",
			logPath:   "/api/v1beta1/pods/foo/log",
			nsInQuery: true,
			pod:       testPod(),
		},
		{
			name:      "v1beta3 - pod log",
			version:   "v1beta3",
			podPath:   "/api/v1beta3/namespaces/test/pods/foo",
			logPath:   "/api/v1beta3/namespaces/test/pods/foo/log",
			nsInQuery: false,
			pod:       testPod(),
		},
	}
	for _, test := range tests {
		logContent := "test log content"
		f, tf, codec := NewAPIFactory()
		tf.Client = &client.FakeRESTClient{
			Codec: codec,
			Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
				switch p, m := req.URL.Path, req.Method; {
				case p == test.podPath && m == "GET":
					if test.nsInQuery {
						if ns := req.URL.Query().Get("namespace"); ns != "test" {
							t.Errorf("%s: did not get expected namespace: %s\n", test.name, ns)
						}
					}
					body := objBody(codec, test.pod)
					return &http.Response{StatusCode: 200, Body: body}, nil
				case p == test.logPath && m == "GET":
					if test.nsInQuery {
						if ns := req.URL.Query().Get("namespace"); ns != "test" {
							t.Errorf("%s: did not get expected namespace: %s\n", test.name, ns)
						}
					}
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
