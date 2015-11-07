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
			podPath: "/api/v1/namespaces/test/pods/foo",
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
