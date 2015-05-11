/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"net/http"
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

func TestRunExposeService(t *testing.T) {
	tests := []struct {
		name   string
		args   []string
		input  runtime.Object
		flags  map[string]string
		output runtime.Object
		status int
	}{
		{
			name: "expose-service-from-service",
			args: []string{"service", "baz"},
			input: &api.Service{
				ObjectMeta: api.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
				Spec: api.ServiceSpec{
					Selector: map[string]string{"app": "go"},
				},
			},
			flags: map[string]string{"selector": "func=stream", "protocol": "UDP", "port": "14", "service-name": "foo"},
			output: &api.Service{
				ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "test", ResourceVersion: "12"},
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{
							Name:     "default",
							Protocol: api.Protocol("UDP"),
							Port:     14,
						},
					},
					Selector: map[string]string{"func": "stream"},
				},
			},
			status: 200,
		},
	}

	for _, test := range tests {
		f, tf, codec := NewAPIFactory()
		tf.Printer = &testPrinter{}
		tf.Client = &client.FakeRESTClient{
			Codec: codec,
			Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
				switch p, m := req.URL.Path, req.Method; {
				case p == "/namespaces/test/services/baz" && m == "GET":
					return &http.Response{StatusCode: test.status, Body: objBody(codec, test.input)}, nil
				case p == "/namespaces/test/services" && m == "POST":
					return &http.Response{StatusCode: test.status, Body: objBody(codec, test.output)}, nil
				default:
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
					return nil, nil
				}
			}),
		}
		tf.Namespace = "test"
		buf := bytes.NewBuffer([]byte{})

		cmd := NewCmdExposeService(f, buf)
		cmd.SetOutput(buf)
		for flag, value := range test.flags {
			cmd.Flags().Set(flag, value)
		}
		cmd.Run(cmd, test.args)

		out := buf.String()
		if strings.Contains(out, "services/foo") {
			t.Errorf("%s: unexpected output: %s", test.name, out)
		}
	}
}
