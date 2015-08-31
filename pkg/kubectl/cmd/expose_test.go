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

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
)

func TestRunExposeService(t *testing.T) {
	tests := []struct {
		name        string
		args        []string
		ns          string
		calls       map[string]string
		input       runtime.Object
		flags       map[string]string
		output      runtime.Object
		expected    string
		status      int
		podSelector string
	}{
		{
			name: "expose-service-from-service-no-selector",
			args: []string{"service", "baz"},
			ns:   "test",
			calls: map[string]string{
				"GET":  "/namespaces/test/services/baz",
				"POST": "/namespaces/test/services",
			},
			input: &api.Service{
				ObjectMeta: api.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
				TypeMeta:   api.TypeMeta{Kind: "Service", APIVersion: "v1"},
				Spec: api.ServiceSpec{
					Selector: map[string]string{"app": "go"},
				},
			},
			podSelector: "app=go",
			flags:       map[string]string{"protocol": "UDP", "port": "14", "name": "foo", "labels": "svc=test"},
			output: &api.Service{
				ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "test", ResourceVersion: "12", Labels: map[string]string{"svc": "test"}},
				TypeMeta:   api.TypeMeta{Kind: "Service", APIVersion: "v1"},
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{
							Name:     "default",
							Protocol: api.Protocol("UDP"),
							Port:     14,
						},
					},
					Selector: map[string]string{"app": "go"},
				},
			},
			status: 200,
		},
		{
			name: "expose-service-from-service",
			args: []string{"service", "baz"},
			ns:   "test",
			calls: map[string]string{
				"GET":  "/namespaces/test/services/baz",
				"POST": "/namespaces/test/services",
			},
			input: &api.Service{
				ObjectMeta: api.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
				TypeMeta:   api.TypeMeta{Kind: "Service", APIVersion: "v1"},
				Spec: api.ServiceSpec{
					Selector: map[string]string{"app": "go"},
				},
			},
			flags: map[string]string{"selector": "func=stream", "protocol": "UDP", "port": "14", "name": "foo", "labels": "svc=test"},
			output: &api.Service{
				ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "test", ResourceVersion: "12", Labels: map[string]string{"svc": "test"}},
				TypeMeta:   api.TypeMeta{Kind: "Service", APIVersion: "v1"},
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
		{
			name: "no-name-passed-from-the-cli",
			args: []string{"service", "mayor"},
			ns:   "default",
			calls: map[string]string{
				"GET":  "/namespaces/default/services/mayor",
				"POST": "/namespaces/default/services",
			},
			input: &api.Service{
				ObjectMeta: api.ObjectMeta{Name: "mayor", Namespace: "default", ResourceVersion: "12"},
				TypeMeta:   api.TypeMeta{Kind: "Service", APIVersion: "v1"},
				Spec: api.ServiceSpec{
					Selector: map[string]string{"run": "this"},
				},
			},
			// No --name flag specified below. Service will use the rc's name passed via the 'default-name' parameter
			flags: map[string]string{"selector": "run=this", "port": "80", "labels": "runas=amayor"},
			output: &api.Service{
				ObjectMeta: api.ObjectMeta{Name: "mayor", Namespace: "default", ResourceVersion: "12", Labels: map[string]string{"runas": "amayor"}},
				TypeMeta:   api.TypeMeta{Kind: "Service", APIVersion: "v1"},
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{
							Name:     "default",
							Protocol: api.Protocol("TCP"),
							Port:     80,
						},
					},
					Selector: map[string]string{"run": "this"},
				},
			},
			status: 200,
		},
		{
			name: "expose-external-service",
			args: []string{"service", "baz"},
			ns:   "test",
			calls: map[string]string{
				"GET":  "/namespaces/test/services/baz",
				"POST": "/namespaces/test/services",
			},
			input: &api.Service{
				ObjectMeta: api.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
				TypeMeta:   api.TypeMeta{Kind: "Service", APIVersion: "v1"},
				Spec: api.ServiceSpec{
					Selector: map[string]string{"app": "go"},
				},
			},
			flags: map[string]string{"selector": "func=stream", "protocol": "UDP", "port": "14", "name": "foo", "labels": "svc=test", "create-external-load-balancer": "true"},
			output: &api.Service{
				ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "test", ResourceVersion: "12", Labels: map[string]string{"svc": "test"}},
				TypeMeta:   api.TypeMeta{Kind: "Service", APIVersion: "v1"},
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{
							Name:       "default",
							Protocol:   api.Protocol("UDP"),
							Port:       14,
							TargetPort: util.NewIntOrStringFromInt(14),
						},
					},
					Selector: map[string]string{"func": "stream"},
					Type:     api.ServiceTypeLoadBalancer,
				},
			},
			status: 200,
		},
		{
			name: "expose-external-affinity-service",
			args: []string{"service", "baz"},
			ns:   "test",
			calls: map[string]string{
				"GET":  "/namespaces/test/services/baz",
				"POST": "/namespaces/test/services",
			},
			input: &api.Service{
				ObjectMeta: api.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
				TypeMeta:   api.TypeMeta{Kind: "Service", APIVersion: "v1"},
				Spec: api.ServiceSpec{
					Selector: map[string]string{"app": "go"},
				},
			},
			flags: map[string]string{"selector": "func=stream", "protocol": "UDP", "port": "14", "name": "foo", "labels": "svc=test", "create-external-load-balancer": "true", "session-affinity": "ClientIP"},
			output: &api.Service{
				ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "test", ResourceVersion: "12", Labels: map[string]string{"svc": "test"}},
				TypeMeta:   api.TypeMeta{Kind: "Service", APIVersion: "v1"},
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{
							Name:       "default",
							Protocol:   api.Protocol("UDP"),
							Port:       14,
							TargetPort: util.NewIntOrStringFromInt(14),
						},
					},
					Selector:        map[string]string{"func": "stream"},
					Type:            api.ServiceTypeLoadBalancer,
					SessionAffinity: api.ServiceAffinityClientIP,
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
				case p == test.calls[m] && m == "GET":
					return &http.Response{StatusCode: test.status, Body: objBody(codec, test.input)}, nil
				case p == test.calls[m] && m == "POST":
					return &http.Response{StatusCode: test.status, Body: objBody(codec, test.output)}, nil
				default:
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
					return nil, nil
				}
			}),
		}
		tf.Namespace = test.ns
		f.PodSelectorForObject = func(obj runtime.Object) (string, error) { return test.podSelector, nil }

		buf := bytes.NewBuffer([]byte{})

		cmd := NewCmdExposeService(f, buf)
		cmd.SetOutput(buf)
		for flag, value := range test.flags {
			cmd.Flags().Set(flag, value)
		}
		cmd.Run(cmd, test.args)
		if len(test.expected) > 0 {
			out := buf.String()
			if !strings.Contains(out, test.expected) {
				t.Errorf("%s: unexpected output: %s", test.name, out)
			}
		}
	}
}

func TestRunExposeServiceFromFile(t *testing.T) {
	test := struct {
		calls    map[string]string
		input    runtime.Object
		flags    map[string]string
		output   runtime.Object
		expected string
		status   int
	}{
		calls: map[string]string{
			"GET":  "/namespaces/test/services/redis-master",
			"POST": "/namespaces/test/services",
		},
		input: &api.Service{
			ObjectMeta: api.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
			TypeMeta:   api.TypeMeta{Kind: "Service", APIVersion: "v1"},
			Spec: api.ServiceSpec{
				Selector: map[string]string{"app": "go"},
			},
		},
		flags: map[string]string{"selector": "func=stream", "protocol": "UDP", "port": "14", "name": "foo", "labels": "svc=test"},
		output: &api.Service{
			ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "test", ResourceVersion: "12", Labels: map[string]string{"svc": "test"}},
			TypeMeta:   api.TypeMeta{Kind: "Service", APIVersion: "v1"},
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
	}

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Client: client.HTTPClientFunc(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == test.calls[m] && m == "GET":
				return &http.Response{StatusCode: test.status, Body: objBody(codec, test.input)}, nil
			case p == test.calls[m] && m == "POST":
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
	cmd.Flags().Set("filename", "../../../examples/guestbook/redis-master-service.yaml")
	cmd.Run(cmd, []string{})
	if len(test.expected) > 0 {
		out := buf.String()
		if !strings.Contains(out, test.expected) {
			t.Errorf("unexpected output: %s", out)
		}
	}
}
