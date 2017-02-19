/*
Copyright 2015 The Kubernetes Authors.

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
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/rest/fake"
	"k8s.io/kubernetes/pkg/api"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	"k8s.io/kubernetes/pkg/printers"
)

func TestRunExposeService(t *testing.T) {
	tests := []struct {
		name     string
		args     []string
		ns       string
		calls    map[string]string
		input    runtime.Object
		flags    map[string]string
		output   runtime.Object
		expected string
		status   int
	}{
		{
			name: "expose-service-from-service-no-selector-defined",
			args: []string{"service", "baz"},
			ns:   "test",
			calls: map[string]string{
				"GET":  "/namespaces/test/services/baz",
				"POST": "/namespaces/test/services",
			},
			input: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
				Spec: api.ServiceSpec{
					Selector: map[string]string{"app": "go"},
				},
			},
			flags: map[string]string{"protocol": "UDP", "port": "14", "name": "foo", "labels": "svc=test"},
			output: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "", Labels: map[string]string{"svc": "test"}},
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{
							Protocol:   api.ProtocolUDP,
							Port:       14,
							TargetPort: intstr.FromInt(14),
						},
					},
					Selector: map[string]string{"app": "go"},
				},
			},
			expected: "service \"foo\" exposed",
			status:   200,
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
				ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
				Spec: api.ServiceSpec{
					Selector: map[string]string{"app": "go"},
				},
			},
			flags: map[string]string{"selector": "func=stream", "protocol": "UDP", "port": "14", "name": "foo", "labels": "svc=test"},
			output: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "", Labels: map[string]string{"svc": "test"}},
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{
							Protocol:   api.ProtocolUDP,
							Port:       14,
							TargetPort: intstr.FromInt(14),
						},
					},
					Selector: map[string]string{"func": "stream"},
				},
			},
			expected: "service \"foo\" exposed",
			status:   200,
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
				ObjectMeta: metav1.ObjectMeta{Name: "mayor", Namespace: "default", ResourceVersion: "12"},
				Spec: api.ServiceSpec{
					Selector: map[string]string{"run": "this"},
				},
			},
			// No --name flag specified below. Service will use the rc's name passed via the 'default-name' parameter
			flags: map[string]string{"selector": "run=this", "port": "80", "labels": "runas=amayor"},
			output: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "mayor", Namespace: "", Labels: map[string]string{"runas": "amayor"}},
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{
							Protocol:   api.ProtocolTCP,
							Port:       80,
							TargetPort: intstr.FromInt(80),
						},
					},
					Selector: map[string]string{"run": "this"},
				},
			},
			expected: "service \"mayor\" exposed",
			status:   200,
		},
		{
			name: "expose-service",
			args: []string{"service", "baz"},
			ns:   "test",
			calls: map[string]string{
				"GET":  "/namespaces/test/services/baz",
				"POST": "/namespaces/test/services",
			},
			input: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
				Spec: api.ServiceSpec{
					Selector: map[string]string{"app": "go"},
				},
			},
			flags: map[string]string{"selector": "func=stream", "protocol": "UDP", "port": "14", "name": "foo", "labels": "svc=test", "type": "LoadBalancer", "dry-run": "true"},
			output: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "", Labels: map[string]string{"svc": "test"}},
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{
							Protocol:   api.ProtocolUDP,
							Port:       14,
							TargetPort: intstr.FromInt(14),
						},
					},
					Selector: map[string]string{"func": "stream"},
					Type:     api.ServiceTypeLoadBalancer,
				},
			},
			status: 200,
		},
		{
			name: "expose-affinity-service",
			args: []string{"service", "baz"},
			ns:   "test",
			calls: map[string]string{
				"GET":  "/namespaces/test/services/baz",
				"POST": "/namespaces/test/services",
			},
			input: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
				Spec: api.ServiceSpec{
					Selector: map[string]string{"app": "go"},
				},
			},
			flags: map[string]string{"selector": "func=stream", "protocol": "UDP", "port": "14", "name": "foo", "labels": "svc=test", "type": "LoadBalancer", "session-affinity": "ClientIP", "dry-run": "true"},
			output: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "", Labels: map[string]string{"svc": "test"}},
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{
							Protocol:   api.ProtocolUDP,
							Port:       14,
							TargetPort: intstr.FromInt(14),
						},
					},
					Selector:        map[string]string{"func": "stream"},
					Type:            api.ServiceTypeLoadBalancer,
					SessionAffinity: api.ServiceAffinityClientIP,
				},
			},
			status: 200,
		},
		{
			name: "expose-service-cluster-ip",
			args: []string{"service", "baz"},
			ns:   "test",
			calls: map[string]string{
				"GET":  "/namespaces/test/services/baz",
				"POST": "/namespaces/test/services",
			},
			input: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
				Spec: api.ServiceSpec{
					Selector: map[string]string{"app": "go"},
				},
			},
			flags: map[string]string{"selector": "func=stream", "protocol": "UDP", "port": "14", "name": "foo", "labels": "svc=test", "cluster-ip": "10.10.10.10", "dry-run": "true"},
			output: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "", Labels: map[string]string{"svc": "test"}},
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{
							Protocol:   api.ProtocolUDP,
							Port:       14,
							TargetPort: intstr.FromInt(14),
						},
					},
					Selector:  map[string]string{"func": "stream"},
					ClusterIP: "10.10.10.10",
				},
			},
			expected: "service \"foo\" exposed",
			status:   200,
		},
		{
			name: "expose-headless-service",
			args: []string{"service", "baz"},
			ns:   "test",
			calls: map[string]string{
				"GET":  "/namespaces/test/services/baz",
				"POST": "/namespaces/test/services",
			},
			input: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
				Spec: api.ServiceSpec{
					Selector: map[string]string{"app": "go"},
				},
			},
			flags: map[string]string{"selector": "func=stream", "protocol": "UDP", "port": "14", "name": "foo", "labels": "svc=test", "cluster-ip": "None", "dry-run": "true"},
			output: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "", Labels: map[string]string{"svc": "test"}},
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{
							Protocol:   api.ProtocolUDP,
							Port:       14,
							TargetPort: intstr.FromInt(14),
						},
					},
					Selector:  map[string]string{"func": "stream"},
					ClusterIP: api.ClusterIPNone,
				},
			},
			expected: "service \"foo\" exposed",
			status:   200,
		},
		{
			name: "expose-headless-service-no-port",
			args: []string{"service", "baz"},
			ns:   "test",
			calls: map[string]string{
				"GET":  "/namespaces/test/services/baz",
				"POST": "/namespaces/test/services",
			},
			input: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
				Spec: api.ServiceSpec{
					Selector: map[string]string{"app": "go"},
				},
			},
			flags: map[string]string{"selector": "func=stream", "name": "foo", "labels": "svc=test", "cluster-ip": "None", "dry-run": "true"},
			output: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "", Labels: map[string]string{"svc": "test"}},
				Spec: api.ServiceSpec{
					Ports:     []api.ServicePort{},
					Selector:  map[string]string{"func": "stream"},
					ClusterIP: api.ClusterIPNone,
				},
			},
			expected: "service \"foo\" exposed",
			status:   200,
		},
		{
			name: "expose-from-file",
			args: []string{},
			ns:   "test",
			calls: map[string]string{
				"GET":  "/namespaces/test/services/redis-master",
				"POST": "/namespaces/test/services",
			},
			input: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "redis-master", Namespace: "test", ResourceVersion: "12"},
				Spec: api.ServiceSpec{
					Selector: map[string]string{"app": "go"},
				},
			},
			flags: map[string]string{"filename": "../../../examples/guestbook/redis-master-service.yaml", "selector": "func=stream", "protocol": "UDP", "port": "14", "name": "foo", "labels": "svc=test", "dry-run": "true"},
			output: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Labels: map[string]string{"svc": "test"}},
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{
							Protocol:   api.ProtocolUDP,
							Port:       14,
							TargetPort: intstr.FromInt(14),
						},
					},
					Selector: map[string]string{"func": "stream"},
				},
			},
			status: 200,
		},
		{
			name: "truncate-name",
			args: []string{"pod", "a-name-that-is-toooo-big-for-a-service-because-it-can-only-handle-63-characters"},
			ns:   "test",
			calls: map[string]string{
				"GET":  "/namespaces/test/pods/a-name-that-is-toooo-big-for-a-service-because-it-can-only-handle-63-characters",
				"POST": "/namespaces/test/services",
			},
			input: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
			},
			flags: map[string]string{"selector": "svc=frompod", "port": "90", "labels": "svc=frompod", "generator": "service/v2"},
			output: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "a-name-that-is-toooo-big-for-a-service-because-it-can-only-handle-63-characters", Namespace: "", Labels: map[string]string{"svc": "frompod"}},
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{
							Protocol:   api.ProtocolTCP,
							Port:       90,
							TargetPort: intstr.FromInt(90),
						},
					},
					Selector: map[string]string{"svc": "frompod"},
				},
			},
			expected: "service \"a-name-that-is-toooo-big-for-a-service-because-it-can-only-hand\" exposed",
			status:   200,
		},
		{
			name: "expose-multiport-object",
			args: []string{"service", "foo"},
			ns:   "test",
			calls: map[string]string{
				"GET":  "/namespaces/test/services/foo",
				"POST": "/namespaces/test/services",
			},
			input: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "", Labels: map[string]string{"svc": "multiport"}},
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{
							Protocol:   api.ProtocolTCP,
							Port:       80,
							TargetPort: intstr.FromInt(80),
						},
						{
							Protocol:   api.ProtocolTCP,
							Port:       443,
							TargetPort: intstr.FromInt(443),
						},
					},
				},
			},
			flags: map[string]string{"selector": "svc=fromfoo", "generator": "service/v2", "name": "fromfoo", "dry-run": "true"},
			output: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "fromfoo", Namespace: "", Labels: map[string]string{"svc": "multiport"}},
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{
							Name:       "port-1",
							Protocol:   api.ProtocolTCP,
							Port:       80,
							TargetPort: intstr.FromInt(80),
						},
						{
							Name:       "port-2",
							Protocol:   api.ProtocolTCP,
							Port:       443,
							TargetPort: intstr.FromInt(443),
						},
					},
					Selector: map[string]string{"svc": "fromfoo"},
				},
			},
			status: 200,
		},
		{
			name: "expose-multiprotocol-object",
			args: []string{"service", "foo"},
			ns:   "test",
			calls: map[string]string{
				"GET":  "/namespaces/test/services/foo",
				"POST": "/namespaces/test/services",
			},
			input: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "", Labels: map[string]string{"svc": "multiport"}},
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{
							Protocol:   api.ProtocolTCP,
							Port:       80,
							TargetPort: intstr.FromInt(80),
						},
						{
							Protocol:   api.ProtocolUDP,
							Port:       8080,
							TargetPort: intstr.FromInt(8080),
						},
						{
							Protocol:   api.ProtocolUDP,
							Port:       8081,
							TargetPort: intstr.FromInt(8081),
						},
					},
				},
			},
			flags: map[string]string{"selector": "svc=fromfoo", "generator": "service/v2", "name": "fromfoo", "dry-run": "true"},
			output: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "fromfoo", Namespace: "", Labels: map[string]string{"svc": "multiport"}},
				Spec: api.ServiceSpec{
					Ports: []api.ServicePort{
						{
							Name:       "port-1",
							Protocol:   api.ProtocolTCP,
							Port:       80,
							TargetPort: intstr.FromInt(80),
						},
						{
							Name:       "port-2",
							Protocol:   api.ProtocolUDP,
							Port:       8080,
							TargetPort: intstr.FromInt(8080),
						},
						{
							Name:       "port-3",
							Protocol:   api.ProtocolUDP,
							Port:       8081,
							TargetPort: intstr.FromInt(8081),
						},
					},
					Selector: map[string]string{"svc": "fromfoo"},
				},
			},
			status: 200,
		},
	}

	for _, test := range tests {
		f, tf, codec, ns := cmdtesting.NewAPIFactory()
		tf.Printer = &printers.JSONPrinter{}
		tf.Client = &fake.RESTClient{
			APIRegistry:          api.Registry,
			NegotiatedSerializer: ns,
			Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
				switch p, m := req.URL.Path, req.Method; {
				case p == test.calls[m] && m == "GET":
					return &http.Response{StatusCode: test.status, Header: defaultHeader(), Body: objBody(codec, test.input)}, nil
				case p == test.calls[m] && m == "POST":
					return &http.Response{StatusCode: test.status, Header: defaultHeader(), Body: objBody(codec, test.output)}, nil
				default:
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
					return nil, nil
				}
			}),
		}
		tf.Namespace = test.ns
		buf := bytes.NewBuffer([]byte{})

		cmd := NewCmdExposeService(f, buf)
		cmd.SetOutput(buf)
		for flag, value := range test.flags {
			cmd.Flags().Set(flag, value)
		}
		cmd.Run(cmd, test.args)

		out := buf.String()
		if _, ok := test.flags["dry-run"]; ok {
			buf.Reset()
			if err := tf.Printer.PrintObj(test.output, buf); err != nil {
				t.Errorf("%s: Unexpected error: %v", test.name, err)
				continue
			}

			test.expected = fmt.Sprintf("service %q exposed (dry run)", test.flags["name"])
		}

		if !strings.Contains(out, test.expected) {
			t.Errorf("%s: Unexpected output! Expected\n%s\ngot\n%s", test.name, test.expected, out)
		}
	}
}
