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

package expose

import (
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/client-go/rest/fake"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	"k8s.io/kubectl/pkg/scheme"
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
			input: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{"app": "go"},
				},
			},
			flags: map[string]string{"protocol": "UDP", "port": "14", "name": "foo", "labels": "svc=test"},
			output: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "", Labels: map[string]string{"svc": "test"}},
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Protocol:   corev1.ProtocolUDP,
							Port:       14,
							TargetPort: intstr.FromInt32(14),
						},
					},
					Selector: map[string]string{"app": "go"},
				},
			},
			expected: "service/foo exposed",
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
			input: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{"app": "go"},
				},
			},
			flags: map[string]string{"selector": "func=stream", "protocol": "UDP", "port": "14", "name": "foo", "labels": "svc=test"},
			output: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "", Labels: map[string]string{"svc": "test"}},
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Protocol:   corev1.ProtocolUDP,
							Port:       14,
							TargetPort: intstr.FromInt32(14),
						},
					},
					Selector: map[string]string{"func": "stream"},
				},
			},
			expected: "service/foo exposed",
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
			input: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "mayor", Namespace: "default", ResourceVersion: "12"},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{"run": "this"},
				},
			},
			// No --name flag specified below. Service will use the rc's name passed via the 'default-name' parameter
			flags: map[string]string{"selector": "run=this", "port": "80", "labels": "runas=amayor"},
			output: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "mayor", Namespace: "", Labels: map[string]string{"runas": "amayor"}},
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Protocol:   corev1.ProtocolTCP,
							Port:       80,
							TargetPort: intstr.FromInt32(80),
						},
					},
					Selector: map[string]string{"run": "this"},
				},
			},
			expected: "service/mayor exposed",
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
			input: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{"app": "go"},
				},
			},
			flags: map[string]string{"selector": "func=stream", "protocol": "UDP", "port": "14", "name": "foo", "labels": "svc=test", "type": "LoadBalancer", "dry-run": "client"},
			output: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "", Labels: map[string]string{"svc": "test"}},
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Protocol:   corev1.ProtocolUDP,
							Port:       14,
							TargetPort: intstr.FromInt32(14),
						},
					},
					Selector: map[string]string{"func": "stream"},
					Type:     corev1.ServiceTypeLoadBalancer,
				},
			},
			expected: "service/foo exposed (dry run)",
			status:   200,
		},
		{
			name: "expose-affinity-service",
			args: []string{"service", "baz"},
			ns:   "test",
			calls: map[string]string{
				"GET":  "/namespaces/test/services/baz",
				"POST": "/namespaces/test/services",
			},
			input: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{"app": "go"},
				},
			},
			flags: map[string]string{"selector": "func=stream", "protocol": "UDP", "port": "14", "name": "foo", "labels": "svc=test", "type": "LoadBalancer", "session-affinity": "ClientIP", "dry-run": "client"},
			output: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "", Labels: map[string]string{"svc": "test"}},
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Protocol:   corev1.ProtocolUDP,
							Port:       14,
							TargetPort: intstr.FromInt32(14),
						},
					},
					Selector:        map[string]string{"func": "stream"},
					Type:            corev1.ServiceTypeLoadBalancer,
					SessionAffinity: corev1.ServiceAffinityClientIP,
				},
			},
			expected: "service/foo exposed (dry run)",
			status:   200,
		},
		{
			name: "expose-service-cluster-ip",
			args: []string{"service", "baz"},
			ns:   "test",
			calls: map[string]string{
				"GET":  "/namespaces/test/services/baz",
				"POST": "/namespaces/test/services",
			},
			input: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{"app": "go"},
				},
			},
			flags: map[string]string{"selector": "func=stream", "protocol": "UDP", "port": "14", "name": "foo", "labels": "svc=test", "cluster-ip": "10.10.10.10", "dry-run": "client"},
			output: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "", Labels: map[string]string{"svc": "test"}},
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Protocol:   corev1.ProtocolUDP,
							Port:       14,
							TargetPort: intstr.FromInt32(14),
						},
					},
					Selector:  map[string]string{"func": "stream"},
					ClusterIP: "10.10.10.10",
				},
			},
			expected: "service/foo exposed (dry run)",
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
			input: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{"app": "go"},
				},
			},
			flags: map[string]string{"selector": "func=stream", "protocol": "UDP", "port": "14", "name": "foo", "labels": "svc=test", "cluster-ip": "None", "dry-run": "client"},
			output: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "", Labels: map[string]string{"svc": "test"}},
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Protocol:   corev1.ProtocolUDP,
							Port:       14,
							TargetPort: intstr.FromInt32(14),
						},
					},
					Selector:  map[string]string{"func": "stream"},
					ClusterIP: corev1.ClusterIPNone,
				},
			},
			expected: "service/foo exposed (dry run)",
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
			input: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{"app": "go"},
				},
			},
			flags: map[string]string{"selector": "func=stream", "name": "foo", "labels": "svc=test", "cluster-ip": "None", "dry-run": "client"},
			output: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "", Labels: map[string]string{"svc": "test"}},
				Spec: corev1.ServiceSpec{
					Ports:     []corev1.ServicePort{},
					Selector:  map[string]string{"func": "stream"},
					ClusterIP: corev1.ClusterIPNone,
				},
			},
			expected: "service/foo exposed (dry run)",
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
			input: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "redis-master", Namespace: "test", ResourceVersion: "12"},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{"app": "go"},
				},
			},
			flags: map[string]string{"filename": "../../../testdata/redis-master-service.yaml", "selector": "func=stream", "protocol": "UDP", "port": "14", "name": "foo", "labels": "svc=test", "dry-run": "client"},
			output: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Labels: map[string]string{"svc": "test"}},
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Protocol:   corev1.ProtocolUDP,
							Port:       14,
							TargetPort: intstr.FromInt32(14),
						},
					},
					Selector: map[string]string{"func": "stream"},
				},
			},
			expected: "service/foo exposed (dry run)",
			status:   200,
		},
		{
			name: "truncate-name",
			args: []string{"pod", "a-name-that-is-toooo-big-for-a-service-because-it-can-only-handle-63-characters"},
			ns:   "test",
			calls: map[string]string{
				"GET":  "/namespaces/test/pods/a-name-that-is-toooo-big-for-a-service-because-it-can-only-handle-63-characters",
				"POST": "/namespaces/test/services",
			},
			input: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
			},
			flags: map[string]string{"selector": "svc=frompod", "port": "90", "labels": "svc=frompod", "generator": "service/v2"},
			output: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "a-name-that-is-toooo-big-for-a-service-because-it-can-only-handle-63-characters"[:63], Namespace: "", Labels: map[string]string{"svc": "frompod"}},
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Protocol:   corev1.ProtocolTCP,
							Port:       90,
							TargetPort: intstr.FromInt32(90),
						},
					},
					Selector: map[string]string{"svc": "frompod"},
				},
			},
			expected: "service/a-name-that-is-toooo-big-for-a-service-because-it-can-only-hand exposed",
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
			input: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "", Labels: map[string]string{"svc": "multiport"}},
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Protocol:   corev1.ProtocolTCP,
							Port:       80,
							TargetPort: intstr.FromInt32(80),
						},
						{
							Protocol:   corev1.ProtocolTCP,
							Port:       443,
							TargetPort: intstr.FromInt32(443),
						},
					},
				},
			},
			flags: map[string]string{"selector": "svc=fromfoo", "generator": "service/v2", "name": "fromfoo", "dry-run": "client"},
			output: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "fromfoo", Namespace: "", Labels: map[string]string{"svc": "multiport"}},
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Name:       "port-1",
							Protocol:   corev1.ProtocolTCP,
							Port:       80,
							TargetPort: intstr.FromInt32(80),
						},
						{
							Name:       "port-2",
							Protocol:   corev1.ProtocolTCP,
							Port:       443,
							TargetPort: intstr.FromInt32(443),
						},
					},
					Selector: map[string]string{"svc": "fromfoo"},
				},
			},
			expected: "service/fromfoo exposed (dry run)",
			status:   200,
		},
		{
			name: "expose-multiprotocol-object",
			args: []string{"service", "foo"},
			ns:   "test",
			calls: map[string]string{
				"GET":  "/namespaces/test/services/foo",
				"POST": "/namespaces/test/services",
			},
			input: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "", Labels: map[string]string{"svc": "multiport"}},
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Protocol:   corev1.ProtocolTCP,
							Port:       80,
							TargetPort: intstr.FromInt32(80),
						},
						{
							Protocol:   corev1.ProtocolUDP,
							Port:       8080,
							TargetPort: intstr.FromInt32(8080),
						},
						{
							Protocol:   corev1.ProtocolUDP,
							Port:       8081,
							TargetPort: intstr.FromInt32(8081),
						},
						{
							Protocol:   corev1.ProtocolSCTP,
							Port:       8082,
							TargetPort: intstr.FromInt32(8082),
						},
					},
				},
			},
			flags: map[string]string{"selector": "svc=fromfoo", "generator": "service/v2", "name": "fromfoo", "dry-run": "client"},
			output: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "fromfoo", Namespace: "", Labels: map[string]string{"svc": "multiport"}},
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Name:       "port-1",
							Protocol:   corev1.ProtocolTCP,
							Port:       80,
							TargetPort: intstr.FromInt32(80),
						},
						{
							Name:       "port-2",
							Protocol:   corev1.ProtocolUDP,
							Port:       8080,
							TargetPort: intstr.FromInt32(8080),
						},
						{
							Name:       "port-3",
							Protocol:   corev1.ProtocolUDP,
							Port:       8081,
							TargetPort: intstr.FromInt32(8081),
						},
						{
							Name:       "port-4",
							Protocol:   corev1.ProtocolSCTP,
							Port:       8082,
							TargetPort: intstr.FromInt32(8082),
						},
					},
					Selector: map[string]string{"svc": "fromfoo"},
				},
			},
			expected: "service/fromfoo exposed (dry run)",
			status:   200,
		},
		{
			name: "expose-service-from-service-no-selector-defined-sctp",
			args: []string{"service", "baz"},
			ns:   "test",
			calls: map[string]string{
				"GET":  "/namespaces/test/services/baz",
				"POST": "/namespaces/test/services",
			},
			input: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{"app": "go"},
				},
			},
			flags: map[string]string{"protocol": "SCTP", "port": "14", "name": "foo", "labels": "svc=test"},
			output: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "", Labels: map[string]string{"svc": "test"}},
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Protocol:   corev1.ProtocolSCTP,
							Port:       14,
							TargetPort: intstr.FromInt32(14),
						},
					},
					Selector: map[string]string{"app": "go"},
				},
			},
			expected: "service/foo exposed",
			status:   200,
		},
		{
			name: "expose-service-from-service-sctp",
			args: []string{"service", "baz"},
			ns:   "test",
			calls: map[string]string{
				"GET":  "/namespaces/test/services/baz",
				"POST": "/namespaces/test/services",
			},
			input: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{"app": "go"},
				},
			},
			flags: map[string]string{"selector": "func=stream", "protocol": "SCTP", "port": "14", "name": "foo", "labels": "svc=test"},
			output: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "", Labels: map[string]string{"svc": "test"}},
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Protocol:   corev1.ProtocolSCTP,
							Port:       14,
							TargetPort: intstr.FromInt32(14),
						},
					},
					Selector: map[string]string{"func": "stream"},
				},
			},
			expected: "service/foo exposed",
			status:   200,
		},
		{
			name: "expose-service-cluster-ip-sctp",
			args: []string{"service", "baz"},
			ns:   "test",
			calls: map[string]string{
				"GET":  "/namespaces/test/services/baz",
				"POST": "/namespaces/test/services",
			},
			input: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{"app": "go"},
				},
			},
			flags: map[string]string{"selector": "func=stream", "protocol": "SCTP", "port": "14", "name": "foo", "labels": "svc=test", "cluster-ip": "10.10.10.10", "dry-run": "client"},
			output: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "", Labels: map[string]string{"svc": "test"}},
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Protocol:   corev1.ProtocolSCTP,
							Port:       14,
							TargetPort: intstr.FromInt32(14),
						},
					},
					Selector:  map[string]string{"func": "stream"},
					ClusterIP: "10.10.10.10",
				},
			},
			expected: "service/foo exposed (dry run)",
			status:   200,
		},
		{
			name: "expose-headless-service-sctp",
			args: []string{"service", "baz"},
			ns:   "test",
			calls: map[string]string{
				"GET":  "/namespaces/test/services/baz",
				"POST": "/namespaces/test/services",
			},
			input: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{"app": "go"},
				},
			},
			flags: map[string]string{"selector": "func=stream", "protocol": "SCTP", "port": "14", "name": "foo", "labels": "svc=test", "cluster-ip": "None", "dry-run": "client"},
			output: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "", Labels: map[string]string{"svc": "test"}},
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Protocol:   corev1.ProtocolSCTP,
							Port:       14,
							TargetPort: intstr.FromInt32(14),
						},
					},
					Selector:  map[string]string{"func": "stream"},
					ClusterIP: corev1.ClusterIPNone,
				},
			},
			expected: "service/foo exposed (dry run)",
			status:   200,
		},
		{
			name: "namespace-yaml",
			args: []string{"service", "baz"},
			ns:   "testns",
			calls: map[string]string{
				"GET":  "/namespaces/testns/services/baz",
				"POST": "/namespaces/testns/services",
			},
			input: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "testns", ResourceVersion: "12"},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{"app": "go"},
				},
			},
			flags: map[string]string{"selector": "func=stream", "protocol": "UDP", "port": "14", "name": "foo", "labels": "svc=test", "type": "LoadBalancer", "dry-run": "client", "output": "yaml"},
			output: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "", Labels: map[string]string{"svc": "test"}},
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Protocol:   corev1.ProtocolUDP,
							Port:       14,
							TargetPort: intstr.FromInt32(14),
						},
					},
					Selector: map[string]string{"func": "stream"},
					Type:     corev1.ServiceTypeLoadBalancer,
				},
			},
			expected: "namespace: testns",
			status:   200,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tf := cmdtesting.NewTestFactory().WithNamespace(test.ns)
			defer tf.Cleanup()

			codec := runtime.NewCodec(scheme.DefaultJSONEncoder(), scheme.Codecs.UniversalDecoder(scheme.Scheme.PrioritizedVersionsAllGroups()...))
			ns := scheme.Codecs.WithoutConversion()

			tf.Client = &fake.RESTClient{
				GroupVersion:         schema.GroupVersion{Version: "v1"},
				NegotiatedSerializer: ns,
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					switch p, m := req.URL.Path, req.Method; {
					case p == test.calls[m] && m == "GET":
						return &http.Response{StatusCode: test.status, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, test.input)}, nil
					case p == test.calls[m] && m == "POST":
						return &http.Response{StatusCode: test.status, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, test.output)}, nil
					default:
						t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
						return nil, nil
					}
				}),
			}

			ioStreams, _, buf, _ := genericiooptions.NewTestIOStreams()
			cmd := NewCmdExposeService(tf, ioStreams)
			cmd.SetOut(buf)
			cmd.SetErr(buf)
			for flag, value := range test.flags {
				cmd.Flags().Set(flag, value)
			}
			cmd.Run(cmd, test.args)

			out := buf.String()

			if test.expected == "" {
				t.Errorf("%s: Invalid test case. Specify expected result.\n", test.name)
			}

			if !strings.Contains(out, test.expected) {
				t.Errorf("%s: Unexpected output! Expected\n%s\ngot\n%s", test.name, test.expected, out)
			}
		})
	}
}

func TestExposeOverride(t *testing.T) {
	tests := []struct {
		name         string
		overrides    string
		overrideType string
		expected     string
	}{
		{
			name:         "expose with merge override type should replace the entire spec",
			overrides:    `{"spec": {"ports": [{"protocol": "TCP", "port": 1111, "targetPort": 2222}]}, "selector": {"app": "go"}}`,
			overrideType: "merge",
			expected: `apiVersion: v1
kind: Service
metadata:
  creationTimestamp: null
  labels:
    svc: test
  name: foo
  namespace: test
spec:
  ports:
  - port: 1111
    protocol: TCP
    targetPort: 2222
  selector:
    app: go
status:
  loadBalancer: {}
`,
		},
		{
			name:         "expose with strategic override type should add port before existing port",
			overrides:    `{"spec": {"ports": [{"protocol": "TCP", "port": 1111, "targetPort": 2222}]}}`,
			overrideType: "strategic",
			expected: `apiVersion: v1
kind: Service
metadata:
  creationTimestamp: null
  labels:
    svc: test
  name: foo
  namespace: test
spec:
  ports:
  - port: 1111
    protocol: TCP
    targetPort: 2222
  - port: 14
    protocol: UDP
    targetPort: 14
  selector:
    app: go
status:
  loadBalancer: {}
`,
		},
		{
			name: "expose with json override type should add port before existing port",
			overrides: `[
				{"op": "add", "path": "/spec/ports/0", "value": {"port": 1111, "protocol": "TCP", "targetPort": 2222}}
			]`,
			overrideType: "json",
			expected: `apiVersion: v1
kind: Service
metadata:
  creationTimestamp: null
  labels:
    svc: test
  name: foo
  namespace: test
spec:
  ports:
  - port: 1111
    protocol: TCP
    targetPort: 2222
  - port: 14
    protocol: UDP
    targetPort: 14
  selector:
    app: go
status:
  loadBalancer: {}
`,
		},
		{
			name: "expose with json override type should add port after existing port",
			overrides: `[
				{"op": "add", "path": "/spec/ports/1", "value": {"port": 1111, "protocol": "TCP", "targetPort": 2222}}
			]`,
			overrideType: "json",
			expected: `apiVersion: v1
kind: Service
metadata:
  creationTimestamp: null
  labels:
    svc: test
  name: foo
  namespace: test
spec:
  ports:
  - port: 14
    protocol: UDP
    targetPort: 14
  - port: 1111
    protocol: TCP
    targetPort: 2222
  selector:
    app: go
status:
  loadBalancer: {}
`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			codec := runtime.NewCodec(scheme.DefaultJSONEncoder(), scheme.Codecs.UniversalDecoder(scheme.Scheme.PrioritizedVersionsAllGroups()...))
			ns := scheme.Codecs.WithoutConversion()

			tf.Client = &fake.RESTClient{
				GroupVersion:         schema.GroupVersion{Version: "v1"},
				NegotiatedSerializer: ns,
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					switch p, m := req.URL.Path, req.Method; {
					case p == "/namespaces/test/services/baz" && m == "GET":
						return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &corev1.Service{
							ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
							Spec: corev1.ServiceSpec{
								Selector: map[string]string{"app": "go"},
							},
						})}, nil
					case p == "/namespaces/test/services" && m == "POST":
						return &http.Response{StatusCode: 200, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &corev1.Service{
							ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "", Labels: map[string]string{"svc": "test"}},
							Spec: corev1.ServiceSpec{
								Ports: []corev1.ServicePort{
									{
										Protocol:   corev1.ProtocolUDP,
										Port:       14,
										TargetPort: intstr.FromInt32(14),
									},
								},
								Selector: map[string]string{"app": "go"},
							},
						})}, nil
					default:
						t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
						return nil, nil
					}
				}),
			}

			ioStreams, _, buf, _ := genericiooptions.NewTestIOStreams()
			cmd := NewCmdExposeService(tf, ioStreams)
			cmd.SetOut(buf)
			cmd.Flags().Set("protocol", "UDP")
			cmd.Flags().Set("port", "14")
			cmd.Flags().Set("name", "foo")
			cmd.Flags().Set("labels", "svc=test")
			cmd.Flags().Set("dry-run", "client")
			cmd.Flags().Set("overrides", test.overrides)
			cmd.Flags().Set("override-type", test.overrideType)
			cmd.Flags().Set("output", "yaml")
			cmd.Run(cmd, []string{"service", "baz"})

			out := buf.String()

			if test.expected == "" {
				t.Errorf("%s: Invalid test case. Specify expected result.\n", test.name)
			}

			if !strings.Contains(out, test.expected) {
				t.Errorf("%s: Unexpected output! Expected\n%s\ngot\n%s", test.name, test.expected, out)
			}
		})
	}
}

func TestGenerateService(t *testing.T) {
	tests := map[string]struct {
		selector        string
		name            string
		port            string
		protocol        string
		protocols       string
		targetPort      string
		clusterIP       string
		labels          string
		externalIP      string
		serviceType     string
		sessionAffinity string
		setup           func(t *testing.T, exposeServiceOptions *ExposeServiceOptions) func()

		expected  *corev1.Service
		expectErr string
	}{
		"test1": {
			selector:   "foo=bar,baz=blah",
			name:       "test",
			port:       "80",
			protocol:   "TCP",
			targetPort: "1234",
			expected: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []corev1.ServicePort{
						{
							Port:       80,
							Protocol:   "TCP",
							TargetPort: intstr.FromInt32(1234),
						},
					},
				},
			},
		},
		"test2": {
			selector:   "foo=bar,baz=blah",
			name:       "test",
			port:       "80",
			protocol:   "UDP",
			targetPort: "foobar",
			expected: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []corev1.ServicePort{
						{
							Port:       80,
							Protocol:   "UDP",
							TargetPort: intstr.FromString("foobar"),
						},
					},
				},
			},
		},
		"test3": {
			selector:   "foo=bar,baz=blah",
			labels:     "key1=value1,key2=value2",
			name:       "test",
			port:       "80",
			protocol:   "TCP",
			targetPort: "1234",
			expected: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
					Labels: map[string]string{
						"key1": "value1",
						"key2": "value2",
					},
				},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []corev1.ServicePort{
						{
							Port:       80,
							Protocol:   "TCP",
							TargetPort: intstr.FromInt32(1234),
						},
					},
				},
			},
		},
		"test4": {
			selector:   "foo=bar,baz=blah",
			name:       "test",
			port:       "80",
			protocol:   "UDP",
			externalIP: "1.2.3.4",
			targetPort: "foobar",
			expected: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []corev1.ServicePort{
						{
							Port:       80,
							Protocol:   "UDP",
							TargetPort: intstr.FromString("foobar"),
						},
					},
					ExternalIPs: []string{"1.2.3.4"},
				},
			},
		},
		"test5": {
			selector:    "foo=bar,baz=blah",
			name:        "test",
			port:        "80",
			protocol:    "UDP",
			externalIP:  "1.2.3.4",
			serviceType: "LoadBalancer",
			targetPort:  "foobar",
			expected: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []corev1.ServicePort{
						{
							Port:       80,
							Protocol:   "UDP",
							TargetPort: intstr.FromString("foobar"),
						},
					},
					Type:        corev1.ServiceTypeLoadBalancer,
					ExternalIPs: []string{"1.2.3.4"},
				},
			},
		},
		"test6": {
			selector:    "foo=bar,baz=blah",
			name:        "test",
			port:        "80",
			protocol:    "UDP",
			targetPort:  "foobar",
			serviceType: string(corev1.ServiceTypeNodePort),
			expected: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []corev1.ServicePort{
						{
							Port:       80,
							Protocol:   "UDP",
							TargetPort: intstr.FromString("foobar"),
						},
					},
					Type: corev1.ServiceTypeNodePort,
				},
			},
		},
		"test7": {
			selector:    "foo=bar,baz=blah",
			name:        "test",
			port:        "80",
			protocol:    "UDP",
			targetPort:  "foobar",
			serviceType: string(corev1.ServiceTypeNodePort),
			expected: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []corev1.ServicePort{
						{
							Port:       80,
							Protocol:   "UDP",
							TargetPort: intstr.FromString("foobar"),
						},
					},
					Type: corev1.ServiceTypeNodePort,
				},
			},
		},
		"test8": {
			selector:   "foo=bar,baz=blah",
			name:       "test",
			port:       "80",
			protocol:   "TCP",
			targetPort: "1234",
			expected: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []corev1.ServicePort{
						{
							Port:       80,
							Protocol:   "TCP",
							TargetPort: intstr.FromInt32(1234),
						},
					},
				},
			},
		},
		"test9": {
			selector:        "foo=bar,baz=blah",
			name:            "test",
			port:            "80",
			protocol:        "TCP",
			sessionAffinity: "ClientIP",
			targetPort:      "1234",
			expected: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []corev1.ServicePort{
						{
							Port:       80,
							Protocol:   "TCP",
							TargetPort: intstr.FromInt32(1234),
						},
					},
					SessionAffinity: corev1.ServiceAffinityClientIP,
				},
			},
		},
		"test10": {
			selector:   "foo=bar,baz=blah",
			name:       "test",
			port:       "80",
			protocol:   "TCP",
			clusterIP:  "10.10.10.10",
			targetPort: "1234",
			expected: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []corev1.ServicePort{
						{

							Port:       80,
							Protocol:   "TCP",
							TargetPort: intstr.FromInt32(1234),
						},
					},
					ClusterIP: "10.10.10.10",
				},
			},
		},
		"test11": {
			selector:   "foo=bar,baz=blah",
			name:       "test",
			port:       "80",
			protocol:   "TCP",
			clusterIP:  "None",
			targetPort: "1234",
			expected: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []corev1.ServicePort{
						{

							Port:       80,
							Protocol:   "TCP",
							TargetPort: intstr.FromInt32(1234),
						},
					},
					ClusterIP: corev1.ClusterIPNone,
				},
			},
		},
		"test12": {
			selector:   "foo=bar",
			name:       "test",
			port:       "80,443",
			protocol:   "TCP",
			targetPort: "foobar",
			expected: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
					},
					Ports: []corev1.ServicePort{
						{
							Name:       "port-1",
							Port:       80,
							Protocol:   corev1.ProtocolTCP,
							TargetPort: intstr.FromString("foobar"),
						},
						{
							Name:       "port-2",
							Port:       443,
							Protocol:   corev1.ProtocolTCP,
							TargetPort: intstr.FromString("foobar"),
						},
					},
				},
			},
		},
		"test13": {
			selector:   "foo=bar",
			name:       "test",
			port:       "80,443",
			protocol:   "UDP",
			targetPort: "1234",
			expected: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
					},
					Ports: []corev1.ServicePort{
						{
							Name:       "port-1",
							Port:       80,
							Protocol:   corev1.ProtocolUDP,
							TargetPort: intstr.FromInt32(1234),
						},
						{
							Name:       "port-2",
							Port:       443,
							Protocol:   corev1.ProtocolUDP,
							TargetPort: intstr.FromInt32(1234),
						},
					},
				},
			},
		},
		"test14": {
			selector: "foo=bar",
			name:     "test",
			port:     "80,443",
			protocol: "TCP",
			expected: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
					},
					Ports: []corev1.ServicePort{
						{
							Name:       "port-1",
							Port:       80,
							Protocol:   corev1.ProtocolTCP,
							TargetPort: intstr.FromInt32(80),
						},
						{
							Name:       "port-2",
							Port:       443,
							Protocol:   corev1.ProtocolTCP,
							TargetPort: intstr.FromInt32(443),
						},
					},
				},
			},
		},
		"test15": {
			selector:  "foo=bar",
			name:      "test",
			port:      "80,8080",
			protocols: "8080/UDP",
			expected: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
					},
					Ports: []corev1.ServicePort{
						{
							Name:       "port-1",
							Port:       80,
							Protocol:   corev1.ProtocolTCP,
							TargetPort: intstr.FromInt32(80),
						},
						{
							Name:       "port-2",
							Port:       8080,
							Protocol:   corev1.ProtocolUDP,
							TargetPort: intstr.FromInt32(8080),
						},
					},
				},
			},
		},
		"test16": {
			selector:  "foo=bar",
			name:      "test",
			port:      "80,8080,8081",
			protocols: "8080/UDP,8081/TCP",
			expected: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
					},
					Ports: []corev1.ServicePort{
						{
							Name:       "port-1",
							Port:       80,
							Protocol:   corev1.ProtocolTCP,
							TargetPort: intstr.FromInt32(80),
						},
						{
							Name:       "port-2",
							Port:       8080,
							Protocol:   corev1.ProtocolUDP,
							TargetPort: intstr.FromInt32(8080),
						},
						{
							Name:       "port-3",
							Port:       8081,
							Protocol:   corev1.ProtocolTCP,
							TargetPort: intstr.FromInt32(8081),
						},
					},
				},
			},
		},
		"test17": {
			selector:  "foo=bar,baz=blah",
			name:      "test",
			protocol:  "TCP",
			clusterIP: "None",
			expected: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports:     []corev1.ServicePort{},
					ClusterIP: corev1.ClusterIPNone,
				},
			},
		},
		"test18": {
			selector:  "foo=bar",
			name:      "test",
			clusterIP: "None",
			expected: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
					},
					Ports:     []corev1.ServicePort{},
					ClusterIP: corev1.ClusterIPNone,
				},
			},
		},
		"test19": {
			selector:   "foo=bar,baz=blah",
			name:       "test",
			port:       "80",
			protocol:   "SCTP",
			targetPort: "1234",
			expected: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []corev1.ServicePort{
						{

							Port:       80,
							Protocol:   "SCTP",
							TargetPort: intstr.FromInt32(1234),
						},
					},
				},
			},
		},
		"test20": {
			selector:   "foo=bar,baz=blah",
			labels:     "key1=value1,key2=value2",
			name:       "test",
			port:       "80",
			protocol:   "SCTP",
			targetPort: "1234",
			expected: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
					Labels: map[string]string{
						"key1": "value1",
						"key2": "value2",
					},
				},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []corev1.ServicePort{
						{

							Port:       80,
							Protocol:   "SCTP",
							TargetPort: intstr.FromInt32(1234),
						},
					},
				},
			},
		},
		"test21": {
			selector:   "foo=bar,baz=blah",
			name:       "test",
			port:       "80",
			protocol:   "SCTP",
			targetPort: "1234",
			expected: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []corev1.ServicePort{
						{

							Port:       80,
							Protocol:   "SCTP",
							TargetPort: intstr.FromInt32(1234),
						},
					},
				},
			},
		},
		"test22": {
			selector:        "foo=bar,baz=blah",
			name:            "test",
			port:            "80",
			protocol:        "SCTP",
			sessionAffinity: "ClientIP",
			targetPort:      "1234",
			expected: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []corev1.ServicePort{
						{
							Port:       80,
							Protocol:   "SCTP",
							TargetPort: intstr.FromInt32(1234),
						},
					},
					SessionAffinity: corev1.ServiceAffinityClientIP,
				},
			},
		},
		"test23": {
			selector:   "foo=bar,baz=blah",
			name:       "test",
			port:       "80",
			protocol:   "SCTP",
			clusterIP:  "10.10.10.10",
			targetPort: "1234",
			expected: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []corev1.ServicePort{
						{
							Port:       80,
							Protocol:   "SCTP",
							TargetPort: intstr.FromInt32(1234),
						},
					},
					ClusterIP: "10.10.10.10",
				},
			},
		},
		"test24": {
			selector:   "foo=bar,baz=blah",
			name:       "test",
			port:       "80",
			protocol:   "SCTP",
			clusterIP:  "None",
			targetPort: "1234",
			expected: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []corev1.ServicePort{
						{

							Port:       80,
							Protocol:   "SCTP",
							TargetPort: intstr.FromInt32(1234),
						},
					},
					ClusterIP: corev1.ClusterIPNone,
				},
			},
		},
		"test25": {
			selector:   "foo=bar",
			name:       "test",
			port:       "80,443",
			protocol:   "SCTP",
			targetPort: "foobar",
			expected: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
					},
					Ports: []corev1.ServicePort{
						{
							Name:       "port-1",
							Port:       80,
							Protocol:   corev1.ProtocolSCTP,
							TargetPort: intstr.FromString("foobar"),
						},
						{
							Name:       "port-2",
							Port:       443,
							Protocol:   corev1.ProtocolSCTP,
							TargetPort: intstr.FromString("foobar"),
						},
					},
				},
			},
		},
		"test26": {
			selector: "foo=bar",
			name:     "test",
			port:     "80,443",
			protocol: "SCTP",
			expected: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
					},
					Ports: []corev1.ServicePort{
						{
							Name:       "port-1",
							Port:       80,
							Protocol:   corev1.ProtocolSCTP,
							TargetPort: intstr.FromInt32(80),
						},
						{
							Name:       "port-2",
							Port:       443,
							Protocol:   corev1.ProtocolSCTP,
							TargetPort: intstr.FromInt32(443),
						},
					},
				},
			},
		},
		"test27": {
			selector:  "foo=bar",
			name:      "test",
			port:      "80,8080",
			protocols: "8080/SCTP",
			expected: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
					},
					Ports: []corev1.ServicePort{
						{
							Name:       "port-1",
							Port:       80,
							Protocol:   corev1.ProtocolTCP,
							TargetPort: intstr.FromInt32(80),
						},
						{
							Name:       "port-2",
							Port:       8080,
							Protocol:   corev1.ProtocolSCTP,
							TargetPort: intstr.FromInt32(8080),
						},
					},
				},
			},
		},
		"test28": {
			selector:  "foo=bar",
			name:      "test",
			port:      "80,8080,8081,8082",
			protocols: "8080/UDP,8081/TCP,8082/SCTP",
			expected: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
					},
					Ports: []corev1.ServicePort{
						{
							Name:       "port-1",
							Port:       80,
							Protocol:   corev1.ProtocolTCP,
							TargetPort: intstr.FromInt32(80),
						},
						{
							Name:       "port-2",
							Port:       8080,
							Protocol:   corev1.ProtocolUDP,
							TargetPort: intstr.FromInt32(8080),
						},
						{
							Name:       "port-3",
							Port:       8081,
							Protocol:   corev1.ProtocolTCP,
							TargetPort: intstr.FromInt32(8081),
						},
						{
							Name:       "port-4",
							Port:       8082,
							Protocol:   corev1.ProtocolSCTP,
							TargetPort: intstr.FromInt32(8082),
						},
					},
				},
			},
		},
		"test 29": {
			selector:   "foo=bar,baz=blah",
			name:       "test",
			protocol:   "SCTP",
			targetPort: "1234",
			clusterIP:  "None",
			expected: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports:     []corev1.ServicePort{},
					ClusterIP: corev1.ClusterIPNone,
				},
			},
		},
		// Fixed #114402 kubectl expose fails for apps with same-port, different-protocol
		"test #114402": {
			selector:  "foo=bar,baz=blah",
			name:      "test",
			clusterIP: "None",
			protocols: "53/TCP,53/UDP",
			port:      "53",
			expected: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: corev1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []corev1.ServicePort{
						{
							Name:       "port-1-tcp",
							Port:       53,
							Protocol:   corev1.ProtocolTCP,
							TargetPort: intstr.FromInt32(53),
						},
						{
							Name:       "port-1-udp",
							Port:       53,
							Protocol:   corev1.ProtocolUDP,
							TargetPort: intstr.FromInt32(53),
						},
					},
					ClusterIP: corev1.ClusterIPNone,
				},
			},
		},
		"check selector": {
			name:       "test",
			protocol:   "SCTP",
			targetPort: "1234",
			clusterIP:  "None",
			expectErr:  `selector must be specified`,
		},
		"check name": {
			selector:   "foo=bar,baz=blah",
			protocol:   "SCTP",
			targetPort: "1234",
			clusterIP:  "None",
			expectErr:  `name must be specified`,
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			exposeServiceOptions := ExposeServiceOptions{
				Selector:        test.selector,
				Name:            test.name,
				Protocol:        test.protocol,
				Protocols:       test.protocols,
				Port:            test.port,
				ClusterIP:       test.clusterIP,
				TargetPort:      test.targetPort,
				Labels:          test.labels,
				ExternalIP:      test.externalIP,
				Type:            test.serviceType,
				SessionAffinity: test.sessionAffinity,
			}

			service, err := exposeServiceOptions.createService()
			if test.expectErr == "" {
				require.NoError(t, err)
				if !apiequality.Semantic.DeepEqual(service, test.expected) {
					t.Errorf("\nexpected:\n%#v\ngot:\n%#v", test.expected, service)
				}
			} else {
				require.Error(t, err)
				require.EqualError(t, err, test.expectErr)
			}

		})

	}
}
