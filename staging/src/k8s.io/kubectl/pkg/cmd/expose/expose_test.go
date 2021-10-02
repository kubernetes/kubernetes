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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/cli-runtime/pkg/genericclioptions"
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
							TargetPort: intstr.FromInt(14),
						},
					},
					Selector: map[string]string{"func": "stream"},
				},
			},
			expected: "service/foo exposed",
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
			flags: map[string]string{"selector": "func=stream", "protocol": "UDP", "port": "14", "name": "foo", "labels": "svc=test", "serviceType": "L ncer", "dry-run": "client"},
			output: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "", Labels: map[string]string{"svc": "test"}},
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Protocol:   corev1.ProtocolUDP,
							Port:       14,
							TargetPort: intstr.FromInt(14),
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
			flags: map[string]string{"selector": "func=stream", "protocol": "UDP", "port": "14", "name": "foo", "labels": "svc=test", "serviceType": "L ncer", "session-affinity": "ClientIP", "dry-run": "client"},
			output: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "", Labels: map[string]string{"svc": "test"}},
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Protocol:   corev1.ProtocolUDP,
							Port:       14,
							TargetPort: intstr.FromInt(14),
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
							TargetPort: intstr.FromInt(14),
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
							TargetPort: intstr.FromInt(14),
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
							TargetPort: intstr.FromInt(14),
						},
					},
					Selector: map[string]string{"func": "stream"},
				},
			},
			expected: "service/foo exposed (dry run)",
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
							TargetPort: intstr.FromInt(14),
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
							TargetPort: intstr.FromInt(14),
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
							TargetPort: intstr.FromInt(14),
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
			flags: map[string]string{"selector": "func=stream", "protocol": "UDP", "port": "14", "name": "foo", "labels": "svc=test", "serviceType": "L ncer", "dry-run": "client", "output": "yaml"},
			output: &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "", Labels: map[string]string{"svc": "test"}},
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Protocol:   corev1.ProtocolUDP,
							Port:       14,
							TargetPort: intstr.FromInt(14),
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

			codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
			ns := scheme.Codecs.WithoutConversion()

			tf.Client = &fake.RESTClient{
				GroupVersion:         schema.GroupVersion{Version: "corev1"},
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

			ioStreams, _, buf, _ := genericclioptions.NewTestIOStreams()
			cmd := NewCmdExposeService(tf, ioStreams)
			cmd.SetOutput(buf)
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

func TestGenerateService(t *testing.T) {
	tests := map[string]struct {
		selector 		string
		name 	 		string
		port     		string
		protocol    	string
		targetPort  	string
		clusterIP   	string
		labels			string
		externalIP		string
		serviceType 	string
		sessionAffinity string
		setup         func(t *testing.T, exposeServiceOptions *ExposeServiceOptions) func()

		expected  *corev1.Service
		expectErr string
	}{
		"test1": {
			selector:       "foo=bar,baz=blah",
			name:           "test",
			port:           "80",
			protocol:       "TCP",
			targetPort:     "1234",
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
							TargetPort: intstr.FromInt(1234),
						},
					},
				},
			},
		},
		"test2": {
			selector:       "foo=bar,baz=blah",
			name:           "test",
			port:           "80",
			protocol:       "UDP",
			targetPort:		"foobar",
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
			selector:       "foo=bar,baz=blah",
			labels:         "key1=value1,key2=value2",
			name:           "test",
			port:           "80",
			protocol:       "TCP",
			targetPort:		"1234",		
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
							TargetPort: intstr.FromInt(1234),
						},
					},
				},
			},
		},
		"test4": {
			selector:       "foo=bar,baz=blah",
			name:           "test",
			port:           "80",
			protocol:       "UDP",
			externalIP:     "1.2.3.4",
			targetPort:		"foobar",
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
			selector:       "foo=bar,baz=blah",
			name:           "test",
			port:           "80",
			protocol:       "UDP",
			externalIP:     "1.2.3.4",
			serviceType:	"LoadBalancer",
			targetPort:		"foobar",
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
			selector:       "foo=bar,baz=blah",
			name:           "test",
			port:           "80",
			protocol:       "UDP",
			serviceType:     string(corev1.ServiceTypeNodePort),
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
			selector:                      "foo=bar,baz=blah",
			name:                          "test",
			port:                          "80",
			protocol:                      "UDP",
			targetPort:					   "foobar",
			serviceType:                          string(corev1.ServiceTypeNodePort),
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
			selector:       "foo=bar,baz=blah",
			name:           "test",
			port:           "80",
			protocol:       "TCP",	
			targetPort:		"1234",	
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
							Name:       "default",
							Port:       80,
							Protocol:   "TCP",
							TargetPort: intstr.FromInt(1234),
						},
					},
				},
			},
		},
		"test9": {
			selector:         "foo=bar,baz=blah",
			name:             "test",
			port:             "80",
			protocol:         "TCP",
			sessionAffinity:  "ClientIP",
			targetPort:		  "1234",	
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
							Name:       "default",
							Port:       80,
							Protocol:   "TCP",
							TargetPort: intstr.FromInt(1234),
						},
					},
					SessionAffinity: corev1.ServiceAffinityClientIP,
				},
			},
		},
		"test10": {
			selector:       "foo=bar,baz=blah",
			name:           "test",
			port:           "80",
			protocol:       "TCP",		
			clusterIP:      "10.10.10.10",
			targetPort:		"1234",
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
							TargetPort: intstr.FromInt(1234),
						},
					},
					ClusterIP: "10.10.10.10",
				},
			},
		},
		"test11": {
			selector:       "foo=bar,baz=blah",
			name:           "test",
			port:           "80",
			protocol:       "TCP",		
			clusterIP:      "None",
			targetPort:		"1234",
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
							TargetPort: intstr.FromInt(1234),
						},
					},
					ClusterIP: corev1.ClusterIPNone,
				},
			},
		},
		"test12": {
			selector:       "foo=bar",
			name:           "test",
			port:           "80,443",
			protocol:       "TCP",
			targetPort:		"foobar",
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
			selector:    "foo=bar",
			name:        "test",
			port:        "80,443",
			protocol:    "UDP",
			targetPort:  "1234",
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
							TargetPort: intstr.FromInt(1234),
						},
						{
							Name:       "port-2",
							Port:       443,
							Protocol:   corev1.ProtocolUDP,
							TargetPort: intstr.FromInt(1234),
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
							TargetPort: intstr.FromInt(80),
						},
						{
							Name:       "port-2",
							Port:       443,
							Protocol:   corev1.ProtocolTCP,
							TargetPort: intstr.FromInt(443),
						},
					},
				},
			},
		},
		"test15": {
			selector:  "foo=bar",
			name:      "test",
			port:      "80,8080",
			protocol:  "8080/UDP",
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
							TargetPort: intstr.FromInt(80),
						},
						{
							Name:       "port-2",
							Port:       8080,
							Protocol:   corev1.ProtocolUDP,
							TargetPort: intstr.FromInt(8080),
						},
					},
				},
			},
		},
		"test16": {
			selector:  "foo=bar",
			name:      "test",
			port:      "80,8080,8081",
			protocol:  "8080/UDP,8081/TCP",
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
							TargetPort: intstr.FromInt(80),
						},
						{
							Name:       "port-2",
							Port:       8080,
							Protocol:   corev1.ProtocolUDP,
							TargetPort: intstr.FromInt(8080),
						},
						{
							Name:       "port-3",
							Port:       8081,
							Protocol:   corev1.ProtocolTCP,
							TargetPort: intstr.FromInt(8081),
						},
					},
				},
			},
		},
		"test17": {
			selector:       "foo=bar,baz=blah",
			name:           "test",
			protocol:       "TCP",		
			clusterIP:      "None",
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
			selector:   "foo=bar",
			name:       "test",
			clusterIP:  "None",
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
		"test18.5": {
			selector:       "foo=bar,baz=blah",
			name:           "test",
			port:           "80",
			protocol:       "SCTP",		
			targetPort:		"1234",
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
							TargetPort: intstr.FromInt(1234),
						},
					},
				},
			},
		},
		"test19": {
			selector:       "foo=bar,baz=blah",
			labels:         "key1=value1,key2=value2",
			name:           "test",
			port:           "80",
			protocol:       "SCTP",	
			targetPort:		"1234",
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
							TargetPort: intstr.FromInt(1234),
						},
					},
				},
			},
		},
		"test20": {
			selector:       "foo=bar,baz=blah",
			name:           "test",
			port:           "80",
			protocol:       "SCTP",		
			targetPort:		"1234",
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
							Name:       "default",
							Port:       80,
							Protocol:   "SCTP",
							TargetPort: intstr.FromInt(1234),
						},
					},
				},
			},
		},
		"test21": {
			selector:         "foo=bar,baz=blah",
			name:             "test",
			port:             "80",
			protocol:         "SCTP",
			sessionAffinity:  "ClientIP",
			targetPort:		  "1234",
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
							Name:       "default",
							Port:       80,
							Protocol:   "SCTP",
							TargetPort: intstr.FromInt(1234),
						},
					},
					SessionAffinity: corev1.ServiceAffinityClientIP,
				},
			},
		},
		"test22": {
			selector:       "foo=bar,baz=blah",
			name:           "test",
			port:           "80",
			protocol:       "SCTP",		
			clusterIP:      "10.10.10.10",
			targetPort:		"1234",
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
							TargetPort: intstr.FromInt(1234),
						},
					},
					ClusterIP: "10.10.10.10",
				},
			},
		},
		"test23": {
			selector:       "foo=bar,baz=blah",
			name:           "test",
			port:           "80",
			protocol:       "SCTP",		
			clusterIP:      "None",
			targetPort:		"1234",
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
							TargetPort: intstr.FromInt(1234),
						},
					},
					ClusterIP: corev1.ClusterIPNone,
				},
			},
		},
		"test24": {
			selector:       "foo=bar",
			name:           "test",
			port:           "80,443",
			protocol:       "SCTP",
			targetPort:		"foobar",
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
		"test25": {
			selector: "foo=bar",
			name:     "test",
			port:    "80,443",
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
							TargetPort: intstr.FromInt(80),
						},
						{
							Name:       "port-2",
							Port:       443,
							Protocol:   corev1.ProtocolSCTP,
							TargetPort: intstr.FromInt(443),
						},
					},
				},
			},
		},
		"test26": {
			selector:  "foo=bar",
			name:      "test",
			port:      "80,8080",
			protocol:  "8080/SCTP",
			targetPort:"foobar",
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
							TargetPort: intstr.FromInt(80),
						},
						{
							Name:       "port-2",
							Port:       8080,
							Protocol:   corev1.ProtocolSCTP,
							TargetPort: intstr.FromInt(8080),
						},
					},
				},
			},
		},
		"test27": {
			selector:  "foo=bar",
			name:      "test",
			port:     "80,8080,8081,8082",
			protocol: "8080/UDP,8081/TCP,8082/SCTP",
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
							TargetPort: intstr.FromInt(80),
						},
						{
							Name:       "port-2",
							Port:       8080,
							Protocol:   corev1.ProtocolUDP,
							TargetPort: intstr.FromInt(8080),
						},
						{
							Name:       "port-3",
							Port:       8081,
							Protocol:   corev1.ProtocolTCP,
							TargetPort: intstr.FromInt(8081),
						},
						{
							Name:       "port-4",
							Port:       8082,
							Protocol:   corev1.ProtocolSCTP,
							TargetPort: intstr.FromInt(8082),
						},
					},
				},
			},
		},
		"test 28": {
			selector:       "foo=bar,baz=blah",
			name:           "test",
			protocol:       "SCTP",
			targetPort:     "1234",
			clusterIP:     	"None",
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
		"check selector": {
			name:           "test",
			protocol:       "SCTP",
			targetPort:     "1234",
			clusterIP:     	"None",
			expectErr:      `selector must be specified`,
		},
		"check name": {
			selector:       "foo=bar,baz=blah",
			protocol:       "SCTP",
			targetPort:     "1234",
			clusterIP:     	"None",
			expectErr:      `name must be specified`,
		},
		"check ports": {
			selector:       "foo=bar,baz=blah",
			name:           "test",
			protocol:       "SCTP",
			targetPort:     "1234",
			expectErr:      `'ports' or 'port' is a required parameter`,
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			var service *corev1.Service = nil
			exposeServiceOptions := ExposeServiceOptions{
				Selector:		 test.selector,
				Name:			 test.name,
				Protocol:		 test.protocol,
				Port:			 test.port,
				ClusterIP:		 test.clusterIP,
				TargetPort: 	 test.targetPort,
				Labels:			 test.labels,
				ExternalIP: 	 test.externalIP,
				Type:			 test.serviceType,
				SessionAffinity: test.sessionAffinity,
			}
	
			if test.setup != nil {
				if teardown := test.setup(t, &exposeServiceOptions); teardown != nil {
					defer teardown()
				}
			}

			err := exposeServiceOptions.Validate()

			if err == nil {
				service, err = exposeServiceOptions.createService()
			}
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