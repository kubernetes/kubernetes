/*
Copyright 2017 The Kubernetes Authors.

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
	"io"
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/rest/fake"
	"k8s.io/kubernetes/pkg/apis/extensions"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
)

type testIngressPrinter struct {
	CachedIngress *extensions.Ingress
}

func (t *testIngressPrinter) PrintObj(obj runtime.Object, out io.Writer) error {
	t.CachedIngress = obj.(*extensions.Ingress)
	return nil
}

func (t *testIngressPrinter) AfterPrint(output io.Writer, res string) error {
	return nil
}

func (t *testIngressPrinter) HandledResources() []string {
	return []string{}
}

func TestCreateIngress(t *testing.T) {
	f, tf, _, _ := cmdtesting.NewAPIFactory()
	printer := &testIngressPrinter{}
	tf.Printer = printer
	tf.Namespace = "test"
	tf.Client = &fake.RESTClient{}
	tf.ClientConfig = defaultClientConfig()

	tests := []struct {
		name        string
		host        []string
		serviceName string
		servicePort string
		tlsAcme     bool
		expected    *extensions.Ingress
	}{
		{
			name:    "acme-example",
			host:    []string{"a.example.com", "b.example.com"},
			tlsAcme: true,
			expected: &extensions.Ingress{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "acme-example",
					Labels:      map[string]string{"app": "acme-example"},
					Annotations: map[string]string{"kubernetes.io/tls-acme": "true"},
				},
				Spec: extensions.IngressSpec{
					Rules: []extensions.IngressRule{
						{
							Host: "a.example.com",
							IngressRuleValue: extensions.IngressRuleValue{
								HTTP: &extensions.HTTPIngressRuleValue{
									Paths: []extensions.HTTPIngressPath{
										{
											Path: "/",
											Backend: extensions.IngressBackend{
												ServiceName: "acme-example",
												ServicePort: intstr.FromInt(80),
											},
										},
									},
								},
							},
						},
						{
							Host: "b.example.com",
							IngressRuleValue: extensions.IngressRuleValue{
								HTTP: &extensions.HTTPIngressRuleValue{
									Paths: []extensions.HTTPIngressPath{
										{
											Path: "/",
											Backend: extensions.IngressBackend{
												ServiceName: "acme-example",
												ServicePort: intstr.FromInt(80),
											},
										},
									},
								},
							},
						},
					},
					TLS: []extensions.IngressTLS{
						{
							Hosts:      []string{"a.example.com", "b.example.com"},
							SecretName: "tls-acme-example",
						},
					},
				},
			},
		},

		{
			name:        "specified-backend",
			host:        []string{"specified-backend.example.com"},
			serviceName: "override-name",
			servicePort: "override-port",
			expected: &extensions.Ingress{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "specified-backend",
					Labels: map[string]string{"app": "specified-backend"},
				},
				Spec: extensions.IngressSpec{
					Rules: []extensions.IngressRule{
						{
							Host: "specified-backend.example.com",
							IngressRuleValue: extensions.IngressRuleValue{
								HTTP: &extensions.HTTPIngressRuleValue{
									Paths: []extensions.HTTPIngressPath{
										{
											Path: "/",
											Backend: extensions.IngressBackend{
												ServiceName: "override-name",
												ServicePort: intstr.FromString("override-port"),
											},
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	for _, test := range tests {
		buf := bytes.NewBuffer([]byte{})
		cmd := NewCmdCreateIngress(f, buf)
		cmd.Flags().Set("dry-run", "true")
		cmd.Flags().Set("output", "object")
		for _, host := range test.host {
			cmd.Flags().Set("host", host)
		}
		if test.tlsAcme {
			cmd.Flags().Set("tls-acme", "true")
		}
		cmd.Flags().Set("service-name", test.serviceName)
		if test.servicePort != "" {
			cmd.Flags().Set("service-port", test.servicePort)
		}
		cmd.Run(cmd, []string{test.name})
		if !reflect.DeepEqual(test.expected, printer.CachedIngress) {
			t.Errorf("%s:\nexpected:\n%#v\nsaw:\n%#v", test.name, test.expected, printer.CachedIngress)
		}
	}
}
