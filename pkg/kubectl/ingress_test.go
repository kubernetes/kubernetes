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

package kubectl

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/apis/extensions"
)

func TestIngressBasicGenerate(t *testing.T) {
	tests := []struct {
		name        string
		serviceName string
		servicePort intstr.IntOrString
		host        []string
		tlsAcme     bool
		expected    *extensions.Ingress
		expectErr   string
	}{
		{
			name: "minimal-ok",
			host: []string{"minimal-ok.example.com"},
			expected: &extensions.Ingress{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "minimal-ok",
					Labels: map[string]string{"app": "minimal-ok"},
				},
				Spec: extensions.IngressSpec{
					Rules: []extensions.IngressRule{
						{
							Host: "minimal-ok.example.com",
							IngressRuleValue: extensions.IngressRuleValue{
								HTTP: &extensions.HTTPIngressRuleValue{
									Paths: []extensions.HTTPIngressPath{
										{
											Path: "/",
											Backend: extensions.IngressBackend{
												ServiceName: "minimal-ok",
												ServicePort: intstr.FromInt(80),
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
		{
			name:      "hosts-missing",
			host:      []string{},
			expectErr: "at least one host must be specified",
		},
		{
			name:        "bad-int-port",
			servicePort: intstr.FromInt(123456),
			host:        []string{"bad-int-port.example.com"},
			expectErr:   "must be between 1 and 65535",
		},
		{
			name:        "bad-string-port",
			servicePort: intstr.FromString("port.names.cannot.contain.dots"),
			host:        []string{"bad-string-port.example.com"},
			expectErr:   "must be no more than 15 characters",
		},
		{
			name: "multiple-hosts",
			host: []string{"a.example.com", "b.example.com"},
			expected: &extensions.Ingress{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "multiple-hosts",
					Labels: map[string]string{"app": "multiple-hosts"},
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
												ServiceName: "multiple-hosts",
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
												ServiceName: "multiple-hosts",
												ServicePort: intstr.FromInt(80),
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
			expectErr: "name must be specified",
		},
		{
			name:        "specified-backend",
			host:        []string{"specified-backend.example.com"},
			serviceName: "override-name",
			servicePort: intstr.FromString("override-port"),
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
		generator := IngressV1Beta1{
			Name:        test.name,
			Host:        test.host,
			TLSAcme:     test.tlsAcme,
			ServiceName: test.serviceName,
			ServicePort: test.servicePort,
		}
		obj, err := generator.StructuredGenerate()
		if test.expectErr == "" && err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}
		if test.expectErr != "" && err != nil {
			actualErr := fmt.Sprintf("%v", err)
			if !strings.Contains(actualErr, test.expectErr) {
				t.Errorf("expected error %q to contain %q", err, test.expectErr)
				continue
			}
			continue
		}
		if !reflect.DeepEqual(obj.(*extensions.Ingress), test.expected) {
			t.Errorf("test: %v\nexpected:\n%#v\nsaw:\n%#v", test.name, test.expected, obj.(*extensions.Ingress))
		}
	}
}
