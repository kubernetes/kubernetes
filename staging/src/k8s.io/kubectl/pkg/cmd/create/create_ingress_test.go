/*
Copyright 2020 The Kubernetes Authors.

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

package create

import (
	"testing"

	networkingv1 "k8s.io/api/networking/v1"
	v1 "k8s.io/api/networking/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestCreateIngressValidation(t *testing.T) {
	tests := map[string]struct {
		defaultbackend string
		ingressclass   string
		rules          []string
		expected       string
	}{
		"no default backend and rule": {
			defaultbackend: "",
			rules:          []string{},
			expected:       "not enough information provided: every ingress has to either specify a default-backend (which catches all traffic) or a list of rules (which catch specific paths)",
		},
		"invalid default backend separator": {
			defaultbackend: "xpto,4444",
			expected:       "default-backend should be in format servicename:serviceport",
		},
		"default backend without port": {
			defaultbackend: "xpto",
			expected:       "default-backend should be in format servicename:serviceport",
		},
		"default backend is ok": {
			defaultbackend: "xpto:4444",
			expected:       "",
		},
		"multiple conformant rules": {
			rules: []string{
				"foo.com/path*=svc:8080",
				"bar.com/admin*=svc2:http",
			},
			expected: "",
		},
		"one invalid and two valid rules": {
			rules: []string{
				"foo.com=svc:redis,tls",
				"foo.com/path/subpath*=othersvc:8080",
				"foo.com/*=svc:8080,tls=secret1",
			},
			expected: "rule foo.com=svc:redis,tls is invalid and should be in format host/path=svcname:svcport[,tls[=secret]]",
		},
		"service without port": {
			rules: []string{
				"foo.com/=svc,tls",
			},
			expected: "rule foo.com/=svc,tls is invalid and should be in format host/path=svcname:svcport[,tls[=secret]]",
		},
		"valid tls rule without secret": {
			rules: []string{
				"foo.com/=svc:http,tls=",
			},
			expected: "",
		},
		"valid tls rule with secret": {
			rules: []string{
				"foo.com/=svc:http,tls=secret123",
			},
			expected: "",
		},
		"valid path with type prefix": {
			rules: []string{
				"foo.com/admin*=svc:8080",
			},
			expected: "",
		},
		"wildcard host": {
			rules: []string{
				"*.foo.com/admin*=svc:8080",
			},
			expected: "",
		},
		"invalid separation between ingress and service": {
			rules: []string{
				"*.foo.com/path,svc:8080",
			},
			expected: "rule *.foo.com/path,svc:8080 is invalid and should be in format host/path=svcname:svcport[,tls[=secret]]",
		},
		"two invalid and one valid rule": {
			rules: []string{
				"foo.com/path/subpath*=svc:redis,tls=blo",
				"foo.com=othersvc:8080",
				"foo.com/admin=svc,tls=secret1",
			},
			expected: "rule foo.com=othersvc:8080 is invalid and should be in format host/path=svcname:svcport[,tls[=secret]]",
		},
		"valid catch all rule": {
			rules: []string{
				"/path/subpath*=svc:redis,tls=blo",
			},
			expected: "",
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			o := &CreateIngressOptions{
				DefaultBackend: tc.defaultbackend,
				Rules:          tc.rules,
				IngressClass:   tc.ingressclass,
			}

			err := o.Validate()
			if err != nil && err.Error() != tc.expected {
				t.Errorf("unexpected error: %v", err)
			}
			if tc.expected != "" && err == nil {
				t.Errorf("expected error, got no error")
			}

		})
	}
}

func TestCreateIngress(t *testing.T) {
	ingressName := "test-ingress"
	ingressClass := "nginx"
	pathTypeExact := networkingv1.PathTypeExact
	pathTypePrefix := networkingv1.PathTypePrefix
	tests := map[string]struct {
		defaultbackend string
		rules          []string
		ingressclass   string
		annotations    []string
		expected       *networkingv1.Ingress
	}{
		"catch all host and default backend with default TLS returns empty TLS": {
			rules: []string{
				"/=catchall:8080,tls=",
			},
			ingressclass:   ingressClass,
			defaultbackend: "service1:https",
			annotations:    []string{},
			expected: &networkingv1.Ingress{
				TypeMeta: metav1.TypeMeta{
					APIVersion: networkingv1.SchemeGroupVersion.String(),
					Kind:       "Ingress",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:        ingressName,
					Annotations: map[string]string{},
				},
				Spec: networkingv1.IngressSpec{
					IngressClassName: &ingressClass,
					DefaultBackend: &networkingv1.IngressBackend{
						Service: &networkingv1.IngressServiceBackend{
							Name: "service1",
							Port: networkingv1.ServiceBackendPort{
								Name: "https",
							},
						},
					},
					TLS: []v1.IngressTLS{},
					Rules: []networkingv1.IngressRule{
						{
							Host: "",
							IngressRuleValue: networkingv1.IngressRuleValue{
								HTTP: &networkingv1.HTTPIngressRuleValue{
									Paths: []networkingv1.HTTPIngressPath{
										{
											Path:     "/",
											PathType: &pathTypeExact,
											Backend: networkingv1.IngressBackend{
												Service: &networkingv1.IngressServiceBackend{
													Name: "catchall",
													Port: networkingv1.ServiceBackendPort{
														Number: 8080,
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
			},
		},
		"catch all with path of type prefix and secret name": {
			rules: []string{
				"/path*=catchall:8080,tls=secret1",
			},
			ingressclass:   ingressClass,
			defaultbackend: "service1:https",
			annotations:    []string{},
			expected: &networkingv1.Ingress{
				TypeMeta: metav1.TypeMeta{
					APIVersion: networkingv1.SchemeGroupVersion.String(),
					Kind:       "Ingress",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:        ingressName,
					Annotations: map[string]string{},
				},
				Spec: networkingv1.IngressSpec{
					IngressClassName: &ingressClass,
					DefaultBackend: &networkingv1.IngressBackend{
						Service: &networkingv1.IngressServiceBackend{
							Name: "service1",
							Port: networkingv1.ServiceBackendPort{
								Name: "https",
							},
						},
					},
					TLS: []v1.IngressTLS{
						{
							SecretName: "secret1",
						},
					},
					Rules: []networkingv1.IngressRule{
						{
							Host: "",
							IngressRuleValue: networkingv1.IngressRuleValue{
								HTTP: &networkingv1.HTTPIngressRuleValue{
									Paths: []networkingv1.HTTPIngressPath{
										{
											Path:     "/path",
											PathType: &pathTypePrefix,
											Backend: networkingv1.IngressBackend{
												Service: &networkingv1.IngressServiceBackend{
													Name: "catchall",
													Port: networkingv1.ServiceBackendPort{
														Number: 8080,
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
			},
		},
		"mixed hosts with mixed TLS configuration and a default backend": {
			rules: []string{
				"foo.com/=foo-svc:8080,tls=",
				"foo.com/admin=foo-admin-svc:http,tls=",
				"bar.com/prefix*=bar-svc:8080,tls=bar-secret",
				"bar.com/noprefix=barnp-svc:8443,tls",
				"foobar.com/*=foobar-svc:https",
				"foobar1.com/*=foobar1-svc:https,tls=bar-secret",
			},
			defaultbackend: "service2:8080",
			annotations:    []string{},
			expected: &networkingv1.Ingress{
				TypeMeta: metav1.TypeMeta{
					APIVersion: networkingv1.SchemeGroupVersion.String(),
					Kind:       "Ingress",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:        ingressName,
					Annotations: map[string]string{},
				},
				Spec: networkingv1.IngressSpec{
					DefaultBackend: &networkingv1.IngressBackend{
						Service: &networkingv1.IngressServiceBackend{
							Name: "service2",
							Port: networkingv1.ServiceBackendPort{
								Number: 8080,
							},
						},
					},
					TLS: []v1.IngressTLS{
						{
							Hosts: []string{
								"foo.com",
							},
						},
						{
							Hosts: []string{
								"bar.com",
								"foobar1.com",
							},
							SecretName: "bar-secret",
						},
					},
					Rules: []networkingv1.IngressRule{
						{
							Host: "foo.com",
							IngressRuleValue: networkingv1.IngressRuleValue{
								HTTP: &networkingv1.HTTPIngressRuleValue{
									Paths: []networkingv1.HTTPIngressPath{
										{
											Path:     "/",
											PathType: &pathTypeExact,
											Backend: networkingv1.IngressBackend{
												Service: &networkingv1.IngressServiceBackend{
													Name: "foo-svc",
													Port: networkingv1.ServiceBackendPort{
														Number: 8080,
													},
												},
											},
										},
										{
											Path:     "/admin",
											PathType: &pathTypeExact,
											Backend: networkingv1.IngressBackend{
												Service: &networkingv1.IngressServiceBackend{
													Name: "foo-admin-svc",
													Port: networkingv1.ServiceBackendPort{
														Name: "http",
													},
												},
											},
										},
									},
								},
							},
						},
						{
							Host: "bar.com",
							IngressRuleValue: networkingv1.IngressRuleValue{
								HTTP: &networkingv1.HTTPIngressRuleValue{
									Paths: []networkingv1.HTTPIngressPath{
										{
											Path:     "/prefix",
											PathType: &pathTypePrefix,
											Backend: networkingv1.IngressBackend{
												Service: &networkingv1.IngressServiceBackend{
													Name: "bar-svc",
													Port: networkingv1.ServiceBackendPort{
														Number: 8080,
													},
												},
											},
										},
										{
											Path:     "/noprefix",
											PathType: &pathTypeExact,
											Backend: networkingv1.IngressBackend{
												Service: &networkingv1.IngressServiceBackend{
													Name: "barnp-svc",
													Port: networkingv1.ServiceBackendPort{
														Number: 8443,
													},
												},
											},
										},
									},
								},
							},
						},
						{
							Host: "foobar.com",
							IngressRuleValue: networkingv1.IngressRuleValue{
								HTTP: &networkingv1.HTTPIngressRuleValue{
									Paths: []networkingv1.HTTPIngressPath{
										{
											Path:     "/",
											PathType: &pathTypePrefix,
											Backend: networkingv1.IngressBackend{
												Service: &networkingv1.IngressServiceBackend{
													Name: "foobar-svc",
													Port: networkingv1.ServiceBackendPort{
														Name: "https",
													},
												},
											},
										},
									},
								},
							},
						},
						{
							Host: "foobar1.com",
							IngressRuleValue: networkingv1.IngressRuleValue{
								HTTP: &networkingv1.HTTPIngressRuleValue{
									Paths: []networkingv1.HTTPIngressPath{
										{
											Path:     "/",
											PathType: &pathTypePrefix,
											Backend: networkingv1.IngressBackend{
												Service: &networkingv1.IngressServiceBackend{
													Name: "foobar1-svc",
													Port: networkingv1.ServiceBackendPort{
														Name: "https",
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
			},
		},
		"simple ingress with annotation": {
			rules: []string{
				"foo.com/=svc:8080,tls=secret1",
				"foo.com/subpath*=othersvc:8080,tls=secret1",
			},
			annotations: []string{
				"ingress.kubernetes.io/annotation1=bla",
				"ingress.kubernetes.io/annotation2=blo",
				"ingress.kubernetes.io/annotation3=ble",
			},
			expected: &networkingv1.Ingress{
				TypeMeta: metav1.TypeMeta{
					APIVersion: networkingv1.SchemeGroupVersion.String(),
					Kind:       "Ingress",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: ingressName,
					Annotations: map[string]string{
						"ingress.kubernetes.io/annotation1": "bla",
						"ingress.kubernetes.io/annotation3": "ble",
						"ingress.kubernetes.io/annotation2": "blo",
					},
				},
				Spec: networkingv1.IngressSpec{
					TLS: []v1.IngressTLS{
						{
							Hosts: []string{
								"foo.com",
							},
							SecretName: "secret1",
						},
					},
					Rules: []networkingv1.IngressRule{
						{
							Host: "foo.com",
							IngressRuleValue: networkingv1.IngressRuleValue{
								HTTP: &networkingv1.HTTPIngressRuleValue{
									Paths: []networkingv1.HTTPIngressPath{
										{
											Path:     "/",
											PathType: &pathTypeExact,
											Backend: networkingv1.IngressBackend{
												Service: &networkingv1.IngressServiceBackend{
													Name: "svc",
													Port: networkingv1.ServiceBackendPort{
														Number: 8080,
													},
												},
											},
										},
										{
											Path:     "/subpath",
											PathType: &pathTypePrefix,
											Backend: networkingv1.IngressBackend{
												Service: &networkingv1.IngressServiceBackend{
													Name: "othersvc",
													Port: networkingv1.ServiceBackendPort{
														Number: 8080,
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
			},
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			o := &CreateIngressOptions{
				Name:           ingressName,
				IngressClass:   tc.ingressclass,
				Annotations:    tc.annotations,
				DefaultBackend: tc.defaultbackend,
				Rules:          tc.rules,
			}
			ingress := o.createIngress()
			if !apiequality.Semantic.DeepEqual(ingress, tc.expected) {
				t.Errorf("expected:\n%#v\ngot:\n%#v", tc.expected, ingress)
			}
		})
	}
}
