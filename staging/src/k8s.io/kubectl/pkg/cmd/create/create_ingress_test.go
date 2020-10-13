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
	"strings"
	"testing"

	"k8s.io/api/networking/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestCreateIngress(t *testing.T) {
	ingressName := "fake-ingress"
	tests := map[string]struct {
		name         string
		host         string
		serviceName  string
		servicePort  string
		path         string
		expectErrMsg string
		expect       *v1.Ingress
	}{
		"test-valid-case": {
			name:        "fake-ingress",
			host:        "foo.bar.com",
			serviceName: "fake-service",
			servicePort: "https",
			path:        "/api",
			expect: &v1.Ingress{
				TypeMeta: metav1.TypeMeta{APIVersion: v1.SchemeGroupVersion.String(), Kind: "Ingress"},
				ObjectMeta: metav1.ObjectMeta{
					Name: "fake-ingress",
				},
				Spec: v1.IngressSpec{
					Rules: []v1.IngressRule{
						{
							Host: "foo.bar.com",
							IngressRuleValue: v1.IngressRuleValue{
								HTTP: &v1.HTTPIngressRuleValue{
									Paths: []v1.HTTPIngressPath{
										{
											Path: "/api",
											Backend: v1.IngressBackend{
												Service: &v1.IngressServiceBackend{
													Name: "fake-service",
													Port: v1.ServiceBackendPort{
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
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			o := &CreateIngressOptions{
				Name:        ingressName,
				Host:        tc.host,
				ServiceName: tc.serviceName,
				ServicePort: tc.servicePort,
				Path:        tc.path,
			}
			ingress := o.createIngress()
			if !apiequality.Semantic.DeepEqual(ingress, tc.expect) {
				t.Errorf("expected:\n%+v\ngot:\n%+v", tc.expect, ingress)
			}
		})
	}

}

func TestCreateIngressValidation(t *testing.T) {
	tests := map[string]struct {
		name        string
		host        string
		serviceName string
		servicePort string
		path        string
		expect      string
	}{
		"test-missing-host": {
			serviceName: "fake-ingress",
			expect:      "--host must be specified",
		},
		"test-missing-service": {
			host:   "foo.bar.com",
			expect: "--service-name must be specified",
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			o := &CreateIngressOptions{
				Host:        tc.host,
				ServiceName: tc.serviceName,
				ServicePort: tc.servicePort,
				Path:        tc.path,
			}

			err := o.Validate()
			if err != nil && !strings.Contains(err.Error(), tc.expect) {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}
