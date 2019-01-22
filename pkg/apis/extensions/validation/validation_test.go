/*
Copyright 2014 The Kubernetes Authors.

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

package validation

import (
	"fmt"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/extensions"
)

func TestValidateIngress(t *testing.T) {
	defaultBackend := extensions.IngressBackend{
		ServiceName: "default-backend",
		ServicePort: intstr.FromInt(80),
	}

	newValid := func() extensions.Ingress {
		return extensions.Ingress{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "foo",
				Namespace: metav1.NamespaceDefault,
			},
			Spec: extensions.IngressSpec{
				Backend: &extensions.IngressBackend{
					ServiceName: "default-backend",
					ServicePort: intstr.FromInt(80),
				},
				Rules: []extensions.IngressRule{
					{
						Host: "foo.bar.com",
						IngressRuleValue: extensions.IngressRuleValue{
							HTTP: &extensions.HTTPIngressRuleValue{
								Paths: []extensions.HTTPIngressPath{
									{
										Path:    "/foo",
										Backend: defaultBackend,
									},
								},
							},
						},
					},
				},
			},
			Status: extensions.IngressStatus{
				LoadBalancer: api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{
						{IP: "127.0.0.1"},
					},
				},
			},
		}
	}
	servicelessBackend := newValid()
	servicelessBackend.Spec.Backend.ServiceName = ""
	invalidNameBackend := newValid()
	invalidNameBackend.Spec.Backend.ServiceName = "defaultBackend"
	noPortBackend := newValid()
	noPortBackend.Spec.Backend = &extensions.IngressBackend{ServiceName: defaultBackend.ServiceName}
	noForwardSlashPath := newValid()
	noForwardSlashPath.Spec.Rules[0].IngressRuleValue.HTTP.Paths = []extensions.HTTPIngressPath{
		{
			Path:    "invalid",
			Backend: defaultBackend,
		},
	}
	noPaths := newValid()
	noPaths.Spec.Rules[0].IngressRuleValue.HTTP.Paths = []extensions.HTTPIngressPath{}
	badHost := newValid()
	badHost.Spec.Rules[0].Host = "foobar:80"
	badRegexPath := newValid()
	badPathExpr := "/invalid["
	badRegexPath.Spec.Rules[0].IngressRuleValue.HTTP.Paths = []extensions.HTTPIngressPath{
		{
			Path:    badPathExpr,
			Backend: defaultBackend,
		},
	}
	badPathErr := fmt.Sprintf("spec.rules[0].http.paths[0].path: Invalid value: '%v'", badPathExpr)
	hostIP := "127.0.0.1"
	badHostIP := newValid()
	badHostIP.Spec.Rules[0].Host = hostIP
	badHostIPErr := fmt.Sprintf("spec.rules[0].host: Invalid value: '%v'", hostIP)

	errorCases := map[string]extensions.Ingress{
		"spec.backend.serviceName: Required value":        servicelessBackend,
		"spec.backend.serviceName: Invalid value":         invalidNameBackend,
		"spec.backend.servicePort: Invalid value":         noPortBackend,
		"spec.rules[0].host: Invalid value":               badHost,
		"spec.rules[0].http.paths: Required value":        noPaths,
		"spec.rules[0].http.paths[0].path: Invalid value": noForwardSlashPath,
	}
	errorCases[badPathErr] = badRegexPath
	errorCases[badHostIPErr] = badHostIP

	wildcardHost := "foo.*.bar.com"
	badWildcard := newValid()
	badWildcard.Spec.Rules[0].Host = wildcardHost
	badWildcardErr := fmt.Sprintf("spec.rules[0].host: Invalid value: '%v'", wildcardHost)
	errorCases[badWildcardErr] = badWildcard

	for k, v := range errorCases {
		errs := ValidateIngress(&v)
		if len(errs) == 0 {
			t.Errorf("expected failure for %q", k)
		} else {
			s := strings.Split(k, ":")
			err := errs[0]
			if err.Field != s[0] || !strings.Contains(err.Error(), s[1]) {
				t.Errorf("unexpected error: %q, expected: %q", err, k)
			}
		}
	}
}

func TestValidateIngressTLS(t *testing.T) {
	defaultBackend := extensions.IngressBackend{
		ServiceName: "default-backend",
		ServicePort: intstr.FromInt(80),
	}

	newValid := func() extensions.Ingress {
		return extensions.Ingress{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "foo",
				Namespace: metav1.NamespaceDefault,
			},
			Spec: extensions.IngressSpec{
				Backend: &extensions.IngressBackend{
					ServiceName: "default-backend",
					ServicePort: intstr.FromInt(80),
				},
				Rules: []extensions.IngressRule{
					{
						Host: "foo.bar.com",
						IngressRuleValue: extensions.IngressRuleValue{
							HTTP: &extensions.HTTPIngressRuleValue{
								Paths: []extensions.HTTPIngressPath{
									{
										Path:    "/foo",
										Backend: defaultBackend,
									},
								},
							},
						},
					},
				},
			},
			Status: extensions.IngressStatus{
				LoadBalancer: api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{
						{IP: "127.0.0.1"},
					},
				},
			},
		}
	}

	errorCases := map[string]extensions.Ingress{}

	wildcardHost := "foo.*.bar.com"
	badWildcardTLS := newValid()
	badWildcardTLS.Spec.Rules[0].Host = "*.foo.bar.com"
	badWildcardTLS.Spec.TLS = []extensions.IngressTLS{
		{
			Hosts: []string{wildcardHost},
		},
	}
	badWildcardTLSErr := fmt.Sprintf("spec.tls[0].hosts: Invalid value: '%v'", wildcardHost)
	errorCases[badWildcardTLSErr] = badWildcardTLS

	for k, v := range errorCases {
		errs := ValidateIngress(&v)
		if len(errs) == 0 {
			t.Errorf("expected failure for %q", k)
		} else {
			s := strings.Split(k, ":")
			err := errs[0]
			if err.Field != s[0] || !strings.Contains(err.Error(), s[1]) {
				t.Errorf("unexpected error: %q, expected: %q", err, k)
			}
		}
	}
}

func TestValidateIngressStatusUpdate(t *testing.T) {
	defaultBackend := extensions.IngressBackend{
		ServiceName: "default-backend",
		ServicePort: intstr.FromInt(80),
	}

	newValid := func() extensions.Ingress {
		return extensions.Ingress{
			ObjectMeta: metav1.ObjectMeta{
				Name:            "foo",
				Namespace:       metav1.NamespaceDefault,
				ResourceVersion: "9",
			},
			Spec: extensions.IngressSpec{
				Backend: &extensions.IngressBackend{
					ServiceName: "default-backend",
					ServicePort: intstr.FromInt(80),
				},
				Rules: []extensions.IngressRule{
					{
						Host: "foo.bar.com",
						IngressRuleValue: extensions.IngressRuleValue{
							HTTP: &extensions.HTTPIngressRuleValue{
								Paths: []extensions.HTTPIngressPath{
									{
										Path:    "/foo",
										Backend: defaultBackend,
									},
								},
							},
						},
					},
				},
			},
			Status: extensions.IngressStatus{
				LoadBalancer: api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{
						{IP: "127.0.0.1", Hostname: "foo.bar.com"},
					},
				},
			},
		}
	}
	oldValue := newValid()
	newValue := newValid()
	newValue.Status = extensions.IngressStatus{
		LoadBalancer: api.LoadBalancerStatus{
			Ingress: []api.LoadBalancerIngress{
				{IP: "127.0.0.2", Hostname: "foo.com"},
			},
		},
	}
	invalidIP := newValid()
	invalidIP.Status = extensions.IngressStatus{
		LoadBalancer: api.LoadBalancerStatus{
			Ingress: []api.LoadBalancerIngress{
				{IP: "abcd", Hostname: "foo.com"},
			},
		},
	}
	invalidHostname := newValid()
	invalidHostname.Status = extensions.IngressStatus{
		LoadBalancer: api.LoadBalancerStatus{
			Ingress: []api.LoadBalancerIngress{
				{IP: "127.0.0.1", Hostname: "127.0.0.1"},
			},
		},
	}

	errs := ValidateIngressStatusUpdate(&newValue, &oldValue)
	if len(errs) != 0 {
		t.Errorf("Unexpected error %v", errs)
	}

	errorCases := map[string]extensions.Ingress{
		"status.loadBalancer.ingress[0].ip: Invalid value":       invalidIP,
		"status.loadBalancer.ingress[0].hostname: Invalid value": invalidHostname,
	}
	for k, v := range errorCases {
		errs := ValidateIngressStatusUpdate(&v, &oldValue)
		if len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		} else {
			s := strings.Split(k, ":")
			err := errs[0]
			if err.Field != s[0] || !strings.Contains(err.Error(), s[1]) {
				t.Errorf("unexpected error: %q, expected: %q", err, k)
			}
		}
	}
}
