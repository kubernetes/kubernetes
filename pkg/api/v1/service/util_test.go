/*
Copyright 2016 The Kubernetes Authors.

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

package service

import (
	"testing"

	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/v1"
	netsets "k8s.io/kubernetes/pkg/util/net/sets"
)

func TestGetLoadBalancerSourceRanges(t *testing.T) {
	checkError := func(v string) {
		annotations := make(map[string]string)
		annotations[AnnotationLoadBalancerSourceRangesKey] = v
		svc := v1.Service{}
		svc.Annotations = annotations
		_, err := GetLoadBalancerSourceRanges(&svc)
		if err == nil {
			t.Errorf("Expected error parsing: %q", v)
		}
		svc = v1.Service{}
		svc.Spec.LoadBalancerSourceRanges = strings.Split(v, ",")
		_, err = GetLoadBalancerSourceRanges(&svc)
		if err == nil {
			t.Errorf("Expected error parsing: %q", v)
		}
	}
	checkError("10.0.0.1/33")
	checkError("foo.bar")
	checkError("10.0.0.1/32,*")
	checkError("10.0.0.1/32,")
	checkError("10.0.0.1/32, ")
	checkError("10.0.0.1")

	checkOK := func(v string) netsets.IPNet {
		annotations := make(map[string]string)
		annotations[AnnotationLoadBalancerSourceRangesKey] = v
		svc := v1.Service{}
		svc.Annotations = annotations
		cidrs, err := GetLoadBalancerSourceRanges(&svc)
		if err != nil {
			t.Errorf("Unexpected error parsing: %q", v)
		}
		svc = v1.Service{}
		svc.Spec.LoadBalancerSourceRanges = strings.Split(v, ",")
		cidrs, err = GetLoadBalancerSourceRanges(&svc)
		if err != nil {
			t.Errorf("Unexpected error parsing: %q", v)
		}
		return cidrs
	}
	cidrs := checkOK("192.168.0.1/32")
	if len(cidrs) != 1 {
		t.Errorf("Expected exactly one CIDR: %v", cidrs.StringSlice())
	}
	cidrs = checkOK("192.168.0.1/32,192.168.0.1/32")
	if len(cidrs) != 1 {
		t.Errorf("Expected exactly one CIDR (after de-dup): %v", cidrs.StringSlice())
	}
	cidrs = checkOK("192.168.0.1/32,192.168.0.2/32")
	if len(cidrs) != 2 {
		t.Errorf("Expected two CIDRs: %v", cidrs.StringSlice())
	}
	cidrs = checkOK("  192.168.0.1/32 , 192.168.0.2/32   ")
	if len(cidrs) != 2 {
		t.Errorf("Expected two CIDRs: %v", cidrs.StringSlice())
	}
	// check LoadBalancerSourceRanges not specified
	svc := v1.Service{}
	cidrs, err := GetLoadBalancerSourceRanges(&svc)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(cidrs) != 1 {
		t.Errorf("Expected exactly one CIDR: %v", cidrs.StringSlice())
	}
	if !IsAllowAll(cidrs) {
		t.Errorf("Expected default to be allow-all: %v", cidrs.StringSlice())
	}
	// check SourceRanges annotation is empty
	annotations := make(map[string]string)
	annotations[AnnotationLoadBalancerSourceRangesKey] = ""
	svc = v1.Service{}
	svc.Annotations = annotations
	cidrs, err = GetLoadBalancerSourceRanges(&svc)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(cidrs) != 1 {
		t.Errorf("Expected exactly one CIDR: %v", cidrs.StringSlice())
	}
	if !IsAllowAll(cidrs) {
		t.Errorf("Expected default to be allow-all: %v", cidrs.StringSlice())
	}
}

func TestRequestsOnlyLocalTraffic(t *testing.T) {
	checkRequestsOnlyLocalTraffic := func(requestsOnlyLocalTraffic bool, service *v1.Service) {
		res := RequestsOnlyLocalTraffic(service)
		if res != requestsOnlyLocalTraffic {
			t.Errorf("Expected requests OnlyLocal traffic = %v, got %v",
				requestsOnlyLocalTraffic, res)
		}
	}

	checkRequestsOnlyLocalTraffic(false, &v1.Service{})
	checkRequestsOnlyLocalTraffic(false, &v1.Service{
		Spec: v1.ServiceSpec{
			Type: v1.ServiceTypeClusterIP,
		},
	})
	checkRequestsOnlyLocalTraffic(false, &v1.Service{
		Spec: v1.ServiceSpec{
			Type: v1.ServiceTypeNodePort,
		},
	})
	checkRequestsOnlyLocalTraffic(false, &v1.Service{
		Spec: v1.ServiceSpec{
			Type:            v1.ServiceTypeNodePort,
			ExternalTraffic: v1.ServiceExternalTrafficTypeGlobal,
		},
	})
	checkRequestsOnlyLocalTraffic(true, &v1.Service{
		Spec: v1.ServiceSpec{
			Type:            v1.ServiceTypeNodePort,
			ExternalTraffic: v1.ServiceExternalTrafficTypeOnlyLocal,
		},
	})
	checkRequestsOnlyLocalTraffic(false, &v1.Service{
		Spec: v1.ServiceSpec{
			Type:            v1.ServiceTypeLoadBalancer,
			ExternalTraffic: v1.ServiceExternalTrafficTypeGlobal,
		},
	})
	checkRequestsOnlyLocalTraffic(true, &v1.Service{
		Spec: v1.ServiceSpec{
			Type:            v1.ServiceTypeLoadBalancer,
			ExternalTraffic: v1.ServiceExternalTrafficTypeOnlyLocal,
		},
	})
}

func TestAllowAll(t *testing.T) {
	checkAllowAll := func(allowAll bool, cidrs ...string) {
		ipnets, err := netsets.ParseIPNets(cidrs...)
		if err != nil {
			t.Errorf("Unexpected error parsing cidrs: %v", cidrs)
		}
		if allowAll != IsAllowAll(ipnets) {
			t.Errorf("IsAllowAll did not return expected value for %v", cidrs)
		}
	}
	checkAllowAll(false, "10.0.0.1/32")
	checkAllowAll(false, "10.0.0.1/32", "10.0.0.2/32")
	checkAllowAll(false, "10.0.0.1/32", "10.0.0.1/32")

	checkAllowAll(true, "0.0.0.0/0")
	checkAllowAll(true, "192.168.0.0/0")
	checkAllowAll(true, "192.168.0.1/32", "0.0.0.0/0")
}

func TestNeedsHealthCheck(t *testing.T) {
	checkNeedsHealthCheck := func(needsHealthCheck bool, service *v1.Service) {
		res := NeedsHealthCheck(service)
		if res != needsHealthCheck {
			t.Errorf("Expected needs health check = %v, got %v",
				needsHealthCheck, res)
		}
	}

	checkNeedsHealthCheck(false, &v1.Service{
		Spec: v1.ServiceSpec{
			Type: v1.ServiceTypeClusterIP,
		},
	})
	checkNeedsHealthCheck(false, &v1.Service{
		Spec: v1.ServiceSpec{
			Type:            v1.ServiceTypeNodePort,
			ExternalTraffic: v1.ServiceExternalTrafficTypeGlobal,
		},
	})
	checkNeedsHealthCheck(false, &v1.Service{
		Spec: v1.ServiceSpec{
			Type:            v1.ServiceTypeNodePort,
			ExternalTraffic: v1.ServiceExternalTrafficTypeOnlyLocal,
		},
	})
	checkNeedsHealthCheck(false, &v1.Service{
		Spec: v1.ServiceSpec{
			Type:            v1.ServiceTypeLoadBalancer,
			ExternalTraffic: v1.ServiceExternalTrafficTypeGlobal,
		},
	})
	checkNeedsHealthCheck(true, &v1.Service{
		Spec: v1.ServiceSpec{
			Type:            v1.ServiceTypeLoadBalancer,
			ExternalTraffic: v1.ServiceExternalTrafficTypeOnlyLocal,
		},
	})

	checkNeedsHealthCheck(false, &v1.Service{
		Spec: v1.ServiceSpec{
			Type: v1.ServiceTypeLoadBalancer,
		},
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				BetaAnnotationExternalTraffic: "invalid",
			},
		},
	})
	checkNeedsHealthCheck(false, &v1.Service{
		Spec: v1.ServiceSpec{
			Type: v1.ServiceTypeLoadBalancer,
		},
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				BetaAnnotationExternalTraffic: string(v1.ServiceExternalTrafficTypeGlobal),
			},
		},
	})
	checkNeedsHealthCheck(true, &v1.Service{
		Spec: v1.ServiceSpec{
			Type: v1.ServiceTypeLoadBalancer,
		},
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				BetaAnnotationExternalTraffic: string(v1.ServiceExternalTrafficTypeOnlyLocal),
			},
		},
	})
}

func TestGetServiceHealthCheckNodePort(t *testing.T) {
	checkGetServiceHealthCheckNodePort := func(healthCheckNodePort int32, service *v1.Service) {
		res := GetServiceHealthCheckNodePort(service)
		if res != healthCheckNodePort {
			t.Errorf("Expected health check node port = %v, got %v",
				healthCheckNodePort, res)
		}
	}

	checkGetServiceHealthCheckNodePort(0, &v1.Service{
		Spec: v1.ServiceSpec{
			Type: v1.ServiceTypeClusterIP,
		},
	})
	checkGetServiceHealthCheckNodePort(0, &v1.Service{
		Spec: v1.ServiceSpec{
			Type:            v1.ServiceTypeNodePort,
			ExternalTraffic: v1.ServiceExternalTrafficTypeGlobal,
		},
	})
	checkGetServiceHealthCheckNodePort(0, &v1.Service{
		Spec: v1.ServiceSpec{
			Type:                v1.ServiceTypeNodePort,
			ExternalTraffic:     v1.ServiceExternalTrafficTypeOnlyLocal,
			HealthCheckNodePort: int32(34567),
		},
	})
	checkGetServiceHealthCheckNodePort(0, &v1.Service{
		Spec: v1.ServiceSpec{
			Type:            v1.ServiceTypeLoadBalancer,
			ExternalTraffic: v1.ServiceExternalTrafficTypeGlobal,
		},
	})
	checkGetServiceHealthCheckNodePort(0, &v1.Service{
		Spec: v1.ServiceSpec{
			Type:                v1.ServiceTypeLoadBalancer,
			ExternalTraffic:     v1.ServiceExternalTrafficTypeGlobal,
			HealthCheckNodePort: int32(34567),
		},
	})
	checkGetServiceHealthCheckNodePort(34567, &v1.Service{
		Spec: v1.ServiceSpec{
			Type:                v1.ServiceTypeLoadBalancer,
			ExternalTraffic:     v1.ServiceExternalTrafficTypeOnlyLocal,
			HealthCheckNodePort: int32(34567),
		},
	})

	checkGetServiceHealthCheckNodePort(0, &v1.Service{
		Spec: v1.ServiceSpec{
			Type: v1.ServiceTypeLoadBalancer,
		},
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				BetaAnnotationExternalTraffic:     "invalid",
				BetaAnnotationHealthCheckNodePort: "34567",
			},
		},
	})
	checkGetServiceHealthCheckNodePort(0, &v1.Service{
		Spec: v1.ServiceSpec{
			Type: v1.ServiceTypeLoadBalancer,
		},
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				BetaAnnotationExternalTraffic:     string(v1.ServiceExternalTrafficTypeGlobal),
				BetaAnnotationHealthCheckNodePort: "34567",
			},
		},
	})
	checkGetServiceHealthCheckNodePort(34567, &v1.Service{
		Spec: v1.ServiceSpec{
			Type: v1.ServiceTypeLoadBalancer,
		},
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				BetaAnnotationExternalTraffic:     string(v1.ServiceExternalTrafficTypeOnlyLocal),
				BetaAnnotationHealthCheckNodePort: "34567",
			},
		},
	})
}

func TestGetServiceHealthCheckPathPort(t *testing.T) {
	checkGetServiceHealthCheckPathPort := func(path string, healthCheckNodePort int32, service *v1.Service) {
		resPath, resPort := GetServiceHealthCheckPathPort(service)
		if resPath != path || resPort != healthCheckNodePort {
			t.Errorf("Expected path %v and port %v, got path %v and port %v",
				path, healthCheckNodePort, resPath, resPort)
		}
	}

	checkGetServiceHealthCheckPathPort("", 0, &v1.Service{
		Spec: v1.ServiceSpec{
			Type: v1.ServiceTypeClusterIP,
		},
	})
	checkGetServiceHealthCheckPathPort("", 0, &v1.Service{
		Spec: v1.ServiceSpec{
			Type:            v1.ServiceTypeNodePort,
			ExternalTraffic: v1.ServiceExternalTrafficTypeGlobal,
		},
	})
	checkGetServiceHealthCheckPathPort("", 0, &v1.Service{
		Spec: v1.ServiceSpec{
			Type:                v1.ServiceTypeNodePort,
			ExternalTraffic:     v1.ServiceExternalTrafficTypeOnlyLocal,
			HealthCheckNodePort: int32(34567),
		},
	})
	checkGetServiceHealthCheckPathPort("", 0, &v1.Service{
		Spec: v1.ServiceSpec{
			Type:            v1.ServiceTypeLoadBalancer,
			ExternalTraffic: v1.ServiceExternalTrafficTypeGlobal,
		},
	})
	checkGetServiceHealthCheckPathPort("/healthz", 34567, &v1.Service{
		Spec: v1.ServiceSpec{
			Type:                v1.ServiceTypeLoadBalancer,
			ExternalTraffic:     v1.ServiceExternalTrafficTypeOnlyLocal,
			HealthCheckNodePort: int32(34567),
		},
	})

	checkGetServiceHealthCheckPathPort("", 0, &v1.Service{
		Spec: v1.ServiceSpec{
			Type: v1.ServiceTypeLoadBalancer,
		},
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				BetaAnnotationExternalTraffic:     "invalid",
				BetaAnnotationHealthCheckNodePort: "34567",
			},
		},
	})
	checkGetServiceHealthCheckPathPort("", 0, &v1.Service{
		Spec: v1.ServiceSpec{
			Type: v1.ServiceTypeLoadBalancer,
		},
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				BetaAnnotationExternalTraffic:     string(v1.ServiceExternalTrafficTypeGlobal),
				BetaAnnotationHealthCheckNodePort: "34567",
			},
		},
	})
	checkGetServiceHealthCheckPathPort("/healthz", 34567, &v1.Service{
		Spec: v1.ServiceSpec{
			Type: v1.ServiceTypeLoadBalancer,
		},
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				BetaAnnotationExternalTraffic:     string(v1.ServiceExternalTrafficTypeOnlyLocal),
				BetaAnnotationHealthCheckNodePort: "34567",
			},
		},
	})
}
