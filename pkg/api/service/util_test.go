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

	api "k8s.io/kubernetes/pkg/apis/core"
)

func TestExternallyAccessible(t *testing.T) {
	checkExternallyAccessible := func(expect bool, service *api.Service) {
		t.Helper()
		res := ExternallyAccessible(service)
		if res != expect {
			t.Errorf("Expected ExternallyAccessible = %v, got %v", expect, res)
		}
	}

	checkExternallyAccessible(false, &api.Service{})
	checkExternallyAccessible(false, &api.Service{
		Spec: api.ServiceSpec{
			Type: api.ServiceTypeClusterIP,
		},
	})
	checkExternallyAccessible(true, &api.Service{
		Spec: api.ServiceSpec{
			Type:        api.ServiceTypeClusterIP,
			ExternalIPs: []string{"1.2.3.4"},
		},
	})
	checkExternallyAccessible(true, &api.Service{
		Spec: api.ServiceSpec{
			Type: api.ServiceTypeLoadBalancer,
		},
	})
	checkExternallyAccessible(true, &api.Service{
		Spec: api.ServiceSpec{
			Type: api.ServiceTypeNodePort,
		},
	})
	checkExternallyAccessible(false, &api.Service{
		Spec: api.ServiceSpec{
			Type: api.ServiceTypeExternalName,
		},
	})
	checkExternallyAccessible(false, &api.Service{
		Spec: api.ServiceSpec{
			Type:        api.ServiceTypeExternalName,
			ExternalIPs: []string{"1.2.3.4"},
		},
	})
}

func TestRequestsOnlyLocalTraffic(t *testing.T) {
	checkRequestsOnlyLocalTraffic := func(requestsOnlyLocalTraffic bool, service *api.Service) {
		t.Helper()
		res := RequestsOnlyLocalTraffic(service)
		if res != requestsOnlyLocalTraffic {
			t.Errorf("Expected requests OnlyLocal traffic = %v, got %v",
				requestsOnlyLocalTraffic, res)
		}
	}

	checkRequestsOnlyLocalTraffic(false, &api.Service{})
	checkRequestsOnlyLocalTraffic(false, &api.Service{
		Spec: api.ServiceSpec{
			Type: api.ServiceTypeClusterIP,
		},
	})
	checkRequestsOnlyLocalTraffic(false, &api.Service{
		Spec: api.ServiceSpec{
			Type: api.ServiceTypeNodePort,
		},
	})
	checkRequestsOnlyLocalTraffic(false, &api.Service{
		Spec: api.ServiceSpec{
			Type:                  api.ServiceTypeNodePort,
			ExternalTrafficPolicy: api.ServiceExternalTrafficPolicyCluster,
		},
	})
	checkRequestsOnlyLocalTraffic(true, &api.Service{
		Spec: api.ServiceSpec{
			Type:                  api.ServiceTypeNodePort,
			ExternalTrafficPolicy: api.ServiceExternalTrafficPolicyLocal,
		},
	})
	checkRequestsOnlyLocalTraffic(false, &api.Service{
		Spec: api.ServiceSpec{
			Type:                  api.ServiceTypeLoadBalancer,
			ExternalTrafficPolicy: api.ServiceExternalTrafficPolicyCluster,
		},
	})
	checkRequestsOnlyLocalTraffic(true, &api.Service{
		Spec: api.ServiceSpec{
			Type:                  api.ServiceTypeLoadBalancer,
			ExternalTrafficPolicy: api.ServiceExternalTrafficPolicyLocal,
		},
	})
}

func TestNeedsHealthCheck(t *testing.T) {
	checkNeedsHealthCheck := func(needsHealthCheck bool, service *api.Service) {
		t.Helper()
		res := NeedsHealthCheck(service)
		if res != needsHealthCheck {
			t.Errorf("Expected needs health check = %v, got %v",
				needsHealthCheck, res)
		}
	}

	checkNeedsHealthCheck(false, &api.Service{
		Spec: api.ServiceSpec{
			Type: api.ServiceTypeClusterIP,
		},
	})
	checkNeedsHealthCheck(false, &api.Service{
		Spec: api.ServiceSpec{
			Type:                  api.ServiceTypeNodePort,
			ExternalTrafficPolicy: api.ServiceExternalTrafficPolicyCluster,
		},
	})
	checkNeedsHealthCheck(false, &api.Service{
		Spec: api.ServiceSpec{
			Type:                  api.ServiceTypeNodePort,
			ExternalTrafficPolicy: api.ServiceExternalTrafficPolicyLocal,
		},
	})
	checkNeedsHealthCheck(false, &api.Service{
		Spec: api.ServiceSpec{
			Type:                  api.ServiceTypeLoadBalancer,
			ExternalTrafficPolicy: api.ServiceExternalTrafficPolicyCluster,
		},
	})
	checkNeedsHealthCheck(true, &api.Service{
		Spec: api.ServiceSpec{
			Type:                  api.ServiceTypeLoadBalancer,
			ExternalTrafficPolicy: api.ServiceExternalTrafficPolicyLocal,
		},
	})
}
