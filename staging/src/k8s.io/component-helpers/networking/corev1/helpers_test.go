/*
Copyright 2021 The Kubernetes Authors.

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

package corev1

import (
	"testing"

	v1 "k8s.io/api/core/v1"
)

func TestIsServiceIPSet(t *testing.T) {
	testCases := []struct {
		input  v1.ServiceSpec
		output bool
		name   string
	}{
		{
			name: "nil cluster ip",
			input: v1.ServiceSpec{
				ClusterIPs: nil,
			},

			output: false,
		},
		{
			name: "headless service",
			input: v1.ServiceSpec{
				ClusterIP:  "None",
				ClusterIPs: []string{"None"},
			},
			output: false,
		},
		// true cases
		{
			name: "one ipv4",
			input: v1.ServiceSpec{
				ClusterIP:  "1.2.3.4",
				ClusterIPs: []string{"1.2.3.4"},
			},
			output: true,
		},
		{
			name: "one ipv6",
			input: v1.ServiceSpec{
				ClusterIP:  "2001::1",
				ClusterIPs: []string{"2001::1"},
			},
			output: true,
		},
		{
			name: "v4, v6",
			input: v1.ServiceSpec{
				ClusterIP:  "1.2.3.4",
				ClusterIPs: []string{"1.2.3.4", "2001::1"},
			},
			output: true,
		},
		{
			name: "v6, v4",
			input: v1.ServiceSpec{
				ClusterIP:  "2001::1",
				ClusterIPs: []string{"2001::1", "1.2.3.4"},
			},

			output: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			s := v1.Service{
				Spec: tc.input,
			}
			if IsServiceIPSet(&s) != tc.output {
				t.Errorf("case, input: %v, expected: %v, got: %v", tc.input, tc.output, !tc.output)
			}
		})
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
			Type:                  v1.ServiceTypeNodePort,
			ExternalTrafficPolicy: v1.ServiceExternalTrafficPolicyTypeCluster,
		},
	})
	checkRequestsOnlyLocalTraffic(true, &v1.Service{
		Spec: v1.ServiceSpec{
			Type:                  v1.ServiceTypeNodePort,
			ExternalTrafficPolicy: v1.ServiceExternalTrafficPolicyTypeLocal,
		},
	})
	checkRequestsOnlyLocalTraffic(false, &v1.Service{
		Spec: v1.ServiceSpec{
			Type:                  v1.ServiceTypeLoadBalancer,
			ExternalTrafficPolicy: v1.ServiceExternalTrafficPolicyTypeCluster,
		},
	})
	checkRequestsOnlyLocalTraffic(true, &v1.Service{
		Spec: v1.ServiceSpec{
			Type:                  v1.ServiceTypeLoadBalancer,
			ExternalTrafficPolicy: v1.ServiceExternalTrafficPolicyTypeLocal,
		},
	})
}

func TestRequestsOnlyLocalTrafficForInternal(t *testing.T) {
	checkRequestsOnlyLocalTrafficForInternal := func(expected bool, service *v1.Service) {
		res := RequestsOnlyLocalTrafficForInternal(service)
		if res != expected {
			t.Errorf("Expected internal local traffic = %v, got %v",
				expected, res)
		}
	}

	// default InternalTrafficPolicy is nil
	checkRequestsOnlyLocalTrafficForInternal(false, &v1.Service{})

	local := v1.ServiceInternalTrafficPolicyLocal
	checkRequestsOnlyLocalTrafficForInternal(true, &v1.Service{
		Spec: v1.ServiceSpec{
			InternalTrafficPolicy: &local,
		},
	})

	cluster := v1.ServiceInternalTrafficPolicyCluster
	checkRequestsOnlyLocalTrafficForInternal(false, &v1.Service{
		Spec: v1.ServiceSpec{
			InternalTrafficPolicy: &cluster,
		},
	})
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
			Type:                  v1.ServiceTypeNodePort,
			ExternalTrafficPolicy: v1.ServiceExternalTrafficPolicyTypeCluster,
		},
	})
	checkNeedsHealthCheck(false, &v1.Service{
		Spec: v1.ServiceSpec{
			Type:                  v1.ServiceTypeNodePort,
			ExternalTrafficPolicy: v1.ServiceExternalTrafficPolicyTypeLocal,
		},
	})
	checkNeedsHealthCheck(false, &v1.Service{
		Spec: v1.ServiceSpec{
			Type:                  v1.ServiceTypeLoadBalancer,
			ExternalTrafficPolicy: v1.ServiceExternalTrafficPolicyTypeCluster,
		},
	})
	checkNeedsHealthCheck(true, &v1.Service{
		Spec: v1.ServiceSpec{
			Type:                  v1.ServiceTypeLoadBalancer,
			ExternalTrafficPolicy: v1.ServiceExternalTrafficPolicyTypeLocal,
		},
	})
}
