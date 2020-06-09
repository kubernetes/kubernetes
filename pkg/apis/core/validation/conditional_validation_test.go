/*
Copyright 2019 The Kubernetes Authors.

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
	"reflect"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/util/diff"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
)

func TestValidatePodSCTP(t *testing.T) {
	objectWithValue := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				Containers:     []api.Container{{Name: "container1", Image: "testimage", Ports: []api.ContainerPort{{ContainerPort: 80, Protocol: api.ProtocolSCTP}}}},
				InitContainers: []api.Container{{Name: "container2", Image: "testimage", Ports: []api.ContainerPort{{ContainerPort: 90, Protocol: api.ProtocolSCTP}}}},
			},
		}
	}
	objectWithoutValue := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				Containers:     []api.Container{{Name: "container1", Image: "testimage", Ports: []api.ContainerPort{{ContainerPort: 80, Protocol: api.ProtocolTCP}}}},
				InitContainers: []api.Container{{Name: "container2", Image: "testimage", Ports: []api.ContainerPort{{ContainerPort: 90, Protocol: api.ProtocolTCP}}}},
			},
		}
	}

	objectInfo := []struct {
		description string
		hasValue    bool
		object      func() *api.Pod
	}{
		{
			description: "has value",
			hasValue:    true,
			object:      objectWithValue,
		},
		{
			description: "does not have value",
			hasValue:    false,
			object:      objectWithoutValue,
		},
		{
			description: "is nil",
			hasValue:    false,
			object:      func() *api.Pod { return nil },
		},
	}

	for _, enabled := range []bool{true, false} {
		for _, oldPodInfo := range objectInfo {
			for _, newPodInfo := range objectInfo {
				oldPodHasValue, oldPod := oldPodInfo.hasValue, oldPodInfo.object()
				newPodHasValue, newPod := newPodInfo.hasValue, newPodInfo.object()
				if newPod == nil {
					continue
				}

				t.Run(fmt.Sprintf("feature enabled=%v, old object %v, new object %v", enabled, oldPodInfo.description, newPodInfo.description), func(t *testing.T) {
					defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SCTPSupport, enabled)()
					errs := ValidateConditionalPod(newPod, oldPod, nil)
					// objects should never be changed
					if !reflect.DeepEqual(oldPod, oldPodInfo.object()) {
						t.Errorf("old object changed: %v", diff.ObjectReflectDiff(oldPod, oldPodInfo.object()))
					}
					if !reflect.DeepEqual(newPod, newPodInfo.object()) {
						t.Errorf("new object changed: %v", diff.ObjectReflectDiff(newPod, newPodInfo.object()))
					}

					switch {
					case enabled || oldPodHasValue || !newPodHasValue:
						if len(errs) > 0 {
							t.Errorf("unexpected errors: %v", errs)
						}
					default:
						if len(errs) != 2 {
							t.Errorf("expected 2 errors, got %v", errs)
						}
					}
				})
			}
		}
	}
}

func TestValidateServiceSCTP(t *testing.T) {
	objectWithValue := func() *api.Service {
		return &api.Service{
			Spec: api.ServiceSpec{
				Ports: []api.ServicePort{{Protocol: api.ProtocolSCTP}},
			},
		}
	}
	objectWithoutValue := func() *api.Service {
		return &api.Service{
			Spec: api.ServiceSpec{
				Ports: []api.ServicePort{{Protocol: api.ProtocolTCP}},
			},
		}
	}

	objectInfo := []struct {
		description string
		hasValue    bool
		object      func() *api.Service
	}{
		{
			description: "has value",
			hasValue:    true,
			object:      objectWithValue,
		},
		{
			description: "does not have value",
			hasValue:    false,
			object:      objectWithoutValue,
		},
		{
			description: "is nil",
			hasValue:    false,
			object:      func() *api.Service { return nil },
		},
	}

	for _, enabled := range []bool{true, false} {
		for _, oldServiceInfo := range objectInfo {
			for _, newServiceInfo := range objectInfo {
				oldServiceHasValue, oldService := oldServiceInfo.hasValue, oldServiceInfo.object()
				newServiceHasValue, newService := newServiceInfo.hasValue, newServiceInfo.object()
				if newService == nil {
					continue
				}

				t.Run(fmt.Sprintf("feature enabled=%v, old object %v, new object %v", enabled, oldServiceInfo.description, newServiceInfo.description), func(t *testing.T) {
					defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SCTPSupport, enabled)()
					errs := ValidateConditionalService(newService, oldService, []api.IPFamily{api.IPv4Protocol})
					// objects should never be changed
					if !reflect.DeepEqual(oldService, oldServiceInfo.object()) {
						t.Errorf("old object changed: %v", diff.ObjectReflectDiff(oldService, oldServiceInfo.object()))
					}
					if !reflect.DeepEqual(newService, newServiceInfo.object()) {
						t.Errorf("new object changed: %v", diff.ObjectReflectDiff(newService, newServiceInfo.object()))
					}

					switch {
					case enabled || oldServiceHasValue || !newServiceHasValue:
						if len(errs) > 0 {
							t.Errorf("unexpected errors: %v", errs)
						}
					default:
						if len(errs) != 1 {
							t.Errorf("expected 1 error, got %v", errs)
						}
					}
				})
			}
		}
	}
}

func TestValidateEndpointsSCTP(t *testing.T) {
	objectWithValue := func() *api.Endpoints {
		return &api.Endpoints{
			Subsets: []api.EndpointSubset{
				{Ports: []api.EndpointPort{{Protocol: api.ProtocolSCTP}}},
			},
		}
	}
	objectWithoutValue := func() *api.Endpoints {
		return &api.Endpoints{
			Subsets: []api.EndpointSubset{
				{Ports: []api.EndpointPort{{Protocol: api.ProtocolTCP}}},
			},
		}
	}

	objectInfo := []struct {
		description string
		hasValue    bool
		object      func() *api.Endpoints
	}{
		{
			description: "has value",
			hasValue:    true,
			object:      objectWithValue,
		},
		{
			description: "does not have value",
			hasValue:    false,
			object:      objectWithoutValue,
		},
		{
			description: "is nil",
			hasValue:    false,
			object:      func() *api.Endpoints { return nil },
		},
	}

	for _, enabled := range []bool{true, false} {
		for _, oldEndpointsInfo := range objectInfo {
			for _, newEndpointsInfo := range objectInfo {
				oldEndpointsHasValue, oldEndpoints := oldEndpointsInfo.hasValue, oldEndpointsInfo.object()
				newEndpointsHasValue, newEndpoints := newEndpointsInfo.hasValue, newEndpointsInfo.object()
				if newEndpoints == nil {
					continue
				}

				t.Run(fmt.Sprintf("feature enabled=%v, old object %v, new object %v", enabled, oldEndpointsInfo.description, newEndpointsInfo.description), func(t *testing.T) {
					defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SCTPSupport, enabled)()
					errs := ValidateConditionalEndpoints(newEndpoints, oldEndpoints)
					// objects should never be changed
					if !reflect.DeepEqual(oldEndpoints, oldEndpointsInfo.object()) {
						t.Errorf("old object changed: %v", diff.ObjectReflectDiff(oldEndpoints, oldEndpointsInfo.object()))
					}
					if !reflect.DeepEqual(newEndpoints, newEndpointsInfo.object()) {
						t.Errorf("new object changed: %v", diff.ObjectReflectDiff(newEndpoints, newEndpointsInfo.object()))
					}

					switch {
					case enabled || oldEndpointsHasValue || !newEndpointsHasValue:
						if len(errs) > 0 {
							t.Errorf("unexpected errors: %v", errs)
						}
					default:
						if len(errs) != 1 {
							t.Errorf("expected 1 error, got %v", errs)
						}
					}
				})
			}
		}
	}
}

func TestValidateServiceIPFamily(t *testing.T) {
	ipv4 := api.IPv4Protocol
	ipv6 := api.IPv6Protocol
	var unknown api.IPFamily = "Unknown"
	testCases := []struct {
		name             string
		dualStackEnabled bool
		ipFamilies       []api.IPFamily
		svc              *api.Service
		oldSvc           *api.Service
		expectErr        []string
	}{
		{
			name:             "allowed ipv4",
			dualStackEnabled: true,
			ipFamilies:       []api.IPFamily{api.IPv4Protocol},
			svc: &api.Service{
				Spec: api.ServiceSpec{
					IPFamily: &ipv4,
				},
			},
		},
		{
			name:             "allowed ipv6",
			dualStackEnabled: true,
			ipFamilies:       []api.IPFamily{api.IPv6Protocol},
			svc: &api.Service{
				Spec: api.ServiceSpec{
					IPFamily: &ipv6,
				},
			},
		},
		{
			name:             "allowed ipv4 dual stack default IPv6",
			dualStackEnabled: true,
			ipFamilies:       []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			svc: &api.Service{
				Spec: api.ServiceSpec{
					IPFamily: &ipv4,
				},
			},
		},
		{
			name:             "allowed ipv4 dual stack default IPv4",
			dualStackEnabled: true,
			ipFamilies:       []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			svc: &api.Service{
				Spec: api.ServiceSpec{
					IPFamily: &ipv4,
				},
			},
		},
		{
			name:             "allowed ipv6 dual stack default IPv6",
			dualStackEnabled: true,
			ipFamilies:       []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			svc: &api.Service{
				Spec: api.ServiceSpec{
					IPFamily: &ipv6,
				},
			},
		},
		{
			name:             "allowed ipv6 dual stack default IPv4",
			dualStackEnabled: true,
			ipFamilies:       []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			svc: &api.Service{
				Spec: api.ServiceSpec{
					IPFamily: &ipv6,
				},
			},
		},
		{
			name:             "allow ipfamily to remain invalid if update doesn't change it",
			dualStackEnabled: true,
			ipFamilies:       []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			svc: &api.Service{
				Spec: api.ServiceSpec{
					IPFamily: &unknown,
				},
			},
			oldSvc: &api.Service{
				Spec: api.ServiceSpec{
					IPFamily: &unknown,
				},
			},
		},
		{
			name:             "not allowed ipfamily/clusterip mismatch",
			dualStackEnabled: true,
			ipFamilies:       []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			svc: &api.Service{
				Spec: api.ServiceSpec{
					IPFamily:  &ipv4,
					ClusterIP: "ffd0::1",
				},
			},
			expectErr: []string{"spec.ipFamily: Invalid value: \"IPv4\": does not match IPv6 cluster IP"},
		},
		{
			name:             "not allowed unknown family",
			dualStackEnabled: true,
			ipFamilies:       []api.IPFamily{api.IPv4Protocol},
			svc: &api.Service{
				Spec: api.ServiceSpec{
					IPFamily: &unknown,
				},
			},
			expectErr: []string{"spec.ipFamily: Invalid value: \"Unknown\": only the following families are allowed: IPv4"},
		},
		{
			name:             "not allowed ipv4 cluster ip without family",
			dualStackEnabled: true,
			ipFamilies:       []api.IPFamily{api.IPv6Protocol},
			svc: &api.Service{
				Spec: api.ServiceSpec{
					ClusterIP: "127.0.0.1",
				},
			},
			expectErr: []string{"spec.ipFamily: Required value: programmer error, must be set or defaulted by other fields"},
		},
		{
			name:             "not allowed ipv6 cluster ip without family",
			dualStackEnabled: true,
			ipFamilies:       []api.IPFamily{api.IPv4Protocol},
			svc: &api.Service{
				Spec: api.ServiceSpec{
					ClusterIP: "ffd0::1",
				},
			},
			expectErr: []string{"spec.ipFamily: Required value: programmer error, must be set or defaulted by other fields"},
		},

		{
			name:             "not allowed to change ipfamily for default type",
			dualStackEnabled: true,
			ipFamilies:       []api.IPFamily{api.IPv4Protocol},
			svc: &api.Service{
				Spec: api.ServiceSpec{
					IPFamily: &ipv4,
				},
			},
			oldSvc: &api.Service{
				Spec: api.ServiceSpec{
					IPFamily: &ipv6,
				},
			},
			expectErr: []string{"spec.ipFamily: Invalid value: \"IPv4\": field is immutable"},
		},
		{
			name:             "allowed to change ipfamily for external name",
			dualStackEnabled: true,
			ipFamilies:       []api.IPFamily{api.IPv4Protocol},
			svc: &api.Service{
				Spec: api.ServiceSpec{
					Type:     api.ServiceTypeExternalName,
					IPFamily: &ipv4,
				},
			},
			oldSvc: &api.Service{
				Spec: api.ServiceSpec{
					Type:     api.ServiceTypeExternalName,
					IPFamily: &ipv6,
				},
			},
		},

		{
			name:             "ipfamily allowed to be empty when dual stack is off",
			dualStackEnabled: false,
			ipFamilies:       []api.IPFamily{api.IPv4Protocol},
			svc: &api.Service{
				Spec: api.ServiceSpec{
					ClusterIP: "127.0.0.1",
				},
			},
		},
		{
			name:             "ipfamily must be empty when dual stack is off",
			dualStackEnabled: false,
			ipFamilies:       []api.IPFamily{api.IPv4Protocol},
			svc: &api.Service{
				Spec: api.ServiceSpec{
					IPFamily:  &ipv4,
					ClusterIP: "127.0.0.1",
				},
			},
			expectErr: []string{"spec.ipFamily: Forbidden: programmer error, must be cleared when the dual-stack feature gate is off"},
		},
		{
			name:             "ipfamily allowed to be cleared when dual stack is off",
			dualStackEnabled: false,
			ipFamilies:       []api.IPFamily{api.IPv4Protocol},
			svc: &api.Service{
				Spec: api.ServiceSpec{
					Type:      api.ServiceTypeClusterIP,
					ClusterIP: "127.0.0.1",
				},
			},
			oldSvc: &api.Service{
				Spec: api.ServiceSpec{
					Type:      api.ServiceTypeClusterIP,
					ClusterIP: "127.0.0.1",
					IPFamily:  &ipv4,
				},
			},
			expectErr: nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.IPv6DualStack, tc.dualStackEnabled)()
			oldSvc := tc.oldSvc.DeepCopy()
			newSvc := tc.svc.DeepCopy()
			originalNewSvc := newSvc.DeepCopy()
			errs := ValidateConditionalService(newSvc, oldSvc, tc.ipFamilies)
			// objects should never be changed
			if !reflect.DeepEqual(oldSvc, tc.oldSvc) {
				t.Errorf("old object changed: %v", diff.ObjectReflectDiff(oldSvc, tc.svc))
			}
			if !reflect.DeepEqual(newSvc, originalNewSvc) {
				t.Errorf("new object changed: %v", diff.ObjectReflectDiff(newSvc, originalNewSvc))
			}

			if len(errs) != len(tc.expectErr) {
				t.Fatalf("unexpected number of errors: %v", errs)
			}
			for i := range errs {
				if !strings.Contains(errs[i].Error(), tc.expectErr[i]) {
					t.Errorf("unexpected error %d: %v", i, errs[i])
				}
			}
		})
	}
}
