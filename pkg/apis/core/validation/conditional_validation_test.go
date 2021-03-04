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
	"strings"
	"testing"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
)

func TestValidateMixedProtocolLBService(t *testing.T) {
	newLBServiceDifferentProtocols := &api.Service{
		Spec: api.ServiceSpec{
			Ports: []api.ServicePort{
				{
					Protocol: api.ProtocolTCP,
				},
				{
					Protocol: api.ProtocolUDP,
				},
			},
			Type: api.ServiceTypeLoadBalancer,
		},
	}
	newLBServiceSameProtocols := &api.Service{
		Spec: api.ServiceSpec{
			Ports: []api.ServicePort{
				{
					Protocol: api.ProtocolTCP,
				},
				{
					Protocol: api.ProtocolTCP,
				},
			},
			Type: api.ServiceTypeLoadBalancer,
		},
	}
	newNonLBServiceDifferentProtocols := &api.Service{
		Spec: api.ServiceSpec{
			Ports: []api.ServicePort{
				{
					Protocol: api.ProtocolTCP,
				},
				{
					Protocol: api.ProtocolUDP,
				},
			},
			Type: api.ServiceTypeNodePort,
		},
	}
	newNonLBServiceSameProtocols := &api.Service{
		Spec: api.ServiceSpec{
			Ports: []api.ServicePort{
				{
					Protocol: api.ProtocolUDP,
				},
				{
					Protocol: api.ProtocolUDP,
				},
			},
			Type: api.ServiceTypeNodePort,
		},
	}
	oldLBServiceDifferentProtocols := &api.Service{
		Spec: api.ServiceSpec{
			Ports: []api.ServicePort{
				{
					Protocol: api.ProtocolTCP,
				},
				{
					Protocol: api.ProtocolUDP,
				},
			},
			Type: api.ServiceTypeLoadBalancer,
		},
	}
	oldLBServiceSameProtocols := &api.Service{
		Spec: api.ServiceSpec{
			Ports: []api.ServicePort{
				{
					Protocol: api.ProtocolTCP,
				},
				{
					Protocol: api.ProtocolTCP,
				},
			},
			Type: api.ServiceTypeLoadBalancer,
		},
	}
	oldNonLBServiceDifferentProtocols := &api.Service{
		Spec: api.ServiceSpec{
			Ports: []api.ServicePort{
				{
					Protocol: api.ProtocolTCP,
				},
				{
					Protocol: api.ProtocolUDP,
				},
			},
			Type: api.ServiceTypeNodePort,
		},
	}
	oldNonLBServiceSameProtocols := &api.Service{
		Spec: api.ServiceSpec{
			Ports: []api.ServicePort{
				{
					Protocol: api.ProtocolUDP,
				},
				{
					Protocol: api.ProtocolUDP,
				},
			},
			Type: api.ServiceTypeNodePort,
		},
	}
	cases := map[string]struct {
		oldService    *api.Service
		newService    *api.Service
		fgEnabled     bool
		expectedError []string
	}{
		"Old service is nil, new service has different protocols, feature gate false": {
			oldService:    nil,
			newService:    newLBServiceDifferentProtocols,
			fgEnabled:     false,
			expectedError: []string{`spec.ports: Invalid value: []core.ServicePort{core.ServicePort{Name:"", Protocol:"TCP", AppProtocol:(*string)(nil), Port:0, TargetPort:intstr.IntOrString{Type:0, IntVal:0, StrVal:""}, NodePort:0}, core.ServicePort{Name:"", Protocol:"UDP", AppProtocol:(*string)(nil), Port:0, TargetPort:intstr.IntOrString{Type:0, IntVal:0, StrVal:""}, NodePort:0}}: may not contain more than 1 protocol when type is 'LoadBalancer'`},
		},
		"Old service is nil, new service has different protocols, feature gate true": {
			oldService: nil,
			newService: newLBServiceDifferentProtocols,
			fgEnabled:  true,
		},
		"Old service is nil, new service does not have different protocols, feature gate false": {
			oldService: nil,
			newService: newLBServiceSameProtocols,
			fgEnabled:  false,
		},
		"Old service is nil, new service does not have different protocols, feature gate true": {
			oldService: nil,
			newService: newLBServiceSameProtocols,
			fgEnabled:  true,
		},
		"Old service is nil, new non-LB service has different protocols, feature gate false": {
			oldService: nil,
			newService: newNonLBServiceDifferentProtocols,
			fgEnabled:  false,
		},
		"Old service is nil, new non-LB service has different protocols, feature gate true": {
			oldService: nil,
			newService: newNonLBServiceDifferentProtocols,
			fgEnabled:  true,
		},
		"Old service is nil, new non-LB service does not have different protocols, feature gate false": {
			oldService: nil,
			newService: newNonLBServiceSameProtocols,
			fgEnabled:  false,
		},
		"Old service is nil, new non-LB service does not have different protocols, feature gate true": {
			oldService: nil,
			newService: newNonLBServiceSameProtocols,
			fgEnabled:  true,
		},
		"Non-LB services, both services have different protocols, feature gate false": {
			oldService: oldNonLBServiceDifferentProtocols,
			newService: newNonLBServiceDifferentProtocols,
			fgEnabled:  false,
		},
		"Non-LB services, old service has same protocols, new service has different protocols, feature gate false": {
			oldService: oldNonLBServiceSameProtocols,
			newService: newNonLBServiceDifferentProtocols,
			fgEnabled:  false,
		},
		"Non-LB services, old service has different protocols, new service has identical protocols, feature gate false": {
			oldService: oldNonLBServiceDifferentProtocols,
			newService: newNonLBServiceSameProtocols,
			fgEnabled:  false,
		},
		"Non-LB services, both services have same protocols, feature gate false": {
			oldService: oldNonLBServiceSameProtocols,
			newService: newNonLBServiceSameProtocols,
			fgEnabled:  false,
		},
		"Non-LB services, both services have different protocols, feature gate true": {
			oldService: oldNonLBServiceDifferentProtocols,
			newService: newNonLBServiceDifferentProtocols,
			fgEnabled:  true,
		},
		"Non-LB services, old service has same protocols, new service has different protocols, feature gate true": {
			oldService: oldNonLBServiceSameProtocols,
			newService: newNonLBServiceDifferentProtocols,
			fgEnabled:  true,
		},
		"Non-LB services, old service has different protocols, new service has identical protocols, feature gate true": {
			oldService: oldNonLBServiceDifferentProtocols,
			newService: newNonLBServiceSameProtocols,
			fgEnabled:  true,
		},
		"Non-LB services, both services have same protocols, feature gate true": {
			oldService: oldNonLBServiceSameProtocols,
			newService: newNonLBServiceSameProtocols,
			fgEnabled:  true,
		},
		"LB service, neither service has different protocols, feature gate false": {
			oldService: oldLBServiceSameProtocols,
			newService: newLBServiceSameProtocols,
			fgEnabled:  false,
		},
		"LB service, old service does not have different protocols, new service has different protocols, feature gate false": {
			oldService:    oldLBServiceSameProtocols,
			newService:    newLBServiceDifferentProtocols,
			fgEnabled:     false,
			expectedError: []string{`spec.ports: Invalid value: []core.ServicePort{core.ServicePort{Name:"", Protocol:"TCP", AppProtocol:(*string)(nil), Port:0, TargetPort:intstr.IntOrString{Type:0, IntVal:0, StrVal:""}, NodePort:0}, core.ServicePort{Name:"", Protocol:"UDP", AppProtocol:(*string)(nil), Port:0, TargetPort:intstr.IntOrString{Type:0, IntVal:0, StrVal:""}, NodePort:0}}: may not contain more than 1 protocol when type is 'LoadBalancer'`},
		},
		"LB service, old service has different protocols, new service does not have different protocols, feature gate false": {
			oldService: oldLBServiceDifferentProtocols,
			newService: newLBServiceSameProtocols,
			fgEnabled:  false,
		},
		"LB service, both services have different protocols, feature gate false": {
			oldService: oldLBServiceDifferentProtocols,
			newService: newLBServiceDifferentProtocols,
			fgEnabled:  false,
		},
		"LB service, neither service has different protocols, feature gate true": {
			oldService: oldLBServiceSameProtocols,
			newService: newLBServiceSameProtocols,
			fgEnabled:  true,
		},
		"LB service, old service has different protocols, new service does not have different protocols, feature gate true": {
			oldService: oldLBServiceDifferentProtocols,
			newService: newLBServiceSameProtocols,
			fgEnabled:  true,
		},
		"LB service, old service does not have different protocols, new service has different protocols, feature gate true": {
			oldService: oldLBServiceSameProtocols,
			newService: newLBServiceDifferentProtocols,
			fgEnabled:  true,
		},
		"LB service, both services have different protocols, feature gate true": {
			oldService: oldLBServiceDifferentProtocols,
			newService: newLBServiceDifferentProtocols,
			fgEnabled:  true,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MixedProtocolLBService, tc.fgEnabled)()
			errs := validateMixedProtocolLBService(tc.newService, tc.oldService)
			if len(errs) != len(tc.expectedError) {
				t.Fatalf("unexpected number of errors: %v", errs)
			}
			for i := range errs {
				if !strings.Contains(errs[i].Error(), tc.expectedError[i]) {
					t.Errorf("unexpected error %d: %v", i, errs[i])
				}
			}
		})
	}
}
