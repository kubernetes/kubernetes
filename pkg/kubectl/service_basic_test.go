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

package kubectl

import (
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/intstr"
)

func TestServiceBasicGenerate(t *testing.T) {
	tests := []struct {
		name        string
		serviceType api.ServiceType
		tcp         []string
		clusterip   string
		expected    *api.Service
		expectErr   bool
	}{
		{
			name:        "clusterip-ok",
			tcp:         []string{"456", "321:908"},
			clusterip:   "",
			serviceType: api.ServiceTypeClusterIP,
			expected: &api.Service{
				ObjectMeta: api.ObjectMeta{
					Name:   "clusterip-ok",
					Labels: map[string]string{"app": "clusterip-ok"},
				},
				Spec: api.ServiceSpec{Type: "ClusterIP",
					Ports: []api.ServicePort{{Name: "456", Protocol: "TCP", Port: 456, TargetPort: intstr.IntOrString{Type: 0, IntVal: 456, StrVal: ""}, NodePort: 0},
						{Name: "321-908", Protocol: "TCP", Port: 321, TargetPort: intstr.IntOrString{Type: 0, IntVal: 908, StrVal: ""}, NodePort: 0}},
					Selector:  map[string]string{"app": "clusterip-ok"},
					ClusterIP: "", ExternalIPs: []string(nil), LoadBalancerIP: ""},
			},
			expectErr: false,
		},
		{
			name:        "clusterip-missing",
			serviceType: api.ServiceTypeClusterIP,
			expectErr:   true,
		},
		{
			name:        "clusterip-none and port mapping",
			tcp:         []string{"456:9898"},
			clusterip:   "None",
			serviceType: api.ServiceTypeClusterIP,
			expectErr:   true,
		},
		{
			name:        "clusterip-none-wrong-type",
			tcp:         []string{},
			clusterip:   "None",
			serviceType: api.ServiceTypeNodePort,
			expectErr:   true,
		},
		{
			name:        "clusterip-none-ok",
			tcp:         []string{},
			clusterip:   "None",
			serviceType: api.ServiceTypeClusterIP,
			expected: &api.Service{
				ObjectMeta: api.ObjectMeta{
					Name:   "clusterip-none-ok",
					Labels: map[string]string{"app": "clusterip-none-ok"},
				},
				Spec: api.ServiceSpec{Type: "ClusterIP",
					Ports:     []api.ServicePort{},
					Selector:  map[string]string{"app": "clusterip-none-ok"},
					ClusterIP: "None", ExternalIPs: []string(nil), LoadBalancerIP: ""},
			},
			expectErr: false,
		},
		{
			name:        "loadbalancer-ok",
			tcp:         []string{"456:9898"},
			clusterip:   "",
			serviceType: api.ServiceTypeLoadBalancer,
			expected: &api.Service{
				ObjectMeta: api.ObjectMeta{
					Name:   "loadbalancer-ok",
					Labels: map[string]string{"app": "loadbalancer-ok"},
				},
				Spec: api.ServiceSpec{Type: "LoadBalancer",
					Ports:     []api.ServicePort{{Name: "456-9898", Protocol: "TCP", Port: 456, TargetPort: intstr.IntOrString{Type: 0, IntVal: 9898, StrVal: ""}, NodePort: 0}},
					Selector:  map[string]string{"app": "loadbalancer-ok"},
					ClusterIP: "", ExternalIPs: []string(nil), LoadBalancerIP: ""},
			},
			expectErr: false,
		},
		{
			expectErr: true,
		},
	}
	for _, test := range tests {
		generator := ServiceCommonGeneratorV1{
			Name:      test.name,
			TCP:       test.tcp,
			Type:      test.serviceType,
			ClusterIP: test.clusterip,
		}
		obj, err := generator.StructuredGenerate()
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if test.expectErr && err != nil {
			continue
		}
		if !reflect.DeepEqual(obj.(*api.Service), test.expected) {
			t.Errorf("test: %v\nexpected:\n%#v\nsaw:\n%#v", test.name, test.expected, obj.(*api.Service))
		}
	}
}
