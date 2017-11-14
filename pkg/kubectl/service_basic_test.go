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

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

func TestServiceBasicGenerate(t *testing.T) {
	tests := []struct {
		name        string
		serviceType v1.ServiceType
		tcp         []string
		clusterip   string
		expected    *v1.Service
		expectErr   bool
	}{
		{
			name:        "clusterip-ok",
			tcp:         []string{"456", "321:908"},
			clusterip:   "",
			serviceType: v1.ServiceTypeClusterIP,
			expected: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "clusterip-ok",
					Labels: map[string]string{"app": "clusterip-ok"},
				},
				Spec: v1.ServiceSpec{Type: "ClusterIP",
					Ports: []v1.ServicePort{{Name: "456", Protocol: "TCP", Port: 456, TargetPort: intstr.IntOrString{Type: 0, IntVal: 456, StrVal: ""}, NodePort: 0},
						{Name: "321-908", Protocol: "TCP", Port: 321, TargetPort: intstr.IntOrString{Type: 0, IntVal: 908, StrVal: ""}, NodePort: 0}},
					Selector:  map[string]string{"app": "clusterip-ok"},
					ClusterIP: "", ExternalIPs: []string(nil), LoadBalancerIP: ""},
			},
			expectErr: false,
		},
		{
			name:        "clusterip-missing",
			serviceType: v1.ServiceTypeClusterIP,
			expectErr:   true,
		},
		{
			name:        "clusterip-none-wrong-type",
			tcp:         []string{},
			clusterip:   "None",
			serviceType: v1.ServiceTypeNodePort,
			expectErr:   true,
		},
		{
			name:        "clusterip-none-ok",
			tcp:         []string{},
			clusterip:   "None",
			serviceType: v1.ServiceTypeClusterIP,
			expected: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "clusterip-none-ok",
					Labels: map[string]string{"app": "clusterip-none-ok"},
				},
				Spec: v1.ServiceSpec{Type: "ClusterIP",
					Ports:     []v1.ServicePort{},
					Selector:  map[string]string{"app": "clusterip-none-ok"},
					ClusterIP: "None", ExternalIPs: []string(nil), LoadBalancerIP: ""},
			},
			expectErr: false,
		},
		{
			name:        "clusterip-none-and-port-mapping",
			tcp:         []string{"456:9898"},
			clusterip:   "None",
			serviceType: v1.ServiceTypeClusterIP,
			expected: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "clusterip-none-and-port-mapping",
					Labels: map[string]string{"app": "clusterip-none-and-port-mapping"},
				},
				Spec: v1.ServiceSpec{Type: "ClusterIP",
					Ports:     []v1.ServicePort{{Name: "456-9898", Protocol: "TCP", Port: 456, TargetPort: intstr.IntOrString{Type: 0, IntVal: 9898, StrVal: ""}, NodePort: 0}},
					Selector:  map[string]string{"app": "clusterip-none-and-port-mapping"},
					ClusterIP: "None", ExternalIPs: []string(nil), LoadBalancerIP: ""},
			},
			expectErr: false,
		},
		{
			name:        "loadbalancer-ok",
			tcp:         []string{"456:9898"},
			clusterip:   "",
			serviceType: v1.ServiceTypeLoadBalancer,
			expected: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "loadbalancer-ok",
					Labels: map[string]string{"app": "loadbalancer-ok"},
				},
				Spec: v1.ServiceSpec{Type: "LoadBalancer",
					Ports:     []v1.ServicePort{{Name: "456-9898", Protocol: "TCP", Port: 456, TargetPort: intstr.IntOrString{Type: 0, IntVal: 9898, StrVal: ""}, NodePort: 0}},
					Selector:  map[string]string{"app": "loadbalancer-ok"},
					ClusterIP: "", ExternalIPs: []string(nil), LoadBalancerIP: ""},
			},
			expectErr: false,
		},
		{
			name:        "invalid-port",
			tcp:         []string{"65536"},
			clusterip:   "None",
			serviceType: v1.ServiceTypeClusterIP,
			expectErr:   true,
		},
		{
			name:        "invalid-port-mapping",
			tcp:         []string{"8080:-abc"},
			clusterip:   "None",
			serviceType: v1.ServiceTypeClusterIP,
			expectErr:   true,
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
		if !reflect.DeepEqual(obj.(*v1.Service), test.expected) {
			t.Errorf("test: %v\nexpected:\n%#v\nsaw:\n%#v", test.name, test.expected, obj.(*v1.Service))
		}
	}
}
