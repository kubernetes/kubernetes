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

package versioned

import (
	"reflect"
	"strings"
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

func TestParsePorts(t *testing.T) {
	tests := []struct {
		portString       string
		expectPort       int32
		expectTargetPort intstr.IntOrString
		expectErr        string
	}{
		{
			portString:       "3232",
			expectPort:       3232,
			expectTargetPort: intstr.IntOrString{Type: intstr.Int, IntVal: 3232},
		},
		{
			portString:       "1:65535",
			expectPort:       1,
			expectTargetPort: intstr.IntOrString{Type: intstr.Int, IntVal: 65535},
		},
		{
			portString:       "-5:1234",
			expectPort:       0,
			expectTargetPort: intstr.IntOrString{Type: intstr.Int, IntVal: 0},
			expectErr:        "must be between 1 and 65535, inclusive",
		},
		{
			portString:       "5:65536",
			expectPort:       0,
			expectTargetPort: intstr.IntOrString{Type: intstr.Int, IntVal: 0},
			expectErr:        "must be between 1 and 65535, inclusive",
		},
		{
			portString:       "test-5:443",
			expectPort:       0,
			expectTargetPort: intstr.IntOrString{Type: intstr.Int, IntVal: 0},
			expectErr:        "invalid syntax",
		},
		{
			portString:       "5:test-443",
			expectPort:       5,
			expectTargetPort: intstr.IntOrString{Type: intstr.String, StrVal: "test-443"},
		},
		{
			portString:       "5:test*443",
			expectPort:       0,
			expectTargetPort: intstr.IntOrString{Type: intstr.Int, IntVal: 0},
			expectErr:        "must contain only alpha-numeric characters (a-z, 0-9), and hyphens (-)",
		},
		{
			portString:       "5:",
			expectPort:       0,
			expectTargetPort: intstr.IntOrString{Type: intstr.Int, IntVal: 0},
			expectErr:        "must contain at least one letter or number (a-z, 0-9)",
		},
		{
			portString:       "5:test--443",
			expectPort:       0,
			expectTargetPort: intstr.IntOrString{Type: intstr.Int, IntVal: 0},
			expectErr:        "must not contain consecutive hyphens",
		},
		{
			portString:       "5:test443-",
			expectPort:       0,
			expectTargetPort: intstr.IntOrString{Type: intstr.Int, IntVal: 0},
			expectErr:        "must not begin or end with a hyphen",
		},
		{
			portString:       "3232:1234:4567",
			expectPort:       3232,
			expectTargetPort: intstr.IntOrString{Type: intstr.Int, IntVal: 1234},
		},
	}

	for _, test := range tests {
		t.Run(test.portString, func(t *testing.T) {
			port, targetPort, err := parsePorts(test.portString)
			if len(test.expectErr) != 0 {
				if !strings.Contains(err.Error(), test.expectErr) {
					t.Errorf("parse ports string: %s. Expected err: %s, Got err: %v.", test.portString, test.expectErr, err)
				}
			}
			if !reflect.DeepEqual(targetPort, test.expectTargetPort) || port != test.expectPort {
				t.Errorf("parse ports string: %s. Expected port:%d, targetPort:%v, Got port:%d, targetPort:%v.", test.portString, test.expectPort, test.expectTargetPort, port, targetPort)
			}
		})
	}
}

func TestValidateServiceCommonGeneratorV1(t *testing.T) {
	tests := []struct {
		name      string
		s         ServiceCommonGeneratorV1
		expectErr string
	}{
		{
			name: "validate-ok",
			s: ServiceCommonGeneratorV1{
				Name:      "validate-ok",
				Type:      v1.ServiceTypeClusterIP,
				TCP:       []string{"123", "234:1234"},
				ClusterIP: "",
			},
		},
		{
			name: "Name-none",
			s: ServiceCommonGeneratorV1{
				Type:      v1.ServiceTypeClusterIP,
				TCP:       []string{"123", "234:1234"},
				ClusterIP: "",
			},
			expectErr: "name must be specified",
		},
		{
			name: "Type-none",
			s: ServiceCommonGeneratorV1{
				Name:      "validate-ok",
				TCP:       []string{"123", "234:1234"},
				ClusterIP: "",
			},
			expectErr: "type must be specified",
		},
		{
			name: "invalid-ClusterIPNone",
			s: ServiceCommonGeneratorV1{
				Name:      "validate-ok",
				Type:      v1.ServiceTypeNodePort,
				TCP:       []string{"123", "234:1234"},
				ClusterIP: v1.ClusterIPNone,
			},
			expectErr: "ClusterIP=None can only be used with ClusterIP service type",
		},
		{
			name: "TCP-none",
			s: ServiceCommonGeneratorV1{
				Name:      "validate-ok",
				Type:      v1.ServiceTypeClusterIP,
				ClusterIP: "",
			},
			expectErr: "at least one tcp port specifier must be provided",
		},
		{
			name: "invalid-ExternalName",
			s: ServiceCommonGeneratorV1{
				Name:         "validate-ok",
				Type:         v1.ServiceTypeExternalName,
				TCP:          []string{"123", "234:1234"},
				ClusterIP:    "",
				ExternalName: "@oi:test",
			},
			expectErr: "invalid service external name",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := test.s.validate()
			if err != nil {
				if !strings.Contains(err.Error(), test.expectErr) {
					t.Errorf("validate:%s Expected err: %s, Got err: %v", test.name, test.expectErr, err)
				}
			}
			if err == nil && len(test.expectErr) != 0 {
				t.Errorf("validate:%s Expected success, Got err: %v", test.name, err)
			}
		})
	}
}
