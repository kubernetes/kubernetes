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

package versioned

import (
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubectl/pkg/generate"
)

func TestGenerateService(t *testing.T) {
	tests := []struct {
		name      string
		generator generate.Generator
		params    map[string]interface{}
		expected  v1.Service
	}{
		{
			name:      "test1",
			generator: ServiceGeneratorV2{},
			params: map[string]interface{}{
				"selector":       "foo=bar,baz=blah",
				"name":           "test",
				"port":           "80",
				"protocol":       "TCP",
				"container-port": "1234",
			},
			expected: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []v1.ServicePort{
						{
							Port:       80,
							Protocol:   "TCP",
							TargetPort: intstr.FromInt32(1234),
						},
					},
				},
			},
		},
		{
			name:      "test2",
			generator: ServiceGeneratorV2{},
			params: map[string]interface{}{
				"selector":       "foo=bar,baz=blah",
				"name":           "test",
				"port":           "80",
				"protocol":       "UDP",
				"container-port": "foobar",
			},
			expected: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []v1.ServicePort{
						{
							Port:       80,
							Protocol:   "UDP",
							TargetPort: intstr.FromString("foobar"),
						},
					},
				},
			},
		},
		{
			name:      "test3",
			generator: ServiceGeneratorV2{},
			params: map[string]interface{}{
				"selector":       "foo=bar,baz=blah",
				"labels":         "key1=value1,key2=value2",
				"name":           "test",
				"port":           "80",
				"protocol":       "TCP",
				"container-port": "1234",
			},
			expected: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
					Labels: map[string]string{
						"key1": "value1",
						"key2": "value2",
					},
				},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []v1.ServicePort{
						{
							Port:       80,
							Protocol:   "TCP",
							TargetPort: intstr.FromInt32(1234),
						},
					},
				},
			},
		},
		{
			name:      "test4",
			generator: ServiceGeneratorV2{},
			params: map[string]interface{}{
				"selector":       "foo=bar,baz=blah",
				"name":           "test",
				"port":           "80",
				"protocol":       "UDP",
				"container-port": "foobar",
				"external-ip":    "1.2.3.4",
			},
			expected: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []v1.ServicePort{
						{
							Port:       80,
							Protocol:   "UDP",
							TargetPort: intstr.FromString("foobar"),
						},
					},
					ExternalIPs: []string{"1.2.3.4"},
				},
			},
		},
		{
			name:      "test5",
			generator: ServiceGeneratorV2{},
			params: map[string]interface{}{
				"selector":       "foo=bar,baz=blah",
				"name":           "test",
				"port":           "80",
				"protocol":       "UDP",
				"container-port": "foobar",
				"external-ip":    "1.2.3.4",
				"type":           "LoadBalancer",
			},
			expected: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []v1.ServicePort{
						{
							Port:       80,
							Protocol:   "UDP",
							TargetPort: intstr.FromString("foobar"),
						},
					},
					Type:        v1.ServiceTypeLoadBalancer,
					ExternalIPs: []string{"1.2.3.4"},
				},
			},
		},
		{
			name:      "test6",
			generator: ServiceGeneratorV2{},
			params: map[string]interface{}{
				"selector":       "foo=bar,baz=blah",
				"name":           "test",
				"port":           "80",
				"protocol":       "UDP",
				"container-port": "foobar",
				"type":           string(v1.ServiceTypeNodePort),
			},
			expected: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []v1.ServicePort{
						{
							Port:       80,
							Protocol:   "UDP",
							TargetPort: intstr.FromString("foobar"),
						},
					},
					Type: v1.ServiceTypeNodePort,
				},
			},
		},
		{
			name:      "test7",
			generator: ServiceGeneratorV2{},
			params: map[string]interface{}{
				"selector":                      "foo=bar,baz=blah",
				"name":                          "test",
				"port":                          "80",
				"protocol":                      "UDP",
				"container-port":                "foobar",
				"create-external-load-balancer": "true", // ignored when type is present
				"type":                          string(v1.ServiceTypeNodePort),
			},
			expected: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []v1.ServicePort{
						{
							Port:       80,
							Protocol:   "UDP",
							TargetPort: intstr.FromString("foobar"),
						},
					},
					Type: v1.ServiceTypeNodePort,
				},
			},
		},
		{
			name:      "test8",
			generator: ServiceGeneratorV1{},
			params: map[string]interface{}{
				"selector":       "foo=bar,baz=blah",
				"name":           "test",
				"port":           "80",
				"protocol":       "TCP",
				"container-port": "1234",
			},
			expected: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []v1.ServicePort{
						{
							Name:       "default",
							Port:       80,
							Protocol:   "TCP",
							TargetPort: intstr.FromInt32(1234),
						},
					},
				},
			},
		},
		{
			name:      "test9",
			generator: ServiceGeneratorV1{},
			params: map[string]interface{}{
				"selector":         "foo=bar,baz=blah",
				"name":             "test",
				"port":             "80",
				"protocol":         "TCP",
				"container-port":   "1234",
				"session-affinity": "ClientIP",
			},
			expected: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []v1.ServicePort{
						{
							Name:       "default",
							Port:       80,
							Protocol:   "TCP",
							TargetPort: intstr.FromInt32(1234),
						},
					},
					SessionAffinity: v1.ServiceAffinityClientIP,
				},
			},
		},
		{
			name:      "test10",
			generator: ServiceGeneratorV2{},
			params: map[string]interface{}{
				"selector":       "foo=bar,baz=blah",
				"name":           "test",
				"port":           "80",
				"protocol":       "TCP",
				"container-port": "1234",
				"cluster-ip":     "10.10.10.10",
			},
			expected: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []v1.ServicePort{
						{
							Port:       80,
							Protocol:   "TCP",
							TargetPort: intstr.FromInt32(1234),
						},
					},
					ClusterIP: "10.10.10.10",
				},
			},
		},
		{
			name:      "test11",
			generator: ServiceGeneratorV2{},
			params: map[string]interface{}{
				"selector":       "foo=bar,baz=blah",
				"name":           "test",
				"port":           "80",
				"protocol":       "TCP",
				"container-port": "1234",
				"cluster-ip":     "None",
			},
			expected: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []v1.ServicePort{
						{
							Port:       80,
							Protocol:   "TCP",
							TargetPort: intstr.FromInt32(1234),
						},
					},
					ClusterIP: v1.ClusterIPNone,
				},
			},
		},
		{
			name:      "test12",
			generator: ServiceGeneratorV1{},
			params: map[string]interface{}{
				"selector":       "foo=bar",
				"name":           "test",
				"ports":          "80,443",
				"protocol":       "TCP",
				"container-port": "foobar",
			},
			expected: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
					},
					Ports: []v1.ServicePort{
						{
							Name:       "port-1",
							Port:       80,
							Protocol:   v1.ProtocolTCP,
							TargetPort: intstr.FromString("foobar"),
						},
						{
							Name:       "port-2",
							Port:       443,
							Protocol:   v1.ProtocolTCP,
							TargetPort: intstr.FromString("foobar"),
						},
					},
				},
			},
		},
		{
			name:      "test13",
			generator: ServiceGeneratorV2{},
			params: map[string]interface{}{
				"selector":    "foo=bar",
				"name":        "test",
				"ports":       "80,443",
				"protocol":    "UDP",
				"target-port": "1234",
			},
			expected: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
					},
					Ports: []v1.ServicePort{
						{
							Name:       "port-1",
							Port:       80,
							Protocol:   v1.ProtocolUDP,
							TargetPort: intstr.FromInt32(1234),
						},
						{
							Name:       "port-2",
							Port:       443,
							Protocol:   v1.ProtocolUDP,
							TargetPort: intstr.FromInt32(1234),
						},
					},
				},
			},
		},
		{
			name:      "test14",
			generator: ServiceGeneratorV2{},
			params: map[string]interface{}{
				"selector": "foo=bar",
				"name":     "test",
				"ports":    "80,443",
				"protocol": "TCP",
			},
			expected: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
					},
					Ports: []v1.ServicePort{
						{
							Name:       "port-1",
							Port:       80,
							Protocol:   v1.ProtocolTCP,
							TargetPort: intstr.FromInt32(80),
						},
						{
							Name:       "port-2",
							Port:       443,
							Protocol:   v1.ProtocolTCP,
							TargetPort: intstr.FromInt32(443),
						},
					},
				},
			},
		},
		{
			name:      "test15",
			generator: ServiceGeneratorV2{},
			params: map[string]interface{}{
				"selector":  "foo=bar",
				"name":      "test",
				"ports":     "80,8080",
				"protocols": "8080/UDP",
			},
			expected: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
					},
					Ports: []v1.ServicePort{
						{
							Name:       "port-1",
							Port:       80,
							Protocol:   v1.ProtocolTCP,
							TargetPort: intstr.FromInt32(80),
						},
						{
							Name:       "port-2",
							Port:       8080,
							Protocol:   v1.ProtocolUDP,
							TargetPort: intstr.FromInt32(8080),
						},
					},
				},
			},
		},
		{
			name:      "test16",
			generator: ServiceGeneratorV2{},
			params: map[string]interface{}{
				"selector":  "foo=bar",
				"name":      "test",
				"ports":     "80,8080,8081",
				"protocols": "8080/UDP,8081/TCP",
			},
			expected: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
					},
					Ports: []v1.ServicePort{
						{
							Name:       "port-1",
							Port:       80,
							Protocol:   v1.ProtocolTCP,
							TargetPort: intstr.FromInt32(80),
						},
						{
							Name:       "port-2",
							Port:       8080,
							Protocol:   v1.ProtocolUDP,
							TargetPort: intstr.FromInt32(8080),
						},
						{
							Name:       "port-3",
							Port:       8081,
							Protocol:   v1.ProtocolTCP,
							TargetPort: intstr.FromInt32(8081),
						},
					},
				},
			},
		},
		{
			name:      "test17",
			generator: ServiceGeneratorV2{},
			params: map[string]interface{}{
				"selector":       "foo=bar,baz=blah",
				"name":           "test",
				"protocol":       "TCP",
				"container-port": "1234",
				"cluster-ip":     "None",
			},
			expected: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports:     []v1.ServicePort{},
					ClusterIP: v1.ClusterIPNone,
				},
			},
		},
		{
			name:      "test18",
			generator: ServiceGeneratorV2{},
			params: map[string]interface{}{
				"selector":   "foo=bar",
				"name":       "test",
				"cluster-ip": "None",
			},
			expected: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
					},
					Ports:     []v1.ServicePort{},
					ClusterIP: v1.ClusterIPNone,
				},
			},
		},
		{
			generator: ServiceGeneratorV2{},
			params: map[string]interface{}{
				"selector":       "foo=bar,baz=blah",
				"name":           "test",
				"port":           "80",
				"protocol":       "SCTP",
				"container-port": "1234",
			},
			expected: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []v1.ServicePort{
						{
							Port:       80,
							Protocol:   "SCTP",
							TargetPort: intstr.FromInt32(1234),
						},
					},
				},
			},
		},
		{
			generator: ServiceGeneratorV2{},
			params: map[string]interface{}{
				"selector":       "foo=bar,baz=blah",
				"labels":         "key1=value1,key2=value2",
				"name":           "test",
				"port":           "80",
				"protocol":       "SCTP",
				"container-port": "1234",
			},
			expected: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
					Labels: map[string]string{
						"key1": "value1",
						"key2": "value2",
					},
				},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []v1.ServicePort{
						{
							Port:       80,
							Protocol:   "SCTP",
							TargetPort: intstr.FromInt32(1234),
						},
					},
				},
			},
		},
		{
			generator: ServiceGeneratorV1{},
			params: map[string]interface{}{
				"selector":       "foo=bar,baz=blah",
				"name":           "test",
				"port":           "80",
				"protocol":       "SCTP",
				"container-port": "1234",
			},
			expected: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []v1.ServicePort{
						{
							Name:       "default",
							Port:       80,
							Protocol:   "SCTP",
							TargetPort: intstr.FromInt32(1234),
						},
					},
				},
			},
		},
		{
			generator: ServiceGeneratorV1{},
			params: map[string]interface{}{
				"selector":         "foo=bar,baz=blah",
				"name":             "test",
				"port":             "80",
				"protocol":         "SCTP",
				"container-port":   "1234",
				"session-affinity": "ClientIP",
			},
			expected: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []v1.ServicePort{
						{
							Name:       "default",
							Port:       80,
							Protocol:   "SCTP",
							TargetPort: intstr.FromInt32(1234),
						},
					},
					SessionAffinity: v1.ServiceAffinityClientIP,
				},
			},
		},
		{
			generator: ServiceGeneratorV2{},
			params: map[string]interface{}{
				"selector":       "foo=bar,baz=blah",
				"name":           "test",
				"port":           "80",
				"protocol":       "SCTP",
				"container-port": "1234",
				"cluster-ip":     "10.10.10.10",
			},
			expected: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []v1.ServicePort{
						{
							Port:       80,
							Protocol:   "SCTP",
							TargetPort: intstr.FromInt32(1234),
						},
					},
					ClusterIP: "10.10.10.10",
				},
			},
		},
		{
			generator: ServiceGeneratorV2{},
			params: map[string]interface{}{
				"selector":       "foo=bar,baz=blah",
				"name":           "test",
				"port":           "80",
				"protocol":       "SCTP",
				"container-port": "1234",
				"cluster-ip":     "None",
			},
			expected: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports: []v1.ServicePort{
						{
							Port:       80,
							Protocol:   "SCTP",
							TargetPort: intstr.FromInt32(1234),
						},
					},
					ClusterIP: v1.ClusterIPNone,
				},
			},
		},
		{
			generator: ServiceGeneratorV1{},
			params: map[string]interface{}{
				"selector":       "foo=bar",
				"name":           "test",
				"ports":          "80,443",
				"protocol":       "SCTP",
				"container-port": "foobar",
			},
			expected: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
					},
					Ports: []v1.ServicePort{
						{
							Name:       "port-1",
							Port:       80,
							Protocol:   v1.ProtocolSCTP,
							TargetPort: intstr.FromString("foobar"),
						},
						{
							Name:       "port-2",
							Port:       443,
							Protocol:   v1.ProtocolSCTP,
							TargetPort: intstr.FromString("foobar"),
						},
					},
				},
			},
		},
		{
			generator: ServiceGeneratorV2{},
			params: map[string]interface{}{
				"selector": "foo=bar",
				"name":     "test",
				"ports":    "80,443",
				"protocol": "SCTP",
			},
			expected: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
					},
					Ports: []v1.ServicePort{
						{
							Name:       "port-1",
							Port:       80,
							Protocol:   v1.ProtocolSCTP,
							TargetPort: intstr.FromInt32(80),
						},
						{
							Name:       "port-2",
							Port:       443,
							Protocol:   v1.ProtocolSCTP,
							TargetPort: intstr.FromInt32(443),
						},
					},
				},
			},
		},
		{
			generator: ServiceGeneratorV2{},
			params: map[string]interface{}{
				"selector":  "foo=bar",
				"name":      "test",
				"ports":     "80,8080",
				"protocols": "8080/SCTP",
			},
			expected: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
					},
					Ports: []v1.ServicePort{
						{
							Name:       "port-1",
							Port:       80,
							Protocol:   v1.ProtocolTCP,
							TargetPort: intstr.FromInt32(80),
						},
						{
							Name:       "port-2",
							Port:       8080,
							Protocol:   v1.ProtocolSCTP,
							TargetPort: intstr.FromInt32(8080),
						},
					},
				},
			},
		},
		{
			generator: ServiceGeneratorV2{},
			params: map[string]interface{}{
				"selector":  "foo=bar",
				"name":      "test",
				"ports":     "80,8080,8081,8082",
				"protocols": "8080/UDP,8081/TCP,8082/SCTP",
			},
			expected: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
					},
					Ports: []v1.ServicePort{
						{
							Name:       "port-1",
							Port:       80,
							Protocol:   v1.ProtocolTCP,
							TargetPort: intstr.FromInt32(80),
						},
						{
							Name:       "port-2",
							Port:       8080,
							Protocol:   v1.ProtocolUDP,
							TargetPort: intstr.FromInt32(8080),
						},
						{
							Name:       "port-3",
							Port:       8081,
							Protocol:   v1.ProtocolTCP,
							TargetPort: intstr.FromInt32(8081),
						},
						{
							Name:       "port-4",
							Port:       8082,
							Protocol:   v1.ProtocolSCTP,
							TargetPort: intstr.FromInt32(8082),
						},
					},
				},
			},
		},
		{
			generator: ServiceGeneratorV2{},
			params: map[string]interface{}{
				"selector":       "foo=bar,baz=blah",
				"name":           "test",
				"protocol":       "SCTP",
				"container-port": "1234",
				"cluster-ip":     "None",
			},
			expected: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
					Ports:     []v1.ServicePort{},
					ClusterIP: v1.ClusterIPNone,
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			obj, err := tt.generator.Generate(tt.params)
			if !reflect.DeepEqual(obj, &tt.expected) {
				t.Errorf("expected:\n%#v\ngot\n%#v\n", &tt.expected, obj)
			}
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}
