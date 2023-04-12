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

package create

import (
	"testing"

	restclient "k8s.io/client-go/rest"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/cli-runtime/pkg/genericiooptions"
)

func TestCreateServices(t *testing.T) {

	tests := []struct {
		name         string
		serviceType  v1.ServiceType
		tcp          []string
		clusterip    string
		externalName string
		nodeport     int
		expected     *v1.Service
		expectErr    bool
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
		{
			name:        "validate-ok",
			serviceType: v1.ServiceTypeClusterIP,
			tcp:         []string{"123", "234:1234"},
			clusterip:   "",
			expected: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "validate-ok",
					Labels: map[string]string{"app": "validate-ok"},
				},
				Spec: v1.ServiceSpec{Type: "ClusterIP",
					Ports: []v1.ServicePort{
						{Name: "123", Protocol: "TCP", Port: 123, TargetPort: intstr.IntOrString{Type: 0, IntVal: 123, StrVal: ""}, NodePort: 0},
						{Name: "234-1234", Protocol: "TCP", Port: 234, TargetPort: intstr.IntOrString{Type: 0, IntVal: 1234, StrVal: ""}, NodePort: 0},
					},
					Selector:  map[string]string{"app": "validate-ok"},
					ClusterIP: "", ExternalIPs: []string(nil), LoadBalancerIP: ""},
			},
			expectErr: false,
		},
		{
			name:        "invalid-ClusterIPNone",
			serviceType: v1.ServiceTypeNodePort,
			tcp:         []string{"123", "234:1234"},
			clusterip:   v1.ClusterIPNone,
			expectErr:   true,
		},
		{
			name:        "TCP-none",
			serviceType: v1.ServiceTypeClusterIP,
			clusterip:   "",
			expectErr:   true,
		},
		{
			name:         "invalid-ExternalName",
			serviceType:  v1.ServiceTypeExternalName,
			tcp:          []string{"123", "234:1234"},
			clusterip:    "",
			externalName: "@oi:test",
			expectErr:    true,
		},
		{
			name:         "externalName-ok",
			serviceType:  v1.ServiceTypeExternalName,
			tcp:          []string{"123", "234:1234"},
			clusterip:    "",
			externalName: "www.externalname.com",
			expected: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "externalName-ok",
					Labels: map[string]string{"app": "externalName-ok"},
				},
				Spec: v1.ServiceSpec{Type: "ExternalName",
					Ports: []v1.ServicePort{
						{Name: "123", Protocol: "TCP", Port: 123, TargetPort: intstr.IntOrString{Type: 0, IntVal: 123, StrVal: ""}, NodePort: 0},
						{Name: "234-1234", Protocol: "TCP", Port: 234, TargetPort: intstr.IntOrString{Type: 0, IntVal: 1234, StrVal: ""}, NodePort: 0},
					},
					Selector:  map[string]string{"app": "externalName-ok"},
					ClusterIP: "", ExternalIPs: []string(nil), LoadBalancerIP: "", ExternalName: "www.externalname.com"},
			},
			expectErr: false,
		},
		{
			name:        "my-node-port-service-ok",
			serviceType: v1.ServiceTypeNodePort,
			tcp:         []string{"443:https", "30000:8000"},
			clusterip:   "",
			expected: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "my-node-port-service-ok",
					Labels: map[string]string{"app": "my-node-port-service-ok"},
				},
				Spec: v1.ServiceSpec{Type: "NodePort",
					Ports: []v1.ServicePort{
						{Name: "443-https", Protocol: "TCP", Port: 443, TargetPort: intstr.IntOrString{Type: 1, IntVal: 0, StrVal: "https"}, NodePort: 0},
						{Name: "30000-8000", Protocol: "TCP", Port: 30000, TargetPort: intstr.IntOrString{Type: 0, IntVal: 8000, StrVal: ""}, NodePort: 0},
					},
					Selector:  map[string]string{"app": "my-node-port-service-ok"},
					ClusterIP: "", ExternalIPs: []string(nil), LoadBalancerIP: ""},
			},
			expectErr: false,
		},
		{
			name:        "my-node-port-service-ok2",
			serviceType: v1.ServiceTypeNodePort,
			tcp:         []string{"80:http"},
			clusterip:   "",
			nodeport:    4444,
			expected: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "my-node-port-service-ok2",
					Labels: map[string]string{"app": "my-node-port-service-ok2"},
				},
				Spec: v1.ServiceSpec{Type: "NodePort",
					Ports: []v1.ServicePort{
						{Name: "80-http", Protocol: "TCP", Port: 80, TargetPort: intstr.IntOrString{Type: 1, IntVal: 0, StrVal: "http"}, NodePort: 4444},
					},
					Selector:  map[string]string{"app": "my-node-port-service-ok2"},
					ClusterIP: "", ExternalIPs: []string(nil), LoadBalancerIP: ""},
			},
			expectErr: false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			options := ServiceOptions{
				Name:         tc.name,
				Type:         tc.serviceType,
				TCP:          tc.tcp,
				ClusterIP:    tc.clusterip,
				NodePort:     tc.nodeport,
				ExternalName: tc.externalName,
			}

			var service *v1.Service

			err := options.Validate()
			if err == nil {
				service, err = options.createService()
			}
			if tc.expectErr && err == nil {
				t.Errorf("%s: expected an error, but createService passes.", tc.name)
			}
			if !tc.expectErr && err != nil {
				t.Errorf("%s: unexpected error: %v", tc.name, err)
			}
			if !apiequality.Semantic.DeepEqual(service, tc.expected) {
				t.Errorf("%s: expected:\n%#v\ngot:\n%#v", tc.name, tc.expected, service)
			}
		})
	}
}

func TestCreateServiceWithNamespace(t *testing.T) {
	svcName := "test-service"
	ns := "test"
	tf := cmdtesting.NewTestFactory().WithNamespace(ns)
	defer tf.Cleanup()

	tf.ClientConfigVal = &restclient.Config{}

	ioStreams, _, buf, _ := genericiooptions.NewTestIOStreams()
	cmd := NewCmdCreateServiceClusterIP(tf, ioStreams)
	cmd.Flags().Set("dry-run", "client")
	cmd.Flags().Set("output", "jsonpath={.metadata.namespace}")
	cmd.Flags().Set("clusterip", "None")
	cmd.Run(cmd, []string{svcName})
	if buf.String() != ns {
		t.Errorf("expected output: %s, but got: %s", ns, buf.String())
	}
}
