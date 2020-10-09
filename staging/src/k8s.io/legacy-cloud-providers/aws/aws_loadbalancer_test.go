// +build !providerless

/*
Copyright 2017 The Kubernetes Authors.

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

package aws

import (
	"k8s.io/apimachinery/pkg/types"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/elb"
	"github.com/stretchr/testify/assert"

	"k8s.io/api/core/v1"
)

func TestElbProtocolsAreEqual(t *testing.T) {
	grid := []struct {
		L        *string
		R        *string
		Expected bool
	}{
		{
			L:        aws.String("http"),
			R:        aws.String("http"),
			Expected: true,
		},
		{
			L:        aws.String("HTTP"),
			R:        aws.String("http"),
			Expected: true,
		},
		{
			L:        aws.String("HTTP"),
			R:        aws.String("TCP"),
			Expected: false,
		},
		{
			L:        aws.String(""),
			R:        aws.String("TCP"),
			Expected: false,
		},
		{
			L:        aws.String(""),
			R:        aws.String(""),
			Expected: true,
		},
		{
			L:        nil,
			R:        aws.String(""),
			Expected: false,
		},
		{
			L:        aws.String(""),
			R:        nil,
			Expected: false,
		},
		{
			L:        nil,
			R:        nil,
			Expected: true,
		},
	}
	for _, g := range grid {
		actual := elbProtocolsAreEqual(g.L, g.R)
		if actual != g.Expected {
			t.Errorf("unexpected result from protocolsEquals(%v, %v)", g.L, g.R)
		}
	}
}

func TestAWSARNEquals(t *testing.T) {
	grid := []struct {
		L        *string
		R        *string
		Expected bool
	}{
		{
			L:        aws.String("arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012"),
			R:        aws.String("arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012"),
			Expected: true,
		},
		{
			L:        aws.String("ARN:AWS:ACM:US-EAST-1:123456789012:CERTIFICATE/12345678-1234-1234-1234-123456789012"),
			R:        aws.String("arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012"),
			Expected: true,
		},
		{
			L:        aws.String("arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012"),
			R:        aws.String(""),
			Expected: false,
		},
		{
			L:        aws.String(""),
			R:        aws.String(""),
			Expected: true,
		},
		{
			L:        nil,
			R:        aws.String(""),
			Expected: false,
		},
		{
			L:        aws.String(""),
			R:        nil,
			Expected: false,
		},
		{
			L:        nil,
			R:        nil,
			Expected: true,
		},
	}
	for _, g := range grid {
		actual := awsArnEquals(g.L, g.R)
		if actual != g.Expected {
			t.Errorf("unexpected result from awsArnEquals(%v, %v)", g.L, g.R)
		}
	}
}

func TestIsNLB(t *testing.T) {
	tests := []struct {
		name string

		annotations map[string]string
		want        bool
	}{
		{
			"NLB annotation provided",
			map[string]string{"service.beta.kubernetes.io/aws-load-balancer-type": "nlb"},
			true,
		},
		{
			"NLB annotation has invalid value",
			map[string]string{"service.beta.kubernetes.io/aws-load-balancer-type": "elb"},
			false,
		},
		{
			"NLB annotation absent",
			map[string]string{},
			false,
		},
	}

	for _, test := range tests {
		t.Logf("Running test case %s", test.name)
		got := isNLB(test.annotations)

		if got != test.want {
			t.Errorf("Incorrect value for isNLB() case %s. Got %t, expected %t.", test.name, got, test.want)
		}
	}
}

func TestIsLBExternal(t *testing.T) {
	tests := []struct {
		name        string
		annotations map[string]string
		want        bool
	}{
		{
			name:        "No annotation",
			annotations: map[string]string{},
			want:        false,
		},
		{
			name:        "Type NLB",
			annotations: map[string]string{"service.beta.kubernetes.io/aws-load-balancer-type": "nlb"},
			want:        false,
		},
		{
			name:        "Type NLB-IP",
			annotations: map[string]string{"service.beta.kubernetes.io/aws-load-balancer-type": "nlb-ip"},
			want:        true,
		},
		{
			name:        "Type External",
			annotations: map[string]string{"service.beta.kubernetes.io/aws-load-balancer-type": "external"},
			want:        true,
		},
	}
	for _, test := range tests {
		t.Logf("Running test case %s", test.name)
		got := isLBExternal(test.annotations)

		if got != test.want {
			t.Errorf("Incorrect value for isLBExternal() case %s. Got %t, expected %t.", test.name, got, test.want)
		}
	}
}

func TestSyncElbListeners(t *testing.T) {
	tests := []struct {
		name                 string
		loadBalancerName     string
		listeners            []*elb.Listener
		listenerDescriptions []*elb.ListenerDescription
		toCreate             []*elb.Listener
		toDelete             []*int64
	}{
		{
			name:             "no edge cases",
			loadBalancerName: "lb_one",
			listeners: []*elb.Listener{
				{InstancePort: aws.Int64(443), InstanceProtocol: aws.String("HTTP"), LoadBalancerPort: aws.Int64(443), Protocol: aws.String("HTTP"), SSLCertificateId: aws.String("abc-123")},
				{InstancePort: aws.Int64(80), InstanceProtocol: aws.String("TCP"), LoadBalancerPort: aws.Int64(80), Protocol: aws.String("TCP"), SSLCertificateId: aws.String("def-456")},
				{InstancePort: aws.Int64(8443), InstanceProtocol: aws.String("TCP"), LoadBalancerPort: aws.Int64(8443), Protocol: aws.String("TCP"), SSLCertificateId: aws.String("def-456")},
			},
			listenerDescriptions: []*elb.ListenerDescription{
				{Listener: &elb.Listener{InstancePort: aws.Int64(80), InstanceProtocol: aws.String("TCP"), LoadBalancerPort: aws.Int64(80), Protocol: aws.String("TCP")}},
				{Listener: &elb.Listener{InstancePort: aws.Int64(8443), InstanceProtocol: aws.String("TCP"), LoadBalancerPort: aws.Int64(8443), Protocol: aws.String("TCP"), SSLCertificateId: aws.String("def-456")}},
			},
			toDelete: []*int64{
				aws.Int64(80),
			},
			toCreate: []*elb.Listener{
				{InstancePort: aws.Int64(443), InstanceProtocol: aws.String("HTTP"), LoadBalancerPort: aws.Int64(443), Protocol: aws.String("HTTP"), SSLCertificateId: aws.String("abc-123")},
				{InstancePort: aws.Int64(80), InstanceProtocol: aws.String("TCP"), LoadBalancerPort: aws.Int64(80), Protocol: aws.String("TCP"), SSLCertificateId: aws.String("def-456")},
			},
		},
		{
			name:             "no listeners to delete",
			loadBalancerName: "lb_two",
			listeners: []*elb.Listener{
				{InstancePort: aws.Int64(443), InstanceProtocol: aws.String("HTTP"), LoadBalancerPort: aws.Int64(443), Protocol: aws.String("HTTP"), SSLCertificateId: aws.String("abc-123")},
				{InstancePort: aws.Int64(80), InstanceProtocol: aws.String("TCP"), LoadBalancerPort: aws.Int64(80), Protocol: aws.String("TCP"), SSLCertificateId: aws.String("def-456")},
			},
			listenerDescriptions: []*elb.ListenerDescription{
				{Listener: &elb.Listener{InstancePort: aws.Int64(443), InstanceProtocol: aws.String("HTTP"), LoadBalancerPort: aws.Int64(443), Protocol: aws.String("HTTP"), SSLCertificateId: aws.String("abc-123")}},
			},
			toCreate: []*elb.Listener{
				{InstancePort: aws.Int64(80), InstanceProtocol: aws.String("TCP"), LoadBalancerPort: aws.Int64(80), Protocol: aws.String("TCP"), SSLCertificateId: aws.String("def-456")},
			},
			toDelete: []*int64{},
		},
		{
			name:             "no listeners to create",
			loadBalancerName: "lb_three",
			listeners: []*elb.Listener{
				{InstancePort: aws.Int64(443), InstanceProtocol: aws.String("HTTP"), LoadBalancerPort: aws.Int64(443), Protocol: aws.String("HTTP"), SSLCertificateId: aws.String("abc-123")},
			},
			listenerDescriptions: []*elb.ListenerDescription{
				{Listener: &elb.Listener{InstancePort: aws.Int64(80), InstanceProtocol: aws.String("TCP"), LoadBalancerPort: aws.Int64(80), Protocol: aws.String("TCP")}},
				{Listener: &elb.Listener{InstancePort: aws.Int64(443), InstanceProtocol: aws.String("HTTP"), LoadBalancerPort: aws.Int64(443), Protocol: aws.String("HTTP"), SSLCertificateId: aws.String("abc-123")}},
			},
			toDelete: []*int64{
				aws.Int64(80),
			},
			toCreate: []*elb.Listener{},
		},
		{
			name:             "nil actual listener",
			loadBalancerName: "lb_four",
			listeners: []*elb.Listener{
				{InstancePort: aws.Int64(443), InstanceProtocol: aws.String("HTTP"), LoadBalancerPort: aws.Int64(443), Protocol: aws.String("HTTP")},
			},
			listenerDescriptions: []*elb.ListenerDescription{
				{Listener: &elb.Listener{InstancePort: aws.Int64(443), InstanceProtocol: aws.String("HTTP"), LoadBalancerPort: aws.Int64(443), Protocol: aws.String("HTTP"), SSLCertificateId: aws.String("abc-123")}},
				{Listener: nil},
			},
			toDelete: []*int64{
				aws.Int64(443),
			},
			toCreate: []*elb.Listener{
				{InstancePort: aws.Int64(443), InstanceProtocol: aws.String("HTTP"), LoadBalancerPort: aws.Int64(443), Protocol: aws.String("HTTP")},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			additions, removals := syncElbListeners(test.loadBalancerName, test.listeners, test.listenerDescriptions)
			assert.Equal(t, additions, test.toCreate)
			assert.Equal(t, removals, test.toDelete)
		})
	}
}

func TestElbListenersAreEqual(t *testing.T) {
	tests := []struct {
		name             string
		expected, actual *elb.Listener
		equal            bool
	}{
		{
			name:     "should be equal",
			expected: &elb.Listener{InstancePort: aws.Int64(80), InstanceProtocol: aws.String("TCP"), LoadBalancerPort: aws.Int64(80), Protocol: aws.String("TCP")},
			actual:   &elb.Listener{InstancePort: aws.Int64(80), InstanceProtocol: aws.String("TCP"), LoadBalancerPort: aws.Int64(80), Protocol: aws.String("TCP")},
			equal:    true,
		},
		{
			name:     "instance port should be different",
			expected: &elb.Listener{InstancePort: aws.Int64(443), InstanceProtocol: aws.String("TCP"), LoadBalancerPort: aws.Int64(80), Protocol: aws.String("TCP")},
			actual:   &elb.Listener{InstancePort: aws.Int64(80), InstanceProtocol: aws.String("TCP"), LoadBalancerPort: aws.Int64(80), Protocol: aws.String("TCP")},
			equal:    false,
		},
		{
			name:     "instance protocol should be different",
			expected: &elb.Listener{InstancePort: aws.Int64(80), InstanceProtocol: aws.String("HTTP"), LoadBalancerPort: aws.Int64(80), Protocol: aws.String("TCP")},
			actual:   &elb.Listener{InstancePort: aws.Int64(80), InstanceProtocol: aws.String("TCP"), LoadBalancerPort: aws.Int64(80), Protocol: aws.String("TCP")},
			equal:    false,
		},
		{
			name:     "load balancer port should be different",
			expected: &elb.Listener{InstancePort: aws.Int64(443), InstanceProtocol: aws.String("TCP"), LoadBalancerPort: aws.Int64(443), Protocol: aws.String("TCP")},
			actual:   &elb.Listener{InstancePort: aws.Int64(443), InstanceProtocol: aws.String("TCP"), LoadBalancerPort: aws.Int64(80), Protocol: aws.String("TCP")},
			equal:    false,
		},
		{
			name:     "protocol should be different",
			expected: &elb.Listener{InstancePort: aws.Int64(443), InstanceProtocol: aws.String("TCP"), LoadBalancerPort: aws.Int64(80), Protocol: aws.String("TCP")},
			actual:   &elb.Listener{InstancePort: aws.Int64(443), InstanceProtocol: aws.String("TCP"), LoadBalancerPort: aws.Int64(80), Protocol: aws.String("HTTP")},
			equal:    false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			assert.Equal(t, test.equal, elbListenersAreEqual(test.expected, test.actual))
		})
	}
}

func TestBuildTargetGroupName(t *testing.T) {
	type args struct {
		serviceName    types.NamespacedName
		servicePort    int64
		nodePort       int64
		targetProtocol string
		targetType     string
		nlbConfig      nlbPortMapping
	}
	tests := []struct {
		name      string
		clusterID string
		args      args
		want      string
	}{
		{
			name:      "base case",
			clusterID: "cluster-a",
			args: args{
				serviceName:    types.NamespacedName{Namespace: "default", Name: "service-a"},
				servicePort:    80,
				nodePort:       8080,
				targetProtocol: "TCP",
				targetType:     "instance",
				nlbConfig:      nlbPortMapping{},
			},
			want: "k8s-default-servicea-7fa2e07508",
		},
		{
			name:      "base case & clusterID changed",
			clusterID: "cluster-b",
			args: args{
				serviceName:    types.NamespacedName{Namespace: "default", Name: "service-a"},
				servicePort:    80,
				nodePort:       8080,
				targetProtocol: "TCP",
				targetType:     "instance",
				nlbConfig:      nlbPortMapping{},
			},
			want: "k8s-default-servicea-719ee635da",
		},
		{
			name:      "base case & serviceNamespace changed",
			clusterID: "cluster-a",
			args: args{
				serviceName:    types.NamespacedName{Namespace: "another", Name: "service-a"},
				servicePort:    80,
				nodePort:       8080,
				targetProtocol: "TCP",
				targetType:     "instance",
				nlbConfig:      nlbPortMapping{},
			},
			want: "k8s-another-servicea-f66e09847d",
		},
		{
			name:      "base case & serviceName changed",
			clusterID: "cluster-a",
			args: args{
				serviceName:    types.NamespacedName{Namespace: "default", Name: "service-b"},
				servicePort:    80,
				nodePort:       8080,
				targetProtocol: "TCP",
				targetType:     "instance",
				nlbConfig:      nlbPortMapping{},
			},
			want: "k8s-default-serviceb-196c19c881",
		},
		{
			name:      "base case & servicePort changed",
			clusterID: "cluster-a",
			args: args{
				serviceName:    types.NamespacedName{Namespace: "default", Name: "service-a"},
				servicePort:    9090,
				nodePort:       8080,
				targetProtocol: "TCP",
				targetType:     "instance",
				nlbConfig:      nlbPortMapping{},
			},
			want: "k8s-default-servicea-06876706cb",
		},
		{
			name:      "base case & nodePort changed",
			clusterID: "cluster-a",
			args: args{
				serviceName:    types.NamespacedName{Namespace: "default", Name: "service-a"},
				servicePort:    80,
				nodePort:       9090,
				targetProtocol: "TCP",
				targetType:     "instance",
				nlbConfig:      nlbPortMapping{},
			},
			want: "k8s-default-servicea-119f844ec0",
		},
		{
			name:      "base case & targetProtocol changed",
			clusterID: "cluster-a",
			args: args{
				serviceName:    types.NamespacedName{Namespace: "default", Name: "service-a"},
				servicePort:    80,
				nodePort:       8080,
				targetProtocol: "UDP",
				targetType:     "instance",
				nlbConfig:      nlbPortMapping{},
			},
			want: "k8s-default-servicea-3868761686",
		},
		{
			name:      "base case & targetType changed",
			clusterID: "cluster-a",
			args: args{
				serviceName:    types.NamespacedName{Namespace: "default", Name: "service-a"},
				servicePort:    80,
				nodePort:       8080,
				targetProtocol: "TCP",
				targetType:     "ip",
				nlbConfig:      nlbPortMapping{},
			},
			want: "k8s-default-servicea-0fa31f4b0f",
		},
		{
			name:      "custom healthcheck config",
			clusterID: "cluster-a",
			args: args{
				serviceName:    types.NamespacedName{Namespace: "default", Name: "service-a"},
				servicePort:    80,
				nodePort:       8080,
				targetProtocol: "TCP",
				targetType:     "ip",
				nlbConfig: nlbPortMapping{
					HealthCheckConfig: healthCheckConfig{
						Protocol: "HTTP",
						Interval: 10,
					},
				},
			},
			want: "k8s-default-servicea-4028e49618",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &Cloud{
				tagging: awsTagging{ClusterID: tt.clusterID},
			}
			if got := c.buildTargetGroupName(tt.args.serviceName, tt.args.servicePort, tt.args.nodePort, tt.args.targetProtocol, tt.args.targetType, tt.args.nlbConfig); got != tt.want {
				assert.Equal(t, tt.want, got)
			}
		})
	}
}

func TestFilterTargetNodes(t *testing.T) {
	tests := []struct {
		name                    string
		nodeLabels, annotations map[string]string
		nodeTargeted            bool
	}{
		{
			name:         "when no filter is provided, node should be targeted",
			nodeLabels:   map[string]string{"k1": "v1"},
			nodeTargeted: true,
		},
		{
			name:         "when all key-value filters match, node should be targeted",
			nodeLabels:   map[string]string{"k1": "v1", "k2": "v2"},
			annotations:  map[string]string{ServiceAnnotationLoadBalancerTargetNodeLabels: "k1=v1,k2=v2"},
			nodeTargeted: true,
		},
		{
			name:         "when all just-key filter match, node should be targeted",
			nodeLabels:   map[string]string{"k1": "v1", "k2": "v2"},
			annotations:  map[string]string{ServiceAnnotationLoadBalancerTargetNodeLabels: "k1,k2"},
			nodeTargeted: true,
		},
		{
			name:         "when some filters do not match, node should not be targeted",
			nodeLabels:   map[string]string{"k1": "v1"},
			annotations:  map[string]string{ServiceAnnotationLoadBalancerTargetNodeLabels: "k1=v1,k2"},
			nodeTargeted: false,
		},
		{
			name:         "when no filter matches, node should not be targeted",
			nodeLabels:   map[string]string{"k1": "v1", "k2": "v2"},
			annotations:  map[string]string{ServiceAnnotationLoadBalancerTargetNodeLabels: "k3=v3"},
			nodeTargeted: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			node := &v1.Node{}
			node.Labels = test.nodeLabels

			nodes := []*v1.Node{node}
			targetNodes := filterTargetNodes(nodes, test.annotations)

			if test.nodeTargeted {
				assert.Equal(t, nodes, targetNodes)
			} else {
				assert.Empty(t, targetNodes)
			}
		})
	}
}

func TestProxyProtocolPortsWithNewELB(t *testing.T) {
	tests := []struct {
		name                    string
		listeners               []*elb.Listener
		proxyProtocolAnnotation string
		expectResults           []int64
	}{
		{
			name: "All ports case",
			listeners: []*elb.Listener{
				{LoadBalancerPort: aws.Int64(80), InstancePort: aws.Int64(30080), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("HTTP"), SSLCertificateId: aws.String("abc-123")},
				{LoadBalancerPort: aws.Int64(443), InstancePort: aws.Int64(30443), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("TCP"), SSLCertificateId: aws.String("def-456")},
				{LoadBalancerPort: aws.Int64(8443), InstancePort: aws.Int64(38443), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("TCP"), SSLCertificateId: aws.String("def-456")},
			},
			proxyProtocolAnnotation: "*",
			expectResults:           []int64{80, 443, 8443},
		},
		{
			name: "Empty proxyProtocolAnnotation case",
			listeners: []*elb.Listener{
				{LoadBalancerPort: aws.Int64(80), InstancePort: aws.Int64(30080), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("HTTP"), SSLCertificateId: aws.String("abc-123")},
				{LoadBalancerPort: aws.Int64(443), InstancePort: aws.Int64(30443), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("TCP"), SSLCertificateId: aws.String("def-456")},
				{LoadBalancerPort: aws.Int64(8443), InstancePort: aws.Int64(38443), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("TCP"), SSLCertificateId: aws.String("def-456")},
			},
			proxyProtocolAnnotation: "",
			expectResults:           []int64{},
		},
		{
			name: "Comma-separated list of ports proxyProtocolAnnotation case",
			listeners: []*elb.Listener{
				{LoadBalancerPort: aws.Int64(80), InstancePort: aws.Int64(30080), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("HTTP"), SSLCertificateId: aws.String("abc-123")},
				{LoadBalancerPort: aws.Int64(443), InstancePort: aws.Int64(30443), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("TCP"), SSLCertificateId: aws.String("def-456")},
				{LoadBalancerPort: aws.Int64(8443), InstancePort: aws.Int64(38443), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("TCP"), SSLCertificateId: aws.String("def-456")},
			},
			proxyProtocolAnnotation: "80,443",
			expectResults:           []int64{80, 443},
		},
	}

	for _, test := range tests {
		t.Logf("Running test case %s", test.name)
		annotations := make(map[string]string)
		if test.proxyProtocolAnnotation != "" {
			annotations[ServiceAnnotationLoadBalancerProxyProtocol] = test.proxyProtocolAnnotation
		}
		proxyProtocolPorts := getPortSets(test.proxyProtocolAnnotation)
		proxyProtocolAnnotation := annotations[ServiceAnnotationLoadBalancerProxyProtocol]
		var results []int64
		if proxyProtocolPorts != nil || proxyProtocolAnnotation == "*" {

			for _, listener := range test.listeners {
				if proxyProtocolAnnotation == "*" || (proxyProtocolPorts != nil && proxyProtocolPorts.numbers.Has(*listener.LoadBalancerPort)) {
					results = append(results, *listener.LoadBalancerPort)
				}
			}
		}

		assert.ElementsMatch(t, results, test.expectResults,
			"Incorrect (%#v vs expected %#v) for case: %s",
			results, test.expectResults, test.name)

	}
}

func TestProxyProtocolPortsWithExistsELB(t *testing.T) {
	tests := []struct {
		name                    string
		listeners               []*elb.Listener
		proxyProtocolAnnotation string
		backendListener         []*elb.BackendServerDescription
		expectResultsForAdd     []int64
		expectResultsForRemove  []int64
	}{
		{
			name: "Set proxy protocol on all ELB backend with *",
			listeners: []*elb.Listener{
				{LoadBalancerPort: aws.Int64(80), InstancePort: aws.Int64(30080), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("HTTP"), SSLCertificateId: aws.String("abc-123")},
				{LoadBalancerPort: aws.Int64(443), InstancePort: aws.Int64(30443), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("TCP"), SSLCertificateId: aws.String("def-456")},
				{LoadBalancerPort: aws.Int64(8443), InstancePort: aws.Int64(38443), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("TCP"), SSLCertificateId: aws.String("def-456")},
			},
			backendListener:         []*elb.BackendServerDescription{},
			proxyProtocolAnnotation: "*",
			expectResultsForAdd:     []int64{30080, 30443, 38443},
			expectResultsForRemove:  []int64{},
		},
		{
			name: "Set proxy protocol on all ELB backend with comma-separated ports",
			listeners: []*elb.Listener{
				{LoadBalancerPort: aws.Int64(80), InstancePort: aws.Int64(30080), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("HTTP"), SSLCertificateId: aws.String("abc-123")},
				{LoadBalancerPort: aws.Int64(443), InstancePort: aws.Int64(30443), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("TCP"), SSLCertificateId: aws.String("def-456")},
				{LoadBalancerPort: aws.Int64(8443), InstancePort: aws.Int64(38443), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("TCP"), SSLCertificateId: aws.String("def-456")},
			},
			backendListener:         []*elb.BackendServerDescription{},
			proxyProtocolAnnotation: "80,443,8443",
			expectResultsForAdd:     []int64{30080, 30443, 38443},
			expectResultsForRemove:  []int64{},
		},
		{
			name: "Set proxy protocol on ELB that proxy already set with *",
			listeners: []*elb.Listener{
				{LoadBalancerPort: aws.Int64(80), InstancePort: aws.Int64(30080), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("HTTP"), SSLCertificateId: aws.String("abc-123")},
				{LoadBalancerPort: aws.Int64(443), InstancePort: aws.Int64(30443), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("TCP"), SSLCertificateId: aws.String("def-456")},
				{LoadBalancerPort: aws.Int64(8443), InstancePort: aws.Int64(38443), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("TCP"), SSLCertificateId: aws.String("def-456")},
			},
			backendListener: []*elb.BackendServerDescription{
				{InstancePort: aws.Int64(30080), PolicyNames: []*string{aws.String(ProxyProtocolPolicyName)}},
				{InstancePort: aws.Int64(30443), PolicyNames: []*string{aws.String(ProxyProtocolPolicyName)}},
				{InstancePort: aws.Int64(38443), PolicyNames: []*string{aws.String(ProxyProtocolPolicyName)}},
			},
			proxyProtocolAnnotation: "*",
			expectResultsForAdd:     []int64{},
			expectResultsForRemove:  []int64{},
		},
		{
			name: "Set proxy protocol on ELB that proxy already set with comma-separated list",
			listeners: []*elb.Listener{
				{LoadBalancerPort: aws.Int64(80), InstancePort: aws.Int64(30080), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("HTTP"), SSLCertificateId: aws.String("abc-123")},
				{LoadBalancerPort: aws.Int64(443), InstancePort: aws.Int64(30443), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("TCP"), SSLCertificateId: aws.String("def-456")},
				{LoadBalancerPort: aws.Int64(8443), InstancePort: aws.Int64(38443), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("TCP"), SSLCertificateId: aws.String("def-456")},
			},
			backendListener: []*elb.BackendServerDescription{
				{InstancePort: aws.Int64(30080), PolicyNames: []*string{aws.String(ProxyProtocolPolicyName)}},
				{InstancePort: aws.Int64(30443), PolicyNames: []*string{aws.String(ProxyProtocolPolicyName)}},
				{InstancePort: aws.Int64(38443), PolicyNames: []*string{aws.String(ProxyProtocolPolicyName)}},
			},
			proxyProtocolAnnotation: "80,443,8443",
			expectResultsForAdd:     []int64{},
			expectResultsForRemove:  []int64{},
		},
		{
			name: "Set proxy protocol on new listener with *",
			listeners: []*elb.Listener{
				{LoadBalancerPort: aws.Int64(80), InstancePort: aws.Int64(30080), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("HTTP"), SSLCertificateId: aws.String("abc-123")},
				{LoadBalancerPort: aws.Int64(443), InstancePort: aws.Int64(30443), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("TCP"), SSLCertificateId: aws.String("def-456")},
				{LoadBalancerPort: aws.Int64(8443), InstancePort: aws.Int64(38443), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("TCP"), SSLCertificateId: aws.String("def-456")},
			},
			backendListener: []*elb.BackendServerDescription{
				{InstancePort: aws.Int64(30080), PolicyNames: []*string{aws.String(ProxyProtocolPolicyName)}},
			},
			proxyProtocolAnnotation: "*",
			expectResultsForAdd:     []int64{30443, 38443},
			expectResultsForRemove:  []int64{},
		},
		{
			name: "Unset proxy protocol on deleted listener",
			listeners: []*elb.Listener{
				{LoadBalancerPort: aws.Int64(80), InstancePort: aws.Int64(30080), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("HTTP"), SSLCertificateId: aws.String("abc-123")},
				{LoadBalancerPort: aws.Int64(443), InstancePort: aws.Int64(30443), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("TCP"), SSLCertificateId: aws.String("def-456")},
				{LoadBalancerPort: aws.Int64(8443), InstancePort: aws.Int64(38443), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("TCP"), SSLCertificateId: aws.String("def-456")},
			},
			backendListener: []*elb.BackendServerDescription{
				{InstancePort: aws.Int64(30080), PolicyNames: []*string{aws.String(ProxyProtocolPolicyName)}},
				{InstancePort: aws.Int64(30443), PolicyNames: []*string{aws.String(ProxyProtocolPolicyName)}},
				{InstancePort: aws.Int64(38443), PolicyNames: []*string{aws.String(ProxyProtocolPolicyName)}},
				{InstancePort: aws.Int64(39443), PolicyNames: []*string{aws.String(ProxyProtocolPolicyName)}},
			},
			proxyProtocolAnnotation: "*",
			expectResultsForAdd:     []int64{},
			expectResultsForRemove:  []int64{39443},
		},
		{
			name: "Proxy protocol needs to be removed with empty proxyProtocolAnnotation",
			listeners: []*elb.Listener{
				{LoadBalancerPort: aws.Int64(80), InstancePort: aws.Int64(30080), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("HTTP"), SSLCertificateId: aws.String("abc-123")},
				{LoadBalancerPort: aws.Int64(443), InstancePort: aws.Int64(30443), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("TCP"), SSLCertificateId: aws.String("def-456")},
				{LoadBalancerPort: aws.Int64(8443), InstancePort: aws.Int64(38443), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("TCP"), SSLCertificateId: aws.String("def-456")},
			},
			backendListener: []*elb.BackendServerDescription{
				{InstancePort: aws.Int64(30080), PolicyNames: []*string{aws.String(ProxyProtocolPolicyName)}},
				{InstancePort: aws.Int64(30443), PolicyNames: []*string{aws.String(ProxyProtocolPolicyName)}},
				{InstancePort: aws.Int64(38443), PolicyNames: []*string{aws.String(ProxyProtocolPolicyName)}},
				{InstancePort: aws.Int64(39443), PolicyNames: []*string{aws.String(ProxyProtocolPolicyName)}},
			},
			proxyProtocolAnnotation: "",
			expectResultsForAdd:     []int64{},
			expectResultsForRemove:  []int64{30080, 30443, 38443, 39443},
		},
		{
			name: "Empty proxyProtocolAnnotation and empty backendListener",
			listeners: []*elb.Listener{
				{LoadBalancerPort: aws.Int64(80), InstancePort: aws.Int64(30080), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("HTTP"), SSLCertificateId: aws.String("abc-123")},
				{LoadBalancerPort: aws.Int64(443), InstancePort: aws.Int64(30443), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("TCP"), SSLCertificateId: aws.String("def-456")},
				{LoadBalancerPort: aws.Int64(8443), InstancePort: aws.Int64(38443), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("TCP"), SSLCertificateId: aws.String("def-456")},
			},
			backendListener:         []*elb.BackendServerDescription{},
			proxyProtocolAnnotation: "",
			expectResultsForAdd:     []int64{},
			expectResultsForRemove:  []int64{},
		},
		{
			name: "Remove policy which ports not specified in annotation",
			listeners: []*elb.Listener{
				{LoadBalancerPort: aws.Int64(80), InstancePort: aws.Int64(30080), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("HTTP"), SSLCertificateId: aws.String("abc-123")},
				{LoadBalancerPort: aws.Int64(443), InstancePort: aws.Int64(30443), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("TCP"), SSLCertificateId: aws.String("def-456")},
				{LoadBalancerPort: aws.Int64(8443), InstancePort: aws.Int64(38443), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("TCP"), SSLCertificateId: aws.String("def-456")},
			},
			backendListener: []*elb.BackendServerDescription{
				{InstancePort: aws.Int64(30080), PolicyNames: []*string{aws.String(ProxyProtocolPolicyName)}},
				{InstancePort: aws.Int64(30443), PolicyNames: []*string{aws.String(ProxyProtocolPolicyName)}},
				{InstancePort: aws.Int64(39443), PolicyNames: []*string{aws.String(ProxyProtocolPolicyName)}},
			},
			proxyProtocolAnnotation: "44444",
			expectResultsForAdd:     []int64{},
			expectResultsForRemove:  []int64{30080, 30443, 39443},
		},
		{
			name: "proxy protocol needs to be added and removed",
			listeners: []*elb.Listener{
				{LoadBalancerPort: aws.Int64(80), InstancePort: aws.Int64(30080), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("HTTP"), SSLCertificateId: aws.String("abc-123")},
				{LoadBalancerPort: aws.Int64(443), InstancePort: aws.Int64(30443), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("TCP"), SSLCertificateId: aws.String("def-456")},
				{LoadBalancerPort: aws.Int64(8443), InstancePort: aws.Int64(38443), InstanceProtocol: aws.String("TCP"), Protocol: aws.String("TCP"), SSLCertificateId: aws.String("def-456")},
			},
			backendListener: []*elb.BackendServerDescription{
				{InstancePort: aws.Int64(38443), PolicyNames: []*string{aws.String(ProxyProtocolPolicyName)}},
				{InstancePort: aws.Int64(39443), PolicyNames: []*string{aws.String(ProxyProtocolPolicyName)}},
			},
			proxyProtocolAnnotation: "80,443,8443",
			expectResultsForAdd:     []int64{30080, 30443},
			expectResultsForRemove:  []int64{39443},
		},
	}

	for _, test := range tests {
		t.Logf("Running test case %s", test.name)
		annotations := make(map[string]string)
		if test.proxyProtocolAnnotation != "" {
			annotations[ServiceAnnotationLoadBalancerProxyProtocol] = test.proxyProtocolAnnotation
		}
		proxyProtocolPorts := getPortSets(test.proxyProtocolAnnotation)
		proxyProtocolAnnotation := annotations[ServiceAnnotationLoadBalancerProxyProtocol]

		var resultsForAdd, resultsForRemove []int64

		keepProxyProtocolBackends := make(map[int64]bool)
		proxyProtocolBackends := make(map[int64]bool)
		for _, backendListener := range test.backendListener {
			keepProxyProtocolBackends[*backendListener.InstancePort] = false
			proxyProtocolBackends[*backendListener.InstancePort] = proxyProtocolEnabled(backendListener)
		}

		// test add proxy protocol
		for _, listener := range test.listeners {
			setPolicy := false
			instancePort := *listener.InstancePort
			loadBalancerPort := *listener.LoadBalancerPort

			proxyProtocol := proxyProtocolAnnotation == "*" || (proxyProtocolPorts != nil && proxyProtocolPorts.numbers.Has(loadBalancerPort))

			if currentState, ok := proxyProtocolBackends[instancePort]; !ok {
				// This is a new ELB backend so we only need to worry about
				// potentially adding a policy and not removing an
				// existing one
				setPolicy = proxyProtocol
			} else {
				keepProxyProtocolBackends[instancePort] = proxyProtocol
				// This is an existing ELB backend so we need to determine
				// if the state changed
				if !currentState {
					setPolicy = proxyProtocol
				}
			}

			if setPolicy {
				resultsForAdd = append(resultsForAdd, *listener.InstancePort)
			}
		}
		// test remove proxy protocol on instancePort
		for instancePort, keep := range keepProxyProtocolBackends {
			if !keep {
				resultsForRemove = append(resultsForRemove, instancePort)
			}
		}

		assert.ElementsMatch(t, resultsForAdd, test.expectResultsForAdd,
			"Incorrect add (%#v vs expected %#v) for case: %s",
			resultsForAdd, test.expectResultsForAdd, test.name)

		assert.ElementsMatch(t, resultsForRemove, test.expectResultsForRemove,
			"Incorrect remove (%#v vs expected %#v) for case: %s",
			resultsForRemove, test.expectResultsForRemove, test.name)
	}
}
