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

package proxy

import (
	"net"
	"reflect"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/dump"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/version"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	netutils "k8s.io/utils/net"
)

const testHostname = "test-hostname"

func makeTestServiceInfo(clusterIP string, port int, protocol string, healthcheckNodePort int, svcInfoFuncs ...func(*BaseServicePortInfo)) *BaseServicePortInfo {
	bsvcPortInfo := &BaseServicePortInfo{
		clusterIP: netutils.ParseIPSloppy(clusterIP),
		port:      port,
		protocol:  v1.Protocol(protocol),
	}
	if healthcheckNodePort != 0 {
		bsvcPortInfo.healthCheckNodePort = healthcheckNodePort
	}
	for _, svcInfoFunc := range svcInfoFuncs {
		svcInfoFunc(bsvcPortInfo)
	}
	return bsvcPortInfo
}

func makeTestService(namespace, name string, svcFunc func(*v1.Service)) *v1.Service {
	svc := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:        name,
			Namespace:   namespace,
			Annotations: map[string]string{},
		},
		Spec:   v1.ServiceSpec{},
		Status: v1.ServiceStatus{},
	}
	svcFunc(svc)
	return svc
}

func addTestPort(array []v1.ServicePort, name string, protocol v1.Protocol, port, nodeport int32, targetPort int) []v1.ServicePort {
	svcPort := v1.ServicePort{
		Name:       name,
		Protocol:   protocol,
		Port:       port,
		NodePort:   nodeport,
		TargetPort: intstr.FromInt32(int32(targetPort)),
	}
	return append(array, svcPort)
}

func makeNSN(namespace, name string) types.NamespacedName {
	return types.NamespacedName{Namespace: namespace, Name: name}
}

func makeServicePortName(ns, name, port string, protocol v1.Protocol) ServicePortName {
	return ServicePortName{
		NamespacedName: makeNSN(ns, name),
		Port:           port,
		Protocol:       protocol,
	}
}
func makeIPs(ipStr ...string) []net.IP {
	var ips []net.IP
	for _, s := range ipStr {
		ips = append(ips, netutils.ParseIPSloppy(s))
	}
	return ips
}
func mustMakeCIDRs(cidrStr ...string) []*net.IPNet {
	var cidrs []*net.IPNet
	for _, s := range cidrStr {
		if _, n, err := netutils.ParseCIDRSloppy(s); err == nil {
			cidrs = append(cidrs, n)
		} else {
			panic(err)
		}
	}
	return cidrs
}

func TestServiceToServiceMap(t *testing.T) {
	testClusterIPv4 := "10.0.0.1"
	testExternalIPv4 := "8.8.8.8"
	testSourceRangeIPv4 := "0.0.0.0/1"
	testClusterIPv6 := "2001:db8:85a3:0:0:8a2e:370:7334"
	testExternalIPv6 := "2001:db8:85a3:0:0:8a2e:370:7335"
	testSourceRangeIPv6 := "2001:db8::/32"
	ipModeVIP := v1.LoadBalancerIPModeVIP
	ipModeProxy := v1.LoadBalancerIPModeProxy

	testCases := []struct {
		desc          string
		service       *v1.Service
		expected      map[ServicePortName]*BaseServicePortInfo
		ipFamily      v1.IPFamily
		ipModeEnabled bool
	}{
		{
			desc:     "nothing",
			ipFamily: v1.IPv4Protocol,

			service:  nil,
			expected: map[ServicePortName]*BaseServicePortInfo{},
		},
		{
			desc:     "headless service",
			ipFamily: v1.IPv4Protocol,

			service: makeTestService("ns2", "headless", func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeClusterIP
				svc.Spec.ClusterIP = v1.ClusterIPNone
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "rpc", "UDP", 1234, 0, 0)
			}),
			expected: map[ServicePortName]*BaseServicePortInfo{},
		},
		{
			desc:     "headless sctp service",
			ipFamily: v1.IPv4Protocol,

			service: makeTestService("ns2", "headless", func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeClusterIP
				svc.Spec.ClusterIP = v1.ClusterIPNone
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "sip", "SCTP", 7777, 0, 0)
			}),
			expected: map[ServicePortName]*BaseServicePortInfo{},
		},
		{
			desc:     "headless service without port",
			ipFamily: v1.IPv4Protocol,

			service: makeTestService("ns2", "headless-without-port", func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeClusterIP
				svc.Spec.ClusterIP = v1.ClusterIPNone
			}),
			expected: map[ServicePortName]*BaseServicePortInfo{},
		},
		{
			desc:     "cluster ip service",
			ipFamily: v1.IPv4Protocol,

			service: makeTestService("ns2", "cluster-ip", func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeClusterIP
				svc.Spec.ClusterIP = "172.16.55.4"
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "p1", "UDP", 1234, 4321, 0)
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "p2", "UDP", 1235, 5321, 0)
			}),
			expected: map[ServicePortName]*BaseServicePortInfo{
				makeServicePortName("ns2", "cluster-ip", "p1", v1.ProtocolUDP): makeTestServiceInfo("172.16.55.4", 1234, "UDP", 0),
				makeServicePortName("ns2", "cluster-ip", "p2", v1.ProtocolUDP): makeTestServiceInfo("172.16.55.4", 1235, "UDP", 0),
			},
		},
		{
			desc:     "nodeport service",
			ipFamily: v1.IPv4Protocol,

			service: makeTestService("ns2", "node-port", func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeNodePort
				svc.Spec.ClusterIP = "172.16.55.10"
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "port1", "UDP", 345, 678, 0)
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "port2", "TCP", 344, 677, 0)
			}),
			expected: map[ServicePortName]*BaseServicePortInfo{
				makeServicePortName("ns2", "node-port", "port1", v1.ProtocolUDP): makeTestServiceInfo("172.16.55.10", 345, "UDP", 0),
				makeServicePortName("ns2", "node-port", "port2", v1.ProtocolTCP): makeTestServiceInfo("172.16.55.10", 344, "TCP", 0),
			},
		},
		{
			desc:     "load balancer service",
			ipFamily: v1.IPv4Protocol,

			service: makeTestService("ns1", "load-balancer", func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeLoadBalancer
				svc.Spec.ClusterIP = "172.16.55.11"
				svc.Spec.LoadBalancerIP = "5.6.7.8"
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "port3", "UDP", 8675, 30061, 7000)
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "port4", "UDP", 8676, 30062, 7001)
				svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{IP: "10.1.2.4"}}
			}),
			expected: map[ServicePortName]*BaseServicePortInfo{
				makeServicePortName("ns1", "load-balancer", "port3", v1.ProtocolUDP): makeTestServiceInfo("172.16.55.11", 8675, "UDP", 0, func(bsvcPortInfo *BaseServicePortInfo) {
					bsvcPortInfo.loadBalancerVIPs = makeIPs("10.1.2.4")
				}),
				makeServicePortName("ns1", "load-balancer", "port4", v1.ProtocolUDP): makeTestServiceInfo("172.16.55.11", 8676, "UDP", 0, func(bsvcPortInfo *BaseServicePortInfo) {
					bsvcPortInfo.loadBalancerVIPs = makeIPs("10.1.2.4")
				}),
			},
		},
		{
			desc:     "load balancer service ipMode VIP feature gate disable",
			ipFamily: v1.IPv4Protocol,

			service: makeTestService("ns1", "load-balancer", func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeLoadBalancer
				svc.Spec.ClusterIP = "172.16.55.11"
				svc.Spec.LoadBalancerIP = "5.6.7.8"
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "port3", "UDP", 8675, 30061, 7000)
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "port4", "UDP", 8676, 30062, 7001)
				svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{IP: "10.1.2.4", IPMode: &ipModeVIP}}
			}),
			expected: map[ServicePortName]*BaseServicePortInfo{
				makeServicePortName("ns1", "load-balancer", "port3", v1.ProtocolUDP): makeTestServiceInfo("172.16.55.11", 8675, "UDP", 0, func(bsvcPortInfo *BaseServicePortInfo) {
					bsvcPortInfo.loadBalancerVIPs = makeIPs("10.1.2.4")
				}),
				makeServicePortName("ns1", "load-balancer", "port4", v1.ProtocolUDP): makeTestServiceInfo("172.16.55.11", 8676, "UDP", 0, func(bsvcPortInfo *BaseServicePortInfo) {
					bsvcPortInfo.loadBalancerVIPs = makeIPs("10.1.2.4")
				}),
			},
		},
		{
			desc:     "load balancer service ipMode Proxy feature gate disable",
			ipFamily: v1.IPv4Protocol,

			service: makeTestService("ns1", "load-balancer", func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeLoadBalancer
				svc.Spec.ClusterIP = "172.16.55.11"
				svc.Spec.LoadBalancerIP = "5.6.7.8"
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "port3", "UDP", 8675, 30061, 7000)
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "port4", "UDP", 8676, 30062, 7001)
				svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{IP: "10.1.2.4", IPMode: &ipModeProxy}}
			}),
			expected: map[ServicePortName]*BaseServicePortInfo{
				makeServicePortName("ns1", "load-balancer", "port3", v1.ProtocolUDP): makeTestServiceInfo("172.16.55.11", 8675, "UDP", 0, func(bsvcPortInfo *BaseServicePortInfo) {
					bsvcPortInfo.loadBalancerVIPs = makeIPs("10.1.2.4")
				}),
				makeServicePortName("ns1", "load-balancer", "port4", v1.ProtocolUDP): makeTestServiceInfo("172.16.55.11", 8676, "UDP", 0, func(bsvcPortInfo *BaseServicePortInfo) {
					bsvcPortInfo.loadBalancerVIPs = makeIPs("10.1.2.4")
				}),
			},
		},
		{
			desc:          "load balancer service ipMode VIP feature gate enabled",
			ipFamily:      v1.IPv4Protocol,
			ipModeEnabled: true,

			service: makeTestService("ns1", "load-balancer", func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeLoadBalancer
				svc.Spec.ClusterIP = "172.16.55.11"
				svc.Spec.LoadBalancerIP = "5.6.7.8"
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "port3", "UDP", 8675, 30061, 7000)
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "port4", "UDP", 8676, 30062, 7001)
				svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{IP: "10.1.2.4", IPMode: &ipModeVIP}}
			}),
			expected: map[ServicePortName]*BaseServicePortInfo{
				makeServicePortName("ns1", "load-balancer", "port3", v1.ProtocolUDP): makeTestServiceInfo("172.16.55.11", 8675, "UDP", 0, func(bsvcPortInfo *BaseServicePortInfo) {
					bsvcPortInfo.loadBalancerVIPs = makeIPs("10.1.2.4")
				}),
				makeServicePortName("ns1", "load-balancer", "port4", v1.ProtocolUDP): makeTestServiceInfo("172.16.55.11", 8676, "UDP", 0, func(bsvcPortInfo *BaseServicePortInfo) {
					bsvcPortInfo.loadBalancerVIPs = makeIPs("10.1.2.4")
				}),
			},
		},
		{
			desc:          "load balancer service ipMode Proxy feature gate enabled",
			ipFamily:      v1.IPv4Protocol,
			ipModeEnabled: true,

			service: makeTestService("ns1", "load-balancer", func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeLoadBalancer
				svc.Spec.ClusterIP = "172.16.55.11"
				svc.Spec.LoadBalancerIP = "5.6.7.8"
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "port3", "UDP", 8675, 30061, 7000)
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "port4", "UDP", 8676, 30062, 7001)
				svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{IP: "10.1.2.4", IPMode: &ipModeProxy}}
			}),
			expected: map[ServicePortName]*BaseServicePortInfo{
				makeServicePortName("ns1", "load-balancer", "port3", v1.ProtocolUDP): makeTestServiceInfo("172.16.55.11", 8675, "UDP", 0, func(bsvcPortInfo *BaseServicePortInfo) {
				}),
				makeServicePortName("ns1", "load-balancer", "port4", v1.ProtocolUDP): makeTestServiceInfo("172.16.55.11", 8676, "UDP", 0, func(bsvcPortInfo *BaseServicePortInfo) {
				}),
			},
		},
		{
			desc:     "load balancer service with only local traffic policy",
			ipFamily: v1.IPv4Protocol,

			service: makeTestService("ns1", "only-local-load-balancer", func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeLoadBalancer
				svc.Spec.ClusterIP = "172.16.55.12"
				svc.Spec.LoadBalancerIP = "5.6.7.8"
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "portx", "UDP", 8677, 30063, 7002)
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "porty", "UDP", 8678, 30064, 7003)
				svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{IP: "10.1.2.3"}}
				svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyLocal
				svc.Spec.HealthCheckNodePort = 345
			}),
			expected: map[ServicePortName]*BaseServicePortInfo{
				makeServicePortName("ns1", "only-local-load-balancer", "portx", v1.ProtocolUDP): makeTestServiceInfo("172.16.55.12", 8677, "UDP", 345, func(bsvcPortInfo *BaseServicePortInfo) {
					bsvcPortInfo.loadBalancerVIPs = makeIPs("10.1.2.3")
				}),
				makeServicePortName("ns1", "only-local-load-balancer", "porty", v1.ProtocolUDP): makeTestServiceInfo("172.16.55.12", 8678, "UDP", 345, func(bsvcPortInfo *BaseServicePortInfo) {
					bsvcPortInfo.loadBalancerVIPs = makeIPs("10.1.2.3")
				}),
			},
		},
		{
			desc:     "external name service",
			ipFamily: v1.IPv4Protocol,

			service: makeTestService("ns2", "external-name", func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeExternalName
				svc.Spec.ClusterIP = "172.16.55.4" // Should be ignored
				svc.Spec.ExternalName = "foo2.bar.com"
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "portz", "UDP", 1235, 5321, 0)
			}),
			expected: map[ServicePortName]*BaseServicePortInfo{},
		},
		{
			desc:     "service with ipv6 clusterIP under ipv4 mode, service should be filtered",
			ipFamily: v1.IPv4Protocol,

			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "invalidIPv6InIPV4Mode",
					Namespace: "test",
				},
				Spec: v1.ServiceSpec{
					ClusterIP: testClusterIPv6,
					Ports: []v1.ServicePort{
						{
							Name:     "testPort",
							Port:     int32(12345),
							Protocol: v1.ProtocolTCP,
						},
					},
				},
				Status: v1.ServiceStatus{
					LoadBalancer: v1.LoadBalancerStatus{
						Ingress: []v1.LoadBalancerIngress{
							{IP: testExternalIPv4},
							{IP: testExternalIPv6},
						},
					},
				},
			},
		},
		{
			desc:     "service with ipv4 clusterIP under ipv6 mode, service should be filtered",
			ipFamily: v1.IPv6Protocol,

			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "invalidIPv4InIPV6Mode",
					Namespace: "test",
				},
				Spec: v1.ServiceSpec{
					ClusterIP: testClusterIPv4,
					Ports: []v1.ServicePort{
						{
							Name:     "testPort",
							Port:     int32(12345),
							Protocol: v1.ProtocolTCP,
						},
					},
				},
				Status: v1.ServiceStatus{
					LoadBalancer: v1.LoadBalancerStatus{
						Ingress: []v1.LoadBalancerIngress{
							{IP: testExternalIPv4},
							{IP: testExternalIPv6},
						},
					},
				},
			},
		},
		{
			desc:     "service with ipv4 configurations under ipv4 mode",
			ipFamily: v1.IPv4Protocol,

			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "validIPv4",
					Namespace: "test",
				},
				Spec: v1.ServiceSpec{
					ClusterIP:                testClusterIPv4,
					ExternalIPs:              []string{testExternalIPv4},
					LoadBalancerSourceRanges: []string{testSourceRangeIPv4},
					Ports: []v1.ServicePort{
						{
							Name:     "testPort",
							Port:     int32(12345),
							Protocol: v1.ProtocolTCP,
						},
					},
				},
				Status: v1.ServiceStatus{
					LoadBalancer: v1.LoadBalancerStatus{
						Ingress: []v1.LoadBalancerIngress{
							{IP: testExternalIPv4},
							{IP: testExternalIPv6},
						},
					},
				},
			},
			expected: map[ServicePortName]*BaseServicePortInfo{
				makeServicePortName("test", "validIPv4", "testPort", v1.ProtocolTCP): makeTestServiceInfo(testClusterIPv4, 12345, "TCP", 0, func(bsvcPortInfo *BaseServicePortInfo) {
					bsvcPortInfo.externalIPs = makeIPs(testExternalIPv4)
					bsvcPortInfo.loadBalancerSourceRanges = mustMakeCIDRs(testSourceRangeIPv4)
					bsvcPortInfo.loadBalancerVIPs = makeIPs(testExternalIPv4)
				}),
			},
		},
		{
			desc:     "service with ipv6 configurations under ipv6 mode",
			ipFamily: v1.IPv6Protocol,

			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "validIPv6",
					Namespace: "test",
				},
				Spec: v1.ServiceSpec{
					ClusterIP:                testClusterIPv6,
					ExternalIPs:              []string{testExternalIPv6},
					LoadBalancerSourceRanges: []string{testSourceRangeIPv6},
					Ports: []v1.ServicePort{
						{
							Name:     "testPort",
							Port:     int32(12345),
							Protocol: v1.ProtocolTCP,
						},
					},
				},
				Status: v1.ServiceStatus{
					LoadBalancer: v1.LoadBalancerStatus{
						Ingress: []v1.LoadBalancerIngress{
							{IP: testExternalIPv4},
							{IP: testExternalIPv6},
						},
					},
				},
			},
			expected: map[ServicePortName]*BaseServicePortInfo{
				makeServicePortName("test", "validIPv6", "testPort", v1.ProtocolTCP): makeTestServiceInfo(testClusterIPv6, 12345, "TCP", 0, func(bsvcPortInfo *BaseServicePortInfo) {
					bsvcPortInfo.externalIPs = makeIPs(testExternalIPv6)
					bsvcPortInfo.loadBalancerSourceRanges = mustMakeCIDRs(testSourceRangeIPv6)
					bsvcPortInfo.loadBalancerVIPs = makeIPs(testExternalIPv6)
				}),
			},
		},
		{
			desc:     "service with both ipv4 and ipv6 configurations under ipv4 mode, ipv6 fields should be filtered",
			ipFamily: v1.IPv4Protocol,

			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "filterIPv6InIPV4Mode",
					Namespace: "test",
				},
				Spec: v1.ServiceSpec{
					ClusterIP:                testClusterIPv4,
					ExternalIPs:              []string{testExternalIPv4, testExternalIPv6},
					LoadBalancerSourceRanges: []string{testSourceRangeIPv4, testSourceRangeIPv6},
					Ports: []v1.ServicePort{
						{
							Name:     "testPort",
							Port:     int32(12345),
							Protocol: v1.ProtocolTCP,
						},
					},
				},
				Status: v1.ServiceStatus{
					LoadBalancer: v1.LoadBalancerStatus{
						Ingress: []v1.LoadBalancerIngress{
							{IP: testExternalIPv4},
							{IP: testExternalIPv6},
						},
					},
				},
			},
			expected: map[ServicePortName]*BaseServicePortInfo{
				makeServicePortName("test", "filterIPv6InIPV4Mode", "testPort", v1.ProtocolTCP): makeTestServiceInfo(testClusterIPv4, 12345, "TCP", 0, func(bsvcPortInfo *BaseServicePortInfo) {
					bsvcPortInfo.externalIPs = makeIPs(testExternalIPv4)
					bsvcPortInfo.loadBalancerSourceRanges = mustMakeCIDRs(testSourceRangeIPv4)
					bsvcPortInfo.loadBalancerVIPs = makeIPs(testExternalIPv4)
				}),
			},
		},
		{
			desc:     "service with both ipv4 and ipv6 configurations under ipv6 mode, ipv4 fields should be filtered",
			ipFamily: v1.IPv6Protocol,

			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "filterIPv4InIPV6Mode",
					Namespace: "test",
				},
				Spec: v1.ServiceSpec{
					ClusterIP:                testClusterIPv6,
					ExternalIPs:              []string{testExternalIPv4, testExternalIPv6},
					LoadBalancerSourceRanges: []string{testSourceRangeIPv4, testSourceRangeIPv6},
					Ports: []v1.ServicePort{
						{
							Name:     "testPort",
							Port:     int32(12345),
							Protocol: v1.ProtocolTCP,
						},
					},
				},
				Status: v1.ServiceStatus{
					LoadBalancer: v1.LoadBalancerStatus{
						Ingress: []v1.LoadBalancerIngress{
							{IP: testExternalIPv4},
							{IP: testExternalIPv6},
						},
					},
				},
			},
			expected: map[ServicePortName]*BaseServicePortInfo{
				makeServicePortName("test", "filterIPv4InIPV6Mode", "testPort", v1.ProtocolTCP): makeTestServiceInfo(testClusterIPv6, 12345, "TCP", 0, func(bsvcPortInfo *BaseServicePortInfo) {
					bsvcPortInfo.externalIPs = makeIPs(testExternalIPv6)
					bsvcPortInfo.loadBalancerSourceRanges = mustMakeCIDRs(testSourceRangeIPv6)
					bsvcPortInfo.loadBalancerVIPs = makeIPs(testExternalIPv6)
				}),
			},
		},
		{
			desc:     "service with extra space in LoadBalancerSourceRanges",
			ipFamily: v1.IPv4Protocol,

			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "extra-space",
					Namespace: "test",
				},
				Spec: v1.ServiceSpec{
					ClusterIP:                testClusterIPv4,
					LoadBalancerSourceRanges: []string{" 10.1.2.0/28"},
					Ports: []v1.ServicePort{
						{
							Name:     "testPort",
							Port:     int32(12345),
							Protocol: v1.ProtocolTCP,
						},
					},
				},
			},
			expected: map[ServicePortName]*BaseServicePortInfo{
				makeServicePortName("test", "extra-space", "testPort", v1.ProtocolTCP): makeTestServiceInfo(testClusterIPv4, 12345, "TCP", 0, func(bsvcPortInfo *BaseServicePortInfo) {
					bsvcPortInfo.loadBalancerSourceRanges = mustMakeCIDRs("10.1.2.0/28")
				}),
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			if !tc.ipModeEnabled {
				featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse("1.31"))
			}
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.LoadBalancerIPMode, tc.ipModeEnabled)
			svcTracker := NewServiceChangeTracker(tc.ipFamily, nil, nil)
			// outputs
			newServices := svcTracker.serviceToServiceMap(tc.service)

			if len(newServices) != len(tc.expected) {
				t.Fatalf("expected %d new, got %d: %v", len(tc.expected), len(newServices), dump.Pretty(newServices))
			}
			for svcKey, expectedInfo := range tc.expected {
				svcInfo, exists := newServices[svcKey].(*BaseServicePortInfo)
				if !exists {
					t.Fatalf("[%s] expected to find key %s", tc.desc, svcKey)
				}

				if !svcInfo.clusterIP.Equal(expectedInfo.clusterIP) ||
					svcInfo.port != expectedInfo.port ||
					svcInfo.protocol != expectedInfo.protocol ||
					svcInfo.healthCheckNodePort != expectedInfo.healthCheckNodePort ||
					!reflect.DeepEqual(svcInfo.externalIPs, expectedInfo.externalIPs) ||
					!reflect.DeepEqual(svcInfo.loadBalancerSourceRanges, expectedInfo.loadBalancerSourceRanges) ||
					!reflect.DeepEqual(svcInfo.loadBalancerVIPs, expectedInfo.loadBalancerVIPs) {
					t.Errorf("[%s] expected new[%v]to be %v, got %v", tc.desc, svcKey, expectedInfo, *svcInfo)
				}
				for svcKey, expectedInfo := range tc.expected {
					svcInfo, _ := newServices[svcKey].(*BaseServicePortInfo)
					if !svcInfo.clusterIP.Equal(expectedInfo.clusterIP) ||
						svcInfo.port != expectedInfo.port ||
						svcInfo.protocol != expectedInfo.protocol ||
						svcInfo.healthCheckNodePort != expectedInfo.healthCheckNodePort ||
						!reflect.DeepEqual(svcInfo.externalIPs, expectedInfo.externalIPs) ||
						!reflect.DeepEqual(svcInfo.loadBalancerSourceRanges, expectedInfo.loadBalancerSourceRanges) ||
						!reflect.DeepEqual(svcInfo.loadBalancerVIPs, expectedInfo.loadBalancerVIPs) {
						t.Errorf("expected new[%v]to be %v, got %v", svcKey, expectedInfo, *svcInfo)
					}
				}
			}
		})
	}
}

type FakeProxier struct {
	endpointsChanges *EndpointsChangeTracker
	serviceChanges   *ServiceChangeTracker
	svcPortMap       ServicePortMap
	endpointsMap     EndpointsMap
}

func newFakeProxier(ipFamily v1.IPFamily, t time.Time) *FakeProxier {
	ect := NewEndpointsChangeTracker(ipFamily, testHostname, nil, nil)
	ect.trackerStartTime = t
	return &FakeProxier{
		svcPortMap:       make(ServicePortMap),
		serviceChanges:   NewServiceChangeTracker(ipFamily, nil, nil),
		endpointsMap:     make(EndpointsMap),
		endpointsChanges: ect,
	}
}

func makeServiceMap(fake *FakeProxier, allServices ...*v1.Service) {
	for i := range allServices {
		fake.addService(allServices[i])
	}
}

func (proxier *FakeProxier) addService(service *v1.Service) {
	proxier.serviceChanges.Update(nil, service)
}

func (proxier *FakeProxier) updateService(oldService *v1.Service, service *v1.Service) {
	proxier.serviceChanges.Update(oldService, service)
}

func (proxier *FakeProxier) deleteService(service *v1.Service) {
	proxier.serviceChanges.Update(service, nil)
}

func TestServiceMapUpdateHeadless(t *testing.T) {
	fp := newFakeProxier(v1.IPv4Protocol, time.Time{})

	makeServiceMap(fp,
		makeTestService("ns2", "headless", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeClusterIP
			svc.Spec.ClusterIP = v1.ClusterIPNone
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "rpc", "UDP", 1234, 0, 0)
		}),
		makeTestService("ns2", "headless-without-port", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeClusterIP
			svc.Spec.ClusterIP = v1.ClusterIPNone
		}),
	)

	// Headless service should be ignored
	result := fp.svcPortMap.Update(fp.serviceChanges)
	if len(fp.svcPortMap) != 0 {
		t.Errorf("expected service map length 0, got %d", len(fp.svcPortMap))
	}

	if len(result.UpdatedServices) != 0 {
		t.Errorf("expected 0 updated services, got %d", len(result.UpdatedServices))
	}

	// No proxied services, so no healthchecks
	healthCheckNodePorts := fp.svcPortMap.HealthCheckNodePorts()
	if len(healthCheckNodePorts) != 0 {
		t.Errorf("expected healthcheck ports length 0, got %d", len(healthCheckNodePorts))
	}
}

func TestUpdateServiceTypeExternalName(t *testing.T) {
	fp := newFakeProxier(v1.IPv4Protocol, time.Time{})

	makeServiceMap(fp,
		makeTestService("ns2", "external-name", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeExternalName
			svc.Spec.ClusterIP = "172.16.55.4" // Should be ignored
			svc.Spec.ExternalName = "foo2.bar.com"
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "blah", "UDP", 1235, 5321, 0)
		}),
	)

	result := fp.svcPortMap.Update(fp.serviceChanges)
	if len(fp.svcPortMap) != 0 {
		t.Errorf("expected service map length 0, got %v", fp.svcPortMap)
	}
	if len(result.UpdatedServices) != 0 {
		t.Errorf("expected 0 updated services, got %v", result.UpdatedServices)
	}

	// No proxied services, so no healthchecks
	healthCheckNodePorts := fp.svcPortMap.HealthCheckNodePorts()
	if len(healthCheckNodePorts) != 0 {
		t.Errorf("expected healthcheck ports length 0, got %v", healthCheckNodePorts)
	}
}

func TestBuildServiceMapAddRemove(t *testing.T) {
	fp := newFakeProxier(v1.IPv4Protocol, time.Time{})

	services := []*v1.Service{
		makeTestService("ns2", "cluster-ip", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeClusterIP
			svc.Spec.ClusterIP = "172.16.55.4"
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "port1", "UDP", 1234, 4321, 0)
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "port2", "UDP", 1235, 5321, 0)
		}),
		makeTestService("ns2", "node-port", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeNodePort
			svc.Spec.ClusterIP = "172.16.55.10"
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "port1", "UDP", 345, 678, 0)
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "port2", "TCP", 344, 677, 0)
		}),
		makeTestService("ns1", "load-balancer", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeLoadBalancer
			svc.Spec.ClusterIP = "172.16.55.11"
			svc.Spec.LoadBalancerIP = "5.6.7.8"
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "foobar", "UDP", 8675, 30061, 7000)
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "baz", "UDP", 8676, 30062, 7001)
			svc.Status.LoadBalancer = v1.LoadBalancerStatus{
				Ingress: []v1.LoadBalancerIngress{
					{IP: "10.1.2.4"},
				},
			}
		}),
		makeTestService("ns1", "only-local-load-balancer", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeLoadBalancer
			svc.Spec.ClusterIP = "172.16.55.12"
			svc.Spec.LoadBalancerIP = "5.6.7.8"
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "foobar2", "UDP", 8677, 30063, 7002)
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "baz", "UDP", 8678, 30064, 7003)
			svc.Status.LoadBalancer = v1.LoadBalancerStatus{
				Ingress: []v1.LoadBalancerIngress{
					{IP: "10.1.2.3"},
				},
			}
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyLocal
			svc.Spec.HealthCheckNodePort = 345
		}),
	}

	for i := range services {
		fp.addService(services[i])
	}

	result := fp.svcPortMap.Update(fp.serviceChanges)
	if len(fp.svcPortMap) != 8 {
		t.Errorf("expected service map length 2, got %v", fp.svcPortMap)
	}
	for i := range services {
		name := makeNSN(services[i].Namespace, services[i].Name)
		if !result.UpdatedServices.Has(name) {
			t.Errorf("expected updated service for %q", name)
		}
	}
	if len(result.UpdatedServices) != len(services) {
		t.Errorf("expected %d updated services, got %d", len(services), len(result.UpdatedServices))
	}

	// The only-local-loadbalancer ones get added
	healthCheckNodePorts := fp.svcPortMap.HealthCheckNodePorts()
	if len(healthCheckNodePorts) != 1 {
		t.Errorf("expected 1 healthcheck port, got %v", healthCheckNodePorts)
	} else {
		nsn := makeNSN("ns1", "only-local-load-balancer")
		if port, found := healthCheckNodePorts[nsn]; !found || port != 345 {
			t.Errorf("expected healthcheck port [%q]=345: got %v", nsn, healthCheckNodePorts)
		}
	}

	// Remove some stuff
	// oneService is a modification of services[0] with removed first port.
	oneService := makeTestService("ns2", "cluster-ip", func(svc *v1.Service) {
		svc.Spec.Type = v1.ServiceTypeClusterIP
		svc.Spec.ClusterIP = "172.16.55.4"
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "p2", "UDP", 1235, 5321, 0)
	})

	fp.updateService(services[0], oneService)
	fp.deleteService(services[1])
	fp.deleteService(services[2])
	fp.deleteService(services[3])

	result = fp.svcPortMap.Update(fp.serviceChanges)
	if len(fp.svcPortMap) != 1 {
		t.Errorf("expected service map length 1, got %v", fp.svcPortMap)
	}
	if len(result.UpdatedServices) != 4 {
		t.Errorf("expected 4 updated services, got %d", len(result.UpdatedServices))
	}

	healthCheckNodePorts = fp.svcPortMap.HealthCheckNodePorts()
	if len(healthCheckNodePorts) != 0 {
		t.Errorf("expected 0 healthcheck ports, got %v", healthCheckNodePorts)
	}
}

func TestBuildServiceMapServiceUpdate(t *testing.T) {
	fp := newFakeProxier(v1.IPv4Protocol, time.Time{})

	servicev1 := makeTestService("ns1", "svc1", func(svc *v1.Service) {
		svc.Spec.Type = v1.ServiceTypeClusterIP
		svc.Spec.ClusterIP = "172.16.55.4"
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "p1", "UDP", 1234, 4321, 0)
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "p2", "TCP", 1235, 5321, 0)
	})
	servicev2 := makeTestService("ns1", "svc1", func(svc *v1.Service) {
		svc.Spec.Type = v1.ServiceTypeLoadBalancer
		svc.Spec.ClusterIP = "172.16.55.4"
		svc.Spec.LoadBalancerIP = "5.6.7.8"
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "p1", "UDP", 1234, 4321, 7002)
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "p2", "TCP", 1235, 5321, 7003)
		svc.Status.LoadBalancer = v1.LoadBalancerStatus{
			Ingress: []v1.LoadBalancerIngress{
				{IP: "10.1.2.3"},
			},
		}
		svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyLocal
		svc.Spec.HealthCheckNodePort = 345
	})

	fp.addService(servicev1)

	result := fp.svcPortMap.Update(fp.serviceChanges)
	if len(fp.svcPortMap) != 2 {
		t.Errorf("expected service map length 2, got %v", fp.svcPortMap)
	}
	if len(result.UpdatedServices) != 1 {
		t.Errorf("expected 1 updated service, got %d", len(result.UpdatedServices))
	}

	healthCheckNodePorts := fp.svcPortMap.HealthCheckNodePorts()
	if len(healthCheckNodePorts) != 0 {
		t.Errorf("expected healthcheck ports length 0, got %v", healthCheckNodePorts)
	}

	// Change service to load-balancer
	fp.updateService(servicev1, servicev2)
	result = fp.svcPortMap.Update(fp.serviceChanges)
	if len(fp.svcPortMap) != 2 {
		t.Errorf("expected service map length 2, got %v", fp.svcPortMap)
	}
	if len(result.UpdatedServices) != 1 {
		t.Errorf("expected 1 updated service, got %d", len(result.UpdatedServices))
	}

	healthCheckNodePorts = fp.svcPortMap.HealthCheckNodePorts()
	if len(healthCheckNodePorts) != 1 {
		t.Errorf("expected healthcheck ports length 1, got %v", healthCheckNodePorts)
	}

	// No change; make sure the service map stays the same and there are
	// no health-check changes
	fp.updateService(servicev2, servicev2)
	result = fp.svcPortMap.Update(fp.serviceChanges)
	if len(fp.svcPortMap) != 2 {
		t.Errorf("expected service map length 2, got %v", fp.svcPortMap)
	}
	if len(result.UpdatedServices) != 0 {
		t.Errorf("expected 0 updated services, got %d", len(result.UpdatedServices))
	}

	healthCheckNodePorts = fp.svcPortMap.HealthCheckNodePorts()
	if len(healthCheckNodePorts) != 1 {
		t.Errorf("expected healthcheck ports length 1, got %v", healthCheckNodePorts)
	}

	// And back to ClusterIP
	fp.updateService(servicev2, servicev1)
	result = fp.svcPortMap.Update(fp.serviceChanges)
	if len(fp.svcPortMap) != 2 {
		t.Errorf("expected service map length 2, got %v", fp.svcPortMap)
	}
	if len(result.UpdatedServices) != 1 {
		t.Errorf("expected 1 updated service, got %d", len(result.UpdatedServices))
	}

	healthCheckNodePorts = fp.svcPortMap.HealthCheckNodePorts()
	if len(healthCheckNodePorts) != 0 {
		t.Errorf("expected healthcheck ports length 0, got %v", healthCheckNodePorts)
	}
}

func TestServiceCacheLeaks(t *testing.T) {
	fp := newFakeProxier(v1.IPv4Protocol, time.Time{})

	service := makeTestService("ns1", "svc1", func(svc *v1.Service) {
		svc.Spec.Type = v1.ServiceTypeClusterIP
		svc.Spec.ClusterIP = "172.16.55.4"
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "p1", "UDP", 1234, 4321, 0)
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "p2", "TCP", 1235, 5321, 0)
	})
	fp.addService(service)
	if len(fp.serviceChanges.items) != 1 {
		t.Errorf("Found %d items on the cache, 1 expected", len(fp.serviceChanges.items))
	}

	fp.deleteService(service)
	if len(fp.serviceChanges.items) > 0 {
		t.Errorf("Found %d items on the cache, 0 expected", len(fp.serviceChanges.items))
	}
}
