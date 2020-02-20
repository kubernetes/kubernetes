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

	"github.com/davecgh/go-spew/spew"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
)

const testHostname = "test-hostname"

func makeTestServiceInfo(clusterIP string, port int, protocol string, healthcheckNodePort int, svcInfoFuncs ...func(*BaseServiceInfo)) *BaseServiceInfo {
	info := &BaseServiceInfo{
		clusterIP: net.ParseIP(clusterIP),
		port:      port,
		protocol:  v1.Protocol(protocol),
	}
	if healthcheckNodePort != 0 {
		info.healthCheckNodePort = healthcheckNodePort
	}
	for _, svcInfoFunc := range svcInfoFuncs {
		svcInfoFunc(info)
	}
	return info
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
		TargetPort: intstr.FromInt(targetPort),
	}
	return append(array, svcPort)
}

func makeNSN(namespace, name string) types.NamespacedName {
	return types.NamespacedName{Namespace: namespace, Name: name}
}

func makeServicePortName(ns, name, port string) ServicePortName {
	return ServicePortName{
		NamespacedName: makeNSN(ns, name),
		Port:           port,
	}
}

func TestServiceToServiceMap(t *testing.T) {
	svcTracker := NewServiceChangeTracker(nil, nil, nil)

	trueVal := true
	falseVal := false
	testClusterIPv4 := "10.0.0.1"
	testExternalIPv4 := "8.8.8.8"
	testSourceRangeIPv4 := "0.0.0.0/1"
	testClusterIPv6 := "2001:db8:85a3:0:0:8a2e:370:7334"
	testExternalIPv6 := "2001:db8:85a3:0:0:8a2e:370:7335"
	testSourceRangeIPv6 := "2001:db8::/32"

	testCases := []struct {
		desc       string
		service    *v1.Service
		expected   map[ServicePortName]*BaseServiceInfo
		isIPv6Mode *bool
	}{
		{
			desc:     "nothing",
			service:  nil,
			expected: map[ServicePortName]*BaseServiceInfo{},
		},
		{
			desc: "headless service",
			service: makeTestService("ns2", "headless", func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeClusterIP
				svc.Spec.ClusterIP = v1.ClusterIPNone
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "rpc", "UDP", 1234, 0, 0)
			}),
			expected: map[ServicePortName]*BaseServiceInfo{},
		},
		{
			desc: "headless sctp service",
			service: makeTestService("ns2", "headless", func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeClusterIP
				svc.Spec.ClusterIP = v1.ClusterIPNone
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "sip", "SCTP", 7777, 0, 0)
			}),
			expected: map[ServicePortName]*BaseServiceInfo{},
		},
		{
			desc: "headless service without port",
			service: makeTestService("ns2", "headless-without-port", func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeClusterIP
				svc.Spec.ClusterIP = v1.ClusterIPNone
			}),
			expected: map[ServicePortName]*BaseServiceInfo{},
		},
		{
			desc: "cluster ip service",
			service: makeTestService("ns2", "cluster-ip", func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeClusterIP
				svc.Spec.ClusterIP = "172.16.55.4"
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "p1", "UDP", 1234, 4321, 0)
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "p2", "UDP", 1235, 5321, 0)
			}),
			expected: map[ServicePortName]*BaseServiceInfo{
				makeServicePortName("ns2", "cluster-ip", "p1"): makeTestServiceInfo("172.16.55.4", 1234, "UDP", 0),
				makeServicePortName("ns2", "cluster-ip", "p2"): makeTestServiceInfo("172.16.55.4", 1235, "UDP", 0),
			},
		},
		{
			desc: "nodeport service",
			service: makeTestService("ns2", "node-port", func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeNodePort
				svc.Spec.ClusterIP = "172.16.55.10"
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "port1", "UDP", 345, 678, 0)
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "port2", "TCP", 344, 677, 0)
			}),
			expected: map[ServicePortName]*BaseServiceInfo{
				makeServicePortName("ns2", "node-port", "port1"): makeTestServiceInfo("172.16.55.10", 345, "UDP", 0),
				makeServicePortName("ns2", "node-port", "port2"): makeTestServiceInfo("172.16.55.10", 344, "TCP", 0),
			},
		},
		{
			desc: "load balancer service",
			service: makeTestService("ns1", "load-balancer", func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeLoadBalancer
				svc.Spec.ClusterIP = "172.16.55.11"
				svc.Spec.LoadBalancerIP = "5.6.7.8"
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "port3", "UDP", 8675, 30061, 7000)
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "port4", "UDP", 8676, 30062, 7001)
				svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{IP: "10.1.2.4"}}
			}),
			expected: map[ServicePortName]*BaseServiceInfo{
				makeServicePortName("ns1", "load-balancer", "port3"): makeTestServiceInfo("172.16.55.11", 8675, "UDP", 0, func(info *BaseServiceInfo) {
					info.loadBalancerStatus.Ingress = []v1.LoadBalancerIngress{{IP: "10.1.2.4"}}
				}),
				makeServicePortName("ns1", "load-balancer", "port4"): makeTestServiceInfo("172.16.55.11", 8676, "UDP", 0, func(info *BaseServiceInfo) {
					info.loadBalancerStatus.Ingress = []v1.LoadBalancerIngress{{IP: "10.1.2.4"}}
				}),
			},
		},
		{
			desc: "load balancer service with only local traffic policy",
			service: makeTestService("ns1", "only-local-load-balancer", func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeLoadBalancer
				svc.Spec.ClusterIP = "172.16.55.12"
				svc.Spec.LoadBalancerIP = "5.6.7.8"
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "portx", "UDP", 8677, 30063, 7002)
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "porty", "UDP", 8678, 30064, 7003)
				svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{IP: "10.1.2.3"}}
				svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
				svc.Spec.HealthCheckNodePort = 345
			}),
			expected: map[ServicePortName]*BaseServiceInfo{
				makeServicePortName("ns1", "only-local-load-balancer", "portx"): makeTestServiceInfo("172.16.55.12", 8677, "UDP", 345, func(info *BaseServiceInfo) {
					info.loadBalancerStatus.Ingress = []v1.LoadBalancerIngress{{IP: "10.1.2.3"}}
				}),
				makeServicePortName("ns1", "only-local-load-balancer", "porty"): makeTestServiceInfo("172.16.55.12", 8678, "UDP", 345, func(info *BaseServiceInfo) {
					info.loadBalancerStatus.Ingress = []v1.LoadBalancerIngress{{IP: "10.1.2.3"}}
				}),
			},
		},
		{
			desc: "external name service",
			service: makeTestService("ns2", "external-name", func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeExternalName
				svc.Spec.ClusterIP = "172.16.55.4" // Should be ignored
				svc.Spec.ExternalName = "foo2.bar.com"
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "portz", "UDP", 1235, 5321, 0)
			}),
			expected: map[ServicePortName]*BaseServiceInfo{},
		},
		{
			desc: "service with ipv6 clusterIP under ipv4 mode, service should be filtered",
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
			isIPv6Mode: &falseVal,
		},
		{
			desc: "service with ipv4 clusterIP under ipv6 mode, service should be filtered",
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
			isIPv6Mode: &trueVal,
		},
		{
			desc: "service with ipv4 configurations under ipv4 mode",
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
			expected: map[ServicePortName]*BaseServiceInfo{
				makeServicePortName("test", "validIPv4", "testPort"): makeTestServiceInfo(testClusterIPv4, 12345, "TCP", 0, func(info *BaseServiceInfo) {
					info.externalIPs = []string{testExternalIPv4}
					info.loadBalancerSourceRanges = []string{testSourceRangeIPv4}
					info.loadBalancerStatus.Ingress = []v1.LoadBalancerIngress{{IP: testExternalIPv4}}
				}),
			},
			isIPv6Mode: &falseVal,
		},
		{
			desc: "service with ipv6 configurations under ipv6 mode",
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
			expected: map[ServicePortName]*BaseServiceInfo{
				makeServicePortName("test", "validIPv6", "testPort"): makeTestServiceInfo(testClusterIPv6, 12345, "TCP", 0, func(info *BaseServiceInfo) {
					info.externalIPs = []string{testExternalIPv6}
					info.loadBalancerSourceRanges = []string{testSourceRangeIPv6}
					info.loadBalancerStatus.Ingress = []v1.LoadBalancerIngress{{IP: testExternalIPv6}}
				}),
			},
			isIPv6Mode: &trueVal,
		},
		{
			desc: "service with both ipv4 and ipv6 configurations under ipv4 mode, ipv6 fields should be filtered",
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
			expected: map[ServicePortName]*BaseServiceInfo{
				makeServicePortName("test", "filterIPv6InIPV4Mode", "testPort"): makeTestServiceInfo(testClusterIPv4, 12345, "TCP", 0, func(info *BaseServiceInfo) {
					info.externalIPs = []string{testExternalIPv4}
					info.loadBalancerSourceRanges = []string{testSourceRangeIPv4}
					info.loadBalancerStatus.Ingress = []v1.LoadBalancerIngress{{IP: testExternalIPv4}}
				}),
			},
			isIPv6Mode: &falseVal,
		},
		{
			desc: "service with both ipv4 and ipv6 configurations under ipv6 mode, ipv4 fields should be filtered",
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
			expected: map[ServicePortName]*BaseServiceInfo{
				makeServicePortName("test", "filterIPv4InIPV6Mode", "testPort"): makeTestServiceInfo(testClusterIPv6, 12345, "TCP", 0, func(info *BaseServiceInfo) {
					info.externalIPs = []string{testExternalIPv6}
					info.loadBalancerSourceRanges = []string{testSourceRangeIPv6}
					info.loadBalancerStatus.Ingress = []v1.LoadBalancerIngress{{IP: testExternalIPv6}}
				}),
			},
			isIPv6Mode: &trueVal,
		},
	}

	for _, tc := range testCases {
		svcTracker.isIPv6Mode = tc.isIPv6Mode
		// outputs
		newServices := svcTracker.serviceToServiceMap(tc.service)

		if len(newServices) != len(tc.expected) {
			t.Errorf("[%s] expected %d new, got %d: %v", tc.desc, len(tc.expected), len(newServices), spew.Sdump(newServices))
		}
		for svcKey, expectedInfo := range tc.expected {
			svcInfo := newServices[svcKey].(*BaseServiceInfo)
			if !svcInfo.clusterIP.Equal(expectedInfo.clusterIP) ||
				svcInfo.port != expectedInfo.port ||
				svcInfo.protocol != expectedInfo.protocol ||
				svcInfo.healthCheckNodePort != expectedInfo.healthCheckNodePort ||
				!sets.NewString(svcInfo.externalIPs...).Equal(sets.NewString(expectedInfo.externalIPs...)) ||
				!sets.NewString(svcInfo.loadBalancerSourceRanges...).Equal(sets.NewString(expectedInfo.loadBalancerSourceRanges...)) ||
				!reflect.DeepEqual(svcInfo.loadBalancerStatus, expectedInfo.loadBalancerStatus) {
				t.Errorf("[%s] expected new[%v]to be %v, got %v", tc.desc, svcKey, expectedInfo, *svcInfo)
			}
		}
	}
}

type FakeProxier struct {
	endpointsChanges *EndpointChangeTracker
	serviceChanges   *ServiceChangeTracker
	serviceMap       ServiceMap
	endpointsMap     EndpointsMap
	hostname         string
}

func newFakeProxier() *FakeProxier {
	return &FakeProxier{
		serviceMap:       make(ServiceMap),
		serviceChanges:   NewServiceChangeTracker(nil, nil, nil),
		endpointsMap:     make(EndpointsMap),
		endpointsChanges: NewEndpointChangeTracker(testHostname, nil, nil, nil, false),
	}
}

func makeServiceMap(fake *FakeProxier, allServices ...*v1.Service) {
	for i := range allServices {
		fake.addService(allServices[i])
	}
}

func (fake *FakeProxier) addService(service *v1.Service) {
	fake.serviceChanges.Update(nil, service)
}

func (fake *FakeProxier) updateService(oldService *v1.Service, service *v1.Service) {
	fake.serviceChanges.Update(oldService, service)
}

func (fake *FakeProxier) deleteService(service *v1.Service) {
	fake.serviceChanges.Update(service, nil)
}

func TestUpdateServiceMapHeadless(t *testing.T) {
	fp := newFakeProxier()

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
	result := UpdateServiceMap(fp.serviceMap, fp.serviceChanges)
	if len(fp.serviceMap) != 0 {
		t.Errorf("expected service map length 0, got %d", len(fp.serviceMap))
	}

	// No proxied services, so no healthchecks
	if len(result.HCServiceNodePorts) != 0 {
		t.Errorf("expected healthcheck ports length 0, got %d", len(result.HCServiceNodePorts))
	}

	if len(result.UDPStaleClusterIP) != 0 {
		t.Errorf("expected stale UDP services length 0, got %d", len(result.UDPStaleClusterIP))
	}
}

func TestUpdateServiceTypeExternalName(t *testing.T) {
	fp := newFakeProxier()

	makeServiceMap(fp,
		makeTestService("ns2", "external-name", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeExternalName
			svc.Spec.ClusterIP = "172.16.55.4" // Should be ignored
			svc.Spec.ExternalName = "foo2.bar.com"
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "blah", "UDP", 1235, 5321, 0)
		}),
	)

	result := UpdateServiceMap(fp.serviceMap, fp.serviceChanges)
	if len(fp.serviceMap) != 0 {
		t.Errorf("expected service map length 0, got %v", fp.serviceMap)
	}
	// No proxied services, so no healthchecks
	if len(result.HCServiceNodePorts) != 0 {
		t.Errorf("expected healthcheck ports length 0, got %v", result.HCServiceNodePorts)
	}
	if len(result.UDPStaleClusterIP) != 0 {
		t.Errorf("expected stale UDP services length 0, got %v", result.UDPStaleClusterIP)
	}
}

func TestBuildServiceMapAddRemove(t *testing.T) {
	fp := newFakeProxier()

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
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
			svc.Spec.HealthCheckNodePort = 345
		}),
	}

	for i := range services {
		fp.addService(services[i])
	}
	result := UpdateServiceMap(fp.serviceMap, fp.serviceChanges)
	if len(fp.serviceMap) != 8 {
		t.Errorf("expected service map length 2, got %v", fp.serviceMap)
	}

	// The only-local-loadbalancer ones get added
	if len(result.HCServiceNodePorts) != 1 {
		t.Errorf("expected 1 healthcheck port, got %v", result.HCServiceNodePorts)
	} else {
		nsn := makeNSN("ns1", "only-local-load-balancer")
		if port, found := result.HCServiceNodePorts[nsn]; !found || port != 345 {
			t.Errorf("expected healthcheck port [%q]=345: got %v", nsn, result.HCServiceNodePorts)
		}
	}

	if len(result.UDPStaleClusterIP) != 0 {
		// Services only added, so nothing stale yet
		t.Errorf("expected stale UDP services length 0, got %d", len(result.UDPStaleClusterIP))
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

	result = UpdateServiceMap(fp.serviceMap, fp.serviceChanges)
	if len(fp.serviceMap) != 1 {
		t.Errorf("expected service map length 1, got %v", fp.serviceMap)
	}

	if len(result.HCServiceNodePorts) != 0 {
		t.Errorf("expected 0 healthcheck ports, got %v", result.HCServiceNodePorts)
	}

	// All services but one were deleted. While you'd expect only the ClusterIPs
	// from the three deleted services here, we still have the ClusterIP for
	// the not-deleted service, because one of it's ServicePorts was deleted.
	expectedStaleUDPServices := []string{"172.16.55.10", "172.16.55.4", "172.16.55.11", "172.16.55.12"}
	if len(result.UDPStaleClusterIP) != len(expectedStaleUDPServices) {
		t.Errorf("expected stale UDP services length %d, got %v", len(expectedStaleUDPServices), result.UDPStaleClusterIP.UnsortedList())
	}
	for _, ip := range expectedStaleUDPServices {
		if !result.UDPStaleClusterIP.Has(ip) {
			t.Errorf("expected stale UDP service service %s", ip)
		}
	}
}

func TestBuildServiceMapServiceUpdate(t *testing.T) {
	fp := newFakeProxier()

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
		svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
		svc.Spec.HealthCheckNodePort = 345
	})

	fp.addService(servicev1)

	result := UpdateServiceMap(fp.serviceMap, fp.serviceChanges)
	if len(fp.serviceMap) != 2 {
		t.Errorf("expected service map length 2, got %v", fp.serviceMap)
	}
	if len(result.HCServiceNodePorts) != 0 {
		t.Errorf("expected healthcheck ports length 0, got %v", result.HCServiceNodePorts)
	}
	if len(result.UDPStaleClusterIP) != 0 {
		// Services only added, so nothing stale yet
		t.Errorf("expected stale UDP services length 0, got %d", len(result.UDPStaleClusterIP))
	}

	// Change service to load-balancer
	fp.updateService(servicev1, servicev2)
	result = UpdateServiceMap(fp.serviceMap, fp.serviceChanges)
	if len(fp.serviceMap) != 2 {
		t.Errorf("expected service map length 2, got %v", fp.serviceMap)
	}
	if len(result.HCServiceNodePorts) != 1 {
		t.Errorf("expected healthcheck ports length 1, got %v", result.HCServiceNodePorts)
	}
	if len(result.UDPStaleClusterIP) != 0 {
		t.Errorf("expected stale UDP services length 0, got %v", result.UDPStaleClusterIP.UnsortedList())
	}

	// No change; make sure the service map stays the same and there are
	// no health-check changes
	fp.updateService(servicev2, servicev2)
	result = UpdateServiceMap(fp.serviceMap, fp.serviceChanges)
	if len(fp.serviceMap) != 2 {
		t.Errorf("expected service map length 2, got %v", fp.serviceMap)
	}
	if len(result.HCServiceNodePorts) != 1 {
		t.Errorf("expected healthcheck ports length 1, got %v", result.HCServiceNodePorts)
	}
	if len(result.UDPStaleClusterIP) != 0 {
		t.Errorf("expected stale UDP services length 0, got %v", result.UDPStaleClusterIP.UnsortedList())
	}

	// And back to ClusterIP
	fp.updateService(servicev2, servicev1)
	result = UpdateServiceMap(fp.serviceMap, fp.serviceChanges)
	if len(fp.serviceMap) != 2 {
		t.Errorf("expected service map length 2, got %v", fp.serviceMap)
	}
	if len(result.HCServiceNodePorts) != 0 {
		t.Errorf("expected healthcheck ports length 0, got %v", result.HCServiceNodePorts)
	}
	if len(result.UDPStaleClusterIP) != 0 {
		// Services only added, so nothing stale yet
		t.Errorf("expected stale UDP services length 0, got %d", len(result.UDPStaleClusterIP))
	}
}
