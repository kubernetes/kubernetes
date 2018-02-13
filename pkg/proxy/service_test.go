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
	"fmt"
	"net"
	"reflect"
	"testing"

	"github.com/davecgh/go-spew/spew"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	apiservice "k8s.io/kubernetes/pkg/api/service"
	api "k8s.io/kubernetes/pkg/apis/core"
)

const testHostname = "test-hostname"

// fake implementation for service info.
type fakeServiceInfo struct {
	clusterIP           net.IP
	port                int
	protocol            api.Protocol
	healthCheckNodePort int
}

func (f *fakeServiceInfo) String() string {
	return fmt.Sprintf("%s:%d/%s", f.clusterIP, f.port, f.protocol)
}

func (f *fakeServiceInfo) ClusterIP() string {
	return f.clusterIP.String()
}

func (f *fakeServiceInfo) Protocol() api.Protocol {
	return f.protocol
}

func (f *fakeServiceInfo) HealthCheckNodePort() int {
	return f.healthCheckNodePort
}

func makeTestServiceInfo(clusterIP string, port int, protocol string, healthcheckNodePort int) *fakeServiceInfo {
	info := &fakeServiceInfo{
		clusterIP: net.ParseIP(clusterIP),
		port:      port,
		protocol:  api.Protocol(protocol),
	}
	if healthcheckNodePort != 0 {
		info.healthCheckNodePort = healthcheckNodePort
	}
	return info
}

func newFakeServiceInfo(servicePort *api.ServicePort, service *api.Service) ServicePort {
	info := &fakeServiceInfo{
		clusterIP: net.ParseIP(service.Spec.ClusterIP),
		port:      int(servicePort.Port),
		protocol:  servicePort.Protocol,
	}
	if apiservice.NeedsHealthCheck(service) {
		p := service.Spec.HealthCheckNodePort
		if p != 0 {
			info.healthCheckNodePort = int(p)
		}
	}
	return info
}

func makeTestService(namespace, name string, svcFunc func(*api.Service)) *api.Service {
	svc := &api.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:        name,
			Namespace:   namespace,
			Annotations: map[string]string{},
		},
		Spec:   api.ServiceSpec{},
		Status: api.ServiceStatus{},
	}
	svcFunc(svc)
	return svc
}

func addTestPort(array []api.ServicePort, name string, protocol api.Protocol, port, nodeport int32, targetPort int) []api.ServicePort {
	svcPort := api.ServicePort{
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

func Test_serviceToServiceMap(t *testing.T) {
	testCases := []struct {
		service  *api.Service
		expected map[ServicePortName]*fakeServiceInfo
	}{
		{
			// Case[0]: nothing
			service:  nil,
			expected: map[ServicePortName]*fakeServiceInfo{},
		},
		{
			// Case[1]: headless service
			service: makeTestService("ns2", "headless", func(svc *api.Service) {
				svc.Spec.Type = api.ServiceTypeClusterIP
				svc.Spec.ClusterIP = api.ClusterIPNone
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "rpc", "UDP", 1234, 0, 0)
			}),
			expected: map[ServicePortName]*fakeServiceInfo{},
		},
		{
			// Case[2]: headless service without port
			service: makeTestService("ns2", "headless-without-port", func(svc *api.Service) {
				svc.Spec.Type = api.ServiceTypeClusterIP
				svc.Spec.ClusterIP = api.ClusterIPNone
			}),
			expected: map[ServicePortName]*fakeServiceInfo{},
		},
		{
			// Case[3]: cluster ip service
			service: makeTestService("ns2", "cluster-ip", func(svc *api.Service) {
				svc.Spec.Type = api.ServiceTypeClusterIP
				svc.Spec.ClusterIP = "172.16.55.4"
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "p1", "UDP", 1234, 4321, 0)
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "p2", "UDP", 1235, 5321, 0)
			}),
			expected: map[ServicePortName]*fakeServiceInfo{
				makeServicePortName("ns2", "cluster-ip", "p1"): makeTestServiceInfo("172.16.55.4", 1234, "UDP", 0),
				makeServicePortName("ns2", "cluster-ip", "p2"): makeTestServiceInfo("172.16.55.4", 1235, "UDP", 0),
			},
		},
		{
			// Case[4]: nodeport service
			service: makeTestService("ns2", "node-port", func(svc *api.Service) {
				svc.Spec.Type = api.ServiceTypeNodePort
				svc.Spec.ClusterIP = "172.16.55.10"
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "port1", "UDP", 345, 678, 0)
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "port2", "TCP", 344, 677, 0)
			}),
			expected: map[ServicePortName]*fakeServiceInfo{
				makeServicePortName("ns2", "node-port", "port1"): makeTestServiceInfo("172.16.55.10", 345, "UDP", 0),
				makeServicePortName("ns2", "node-port", "port2"): makeTestServiceInfo("172.16.55.10", 344, "TCP", 0),
			},
		},
		{
			// Case[5]: load balancer service
			service: makeTestService("ns1", "load-balancer", func(svc *api.Service) {
				svc.Spec.Type = api.ServiceTypeLoadBalancer
				svc.Spec.ClusterIP = "172.16.55.11"
				svc.Spec.LoadBalancerIP = "5.6.7.8"
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "port3", "UDP", 8675, 30061, 7000)
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "port4", "UDP", 8676, 30062, 7001)
				svc.Status.LoadBalancer = api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{
						{IP: "10.1.2.4"},
					},
				}
			}),
			expected: map[ServicePortName]*fakeServiceInfo{
				makeServicePortName("ns1", "load-balancer", "port3"): makeTestServiceInfo("172.16.55.11", 8675, "UDP", 0),
				makeServicePortName("ns1", "load-balancer", "port4"): makeTestServiceInfo("172.16.55.11", 8676, "UDP", 0),
			},
		},
		{
			// Case[6]: load balancer service with only local traffic policy
			service: makeTestService("ns1", "only-local-load-balancer", func(svc *api.Service) {
				svc.Spec.Type = api.ServiceTypeLoadBalancer
				svc.Spec.ClusterIP = "172.16.55.12"
				svc.Spec.LoadBalancerIP = "5.6.7.8"
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "portx", "UDP", 8677, 30063, 7002)
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "porty", "UDP", 8678, 30064, 7003)
				svc.Status.LoadBalancer = api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{
						{IP: "10.1.2.3"},
					},
				}
				svc.Spec.ExternalTrafficPolicy = api.ServiceExternalTrafficPolicyTypeLocal
				svc.Spec.HealthCheckNodePort = 345
			}),
			expected: map[ServicePortName]*fakeServiceInfo{
				makeServicePortName("ns1", "only-local-load-balancer", "portx"): makeTestServiceInfo("172.16.55.12", 8677, "UDP", 345),
				makeServicePortName("ns1", "only-local-load-balancer", "porty"): makeTestServiceInfo("172.16.55.12", 8678, "UDP", 345),
			},
		},
		{
			// Case[7]: external name service
			service: makeTestService("ns2", "external-name", func(svc *api.Service) {
				svc.Spec.Type = api.ServiceTypeExternalName
				svc.Spec.ClusterIP = "172.16.55.4" // Should be ignored
				svc.Spec.ExternalName = "foo2.bar.com"
				svc.Spec.Ports = addTestPort(svc.Spec.Ports, "portz", "UDP", 1235, 5321, 0)
			}),
			expected: map[ServicePortName]*fakeServiceInfo{},
		},
	}

	for tci, tc := range testCases {
		// outputs
		newServices := serviceToServiceMap(tc.service, newFakeServiceInfo)

		if len(newServices) != len(tc.expected) {
			t.Errorf("[%d] expected %d new, got %d: %v", tci, len(tc.expected), len(newServices), spew.Sdump(newServices))
		}
		for x := range tc.expected {
			svc := newServices[x].(*fakeServiceInfo)
			if !reflect.DeepEqual(svc, tc.expected[x]) {
				t.Errorf("[%d] expected new[%v]to be %v, got %v", tci, x, tc.expected[x], *svc)
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
		serviceChanges:   NewServiceChangeTracker(),
		endpointsMap:     make(EndpointsMap),
		endpointsChanges: NewEndpointChangeTracker(testHostname),
	}
}

func makeServiceMap(fake *FakeProxier, allServices ...*api.Service) {
	for i := range allServices {
		fake.addService(allServices[i])
	}
}

func (fake *FakeProxier) addService(service *api.Service) {
	fake.serviceChanges.Update(nil, service, makeServicePort)
}

func (fake *FakeProxier) updateService(oldService *api.Service, service *api.Service) {
	fake.serviceChanges.Update(oldService, service, makeServicePort)
}

func (fake *FakeProxier) deleteService(service *api.Service) {
	fake.serviceChanges.Update(service, nil, makeServicePort)
}

func makeServicePort(port *api.ServicePort, service *api.Service) ServicePort {
	info := &fakeServiceInfo{
		clusterIP: net.ParseIP(service.Spec.ClusterIP),
		port:      int(port.Port),
		protocol:  port.Protocol,
	}
	if apiservice.NeedsHealthCheck(service) {
		p := service.Spec.HealthCheckNodePort
		if p != 0 {
			info.healthCheckNodePort = int(p)
		}
	}
	return info
}

func TestUpdateServiceMapHeadless(t *testing.T) {
	fp := newFakeProxier()

	makeServiceMap(fp,
		makeTestService("ns2", "headless", func(svc *api.Service) {
			svc.Spec.Type = api.ServiceTypeClusterIP
			svc.Spec.ClusterIP = api.ClusterIPNone
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "rpc", "UDP", 1234, 0, 0)
		}),
		makeTestService("ns2", "headless-without-port", func(svc *api.Service) {
			svc.Spec.Type = api.ServiceTypeClusterIP
			svc.Spec.ClusterIP = api.ClusterIPNone
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
		makeTestService("ns2", "external-name", func(svc *api.Service) {
			svc.Spec.Type = api.ServiceTypeExternalName
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

	services := []*api.Service{
		makeTestService("ns2", "cluster-ip", func(svc *api.Service) {
			svc.Spec.Type = api.ServiceTypeClusterIP
			svc.Spec.ClusterIP = "172.16.55.4"
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "port1", "UDP", 1234, 4321, 0)
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "port2", "UDP", 1235, 5321, 0)
		}),
		makeTestService("ns2", "node-port", func(svc *api.Service) {
			svc.Spec.Type = api.ServiceTypeNodePort
			svc.Spec.ClusterIP = "172.16.55.10"
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "port1", "UDP", 345, 678, 0)
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "port2", "TCP", 344, 677, 0)
		}),
		makeTestService("ns1", "load-balancer", func(svc *api.Service) {
			svc.Spec.Type = api.ServiceTypeLoadBalancer
			svc.Spec.ClusterIP = "172.16.55.11"
			svc.Spec.LoadBalancerIP = "5.6.7.8"
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "foobar", "UDP", 8675, 30061, 7000)
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "baz", "UDP", 8676, 30062, 7001)
			svc.Status.LoadBalancer = api.LoadBalancerStatus{
				Ingress: []api.LoadBalancerIngress{
					{IP: "10.1.2.4"},
				},
			}
		}),
		makeTestService("ns1", "only-local-load-balancer", func(svc *api.Service) {
			svc.Spec.Type = api.ServiceTypeLoadBalancer
			svc.Spec.ClusterIP = "172.16.55.12"
			svc.Spec.LoadBalancerIP = "5.6.7.8"
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "foobar2", "UDP", 8677, 30063, 7002)
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "baz", "UDP", 8678, 30064, 7003)
			svc.Status.LoadBalancer = api.LoadBalancerStatus{
				Ingress: []api.LoadBalancerIngress{
					{IP: "10.1.2.3"},
				},
			}
			svc.Spec.ExternalTrafficPolicy = api.ServiceExternalTrafficPolicyTypeLocal
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
	oneService := makeTestService("ns2", "cluster-ip", func(svc *api.Service) {
		svc.Spec.Type = api.ServiceTypeClusterIP
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

	servicev1 := makeTestService("ns1", "svc1", func(svc *api.Service) {
		svc.Spec.Type = api.ServiceTypeClusterIP
		svc.Spec.ClusterIP = "172.16.55.4"
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "p1", "UDP", 1234, 4321, 0)
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "p2", "TCP", 1235, 5321, 0)
	})
	servicev2 := makeTestService("ns1", "svc1", func(svc *api.Service) {
		svc.Spec.Type = api.ServiceTypeLoadBalancer
		svc.Spec.ClusterIP = "172.16.55.4"
		svc.Spec.LoadBalancerIP = "5.6.7.8"
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "p1", "UDP", 1234, 4321, 7002)
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "p2", "TCP", 1235, 5321, 7003)
		svc.Status.LoadBalancer = api.LoadBalancerStatus{
			Ingress: []api.LoadBalancerIngress{
				{IP: "10.1.2.3"},
			},
		}
		svc.Spec.ExternalTrafficPolicy = api.ServiceExternalTrafficPolicyTypeLocal
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
