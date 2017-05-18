// +build linux

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

package ipvs

import (
	"net"
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/util/exec"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"

	proxyutil "k8s.io/kubernetes/pkg/proxy/util"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	iptablestest "k8s.io/kubernetes/pkg/util/iptables/testing"
	utilipvs "k8s.io/kubernetes/pkg/util/ipvs"
	ipvstest "k8s.io/kubernetes/pkg/util/ipvs/testing"
)

const testHostname = "test-hostname"

type fakeIPGetter struct {
	nodeIPs []net.IP
}

func (f *fakeIPGetter) NodeIPs() ([]net.IP, error) {
	return f.nodeIPs, nil
}

type fakeHealthChecker struct {
	services  map[types.NamespacedName]uint16
	Endpoints map[types.NamespacedName]int
}

func newFakeHealthChecker() *fakeHealthChecker {
	return &fakeHealthChecker{
		services:  map[types.NamespacedName]uint16{},
		Endpoints: map[types.NamespacedName]int{},
	}
}

// fakePortOpener implements portOpener.
type fakePortOpener struct {
	openPorts []*proxyutil.LocalPort
}

// OpenLocalPort fakes out the listen() and bind() used by syncProxyRules
// to lock a local port.
func (f *fakePortOpener) OpenLocalPort(lp *proxyutil.LocalPort) (proxyutil.Closeable, error) {
	f.openPorts = append(f.openPorts, lp)
	return nil, nil
}

func (fake *fakeHealthChecker) SyncServices(newServices map[types.NamespacedName]uint16) error {
	fake.services = newServices
	return nil
}

func (fake *fakeHealthChecker) SyncEndpoints(newEndpoints map[types.NamespacedName]int) error {
	fake.Endpoints = newEndpoints
	return nil
}

func NewFakeProxier(ipt utiliptables.Interface, ipvs utilipvs.Interface, nodeIPs []net.IP) *Proxier {
	return &Proxier{
		exec:             &exec.FakeExec{},
		serviceMap:       make(proxyutil.ProxyServiceMap),
		serviceChanges:   proxyutil.NewServiceChangeMap(),
		endpointsMap:     make(proxyutil.ProxyEndpointsMap),
		endpointsChanges: proxyutil.NewEndpointsChangeMap(),
		iptables:         ipt,
		ipvs:             ipvs,
		clusterCIDR:      "10.0.0.0/24",
		hostname:         testHostname,
		portsMap:         make(map[proxyutil.LocalPort]proxyutil.Closeable),
		portMapper:       &fakePortOpener{[]*proxyutil.LocalPort{}},
		healthChecker:    newFakeHealthChecker(),
		ipvsScheduler:    utilipvs.DefaultIPVSScheduler,
		ipGetter:         &fakeIPGetter{nodeIPs: nodeIPs},
	}
}

func makeNSN(namespace, name string) types.NamespacedName {
	return types.NamespacedName{Namespace: namespace, Name: name}
}

func makeServiceMap(proxier *Proxier, allServices ...*api.Service) {
	for i := range allServices {
		proxier.OnServiceAdd(allServices[i])
	}

	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	proxier.servicesSynced = true
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

func makeEndpointsMap(proxier *Proxier, allEndpoints ...*api.Endpoints) {
	for i := range allEndpoints {
		proxier.OnEndpointsAdd(allEndpoints[i])
	}

	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	proxier.endpointsSynced = true
}

func makeTestEndpoints(namespace, name string, eptFunc func(*api.Endpoints)) *api.Endpoints {
	ept := &api.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
	}
	eptFunc(ept)
	return ept
}

func TestNodePort(t *testing.T) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	nodeIP := net.ParseIP("100.101.102.103")
	fp := NewFakeProxier(ipt, ipvs, []net.IP{nodeIP})
	svcIP := "10.20.30.41"
	svcPort := 80
	svcNodePort := 3001
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
	}

	makeServiceMap(fp,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *api.Service) {
			svc.Spec.Type = "NodePort"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.Ports = []api.ServicePort{{
				Name:     svcPortName.Port,
				Port:     int32(svcPort),
				Protocol: api.ProtocolTCP,
				NodePort: int32(svcNodePort),
			}}
		}),
	)
	epIP := "10.180.0.1"
	makeEndpointsMap(fp,
		makeTestEndpoints(svcPortName.Namespace, svcPortName.Name, func(ept *api.Endpoints) {
			ept.Subsets = []api.EndpointSubset{{
				Addresses: []api.EndpointAddress{{
					IP: epIP,
				}},
				Ports: []api.EndpointPort{{
					Name: svcPortName.Port,
					Port: int32(svcPort),
				}},
			}}
		}),
	)

	fp.syncProxyRules(syncReasonForce)

	// Tcheck ipvs service and destinations
	services, err := ipvs.GetServices()
	if err != nil {
		t.Errorf("Failed to get ipvs services, err: %v")
	}
	if len(services) != 2 {
		t.Errorf("Expect 2 ipvs services, got %d", len(services))
	}
	found := false
	for _, svc := range services {
		if svc.Address.Equal(nodeIP) && svc.Port == uint16(svcNodePort) && svc.Protocol == string(api.ProtocolTCP) {
			found = true
			destinations, err := ipvs.GetDestinations(svc)
			if err != nil {
				t.Errorf("Failed to get ipvs destinations, err: %v", err)
			}
			for _, dest := range destinations {
				if dest.Address.To4().String() != epIP || dest.Port != uint16(svcPort) {
					t.Errorf("service Endpoint mismatch ipvs service destination")
				}
			}
			break
		}
	}
	if !found {
		t.Errorf("Expect node port type service, got none")
	}
}

func TestNodePortNoEndpoint(t *testing.T) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	nodeIP := net.ParseIP("100.101.102.103")
	fp := NewFakeProxier(ipt, ipvs, []net.IP{nodeIP})
	svcIP := "10.20.30.41"
	svcPort := 80
	svcNodePort := 3001
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
	}

	makeServiceMap(fp,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *api.Service) {
			svc.Spec.Type = "NodePort"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.Ports = []api.ServicePort{{
				Name:     svcPortName.Port,
				Port:     int32(svcPort),
				Protocol: api.ProtocolTCP,
				NodePort: int32(svcNodePort),
			}}
		}),
	)
	makeEndpointsMap(fp)

	fp.syncProxyRules(syncReasonForce)

	// Tcheck ipvs service and destinations
	services, err := ipvs.GetServices()
	if err != nil {
		t.Errorf("Failed to get ipvs services, err: %v")
	}
	if len(services) != 2 {
		t.Errorf("Expect 2 ipvs services, got %d", len(services))
	}
	found := false
	for _, svc := range services {
		if svc.Address.Equal(nodeIP) && svc.Port == uint16(svcNodePort) && svc.Protocol == string(api.ProtocolTCP) {
			found = true
			destinations, _ := ipvs.GetDestinations(svc)
			if len(destinations) != 0 {
				t.Errorf("Unexpected %d destinations, expect 0 destinations")
			}
			break
		}
	}
	if !found {
		t.Errorf("Expect node port type service, got none")
	}
}

func TestClusterIPNoEndpoint(t *testing.T) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	fp := NewFakeProxier(ipt, ipvs, nil)
	svcIP := "10.20.30.41"
	svcPort := 80
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
	}

	makeServiceMap(fp,
		makeTestService(svcPortName.Namespace, svcPortName.Namespace, func(svc *api.Service) {
			svc.Spec.ClusterIP = svcIP
			svc.Spec.Ports = []api.ServicePort{{
				Name:     svcPortName.Port,
				Port:     int32(svcPort),
				Protocol: api.ProtocolTCP,
			}}
		}),
	)
	makeEndpointsMap(fp)
	fp.syncProxyRules(syncReasonForce)

	// check ipvs service and destinations
	services, err := ipvs.GetServices()
	if err != nil {
		t.Errorf("Failed to get ipvs services, err: %v")
	}
	if len(services) != 1 {
		t.Errorf("Expect 1 ipvs services, got %d", len(services))
	} else {
		if services[0].Address.To4().String() != svcIP || services[0].Port != uint16(svcPort) && services[0].Protocol == string(api.ProtocolTCP) {
			t.Errorf("Unexpected mismatch service")
		} else {
			destinations, _ := ipvs.GetDestinations(services[0])
			if len(destinations) != 0 {
				t.Errorf("Unexpected %d destinations, expect 0 destinations")
			}
		}
	}
}

func TestClusterIP(t *testing.T) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	fp := NewFakeProxier(ipt, ipvs, nil)
	svcIP := "10.20.30.41"
	svcPort := 80
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
	}

	makeServiceMap(fp,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *api.Service) {
			svc.Spec.ClusterIP = svcIP
			svc.Spec.Ports = []api.ServicePort{{
				Name:     svcPortName.Port,
				Port:     int32(svcPort),
				Protocol: api.ProtocolTCP,
			}}
		}),
	)

	epIP := "10.180.0.1"
	makeEndpointsMap(fp,
		makeTestEndpoints(svcPortName.Namespace, svcPortName.Name, func(ept *api.Endpoints) {
			ept.Subsets = []api.EndpointSubset{{
				Addresses: []api.EndpointAddress{{
					IP: epIP,
				}},
				Ports: []api.EndpointPort{{
					Name: svcPortName.Port,
					Port: int32(svcPort),
				}},
			}}
		}),
	)

	fp.syncProxyRules(syncReasonForce)

	// check ipvs service and destinations
	services, err := ipvs.GetServices()
	if err != nil {
		t.Errorf("Failed to get ipvs services, err: %v")
	}
	if len(services) != 1 {
		t.Errorf("Expect 1 ipvs services, got %d", len(services))
	} else {
		if services[0].Address.To4().String() != svcIP || services[0].Port != uint16(svcPort) && services[0].Protocol == string(api.ProtocolTCP) {
			t.Errorf("Unexpected mismatch service")
		} else {
			destinations, _ := ipvs.GetDestinations(services[0])
			if len(destinations) != 1 {
				t.Errorf("Unexpected %d destinations, expect 0 destinations")
			} else if destinations[0].Address.To4().String() != epIP || destinations[0].Port != uint16(svcPort) {
				t.Errorf("Unexpected mismatch destinations")
			}
		}
	}
}

func TestExternalIPsNoEndpoint(t *testing.T) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	fp := NewFakeProxier(ipt, ipvs, nil)
	svcIP := "10.20.30.41"
	svcPort := 80
	svcExternalIPs := "50.60.70.81"
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
	}

	makeServiceMap(fp,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *api.Service) {
			svc.Spec.Type = "ClusterIP"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.ExternalIPs = []string{svcExternalIPs}
			svc.Spec.Ports = []api.ServicePort{{
				Name:       svcPortName.Port,
				Port:       int32(svcPort),
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(svcPort),
			}}
		}),
	)

	makeEndpointsMap(fp)

	fp.syncProxyRules(syncReasonForce)

	// check ipvs service and destinations
	services, err := ipvs.GetServices()
	if err != nil {
		t.Errorf("Failed to get ipvs services, err: %v")
	}
	if len(services) != 2 {
		t.Errorf("Expect 2 ipvs services, got %d", len(services))
	}
	found := false
	for _, svc := range services {
		if svc.Address.To4().String() == svcExternalIPs && svc.Port == uint16(svcPort) && svc.Protocol == string(api.ProtocolTCP) {
			found = true
			destinations, _ := ipvs.GetDestinations(svc)
			if len(destinations) != 0 {
				t.Errorf("Unexpected %d destinations, expect 0 destinations")
			}
			break
		}
	}
	if !found {
		t.Errorf("Expect external ip type service, got none")
	}
}

func TestExternalIPs(t *testing.T) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	fp := NewFakeProxier(ipt, ipvs, nil)
	svcIP := "10.20.30.41"
	svcPort := 80
	svcExternalIPs := "50.60.70.81"
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
	}

	makeServiceMap(fp,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *api.Service) {
			svc.Spec.Type = "ClusterIP"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.ExternalIPs = []string{svcExternalIPs}
			svc.Spec.Ports = []api.ServicePort{{
				Name:       svcPortName.Port,
				Port:       int32(svcPort),
				Protocol:   api.ProtocolTCP,
				TargetPort: intstr.FromInt(svcPort),
			}}
		}),
	)

	epIP := "10.180.0.1"
	makeEndpointsMap(fp,
		makeTestEndpoints(svcPortName.Namespace, svcPortName.Name, func(ept *api.Endpoints) {
			ept.Subsets = []api.EndpointSubset{{
				Addresses: []api.EndpointAddress{{
					IP: epIP,
				}},
				Ports: []api.EndpointPort{{
					Name: svcPortName.Port,
					Port: int32(svcPort),
				}},
			}}
		}),
	)

	fp.syncProxyRules(syncReasonForce)

	// check ipvs service and destinations
	services, err := ipvs.GetServices()
	if err != nil {
		t.Errorf("Failed to get ipvs services, err: %v")
	}
	if len(services) != 2 {
		t.Errorf("Expect 2 ipvs services, got %d", len(services))
	}
	found := false
	for _, svc := range services {
		if svc.Address.To4().String() == svcExternalIPs && svc.Port == uint16(svcPort) && svc.Protocol == string(api.ProtocolTCP) {
			found = true
			destinations, _ := ipvs.GetDestinations(svc)
			for _, dest := range destinations {
				if dest.Address.To4().String() != epIP || dest.Port != uint16(svcPort) {
					t.Errorf("service Endpoint mismatch ipvs service destination")
				}
			}
			break
		}
	}
	if !found {
		t.Errorf("Expect external ip type service, got none")
	}
}

func TestLoadBalancer(t *testing.T) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	fp := NewFakeProxier(ipt, ipvs, nil)
	svcIP := "10.20.30.41"
	svcPort := 80
	svcNodePort := 3001
	svcLBIP := "1.2.3.4"
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
	}

	makeServiceMap(fp,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *api.Service) {
			svc.Spec.Type = "LoadBalancer"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.Ports = []api.ServicePort{{
				Name:     svcPortName.Port,
				Port:     int32(svcPort),
				Protocol: api.ProtocolTCP,
				NodePort: int32(svcNodePort),
			}}
			svc.Status.LoadBalancer.Ingress = []api.LoadBalancerIngress{{
				IP: svcLBIP,
			}}
		}),
	)

	epIP := "10.180.0.1"
	makeEndpointsMap(fp,
		makeTestEndpoints(svcPortName.Namespace, svcPortName.Name, func(ept *api.Endpoints) {
			ept.Subsets = []api.EndpointSubset{{
				Addresses: []api.EndpointAddress{{
					IP: epIP,
				}},
				Ports: []api.EndpointPort{{
					Name: svcPortName.Port,
					Port: int32(svcPort),
				}},
			}}
		}),
	)

	fp.syncProxyRules(syncReasonForce)
}

func strPtr(s string) *string {
	return &s
}

func TestOnlyLocalNodePorts(t *testing.T) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	nodeIP := net.ParseIP("100.101.102.103")
	fp := NewFakeProxier(ipt, ipvs, []net.IP{nodeIP})
	svcIP := "10.20.30.41"
	svcPort := 80
	svcNodePort := 3001
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
	}

	makeServiceMap(fp,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *api.Service) {
			svc.Spec.Type = "NodePort"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.Ports = []api.ServicePort{{
				Name:     svcPortName.Port,
				Port:     int32(svcPort),
				Protocol: api.ProtocolTCP,
				NodePort: int32(svcNodePort),
			}}
			svc.Annotations[api.BetaAnnotationExternalTraffic] = api.AnnotationValueExternalTrafficLocal
		}),
	)

	epIP1 := "10.180.0.1"
	epIP2 := "10.180.2.1"
	makeEndpointsMap(fp,
		makeTestEndpoints(svcPortName.Namespace, svcPortName.Name, func(ept *api.Endpoints) {
			ept.Subsets = []api.EndpointSubset{{
				Addresses: []api.EndpointAddress{{
					IP:       epIP1,
					NodeName: nil,
				}, {
					IP:       epIP2,
					NodeName: strPtr(testHostname),
				}},
				Ports: []api.EndpointPort{{
					Name: svcPortName.Port,
					Port: int32(svcPort),
				}},
			}}
		}),
	)

	fp.syncProxyRules(syncReasonForce)

	// Expect 2 services and 1 destination
	services, err := ipvs.GetServices()
	if err != nil {
		t.Errorf("Failed to get ipvs services, err: %v")
	}
	if len(services) != 2 {
		t.Errorf("Expect 2 ipvs services, got %d", len(services))
	}
	found := false
	for _, svc := range services {
		if svc.Address.Equal(nodeIP) && svc.Port == uint16(svcNodePort) && svc.Protocol == string(api.ProtocolTCP) {
			found = true
			destinations, err := ipvs.GetDestinations(svc)
			if err != nil {
				t.Errorf("Failed to get ipvs destinations, err: %v", err)
			}
			if len(destinations) != 1 {
				t.Errorf("Expect 1 ipvs destination, got %d", len(destinations))
			} else {
				if destinations[0].Address.To4().String() != epIP2 || destinations[0].Port != uint16(svcPort) {
					t.Errorf("service Endpoint mismatch ipvs service destination")
				}
			}
			break
		}
	}
	if !found {
		t.Errorf("Expect node port type service, got none")
	}
}

// NO help
func TestOnlyLocalLoadBalancing(t *testing.T) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	fp := NewFakeProxier(ipt, ipvs, nil)
	svcIP := "10.20.30.41"
	svcPort := 80
	svcNodePort := 3001
	svcLBIP := "1.2.3.4"
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
	}

	makeServiceMap(fp,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *api.Service) {
			svc.Spec.Type = "LoadBalancer"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.Ports = []api.ServicePort{{
				Name:     svcPortName.Port,
				Port:     int32(svcPort),
				Protocol: api.ProtocolTCP,
				NodePort: int32(svcNodePort),
			}}
			svc.Status.LoadBalancer.Ingress = []api.LoadBalancerIngress{{
				IP: svcLBIP,
			}}
			svc.Annotations[api.BetaAnnotationExternalTraffic] = api.AnnotationValueExternalTrafficLocal
		}),
	)

	epIP1 := "10.180.0.1"
	epIP2 := "10.180.2.1"
	makeEndpointsMap(fp,
		makeTestEndpoints(svcPortName.Namespace, svcPortName.Name, func(ept *api.Endpoints) {
			ept.Subsets = []api.EndpointSubset{{
				Addresses: []api.EndpointAddress{{
					IP:       epIP1,
					NodeName: nil,
				}, {
					IP:       epIP2,
					NodeName: strPtr(testHostname),
				}},
				Ports: []api.EndpointPort{{
					Name: svcPortName.Port,
					Port: int32(svcPort),
				}},
			}}
		}),
	)

	fp.syncProxyRules(syncReasonForce)
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

func TestBuildServiceMapAddRemove(t *testing.T) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	fp := NewFakeProxier(ipt, ipvs, nil)

	services := []*api.Service{
		makeTestService("somewhere-else", "cluster-ip", func(svc *api.Service) {
			svc.Spec.Type = api.ServiceTypeClusterIP
			svc.Spec.ClusterIP = "172.16.55.4"
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "something", "UDP", 1234, 4321, 0)
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "somethingelse", "UDP", 1235, 5321, 0)
		}),
		makeTestService("somewhere-else", "node-port", func(svc *api.Service) {
			svc.Spec.Type = api.ServiceTypeNodePort
			svc.Spec.ClusterIP = "172.16.55.10"
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "blahblah", "UDP", 345, 678, 0)
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "moreblahblah", "TCP", 344, 677, 0)
		}),
		makeTestService("somewhere", "load-balancer", func(svc *api.Service) {
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
		makeTestService("somewhere", "only-local-load-balancer", func(svc *api.Service) {
			svc.ObjectMeta.Annotations = map[string]string{
				api.BetaAnnotationExternalTraffic:     api.AnnotationValueExternalTrafficLocal,
				api.BetaAnnotationHealthCheckNodePort: "345",
			}
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
		}),
	}

	for i := range services {
		fp.OnServiceAdd(services[i])
	}
	_, hcPorts, staleUDPServices := proxyutil.UpdateServiceMap(fp.serviceMap, &fp.serviceChanges)
	if len(fp.serviceMap) != 8 {
		t.Errorf("expected service map length 8, got %v", fp.serviceMap)
	}

	// The only-local-loadbalancer ones get added
	if len(hcPorts) != 1 {
		t.Errorf("expected 1 healthcheck port, got %v", hcPorts)
	} else {
		nsn := makeNSN("somewhere", "only-local-load-balancer")
		if port, found := hcPorts[nsn]; !found || port != 345 {
			t.Errorf("expected healthcheck port [%q]=345: got %v", nsn, hcPorts)
		}
	}

	if len(staleUDPServices) != 0 {
		// Services only added, so nothing stale yet
		t.Errorf("expected stale UDP services length 0, got %d", len(staleUDPServices))
	}

	// Remove some stuff
	// oneService is a modification of services[0] with removed first port.
	oneService := makeTestService("somewhere-else", "cluster-ip", func(svc *api.Service) {
		svc.Spec.Type = api.ServiceTypeClusterIP
		svc.Spec.ClusterIP = "172.16.55.4"
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "somethingelse", "UDP", 1235, 5321, 0)
	})

	fp.OnServiceUpdate(services[0], oneService)
	fp.OnServiceDelete(services[1])
	fp.OnServiceDelete(services[2])
	fp.OnServiceDelete(services[3])

	_, hcPorts, staleUDPServices = proxyutil.UpdateServiceMap(fp.serviceMap, &fp.serviceChanges)
	if len(fp.serviceMap) != 1 {
		t.Errorf("expected service map length 1, got %v", fp.serviceMap)
	}

	if len(hcPorts) != 0 {
		t.Errorf("expected 0 healthcheck ports, got %v", hcPorts)
	}

	// All services but one were deleted. While you'd expect only the ClusterIPs
	// from the three deleted services here, we still have the ClusterIP for
	// the not-deleted service, because one of it's ServicePorts was deleted.
	expectedStaleUDPServices := []string{"172.16.55.10", "172.16.55.4", "172.16.55.11", "172.16.55.12"}
	if len(staleUDPServices) != len(expectedStaleUDPServices) {
		t.Errorf("expected stale UDP services length %d, got %v", len(expectedStaleUDPServices), staleUDPServices.List())
	}
	for _, ip := range expectedStaleUDPServices {
		if !staleUDPServices.Has(ip) {
			t.Errorf("expected stale UDP service service %s", ip)
		}
	}
}

func TestBuildServiceMapServiceHeadless(t *testing.T) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	fp := NewFakeProxier(ipt, ipvs, nil)

	makeServiceMap(fp,
		makeTestService("somewhere-else", "headless", func(svc *api.Service) {
			svc.Spec.Type = api.ServiceTypeClusterIP
			svc.Spec.ClusterIP = api.ClusterIPNone
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "rpc", "UDP", 1234, 0, 0)
		}),
	)

	// Headless service should be ignored
	_, hcPorts, staleUDPServices := proxyutil.UpdateServiceMap(fp.serviceMap, &fp.serviceChanges)
	if len(fp.serviceMap) != 0 {
		t.Errorf("expected service map length 0, got %d", len(fp.serviceMap))
	}

	// No proxied services, so no healthchecks
	if len(hcPorts) != 0 {
		t.Errorf("expected healthcheck ports length 0, got %d", len(hcPorts))
	}

	if len(staleUDPServices) != 0 {
		t.Errorf("expected stale UDP services length 0, got %d", len(staleUDPServices))
	}
}

func TestBuildServiceMapServiceTypeExternalName(t *testing.T) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	fp := NewFakeProxier(ipt, ipvs, nil)

	makeServiceMap(fp,
		makeTestService("somewhere-else", "external-name", func(svc *api.Service) {
			svc.Spec.Type = api.ServiceTypeExternalName
			svc.Spec.ClusterIP = "172.16.55.4" // Should be ignored
			svc.Spec.ExternalName = "foo2.bar.com"
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "blah", "UDP", 1235, 5321, 0)
		}),
	)

	_, hcPorts, staleUDPServices := proxyutil.UpdateServiceMap(fp.serviceMap, &fp.serviceChanges)
	if len(fp.serviceMap) != 0 {
		t.Errorf("expected service map length 0, got %v", fp.serviceMap)
	}
	// No proxied services, so no healthchecks
	if len(hcPorts) != 0 {
		t.Errorf("expected healthcheck ports length 0, got %v", hcPorts)
	}
	if len(staleUDPServices) != 0 {
		t.Errorf("expected stale UDP services length 0, got %v", staleUDPServices)
	}
}

func TestBuildServiceMapServiceUpdate(t *testing.T) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	fp := NewFakeProxier(ipt, ipvs, nil)

	servicev1 := makeTestService("somewhere", "some-service", func(svc *api.Service) {
		svc.Spec.Type = api.ServiceTypeClusterIP
		svc.Spec.ClusterIP = "172.16.55.4"
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "something", "UDP", 1234, 4321, 0)
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "somethingelse", "TCP", 1235, 5321, 0)
	})
	servicev2 := makeTestService("somewhere", "some-service", func(svc *api.Service) {
		svc.ObjectMeta.Annotations = map[string]string{
			api.BetaAnnotationExternalTraffic:     api.AnnotationValueExternalTrafficLocal,
			api.BetaAnnotationHealthCheckNodePort: "345",
		}
		svc.Spec.Type = api.ServiceTypeLoadBalancer
		svc.Spec.ClusterIP = "172.16.55.4"
		svc.Spec.LoadBalancerIP = "5.6.7.8"
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "something", "UDP", 1234, 4321, 7002)
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "somethingelse", "TCP", 1235, 5321, 7003)
		svc.Status.LoadBalancer = api.LoadBalancerStatus{
			Ingress: []api.LoadBalancerIngress{
				{IP: "10.1.2.3"},
			},
		}
	})

	fp.OnServiceAdd(servicev1)

	syncRequired, hcPorts, staleUDPServices := proxyutil.UpdateServiceMap(fp.serviceMap, &fp.serviceChanges)
	if !syncRequired {
		t.Errorf("expected sync required, got %t", syncRequired)
	}
	if len(fp.serviceMap) != 2 {
		t.Errorf("expected service map length 2, got %v", fp.serviceMap)
	}
	if len(hcPorts) != 0 {
		t.Errorf("expected healthcheck ports length 0, got %v", hcPorts)
	}
	if len(staleUDPServices) != 0 {
		// Services only added, so nothing stale yet
		t.Errorf("expected stale UDP services length 0, got %d", len(staleUDPServices))
	}

	// Change service to load-balancer
	fp.OnServiceUpdate(servicev1, servicev2)
	syncRequired, hcPorts, staleUDPServices = proxyutil.UpdateServiceMap(fp.serviceMap, &fp.serviceChanges)
	if !syncRequired {
		t.Errorf("expected sync required, got %t", syncRequired)
	}
	if len(fp.serviceMap) != 2 {
		t.Errorf("expected service map length 2, got %v", fp.serviceMap)
	}
	if len(hcPorts) != 1 {
		t.Errorf("expected healthcheck ports length 1, got %v", hcPorts)
	}
	if len(staleUDPServices) != 0 {
		t.Errorf("expected stale UDP services length 0, got %v", staleUDPServices.List())
	}

	// No change; make sure the service map stays the same and there are
	// no health-check changes
	fp.OnServiceUpdate(servicev2, servicev2)
	syncRequired, hcPorts, staleUDPServices = proxyutil.UpdateServiceMap(fp.serviceMap, &fp.serviceChanges)
	if syncRequired {
		t.Errorf("not expected sync required, got %t", syncRequired)
	}
	if len(fp.serviceMap) != 2 {
		t.Errorf("expected service map length 2, got %v", fp.serviceMap)
	}
	if len(hcPorts) != 1 {
		t.Errorf("expected healthcheck ports length 1, got %v", hcPorts)
	}
	if len(staleUDPServices) != 0 {
		t.Errorf("expected stale UDP services length 0, got %v", staleUDPServices.List())
	}

	// And back to ClusterIP
	fp.OnServiceUpdate(servicev2, servicev1)
	syncRequired, hcPorts, staleUDPServices = proxyutil.UpdateServiceMap(fp.serviceMap, &fp.serviceChanges)
	if !syncRequired {
		t.Errorf("expected sync required, got %t", syncRequired)
	}
	if len(fp.serviceMap) != 2 {
		t.Errorf("expected service map length 2, got %v", fp.serviceMap)
	}
	if len(hcPorts) != 0 {
		t.Errorf("expected healthcheck ports length 0, got %v", hcPorts)
	}
	if len(staleUDPServices) != 0 {
		// Services only added, so nothing stale yet
		t.Errorf("expected stale UDP services length 0, got %d", len(staleUDPServices))
	}
}

func TestSessionAffinity(t *testing.T) {
	ipt := iptablestest.NewFake()
	ipvs := ipvstest.NewFake()
	nodeIP := net.ParseIP("100.101.102.103")
	fp := NewFakeProxier(ipt, ipvs, []net.IP{nodeIP})
	svcIP := "10.20.30.41"
	svcPort := 80
	svcNodePort := 3001
	svcExternalIPs := "50.60.70.81"
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
	}

	makeServiceMap(fp,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *api.Service) {
			svc.Spec.Type = "NodePort"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.ExternalIPs = []string{svcExternalIPs}
			svc.Spec.SessionAffinity = api.ServiceAffinityClientIP
			svc.Spec.Ports = []api.ServicePort{{
				Name:     svcPortName.Port,
				Port:     int32(svcPort),
				Protocol: api.ProtocolTCP,
				NodePort: int32(svcNodePort),
			}}
		}),
	)
	makeEndpointsMap(fp)

	fp.syncProxyRules(syncReasonForce)

	// check ipvs service and destinations
	services, err := ipvs.GetServices()
	if err != nil {
		t.Errorf("Failed to get ipvs services, err: %v")
	}
	for _, svc := range services {
		if svc.Timeout != uint32(180*60) {
			t.Errorf("expected mismatch ipvs service session affinity timeout: %d, expected: %d", svc.Timeout, 180*60)
		}
	}
}

func makeServicePortName(ns, name, port string) proxy.ServicePortName {
	return proxy.ServicePortName{
		NamespacedName: makeNSN(ns, name),
		Port:           port,
	}
}

func Test_updateEndpointsMap(t *testing.T) {
	var nodeName = "host"

	unnamedPort := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP: "1.1.1.1",
			}},
			Ports: []api.EndpointPort{{
				Port: 11,
			}},
		}}
	}
	unnamedPortLocal := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP:       "1.1.1.1",
				NodeName: &nodeName,
			}},
			Ports: []api.EndpointPort{{
				Port: 11,
			}},
		}}
	}
	namedPortLocal := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP:       "1.1.1.1",
				NodeName: &nodeName,
			}},
			Ports: []api.EndpointPort{{
				Name: "p11",
				Port: 11,
			}},
		}}
	}
	namedPort := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP: "1.1.1.1",
			}},
			Ports: []api.EndpointPort{{
				Name: "p11",
				Port: 11,
			}},
		}}
	}
	namedPortRenamed := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP: "1.1.1.1",
			}},
			Ports: []api.EndpointPort{{
				Name: "p11-2",
				Port: 11,
			}},
		}}
	}
	namedPortRenumbered := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP: "1.1.1.1",
			}},
			Ports: []api.EndpointPort{{
				Name: "p11",
				Port: 22,
			}},
		}}
	}
	namedPortsLocalNoLocal := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP: "1.1.1.1",
			}, {
				IP:       "1.1.1.2",
				NodeName: &nodeName,
			}},
			Ports: []api.EndpointPort{{
				Name: "p11",
				Port: 11,
			}, {
				Name: "p12",
				Port: 12,
			}},
		}}
	}
	multipleSubsets := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP: "1.1.1.1",
			}},
			Ports: []api.EndpointPort{{
				Name: "p11",
				Port: 11,
			}},
		}, {
			Addresses: []api.EndpointAddress{{
				IP: "1.1.1.2",
			}},
			Ports: []api.EndpointPort{{
				Name: "p12",
				Port: 12,
			}},
		}}
	}
	multipleSubsetsWithLocal := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP: "1.1.1.1",
			}},
			Ports: []api.EndpointPort{{
				Name: "p11",
				Port: 11,
			}},
		}, {
			Addresses: []api.EndpointAddress{{
				IP:       "1.1.1.2",
				NodeName: &nodeName,
			}},
			Ports: []api.EndpointPort{{
				Name: "p12",
				Port: 12,
			}},
		}}
	}
	multipleSubsetsMultiplePortsLocal := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP:       "1.1.1.1",
				NodeName: &nodeName,
			}},
			Ports: []api.EndpointPort{{
				Name: "p11",
				Port: 11,
			}, {
				Name: "p12",
				Port: 12,
			}},
		}, {
			Addresses: []api.EndpointAddress{{
				IP: "1.1.1.3",
			}},
			Ports: []api.EndpointPort{{
				Name: "p13",
				Port: 13,
			}},
		}}
	}
	multipleSubsetsIPsPorts1 := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP: "1.1.1.1",
			}, {
				IP:       "1.1.1.2",
				NodeName: &nodeName,
			}},
			Ports: []api.EndpointPort{{
				Name: "p11",
				Port: 11,
			}, {
				Name: "p12",
				Port: 12,
			}},
		}, {
			Addresses: []api.EndpointAddress{{
				IP: "1.1.1.3",
			}, {
				IP:       "1.1.1.4",
				NodeName: &nodeName,
			}},
			Ports: []api.EndpointPort{{
				Name: "p13",
				Port: 13,
			}, {
				Name: "p14",
				Port: 14,
			}},
		}}
	}
	multipleSubsetsIPsPorts2 := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP: "2.2.2.1",
			}, {
				IP:       "2.2.2.2",
				NodeName: &nodeName,
			}},
			Ports: []api.EndpointPort{{
				Name: "p21",
				Port: 21,
			}, {
				Name: "p22",
				Port: 22,
			}},
		}}
	}
	complexBefore1 := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP: "1.1.1.1",
			}},
			Ports: []api.EndpointPort{{
				Name: "p11",
				Port: 11,
			}},
		}}
	}
	complexBefore2 := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP:       "2.2.2.2",
				NodeName: &nodeName,
			}, {
				IP:       "2.2.2.22",
				NodeName: &nodeName,
			}},
			Ports: []api.EndpointPort{{
				Name: "p22",
				Port: 22,
			}},
		}, {
			Addresses: []api.EndpointAddress{{
				IP:       "2.2.2.3",
				NodeName: &nodeName,
			}},
			Ports: []api.EndpointPort{{
				Name: "p23",
				Port: 23,
			}},
		}}
	}
	complexBefore4 := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP:       "4.4.4.4",
				NodeName: &nodeName,
			}, {
				IP:       "4.4.4.5",
				NodeName: &nodeName,
			}},
			Ports: []api.EndpointPort{{
				Name: "p44",
				Port: 44,
			}},
		}, {
			Addresses: []api.EndpointAddress{{
				IP:       "4.4.4.6",
				NodeName: &nodeName,
			}},
			Ports: []api.EndpointPort{{
				Name: "p45",
				Port: 45,
			}},
		}}
	}
	complexAfter1 := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP: "1.1.1.1",
			}, {
				IP: "1.1.1.11",
			}},
			Ports: []api.EndpointPort{{
				Name: "p11",
				Port: 11,
			}},
		}, {
			Addresses: []api.EndpointAddress{{
				IP: "1.1.1.2",
			}},
			Ports: []api.EndpointPort{{
				Name: "p12",
				Port: 12,
			}, {
				Name: "p122",
				Port: 122,
			}},
		}}
	}
	complexAfter3 := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP: "3.3.3.3",
			}},
			Ports: []api.EndpointPort{{
				Name: "p33",
				Port: 33,
			}},
		}}
	}
	complexAfter4 := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{
				IP:       "4.4.4.4",
				NodeName: &nodeName,
			}},
			Ports: []api.EndpointPort{{
				Name: "p44",
				Port: 44,
			}},
		}}
	}

	testCases := []struct {
		// previousEndpoints and currentEndpoints are used to call appropriate
		// handlers OnEndpoints* (based on whether corresponding values are nil
		// or non-nil) and must be of equal length.
		previousEndpoints    []*api.Endpoints
		currentEndpoints     []*api.Endpoints
		oldEndpoints         map[proxy.ServicePortName][]*proxyutil.EndpointsInfo
		expectedResult       map[proxy.ServicePortName][]*proxyutil.EndpointsInfo
		expectedStale        []proxyutil.EndpointServicePair
		expectedHealthchecks map[types.NamespacedName]int
	}{{
		// Case[0]: nothing
		oldEndpoints:         map[proxy.ServicePortName][]*proxyutil.EndpointsInfo{},
		expectedResult:       map[proxy.ServicePortName][]*proxyutil.EndpointsInfo{},
		expectedStale:        []proxyutil.EndpointServicePair{},
		expectedHealthchecks: map[types.NamespacedName]int{},
	}, {
		// Case[1]: no change, unnamed port
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", unnamedPort),
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", unnamedPort),
		},
		oldEndpoints: map[proxy.ServicePortName][]*proxyutil.EndpointsInfo{
			makeServicePortName("ns1", "ep1", ""): {
				{"1.1.1.1:11", false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxyutil.EndpointsInfo{
			makeServicePortName("ns1", "ep1", ""): {
				{"1.1.1.1:11", false},
			},
		},
		expectedStale:        []proxyutil.EndpointServicePair{},
		expectedHealthchecks: map[types.NamespacedName]int{},
	}, {
		// Case[2]: no change, named port, local
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPortLocal),
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPortLocal),
		},
		oldEndpoints: map[proxy.ServicePortName][]*proxyutil.EndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{"1.1.1.1:11", true},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxyutil.EndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{"1.1.1.1:11", true},
			},
		},
		expectedStale: []proxyutil.EndpointServicePair{},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 1,
		},
	}, {
		// Case[3]: no change, multiple subsets
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsets),
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsets),
		},
		oldEndpoints: map[proxy.ServicePortName][]*proxyutil.EndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{"1.1.1.1:11", false},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{"1.1.1.2:12", false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxyutil.EndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{"1.1.1.1:11", false},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{"1.1.1.2:12", false},
			},
		},
		expectedStale:        []proxyutil.EndpointServicePair{},
		expectedHealthchecks: map[types.NamespacedName]int{},
	}, {
		// Case[4]: no change, multiple subsets, multiple ports, local
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsetsMultiplePortsLocal),
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsetsMultiplePortsLocal),
		},
		oldEndpoints: map[proxy.ServicePortName][]*proxyutil.EndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{"1.1.1.1:11", true},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{"1.1.1.1:12", true},
			},
			makeServicePortName("ns1", "ep1", "p13"): {
				{"1.1.1.3:13", false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxyutil.EndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{"1.1.1.1:11", true},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{"1.1.1.1:12", true},
			},
			makeServicePortName("ns1", "ep1", "p13"): {
				{"1.1.1.3:13", false},
			},
		},
		expectedStale: []proxyutil.EndpointServicePair{},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 1,
		},
	}, {
		// Case[5]: no change, multiple Endpoints, subsets, IPs, and ports
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsetsIPsPorts1),
			makeTestEndpoints("ns2", "ep2", multipleSubsetsIPsPorts2),
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsetsIPsPorts1),
			makeTestEndpoints("ns2", "ep2", multipleSubsetsIPsPorts2),
		},
		oldEndpoints: map[proxy.ServicePortName][]*proxyutil.EndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{"1.1.1.1:11", false},
				{"1.1.1.2:11", true},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{"1.1.1.1:12", false},
				{"1.1.1.2:12", true},
			},
			makeServicePortName("ns1", "ep1", "p13"): {
				{"1.1.1.3:13", false},
				{"1.1.1.4:13", true},
			},
			makeServicePortName("ns1", "ep1", "p14"): {
				{"1.1.1.3:14", false},
				{"1.1.1.4:14", true},
			},
			makeServicePortName("ns2", "ep2", "p21"): {
				{"2.2.2.1:21", false},
				{"2.2.2.2:21", true},
			},
			makeServicePortName("ns2", "ep2", "p22"): {
				{"2.2.2.1:22", false},
				{"2.2.2.2:22", true},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxyutil.EndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{"1.1.1.1:11", false},
				{"1.1.1.2:11", true},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{"1.1.1.1:12", false},
				{"1.1.1.2:12", true},
			},
			makeServicePortName("ns1", "ep1", "p13"): {
				{"1.1.1.3:13", false},
				{"1.1.1.4:13", true},
			},
			makeServicePortName("ns1", "ep1", "p14"): {
				{"1.1.1.3:14", false},
				{"1.1.1.4:14", true},
			},
			makeServicePortName("ns2", "ep2", "p21"): {
				{"2.2.2.1:21", false},
				{"2.2.2.2:21", true},
			},
			makeServicePortName("ns2", "ep2", "p22"): {
				{"2.2.2.1:22", false},
				{"2.2.2.2:22", true},
			},
		},
		expectedStale: []proxyutil.EndpointServicePair{},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 2,
			makeNSN("ns2", "ep2"): 1,
		},
	}, {
		// Case[6]: add an Endpoints
		previousEndpoints: []*api.Endpoints{
			nil,
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", unnamedPortLocal),
		},
		oldEndpoints: map[proxy.ServicePortName][]*proxyutil.EndpointsInfo{},
		expectedResult: map[proxy.ServicePortName][]*proxyutil.EndpointsInfo{
			makeServicePortName("ns1", "ep1", ""): {
				{"1.1.1.1:11", true},
			},
		},
		expectedStale: []proxyutil.EndpointServicePair{},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 1,
		},
	}, {
		// Case[7]: remove an Endpoints
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", unnamedPortLocal),
		},
		currentEndpoints: []*api.Endpoints{
			nil,
		},
		oldEndpoints: map[proxy.ServicePortName][]*proxyutil.EndpointsInfo{
			makeServicePortName("ns1", "ep1", ""): {
				{"1.1.1.1:11", true},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxyutil.EndpointsInfo{},
		expectedStale: []proxyutil.EndpointServicePair{{
			Endpoint:        "1.1.1.1:11",
			ServicePortName: makeServicePortName("ns1", "ep1", ""),
		}},
		expectedHealthchecks: map[types.NamespacedName]int{},
	}, {
		// Case[8]: add an IP and port
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPort),
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPortsLocalNoLocal),
		},
		oldEndpoints: map[proxy.ServicePortName][]*proxyutil.EndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{"1.1.1.1:11", false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxyutil.EndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{"1.1.1.1:11", false},
				{"1.1.1.2:11", true},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{"1.1.1.1:12", false},
				{"1.1.1.2:12", true},
			},
		},
		expectedStale: []proxyutil.EndpointServicePair{},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 1,
		},
	}, {
		// Case[9]: remove an IP and port
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPortsLocalNoLocal),
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPort),
		},
		oldEndpoints: map[proxy.ServicePortName][]*proxyutil.EndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{"1.1.1.1:11", false},
				{"1.1.1.2:11", true},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{"1.1.1.1:12", false},
				{"1.1.1.2:12", true},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxyutil.EndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{"1.1.1.1:11", false},
			},
		},
		expectedStale: []proxyutil.EndpointServicePair{{
			Endpoint:        "1.1.1.2:11",
			ServicePortName: makeServicePortName("ns1", "ep1", "p11"),
		}, {
			Endpoint:        "1.1.1.1:12",
			ServicePortName: makeServicePortName("ns1", "ep1", "p12"),
		}, {
			Endpoint:        "1.1.1.2:12",
			ServicePortName: makeServicePortName("ns1", "ep1", "p12"),
		}},
		expectedHealthchecks: map[types.NamespacedName]int{},
	}, {
		// Case[10]: add a subset
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPort),
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsetsWithLocal),
		},
		oldEndpoints: map[proxy.ServicePortName][]*proxyutil.EndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{"1.1.1.1:11", false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxyutil.EndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{"1.1.1.1:11", false},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{"1.1.1.2:12", true},
			},
		},
		expectedStale: []proxyutil.EndpointServicePair{},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 1,
		},
	}, {
		// Case[11]: remove a subset
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsets),
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPort),
		},
		oldEndpoints: map[proxy.ServicePortName][]*proxyutil.EndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{"1.1.1.1:11", false},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{"1.1.1.2:12", false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxyutil.EndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{"1.1.1.1:11", false},
			},
		},
		expectedStale: []proxyutil.EndpointServicePair{{
			Endpoint:        "1.1.1.2:12",
			ServicePortName: makeServicePortName("ns1", "ep1", "p12"),
		}},
		expectedHealthchecks: map[types.NamespacedName]int{},
	}, {
		// Case[12]: rename a port
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPort),
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPortRenamed),
		},
		oldEndpoints: map[proxy.ServicePortName][]*proxyutil.EndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{"1.1.1.1:11", false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxyutil.EndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11-2"): {
				{"1.1.1.1:11", false},
			},
		},
		expectedStale: []proxyutil.EndpointServicePair{{
			Endpoint:        "1.1.1.1:11",
			ServicePortName: makeServicePortName("ns1", "ep1", "p11"),
		}},
		expectedHealthchecks: map[types.NamespacedName]int{},
	}, {
		// Case[13]: renumber a port
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPort),
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPortRenumbered),
		},
		oldEndpoints: map[proxy.ServicePortName][]*proxyutil.EndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{"1.1.1.1:11", false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxyutil.EndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{"1.1.1.1:22", false},
			},
		},
		expectedStale: []proxyutil.EndpointServicePair{{
			Endpoint:        "1.1.1.1:11",
			ServicePortName: makeServicePortName("ns1", "ep1", "p11"),
		}},
		expectedHealthchecks: map[types.NamespacedName]int{},
	}, {
		// Case[14]: complex add and remove
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", complexBefore1),
			makeTestEndpoints("ns2", "ep2", complexBefore2),
			nil,
			makeTestEndpoints("ns4", "ep4", complexBefore4),
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", complexAfter1),
			nil,
			makeTestEndpoints("ns3", "ep3", complexAfter3),
			makeTestEndpoints("ns4", "ep4", complexAfter4),
		},
		oldEndpoints: map[proxy.ServicePortName][]*proxyutil.EndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{"1.1.1.1:11", false},
			},
			makeServicePortName("ns2", "ep2", "p22"): {
				{"2.2.2.2:22", true},
				{"2.2.2.22:22", true},
			},
			makeServicePortName("ns2", "ep2", "p23"): {
				{"2.2.2.3:23", true},
			},
			makeServicePortName("ns4", "ep4", "p44"): {
				{"4.4.4.4:44", true},
				{"4.4.4.5:44", true},
			},
			makeServicePortName("ns4", "ep4", "p45"): {
				{"4.4.4.6:45", true},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*proxyutil.EndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{"1.1.1.1:11", false},
				{"1.1.1.11:11", false},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{"1.1.1.2:12", false},
			},
			makeServicePortName("ns1", "ep1", "p122"): {
				{"1.1.1.2:122", false},
			},
			makeServicePortName("ns3", "ep3", "p33"): {
				{"3.3.3.3:33", false},
			},
			makeServicePortName("ns4", "ep4", "p44"): {
				{"4.4.4.4:44", true},
			},
		},
		expectedStale: []proxyutil.EndpointServicePair{{
			Endpoint:        "2.2.2.2:22",
			ServicePortName: makeServicePortName("ns2", "ep2", "p22"),
		}, {
			Endpoint:        "2.2.2.22:22",
			ServicePortName: makeServicePortName("ns2", "ep2", "p22"),
		}, {
			Endpoint:        "2.2.2.3:23",
			ServicePortName: makeServicePortName("ns2", "ep2", "p23"),
		}, {
			Endpoint:        "4.4.4.5:44",
			ServicePortName: makeServicePortName("ns4", "ep4", "p44"),
		}, {
			Endpoint:        "4.4.4.6:45",
			ServicePortName: makeServicePortName("ns4", "ep4", "p45"),
		}},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns4", "ep4"): 1,
		},
	}}

	for tci, tc := range testCases {
		ipt := iptablestest.NewFake()
		ipvs := ipvstest.NewFake()
		fp := NewFakeProxier(ipt, ipvs, nil)
		fp.hostname = nodeName

		// First check that after adding all previous versions of Endpoints,
		// the fp.oldEndpoints is as we expect.
		for i := range tc.previousEndpoints {
			if tc.previousEndpoints[i] != nil {
				fp.OnEndpointsAdd(tc.previousEndpoints[i])
			}
		}
		proxyutil.UpdateEndpointsMap(fp.endpointsMap, &fp.endpointsChanges, fp.hostname)
		compareEndpointsMaps(t, tci, fp.endpointsMap, tc.oldEndpoints)

		// Now let's call appropriate handlers to get to state we want to be.
		if len(tc.previousEndpoints) != len(tc.currentEndpoints) {
			t.Fatalf("[%d] different lengths of previous and current Endpoints", tci)
			continue
		}

		for i := range tc.previousEndpoints {
			prev, curr := tc.previousEndpoints[i], tc.currentEndpoints[i]
			switch {
			case prev == nil:
				fp.OnEndpointsAdd(curr)
			case curr == nil:
				fp.OnEndpointsDelete(prev)
			default:
				fp.OnEndpointsUpdate(prev, curr)
			}
		}
		_, hcEndpoints, stale := proxyutil.UpdateEndpointsMap(fp.endpointsMap, &fp.endpointsChanges, fp.hostname)
		newMap := fp.endpointsMap
		compareEndpointsMaps(t, tci, newMap, tc.expectedResult)
		if len(stale) != len(tc.expectedStale) {
			t.Errorf("[%d] expected %d stale, got %d: %v", tci, len(tc.expectedStale), len(stale), stale)
		}
		for _, x := range tc.expectedStale {
			if stale[x] != true {
				t.Errorf("[%d] expected stale[%v], but didn't find it: %v", tci, x, stale)
			}
		}
		if !reflect.DeepEqual(hcEndpoints, tc.expectedHealthchecks) {
			t.Errorf("[%d] expected healthchecks %v, got %v", tci, tc.expectedHealthchecks, hcEndpoints)
		}
	}
}

func compareEndpointsMaps(t *testing.T, tci int, newMap, expected map[proxy.ServicePortName][]*proxyutil.EndpointsInfo) {
	if len(newMap) != len(expected) {
		t.Errorf("[%d] expected %d results, got %d: %v", tci, len(expected), len(newMap), newMap)
	}
	for x := range expected {
		if len(newMap[x]) != len(expected[x]) {
			t.Errorf("[%d] expected %d Endpoints for %v, got %d", tci, len(expected[x]), x, len(newMap[x]))
		} else {
			for i := range expected[x] {
				if *(newMap[x][i]) != *(expected[x][i]) {
					t.Errorf("[%d] expected new[%v][%d] to be %v, got %v", tci, x, i, expected[x][i], newMap[x][i])
				}
			}
		}
	}
}

func Test_parseHostPort(t *testing.T) {
	hostPorts := []struct{
		str string
		host string
		port int
	}{{
		str:  "1.2.3.4:80",
		host: "1.2.3.4",
		port: 80,
	}}

	for _, hostPort := range hostPorts {
		host, port := parseHostPort(hostPort.str)
		if host != hostPort.host || port != hostPort.port {
			t.Errorf("mismatch host and port")
		}
	}
}