// +build windows

/*
Copyright 2015 The Kubernetes Authors.

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

package winkernel

import (
	"reflect"
	"testing"
	"time"

	"github.com/davecgh/go-spew/spew"

	"fmt"
	"net"
	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/util/async"
	"k8s.io/utils/exec"
	fakeexec "k8s.io/utils/exec/testing"
)

func newFakeServiceInfo(service proxy.ServicePortName, ip net.IP, port int, protocol api.Protocol, onlyNodeLocalEndpoints bool) *serviceInfo {
	return &serviceInfo{
		sessionAffinityType:    api.ServiceAffinityNone, // default
		stickyMaxAgeMinutes:    180,
		clusterIP:              ip,
		port:                   port,
		protocol:               protocol,
		onlyNodeLocalEndpoints: onlyNodeLocalEndpoints,
	}
}

func TestDeleteEndpointConnections(t *testing.T) {
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
			func() ([]byte, error) { return []byte("1 flow entries have been deleted"), nil },
			func() ([]byte, error) {
				return []byte(""), fmt.Errorf("conntrack v1.4.2 (conntrack-tools): 0 flow entries have been deleted.")
			},
		},
	}
	fexec := fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
		},
		LookPathFunc: func(cmd string) (string, error) { return cmd, nil },
	}

	serviceMap := make(map[proxy.ServicePortName]*serviceInfo)
	svc1 := proxy.ServicePortName{NamespacedName: types.NamespacedName{Namespace: "ns1", Name: "svc1"}, Port: "p80"}
	svc2 := proxy.ServicePortName{NamespacedName: types.NamespacedName{Namespace: "ns1", Name: "svc2"}, Port: "p80"}
	serviceMap[svc1] = newFakeServiceInfo(svc1, net.IPv4(10, 20, 30, 40), 80, api.ProtocolUDP, false)
	serviceMap[svc2] = newFakeServiceInfo(svc1, net.IPv4(10, 20, 30, 41), 80, api.ProtocolTCP, false)

	fakeProxier := Proxier{exec: &fexec, serviceMap: serviceMap}

	testCases := []endpointServicePair{
		{
			endpoint:        "10.240.0.3:80",
			servicePortName: svc1,
		},
		{
			endpoint:        "10.240.0.4:80",
			servicePortName: svc1,
		},
		{
			endpoint:        "10.240.0.5:80",
			servicePortName: svc2,
		},
	}

	expectCommandExecCount := 0
	for i := range testCases {
		input := map[endpointServicePair]bool{testCases[i]: true}
		fakeProxier.deleteEndpointConnections(input)
		svcInfo := fakeProxier.serviceMap[testCases[i].servicePortName]
		if svcInfo.protocol == api.ProtocolUDP {
			svcIp := svcInfo.clusterIP.String()
			endpointIp := strings.Split(testCases[i].endpoint, ":")[0]
			expectCommand := fmt.Sprintf("conntrack -D --orig-dst %s --dst-nat %s -p udp", svcIp, endpointIp)
			execCommand := strings.Join(fcmd.CombinedOutputLog[expectCommandExecCount], " ")
			if expectCommand != execCommand {
				t.Errorf("Exepect comand: %s, but executed %s", expectCommand, execCommand)
			}
			expectCommandExecCount += 1
		}

		if expectCommandExecCount != fexec.CommandCalls {
			t.Errorf("Exepect comand executed %d times, but got %d", expectCommandExecCount, fexec.CommandCalls)
		}
	}
}

type fakeClosable struct {
	closed bool
}

func (c *fakeClosable) Close() error {
	c.closed = true
	return nil
}

func TestRevertPorts(t *testing.T) {
	testCases := []struct {
		replacementPorts []localPort
		existingPorts    []localPort
		expectToBeClose  []bool
	}{
		{
			replacementPorts: []localPort{
				{port: 5001},
				{port: 5002},
				{port: 5003},
			},
			existingPorts:   []localPort{},
			expectToBeClose: []bool{true, true, true},
		},
		{
			replacementPorts: []localPort{},
			existingPorts: []localPort{
				{port: 5001},
				{port: 5002},
				{port: 5003},
			},
			expectToBeClose: []bool{},
		},
		{
			replacementPorts: []localPort{
				{port: 5001},
				{port: 5002},
				{port: 5003},
			},
			existingPorts: []localPort{
				{port: 5001},
				{port: 5002},
				{port: 5003},
			},
			expectToBeClose: []bool{false, false, false},
		},
		{
			replacementPorts: []localPort{
				{port: 5001},
				{port: 5002},
				{port: 5003},
			},
			existingPorts: []localPort{
				{port: 5001},
				{port: 5003},
			},
			expectToBeClose: []bool{false, true, false},
		},
		{
			replacementPorts: []localPort{
				{port: 5001},
				{port: 5002},
				{port: 5003},
			},
			existingPorts: []localPort{
				{port: 5001},
				{port: 5002},
				{port: 5003},
				{port: 5004},
			},
			expectToBeClose: []bool{false, false, false},
		},
	}

	for i, tc := range testCases {
		replacementPortsMap := make(map[localPort]closeable)
		for _, lp := range tc.replacementPorts {
			replacementPortsMap[lp] = &fakeClosable{}
		}
		existingPortsMap := make(map[localPort]closeable)
		for _, lp := range tc.existingPorts {
			existingPortsMap[lp] = &fakeClosable{}
		}
		revertPorts(replacementPortsMap, existingPortsMap)
		for j, expectation := range tc.expectToBeClose {
			if replacementPortsMap[tc.replacementPorts[j]].(*fakeClosable).closed != expectation {
				t.Errorf("Expect replacement localport %v to be %v in test case %v", tc.replacementPorts[j], expectation, i)
			}
		}
		for _, lp := range tc.existingPorts {
			if existingPortsMap[lp].(*fakeClosable).closed == true {
				t.Errorf("Expect existing localport %v to be false in test case %v", lp, i)
			}
		}
	}

}

// fakePortOpener implements portOpener.
type fakePortOpener struct {
	openPorts []*localPort
}

// OpenLocalPort fakes out the listen() and bind() used by syncProxyRules
// to lock a local port.
func (f *fakePortOpener) OpenLocalPort(lp *localPort) (closeable, error) {
	f.openPorts = append(f.openPorts, lp)
	return nil, nil
}

type fakeHealthChecker struct {
	services  map[types.NamespacedName]uint16
	endpoints map[types.NamespacedName]int
}

func newFakeHealthChecker() *fakeHealthChecker {
	return &fakeHealthChecker{
		services:  map[types.NamespacedName]uint16{},
		endpoints: map[types.NamespacedName]int{},
	}
}

func (fake *fakeHealthChecker) SyncServices(newServices map[types.NamespacedName]uint16) error {
	fake.services = newServices
	return nil
}

func (fake *fakeHealthChecker) SyncEndpoints(newEndpoints map[types.NamespacedName]int) error {
	fake.endpoints = newEndpoints
	return nil
}

func getFakeHnsNetwork() *hnsNetworkInfo {
	return &hnsNetworkInfo{
		id:   "00000000-0000-0000-0000-000000000001",
		name: "fakeNetwork",
	}, nil
}

const testHostname = "test-hostname"

func NewFakeProxier() *Proxier {
	fakeHnsNetwork := getFakeHnsNetwork()
	// TODO: Call NewProxier after refactoring out the goroutine
	// invocation into a Run() method.
	p := &Proxier{
		serviceMap:       make(proxyServiceMap),
		serviceChanges:   newServiceChangeMap(),
		endpointsMap:     make(proxyEndpointsMap),
		endpointsChanges: newEndpointsChangeMap(testHostname),
		clusterCIDR:      "10.0.0.0/24",
		hostname:         testHostname,
		portsMap:         make(map[localPort]closeable),
		healthChecker:    newFakeHealthChecker(),
		network:          fakeHnsNetwork,
	}
	p.syncRunner = async.NewBoundedFrequencyRunner("test-sync-runner", p.syncProxyRules, 0, time.Minute, 1)
	return p
}

func errorf(msg string, rules []iptablestest.Rule, t *testing.T) {
	for _, r := range rules {
		t.Logf("%q", r)
	}
	t.Errorf("%v", msg)
}

func TestLoadBalancer(t *testing.T) {
	fp := NewFakeProxier()
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

	fp.syncProxyRules()

	proto := strings.ToLower(string(api.ProtocolTCP))
	fwChain := string(serviceFirewallChainName(svcPortName.String(), proto))
	svcChain := string(servicePortChainName(svcPortName.String(), proto))
	//lbChain := string(serviceLBChainName(svcPortName.String(), proto))

	// TODO

}

func TestNodePort(t *testing.T) {

	fp := NewFakeProxier()
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

	fp.syncProxyRules()

	proto := strings.ToLower(string(api.ProtocolTCP))
	svcChain := string(servicePortChainName(svcPortName.String(), proto))

	// TODO
}

func TestExternalIPsReject(t *testing.T) {

	fp := NewFakeProxier()
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

	fp.syncProxyRules()

}

func TestNodePortReject(t *testing.T) {

	fp := NewFakeProxier()
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

	fp.syncProxyRules()

	// TODO
}

func strPtr(s string) *string {
	return &s
}

func TestOnlyLocalLoadBalancing(t *testing.T) {

	fp := NewFakeProxier()
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
	epStrLocal := fmt.Sprintf("%s:%d", epIP1, svcPort)
	epStrNonLocal := fmt.Sprintf("%s:%d", epIP2, svcPort)
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

	fp.syncProxyRules()

	proto := strings.ToLower(string(api.ProtocolTCP))
	fwChain := string(serviceFirewallChainName(svcPortName.String(), proto))
	lbChain := string(serviceLBChainName(svcPortName.String(), proto))

	nonLocalEpChain := string(servicePortEndpointChainName(svcPortName.String(), strings.ToLower(string(api.ProtocolTCP)), epStrLocal))
	localEpChain := string(servicePortEndpointChainName(svcPortName.String(), strings.ToLower(string(api.ProtocolTCP)), epStrNonLocal))

	// TODO
}

func TestOnlyLocalNodePortsNoClusterCIDR(t *testing.T) {

	fp := NewFakeProxier()
	// set cluster CIDR to empty before test
	fp.clusterCIDR = ""
	onlyLocalNodePorts(t, fp)
}

func TestOnlyLocalNodePorts(t *testing.T) {

	fp := NewFakeProxier()
	onlyLocalNodePorts(t, fp)
}

func onlyLocalNodePorts(t *testing.T, fp *Proxier) {
	shouldLBTOSVCRuleExist := len(fp.clusterCIDR) > 0
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
	epStrLocal := fmt.Sprintf("%s:%d", epIP1, svcPort)
	epStrNonLocal := fmt.Sprintf("%s:%d", epIP2, svcPort)
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

	fp.syncProxyRules()

	// TODO
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

func TestBuildServiceMapAddRemove(t *testing.T) {

	fp := NewFakeProxier()

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
	result := updateServiceMap(fp.serviceMap, &fp.serviceChanges)
	if len(fp.serviceMap) != 8 {
		t.Errorf("expected service map length 8, got %v", fp.serviceMap)
	}

	// The only-local-loadbalancer ones get added
	if len(result.hcServices) != 1 {
		t.Errorf("expected 1 healthcheck port, got %v", result.hcServices)
	} else {
		nsn := makeNSN("somewhere", "only-local-load-balancer")
		if port, found := result.hcServices[nsn]; !found || port != 345 {
			t.Errorf("expected healthcheck port [%q]=345: got %v", nsn, result.hcServices)
		}
	}

	if len(result.staleServices) != 0 {
		// Services only added, so nothing stale yet
		t.Errorf("expected stale UDP services length 0, got %d", len(result.staleServices))
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

	result = updateServiceMap(fp.serviceMap, &fp.serviceChanges)
	if len(fp.serviceMap) != 1 {
		t.Errorf("expected service map length 1, got %v", fp.serviceMap)
	}

	if len(result.hcServices) != 0 {
		t.Errorf("expected 0 healthcheck ports, got %v", result.hcServices)
	}

	// All services but one were deleted. While you'd expect only the ClusterIPs
	// from the three deleted services here, we still have the ClusterIP for
	// the not-deleted service, because one of it's ServicePorts was deleted.
	expectedStaleUDPServices := []string{"172.16.55.10", "172.16.55.4", "172.16.55.11", "172.16.55.12"}
	if len(result.staleServices) != len(expectedStaleUDPServices) {
		t.Errorf("expected stale UDP services length %d, got %v", len(expectedStaleUDPServices), result.staleServices.List())
	}
	for _, ip := range expectedStaleUDPServices {
		if !result.staleServices.Has(ip) {
			t.Errorf("expected stale UDP service service %s", ip)
		}
	}
}

func TestBuildServiceMapServiceHeadless(t *testing.T) {

	fp := NewFakeProxier()

	makeServiceMap(fp,
		makeTestService("somewhere-else", "headless", func(svc *api.Service) {
			svc.Spec.Type = api.ServiceTypeClusterIP
			svc.Spec.ClusterIP = api.ClusterIPNone
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "rpc", "UDP", 1234, 0, 0)
		}),
		makeTestService("somewhere-else", "headless-without-port", func(svc *api.Service) {
			svc.Spec.Type = api.ServiceTypeClusterIP
			svc.Spec.ClusterIP = api.ClusterIPNone
		}),
	)

	// Headless service should be ignored
	result := updateServiceMap(fp.serviceMap, &fp.serviceChanges)
	if len(fp.serviceMap) != 0 {
		t.Errorf("expected service map length 0, got %d", len(fp.serviceMap))
	}

	// No proxied services, so no healthchecks
	if len(result.hcServices) != 0 {
		t.Errorf("expected healthcheck ports length 0, got %d", len(result.hcServices))
	}

	if len(result.staleServices) != 0 {
		t.Errorf("expected stale UDP services length 0, got %d", len(result.staleServices))
	}
}

func TestBuildServiceMapServiceTypeExternalName(t *testing.T) {

	fp := NewFakeProxier()

	makeServiceMap(fp,
		makeTestService("somewhere-else", "external-name", func(svc *api.Service) {
			svc.Spec.Type = api.ServiceTypeExternalName
			svc.Spec.ClusterIP = "172.16.55.4" // Should be ignored
			svc.Spec.ExternalName = "foo2.bar.com"
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "blah", "UDP", 1235, 5321, 0)
		}),
	)

	result := updateServiceMap(fp.serviceMap, &fp.serviceChanges)
	if len(fp.serviceMap) != 0 {
		t.Errorf("expected service map length 0, got %v", fp.serviceMap)
	}
	// No proxied services, so no healthchecks
	if len(result.hcServices) != 0 {
		t.Errorf("expected healthcheck ports length 0, got %v", result.hcServices)
	}
	if len(result.staleServices) != 0 {
		t.Errorf("expected stale UDP services length 0, got %v", result.staleServices)
	}
}

func TestBuildServiceMapServiceUpdate(t *testing.T) {
	fp := NewFakeProxier()

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

	result := updateServiceMap(fp.serviceMap, &fp.serviceChanges)
	if len(fp.serviceMap) != 2 {
		t.Errorf("expected service map length 2, got %v", fp.serviceMap)
	}
	if len(result.hcServices) != 0 {
		t.Errorf("expected healthcheck ports length 0, got %v", result.hcServices)
	}
	if len(result.staleServices) != 0 {
		// Services only added, so nothing stale yet
		t.Errorf("expected stale UDP services length 0, got %d", len(result.staleServices))
	}

	// Change service to load-balancer
	fp.OnServiceUpdate(servicev1, servicev2)
	result = updateServiceMap(fp.serviceMap, &fp.serviceChanges)
	if len(fp.serviceMap) != 2 {
		t.Errorf("expected service map length 2, got %v", fp.serviceMap)
	}
	if len(result.hcServices) != 1 {
		t.Errorf("expected healthcheck ports length 1, got %v", result.hcServices)
	}
	if len(result.staleServices) != 0 {
		t.Errorf("expected stale UDP services length 0, got %v", result.staleServices.List())
	}

	// No change; make sure the service map stays the same and there are
	// no health-check changes
	fp.OnServiceUpdate(servicev2, servicev2)
	result = updateServiceMap(fp.serviceMap, &fp.serviceChanges)
	if len(fp.serviceMap) != 2 {
		t.Errorf("expected service map length 2, got %v", fp.serviceMap)
	}
	if len(result.hcServices) != 1 {
		t.Errorf("expected healthcheck ports length 1, got %v", result.hcServices)
	}
	if len(result.staleServices) != 0 {
		t.Errorf("expected stale UDP services length 0, got %v", result.staleServices.List())
	}

	// And back to ClusterIP
	fp.OnServiceUpdate(servicev2, servicev1)
	result = updateServiceMap(fp.serviceMap, &fp.serviceChanges)
	if len(fp.serviceMap) != 2 {
		t.Errorf("expected service map length 2, got %v", fp.serviceMap)
	}
	if len(result.hcServices) != 0 {
		t.Errorf("expected healthcheck ports length 0, got %v", result.hcServices)
	}
	if len(result.staleServices) != 0 {
		// Services only added, so nothing stale yet
		t.Errorf("expected stale UDP services length 0, got %d", len(result.staleServices))
	}
}

func Test_getLocalIPs(t *testing.T) {
	testCases := []struct {
		endpointsMap map[proxy.ServicePortName][]*endpointsInfo
		expected     map[types.NamespacedName]sets.String
	}{{
		// Case[0]: nothing
		endpointsMap: map[proxy.ServicePortName][]*endpointsInfo{},
		expected:     map[types.NamespacedName]sets.String{},
	}, {
		// Case[1]: unnamed port
		endpointsMap: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", ""): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
		},
		expected: map[types.NamespacedName]sets.String{},
	}, {
		// Case[2]: unnamed port local
		endpointsMap: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", ""): {
				{endpoint: "1.1.1.1:11", isLocal: true},
			},
		},
		expected: map[types.NamespacedName]sets.String{
			{Namespace: "ns1", Name: "ep1"}: sets.NewString("1.1.1.1"),
		},
	}, {
		// Case[3]: named local and non-local ports for the same IP.
		endpointsMap: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
				{endpoint: "1.1.1.2:11", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{endpoint: "1.1.1.1:12", isLocal: false},
				{endpoint: "1.1.1.2:12", isLocal: true},
			},
		},
		expected: map[types.NamespacedName]sets.String{
			{Namespace: "ns1", Name: "ep1"}: sets.NewString("1.1.1.2"),
		},
	}, {
		// Case[4]: named local and non-local ports for different IPs.
		endpointsMap: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
			makeServicePortName("ns2", "ep2", "p22"): {
				{endpoint: "2.2.2.2:22", isLocal: true},
				{endpoint: "2.2.2.22:22", isLocal: true},
			},
			makeServicePortName("ns2", "ep2", "p23"): {
				{endpoint: "2.2.2.3:23", isLocal: true},
			},
			makeServicePortName("ns4", "ep4", "p44"): {
				{endpoint: "4.4.4.4:44", isLocal: true},
				{endpoint: "4.4.4.5:44", isLocal: false},
			},
			makeServicePortName("ns4", "ep4", "p45"): {
				{endpoint: "4.4.4.6:45", isLocal: true},
			},
		},
		expected: map[types.NamespacedName]sets.String{
			{Namespace: "ns2", Name: "ep2"}: sets.NewString("2.2.2.2", "2.2.2.22", "2.2.2.3"),
			{Namespace: "ns4", Name: "ep4"}: sets.NewString("4.4.4.4", "4.4.4.6"),
		},
	}}

	for tci, tc := range testCases {
		// outputs
		localIPs := getLocalIPs(tc.endpointsMap)

		if !reflect.DeepEqual(localIPs, tc.expected) {
			t.Errorf("[%d] expected %#v, got %#v", tci, tc.expected, localIPs)
		}
	}
}

// This is a coarse test, but it offers some modicum of confidence as the code is evolved.
func Test_endpointsToEndpointsMap(t *testing.T) {
	testCases := []struct {
		newEndpoints *api.Endpoints
		expected     map[proxy.ServicePortName][]*endpointsInfo
	}{{
		// Case[0]: nothing
		newEndpoints: makeTestEndpoints("ns1", "ep1", func(ept *api.Endpoints) {}),
		expected:     map[proxy.ServicePortName][]*endpointsInfo{},
	}, {
		// Case[1]: no changes, unnamed port
		newEndpoints: makeTestEndpoints("ns1", "ep1", func(ept *api.Endpoints) {
			ept.Subsets = []api.EndpointSubset{
				{
					Addresses: []api.EndpointAddress{{
						IP: "1.1.1.1",
					}},
					Ports: []api.EndpointPort{{
						Name: "",
						Port: 11,
					}},
				},
			}
		}),
		expected: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", ""): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
		},
	}, {
		// Case[2]: no changes, named port
		newEndpoints: makeTestEndpoints("ns1", "ep1", func(ept *api.Endpoints) {
			ept.Subsets = []api.EndpointSubset{
				{
					Addresses: []api.EndpointAddress{{
						IP: "1.1.1.1",
					}},
					Ports: []api.EndpointPort{{
						Name: "port",
						Port: 11,
					}},
				},
			}
		}),
		expected: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "port"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
		},
	}, {
		// Case[3]: new port
		newEndpoints: makeTestEndpoints("ns1", "ep1", func(ept *api.Endpoints) {
			ept.Subsets = []api.EndpointSubset{
				{
					Addresses: []api.EndpointAddress{{
						IP: "1.1.1.1",
					}},
					Ports: []api.EndpointPort{{
						Port: 11,
					}},
				},
			}
		}),
		expected: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", ""): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
		},
	}, {
		// Case[4]: remove port
		newEndpoints: makeTestEndpoints("ns1", "ep1", func(ept *api.Endpoints) {}),
		expected:     map[proxy.ServicePortName][]*endpointsInfo{},
	}, {
		// Case[5]: new IP and port
		newEndpoints: makeTestEndpoints("ns1", "ep1", func(ept *api.Endpoints) {
			ept.Subsets = []api.EndpointSubset{
				{
					Addresses: []api.EndpointAddress{{
						IP: "1.1.1.1",
					}, {
						IP: "2.2.2.2",
					}},
					Ports: []api.EndpointPort{{
						Name: "p1",
						Port: 11,
					}, {
						Name: "p2",
						Port: 22,
					}},
				},
			}
		}),
		expected: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p1"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
				{endpoint: "2.2.2.2:11", isLocal: false},
			},
			makeServicePortName("ns1", "ep1", "p2"): {
				{endpoint: "1.1.1.1:22", isLocal: false},
				{endpoint: "2.2.2.2:22", isLocal: false},
			},
		},
	}, {
		// Case[6]: remove IP and port
		newEndpoints: makeTestEndpoints("ns1", "ep1", func(ept *api.Endpoints) {
			ept.Subsets = []api.EndpointSubset{
				{
					Addresses: []api.EndpointAddress{{
						IP: "1.1.1.1",
					}},
					Ports: []api.EndpointPort{{
						Name: "p1",
						Port: 11,
					}},
				},
			}
		}),
		expected: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p1"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
		},
	}, {
		// Case[7]: rename port
		newEndpoints: makeTestEndpoints("ns1", "ep1", func(ept *api.Endpoints) {
			ept.Subsets = []api.EndpointSubset{
				{
					Addresses: []api.EndpointAddress{{
						IP: "1.1.1.1",
					}},
					Ports: []api.EndpointPort{{
						Name: "p2",
						Port: 11,
					}},
				},
			}
		}),
		expected: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p2"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
		},
	}, {
		// Case[8]: renumber port
		newEndpoints: makeTestEndpoints("ns1", "ep1", func(ept *api.Endpoints) {
			ept.Subsets = []api.EndpointSubset{
				{
					Addresses: []api.EndpointAddress{{
						IP: "1.1.1.1",
					}},
					Ports: []api.EndpointPort{{
						Name: "p1",
						Port: 22,
					}},
				},
			}
		}),
		expected: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p1"): {
				{endpoint: "1.1.1.1:22", isLocal: false},
			},
		},
	}}

	for tci, tc := range testCases {
		// outputs
		newEndpoints := endpointsToEndpointsMap(tc.newEndpoints, "host")

		if len(newEndpoints) != len(tc.expected) {
			t.Errorf("[%d] expected %d new, got %d: %v", tci, len(tc.expected), len(newEndpoints), spew.Sdump(newEndpoints))
		}
		for x := range tc.expected {
			if len(newEndpoints[x]) != len(tc.expected[x]) {
				t.Errorf("[%d] expected %d endpoints for %v, got %d", tci, len(tc.expected[x]), x, len(newEndpoints[x]))
			} else {
				for i := range newEndpoints[x] {
					if *(newEndpoints[x][i]) != *(tc.expected[x][i]) {
						t.Errorf("[%d] expected new[%v][%d] to be %v, got %v", tci, x, i, tc.expected[x][i], *(newEndpoints[x][i]))
					}
				}
			}
		}
	}
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

func makeEndpointsMap(proxier *Proxier, allEndpoints ...*api.Endpoints) {
	for i := range allEndpoints {
		proxier.OnEndpointsAdd(allEndpoints[i])
	}

	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	proxier.endpointsSynced = true
}

func makeNSN(namespace, name string) types.NamespacedName {
	return types.NamespacedName{Namespace: namespace, Name: name}
}

func makeServicePortName(ns, name, port string) proxy.ServicePortName {
	return proxy.ServicePortName{
		NamespacedName: makeNSN(ns, name),
		Port:           port,
	}
}

func makeServiceMap(proxier *Proxier, allServices ...*api.Service) {
	for i := range allServices {
		proxier.OnServiceAdd(allServices[i])
	}

	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	proxier.servicesSynced = true
}

func compareEndpointsMaps(t *testing.T, tci int, newMap, expected map[proxy.ServicePortName][]*endpointsInfo) {
	if len(newMap) != len(expected) {
		t.Errorf("[%d] expected %d results, got %d: %v", tci, len(expected), len(newMap), newMap)
	}
	for x := range expected {
		if len(newMap[x]) != len(expected[x]) {
			t.Errorf("[%d] expected %d endpoints for %v, got %d", tci, len(expected[x]), x, len(newMap[x]))
		} else {
			for i := range expected[x] {
				if *(newMap[x][i]) != *(expected[x][i]) {
					t.Errorf("[%d] expected new[%v][%d] to be %v, got %v", tci, x, i, expected[x][i], newMap[x][i])
				}
			}
		}
	}
}

func Test_updateEndpointsMap(t *testing.T) {
	var nodeName = testHostname

	emptyEndpoint := func(ept *api.Endpoints) {
		ept.Subsets = []api.EndpointSubset{}
	}
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
		previousEndpoints         []*api.Endpoints
		currentEndpoints          []*api.Endpoints
		oldEndpoints              map[proxy.ServicePortName][]*endpointsInfo
		expectedResult            map[proxy.ServicePortName][]*endpointsInfo
		expectedStaleEndpoints    []endpointServicePair
		expectedStaleServiceNames map[proxy.ServicePortName]bool
		expectedHealthchecks      map[types.NamespacedName]int
	}{{
		// Case[0]: nothing
		oldEndpoints:              map[proxy.ServicePortName][]*endpointsInfo{},
		expectedResult:            map[proxy.ServicePortName][]*endpointsInfo{},
		expectedStaleEndpoints:    []endpointServicePair{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		// Case[1]: no change, unnamed port
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", unnamedPort),
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", unnamedPort),
		},
		oldEndpoints: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", ""): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", ""): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
		},
		expectedStaleEndpoints:    []endpointServicePair{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		// Case[2]: no change, named port, local
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPortLocal),
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPortLocal),
		},
		oldEndpoints: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: true},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: true},
			},
		},
		expectedStaleEndpoints:    []endpointServicePair{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
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
		oldEndpoints: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{endpoint: "1.1.1.2:12", isLocal: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{endpoint: "1.1.1.2:12", isLocal: false},
			},
		},
		expectedStaleEndpoints:    []endpointServicePair{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		// Case[4]: no change, multiple subsets, multiple ports, local
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsetsMultiplePortsLocal),
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsetsMultiplePortsLocal),
		},
		oldEndpoints: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{endpoint: "1.1.1.1:12", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p13"): {
				{endpoint: "1.1.1.3:13", isLocal: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{endpoint: "1.1.1.1:12", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p13"): {
				{endpoint: "1.1.1.3:13", isLocal: false},
			},
		},
		expectedStaleEndpoints:    []endpointServicePair{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 1,
		},
	}, {
		// Case[5]: no change, multiple endpoints, subsets, IPs, and ports
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsetsIPsPorts1),
			makeTestEndpoints("ns2", "ep2", multipleSubsetsIPsPorts2),
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsetsIPsPorts1),
			makeTestEndpoints("ns2", "ep2", multipleSubsetsIPsPorts2),
		},
		oldEndpoints: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
				{endpoint: "1.1.1.2:11", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{endpoint: "1.1.1.1:12", isLocal: false},
				{endpoint: "1.1.1.2:12", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p13"): {
				{endpoint: "1.1.1.3:13", isLocal: false},
				{endpoint: "1.1.1.4:13", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p14"): {
				{endpoint: "1.1.1.3:14", isLocal: false},
				{endpoint: "1.1.1.4:14", isLocal: true},
			},
			makeServicePortName("ns2", "ep2", "p21"): {
				{endpoint: "2.2.2.1:21", isLocal: false},
				{endpoint: "2.2.2.2:21", isLocal: true},
			},
			makeServicePortName("ns2", "ep2", "p22"): {
				{endpoint: "2.2.2.1:22", isLocal: false},
				{endpoint: "2.2.2.2:22", isLocal: true},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
				{endpoint: "1.1.1.2:11", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{endpoint: "1.1.1.1:12", isLocal: false},
				{endpoint: "1.1.1.2:12", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p13"): {
				{endpoint: "1.1.1.3:13", isLocal: false},
				{endpoint: "1.1.1.4:13", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p14"): {
				{endpoint: "1.1.1.3:14", isLocal: false},
				{endpoint: "1.1.1.4:14", isLocal: true},
			},
			makeServicePortName("ns2", "ep2", "p21"): {
				{endpoint: "2.2.2.1:21", isLocal: false},
				{endpoint: "2.2.2.2:21", isLocal: true},
			},
			makeServicePortName("ns2", "ep2", "p22"): {
				{endpoint: "2.2.2.1:22", isLocal: false},
				{endpoint: "2.2.2.2:22", isLocal: true},
			},
		},
		expectedStaleEndpoints:    []endpointServicePair{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
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
		oldEndpoints: map[proxy.ServicePortName][]*endpointsInfo{},
		expectedResult: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", ""): {
				{endpoint: "1.1.1.1:11", isLocal: true},
			},
		},
		expectedStaleEndpoints: []endpointServicePair{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{
			makeServicePortName("ns1", "ep1", ""): true,
		},
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
		oldEndpoints: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", ""): {
				{endpoint: "1.1.1.1:11", isLocal: true},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*endpointsInfo{},
		expectedStaleEndpoints: []endpointServicePair{{
			endpoint:        "1.1.1.1:11",
			servicePortName: makeServicePortName("ns1", "ep1", ""),
		}},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		// Case[8]: add an IP and port
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPort),
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPortsLocalNoLocal),
		},
		oldEndpoints: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
				{endpoint: "1.1.1.2:11", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{endpoint: "1.1.1.1:12", isLocal: false},
				{endpoint: "1.1.1.2:12", isLocal: true},
			},
		},
		expectedStaleEndpoints: []endpointServicePair{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{
			makeServicePortName("ns1", "ep1", "p12"): true,
		},
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
		oldEndpoints: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
				{endpoint: "1.1.1.2:11", isLocal: true},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{endpoint: "1.1.1.1:12", isLocal: false},
				{endpoint: "1.1.1.2:12", isLocal: true},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
		},
		expectedStaleEndpoints: []endpointServicePair{{
			endpoint:        "1.1.1.2:11",
			servicePortName: makeServicePortName("ns1", "ep1", "p11"),
		}, {
			endpoint:        "1.1.1.1:12",
			servicePortName: makeServicePortName("ns1", "ep1", "p12"),
		}, {
			endpoint:        "1.1.1.2:12",
			servicePortName: makeServicePortName("ns1", "ep1", "p12"),
		}},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		// Case[10]: add a subset
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPort),
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", multipleSubsetsWithLocal),
		},
		oldEndpoints: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{endpoint: "1.1.1.2:12", isLocal: true},
			},
		},
		expectedStaleEndpoints: []endpointServicePair{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{
			makeServicePortName("ns1", "ep1", "p12"): true,
		},
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
		oldEndpoints: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{endpoint: "1.1.1.2:12", isLocal: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
		},
		expectedStaleEndpoints: []endpointServicePair{{
			endpoint:        "1.1.1.2:12",
			servicePortName: makeServicePortName("ns1", "ep1", "p12"),
		}},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		// Case[12]: rename a port
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPort),
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPortRenamed),
		},
		oldEndpoints: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11-2"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
		},
		expectedStaleEndpoints: []endpointServicePair{{
			endpoint:        "1.1.1.1:11",
			servicePortName: makeServicePortName("ns1", "ep1", "p11"),
		}},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{
			makeServicePortName("ns1", "ep1", "p11-2"): true,
		},
		expectedHealthchecks: map[types.NamespacedName]int{},
	}, {
		// Case[13]: renumber a port
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPort),
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", namedPortRenumbered),
		},
		oldEndpoints: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:22", isLocal: false},
			},
		},
		expectedStaleEndpoints: []endpointServicePair{{
			endpoint:        "1.1.1.1:11",
			servicePortName: makeServicePortName("ns1", "ep1", "p11"),
		}},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
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
		oldEndpoints: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
			makeServicePortName("ns2", "ep2", "p22"): {
				{endpoint: "2.2.2.2:22", isLocal: true},
				{endpoint: "2.2.2.22:22", isLocal: true},
			},
			makeServicePortName("ns2", "ep2", "p23"): {
				{endpoint: "2.2.2.3:23", isLocal: true},
			},
			makeServicePortName("ns4", "ep4", "p44"): {
				{endpoint: "4.4.4.4:44", isLocal: true},
				{endpoint: "4.4.4.5:44", isLocal: true},
			},
			makeServicePortName("ns4", "ep4", "p45"): {
				{endpoint: "4.4.4.6:45", isLocal: true},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{endpoint: "1.1.1.1:11", isLocal: false},
				{endpoint: "1.1.1.11:11", isLocal: false},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{endpoint: "1.1.1.2:12", isLocal: false},
			},
			makeServicePortName("ns1", "ep1", "p122"): {
				{endpoint: "1.1.1.2:122", isLocal: false},
			},
			makeServicePortName("ns3", "ep3", "p33"): {
				{endpoint: "3.3.3.3:33", isLocal: false},
			},
			makeServicePortName("ns4", "ep4", "p44"): {
				{endpoint: "4.4.4.4:44", isLocal: true},
			},
		},
		expectedStaleEndpoints: []endpointServicePair{{
			endpoint:        "2.2.2.2:22",
			servicePortName: makeServicePortName("ns2", "ep2", "p22"),
		}, {
			endpoint:        "2.2.2.22:22",
			servicePortName: makeServicePortName("ns2", "ep2", "p22"),
		}, {
			endpoint:        "2.2.2.3:23",
			servicePortName: makeServicePortName("ns2", "ep2", "p23"),
		}, {
			endpoint:        "4.4.4.5:44",
			servicePortName: makeServicePortName("ns4", "ep4", "p44"),
		}, {
			endpoint:        "4.4.4.6:45",
			servicePortName: makeServicePortName("ns4", "ep4", "p45"),
		}},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{
			makeServicePortName("ns1", "ep1", "p12"):  true,
			makeServicePortName("ns1", "ep1", "p122"): true,
			makeServicePortName("ns3", "ep3", "p33"):  true,
		},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns4", "ep4"): 1,
		},
	}, {
		// Case[15]: change from 0 endpoint address to 1 unnamed port
		previousEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", emptyEndpoint),
		},
		currentEndpoints: []*api.Endpoints{
			makeTestEndpoints("ns1", "ep1", unnamedPort),
		},
		oldEndpoints: map[proxy.ServicePortName][]*endpointsInfo{},
		expectedResult: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", ""): {
				{endpoint: "1.1.1.1:11", isLocal: false},
			},
		},
		expectedStaleEndpoints: []endpointServicePair{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{
			makeServicePortName("ns1", "ep1", ""): true,
		},
		expectedHealthchecks: map[types.NamespacedName]int{},
	},
	}

	for tci, tc := range testCases {

		fp := NewFakeProxier()
		fp.hostname = nodeName

		// First check that after adding all previous versions of endpoints,
		// the fp.oldEndpoints is as we expect.
		for i := range tc.previousEndpoints {
			if tc.previousEndpoints[i] != nil {
				fp.OnEndpointsAdd(tc.previousEndpoints[i])
			}
		}
		updateEndpointsMap(fp.endpointsMap, &fp.endpointsChanges, fp.hostname)
		compareEndpointsMaps(t, tci, fp.endpointsMap, tc.oldEndpoints)

		// Now let's call appropriate handlers to get to state we want to be.
		if len(tc.previousEndpoints) != len(tc.currentEndpoints) {
			t.Fatalf("[%d] different lengths of previous and current endpoints", tci)
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
		result := updateEndpointsMap(fp.endpointsMap, &fp.endpointsChanges, fp.hostname)
		newMap := fp.endpointsMap
		compareEndpointsMaps(t, tci, newMap, tc.expectedResult)
		if len(result.staleEndpoints) != len(tc.expectedStaleEndpoints) {
			t.Errorf("[%d] expected %d staleEndpoints, got %d: %v", tci, len(tc.expectedStaleEndpoints), len(result.staleEndpoints), result.staleEndpoints)
		}
		for _, x := range tc.expectedStaleEndpoints {
			if result.staleEndpoints[x] != true {
				t.Errorf("[%d] expected staleEndpoints[%v], but didn't find it: %v", tci, x, result.staleEndpoints)
			}
		}
		if len(result.staleServiceNames) != len(tc.expectedStaleServiceNames) {
			t.Errorf("[%d] expected %d staleServiceNames, got %d: %v", tci, len(tc.expectedStaleServiceNames), len(result.staleServiceNames), result.staleServiceNames)
		}
		for svcName := range tc.expectedStaleServiceNames {
			if result.staleServiceNames[svcName] != true {
				t.Errorf("[%d] expected staleServiceNames[%v], but didn't find it: %v", tci, svcName, result.staleServiceNames)
			}
		}
		if !reflect.DeepEqual(result.hcEndpoints, tc.expectedHealthchecks) {
			t.Errorf("[%d] expected healthchecks %v, got %v", tci, tc.expectedHealthchecks, result.hcEndpoints)
		}
	}
}

// TODO(thockin): add *more* tests for syncProxyRules() or break it down further and test the pieces.
