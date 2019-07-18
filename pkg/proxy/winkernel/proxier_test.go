// +build windows

/*
Copyright 2018 The Kubernetes Authors.

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
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/proxy"

	"net"
	"strings"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const testHostName = "test-hostname"
const macAddress = "00-11-22-33-44-55"
const clusterCIDR = "192.168.1.0/24"
const destinationPrefix = "192.168.2.0/24"
const providerAddress = "10.0.0.3"
const guid = "123ABC"

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

type fakeHNS struct{}

func newFakeHNS() *fakeHNS {
	return &fakeHNS{}
}
func (hns fakeHNS) getNetworkByName(name string) (*hnsNetworkInfo, error) {
	var remoteSubnets []*remoteSubnetInfo
	rs := &remoteSubnetInfo{
		destinationPrefix: destinationPrefix,
		isolationID:       4096,
		providerAddress:   providerAddress,
		drMacAddress:      macAddress,
	}
	remoteSubnets = append(remoteSubnets, rs)
	return &hnsNetworkInfo{
		id:            strings.ToUpper(guid),
		name:          name,
		networkType:   "Overlay",
		remoteSubnets: remoteSubnets,
	}, nil
}
func (hns fakeHNS) getEndpointByID(id string) (*endpointsInfo, error) {
	return nil, nil
}
func (hns fakeHNS) getEndpointByIpAddress(ip string, networkName string) (*endpointsInfo, error) {
	_, ipNet, _ := net.ParseCIDR(destinationPrefix)

	if ipNet.Contains(net.ParseIP(ip)) {
		return &endpointsInfo{
			ip:         ip,
			isLocal:    true,
			macAddress: macAddress,
			hnsID:      guid,
			hns:        hns,
		}, nil
	}
	return nil, nil

}
func (hns fakeHNS) createEndpoint(ep *endpointsInfo, networkName string) (*endpointsInfo, error) {
	return &endpointsInfo{
		ip:         ep.ip,
		isLocal:    ep.isLocal,
		macAddress: ep.macAddress,
		hnsID:      guid,
		hns:        hns,
	}, nil
}
func (hns fakeHNS) deleteEndpoint(hnsID string) error {
	return nil
}
func (hns fakeHNS) getLoadBalancer(endpoints []endpointsInfo, flags loadBalancerFlags, sourceVip string, vip string, protocol uint16, internalPort uint16, externalPort uint16) (*loadBalancerInfo, error) {
	return &loadBalancerInfo{
		hnsID: guid,
	}, nil
}
func (hns fakeHNS) deleteLoadBalancer(hnsID string) error {
	return nil
}
func NewFakeProxier(syncPeriod time.Duration, minSyncPeriod time.Duration, clusterCIDR string, hostname string, nodeIP net.IP, networkType string) *Proxier {
	sourceVip := "192.168.1.2"
	hnsNetworkInfo := &hnsNetworkInfo{
		name:        "TestNetwork",
		networkType: networkType,
	}
	proxier := &Proxier{
		portsMap:         make(map[localPort]closeable),
		serviceMap:       make(proxyServiceMap),
		serviceChanges:   newServiceChangeMap(),
		endpointsMap:     make(proxyEndpointsMap),
		endpointsChanges: newEndpointsChangeMap(hostname),
		clusterCIDR:      clusterCIDR,
		hostname:         testHostName,
		nodeIP:           nodeIP,
		healthChecker:    newFakeHealthChecker(),
		network:          *hnsNetworkInfo,
		sourceVip:        sourceVip,
		hostMac:          macAddress,
		isDSR:            false,
		hns:              newFakeHNS(),
	}
	return proxier
}

func TestCreateServiceVip(t *testing.T) {
	syncPeriod := 30 * time.Second
	proxier := NewFakeProxier(syncPeriod, syncPeriod, clusterCIDR, "testhost", net.ParseIP("10.0.0.1"), "Overlay")
	if proxier == nil {
		t.Error()
	}

	svcIP := "10.20.30.41"
	svcPort := 80
	svcNodePort := 3001
	svcExternalIPs := "50.60.70.81"
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
	}
	timeoutSeconds := v1.DefaultClientIPServiceAffinitySeconds

	makeServiceMap(proxier,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = "NodePort"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.ExternalIPs = []string{svcExternalIPs}
			svc.Spec.SessionAffinity = v1.ServiceAffinityClientIP
			svc.Spec.SessionAffinityConfig = &v1.SessionAffinityConfig{
				ClientIP: &v1.ClientIPConfig{
					TimeoutSeconds: &timeoutSeconds,
				},
			}
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     svcPortName.Port,
				Port:     int32(svcPort),
				Protocol: v1.ProtocolTCP,
				NodePort: int32(svcNodePort),
			}}
		}),
	)
	makeEndpointsMap(proxier)

	proxier.syncProxyRules()

	if proxier.serviceMap[svcPortName].remoteEndpoint == nil {
		t.Error()
	}
	if proxier.serviceMap[svcPortName].remoteEndpoint.ip != svcIP {
		t.Error()
	}
}
func TestCreateRemoteEndpointOverlay(t *testing.T) {
	syncPeriod := 30 * time.Second
	proxier := NewFakeProxier(syncPeriod, syncPeriod, clusterCIDR, "testhost", net.ParseIP("10.0.0.1"), "Overlay")
	if proxier == nil {
		t.Error()
	}

	svcIP := "10.20.30.41"
	svcPort := 80
	svcNodePort := 3001
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
	}

	makeServiceMap(proxier,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = "NodePort"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     svcPortName.Port,
				Port:     int32(svcPort),
				Protocol: v1.ProtocolTCP,
				NodePort: int32(svcNodePort),
			}}
		}),
	)
	makeEndpointsMap(proxier,
		makeTestEndpoints(svcPortName.Namespace, svcPortName.Name, func(ept *v1.Endpoints) {
			ept.Subsets = []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{
					IP: epIpAddressRemote,
				}},
				Ports: []v1.EndpointPort{{
					Name: svcPortName.Port,
					Port: int32(svcPort),
				}},
			}}
		}),
	)

	proxier.syncProxyRules()

	if proxier.endpointsMap[svcPortName][0].hnsID != guid {
		t.Errorf("%v does not match %v", proxier.endpointsMap[svcPortName][0].hnsID, guid)
	}
}
func TestCreateRemoteEndpointL2Bridge(t *testing.T) {
	syncPeriod := 30 * time.Second
	proxier := NewFakeProxier(syncPeriod, syncPeriod, clusterCIDR, "testhost", net.ParseIP("10.0.0.1"), "L2Bridge")
	if proxier == nil {
		t.Error()
	}

	svcIP := "10.20.30.41"
	svcPort := 80
	svcNodePort := 3001
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
	}

	makeServiceMap(proxier,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = "NodePort"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     svcPortName.Port,
				Port:     int32(svcPort),
				Protocol: v1.ProtocolTCP,
				NodePort: int32(svcNodePort),
			}}
		}),
	)
	makeEndpointsMap(proxier,
		makeTestEndpoints(svcPortName.Namespace, svcPortName.Name, func(ept *v1.Endpoints) {
			ept.Subsets = []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{
					IP: epIpAddressRemote,
				}},
				Ports: []v1.EndpointPort{{
					Name: svcPortName.Port,
					Port: int32(svcPort),
				}},
			}}
		}),
	)

	proxier.syncProxyRules()

	if proxier.endpointsMap[svcPortName][0].hnsID != guid {
		t.Errorf("%v does not match %v", proxier.endpointsMap[svcPortName][0].hnsID, guid)
	}
}
func TestCreateLoadBalancer(t *testing.T) {
	syncPeriod := 30 * time.Second
	proxier := NewFakeProxier(syncPeriod, syncPeriod, clusterCIDR, "testhost", net.ParseIP("10.0.0.1"), "Overlay")
	if proxier == nil {
		t.Error()
	}

	svcIP := "10.20.30.41"
	svcPort := 80
	svcNodePort := 3001
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
	}

	makeServiceMap(proxier,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = "NodePort"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     svcPortName.Port,
				Port:     int32(svcPort),
				Protocol: v1.ProtocolTCP,
				NodePort: int32(svcNodePort),
			}}
		}),
	)
	makeEndpointsMap(proxier,
		makeTestEndpoints(svcPortName.Namespace, svcPortName.Name, func(ept *v1.Endpoints) {
			ept.Subsets = []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{
					IP: epIpAddressRemote,
				}},
				Ports: []v1.EndpointPort{{
					Name: svcPortName.Port,
					Port: int32(svcPort),
				}},
			}}
		}),
	)

	proxier.syncProxyRules()

	if proxier.serviceMap[svcPortName].hnsID != guid {
		t.Errorf("%v does not match %v", proxier.serviceMap[svcPortName].hnsID, guid)
	}
}
func makeNSN(namespace, name string) types.NamespacedName {
	return types.NamespacedName{Namespace: namespace, Name: name}
}
func makeServiceMap(proxier *Proxier, allServices ...*v1.Service) {
	for i := range allServices {
		proxier.OnServiceAdd(allServices[i])
	}

	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	proxier.servicesSynced = true
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

func makeEndpointsMap(proxier *Proxier, allEndpoints ...*v1.Endpoints) {
	for i := range allEndpoints {
		proxier.OnEndpointsAdd(allEndpoints[i])
	}

	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	proxier.endpointsSynced = true
}

func makeTestEndpoints(namespace, name string, eptFunc func(*v1.Endpoints)) *v1.Endpoints {
	ept := &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
	}
	eptFunc(ept)
	return ept
}
