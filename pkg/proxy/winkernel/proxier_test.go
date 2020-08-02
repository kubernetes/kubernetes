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
	"fmt"
	"k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/proxy/healthcheck"
	utilproxy "k8s.io/kubernetes/pkg/proxy/util"
	utilpointer "k8s.io/utils/pointer"
	"net"
	"strings"
	"testing"
	"time"
)

const testHostName = "test-hostname"
const macAddress = "00-11-22-33-44-55"
const clusterCIDR = "192.168.1.0/24"
const destinationPrefix = "192.168.2.0/24"
const providerAddress = "10.0.0.3"
const guid = "123ABC"

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
			isLocal:    false,
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
func NewFakeProxier(syncPeriod time.Duration, minSyncPeriod time.Duration, clusterCIDR string, hostname string, nodeIP net.IP, networkType string, endpointSliceEnabled bool) *Proxier {
	sourceVip := "192.168.1.2"
	hnsNetworkInfo := &hnsNetworkInfo{
		id:          strings.ToUpper(guid),
		name:        "TestNetwork",
		networkType: networkType,
	}
	proxier := &Proxier{
		portsMap:            make(map[utilproxy.LocalPort]utilproxy.Closeable),
		serviceMap:          make(proxy.ServiceMap),
		endpointsMap:        make(proxy.EndpointsMap),
		clusterCIDR:         clusterCIDR,
		hostname:            testHostName,
		nodeIP:              nodeIP,
		serviceHealthServer: healthcheck.NewFakeServiceHealthServer(),
		network:             *hnsNetworkInfo,
		sourceVip:           sourceVip,
		hostMac:             macAddress,
		isDSR:               false,
		hns:                 newFakeHNS(),
		endPointsRefCount:   make(endPointsReferenceCountMap),
	}

	isIPv6 := false
	serviceChanges := proxy.NewServiceChangeTracker(proxier.newServiceInfo, &isIPv6, nil, proxier.serviceMapChange)
	endpointChangeTracker := proxy.NewEndpointChangeTracker(hostname, proxier.newEndpointInfo, &isIPv6, nil, endpointSliceEnabled, proxier.endpointsMapChange)
	proxier.endpointsChanges = endpointChangeTracker
	proxier.serviceChanges = serviceChanges

	return proxier
}

func TestCreateServiceVip(t *testing.T) {
	syncPeriod := 30 * time.Second
	proxier := NewFakeProxier(syncPeriod, syncPeriod, clusterCIDR, "testhost", net.ParseIP("10.0.0.1"), "Overlay", false)
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
		Protocol:       v1.ProtocolTCP,
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
	proxier.setInitialized(true)
	proxier.syncProxyRules()

	svc := proxier.serviceMap[svcPortName]
	svcInfo, ok := svc.(*serviceInfo)
	if !ok {
		t.Errorf("Failed to cast serviceInfo %q", svcPortName.String())

	} else {
		if svcInfo.remoteEndpoint == nil {
			t.Error()
		}
		if svcInfo.remoteEndpoint.ip != svcIP {
			t.Error()
		}
	}
}
func TestCreateRemoteEndpointOverlay(t *testing.T) {
	syncPeriod := 30 * time.Second
	proxier := NewFakeProxier(syncPeriod, syncPeriod, clusterCIDR, "testhost", net.ParseIP("10.0.0.1"), "Overlay", false)
	if proxier == nil {
		t.Error()
	}

	svcIP := "10.20.30.41"
	svcPort := 80
	svcNodePort := 3001
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
		Protocol:       v1.ProtocolTCP,
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
					Name:     svcPortName.Port,
					Port:     int32(svcPort),
					Protocol: v1.ProtocolTCP,
				}},
			}}
		}),
	)
	proxier.setInitialized(true)
	proxier.syncProxyRules()

	ep := proxier.endpointsMap[svcPortName][0]
	epInfo, ok := ep.(*endpointsInfo)
	if !ok {
		t.Errorf("Failed to cast endpointsInfo %q", svcPortName.String())

	} else {
		if epInfo.hnsID != guid {
			t.Errorf("%v does not match %v", epInfo.hnsID, guid)
		}
	}

	if *proxier.endPointsRefCount[guid] <= 0 {
		t.Errorf("RefCount not incremented. Current value: %v", *proxier.endPointsRefCount[guid])
	}

	if *proxier.endPointsRefCount[guid] != *epInfo.refCount {
		t.Errorf("Global refCount: %v does not match endpoint refCount: %v", *proxier.endPointsRefCount[guid], *epInfo.refCount)
	}
}
func TestCreateRemoteEndpointL2Bridge(t *testing.T) {
	syncPeriod := 30 * time.Second
	proxier := NewFakeProxier(syncPeriod, syncPeriod, clusterCIDR, "testhost", net.ParseIP("10.0.0.1"), "L2Bridge", false)
	if proxier == nil {
		t.Error()
	}

	svcIP := "10.20.30.41"
	svcPort := 80
	svcNodePort := 3001
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
		Protocol:       v1.ProtocolTCP,
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
					Name:     svcPortName.Port,
					Port:     int32(svcPort),
					Protocol: v1.ProtocolTCP,
				}},
			}}
		}),
	)
	proxier.setInitialized(true)
	proxier.syncProxyRules()
	ep := proxier.endpointsMap[svcPortName][0]
	epInfo, ok := ep.(*endpointsInfo)
	if !ok {
		t.Errorf("Failed to cast endpointsInfo %q", svcPortName.String())

	} else {
		if epInfo.hnsID != guid {
			t.Errorf("%v does not match %v", epInfo.hnsID, guid)
		}
	}

	if *proxier.endPointsRefCount[guid] <= 0 {
		t.Errorf("RefCount not incremented. Current value: %v", *proxier.endPointsRefCount[guid])
	}

	if *proxier.endPointsRefCount[guid] != *epInfo.refCount {
		t.Errorf("Global refCount: %v does not match endpoint refCount: %v", *proxier.endPointsRefCount[guid], *epInfo.refCount)
	}
}
func TestSharedRemoteEndpointDelete(t *testing.T) {
	syncPeriod := 30 * time.Second
	proxier := NewFakeProxier(syncPeriod, syncPeriod, clusterCIDR, "testhost", net.ParseIP("10.0.0.1"), "L2Bridge", false)
	if proxier == nil {
		t.Error()
	}

	svcIP1 := "10.20.30.41"
	svcPort1 := 80
	svcNodePort1 := 3001
	svcPortName1 := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
		Protocol:       v1.ProtocolTCP,
	}

	svcIP2 := "10.20.30.42"
	svcPort2 := 80
	svcNodePort2 := 3002
	svcPortName2 := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc2"),
		Port:           "p80",
		Protocol:       v1.ProtocolTCP,
	}

	makeServiceMap(proxier,
		makeTestService(svcPortName1.Namespace, svcPortName1.Name, func(svc *v1.Service) {
			svc.Spec.Type = "NodePort"
			svc.Spec.ClusterIP = svcIP1
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     svcPortName1.Port,
				Port:     int32(svcPort1),
				Protocol: v1.ProtocolTCP,
				NodePort: int32(svcNodePort1),
			}}
		}),
		makeTestService(svcPortName2.Namespace, svcPortName2.Name, func(svc *v1.Service) {
			svc.Spec.Type = "NodePort"
			svc.Spec.ClusterIP = svcIP2
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     svcPortName2.Port,
				Port:     int32(svcPort2),
				Protocol: v1.ProtocolTCP,
				NodePort: int32(svcNodePort2),
			}}
		}),
	)
	makeEndpointsMap(proxier,
		makeTestEndpoints(svcPortName1.Namespace, svcPortName1.Name, func(ept *v1.Endpoints) {
			ept.Subsets = []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{
					IP: epIpAddressRemote,
				}},
				Ports: []v1.EndpointPort{{
					Name:     svcPortName1.Port,
					Port:     int32(svcPort1),
					Protocol: v1.ProtocolTCP,
				}},
			}}
		}),
		makeTestEndpoints(svcPortName2.Namespace, svcPortName2.Name, func(ept *v1.Endpoints) {
			ept.Subsets = []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{
					IP: epIpAddressRemote,
				}},
				Ports: []v1.EndpointPort{{
					Name:     svcPortName2.Port,
					Port:     int32(svcPort2),
					Protocol: v1.ProtocolTCP,
				}},
			}}
		}),
	)
	proxier.setInitialized(true)
	proxier.syncProxyRules()
	ep := proxier.endpointsMap[svcPortName1][0]
	epInfo, ok := ep.(*endpointsInfo)
	if !ok {
		t.Errorf("Failed to cast endpointsInfo %q", svcPortName1.String())

	} else {
		if epInfo.hnsID != guid {
			t.Errorf("%v does not match %v", epInfo.hnsID, guid)
		}
	}

	if *proxier.endPointsRefCount[guid] != 2 {
		t.Errorf("RefCount not incremented. Current value: %v", *proxier.endPointsRefCount[guid])
	}

	if *proxier.endPointsRefCount[guid] != *epInfo.refCount {
		t.Errorf("Global refCount: %v does not match endpoint refCount: %v", *proxier.endPointsRefCount[guid], *epInfo.refCount)
	}

	proxier.setInitialized(false)
	deleteServices(proxier,
		makeTestService(svcPortName2.Namespace, svcPortName2.Name, func(svc *v1.Service) {
			svc.Spec.Type = "NodePort"
			svc.Spec.ClusterIP = svcIP2
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     svcPortName2.Port,
				Port:     int32(svcPort2),
				Protocol: v1.ProtocolTCP,
				NodePort: int32(svcNodePort2),
			}}
		}),
	)

	deleteEndpoints(proxier,
		makeTestEndpoints(svcPortName2.Namespace, svcPortName2.Name, func(ept *v1.Endpoints) {
			ept.Subsets = []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{
					IP: epIpAddressRemote,
				}},
				Ports: []v1.EndpointPort{{
					Name:     svcPortName2.Port,
					Port:     int32(svcPort2),
					Protocol: v1.ProtocolTCP,
				}},
			}}
		}),
	)

	proxier.setInitialized(true)
	proxier.syncProxyRules()

	ep = proxier.endpointsMap[svcPortName1][0]
	epInfo, ok = ep.(*endpointsInfo)
	if !ok {
		t.Errorf("Failed to cast endpointsInfo %q", svcPortName1.String())

	} else {
		if epInfo.hnsID != guid {
			t.Errorf("%v does not match %v", epInfo.hnsID, guid)
		}
	}

	if *epInfo.refCount != 1 {
		t.Errorf("Incorrect Refcount. Current value: %v", *epInfo.refCount)
	}

	if *proxier.endPointsRefCount[guid] != *epInfo.refCount {
		t.Errorf("Global refCount: %v does not match endpoint refCount: %v", *proxier.endPointsRefCount[guid], *epInfo.refCount)
	}
}
func TestSharedRemoteEndpointUpdate(t *testing.T) {
	syncPeriod := 30 * time.Second
	proxier := NewFakeProxier(syncPeriod, syncPeriod, clusterCIDR, "testhost", net.ParseIP("10.0.0.1"), "L2Bridge", false)
	if proxier == nil {
		t.Error()
	}

	svcIP1 := "10.20.30.41"
	svcPort1 := 80
	svcNodePort1 := 3001
	svcPortName1 := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
		Protocol:       v1.ProtocolTCP,
	}

	svcIP2 := "10.20.30.42"
	svcPort2 := 80
	svcNodePort2 := 3002
	svcPortName2 := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc2"),
		Port:           "p80",
		Protocol:       v1.ProtocolTCP,
	}

	makeServiceMap(proxier,
		makeTestService(svcPortName1.Namespace, svcPortName1.Name, func(svc *v1.Service) {
			svc.Spec.Type = "NodePort"
			svc.Spec.ClusterIP = svcIP1
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     svcPortName1.Port,
				Port:     int32(svcPort1),
				Protocol: v1.ProtocolTCP,
				NodePort: int32(svcNodePort1),
			}}
		}),
		makeTestService(svcPortName2.Namespace, svcPortName2.Name, func(svc *v1.Service) {
			svc.Spec.Type = "NodePort"
			svc.Spec.ClusterIP = svcIP2
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     svcPortName2.Port,
				Port:     int32(svcPort2),
				Protocol: v1.ProtocolTCP,
				NodePort: int32(svcNodePort2),
			}}
		}),
	)

	makeEndpointsMap(proxier,
		makeTestEndpoints(svcPortName1.Namespace, svcPortName1.Name, func(ept *v1.Endpoints) {
			ept.Subsets = []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{
					IP: epIpAddressRemote,
				}},
				Ports: []v1.EndpointPort{{
					Name:     svcPortName1.Port,
					Port:     int32(svcPort1),
					Protocol: v1.ProtocolTCP,
				}},
			}}
		}),
		makeTestEndpoints(svcPortName2.Namespace, svcPortName2.Name, func(ept *v1.Endpoints) {
			ept.Subsets = []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{
					IP: epIpAddressRemote,
				}},
				Ports: []v1.EndpointPort{{
					Name:     svcPortName2.Port,
					Port:     int32(svcPort2),
					Protocol: v1.ProtocolTCP,
				}},
			}}
		}),
	)
	proxier.setInitialized(true)
	proxier.syncProxyRules()
	ep := proxier.endpointsMap[svcPortName1][0]
	epInfo, ok := ep.(*endpointsInfo)
	if !ok {
		t.Errorf("Failed to cast endpointsInfo %q", svcPortName1.String())

	} else {
		if epInfo.hnsID != guid {
			t.Errorf("%v does not match %v", epInfo.hnsID, guid)
		}
	}

	if *proxier.endPointsRefCount[guid] != 2 {
		t.Errorf("RefCount not incremented. Current value: %v", *proxier.endPointsRefCount[guid])
	}

	if *proxier.endPointsRefCount[guid] != *epInfo.refCount {
		t.Errorf("Global refCount: %v does not match endpoint refCount: %v", *proxier.endPointsRefCount[guid], *epInfo.refCount)
	}

	proxier.setInitialized(false)

	proxier.OnServiceUpdate(
		makeTestService(svcPortName1.Namespace, svcPortName1.Name, func(svc *v1.Service) {
			svc.Spec.Type = "NodePort"
			svc.Spec.ClusterIP = svcIP1
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     svcPortName1.Port,
				Port:     int32(svcPort1),
				Protocol: v1.ProtocolTCP,
				NodePort: int32(svcNodePort1),
			}}
		}),
		makeTestService(svcPortName1.Namespace, svcPortName1.Name, func(svc *v1.Service) {
			svc.Spec.Type = "NodePort"
			svc.Spec.ClusterIP = svcIP1
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     svcPortName1.Port,
				Port:     int32(svcPort1),
				Protocol: v1.ProtocolTCP,
				NodePort: int32(3003),
			}}
		}))

	proxier.OnEndpointsUpdate(
		makeTestEndpoints(svcPortName1.Namespace, svcPortName1.Name, func(ept *v1.Endpoints) {
			ept.Subsets = []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{
					IP: epIpAddressRemote,
				}},
				Ports: []v1.EndpointPort{{
					Name:     svcPortName1.Port,
					Port:     int32(svcPort1),
					Protocol: v1.ProtocolTCP,
				}},
			}}
		}),
		makeTestEndpoints(svcPortName1.Namespace, svcPortName1.Name, func(ept *v1.Endpoints) {
			ept.Subsets = []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{
					IP: epIpAddressRemote,
				}},
				Ports: []v1.EndpointPort{
					{
						Name:     svcPortName1.Port,
						Port:     int32(svcPort1),
						Protocol: v1.ProtocolTCP,
					},
					{
						Name:     "p443",
						Port:     int32(443),
						Protocol: v1.ProtocolTCP,
					}},
			}}
		}))

	proxier.mu.Lock()
	proxier.endpointsSynced = true
	proxier.mu.Unlock()

	proxier.setInitialized(true)
	proxier.syncProxyRules()

	ep = proxier.endpointsMap[svcPortName1][0]
	epInfo, ok = ep.(*endpointsInfo)

	if !ok {
		t.Errorf("Failed to cast endpointsInfo %q", svcPortName1.String())

	} else {
		if epInfo.hnsID != guid {
			t.Errorf("%v does not match %v", epInfo.hnsID, guid)
		}
	}

	if *epInfo.refCount != 2 {
		t.Errorf("Incorrect refcount. Current value: %v", *epInfo.refCount)
	}

	if *proxier.endPointsRefCount[guid] != *epInfo.refCount {
		t.Errorf("Global refCount: %v does not match endpoint refCount: %v", *proxier.endPointsRefCount[guid], *epInfo.refCount)
	}
}
func TestCreateLoadBalancer(t *testing.T) {
	syncPeriod := 30 * time.Second
	proxier := NewFakeProxier(syncPeriod, syncPeriod, clusterCIDR, "testhost", net.ParseIP("10.0.0.1"), "Overlay", false)
	if proxier == nil {
		t.Error()
	}

	svcIP := "10.20.30.41"
	svcPort := 80
	svcNodePort := 3001
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
		Protocol:       v1.ProtocolTCP,
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
					Name:     svcPortName.Port,
					Port:     int32(svcPort),
					Protocol: v1.ProtocolTCP,
				}},
			}}
		}),
	)

	proxier.setInitialized(true)
	proxier.syncProxyRules()

	svc := proxier.serviceMap[svcPortName]
	svcInfo, ok := svc.(*serviceInfo)
	if !ok {
		t.Errorf("Failed to cast serviceInfo %q", svcPortName.String())

	} else {
		if svcInfo.hnsID != guid {
			t.Errorf("%v does not match %v", svcInfo.hnsID, guid)
		}
	}

}
func TestEndpointSlice(t *testing.T) {
	syncPeriod := 30 * time.Second
	proxier := NewFakeProxier(syncPeriod, syncPeriod, clusterCIDR, "testhost", net.ParseIP("10.0.0.1"), "Overlay", true)
	if proxier == nil {
		t.Error()
	}

	proxier.servicesSynced = true
	proxier.endpointSlicesSynced = true

	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
		Protocol:       v1.ProtocolTCP,
	}

	proxier.OnServiceAdd(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: svcPortName.Name, Namespace: svcPortName.Namespace},
		Spec: v1.ServiceSpec{
			ClusterIP: "172.20.1.1",
			Selector:  map[string]string{"foo": "bar"},
			Ports:     []v1.ServicePort{{Name: svcPortName.Port, TargetPort: intstr.FromInt(80), Protocol: v1.ProtocolTCP}},
		},
	})

	// Add initial endpoint slice
	tcpProtocol := v1.ProtocolTCP
	endpointSlice := &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-1", svcPortName.Name),
			Namespace: svcPortName.Namespace,
			Labels:    map[string]string{discovery.LabelServiceName: svcPortName.Name},
		},
		Ports: []discovery.EndpointPort{{
			Name:     &svcPortName.Port,
			Port:     utilpointer.Int32Ptr(80),
			Protocol: &tcpProtocol,
		}},
		AddressType: discovery.AddressTypeIPv4,
		Endpoints: []discovery.Endpoint{{
			Addresses:  []string{"192.168.2.3"},
			Conditions: discovery.EndpointConditions{Ready: utilpointer.BoolPtr(true)},
			Topology:   map[string]string{"kubernetes.io/hostname": "testhost2"},
		}},
	}

	proxier.OnEndpointSliceAdd(endpointSlice)
	proxier.setInitialized(true)
	proxier.syncProxyRules()

	svc := proxier.serviceMap[svcPortName]
	svcInfo, ok := svc.(*serviceInfo)
	if !ok {
		t.Errorf("Failed to cast serviceInfo %q", svcPortName.String())

	} else {
		if svcInfo.hnsID != guid {
			t.Errorf("The Hns Loadbalancer Id %v does not match %v. ServicePortName %q", svcInfo.hnsID, guid, svcPortName.String())
		}
	}

	ep := proxier.endpointsMap[svcPortName][0]
	epInfo, ok := ep.(*endpointsInfo)
	if !ok {
		t.Errorf("Failed to cast endpointsInfo %q", svcPortName.String())

	} else {
		if epInfo.hnsID != guid {
			t.Errorf("Hns EndpointId %v does not match %v. ServicePortName %q", epInfo.hnsID, guid, svcPortName.String())
		}
	}
}
func TestNoopEndpointSlice(t *testing.T) {
	p := Proxier{}
	p.OnEndpointSliceAdd(&discovery.EndpointSlice{})
	p.OnEndpointSliceUpdate(&discovery.EndpointSlice{}, &discovery.EndpointSlice{})
	p.OnEndpointSliceDelete(&discovery.EndpointSlice{})
	p.OnEndpointSlicesSynced()
}

func TestFindRemoteSubnetProviderAddress(t *testing.T) {
	networkInfo, _ := newFakeHNS().getNetworkByName("TestNetwork")
	pa := networkInfo.findRemoteSubnetProviderAddress(providerAddress)

	if pa != providerAddress {
		t.Errorf("%v does not match %v", pa, providerAddress)
	}

	pa = networkInfo.findRemoteSubnetProviderAddress(epIpAddressRemote)

	if pa != providerAddress {
		t.Errorf("%v does not match %v", pa, providerAddress)
	}

	pa = networkInfo.findRemoteSubnetProviderAddress(serviceVip)

	if len(pa) != 0 {
		t.Errorf("Provider address is not empty as expected")
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
func deleteServices(proxier *Proxier, allServices ...*v1.Service) {
	for i := range allServices {
		proxier.OnServiceDelete(allServices[i])
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

func deleteEndpoints(proxier *Proxier, allEndpoints ...*v1.Endpoints) {
	for i := range allEndpoints {
		proxier.OnEndpointsDelete(allEndpoints[i])
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
