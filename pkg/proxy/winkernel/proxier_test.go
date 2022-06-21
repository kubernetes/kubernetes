//go:build windows
// +build windows

/*
Copyright 2021 The Kubernetes Authors.

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
	"strings"
	"testing"

	"github.com/Microsoft/hcsshim/hcn"
	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/proxy/healthcheck"
	"k8s.io/kubernetes/pkg/proxy/winkernel/hcntesting"
	netutils "k8s.io/utils/net"
	utilpointer "k8s.io/utils/pointer"
)

const (
	testHostName           = "test-hostname"
	macAddress             = "00-11-22-33-44-55"
	clusterCIDR            = "192.168.1.0/24"
	destinationPrefix      = "192.168.2.0/24"
	providerAddress        = "10.0.0.3"
	guid                   = "123ABC"
	svcIP                  = "10.20.30.41"
	svcPort                = 80
	svcNodePort            = 3001
	svcExternalIPs         = "50.60.70.81"
	svcLBIP                = "11.21.31.41"
	svcHealthCheckNodePort = 30000
	LbInternalPort         = 0x50
	LbExternalPort         = 0x50
	LbTCPProtocol          = 0x6
)

//
//syncPeriod 	  30 * time.Second
//minSyncPeriod   30 * time.Second
//clusterCIDR     "192.168.1.0/24"
//hostname		  "testhost"
//nodeIP		  "10.0.0.1"
//networkType     "overlay" or "l2bridge"

func NewFakeProxier(hcnutilsfake *hcnutils, networkType string) *Proxier {
	sourceVip := "192.168.1.2"
	hnsNetworkInfo := &hnsNetworkInfo{
		id:          strings.ToUpper(guid),
		name:        "TestNetwork",
		networkType: networkType,
	}
	proxier := &Proxier{
		serviceMap:          make(proxy.ServiceMap),
		endpointsMap:        make(proxy.EndpointsMap),
		clusterCIDR:         clusterCIDR,
		hostname:            testHostName,
		nodeIP:              netutils.ParseIPSloppy("10.0.0.1"),
		serviceHealthServer: healthcheck.NewFakeServiceHealthServer(),
		network:             *hnsNetworkInfo,
		sourceVip:           sourceVip,
		hostMac:             macAddress,
		isDSR:               true,
		hns:                 hcnutilsfake,
		endPointsRefCount:   make(endPointsReferenceCountMap),
	}

	serviceChanges := proxy.NewServiceChangeTracker(proxier.newServiceInfo, v1.IPv4Protocol, nil, proxier.serviceMapChange)
	endpointChangeTracker := proxy.NewEndpointChangeTracker("testhost", proxier.newEndpointInfo, v1.IPv4Protocol, nil, proxier.endpointsMapChange)
	proxier.endpointsChanges = endpointChangeTracker
	proxier.serviceChanges = serviceChanges

	return proxier
}

// The Tests from this file are partitioned into 3 stages:
// 1. Arrange (setting up the objects)
// 2. Act (executing methods on those objects)
// 3. Assert (testing the expected outcome)

// Each Stage is indicated by comments in the code (//Arrange, //Act, //Assert)

// TestCreateServiceVip verifies if the ServiceVip is successfully created as expected
func TestCreateServiceVip(t *testing.T) {
	// Arrange
	testHNS := NewHCNUtils(&hcntesting.FakeHCN{})
	proxier := NewFakeProxier(testHNS, NETWORK_TYPE_OVERLAY)
	if proxier == nil {
		t.Error()
	}

	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
		Protocol:       v1.ProtocolTCP,
	}
	timeoutSeconds := v1.DefaultClientIPServiceAffinitySeconds

	// Act
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
	proxier.setInitialized(true)
	proxier.syncProxyRules()

	// Assert
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

// TestCreateRemoteEndpointOverlay verifies if the remote endpoint is
// successfully created on a network with type OVERLAY
func TestCreateRemoteEndpointOverlay(t *testing.T) {
	// Arrange
	testHNS := NewHCNUtils(&hcntesting.FakeHCN{})
	proxier := NewFakeProxier(testHNS, NETWORK_TYPE_OVERLAY)
	if proxier == nil {
		t.Error()
	}

	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
		Protocol:       v1.ProtocolTCP,
	}
	tcpProtocol := v1.ProtocolTCP

	expectedEndpoint := &hcn.HostComputeEndpoint{
		Id:                 guid,
		HostComputeNetwork: guid,
		SchemaVersion: hcn.SchemaVersion{
			Major: 2,
			Minor: 0,
		},
		IpConfigurations: []hcn.IpConfig{{IpAddress: epIpAddressRemote, PrefixLength: 24}},
		Flags:            hcn.EndpointFlagsRemoteEndpoint,
	}

	// Act
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
	populateEndpointSlices(proxier,
		makeTestEndpointSlice(svcPortName.Namespace, svcPortName.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIpAddressRemote},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     utilpointer.StringPtr(svcPortName.Port),
				Port:     utilpointer.Int32(int32(svcPort)),
				Protocol: &tcpProtocol,
			}}
		}),
	)
	proxier.setInitialized(true)
	proxier.syncProxyRules()

	// Assert
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

	Endpoint, err := testHNS.hcninstance.GetEndpointByID(guid)

	if err != nil {
		t.Error(err)
	}

	diff := assertHCNDiff(*expectedEndpoint, *Endpoint)
	if diff != "" {
		t.Errorf("GetEndpointById(%s) returned a different Endpoint. Diff: %s ", expectedEndpoint.Id, diff)
	}
}

// TestCreateRemoteEndpointOverlay verifies if the remote endpoint is
// successfully created on a network with type L2BRIDGE
func TestCreateRemoteEndpointL2Bridge(t *testing.T) {
	// Arrange
	testHNS := NewHCNUtils(&hcntesting.FakeHCN{})
	proxier := NewFakeProxier(testHNS, NETWORK_TYPE_L2BRIDGE)
	if proxier == nil {
		t.Error()
	}

	tcpProtocol := v1.ProtocolTCP

	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
		Protocol:       tcpProtocol,
	}

	expectedEndpoint := &hcn.HostComputeEndpoint{
		Id:                 guid,
		HostComputeNetwork: guid,
		SchemaVersion: hcn.SchemaVersion{
			Major: 2,
			Minor: 0,
		},
		IpConfigurations: []hcn.IpConfig{{IpAddress: epIpAddressRemote, PrefixLength: 24}},
		Flags:            hcn.EndpointFlagsRemoteEndpoint,
	}

	// Act
	makeServiceMap(proxier,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = "NodePort"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     svcPortName.Port,
				Port:     int32(svcPort),
				Protocol: tcpProtocol,
				NodePort: int32(svcNodePort),
			}}
		}),
	)
	populateEndpointSlices(proxier,
		makeTestEndpointSlice(svcPortName.Namespace, svcPortName.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIpAddressRemote},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     utilpointer.String(svcPortName.Port),
				Port:     utilpointer.Int32(int32(svcPort)),
				Protocol: &tcpProtocol,
			}}
		}),
	)
	proxier.setInitialized(true)
	proxier.syncProxyRules()

	// Assert
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

	Endpoint, err := testHNS.hcninstance.GetEndpointByID(guid)

	if err != nil {
		t.Error(err)
	}

	diff := assertHCNDiff(*expectedEndpoint, *Endpoint)
	if diff != "" {
		t.Errorf("GetEndpointById(%s) returned a different Endpoint. Diff: %s ", expectedEndpoint.Id, diff)
	}
}

// TestSharedRemoteEndpointDelete tests if shared remote endpoints are
// deleted successfully
func TestSharedRemoteEndpointDelete(t *testing.T) {
	// Arrange
	tcpProtocol := v1.ProtocolTCP
	testHNS := NewHCNUtils(&hcntesting.FakeHCN{})
	proxier := NewFakeProxier(testHNS, NETWORK_TYPE_L2BRIDGE)
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

	// Act
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
	populateEndpointSlices(proxier,
		makeTestEndpointSlice(svcPortName1.Namespace, svcPortName1.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIpAddressRemote},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     utilpointer.StringPtr(svcPortName1.Port),
				Port:     utilpointer.Int32(int32(svcPort1)),
				Protocol: &tcpProtocol,
			}}
		}),
		makeTestEndpointSlice(svcPortName2.Namespace, svcPortName2.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIpAddressRemote},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     utilpointer.StringPtr(svcPortName2.Port),
				Port:     utilpointer.Int32(int32(svcPort2)),
				Protocol: &tcpProtocol,
			}}
		}),
	)
	proxier.setInitialized(true)
	proxier.syncProxyRules()

	// Assert
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

	// Act --- here we delete the shared endpoint created
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

	deleteEndpointSlices(proxier,
		makeTestEndpointSlice(svcPortName2.Namespace, svcPortName2.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIpAddressRemote},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     utilpointer.StringPtr(svcPortName2.Port),
				Port:     utilpointer.Int32(int32(svcPort2)),
				Protocol: &tcpProtocol,
			}}
		}),
	)

	proxier.setInitialized(true)
	proxier.syncProxyRules()

	// Assert --- here we verify that the endpoint was successfully deleted
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

// TestSharedRemoteEndpointUpdate tests if shared remote endpoints are
// updated successfully
func TestSharedRemoteEndpointUpdate(t *testing.T) {
	// Arrange  -- before update
	testHNS := NewHCNUtils(&hcntesting.FakeHCN{})
	proxier := NewFakeProxier(testHNS, NETWORK_TYPE_L2BRIDGE)
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

	expectedEndpoint := &hcn.HostComputeEndpoint{
		Id:                 guid,
		HostComputeNetwork: guid,
		SchemaVersion: hcn.SchemaVersion{
			Major: 2,
			Minor: 0,
		},
		IpConfigurations: []hcn.IpConfig{{IpAddress: epIpAddressRemote, PrefixLength: 24}},
		Flags:            hcn.EndpointFlagsRemoteEndpoint,
	}

	// Act   ---before update
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

	tcpProtocol := v1.ProtocolTCP
	populateEndpointSlices(proxier,
		makeTestEndpointSlice(svcPortName1.Namespace, svcPortName1.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIpAddressRemote},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     utilpointer.StringPtr(svcPortName1.Port),
				Port:     utilpointer.Int32(int32(svcPort1)),
				Protocol: &tcpProtocol,
			}}
		}),
		makeTestEndpointSlice(svcPortName2.Namespace, svcPortName2.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIpAddressRemote},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     utilpointer.StringPtr(svcPortName2.Port),
				Port:     utilpointer.Int32(int32(svcPort2)),
				Protocol: &tcpProtocol,
			}}
		}),
	)
	proxier.setInitialized(true)
	proxier.syncProxyRules()

	// Assert --- before update
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

	Endpoint, err := testHNS.hcninstance.GetEndpointByID(guid)

	if err != nil {
		t.Error(err)
	}

	diff := assertHCNDiff(*expectedEndpoint, *Endpoint)
	if diff != "" {
		t.Errorf("GetEndpointById(%s) returned a different Endpoint. Diff: %s ", expectedEndpoint.Id, diff)
	}

	// Arrange --- after update
	expectedEndpoint = &hcn.HostComputeEndpoint{
		Id:                 guid,
		HostComputeNetwork: guid,
		SchemaVersion: hcn.SchemaVersion{
			Major: 2,
			Minor: 0,
		},
		IpConfigurations: []hcn.IpConfig{{IpAddress: epIpAddressRemote, PrefixLength: 24}},
		Flags:            hcn.EndpointFlagsRemoteEndpoint,
	}

	// Act -- after update
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

	proxier.OnEndpointSliceUpdate(
		makeTestEndpointSlice(svcPortName1.Namespace, svcPortName1.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIpAddressRemote},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     utilpointer.StringPtr(svcPortName1.Port),
				Port:     utilpointer.Int32(int32(svcPort1)),
				Protocol: &tcpProtocol,
			}}
		}),
		makeTestEndpointSlice(svcPortName1.Namespace, svcPortName1.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIpAddressRemote},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     utilpointer.StringPtr(svcPortName1.Port),
				Port:     utilpointer.Int32(int32(svcPort1)),
				Protocol: &tcpProtocol,
			},
				{
					Name:     utilpointer.StringPtr("p443"),
					Port:     utilpointer.Int32(int32(443)),
					Protocol: &tcpProtocol,
				}}
		}))

	proxier.mu.Lock()
	proxier.endpointSlicesSynced = true
	proxier.mu.Unlock()

	proxier.setInitialized(true)
	proxier.syncProxyRules()

	// Assert --- after update
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

	Endpoint, err = testHNS.hcninstance.GetEndpointByID(guid)

	if err != nil {
		t.Error(err)
	}

	diff = assertHCNDiff(*expectedEndpoint, *Endpoint)
	if diff != "" {
		t.Errorf("GetEndpointById(%s) returned a different Endpoint. Diff: %s ", expectedEndpoint.Id, diff)
	}

}

// TestCreateLoadBalancer tests if simple LoadBalancers are created the way they are expected to
func TestCreateLoadBalancer(t *testing.T) {
	// Arrange
	tcpProtocol := v1.ProtocolTCP
	testHNS := NewHCNUtils(&hcntesting.FakeHCN{})
	proxier := NewFakeProxier(testHNS, NETWORK_TYPE_L2BRIDGE)
	if proxier == nil {
		t.Error()
	}

	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
		Protocol:       v1.ProtocolTCP,
	}

	expectedPortMapping := &hcn.LoadBalancerPortMapping{
		Protocol:     LbTCPProtocol,
		InternalPort: LbInternalPort,
		ExternalPort: LbExternalPort,
		Flags:        hcn.LoadBalancerPortMappingFlagsNone,
	}

	expectedPortMappings := []hcn.LoadBalancerPortMapping{*expectedPortMapping}

	expectedLoadBalancer := &hcn.HostComputeLoadBalancer{
		Id:                   guid,
		HostComputeEndpoints: []string{guid},
		SourceVIP:            sourceVip,
		SchemaVersion: hcn.SchemaVersion{
			Major: 2,
			Minor: 0,
		},
		FrontendVIPs: []string{svcIP},
		Flags:        hcn.LoadBalancerFlagsDSR,
		PortMappings: expectedPortMappings,
	}

	// Act
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
	populateEndpointSlices(proxier,
		makeTestEndpointSlice(svcPortName.Namespace, svcPortName.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIpAddressRemote},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     utilpointer.StringPtr(svcPortName.Port),
				Port:     utilpointer.Int32(int32(svcPort)),
				Protocol: &tcpProtocol,
			}}
		}),
	)

	proxier.setInitialized(true)
	proxier.syncProxyRules()

	// Assert
	svc := proxier.serviceMap[svcPortName]
	svcInfo, ok := svc.(*serviceInfo)
	if !ok {
		t.Errorf("Failed to cast serviceInfo %q", svcPortName.String())

	} else {
		if svcInfo.hnsID != guid {
			t.Errorf("%v does not match %v", svcInfo.hnsID, guid)
		}
	}

	LoadBalancer, err := testHNS.hcninstance.GetLoadBalancerByID(guid)
	if err != nil {
		t.Error(err)
	}

	diff := assertHCNDiff(*LoadBalancer, *expectedLoadBalancer)
	if diff != "" {
		t.Errorf("GetLoadBalancerByID(%s) returned a different LoadBalancer. Diff: %s ", expectedLoadBalancer.Id, diff)
	}
}

// TestCreateDsrLoadBalancer tests if simple LoadBalancers are created the way they are expected to
// and makes additional DSR checks
func TestCreateDsrLoadBalancer(t *testing.T) {
	// Arrange
	testHNS := NewHCNUtils(&hcntesting.FakeHCN{})
	proxier := NewFakeProxier(testHNS, NETWORK_TYPE_OVERLAY)
	if proxier == nil {
		t.Error()
	}

	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
		Protocol:       v1.ProtocolTCP,
	}

	expectedPortMapping := &hcn.LoadBalancerPortMapping{
		Protocol:     LbTCPProtocol,
		InternalPort: LbInternalPort,
		ExternalPort: LbExternalPort,
		Flags:        hcn.LoadBalancerPortMappingFlagsNone,
	}

	expectedPortMappings := []hcn.LoadBalancerPortMapping{*expectedPortMapping}

	expectedLoadBalancer := &hcn.HostComputeLoadBalancer{
		Id:                   guid,
		HostComputeEndpoints: []string{guid},
		SourceVIP:            sourceVip,
		SchemaVersion: hcn.SchemaVersion{
			Major: 2,
			Minor: 0,
		},
		FrontendVIPs: []string{svcLBIP},
		Flags:        hcn.LoadBalancerFlagsDSR,
		PortMappings: expectedPortMappings,
	}

	// Act
	makeServiceMap(proxier,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = "NodePort"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     svcPortName.Port,
				Port:     int32(svcPort),
				Protocol: v1.ProtocolTCP,
				NodePort: int32(svcNodePort),
			}}
			svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{
				IP: svcLBIP,
			}}
		}),
	)
	tcpProtocol := v1.ProtocolTCP
	populateEndpointSlices(proxier,
		makeTestEndpointSlice(svcPortName.Namespace, svcPortName.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIpAddressRemote},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     utilpointer.StringPtr(svcPortName.Port),
				Port:     utilpointer.Int32(int32(svcPort)),
				Protocol: &tcpProtocol,
			}}
		}),
	)

	proxier.setInitialized(true)
	proxier.syncProxyRules()

	// Assert
	svc := proxier.serviceMap[svcPortName]
	svcInfo, ok := svc.(*serviceInfo)
	if !ok {
		t.Errorf("Failed to cast serviceInfo %q", svcPortName.String())

	} else {
		if svcInfo.hnsID != guid {
			t.Errorf("%v does not match %v", svcInfo.hnsID, guid)
		}
		if svcInfo.localTrafficDSR != true {
			t.Errorf("Failed to create DSR loadbalancer with local traffic policy")
		}
		if len(svcInfo.loadBalancerIngressIPs) == 0 {
			t.Errorf("svcInfo does not have any loadBalancerIngressIPs, %+v", svcInfo)
		}
	}

	LoadBalancer, err := testHNS.hcninstance.GetLoadBalancerByID(guid)
	if err != nil {
		t.Error(err)
	}

	diff := assertHCNDiff(*LoadBalancer, *expectedLoadBalancer)
	if diff != "" {
		t.Errorf("GetLoadBalancerByID(%s) returned a different LoadBalancer. Diff: %s ", expectedLoadBalancer.Id, diff)
	}
}

// TestEndpointSlice checks if endpoint slices are initialized successfully
func TestEndpointSlice(t *testing.T) {
	// Arrange
	testHNS := NewHCNUtils(&hcntesting.FakeHCN{})
	proxier := NewFakeProxier(testHNS, NETWORK_TYPE_OVERLAY)
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

	expectedEndpoint := &hcn.HostComputeEndpoint{
		Id:                 guid,
		HostComputeNetwork: guid,
		SchemaVersion: hcn.SchemaVersion{
			Major: 2,
			Minor: 0,
		},
		IpConfigurations: []hcn.IpConfig{{IpAddress: epIpAddressRemote, PrefixLength: 24}},
		Flags:            hcn.EndpointFlagsRemoteEndpoint,
	}
	// Act
	proxier.OnServiceAdd(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: svcPortName.Name, Namespace: svcPortName.Namespace},
		Spec: v1.ServiceSpec{
			ClusterIP: svcIP,
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
			Addresses:  []string{epIpAddressRemote},
			Conditions: discovery.EndpointConditions{Ready: utilpointer.BoolPtr(true)},
			NodeName:   utilpointer.StringPtr(testHostName),
		}},
	}

	proxier.OnEndpointSliceAdd(endpointSlice)
	proxier.setInitialized(true)
	proxier.syncProxyRules()

	// Assert
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

	endpoint, err := testHNS.hcninstance.GetEndpointByID(guid)
	if err != nil {
		t.Error(err)
	}

	diff := assertHCNDiff(*expectedEndpoint, *endpoint)
	if diff != "" {
		t.Errorf("GetEndpointById(%s) returned a different LoadBalancer. Diff: %s ", endpoint.Id, diff)
	}
}

// TestLoadBalancer tests that LoadBalancers that are created function as expected
func TestLoadBalancer(t *testing.T) {
	// Arrange
	testHNS := NewHCNUtils(&hcntesting.FakeHCN{})
	proxier := NewFakeProxier(testHNS, NETWORK_TYPE_OVERLAY)
	if proxier == nil {
		t.Error()
	}

	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
		Protocol:       v1.ProtocolTCP,
	}

	expectedPortMapping := &hcn.LoadBalancerPortMapping{
		Protocol:     LbTCPProtocol,
		InternalPort: LbInternalPort,
		ExternalPort: LbExternalPort,
		Flags:        hcn.LoadBalancerPortMappingFlagsNone,
	}

	expectedPortMappings := []hcn.LoadBalancerPortMapping{*expectedPortMapping}

	expectedLoadBalancer := &hcn.HostComputeLoadBalancer{
		Id:                   guid,
		HostComputeEndpoints: []string{guid},
		SourceVIP:            sourceVip,
		SchemaVersion: hcn.SchemaVersion{
			Major: 2,
			Minor: 0,
		},
		FrontendVIPs: []string{svcLBIP},
		Flags:        hcn.LoadBalancerFlagsDSR,
		PortMappings: expectedPortMappings,
	}
	// Act
	makeServiceMap(proxier,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = "LoadBalancer"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     svcPortName.Port,
				Port:     int32(svcPort),
				Protocol: v1.ProtocolTCP,
				NodePort: int32(svcNodePort),
			}}
			svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{
				IP: svcLBIP,
			}}
			svc.Spec.LoadBalancerSourceRanges = []string{" 203.0.113.0/25"}
		}),
	)

	tcpProtocol := v1.ProtocolTCP
	populateEndpointSlices(proxier,
		makeTestEndpointSlice(svcPortName.Namespace, svcPortName.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIpAddressRemote},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     utilpointer.StringPtr(svcPortName.Port),
				Port:     utilpointer.Int32(int32(svcPort)),
				Protocol: &tcpProtocol,
			}}
		}),
	)

	proxier.setInitialized(true)
	proxier.syncProxyRules()

	// Assert
	svc := proxier.serviceMap[svcPortName]
	svcInfo, ok := svc.(*serviceInfo)
	if !ok {
		t.Errorf("Failed to cast serviceInfo %q", svcPortName.String())

	} else {
		if svcInfo.hnsID != guid {
			t.Errorf("%v does not match %v", svcInfo.hnsID, guid)
		}
	}

	LoadBalancer, err := testHNS.hcninstance.GetLoadBalancerByID(guid)
	if err != nil {
		t.Error(err)
	}

	diff := assertHCNDiff(*LoadBalancer, *expectedLoadBalancer)
	if diff != "" {
		t.Errorf("GetLoadBalancerByID(%s) returned a different LoadBalancer. Diff: %s ", expectedLoadBalancer.Id, diff)
	}
}

// TestOnlyLocalLoadBalancing tests if LoadBalancing works as expected on the
// local side
func TestOnlyLocalLoadBalancing(t *testing.T) {
	// Arrange
	testHNS := NewHCNUtils(&hcntesting.FakeHCN{})
	proxier := NewFakeProxier(testHNS, NETWORK_TYPE_OVERLAY)
	if proxier == nil {
		t.Error()
	}

	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
		Protocol:       v1.ProtocolTCP,
	}
	svcSessionAffinityTimeout := int32(10800)

	expectedPortMapping := &hcn.LoadBalancerPortMapping{
		Protocol:     LbTCPProtocol,
		InternalPort: LbInternalPort,
		ExternalPort: LbExternalPort,
		Flags:        hcn.LoadBalancerPortMappingFlagsNone,
	}

	expectedPortMappings := []hcn.LoadBalancerPortMapping{*expectedPortMapping}

	expectedLoadBalancer := &hcn.HostComputeLoadBalancer{
		Id:                   guid,
		HostComputeEndpoints: []string{guid},
		SourceVIP:            sourceVip,
		SchemaVersion: hcn.SchemaVersion{
			Major: 2,
			Minor: 0,
		},
		FrontendVIPs: []string{svcLBIP},
		Flags:        hcn.LoadBalancerFlagsDSR,
		PortMappings: expectedPortMappings,
	}

	// Act
	makeServiceMap(proxier,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = "LoadBalancer"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     svcPortName.Port,
				Port:     int32(svcPort),
				Protocol: v1.ProtocolTCP,
				NodePort: int32(svcNodePort),
			}}
			svc.Spec.HealthCheckNodePort = int32(svcHealthCheckNodePort)
			svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{
				IP: svcLBIP,
			}}
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
			svc.Spec.SessionAffinity = v1.ServiceAffinityClientIP
			svc.Spec.SessionAffinityConfig = &v1.SessionAffinityConfig{
				ClientIP: &v1.ClientIPConfig{TimeoutSeconds: &svcSessionAffinityTimeout},
			}
		}),
	)

	tcpProtocol := v1.ProtocolTCP
	populateEndpointSlices(proxier,
		makeTestEndpointSlice(svcPortName.Namespace, svcPortName.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIpAddressRemote},
			}, {
				Addresses: []string{epIpAddress},
				NodeName:  utilpointer.StringPtr(testHostName),
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     utilpointer.StringPtr(svcPortName.Port),
				Port:     utilpointer.Int32(int32(svcPort)),
				Protocol: &tcpProtocol,
			}}
		}),
	)

	proxier.setInitialized(true)
	proxier.syncProxyRules()

	// Assert
	svc := proxier.serviceMap[svcPortName]
	svcInfo, ok := svc.(*serviceInfo)
	if !ok {
		t.Errorf("Failed to cast serviceInfo %q", svcPortName.String())

	} else {
		if svcInfo.hnsID != guid {
			t.Errorf("%v does not match %v", svcInfo.hnsID, guid)
		}
	}

	LoadBalancer, err := testHNS.hcninstance.GetLoadBalancerByID(guid)
	if err != nil {
		t.Error(err)
	}

	diff := assertHCNDiff(*LoadBalancer, *expectedLoadBalancer)
	if diff != "" {
		t.Errorf("GetLoadBalancerByID(%s) returned a different LoadBalancer. Diff: %s ", expectedLoadBalancer.Id, diff)
	}
}

// TestNodePort creates a NodePort service and checks if it is working the way
// it is expected to
func TestNodePort(t *testing.T) {
	// Arrange
	testHNS := NewHCNUtils(&hcntesting.FakeHCN{})
	proxier := NewFakeProxier(testHNS, NETWORK_TYPE_OVERLAY)
	if proxier == nil {
		t.Error()
	}

	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
		Protocol:       v1.ProtocolTCP,
	}

	expectedPortMapping := &hcn.LoadBalancerPortMapping{
		Protocol:     LbTCPProtocol,
		InternalPort: LbInternalPort,
		ExternalPort: LbExternalPort,
		Flags:        hcn.LoadBalancerPortMappingFlagsNone,
	}

	expectedPortMappings := []hcn.LoadBalancerPortMapping{*expectedPortMapping}

	expectedLoadBalancer := &hcn.HostComputeLoadBalancer{
		Id:                   guid,
		HostComputeEndpoints: []string{guid},
		SourceVIP:            sourceVip,
		SchemaVersion: hcn.SchemaVersion{
			Major: 2,
			Minor: 0,
		},
		FrontendVIPs: []string{svcIP},
		Flags:        hcn.LoadBalancerFlagsDSR,
		PortMappings: expectedPortMappings,
	}

	// Act
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

	tcpProtocol := v1.ProtocolTCP
	populateEndpointSlices(proxier,
		makeTestEndpointSlice(svcPortName.Namespace, svcPortName.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIpAddressRemote},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     utilpointer.StringPtr(svcPortName.Port),
				Port:     utilpointer.Int32(int32(svcPort)),
				Protocol: &tcpProtocol,
			}}
		}),
	)

	proxier.setInitialized(true)
	proxier.syncProxyRules()

	// Assert
	svc := proxier.serviceMap[svcPortName]
	svcInfo, ok := svc.(*serviceInfo)
	if !ok {
		t.Errorf("Failed to cast serviceInfo %q", svcPortName.String())

	} else {
		if svcInfo.hnsID != guid {
			t.Errorf("%v does not match %v", svcInfo.hnsID, guid)
		}

		if svcInfo.nodePorthnsID != guid {
			t.Errorf("%v does not match %v", svcInfo.nodePorthnsID, guid)
		}
	}

	LoadBalancer, err := testHNS.hcninstance.GetLoadBalancerByID(guid)
	if err != nil {
		t.Error(err)
	}

	diff := assertHCNDiff(*LoadBalancer, *expectedLoadBalancer)
	if diff != "" {
		t.Errorf("GetLoadBalancerByID(%s) returned a different LoadBalancer. Diff: %s ", expectedLoadBalancer.Id, diff)
	}
}

// TestNodePortReject creates a non-valid Node Port service and verifies if it is
// successfully rejected
func TestNodePortReject(t *testing.T) {
	// Arrange
	testHNS := NewHCNUtils(&hcntesting.FakeHCN{})
	proxier := NewFakeProxier(testHNS, NETWORK_TYPE_OVERLAY)

	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
	}

	// Act
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

	proxier.setInitialized(true)
	proxier.syncProxyRules()

	// Assert
	svc := proxier.serviceMap[svcPortName]
	_, ok := svc.(*serviceInfo)
	//The Node Port should be rejected so the casting to serviceInfo should not be possible
	if ok {
		t.Errorf("Service ns1/svc1:p80 shoud be REJECTED. Unexpected behaviour!")
	}

	//there should not be created any hcn instances because the service was rejected
	LoadBalancers, err := testHNS.hcninstance.ListLoadBalancers()
	if err != nil {
		t.Error(err)
	}

	if len(LoadBalancers) != 0 {
		t.Errorf("Service ns1/svc1:p80 shoud be REJECTED. Unexpected behaviour!")
	}
}

// TestClusterIpReject creates a non-valid ClusterIP service and verifies if it is
// successfully rejected
func TestClusterIpReject(t *testing.T) {
	// Arrange
	testHNS := NewHCNUtils(&hcntesting.FakeHCN{})
	proxier := NewFakeProxier(testHNS, NETWORK_TYPE_OVERLAY)

	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
	}

	// Act
	makeServiceMap(proxier,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.ClusterIP = svcIP
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     svcPortName.Port,
				Port:     int32(svcPort),
				Protocol: v1.ProtocolTCP,
			}}
		}),
	)

	proxier.setInitialized(true)
	proxier.syncProxyRules()

	// Assert
	svc := proxier.serviceMap[svcPortName]
	_, ok := svc.(*serviceInfo)
	//The ClusterIp should be rejected so the casting to serviceInfo should not be possible
	if ok {
		t.Errorf("Service ns1/svc1:p80 shoud be REJECTED. Unexpected behaviour!")
	}

	//there should not be created any hcn instances because the service was rejected
	LoadBalancers, err := testHNS.hcninstance.ListLoadBalancers()
	if err != nil {
		t.Error(err)
	}

	if len(LoadBalancers) != 0 {
		t.Errorf("Service ns1/svc1:p80 shoud be REJECTED. Unexpected behaviour!")
	}
}

// TestLoadBalancerReject creates a non-valid LoadBalancer service and verifies if it is
// successfully rejected
func TestLoadBalancerReject(t *testing.T) {
	// Arrange
	testHNS := NewHCNUtils(&hcntesting.FakeHCN{})
	proxier := NewFakeProxier(testHNS, NETWORK_TYPE_OVERLAY)

	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
		Protocol:       v1.ProtocolTCP,
	}
	svcSessionAffinityTimeout := int32(10800)

	// Act
	makeServiceMap(proxier,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = "LoadBalancer"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     svcPortName.Port,
				Port:     int32(svcPort),
				Protocol: v1.ProtocolTCP,
				NodePort: int32(svcNodePort),
			}}
			svc.Spec.HealthCheckNodePort = int32(svcHealthCheckNodePort)
			svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{
				IP: svcLBIP,
			}}
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
			svc.Spec.SessionAffinity = v1.ServiceAffinityClientIP
			svc.Spec.SessionAffinityConfig = &v1.SessionAffinityConfig{
				ClientIP: &v1.ClientIPConfig{TimeoutSeconds: &svcSessionAffinityTimeout},
			}
		}),
	)

	proxier.setInitialized(true)
	proxier.syncProxyRules()

	// Assert
	svc := proxier.serviceMap[svcPortName]
	svcInfo, ok := svc.(*serviceInfo)
	if !ok {
		t.Errorf("Failed to cast serviceInfo %q", svcPortName.String())

	} else { //service has no endpoints so the lb policy should not have been applied
		if svcInfo.policyApplied {
			t.Errorf("Service ns1/svc1:p80 has no endpoint information available, but policies are applied. Unexpected behaviour!")
		}
	}

	//no LoadBalancers should have been created so this should fail
	_, err := testHNS.hcninstance.GetLoadBalancerByID(guid)

	if err == nil {
		t.Error(err)
	}
}

// TestExternalIPsReject creates a non-valid ExternalIP service and verifies if it is
// successfully rejected
func TestExternalIPsReject(t *testing.T) {
	// Arrange
	testHNS := NewHCNUtils(&hcntesting.FakeHCN{})
	proxier := NewFakeProxier(testHNS, NETWORK_TYPE_OVERLAY)

	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
	}

	// Act
	makeServiceMap(proxier,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = "ClusterIP"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.ExternalIPs = []string{svcExternalIPs}
			svc.Spec.Ports = []v1.ServicePort{{
				Name:       svcPortName.Port,
				Port:       int32(svcPort),
				Protocol:   v1.ProtocolTCP,
				TargetPort: intstr.FromInt(svcPort),
			}}
		}),
	)

	proxier.setInitialized(true)
	proxier.syncProxyRules()

	// Assert
	svc := proxier.serviceMap[svcPortName]
	_, ok := svc.(*serviceInfo) //The ExternalIP should be rejected so the casting to serviceInfo should not be possible
	if ok {
		t.Errorf("Service ns1/svc1:p80 shoud be REJECTED. Unexpected behaviour!")
	}

	//there should not be created any hcn instances because the service was rejected
	LoadBalancers, err := testHNS.hcninstance.ListLoadBalancers()
	if err != nil {
		t.Error(err)
	}

	if len(LoadBalancers) != 0 {
		t.Errorf("Service ns1/svc1:p80 shoud be REJECTED. Unexpected behaviour!")
	}
}

func TestClusterIPEndpointsJump(t *testing.T) {
	// Arrange
	testHNS := NewHCNUtils(&hcntesting.FakeHCN{})
	proxier := NewFakeProxier(testHNS, NETWORK_TYPE_OVERLAY)

	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
		Protocol:       v1.ProtocolTCP,
	}

	// Act
	makeServiceMap(proxier,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.ClusterIP = svcIP
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     svcPortName.Port,
				Port:     int32(svcPort),
				Protocol: v1.ProtocolTCP,
			}}
		}),
	)

	tcpProtocol := v1.ProtocolTCP
	populateEndpointSlices(proxier,
		makeTestEndpointSlice(svcPortName.Namespace, svcPortName.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIpAddressRemote},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     utilpointer.StringPtr(svcPortName.Port),
				Port:     utilpointer.Int32(int32(svcPort)),
				Protocol: &tcpProtocol,
			}}
		}),
	)

	proxier.setInitialized(true)
	proxier.syncProxyRules()

	// Assert
	svc := proxier.serviceMap[svcPortName]
	svcInfo, ok := svc.(*serviceInfo)
	if !ok {
		t.Errorf("Failed to cast serviceInfo %q", svcPortName.String())

	} else {
		if svcInfo.hnsID != guid {
			t.Errorf("%v does not match %v", svcInfo.hnsID, guid)
		}
		if !svcInfo.BaseServiceInfo.UsesClusterEndpoints() {
			t.Error()
		}
	}
}

// TestOnlyLocalExternalIPs creates a service with local ExternalIPs and checks
// if it is created the way it's expected to. Also it makes additional checks on
// policies to ensure they're local
func TestOnlyLocalExternalIPs(t *testing.T) {
	// Arrange
	testHNS := NewHCNUtils(&hcntesting.FakeHCN{})
	proxier := NewFakeProxier(testHNS, NETWORK_TYPE_OVERLAY)

	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
		Protocol:       v1.ProtocolTCP,
	}

	// Act
	makeServiceMap(proxier,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = "NodePort"
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
			svc.Spec.ClusterIP = svcIP
			svc.Spec.ExternalIPs = []string{svcExternalIPs}
			svc.Spec.Ports = []v1.ServicePort{{
				Name:       svcPortName.Port,
				Port:       int32(svcPort),
				Protocol:   v1.ProtocolTCP,
				TargetPort: intstr.FromInt(svcPort),
			}}
		}),
	)
	tcpProtocol := v1.ProtocolTCP
	populateEndpointSlices(proxier,
		makeTestEndpointSlice(svcPortName.Namespace, svcPortName.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIpAddressRemote},
			}, {
				Addresses: []string{epIpAddressRemote},
				NodeName:  utilpointer.StringPtr(testHostName),
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     utilpointer.StringPtr(svcPortName.Port),
				Port:     utilpointer.Int32(int32(svcPort)),
				Protocol: &tcpProtocol,
			}}
		}),
	)

	proxier.setInitialized(true)
	proxier.syncProxyRules()

	// Assert
	svc := proxier.serviceMap[svcPortName]
	svcInfo, ok := svc.(*serviceInfo)
	if !ok {
		t.Errorf("Failed to cast serviceInfo %q", svcPortName.String())

	} else {
		if svcInfo.hnsID != guid {
			t.Errorf("%v does not match %v", svcInfo.hnsID, guid)
		}
		if svcInfo.remoteEndpoint.ip != svcIP {
			t.Errorf("Ip %v does not match %v", svcInfo.remoteEndpoint.ip, svcIP)
		}
		if !svcInfo.BaseServiceInfo.ExternalPolicyLocal() {
			t.Error("Traffic policy not local")
		}
	}
}

// TestNonLocalExternalIPs creates a service with non-local ExternalIPs and checks
// if it is created the way it's expected to. Also it makes additional checks on
// policies to ensure they're not local
func TestNonLocalExternalIPs(t *testing.T) {
	// Arrange
	testHNS := NewHCNUtils(&hcntesting.FakeHCN{})
	proxier := NewFakeProxier(testHNS, NETWORK_TYPE_OVERLAY)

	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
		Protocol:       v1.ProtocolTCP,
	}

	// Act
	makeServiceMap(proxier,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.ClusterIP = svcIP
			svc.Spec.ExternalIPs = []string{svcExternalIPs}
			svc.Spec.Ports = []v1.ServicePort{{
				Name:       svcPortName.Port,
				Port:       int32(svcPort),
				Protocol:   v1.ProtocolTCP,
				TargetPort: intstr.FromInt(svcPort),
			}}
		}),
	)
	tcpProtocol := v1.ProtocolTCP
	populateEndpointSlices(proxier,
		makeTestEndpointSlice(svcPortName.Namespace, svcPortName.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIpAddressRemote},
				NodeName:  nil,
			}, {
				Addresses: []string{epIpAddressRemote},
				NodeName:  utilpointer.StringPtr(testHostName),
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     utilpointer.StringPtr(svcPortName.Port),
				Port:     utilpointer.Int32(int32(svcPort)),
				Protocol: &tcpProtocol,
			}}
		}),
	)

	proxier.setInitialized(true)
	proxier.syncProxyRules()

	// Assert
	svc := proxier.serviceMap[svcPortName]
	svcInfo, ok := svc.(*serviceInfo)
	if !ok {
		t.Errorf("Failed to cast serviceInfo %q", svcPortName.String())

	} else {
		if svcInfo.hnsID != guid {
			t.Errorf("%v does not match %v", svcInfo.hnsID, guid)
		}
		if svcInfo.remoteEndpoint.ip != svcIP {
			t.Errorf("Ip %v does not match %v", svcInfo.remoteEndpoint.ip, svcIP)
		}
		if svcInfo.BaseServiceInfo.ExternalPolicyLocal() {
			t.Error("Traffic policy is local when it should be non-local")
		}
	}
}

// TestHealthCheckNodePort tests that health check node ports are enabled when expected
func TestHealthCheckNodePort(t *testing.T) {
	// Arrange
	testHNS := NewHCNUtils(&hcntesting.FakeHCN{})
	proxier := NewFakeProxier(testHNS, NETWORK_TYPE_OVERLAY)

	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
		Protocol:       v1.ProtocolTCP,
	}

	// Act
	makeServiceMap(proxier,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = "LoadBalancer"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     svcPortName.Port,
				Port:     int32(svcPort),
				Protocol: v1.ProtocolTCP,
				NodePort: int32(svcNodePort),
			}}
			svc.Spec.HealthCheckNodePort = int32(svcHealthCheckNodePort)
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
		}),
	)

	proxier.setInitialized(true)
	proxier.syncProxyRules()

	// Assert
	svc := proxier.serviceMap[svcPortName]
	svcInfo, ok := svc.(*serviceInfo)
	if !ok {
		t.Errorf("Failed to cast serviceInfo %q", svcPortName.String())

	} else {
		if svcInfo.BaseServiceInfo.HealthCheckNodePort() != svcHealthCheckNodePort {
			t.Errorf("Node port health check is null")
		}
	}
}

// TestBuildServiceMapAddRemove makes tests on simple operations executed on ServiceMap
// (Add, Delete) with different test cases.
func TestBuildServiceMapAddRemove(t *testing.T) {
	// Arrange
	testHNS := NewHCNUtils(&hcntesting.FakeHCN{})
	proxier := NewFakeProxier(testHNS, NETWORK_TYPE_OVERLAY)

	services := []*v1.Service{
		makeTestService("somewhere-else", "cluster-ip", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeClusterIP
			svc.Spec.ClusterIP = "10.20.30.42"
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "something", "UDP", 1234, 4321, 0)
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "somethingelse", "UDP", 1235, 5321, 0)
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "sctpport", "SCTP", 1236, 6321, 0)
		}),
		makeTestService("somewhere-else", "node-port", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeNodePort
			svc.Spec.ClusterIP = "10.20.30.44"
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "blahblah", "UDP", 345, 678, 0)
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "moreblahblah", "TCP", 344, 677, 0)
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "muchmoreblah", "SCTP", 343, 676, 0)
		}),
		makeTestService("somewhere", "load-balancer", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeLoadBalancer
			svc.Spec.ClusterIP = "10.20.30.47"
			svc.Spec.LoadBalancerIP = "11.21.31.41"
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "foobar", "UDP", 8675, 30061, 7000)
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "baz", "UDP", 8676, 30062, 7001)
			svc.Status.LoadBalancer = v1.LoadBalancerStatus{
				Ingress: []v1.LoadBalancerIngress{
					{IP: "11.21.31.41"},
				},
			}
		}),
		makeTestService("somewhere", "only-local-load-balancer", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeLoadBalancer
			svc.Spec.ClusterIP = "10.20.30.49"
			svc.Spec.LoadBalancerIP = "11.21.31.42"
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "foobar2", "UDP", 8677, 30063, 7002)
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "baz", "UDP", 8678, 30064, 7003)
			svc.Status.LoadBalancer = v1.LoadBalancerStatus{
				Ingress: []v1.LoadBalancerIngress{
					{IP: "11.21.31.42"},
				},
			}
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
			svc.Spec.HealthCheckNodePort = 345
		}),
	}

	// Act

	for i := range services {
		proxier.OnServiceAdd(services[i])
	}
	result := proxier.serviceMap.Update(proxier.serviceChanges)
	if len(proxier.serviceMap) != 10 {
		t.Errorf("expected service map length 10, got %v", proxier.serviceMap)
	}

	// The only-local-loadbalancer ones get added
	if len(result.HCServiceNodePorts) != 1 {
		t.Errorf("expected 1 healthcheck port, got %v", result.HCServiceNodePorts)
	} else {
		nsn := makeNSN("somewhere", "only-local-load-balancer")
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
	oneService := makeTestService("somewhere-else", "cluster-ip", func(svc *v1.Service) {
		svc.Spec.Type = v1.ServiceTypeClusterIP
		svc.Spec.ClusterIP = "10.20.30.45"
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "somethingelse", "UDP", 1235, 5321, 0)
	})

	proxier.OnServiceUpdate(services[0], oneService)
	proxier.OnServiceDelete(services[1])
	proxier.OnServiceDelete(services[2])
	proxier.OnServiceDelete(services[3])

	result = proxier.serviceMap.Update(proxier.serviceChanges)

	// Assert

	if len(proxier.serviceMap) != 1 {
		t.Errorf("expected service map length 1, got %v", proxier.serviceMap)
	}

	if len(result.HCServiceNodePorts) != 0 {
		t.Errorf("expected 0 healthcheck ports, got %v", result.HCServiceNodePorts)
	}

	// All services but one were deleted. While you'd expect only the ClusterIPs
	// from the three deleted services here, we still have the ClusterIP for
	// the not-deleted service, because one of it's ServicePorts was deleted.
	expectedStaleUDPServices := []string{"10.20.30.42", "10.20.30.44", "10.20.30.47", "10.20.30.49"}
	if len(result.UDPStaleClusterIP) != len(expectedStaleUDPServices) {
		t.Errorf("expected stale UDP services length %d, got %v", len(expectedStaleUDPServices), result.UDPStaleClusterIP.UnsortedList())
	}
	for _, ip := range expectedStaleUDPServices {
		if !result.UDPStaleClusterIP.Has(ip) {
			t.Errorf("expected stale UDP service service %s", ip)
		}
	}
}

// TestBuildServiceMapServiceHeadless tests if headless services are ignored as expected and
// checks that healthchecks are not enabled when there are no proxied services
func TestBuildServiceMapServiceHeadless(t *testing.T) {
	// Arrange
	testHNS := NewHCNUtils(&hcntesting.FakeHCN{})
	proxier := NewFakeProxier(testHNS, NETWORK_TYPE_OVERLAY)

	// Act
	makeServiceMap(proxier,
		makeTestService("somewhere-else", "headless", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeClusterIP
			svc.Spec.ClusterIP = v1.ClusterIPNone
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "rpc", "UDP", 1234, 0, 0)
		}),
		makeTestService("somewhere-else", "headless-without-port", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeClusterIP
			svc.Spec.ClusterIP = v1.ClusterIPNone
		}),
	)

	// Assert

	// Headless service should be ignored
	result := proxier.serviceMap.Update(proxier.serviceChanges)
	if len(proxier.serviceMap) != 0 {
		t.Errorf("expected service map length 0, got %d", len(proxier.serviceMap))
	}

	// No proxied services, so no healthchecks
	if len(result.HCServiceNodePorts) != 0 {
		t.Errorf("expected healthcheck ports length 0, got %d", len(result.HCServiceNodePorts))
	}

	if len(result.UDPStaleClusterIP) != 0 {
		t.Errorf("expected stale UDP services length 0, got %d", len(result.UDPStaleClusterIP))
	}
}

// TestBuildServiceMapServiceHeadless tests if healthchecks are enabled when
// there are no proxied services (they are expected not to be enabled)
func TestBuildServiceMapServiceTypeExternalName(t *testing.T) {
	// Arrange
	testHNS := NewHCNUtils(&hcntesting.FakeHCN{})
	proxier := NewFakeProxier(testHNS, NETWORK_TYPE_OVERLAY)

	// Act
	makeServiceMap(proxier,
		makeTestService("somewhere-else", "external-name", func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeExternalName
			svc.Spec.ClusterIP = "10.20.30.42" // Should be ignored
			svc.Spec.ExternalName = "foo2.bar.com"
			svc.Spec.Ports = addTestPort(svc.Spec.Ports, "blah", "UDP", 1235, 5321, 0)
		}),
	)

	// Assert
	result := proxier.serviceMap.Update(proxier.serviceChanges)
	if len(proxier.serviceMap) != 0 {
		t.Errorf("expected service map length 0, got %v", proxier.serviceMap)
	}
	// No proxied services, so no healthchecks
	if len(result.HCServiceNodePorts) != 0 {
		t.Errorf("expected healthcheck ports length 0, got %v", result.HCServiceNodePorts)
	}
	if len(result.UDPStaleClusterIP) != 0 {
		t.Errorf("expected stale UDP services length 0, got %v", result.UDPStaleClusterIP)
	}
}

// TestBuildServiceMapServiceUpdate creates a service and updates it to LoadBalancer and ClusterIP
// and checks if it is successfully updated. Also the test checks that the service map
// does not change when is not expected to (when the service is not updated).
func TestBuildServiceMapServiceUpdate(t *testing.T) {
	// Arrange
	testHNS := NewHCNUtils(&hcntesting.FakeHCN{})
	proxier := NewFakeProxier(testHNS, NETWORK_TYPE_OVERLAY)

	servicev1 := makeTestService("somewhere", "some-service", func(svc *v1.Service) {
		svc.Spec.Type = v1.ServiceTypeClusterIP
		svc.Spec.ClusterIP = "10.20.30.42"
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "something", "UDP", 1234, 4321, 0)
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "somethingelse", "TCP", 1235, 5321, 0)
	})
	servicev2 := makeTestService("somewhere", "some-service", func(svc *v1.Service) {
		svc.Spec.Type = v1.ServiceTypeLoadBalancer
		svc.Spec.ClusterIP = "10.20.30.42"
		svc.Spec.LoadBalancerIP = "11.21.31.41"
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "something", "UDP", 1234, 4321, 7002)
		svc.Spec.Ports = addTestPort(svc.Spec.Ports, "somethingelse", "TCP", 1235, 5321, 7003)
		svc.Status.LoadBalancer = v1.LoadBalancerStatus{
			Ingress: []v1.LoadBalancerIngress{
				{IP: "11.21.31.41"},
			},
		}
		svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
		svc.Spec.HealthCheckNodePort = 345
	})

	// Act --- case 1
	proxier.OnServiceAdd(servicev1)

	// Assert --- case 1
	result := proxier.serviceMap.Update(proxier.serviceChanges)
	if len(proxier.serviceMap) != 2 {
		t.Errorf("expected service map length 2, got %v", proxier.serviceMap)
	}
	if len(result.HCServiceNodePorts) != 0 {
		t.Errorf("expected healthcheck ports length 0, got %v", result.HCServiceNodePorts)
	}
	if len(result.UDPStaleClusterIP) != 0 {
		// Services only added, so nothing stale yet
		t.Errorf("expected stale UDP services length 0, got %d", len(result.UDPStaleClusterIP))
	}

	// Act --- case 2
	// Change service to load-balancer
	proxier.OnServiceUpdate(servicev1, servicev2)

	// Assert --- case 2
	result = proxier.serviceMap.Update(proxier.serviceChanges)
	if len(proxier.serviceMap) != 2 {
		t.Errorf("expected service map length 2, got %v", proxier.serviceMap)
	}
	if len(result.HCServiceNodePorts) != 1 {
		t.Errorf("expected healthcheck ports length 1, got %v", result.HCServiceNodePorts)
	}
	if len(result.UDPStaleClusterIP) != 0 {
		t.Errorf("expected stale UDP services length 0, got %v", result.UDPStaleClusterIP.UnsortedList())
	}

	// Act --- case 3
	// No change; make sure the service map stays the same and there are
	// no health-check changes
	proxier.OnServiceUpdate(servicev2, servicev2)

	// Assert --- case 3
	result = proxier.serviceMap.Update(proxier.serviceChanges)
	if len(proxier.serviceMap) != 2 {
		t.Errorf("expected service map length 2, got %v", proxier.serviceMap)
	}
	if len(result.HCServiceNodePorts) != 1 {
		t.Errorf("expected healthcheck ports length 1, got %v", result.HCServiceNodePorts)
	}
	if len(result.UDPStaleClusterIP) != 0 {
		t.Errorf("expected stale UDP services length 0, got %v", result.UDPStaleClusterIP.UnsortedList())
	}

	// Act --- case 4
	// And back to ClusterIP
	proxier.OnServiceUpdate(servicev2, servicev1)

	// Assert --- case 4
	result = proxier.serviceMap.Update(proxier.serviceChanges)
	if len(proxier.serviceMap) != 2 {
		t.Errorf("expected service map length 2, got %v", proxier.serviceMap)
	}
	if len(result.HCServiceNodePorts) != 0 {
		t.Errorf("expected healthcheck ports length 0, got %v", result.HCServiceNodePorts)
	}
	if len(result.UDPStaleClusterIP) != 0 {
		// Services only added, so nothing stale yet
		t.Errorf("expected stale UDP services length 0, got %d", len(result.UDPStaleClusterIP))
	}
}

// TestEndpointSliceE2E ensures that the winkernel proxier supports working with
// EndpointSlices
func TestEndpointSliceE2E(t *testing.T) {
	// Arrange
	testHNS := NewHCNUtils(&hcntesting.FakeHCN{})
	proxier := NewFakeProxier(testHNS, NETWORK_TYPE_OVERLAY)
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
			ClusterIP: svcIP,
			Selector:  map[string]string{"foo": "bar"},
			Ports:     []v1.ServicePort{{Name: svcPortName.Port, TargetPort: intstr.FromInt(80), Protocol: v1.ProtocolTCP}},
		},
	})

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
			Addresses:  []string{epIpAddressRemote},
			Conditions: discovery.EndpointConditions{Ready: utilpointer.BoolPtr(true)},
			NodeName:   utilpointer.StringPtr(testHostName),
		}, {
			Addresses:  []string{"192.168.2.4"},
			Conditions: discovery.EndpointConditions{Ready: utilpointer.BoolPtr(true)},
			NodeName:   utilpointer.StringPtr("node2"),
		}, {
			Addresses:  []string{"192.168.2.5"},
			Conditions: discovery.EndpointConditions{Ready: utilpointer.BoolPtr(true)},
			NodeName:   utilpointer.StringPtr("node3"),
		}, {
			Addresses:  []string{"192.168.2.6"},
			Conditions: discovery.EndpointConditions{Ready: utilpointer.BoolPtr(false)},
			NodeName:   utilpointer.StringPtr("node4"),
		}},
	}

	// Act
	proxier.OnEndpointSliceAdd(endpointSlice)
	proxier.setInitialized(true)
	proxier.syncProxyRules()

	// Assert
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

	proxier.setInitialized(false)
	proxier.OnEndpointSliceDelete(endpointSlice)
	proxier.setInitialized(true)
	proxier.syncProxyRules()

	svc = proxier.serviceMap[svcPortName]
	svcInfo, ok = svc.(*serviceInfo)
	if !ok {
		t.Errorf("Failed to cast serviceInfo %q", svcPortName.String())

	} else { //policy should not be applied here
		if svcInfo.policyApplied {
			t.Error("Service ns1/svc1:p80 has no endpoint information available, but policies are applied. Unexpected behaviour!")
		}
	}
}

// TestEndpointSliceE2E ensures that the winkernel proxier supports working with
// EndpointSlices when internalTrafficPolicy is specified
func TestInternalTrafficPolicyE2E(t *testing.T) {
	// Arrange (test cases)
	type endpoint struct {
		ip       string
		hostname string
	}

	cluster := v1.ServiceInternalTrafficPolicyCluster
	local := v1.ServiceInternalTrafficPolicyLocal

	testCases := []struct {
		name                  string
		internalTrafficPolicy *v1.ServiceInternalTrafficPolicyType
		featureGateOn         bool
		endpoints             []endpoint
		expectEndpointRule    bool
	}{
		{
			name:                  "internalTrafficPolicy is cluster",
			internalTrafficPolicy: &cluster,
			featureGateOn:         true,
			endpoints: []endpoint{
				{epPaAddress, testHostName},
				{"10.0.0.4", "host1"},
				{"10.0.0.5", "host2"},
			},
			expectEndpointRule: true,
		},
		{
			name:                  "internalTrafficPolicy is local and there is non-zero local endpoints",
			internalTrafficPolicy: &local,
			featureGateOn:         true,
			endpoints: []endpoint{
				{epPaAddress, testHostName},
				{"10.0.0.4", "host1"},
				{"10.0.0.5", "host2"},
			},
			expectEndpointRule: true,
		},
		{
			name:                  "internalTrafficPolicy is local and there is zero local endpoint",
			internalTrafficPolicy: &local,
			featureGateOn:         true,
			endpoints: []endpoint{
				{epPaAddress, testHostName},
				{"10.0.0.4", "host1"},
				{"10.0.0.5", "host2"},
			},
			expectEndpointRule: false,
		},
		{
			name:                  "internalTrafficPolicy is local and there is non-zero local endpoint with feature gate off",
			internalTrafficPolicy: &local,
			featureGateOn:         false,
			endpoints: []endpoint{
				{epPaAddress, testHostName},
				{"10.0.0.4", "host1"},
				{"10.0.0.5", "host2"},
			},
			expectEndpointRule: false,
		},
	}

	//Iterate through test cases
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ServiceInternalTrafficPolicy, tc.featureGateOn)()

			// Arrange (individually)
			testHNS := NewHCNUtils(&hcntesting.FakeHCN{})
			proxier := NewFakeProxier(testHNS, NETWORK_TYPE_OVERLAY)
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

			svc := &v1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: svcPortName.Name, Namespace: svcPortName.Namespace},
				Spec: v1.ServiceSpec{
					ClusterIP: svcIP,
					Selector:  map[string]string{"foo": "bar"},
					Ports:     []v1.ServicePort{{Name: svcPortName.Port, Port: 80, Protocol: v1.ProtocolTCP}},
				},
			}
			if tc.internalTrafficPolicy != nil {
				svc.Spec.InternalTrafficPolicy = tc.internalTrafficPolicy
			}

			proxier.OnServiceAdd(svc)

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
			}

			// Act
			for _, ep := range tc.endpoints {
				endpointSlice.Endpoints = append(endpointSlice.Endpoints, discovery.Endpoint{
					Addresses:  []string{ep.ip},
					Conditions: discovery.EndpointConditions{Ready: utilpointer.BoolPtr(true)},
					NodeName:   utilpointer.StringPtr(ep.hostname),
				})
			}

			proxier.OnEndpointSliceAdd(endpointSlice)
			proxier.setInitialized(true)
			proxier.syncProxyRules()

			// Assert
			sVc := proxier.serviceMap[svcPortName]
			svcInfo, ok := sVc.(*serviceInfo)
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

			if tc.expectEndpointRule {
				proxier.setInitialized(false)
				proxier.OnEndpointSliceDelete(endpointSlice)
				proxier.setInitialized(true)
				proxier.syncProxyRules()

				sVc = proxier.serviceMap[svcPortName]
				svcInfo, ok = sVc.(*serviceInfo)
				if !ok {
					t.Errorf("Failed to cast serviceInfo %q", svcPortName.String())

				} else { //policy should not be applied here
					if svcInfo.policyApplied {
						t.Error("Service ns1/svc1:p80 has no endpoint information available, but policies are applied. Unexpected behaviour!")
					}
				}
			}
		})
	}
}

func TestHealthCheckNodePortE2E(t *testing.T) {
	// Arrange
	testHNS := NewHCNUtils(&hcntesting.FakeHCN{})
	proxier := NewFakeProxier(testHNS, NETWORK_TYPE_OVERLAY)
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

	svc := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: svcPortName.Name, Namespace: svcPortName.Namespace},
		Spec: v1.ServiceSpec{
			ClusterIP:             svcIP,
			Selector:              map[string]string{"foo": "bar"},
			Ports:                 []v1.ServicePort{{Name: svcPortName.Port, TargetPort: intstr.FromInt(80), NodePort: 30010, Protocol: v1.ProtocolTCP}},
			Type:                  "LoadBalancer",
			HealthCheckNodePort:   svcHealthCheckNodePort,
			ExternalTrafficPolicy: v1.ServiceExternalTrafficPolicyTypeLocal,
		},
	}

	proxier.OnServiceAdd(svc)

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
			Addresses:  []string{epPaAddress},
			Conditions: discovery.EndpointConditions{Ready: utilpointer.BoolPtr(true)},
			NodeName:   utilpointer.StringPtr(testHostName),
		}, {
			Addresses:  []string{"10.0.0.4"},
			Conditions: discovery.EndpointConditions{Ready: utilpointer.BoolPtr(true)},
			NodeName:   utilpointer.StringPtr("node2"),
		}, {
			Addresses:  []string{"10.0.0.5"},
			Conditions: discovery.EndpointConditions{Ready: utilpointer.BoolPtr(true)},
			NodeName:   utilpointer.StringPtr("node3"),
		}, {
			Addresses:  []string{"10.0.0.6"},
			Conditions: discovery.EndpointConditions{Ready: utilpointer.BoolPtr(false)},
			NodeName:   utilpointer.StringPtr("node4"),
		}},
	}

	// Act
	proxier.OnEndpointSliceAdd(endpointSlice)
	proxier.setInitialized(true)
	proxier.syncProxyRules()

	// Assert
	sVc := proxier.serviceMap[svcPortName]
	svcInfo, ok := sVc.(*serviceInfo)
	if !ok {
		t.Errorf("Failed to cast serviceInfo %q", svcPortName.String())

	} else {
		if svcInfo.hnsID != guid {
			t.Errorf("The Hns Loadbalancer Id %v does not match %v. ServicePortName %q", svcInfo.hnsID, guid, svcPortName.String())
		}
		if svcInfo.BaseServiceInfo.HealthCheckNodePort() != svcHealthCheckNodePort {
			t.Errorf("Node port health check is null")
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

	// Act -- delete service
	proxier.setInitialized(false)
	proxier.OnServiceDelete(svc)
	proxier.setInitialized(true)
	proxier.syncProxyRules()

	// Assert -- delete service
	sVc = proxier.serviceMap[svcPortName]
	svcInfo, ok = sVc.(*serviceInfo)
	if ok {
		t.Errorf("Service ns1/svc1:p80 shoud be REJECTED. Unexpected behaviour!")
	}
}

// TestHealthCheckNodePortWhenTerminating tests that health check node ports are not enabled when all local endpoints are terminating
func TestHealthCheckNodePortWhenTerminating(t *testing.T) {
	// Arrange
	testHNS := NewHCNUtils(&hcntesting.FakeHCN{})
	proxier := NewFakeProxier(testHNS, NETWORK_TYPE_OVERLAY)
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
			ClusterIP:             svcIP,
			Selector:              map[string]string{"foo": "bar"},
			Ports:                 []v1.ServicePort{{Name: svcPortName.Port, TargetPort: intstr.FromInt(80), NodePort: 30010, Protocol: v1.ProtocolTCP}},
			Type:                  "LoadBalancer",
			HealthCheckNodePort:   svcHealthCheckNodePort,
			ExternalTrafficPolicy: v1.ServiceExternalTrafficPolicyTypeLocal,
		},
	})

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
			Addresses:  []string{epPaAddress},
			Conditions: discovery.EndpointConditions{Ready: utilpointer.BoolPtr(true)},
			NodeName:   utilpointer.StringPtr(testHostName),
		}, {
			Addresses:  []string{"10.0.0.4"},
			Conditions: discovery.EndpointConditions{Ready: utilpointer.BoolPtr(true)},
			NodeName:   utilpointer.StringPtr(testHostName),
		}, {
			Addresses:  []string{"10.0.0.5"},
			Conditions: discovery.EndpointConditions{Ready: utilpointer.BoolPtr(true)},
			NodeName:   utilpointer.StringPtr(testHostName),
		}, { // not ready endpoints should be ignored
			Addresses:  []string{"10.0.0.6"},
			Conditions: discovery.EndpointConditions{Ready: utilpointer.BoolPtr(false)},
			NodeName:   utilpointer.StringPtr(testHostName),
		}},
	}

	// Act
	proxier.OnEndpointSliceAdd(endpointSlice)
	proxier.setInitialized(true)
	proxier.syncProxyRules()

	// Assert
	svc := proxier.serviceMap[svcPortName]
	svcInfo, ok := svc.(*serviceInfo)
	if !ok {
		t.Errorf("Failed to cast serviceInfo %q", svcPortName.String())

	} else {
		if svcInfo.hnsID != guid {
			t.Errorf("The Hns Loadbalancer Id %v does not match %v. ServicePortName %q", svcInfo.hnsID, guid, svcPortName.String())
		}

		if svcInfo.BaseServiceInfo.HealthCheckNodePort() != svcHealthCheckNodePort {
			t.Errorf("Node port health check is null")
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

	// Arrange -- case 2
	// set all endpoints to terminating
	endpointSliceTerminating := &discovery.EndpointSlice{
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
			Addresses: []string{epPaAddress},
			Conditions: discovery.EndpointConditions{
				Ready:       utilpointer.BoolPtr(false),
				Serving:     utilpointer.BoolPtr(true),
				Terminating: utilpointer.BoolPtr(false),
			},
			NodeName: utilpointer.StringPtr(testHostName),
		}, {
			Addresses: []string{"10.0.0.4"},
			Conditions: discovery.EndpointConditions{
				Ready:       utilpointer.BoolPtr(false),
				Serving:     utilpointer.BoolPtr(true),
				Terminating: utilpointer.BoolPtr(true),
			},
			NodeName: utilpointer.StringPtr(testHostName),
		}, {
			Addresses: []string{"10.0.0.5"},
			Conditions: discovery.EndpointConditions{
				Ready:       utilpointer.BoolPtr(false),
				Serving:     utilpointer.BoolPtr(true),
				Terminating: utilpointer.BoolPtr(true),
			},
			NodeName: utilpointer.StringPtr(testHostName),
		}, { // not ready endpoints should be ignored
			Addresses: []string{"10.0.0.6"},
			Conditions: discovery.EndpointConditions{
				Ready:       utilpointer.BoolPtr(false),
				Serving:     utilpointer.BoolPtr(false),
				Terminating: utilpointer.BoolPtr(true),
			},
			NodeName: utilpointer.StringPtr(testHostName),
		}},
	}

	// Act -- case 2
	proxier.setInitialized(false)
	proxier.OnEndpointSliceUpdate(endpointSlice, endpointSliceTerminating)
	proxier.setInitialized(true)
	proxier.syncProxyRules()

	// Assert
	svc = proxier.serviceMap[svcPortName]
	svcInfo, ok = svc.(*serviceInfo)
	if !ok {
		t.Errorf("Failed to cast serviceInfo %q", svcPortName.String())

	} else { //service has all endpoints on terminating so the lb policy should not have been applied
		if svcInfo.policyApplied {
			t.Errorf("Service ns1/svc1:p80 has no endpoint information available, but policies are applied. Unexpected behaviour!")
		}
	}
}

// TestUpdateEndpointsMap makes different changes on the endpointsMap and verifies they are
// working like they're expected to
func TestUpdateEndpointsMap(t *testing.T) {
	// Arrange
	var nodeName = testHostName
	udpProtocol := v1.ProtocolUDP

	emptyEndpointSlices := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1, func(*discovery.EndpointSlice) {}),
	}
	subset1 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"10.1.1.1"},
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     utilpointer.String("p11"),
			Port:     utilpointer.Int32(11),
			Protocol: &udpProtocol,
		}}
	}
	subset2 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"10.1.1.2"},
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     utilpointer.String("p12"),
			Port:     utilpointer.Int32(12),
			Protocol: &udpProtocol,
		}}
	}
	namedPortLocal := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1,
			func(eps *discovery.EndpointSlice) {
				eps.AddressType = discovery.AddressTypeIPv4
				eps.Endpoints = []discovery.Endpoint{{
					Addresses: []string{"10.1.1.1"},
					NodeName:  &nodeName,
				}}
				eps.Ports = []discovery.EndpointPort{{
					Name:     utilpointer.String("p11"),
					Port:     utilpointer.Int32(11),
					Protocol: &udpProtocol,
				}}
			}),
	}
	namedPort := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1, subset1),
	}
	namedPortRenamed := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1,
			func(eps *discovery.EndpointSlice) {
				eps.AddressType = discovery.AddressTypeIPv4
				eps.Endpoints = []discovery.Endpoint{{
					Addresses: []string{"10.1.1.1"},
				}}
				eps.Ports = []discovery.EndpointPort{{
					Name:     utilpointer.String("p11-2"),
					Port:     utilpointer.Int32(11),
					Protocol: &udpProtocol,
				}}
			}),
	}
	namedPortRenumbered := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1,
			func(eps *discovery.EndpointSlice) {
				eps.AddressType = discovery.AddressTypeIPv4
				eps.Endpoints = []discovery.Endpoint{{
					Addresses: []string{"10.1.1.1"},
				}}
				eps.Ports = []discovery.EndpointPort{{
					Name:     utilpointer.String("p11"),
					Port:     utilpointer.Int32(22),
					Protocol: &udpProtocol,
				}}
			}),
	}
	namedPortsLocalNoLocal := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1,
			func(eps *discovery.EndpointSlice) {
				eps.AddressType = discovery.AddressTypeIPv4
				eps.Endpoints = []discovery.Endpoint{{
					Addresses: []string{"10.1.1.1"},
				}, {
					Addresses: []string{"10.1.1.2"},
					NodeName:  &nodeName,
				}}
				eps.Ports = []discovery.EndpointPort{{
					Name:     utilpointer.String("p11"),
					Port:     utilpointer.Int32(11),
					Protocol: &udpProtocol,
				}, {
					Name:     utilpointer.String("p12"),
					Port:     utilpointer.Int32(12),
					Protocol: &udpProtocol,
				}}
			}),
	}
	multipleSubsets := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1, subset1),
		makeTestEndpointSlice("ns1", "ep1", 2, subset2),
	}
	subsetLocal := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"10.1.1.2"},
			NodeName:  &nodeName,
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     utilpointer.String("p12"),
			Port:     utilpointer.Int32(12),
			Protocol: &udpProtocol,
		}}
	}
	multipleSubsetsWithLocal := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1, subset1),
		makeTestEndpointSlice("ns1", "ep1", 2, subsetLocal),
	}
	subsetMultiplePortsLocal := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"10.1.1.1"},
			NodeName:  &nodeName,
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     utilpointer.String("p11"),
			Port:     utilpointer.Int32(11),
			Protocol: &udpProtocol,
		}, {
			Name:     utilpointer.String("p12"),
			Port:     utilpointer.Int32(12),
			Protocol: &udpProtocol,
		}}
	}
	subset3 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"10.1.1.3"},
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     utilpointer.String("p13"),
			Port:     utilpointer.Int32(13),
			Protocol: &udpProtocol,
		}}
	}
	multipleSubsetsMultiplePortsLocal := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1, subsetMultiplePortsLocal),
		makeTestEndpointSlice("ns1", "ep1", 2, subset3),
	}
	subsetMultipleIPsPorts1 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"10.1.1.1"},
		}, {
			Addresses: []string{"10.1.1.2"},
			NodeName:  &nodeName,
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     utilpointer.String("p11"),
			Port:     utilpointer.Int32(11),
			Protocol: &udpProtocol,
		}, {
			Name:     utilpointer.String("p12"),
			Port:     utilpointer.Int32(12),
			Protocol: &udpProtocol,
		}}
	}
	subsetMultipleIPsPorts2 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"10.1.1.3"},
		}, {
			Addresses: []string{"10.1.1.4"},
			NodeName:  &nodeName,
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     utilpointer.String("p13"),
			Port:     utilpointer.Int32(13),
			Protocol: &udpProtocol,
		}, {
			Name:     utilpointer.String("p14"),
			Port:     utilpointer.Int32(14),
			Protocol: &udpProtocol,
		}}
	}
	subsetMultipleIPsPorts3 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"10.2.2.1"},
		}, {
			Addresses: []string{"10.2.2.2"},
			NodeName:  &nodeName,
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     utilpointer.String("p21"),
			Port:     utilpointer.Int32(21),
			Protocol: &udpProtocol,
		}, {
			Name:     utilpointer.String("p22"),
			Port:     utilpointer.Int32(22),
			Protocol: &udpProtocol,
		}}
	}
	multipleSubsetsIPsPorts := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1, subsetMultipleIPsPorts1),
		makeTestEndpointSlice("ns1", "ep1", 2, subsetMultipleIPsPorts2),
		makeTestEndpointSlice("ns2", "ep2", 1, subsetMultipleIPsPorts3),
	}
	complexSubset1 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"10.2.2.2"},
			NodeName:  &nodeName,
		}, {
			Addresses: []string{"10.2.2.22"},
			NodeName:  &nodeName,
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     utilpointer.String("p22"),
			Port:     utilpointer.Int32(22),
			Protocol: &udpProtocol,
		}}
	}
	complexSubset2 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"10.2.2.3"},
			NodeName:  &nodeName,
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     utilpointer.String("p23"),
			Port:     utilpointer.Int32(23),
			Protocol: &udpProtocol,
		}}
	}
	complexSubset3 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"10.4.4.4"},
			NodeName:  &nodeName,
		}, {
			Addresses: []string{"10.4.4.5"},
			NodeName:  &nodeName,
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     utilpointer.String("p44"),
			Port:     utilpointer.Int32(44),
			Protocol: &udpProtocol,
		}}
	}
	complexSubset4 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"10.4.4.6"},
			NodeName:  &nodeName,
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     utilpointer.String("p45"),
			Port:     utilpointer.Int32(45),
			Protocol: &udpProtocol,
		}}
	}
	complexSubset5 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"10.1.1.1"},
		}, {
			Addresses: []string{"10.1.1.11"},
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     utilpointer.String("p11"),
			Port:     utilpointer.Int32(11),
			Protocol: &udpProtocol,
		}}
	}
	complexSubset6 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"10.1.1.2"},
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     utilpointer.String("p12"),
			Port:     utilpointer.Int32(12),
			Protocol: &udpProtocol,
		}, {
			Name:     utilpointer.String("p122"),
			Port:     utilpointer.Int32(122),
			Protocol: &udpProtocol,
		}}
	}
	complexSubset7 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"10.3.3.3"},
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     utilpointer.String("p33"),
			Port:     utilpointer.Int32(33),
			Protocol: &udpProtocol,
		}}
	}
	complexSubset8 := func(eps *discovery.EndpointSlice) {
		eps.AddressType = discovery.AddressTypeIPv4
		eps.Endpoints = []discovery.Endpoint{{
			Addresses: []string{"10.4.4.4"},
			NodeName:  &nodeName,
		}}
		eps.Ports = []discovery.EndpointPort{{
			Name:     utilpointer.String("p44"),
			Port:     utilpointer.Int32(44),
			Protocol: &udpProtocol,
		}}
	}
	complexBefore := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1, subset1),
		nil,
		makeTestEndpointSlice("ns2", "ep2", 1, complexSubset1),
		makeTestEndpointSlice("ns2", "ep2", 2, complexSubset2),
		nil,
		makeTestEndpointSlice("ns4", "ep4", 1, complexSubset3),
		makeTestEndpointSlice("ns4", "ep4", 2, complexSubset4),
	}
	complexAfter := []*discovery.EndpointSlice{
		makeTestEndpointSlice("ns1", "ep1", 1, complexSubset5),
		makeTestEndpointSlice("ns1", "ep1", 2, complexSubset6),
		nil,
		nil,
		makeTestEndpointSlice("ns3", "ep3", 1, complexSubset7),
		makeTestEndpointSlice("ns4", "ep4", 1, complexSubset8),
		nil,
	}

	// Arrange (test cases)
	testCases := []struct {
		// previousEndpoints and currentEndpoints are used to call appropriate
		// handlers OnEndpoints* (based on whether corresponding values are nil
		// or non-nil) and must be of equal length.
		previousEndpoints         []*discovery.EndpointSlice
		currentEndpoints          []*discovery.EndpointSlice
		oldEndpoints              map[proxy.ServicePortName][]*endpointsInfo
		expectedResult            map[proxy.ServicePortName][]*endpointsInfo
		expectedStaleEndpoints    []proxy.ServiceEndpoint
		expectedStaleServiceNames map[proxy.ServicePortName]bool
		expectedHealthchecks      map[types.NamespacedName]int
	}{{
		// Case[0]: nothing
		oldEndpoints:              map[proxy.ServicePortName][]*endpointsInfo{},
		expectedResult:            map[proxy.ServicePortName][]*endpointsInfo{},
		expectedStaleEndpoints:    []proxy.ServiceEndpoint{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		// Case[1]: no change, named port, local
		previousEndpoints: namedPortLocal,
		currentEndpoints:  namedPortLocal,
		oldEndpoints: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{ip: "10.1.1.1:11", isLocal: false, ready: true, serving: true, terminating: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{ip: "10.1.1.1:11", isLocal: false, ready: true, serving: true, terminating: false},
			},
		},
		expectedStaleEndpoints:    []proxy.ServiceEndpoint{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 1,
		},
	}, {
		// Case[2]: no change, multiple subsets
		previousEndpoints: multipleSubsets,
		currentEndpoints:  multipleSubsets,
		oldEndpoints: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{ip: "10.1.1.1:11", isLocal: false, ready: true, serving: true, terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{ip: "10.1.1.2:12", isLocal: false, ready: true, serving: true, terminating: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{ip: "10.1.1.1:11", isLocal: false, ready: true, serving: true, terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{ip: "10.1.1.2:12", isLocal: false, ready: true, serving: true, terminating: false},
			},
		},
		expectedStaleEndpoints:    []proxy.ServiceEndpoint{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		// Case[3]: no change, multiple subsets, multiple ports, local
		previousEndpoints: multipleSubsetsMultiplePortsLocal,
		currentEndpoints:  multipleSubsetsMultiplePortsLocal,
		oldEndpoints: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{ip: "10.1.1.1:11", isLocal: false, ready: true, serving: true, terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{ip: "10.1.1.1:12", isLocal: false, ready: true, serving: true, terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p13", v1.ProtocolUDP): {
				{ip: "10.1.1.3:13", isLocal: false, ready: true, serving: true, terminating: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{ip: "10.1.1.1:11", isLocal: false, ready: true, serving: true, terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{ip: "10.1.1.1:12", isLocal: false, ready: true, serving: true, terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p13", v1.ProtocolUDP): {
				{ip: "10.1.1.3:13", isLocal: false, ready: true, serving: true, terminating: false},
			},
		},
		expectedStaleEndpoints:    []proxy.ServiceEndpoint{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 1,
		},
	}, {
		// Case[4]: no change, multiple endpoints, subsets, IPs, and ports
		previousEndpoints: multipleSubsetsIPsPorts,
		currentEndpoints:  multipleSubsetsIPsPorts,
		oldEndpoints: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{ip: "10.1.1.1:11", isLocal: false, ready: true, serving: true, terminating: false},
				{ip: "10.1.1.2:11", isLocal: false, ready: true, serving: true, terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{ip: "10.1.1.1:12", isLocal: false, ready: true, serving: true, terminating: false},
				{ip: "10.1.1.2:12", isLocal: false, ready: true, serving: true, terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p13", v1.ProtocolUDP): {
				{ip: "10.1.1.3:13", isLocal: false, ready: true, serving: true, terminating: false},
				{ip: "10.1.1.4:13", isLocal: false, ready: true, serving: true, terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p14", v1.ProtocolUDP): {
				{ip: "10.1.1.3:14", isLocal: false, ready: true, serving: true, terminating: false},
				{ip: "10.1.1.4:14", isLocal: false, ready: true, serving: true, terminating: false},
			},
			makeServicePortName("ns2", "ep2", "p21", v1.ProtocolUDP): {
				{ip: "10.2.2.1:21", isLocal: false, ready: true, serving: true, terminating: false},
				{ip: "10.2.2.2:21", isLocal: false, ready: true, serving: true, terminating: false},
			},
			makeServicePortName("ns2", "ep2", "p22", v1.ProtocolUDP): {
				{ip: "10.2.2.1:22", isLocal: false, ready: true, serving: true, terminating: false},
				{ip: "10.2.2.2:22", isLocal: false, ready: true, serving: true, terminating: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{ip: "10.1.1.1:11", isLocal: false, ready: true, serving: true, terminating: false},
				{ip: "10.1.1.2:11", isLocal: false, ready: true, serving: true, terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{ip: "10.1.1.1:12", isLocal: false, ready: true, serving: true, terminating: false},
				{ip: "10.1.1.2:12", isLocal: false, ready: true, serving: true, terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p13", v1.ProtocolUDP): {
				{ip: "10.1.1.3:13", isLocal: false, ready: true, serving: true, terminating: false},
				{ip: "10.1.1.4:13", isLocal: false, ready: true, serving: true, terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p14", v1.ProtocolUDP): {
				{ip: "10.1.1.3:14", isLocal: false, ready: true, serving: true, terminating: false},
				{ip: "10.1.1.4:14", isLocal: false, ready: true, serving: true, terminating: false},
			},
			makeServicePortName("ns2", "ep2", "p21", v1.ProtocolUDP): {
				{ip: "10.2.2.1:21", isLocal: false, ready: true, serving: true, terminating: false},
				{ip: "10.2.2.2:21", isLocal: false, ready: true, serving: true, terminating: false},
			},
			makeServicePortName("ns2", "ep2", "p22", v1.ProtocolUDP): {
				{ip: "10.2.2.1:22", isLocal: false, ready: true, serving: true, terminating: false},
				{ip: "10.2.2.2:22", isLocal: false, ready: true, serving: true, terminating: false},
			},
		},
		expectedStaleEndpoints:    []proxy.ServiceEndpoint{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 2,
			makeNSN("ns2", "ep2"): 1,
		},
	}, {
		// Case[5]: add an Endpoints
		previousEndpoints: []*discovery.EndpointSlice{nil},
		currentEndpoints:  namedPortLocal,
		oldEndpoints:      map[proxy.ServicePortName][]*endpointsInfo{},
		expectedResult: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{ip: "10.1.1.1:11", isLocal: false, ready: true, serving: true, terminating: false},
			},
		},
		expectedStaleEndpoints: []proxy.ServiceEndpoint{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): true,
		},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 1,
		},
	}, {
		// Case[6]: remove an Endpoints
		previousEndpoints: namedPortLocal,
		currentEndpoints:  []*discovery.EndpointSlice{nil},
		oldEndpoints: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{ip: "10.1.1.1:11", isLocal: false, ready: true, serving: true, terminating: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*endpointsInfo{},
		expectedStaleEndpoints: []proxy.ServiceEndpoint{{
			Endpoint:        "10.1.1.1:11",
			ServicePortName: makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP),
		}},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		// Case[7]: add an IP and port
		previousEndpoints: namedPort,
		currentEndpoints:  namedPortsLocalNoLocal,
		oldEndpoints: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{ip: "10.1.1.1:11", isLocal: false, ready: true, serving: true, terminating: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{ip: "10.1.1.1:11", isLocal: false, ready: true, serving: true, terminating: false},
				{ip: "10.1.1.2:11", isLocal: false, ready: true, serving: true, terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{ip: "10.1.1.1:12", isLocal: false, ready: true, serving: true, terminating: false},
				{ip: "10.1.1.2:12", isLocal: false, ready: true, serving: true, terminating: false},
			},
		},
		expectedStaleEndpoints: []proxy.ServiceEndpoint{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): true,
		},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 1,
		},
	}, {
		// Case[8]: remove an IP and port
		previousEndpoints: namedPortsLocalNoLocal,
		currentEndpoints:  namedPort,
		oldEndpoints: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{ip: "10.1.1.1:11", isLocal: false, ready: true, serving: true, terminating: false},
				{ip: "10.1.1.2:11", isLocal: false, ready: true, serving: true, terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{ip: "10.1.1.1:12", isLocal: false, ready: true, serving: true, terminating: false},
				{ip: "10.1.1.2:12", isLocal: false, ready: true, serving: true, terminating: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{ip: "10.1.1.1:11", isLocal: false, ready: true, serving: true, terminating: false},
			},
		},
		expectedStaleEndpoints: []proxy.ServiceEndpoint{{
			Endpoint:        "10.1.1.2:11",
			ServicePortName: makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP),
		}, {
			Endpoint:        "10.1.1.1:12",
			ServicePortName: makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP),
		}, {
			Endpoint:        "10.1.1.2:12",
			ServicePortName: makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP),
		}},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		// Case[9]: add a subset
		previousEndpoints: []*discovery.EndpointSlice{namedPort[0], nil},
		currentEndpoints:  multipleSubsetsWithLocal,
		oldEndpoints: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{ip: "10.1.1.1:11", isLocal: false, ready: true, serving: true, terminating: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{ip: "10.1.1.1:11", isLocal: false, ready: true, serving: true, terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{ip: "10.1.1.2:12", isLocal: false, ready: true, serving: true, terminating: false},
			},
		},
		expectedStaleEndpoints: []proxy.ServiceEndpoint{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): true,
		},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns1", "ep1"): 1,
		},
	}, {
		// Case[10]: remove a subset
		previousEndpoints: multipleSubsets,
		currentEndpoints:  []*discovery.EndpointSlice{namedPort[0], nil},
		oldEndpoints: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{ip: "10.1.1.1:11", isLocal: false, ready: true, serving: true, terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{ip: "10.1.1.2:12", isLocal: false, ready: true, serving: true, terminating: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{ip: "10.1.1.1:11", isLocal: false, ready: true, serving: true, terminating: false},
			},
		},
		expectedStaleEndpoints: []proxy.ServiceEndpoint{{
			Endpoint:        "10.1.1.2:12",
			ServicePortName: makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP),
		}},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		// Case[11]: rename a port
		previousEndpoints: namedPort,
		currentEndpoints:  namedPortRenamed,
		oldEndpoints: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{ip: "10.1.1.1:11", isLocal: false, ready: true, serving: true, terminating: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11-2", v1.ProtocolUDP): {
				{ip: "10.1.1.1:11", isLocal: false, ready: true, serving: true, terminating: false},
			},
		},
		expectedStaleEndpoints: []proxy.ServiceEndpoint{{
			Endpoint:        "10.1.1.1:11",
			ServicePortName: makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP),
		}},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{
			makeServicePortName("ns1", "ep1", "p11-2", v1.ProtocolUDP): true,
		},
		expectedHealthchecks: map[types.NamespacedName]int{},
	}, {
		// Case[12]: renumber a port
		previousEndpoints: namedPort,
		currentEndpoints:  namedPortRenumbered,
		oldEndpoints: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{ip: "10.1.1.1:11", isLocal: false, ready: true, serving: true, terminating: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{ip: "10.1.1.1:22", isLocal: false, ready: true, serving: true, terminating: false},
			},
		},
		expectedStaleEndpoints: []proxy.ServiceEndpoint{{
			Endpoint:        "10.1.1.1:11",
			ServicePortName: makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP),
		}},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{},
		expectedHealthchecks:      map[types.NamespacedName]int{},
	}, {
		// Case[13]: complex add and remove
		previousEndpoints: complexBefore,
		currentEndpoints:  complexAfter,
		oldEndpoints: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{ip: "10.1.1.1:11", isLocal: false, ready: true, serving: true, terminating: false},
			},
			makeServicePortName("ns2", "ep2", "p22", v1.ProtocolUDP): {
				{ip: "10.2.2.22:22", isLocal: false, ready: true, serving: true, terminating: false},
				{ip: "10.2.2.2:22", isLocal: false, ready: true, serving: true, terminating: false},
			},
			makeServicePortName("ns2", "ep2", "p23", v1.ProtocolUDP): {
				{ip: "10.2.2.3:23", isLocal: false, ready: true, serving: true, terminating: false},
			},
			makeServicePortName("ns4", "ep4", "p44", v1.ProtocolUDP): {
				{ip: "10.4.4.4:44", isLocal: false, ready: true, serving: true, terminating: false},
				{ip: "10.4.4.5:44", isLocal: false, ready: true, serving: true, terminating: false},
			},
			makeServicePortName("ns4", "ep4", "p45", v1.ProtocolUDP): {
				{ip: "10.4.4.6:45", isLocal: false, ready: true, serving: true, terminating: false},
			},
		},
		expectedResult: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{ip: "10.1.1.11:11", isLocal: false, ready: true, serving: true, terminating: false},
				{ip: "10.1.1.1:11", isLocal: false, ready: true, serving: true, terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP): {
				{ip: "10.1.1.2:12", isLocal: false, ready: true, serving: true, terminating: false},
			},
			makeServicePortName("ns1", "ep1", "p122", v1.ProtocolUDP): {
				{ip: "10.1.1.2:122", isLocal: false, ready: true, serving: true, terminating: false},
			},
			makeServicePortName("ns3", "ep3", "p33", v1.ProtocolUDP): {
				{ip: "10.3.3.3:33", isLocal: false, ready: true, serving: true, terminating: false},
			},
			makeServicePortName("ns4", "ep4", "p44", v1.ProtocolUDP): {
				{ip: "10.4.4.4:44", isLocal: false, ready: true, serving: true, terminating: false},
			},
		},
		expectedStaleEndpoints: []proxy.ServiceEndpoint{{
			Endpoint:        "10.2.2.2:22",
			ServicePortName: makeServicePortName("ns2", "ep2", "p22", v1.ProtocolUDP),
		}, {
			Endpoint:        "10.2.2.22:22",
			ServicePortName: makeServicePortName("ns2", "ep2", "p22", v1.ProtocolUDP),
		}, {
			Endpoint:        "10.2.2.3:23",
			ServicePortName: makeServicePortName("ns2", "ep2", "p23", v1.ProtocolUDP),
		}, {
			Endpoint:        "10.4.4.5:44",
			ServicePortName: makeServicePortName("ns4", "ep4", "p44", v1.ProtocolUDP),
		}, {
			Endpoint:        "10.4.4.6:45",
			ServicePortName: makeServicePortName("ns4", "ep4", "p45", v1.ProtocolUDP),
		}},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{
			makeServicePortName("ns1", "ep1", "p12", v1.ProtocolUDP):  true,
			makeServicePortName("ns1", "ep1", "p122", v1.ProtocolUDP): true,
			makeServicePortName("ns3", "ep3", "p33", v1.ProtocolUDP):  true,
		},
		expectedHealthchecks: map[types.NamespacedName]int{
			makeNSN("ns4", "ep4"): 1,
		},
	}, {
		// Case[14]: change from 0 endpoint address to 1 unnamed port
		previousEndpoints: emptyEndpointSlices,
		currentEndpoints:  namedPort,
		oldEndpoints:      map[proxy.ServicePortName][]*endpointsInfo{},
		expectedResult: map[proxy.ServicePortName][]*endpointsInfo{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): {
				{ip: "10.1.1.1:11", isLocal: false, ready: true, serving: true, terminating: false},
			},
		},
		expectedStaleEndpoints: []proxy.ServiceEndpoint{},
		expectedStaleServiceNames: map[proxy.ServicePortName]bool{
			makeServicePortName("ns1", "ep1", "p11", v1.ProtocolUDP): true,
		},
		expectedHealthchecks: map[types.NamespacedName]int{},
	},
	}

	for tci, tc := range testCases {
		// Arrange (individually)
		testHNS := NewHCNUtils(&hcntesting.FakeHCN{})
		proxier := NewFakeProxier(testHNS, NETWORK_TYPE_OVERLAY)
		proxier.hostname = nodeName

		// Act
		// First check that after adding all previous versions of endpoints,
		// the proxier.oldEndpoints is as we expect.
		for i := range tc.previousEndpoints {
			if tc.previousEndpoints[i] != nil {
				proxier.OnEndpointSliceAdd(tc.previousEndpoints[i])
			}
		}
		proxier.endpointsMap.Update(proxier.endpointsChanges)
		compareEndpointsMapsExceptChainName(t, tci, proxier.endpointsMap, tc.oldEndpoints)

		// Now let's call appropriate handlers to get to state we want to be.
		if len(tc.previousEndpoints) != len(tc.currentEndpoints) {
			t.Fatalf("[%d] different lengths of previous and current endpoints", tci)
			continue
		}

		for i := range tc.previousEndpoints {
			prev, curr := tc.previousEndpoints[i], tc.currentEndpoints[i]
			switch {
			case prev == nil:
				proxier.OnEndpointSliceAdd(curr)
			case curr == nil:
				proxier.OnEndpointSliceDelete(prev)
			default:
				proxier.OnEndpointSliceUpdate(prev, curr)
			}
		}
		result := proxier.endpointsMap.Update(proxier.endpointsChanges)
		newMap := proxier.endpointsMap

		// Assert
		compareEndpointsMapsExceptChainName(t, tci, newMap, tc.expectedResult)
		if len(result.StaleEndpoints) != len(tc.expectedStaleEndpoints) {
			t.Errorf("[%d] expected %d staleEndpoints, got %d: %v", tci, len(tc.expectedStaleEndpoints), len(result.StaleEndpoints), result.StaleEndpoints)
		}
		for _, x := range tc.expectedStaleEndpoints {
			found := false
			for _, stale := range result.StaleEndpoints {
				if stale == x {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("[%d] expected staleEndpoints[%v], but didn't find it: %v", tci, x, result.StaleEndpoints)
			}
		}
		if len(result.StaleServiceNames) != len(tc.expectedStaleServiceNames) {
			t.Errorf("[%d] expected %d staleServiceNames, got %d: %v", tci, len(tc.expectedStaleServiceNames), len(result.StaleServiceNames), result.StaleServiceNames)
		}
		for svcName := range tc.expectedStaleServiceNames {
			found := false
			for _, stale := range result.StaleServiceNames {
				if stale == svcName {
					found = true
				}
			}
			if !found {
				t.Errorf("[%d] expected staleServiceNames[%v], but didn't find it: %v", tci, svcName, result.StaleServiceNames)
			}
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

// TestFindRemoteSubnetProviderAddress checks if the searched RemoteSubnetProviderAddress
// is the one expected
func TestFindRemoteSubnetProviderAddress(t *testing.T) {
	// Arrange
	networkInfo, _ := NewHCNUtils(&hcntesting.FakeHCN{}).getNetworkByName("TestNetwork")

	// Act
	pa := networkInfo.findRemoteSubnetProviderAddress(providerAddress)

	// Assert
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

func deleteEndpointSlices(proxier *Proxier, allEndpointSlices ...*discovery.EndpointSlice) {
	for i := range allEndpointSlices {
		proxier.OnEndpointSliceDelete(allEndpointSlices[i])
	}

	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	proxier.endpointSlicesSynced = true
}

func populateEndpointSlices(proxier *Proxier, allEndpointSlices ...*discovery.EndpointSlice) {
	for i := range allEndpointSlices {
		proxier.OnEndpointSliceAdd(allEndpointSlices[i])
	}
}

func makeTestEndpointSlice(namespace, name string, sliceNum int, epsFunc func(*discovery.EndpointSlice)) *discovery.EndpointSlice {
	eps := &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-%d", name, sliceNum),
			Namespace: namespace,
			Labels:    map[string]string{discovery.LabelServiceName: name},
		},
	}
	epsFunc(eps)
	return eps
}

func addTestPort(array []v1.ServicePort, name string, protocol v1.Protocol, port, nodeport int32, targetPort int) []v1.ServicePort {
	svcport := v1.ServicePort{
		Name:       name,
		Protocol:   protocol,
		Port:       port,
		NodePort:   nodeport,
		TargetPort: intstr.FromInt(targetPort),
	}
	return append(array, svcport)
}

func makeServicePortName(ns, name, port string, protocol v1.Protocol) proxy.ServicePortName {
	return proxy.ServicePortName{
		NamespacedName: makeNSN(ns, name),
		Port:           port,
		Protocol:       protocol,
	}
}

func compareEndpointsMapsExceptChainName(t *testing.T, tci int, newMap proxy.EndpointsMap, expected map[proxy.ServicePortName][]*endpointsInfo) {
	if len(newMap) != len(expected) {
		t.Errorf("[%d] expected %d results, got %d: %v", tci, len(expected), len(newMap), newMap)
	}
	for x := range expected {
		if len(newMap[x]) != len(expected[x]) {
			t.Errorf("[%d] expected %d endpoints for %v, got %d", tci, len(expected[x]), x, len(newMap[x]))
		} else {
			for i := range expected[x] {
				newEp := newMap[x][i]
				if newEp.String() != expected[x][i].ip ||
					newEp.GetIsLocal() != expected[x][i].isLocal {
					t.Errorf("[%d] expected new[%v][%d] to be %v, got %v", tci, x, i, expected[x][i], newEp)
				}
			}
		}
	}
}
