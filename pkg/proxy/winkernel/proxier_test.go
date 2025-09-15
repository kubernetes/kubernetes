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
	"encoding/json"
	"fmt"
	"net"
	"strings"
	"testing"

	"github.com/Microsoft/hnslib/hcn"
	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/proxy/apis/config"
	"k8s.io/kubernetes/pkg/proxy/healthcheck"
	fakehcn "k8s.io/kubernetes/pkg/proxy/winkernel/testing"
	netutils "k8s.io/utils/net"
	"k8s.io/utils/ptr"
)

const (
	testNodeName      = "test-node"
	testNetwork       = "TestNetwork"
	ipAddress         = "10.0.0.1"
	prefixLen         = 24
	macAddress        = "00-11-22-33-44-55"
	destinationPrefix = "192.168.2.0/24"
	providerAddress   = "10.0.0.3"
	guid              = "123ABC"
	endpointGuid1     = "EPID-1"
	loadbalancerGuid1 = "LBID-1"
	endpointLocal     = "EP-LOCAL"
	endpointGw        = "EP-GW"
	epIpAddressGw     = "192.168.2.1"
	epMacAddressGw    = "00-11-22-33-44-66"
)

func newHnsNetwork(networkInfo *hnsNetworkInfo) *hcn.HostComputeNetwork {
	var policies []hcn.NetworkPolicy
	for _, remoteSubnet := range networkInfo.remoteSubnets {
		policySettings := hcn.RemoteSubnetRoutePolicySetting{
			DestinationPrefix:           remoteSubnet.destinationPrefix,
			IsolationId:                 remoteSubnet.isolationID,
			ProviderAddress:             remoteSubnet.providerAddress,
			DistributedRouterMacAddress: remoteSubnet.drMacAddress,
		}
		settings, _ := json.Marshal(policySettings)
		policy := hcn.NetworkPolicy{
			Type:     hcn.RemoteSubnetRoute,
			Settings: settings,
		}
		policies = append(policies, policy)
	}

	network := &hcn.HostComputeNetwork{
		Id:       networkInfo.id,
		Name:     networkInfo.name,
		Type:     hcn.NetworkType(networkInfo.networkType),
		Policies: policies,
	}
	return network
}

func NewFakeProxier(t *testing.T, nodeName string, nodeIP net.IP, networkType string, enableDSR bool) *Proxier {
	sourceVip := "192.168.1.2"

	// enable `WinDSR` feature gate
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, kubefeatures.WinDSR, true)

	config := config.KubeProxyWinkernelConfiguration{
		SourceVip:             sourceVip,
		EnableDSR:             enableDSR,
		NetworkName:           testNetwork,
		ForwardHealthCheckVip: true,
	}

	hcnMock := getHcnMock(networkType)

	proxier, _ := newProxierInternal(
		v1.IPv4Protocol,
		nodeName,
		nodeIP,
		healthcheck.NewFakeServiceHealthServer(),
		nil,
		0,
		hcnMock,
		&testHostMacProvider{macAddress: macAddress},
		config,
		false,
	)

	return proxier
}

func getHcnMock(networkType string) *fakehcn.HcnMock {
	var remoteSubnets []*remoteSubnetInfo
	rs := &remoteSubnetInfo{
		destinationPrefix: destinationPrefix,
		isolationID:       4096,
		providerAddress:   providerAddress,
		drMacAddress:      macAddress,
	}
	remoteSubnets = append(remoteSubnets, rs)
	hnsNetworkInfo := &hnsNetworkInfo{
		id:            strings.ToUpper(guid),
		name:          testNetwork,
		networkType:   networkType,
		remoteSubnets: remoteSubnets,
	}
	hnsNetwork := newHnsNetwork(hnsNetworkInfo)
	hcnMock := fakehcn.NewHcnMock(hnsNetwork)
	return hcnMock
}

func TestCreateServiceVip(t *testing.T) {
	proxier := NewFakeProxier(t, testNodeName, netutils.ParseIPSloppy("10.0.0.1"), NETWORK_TYPE_OVERLAY, true)
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

	makeServiceMap(proxier,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = "NodePort"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.ExternalIPs = []string{svcExternalIPs}
			svc.Spec.SessionAffinity = v1.ServiceAffinityClientIP
			svc.Spec.SessionAffinityConfig = &v1.SessionAffinityConfig{
				ClientIP: &v1.ClientIPConfig{
					TimeoutSeconds: ptr.To[int32](v1.DefaultClientIPServiceAffinitySeconds),
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

	svc := proxier.svcPortMap[svcPortName]
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
	proxier := NewFakeProxier(t, testNodeName, netutils.ParseIPSloppy("10.0.0.1"), NETWORK_TYPE_OVERLAY, false)
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
	populateEndpointSlices(proxier,
		makeTestEndpointSlice(svcPortName.Namespace, svcPortName.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIpAddressRemote},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To(svcPortName.Port),
				Port:     ptr.To(int32(svcPort)),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}),
	)

	proxier.setInitialized(true)
	proxier.syncProxyRules()

	ep := proxier.endpointsMap[svcPortName][0]
	epInfo, ok := ep.(*endpointInfo)
	if !ok {
		t.Errorf("Failed to cast endpointInfo %q", svcPortName.String())

	} else {
		if epInfo.hnsID != "EPID-3" {
			t.Errorf("%v does not match %v", epInfo.hnsID, endpointGuid1)
		}
	}

	if *proxier.endPointsRefCount["EPID-3"] <= 0 {
		t.Errorf("RefCount not incremented. Current value: %v", *proxier.endPointsRefCount[endpointGuid1])
	}

	if *proxier.endPointsRefCount["EPID-3"] != *epInfo.refCount {
		t.Errorf("Global refCount: %v does not match endpoint refCount: %v", *proxier.endPointsRefCount[endpointGuid1], *epInfo.refCount)
	}
}

func TestCreateRemoteEndpointL2Bridge(t *testing.T) {
	proxier := NewFakeProxier(t, testNodeName, netutils.ParseIPSloppy("10.0.0.1"), "L2Bridge", false)
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
	populateEndpointSlices(proxier,
		makeTestEndpointSlice(svcPortName.Namespace, svcPortName.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIpAddressRemote},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To(svcPortName.Port),
				Port:     ptr.To(int32(svcPort)),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}),
	)
	proxier.setInitialized(true)
	proxier.syncProxyRules()
	ep := proxier.endpointsMap[svcPortName][0]
	epInfo, ok := ep.(*endpointInfo)
	if !ok {
		t.Errorf("Failed to cast endpointInfo %q", svcPortName.String())

	} else {
		if epInfo.hnsID != endpointGuid1 {
			t.Errorf("%v does not match %v", epInfo.hnsID, endpointGuid1)
		}
	}

	if *proxier.endPointsRefCount[endpointGuid1] <= 0 {
		t.Errorf("RefCount not incremented. Current value: %v", *proxier.endPointsRefCount[endpointGuid1])
	}

	if *proxier.endPointsRefCount[endpointGuid1] != *epInfo.refCount {
		t.Errorf("Global refCount: %v does not match endpoint refCount: %v", *proxier.endPointsRefCount[endpointGuid1], *epInfo.refCount)
	}
}

func TestDsrEndpointsAreCreatedCorrectly(t *testing.T) {
	proxier := NewFakeProxier(t, "testhost", netutils.ParseIPSloppy("10.0.0.1"), NETWORK_TYPE_OVERLAY, true)
	if proxier == nil {
		t.Fatal("Failed to create proxier")
	}

	svcIP := "10.20.30.41"
	svcPort := 80
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
		Protocol:       v1.ProtocolTCP,
	}

	makeServiceMap(proxier,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = "ClusterIP"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     svcPortName.Port,
				Port:     int32(svcPort),
				Protocol: v1.ProtocolTCP,
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
				Name:     ptr.To(svcPortName.Port),
				Port:     ptr.To(int32(svcPort)),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}),
	)

	proxier.setInitialized(true)
	proxier.syncProxyRules()

	ep := proxier.endpointsMap[svcPortName][0]
	epInfo, ok := ep.(*endpointInfo)
	if !ok {
		t.Errorf("Failed to cast endpointInfo %q", svcPortName.String())
	}

	if epInfo.hnsID == "" {
		t.Errorf("Expected HNS ID to be set for endpoint %s, but got empty value", epIpAddressRemote)
	}
}

func TestDsrNotAppliedToClusterTrafficPolicy(t *testing.T) {
	proxier := NewFakeProxier(t, "testhost", netutils.ParseIPSloppy("10.0.0.1"), NETWORK_TYPE_OVERLAY, true)
	if proxier == nil {
		t.Fatal("Failed to create proxier")
	}

	svcIP := "10.20.30.41"
	svcPort := 80
	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
		Protocol:       v1.ProtocolTCP,
	}

	makeServiceMap(proxier,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = "ClusterIP"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyCluster
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     svcPortName.Port,
				Port:     int32(svcPort),
				Protocol: v1.ProtocolTCP,
			}}
		}),
	)

	proxier.setInitialized(true)
	proxier.syncProxyRules()

	svc := proxier.svcPortMap[svcPortName]
	svcInfo, ok := svc.(*serviceInfo)
	if !ok {
		t.Errorf("Failed to cast serviceInfo %q", svcPortName.String())
	}

	if svcInfo.localTrafficDSR {
		t.Errorf("Expected localTrafficDSR to be false for ExternalTrafficPolicy=Cluster, but got true")
	}
}

func TestSharedRemoteEndpointDelete(t *testing.T) {
	proxier := NewFakeProxier(t, testNodeName, netutils.ParseIPSloppy("10.0.0.1"), "L2Bridge", true)
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
	populateEndpointSlices(proxier,
		makeTestEndpointSlice(svcPortName1.Namespace, svcPortName1.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIpAddressRemote},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To(svcPortName1.Port),
				Port:     ptr.To(int32(svcPort1)),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}),
		makeTestEndpointSlice(svcPortName2.Namespace, svcPortName2.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIpAddressRemote},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To(svcPortName2.Port),
				Port:     ptr.To(int32(svcPort2)),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}),
	)
	proxier.setInitialized(true)
	proxier.syncProxyRules()
	ep := proxier.endpointsMap[svcPortName1][0]
	epInfo, ok := ep.(*endpointInfo)
	if !ok {
		t.Errorf("Failed to cast endpointInfo %q", svcPortName1.String())

	} else {
		if epInfo.hnsID != endpointGuid1 {
			t.Errorf("%v does not match %v", epInfo.hnsID, endpointGuid1)
		}
	}

	if *proxier.endPointsRefCount[endpointGuid1] != 2 {
		t.Errorf("RefCount not incremented. Current value: %v", *proxier.endPointsRefCount[endpointGuid1])
	}

	if *proxier.endPointsRefCount[endpointGuid1] != *epInfo.refCount {
		t.Errorf("Global refCount: %v does not match endpoint refCount: %v", *proxier.endPointsRefCount[endpointGuid1], *epInfo.refCount)
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

	deleteEndpointSlices(proxier,
		makeTestEndpointSlice(svcPortName2.Namespace, svcPortName2.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIpAddressRemote},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To(svcPortName2.Port),
				Port:     ptr.To(int32(svcPort2)),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}),
	)

	proxier.setInitialized(true)
	proxier.syncProxyRules()

	ep = proxier.endpointsMap[svcPortName1][0]
	epInfo, ok = ep.(*endpointInfo)
	if !ok {
		t.Errorf("Failed to cast endpointInfo %q", svcPortName1.String())

	} else {
		if epInfo.hnsID != endpointGuid1 {
			t.Errorf("%v does not match %v", epInfo.hnsID, endpointGuid1)
		}
	}

	if *epInfo.refCount != 1 {
		t.Errorf("Incorrect Refcount. Current value: %v", *epInfo.refCount)
	}

	if *proxier.endPointsRefCount[endpointGuid1] != *epInfo.refCount {
		t.Errorf("Global refCount: %v does not match endpoint refCount: %v", *proxier.endPointsRefCount[endpointGuid1], *epInfo.refCount)
	}
}
func TestSharedRemoteEndpointUpdate(t *testing.T) {
	proxier := NewFakeProxier(t, testNodeName, netutils.ParseIPSloppy("10.0.0.1"), "L2Bridge", true)
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

	populateEndpointSlices(proxier,
		makeTestEndpointSlice(svcPortName1.Namespace, svcPortName1.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIpAddressRemote},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To(svcPortName1.Port),
				Port:     ptr.To(int32(svcPort1)),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}),
		makeTestEndpointSlice(svcPortName2.Namespace, svcPortName2.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIpAddressRemote},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To(svcPortName2.Port),
				Port:     ptr.To(int32(svcPort2)),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}),
	)

	proxier.setInitialized(true)
	proxier.syncProxyRules()
	ep := proxier.endpointsMap[svcPortName1][0]
	epInfo, ok := ep.(*endpointInfo)
	if !ok {
		t.Errorf("Failed to cast endpointInfo %q", svcPortName1.String())

	} else {
		if epInfo.hnsID != endpointGuid1 {
			t.Errorf("%v does not match %v", epInfo.hnsID, endpointGuid1)
		}
	}

	if *proxier.endPointsRefCount[endpointGuid1] != 2 {
		t.Errorf("RefCount not incremented. Current value: %v", *proxier.endPointsRefCount[endpointGuid1])
	}

	if *proxier.endPointsRefCount[endpointGuid1] != *epInfo.refCount {
		t.Errorf("Global refCount: %v does not match endpoint refCount: %v", *proxier.endPointsRefCount[endpointGuid1], *epInfo.refCount)
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

	proxier.OnEndpointSliceUpdate(
		makeTestEndpointSlice(svcPortName1.Namespace, svcPortName1.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIpAddressRemote},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To(svcPortName1.Port),
				Port:     ptr.To(int32(svcPort1)),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}),
		makeTestEndpointSlice(svcPortName1.Namespace, svcPortName1.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIpAddressRemote},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To(svcPortName1.Port),
				Port:     ptr.To(int32(svcPort1)),
				Protocol: ptr.To(v1.ProtocolTCP),
			},
				{
					Name:     ptr.To("p443"),
					Port:     ptr.To[int32](443),
					Protocol: ptr.To(v1.ProtocolTCP),
				}}
		}))

	proxier.mu.Lock()
	proxier.endpointSlicesSynced = true
	proxier.mu.Unlock()

	proxier.setInitialized(true)
	proxier.syncProxyRules()

	ep = proxier.endpointsMap[svcPortName1][0]
	epInfo, ok = ep.(*endpointInfo)

	if !ok {
		t.Errorf("Failed to cast endpointInfo %q", svcPortName1.String())

	} else {
		if epInfo.hnsID != endpointGuid1 {
			t.Errorf("%v does not match %v", epInfo.hnsID, endpointGuid1)
		}
	}

	if *epInfo.refCount != 2 {
		t.Errorf("Incorrect refcount. Current value: %v", *epInfo.refCount)
	}

	if *proxier.endPointsRefCount[endpointGuid1] != *epInfo.refCount {
		t.Errorf("Global refCount: %v does not match endpoint refCount: %v", *proxier.endPointsRefCount[endpointGuid1], *epInfo.refCount)
	}
}

func TestCreateLoadBalancerWithoutDSR(t *testing.T) {
	proxier := NewFakeProxier(t, testNodeName, netutils.ParseIPSloppy("10.0.0.1"), NETWORK_TYPE_OVERLAY, false)
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
	populateEndpointSlices(proxier,
		makeTestEndpointSlice(svcPortName.Namespace, svcPortName.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIpAddressRemote},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To(svcPortName.Port),
				Port:     ptr.To(int32(svcPort)),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}),
	)

	proxier.setInitialized(true)
	proxier.syncProxyRules()

	svc := proxier.svcPortMap[svcPortName]
	svcInfo, ok := svc.(*serviceInfo)
	if !ok {
		t.Errorf("Failed to cast serviceInfo %q", svcPortName.String())

	}

	if svcInfo.hnsID != loadbalancerGuid1 {
		t.Errorf("%v does not match %v", svcInfo.hnsID, loadbalancerGuid1)
	}

	lb, err := proxier.hcn.GetLoadBalancerByID(svcInfo.hnsID)
	if err != nil {
		t.Errorf("Failed to fetch loadbalancer: %v", err)
	}

	if lb == nil {
		t.Errorf("Failed to fetch loadbalancer: %v", err)
	}

	if lb.Flags != hcn.LoadBalancerFlagsNone {
		t.Errorf("Incorrect loadbalancer flags. Current value: %v", lb.Flags)
	}
}

func TestCreateLoadBalancerWithDSR(t *testing.T) {
	proxier := NewFakeProxier(t, testNodeName, netutils.ParseIPSloppy("10.0.0.1"), NETWORK_TYPE_OVERLAY, true)
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
	populateEndpointSlices(proxier,
		makeTestEndpointSlice(svcPortName.Namespace, svcPortName.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIpAddressRemote},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To(svcPortName.Port),
				Port:     ptr.To(int32(svcPort)),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}),
	)

	proxier.setInitialized(true)
	proxier.syncProxyRules()

	svc := proxier.svcPortMap[svcPortName]
	svcInfo, ok := svc.(*serviceInfo)
	if !ok {
		t.Errorf("Failed to cast serviceInfo %q", svcPortName.String())

	}

	if svcInfo.hnsID != loadbalancerGuid1 {
		t.Errorf("%v does not match %v", svcInfo.hnsID, loadbalancerGuid1)
	}

	lb, err := proxier.hcn.GetLoadBalancerByID(svcInfo.hnsID)
	if err != nil {
		t.Errorf("Failed to fetch loadbalancer: %v", err)
	}

	if lb == nil {
		t.Errorf("Failed to fetch loadbalancer: %v", err)
	}

	if lb.Flags != hcn.LoadBalancerFlagsDSR {
		t.Errorf("Incorrect loadbalancer flags. Current value: %v", lb.Flags)
	}
}

func TestUpdateLoadBalancerWhenSupported(t *testing.T) {
	proxier := NewFakeProxier(t, testNodeName, netutils.ParseIPSloppy("10.0.0.1"), NETWORK_TYPE_OVERLAY, true)
	if proxier == nil {
		t.Error()
	}

	proxier.supportedFeatures.ModifyLoadbalancer = true

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
	populateEndpointSlices(proxier,
		makeTestEndpointSlice(svcPortName.Namespace, svcPortName.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIpAddressRemote},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To(svcPortName.Port),
				Port:     ptr.To(int32(svcPort)),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}),
	)

	proxier.setInitialized(true)
	proxier.syncProxyRules()

	svc := proxier.svcPortMap[svcPortName]
	svcInfo, ok := svc.(*serviceInfo)
	if !ok {
		t.Errorf("Failed to cast serviceInfo %q", svcPortName.String())

	} else {
		if svcInfo.hnsID != loadbalancerGuid1 {
			t.Errorf("%v does not match %v", svcInfo.hnsID, loadbalancerGuid1)
		}
	}

	proxier.setInitialized(false)

	proxier.OnEndpointSliceUpdate(
		makeTestEndpointSlice(svcPortName.Namespace, svcPortName.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIpAddressRemote},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To(svcPortName.Port),
				Port:     ptr.To(int32(svcPort)),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}),
		makeTestEndpointSlice(svcPortName.Namespace, svcPortName.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epPaAddress},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To(svcPortName.Port),
				Port:     ptr.To(int32(svcPort)),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}))

	proxier.mu.Lock()
	proxier.endpointSlicesSynced = true
	proxier.mu.Unlock()

	proxier.setInitialized(true)

	epObj, err := proxier.hcn.GetEndpointByID("EPID-3")
	if err != nil || epObj == nil {
		t.Errorf("Failed to fetch endpoint: EPID-3")
	}

	proxier.syncProxyRules()

	// The endpoint should be deleted as it is not present in the new endpoint slice
	epObj, err = proxier.hcn.GetEndpointByID("EPID-3")
	if err == nil || epObj != nil {
		t.Errorf("Failed to fetch endpoint: EPID-3")
	}

	ep := proxier.endpointsMap[svcPortName][0]
	epInfo, ok := ep.(*endpointInfo)

	epObj, err = proxier.hcn.GetEndpointByID("EPID-5")
	if err != nil || epObj == nil {
		t.Errorf("Failed to fetch endpoint: EPID-5")
	}

	if !ok {
		t.Errorf("Failed to cast endpointInfo %q", svcPortName.String())

	} else {
		if epInfo.hnsID != "EPID-5" {
			t.Errorf("%v does not match %v", epInfo.hnsID, "EPID-5")
		}
	}

	if *epInfo.refCount != 1 {
		t.Errorf("Incorrect refcount. Current value: %v", *epInfo.refCount)
	}

	if *proxier.endPointsRefCount["EPID-5"] != *epInfo.refCount {
		t.Errorf("Global refCount: %v does not match endpoint refCount: %v", *proxier.endPointsRefCount[endpointGuid1], *epInfo.refCount)
	}

	svc = proxier.svcPortMap[svcPortName]
	svcInfo, ok = svc.(*serviceInfo)
	if !ok {
		t.Errorf("Failed to cast serviceInfo %q", svcPortName.String())

	} else {
		// Loadbalancer id should not change after the update
		if svcInfo.hnsID != loadbalancerGuid1 {
			t.Errorf("%v does not match %v", svcInfo.hnsID, loadbalancerGuid1)
		}
	}

}

func TestUpdateLoadBalancerWhenUnsupported(t *testing.T) {
	proxier := NewFakeProxier(t, testNodeName, netutils.ParseIPSloppy("10.0.0.1"), NETWORK_TYPE_OVERLAY, true)
	if proxier == nil {
		t.Error()
	}

	// By default the value is false, for the readibility of the test case setting it to false again
	proxier.supportedFeatures.ModifyLoadbalancer = false

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
	populateEndpointSlices(proxier,
		makeTestEndpointSlice(svcPortName.Namespace, svcPortName.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIpAddressRemote},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To(svcPortName.Port),
				Port:     ptr.To(int32(svcPort)),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}),
	)

	proxier.setInitialized(true)
	proxier.syncProxyRules()

	svc := proxier.svcPortMap[svcPortName]
	svcInfo, ok := svc.(*serviceInfo)
	if !ok {
		t.Errorf("Failed to cast serviceInfo %q", svcPortName.String())

	} else {
		if svcInfo.hnsID != loadbalancerGuid1 {
			t.Errorf("%v does not match %v", svcInfo.hnsID, loadbalancerGuid1)
		}
	}

	proxier.setInitialized(false)

	proxier.OnEndpointSliceUpdate(
		makeTestEndpointSlice(svcPortName.Namespace, svcPortName.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIpAddressRemote},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To(svcPortName.Port),
				Port:     ptr.To(int32(svcPort)),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}),
		makeTestEndpointSlice(svcPortName.Namespace, svcPortName.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epPaAddress},
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To(svcPortName.Port),
				Port:     ptr.To(int32(svcPort)),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}))

	proxier.mu.Lock()
	proxier.endpointSlicesSynced = true
	proxier.mu.Unlock()

	proxier.setInitialized(true)

	epObj, err := proxier.hcn.GetEndpointByID("EPID-3")
	if err != nil || epObj == nil {
		t.Errorf("Failed to fetch endpoint: EPID-3")
	}

	proxier.syncProxyRules()

	// The endpoint should be deleted as it is not present in the new endpoint slice
	epObj, err = proxier.hcn.GetEndpointByID("EPID-3")
	if err == nil || epObj != nil {
		t.Errorf("Failed to fetch endpoint: EPID-3")
	}

	ep := proxier.endpointsMap[svcPortName][0]
	epInfo, ok := ep.(*endpointInfo)

	epObj, err = proxier.hcn.GetEndpointByID("EPID-5")
	if err != nil || epObj == nil {
		t.Errorf("Failed to fetch endpoint: EPID-5")
	}

	if !ok {
		t.Errorf("Failed to cast endpointInfo %q", svcPortName.String())

	} else {
		if epInfo.hnsID != "EPID-5" {
			t.Errorf("%v does not match %v", epInfo.hnsID, "EPID-5")
		}
	}

	if *epInfo.refCount != 1 {
		t.Errorf("Incorrect refcount. Current value: %v", *epInfo.refCount)
	}

	if *proxier.endPointsRefCount["EPID-5"] != *epInfo.refCount {
		t.Errorf("Global refCount: %v does not match endpoint refCount: %v", *proxier.endPointsRefCount[endpointGuid1], *epInfo.refCount)
	}

	svc = proxier.svcPortMap[svcPortName]
	svcInfo, ok = svc.(*serviceInfo)
	if !ok {
		t.Errorf("Failed to cast serviceInfo %q", svcPortName.String())

	} else {
		// Loadbalancer id should change after the update
		if svcInfo.hnsID != "LBID-3" {
			t.Errorf("%v does not match %v", svcInfo.hnsID, "LBID-3")
		}
	}

}

func TestCreateDsrLoadBalancer(t *testing.T) {
	proxier := NewFakeProxier(t, testNodeName, netutils.ParseIPSloppy("10.0.0.1"), NETWORK_TYPE_OVERLAY, true)
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
	lbIP := "11.21.31.41"

	makeServiceMap(proxier,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = "NodePort"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyLocal
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     svcPortName.Port,
				Port:     int32(svcPort),
				Protocol: v1.ProtocolTCP,
				NodePort: int32(svcNodePort),
			}}
			svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{
				IP: lbIP,
			}}
		}),
	)
	populateEndpointSlices(proxier,
		makeTestEndpointSlice(svcPortName.Namespace, svcPortName.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIpAddressRemote},
				NodeName:  ptr.To(testNodeName),
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To(svcPortName.Port),
				Port:     ptr.To(int32(svcPort)),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}),
	)

	hcn := (proxier.hcn).(*fakehcn.HcnMock)
	proxier.rootHnsEndpointName = endpointGw
	hcn.PopulateQueriedEndpoints(endpointLocal, guid, epIpAddressRemote, macAddress, prefixLen)
	hcn.PopulateQueriedEndpoints(endpointGw, guid, epIpAddressGw, epMacAddressGw, prefixLen)
	proxier.setInitialized(true)
	proxier.syncProxyRules()

	svc := proxier.svcPortMap[svcPortName]
	svcInfo, ok := svc.(*serviceInfo)
	if !ok {
		t.Errorf("Failed to cast serviceInfo %q", svcPortName.String())

	} else {
		if svcInfo.hnsID != loadbalancerGuid1 {
			t.Errorf("%v does not match %v", svcInfo.hnsID, loadbalancerGuid1)
		}
		if svcInfo.localTrafficDSR != true {
			t.Errorf("Failed to create DSR loadbalancer with local traffic policy")
		}
		if len(svcInfo.loadBalancerIngressIPs) == 0 {
			t.Errorf("svcInfo does not have any loadBalancerIngressIPs, %+v", svcInfo)
		}
		if svcInfo.loadBalancerIngressIPs[0].healthCheckHnsID != "LBID-4" {
			t.Errorf("The Hns Loadbalancer HealthCheck Id %v does not match %v. ServicePortName %q", svcInfo.loadBalancerIngressIPs[0].healthCheckHnsID, loadbalancerGuid1, svcPortName.String())
		}
	}
}

// TestClusterIPLBInCreateDsrLoadBalancer tests, if the available endpoints are remote,
// syncproxyrules only creates ClusterIP Loadbalancer and no NodePort, External IP or IngressIP
// loadbalancers will be created.
func TestClusterIPLBInCreateDsrLoadBalancer(t *testing.T) {
	proxier := NewFakeProxier(t, testNodeName, netutils.ParseIPSloppy("10.0.0.1"), NETWORK_TYPE_OVERLAY, false)

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
	lbIP := "11.21.31.41"

	makeServiceMap(proxier,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = "NodePort"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyLocal
			svc.Spec.Ports = []v1.ServicePort{{
				Name:     svcPortName.Port,
				Port:     int32(svcPort),
				Protocol: v1.ProtocolTCP,
				NodePort: int32(svcNodePort),
			}}
			svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{
				IP: lbIP,
			}}
		}),
	)
	populateEndpointSlices(proxier,
		makeTestEndpointSlice(svcPortName.Namespace, svcPortName.Name, 1, func(eps *discovery.EndpointSlice) {
			eps.AddressType = discovery.AddressTypeIPv4
			eps.Endpoints = []discovery.Endpoint{{
				Addresses: []string{epIpAddressRemote},
				NodeName:  ptr.To("testhost2"), // This will make this endpoint as a remote endpoint
			}}
			eps.Ports = []discovery.EndpointPort{{
				Name:     ptr.To(svcPortName.Port),
				Port:     ptr.To(int32(svcPort)),
				Protocol: ptr.To(v1.ProtocolTCP),
			}}
		}),
	)

	proxier.setInitialized(true)
	proxier.syncProxyRules()

	svc := proxier.svcPortMap[svcPortName]
	svcInfo, ok := svc.(*serviceInfo)
	if !ok {
		t.Errorf("Failed to cast serviceInfo %q", svcPortName.String())

	} else {
		// Checking ClusterIP Loadbalancer is created
		if svcInfo.hnsID != loadbalancerGuid1 {
			t.Errorf("%v does not match %v", svcInfo.hnsID, loadbalancerGuid1)
		}
		// Verifying NodePort Loadbalancer is not created
		if svcInfo.nodePorthnsID != "" {
			t.Errorf("NodePortHnsID %v is not empty.", svcInfo.nodePorthnsID)
		}
		// Verifying ExternalIP Loadbalancer is not created
		for _, externalIP := range svcInfo.externalIPs {
			if externalIP.hnsID != "" {
				t.Errorf("ExternalLBID %v is not empty.", externalIP.hnsID)
			}
		}
		// Verifying IngressIP Loadbalancer is not created
		for _, ingressIP := range svcInfo.loadBalancerIngressIPs {
			if ingressIP.hnsID != "" {
				t.Errorf("IngressLBID %v is not empty.", ingressIP.hnsID)
			}
		}
	}
}

func TestEndpointSliceWithInternalPortDifferentFromServicePort(t *testing.T) {
	proxier := NewFakeProxier(t, testNodeName, netutils.ParseIPSloppy("10.0.0.1"), NETWORK_TYPE_OVERLAY, true)
	assert.NotNil(t, proxier, "Failed to create proxier")

	proxier.servicesSynced = true
	proxier.endpointSlicesSynced = true

	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
		Protocol:       v1.ProtocolTCP,
	}

	svcSpec := v1.ServiceSpec{
		ClusterIP: "172.20.1.1",
		Selector:  map[string]string{"foo": "bar"},
		Ports: []v1.ServicePort{
			{Name: svcPortName.Port, Port: 80, TargetPort: intstr.FromInt32(80), Protocol: v1.ProtocolTCP}, // Mocking TargetPort as to same as service port (80)
		},
	}

	proxier.OnServiceAdd(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: svcPortName.Name, Namespace: svcPortName.Namespace},
		Spec:       svcSpec,
	})

	// Add initial endpoint slice
	endpointSlice := &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-1", svcPortName.Name),
			Namespace: svcPortName.Namespace,
			Labels:    map[string]string{discovery.LabelServiceName: svcPortName.Name},
		},
		Ports: []discovery.EndpointPort{{
			Name:     &svcPortName.Port,
			Port:     ptr.To[int32](8080), // Using container port 8080 which is different from service port 80
			Protocol: ptr.To(v1.ProtocolTCP),
		}},
		AddressType: discovery.AddressTypeIPv4,
		Endpoints: []discovery.Endpoint{{
			Addresses:  []string{"192.168.2.3"},
			Conditions: discovery.EndpointConditions{Ready: ptr.To(true)},
			NodeName:   ptr.To("testhost2"),
		}},
	}

	proxier.OnEndpointSliceAdd(endpointSlice)
	proxier.setInitialized(true)
	proxier.syncProxyRules()

	svc := proxier.svcPortMap[svcPortName]
	svcInfo, ok := svc.(*serviceInfo)
	assert.True(t, ok, "Failed to cast serviceInfo %q", svcPortName.String())
	assert.Equal(t, svcInfo.hnsID, loadbalancerGuid1, "The Hns Loadbalancer Id %v does not match %v. ServicePortName %q", svcInfo.hnsID, loadbalancerGuid1, svcPortName.String())

	lb, err := proxier.hcn.GetLoadBalancerByID(svcInfo.hnsID)
	assert.Equal(t, nil, err, "Failed to fetch loadbalancer: %v", err)
	assert.NotEqual(t, nil, lb, "Loadbalancer object should not be nil")
	assert.Equal(t, len(lb.PortMappings), 1, "PortMappings should have one and only one entry")
	assert.Equal(t, lb.PortMappings[0].InternalPort, uint16(8080), "InternalPort should be 8080")
	assert.Equal(t, lb.PortMappings[0].ExternalPort, uint16(80), "ExternalPort should be 80")

	ep := proxier.endpointsMap[svcPortName][0]
	epInfo, ok := ep.(*endpointInfo)
	assert.True(t, ok, "Failed to cast endpointInfo %q", svcPortName.String())
	assert.Equal(t, epInfo.hnsID, "EPID-3", "Hns EndpointId %v does not match %v. ServicePortName %q", epInfo.hnsID, endpointGuid1, svcPortName.String())
}

func TestEndpointSlice(t *testing.T) {
	proxier := NewFakeProxier(t, testNodeName, netutils.ParseIPSloppy("10.0.0.1"), NETWORK_TYPE_OVERLAY, true)
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
			Ports:     []v1.ServicePort{{Name: svcPortName.Port, TargetPort: intstr.FromInt32(80), Protocol: v1.ProtocolTCP}},
		},
	})

	// Add initial endpoint slice
	endpointSlice := &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-1", svcPortName.Name),
			Namespace: svcPortName.Namespace,
			Labels:    map[string]string{discovery.LabelServiceName: svcPortName.Name},
		},
		Ports: []discovery.EndpointPort{{
			Name:     &svcPortName.Port,
			Port:     ptr.To[int32](80),
			Protocol: ptr.To(v1.ProtocolTCP),
		}},
		AddressType: discovery.AddressTypeIPv4,
		Endpoints: []discovery.Endpoint{{
			Addresses:  []string{"192.168.2.3"},
			Conditions: discovery.EndpointConditions{Ready: ptr.To(true)},
			NodeName:   ptr.To("testhost2"),
		}},
	}

	proxier.OnEndpointSliceAdd(endpointSlice)
	proxier.setInitialized(true)
	proxier.syncProxyRules()

	svc := proxier.svcPortMap[svcPortName]
	svcInfo, ok := svc.(*serviceInfo)
	if !ok {
		t.Errorf("Failed to cast serviceInfo %q", svcPortName.String())

	} else {
		if svcInfo.hnsID != loadbalancerGuid1 {
			t.Errorf("The Hns Loadbalancer Id %v does not match %v. ServicePortName %q", svcInfo.hnsID, loadbalancerGuid1, svcPortName.String())
		}
	}

	ep := proxier.endpointsMap[svcPortName][0]
	epInfo, ok := ep.(*endpointInfo)
	if !ok {
		t.Errorf("Failed to cast endpointInfo %q", svcPortName.String())

	} else {
		if epInfo.hnsID != "EPID-3" {
			t.Errorf("Hns EndpointId %v does not match %v. ServicePortName %q", epInfo.hnsID, endpointGuid1, svcPortName.String())
		}
	}
}

func TestNoopEndpointSlice(t *testing.T) {
	p := Proxier{}
	p.endpointsChanges = proxy.NewEndpointsChangeTracker(v1.IPv4Protocol, "", nil, nil)
	p.OnEndpointSliceAdd(&discovery.EndpointSlice{})
	p.OnEndpointSliceUpdate(&discovery.EndpointSlice{}, &discovery.EndpointSlice{})
	p.OnEndpointSliceDelete(&discovery.EndpointSlice{})
	p.OnEndpointSlicesSynced()
}

func TestFindRemoteSubnetProviderAddress(t *testing.T) {
	proxier := NewFakeProxier(t, testNodeName, netutils.ParseIPSloppy("10.0.0.1"), NETWORK_TYPE_OVERLAY, true)
	if proxier == nil {
		t.Error()
	}

	networkInfo, _ := proxier.hns.getNetworkByName(testNetwork)
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

func TestWinDSRWithOverlayEnabled(t *testing.T) {
	proxier := NewFakeProxier(t, testNodeName, netutils.ParseIPSloppy("10.0.0.1"), NETWORK_TYPE_OVERLAY, true)
	if proxier == nil {
		t.Error("Failed to create proxier")
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
			svc.Spec.Type = "ClusterIP"
			svc.Spec.ClusterIP = svcIP
			svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyLocal
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

	svc := proxier.svcPortMap[svcPortName]
	svcInfo, ok := svc.(*serviceInfo)
	if !ok {
		t.Errorf("Failed to cast serviceInfo %q", svcPortName.String())
	}

	if !svcInfo.localTrafficDSR {
		t.Errorf("Expected localTrafficDSR to be enabled but got false")
	}
}

func TestDSRFeatureGateValidation(t *testing.T) {
	testCases := []struct {
		name          string
		enableDSR     bool
		featureGate   bool
		expectFailure bool
	}{
		{
			name:          "DSR enabled but feature gate disabled",
			enableDSR:     true,
			featureGate:   false,
			expectFailure: true,
		},
		{
			name:          "DSR enabled and feature gate enabled",
			enableDSR:     true,
			featureGate:   true,
			expectFailure: false,
		},
		{
			name:          "DSR disabled, feature gate does not matter",
			enableDSR:     false,
			featureGate:   false,
			expectFailure: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Mock feature gate
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, kubefeatures.WinDSR, tc.featureGate)

			config := config.KubeProxyWinkernelConfiguration{
				EnableDSR:   tc.enableDSR,
				NetworkName: testNetwork,
				SourceVip:   serviceVip,
			}

			hostMacProvider := &testHostMacProvider{macAddress: macAddress}

			hcnMock := getHcnMock(NETWORK_TYPE_OVERLAY)

			_, err := newProxierInternal(
				v1.IPv4Protocol,                       // ipFamily
				testNodeName,                          // nodeName
				netutils.ParseIPSloppy("192.168.1.1"), // nodeIP
				nil,                                   // serviceHealthServer (not needed in this unit test)
				nil,                                   // healthzServer (not needed in this unit test)
				0,                                     // healthzPort
				hcnMock,                               // hcnImpl
				hostMacProvider,                       // hostMacProvider
				config,                                // kube-proxy config
				false,                                 // waitForHNSOverlay
			)

			if tc.expectFailure {
				if err == nil {
					t.Errorf("Expected failure for case %q, but got success", tc.name)
				}
			} else {
				if err != nil {
					t.Errorf("Expected success for case %q, but got error: %v", tc.name, err)
				}
			}
		})
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

type testHostMacProvider struct {
	macAddress string
}

func (r *testHostMacProvider) GetHostMac(nodeIP net.IP) string {
	return r.macAddress
}
