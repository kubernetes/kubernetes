//go:build windows
// +build windows

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

package winkernel

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/proxy"
	mockhcn "k8s.io/kubernetes/pkg/proxy/winkernel/testing"
	netutils "k8s.io/utils/net"
	"k8s.io/utils/ptr"
)

// Helper function to create test endpoints
func createTestEndpoints(ips []string, port int) []endpointInfo {
	endpoints := make([]endpointInfo, len(ips))
	for i, ip := range ips {
		endpoints[i] = endpointInfo{
			hnsID:   fmt.Sprintf("test-ep-%d", i),
			ip:      ip,
			port:    uint16(port),
			isLocal: false,
		}
	}
	return endpoints
}

func TestManageLoadbalancer(t *testing.T) {
	tests := []struct {
		name                    string
		expectedLBIDs           []string
		testUpdateLB            bool
		modifyLBSupported       bool
		endpointsAvailableForLB bool
		endpointCount           int
		expectedSuccess         bool
		shouldFailUpdate        bool
		shouldFailCreate        bool
	}{
		{
			name:                    "successful create when update supported",
			expectedLBIDs:           []string{"LBID-1"},
			modifyLBSupported:       true,
			endpointsAvailableForLB: true,
			endpointCount:           2,
			expectedSuccess:         true,
			shouldFailUpdate:        false,
		},
		{
			name:                    "successful create when update not supported",
			expectedLBIDs:           []string{"LBID-1"},
			modifyLBSupported:       false,
			endpointsAvailableForLB: true,
			endpointCount:           2,
			expectedSuccess:         true,
			shouldFailCreate:        false,
		},
		{
			name:                    "successful update when update supported",
			expectedLBIDs:           []string{"LBID-1", "LBID-1"},
			modifyLBSupported:       true,
			testUpdateLB:            true,
			endpointsAvailableForLB: true,
			endpointCount:           2,
			expectedSuccess:         true,
			shouldFailUpdate:        false,
		},
		{
			name:                    "successful update when update not supported",
			expectedLBIDs:           []string{"LBID-1", "LBID-2"},
			modifyLBSupported:       false,
			testUpdateLB:            true,
			endpointsAvailableForLB: true,
			endpointCount:           2,
			expectedSuccess:         true,
			shouldFailCreate:        false,
		},
		{
			name:                    "skip create when no endpoints available",
			endpointsAvailableForLB: false,
			endpointCount:           0,
			expectedSuccess:         true,
		},
		{
			name:                    "fail when update fails",
			expectedLBIDs:           []string{"LBID-1", "LBID-1"},
			modifyLBSupported:       true,
			testUpdateLB:            true,
			endpointsAvailableForLB: true,
			endpointCount:           2,
			expectedSuccess:         true,
			shouldFailUpdate:        true,
		},
		{
			name:                    "fail when create fails",
			modifyLBSupported:       true,
			testUpdateLB:            false,
			endpointsAvailableForLB: true,
			endpointCount:           2,
			expectedSuccess:         false,
			shouldFailCreate:        true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			proxier := NewFakeProxier(t, testNodeName, netutils.ParseIPSloppy("10.0.0.1"), NETWORK_TYPE_OVERLAY, true)
			hcnMock := (proxier.hcn).(*mockhcn.HcnMock)
			proxier.supportedFeatures.ModifyLoadbalancer = tt.modifyLBSupported
			hcnMock.ShouldFailCreateLB = tt.shouldFailCreate
			hcnMock.ShouldFailUpdateLB = tt.shouldFailUpdate
			endpoints := createTestEndpoints([]string{"10.0.0.10", "10.0.0.11"}, 8080)
			if tt.endpointCount == 0 {
				endpoints = []endpointInfo{}
			}

			lbConfig := &loadbalancerConfig{
				loadbalancerType:        loadbalancerTypeClusterIP,
				hnsID:                   "",
				srcVip:                  "10.0.0.100",
				vip:                     "10.0.0.200",
				protocol:                6, // TCP
				internalPort:            8080,
				externalPort:            80,
				winProxyOptimization:    true,
				endpointsAvailableForLB: tt.endpointsAvailableForLB,
				lbFlags:                 loadBalancerFlags{},
				endpoints:               endpoints,
				queriedLoadBalancers:    make(map[loadBalancerIdentifier]*loadBalancerInfo),
			}

			success := proxier.manageLoadbalancer(lbConfig)

			if tt.shouldFailCreate {
				assert.False(t, success, "LB Create: Expected Failure, got %v", success)
			} else {
				assert.True(t, success, "LB Create: Expected success, got %v", success)
			}

			if len(tt.expectedLBIDs) > 0 {
				assert.Equal(t, tt.expectedLBIDs[0], lbConfig.hnsID, "LB Create: Expected Hns Loadbalancer Id %v does not match actual Loadbalancer Id: %v.", tt.expectedLBIDs[0], lbConfig.hnsID)
			} else {
				assert.Empty(t, lbConfig.hnsID, "LB Create: Expected no Hns Loadbalancer Id, got: %v", lbConfig.hnsID)
			}

			if tt.testUpdateLB {
				lbConfig.endpoints = createTestEndpoints([]string{"10.0.0.10", "10.0.0.11", "10.0.0.12"}, 8080)
				success = proxier.manageLoadbalancer(lbConfig)
				if tt.shouldFailUpdate {
					assert.False(t, success, "LB Update: Expected Failure, got %v", success)
				} else {
					assert.True(t, success, "LB Update: Expected success, got %v", success)
				}
				if len(tt.expectedLBIDs) > 1 {
					if tt.modifyLBSupported {
						assert.Equal(t, tt.expectedLBIDs[0], lbConfig.hnsID, "LB Update: Expected Hns Loadbalancer Id %v does not match actual Loadbalancer Id: %v.", tt.expectedLBIDs[0], lbConfig.hnsID)
					} else {
						assert.Equal(t, tt.expectedLBIDs[1], lbConfig.hnsID, "LB Update: Expected Hns Loadbalancer Id %v does not match actual Loadbalancer Id: %v.", tt.expectedLBIDs[1], lbConfig.hnsID)
					}
				}
			}
		})
	}
}

func TestManageClusterIPLoadbalancer(t *testing.T) {
	proxier := NewFakeProxier(t, testNodeName, netutils.ParseIPSloppy("10.0.0.1"), NETWORK_TYPE_OVERLAY, true)
	// svcInfo := createTestServiceInfo("test-svc", "10.0.0.200", 80, 8080, 0)
	endpoints := createTestEndpoints([]string{"10.0.0.10", "10.0.0.11"}, 8080)
	queriedLBs := make(map[loadBalancerIdentifier]*loadBalancerInfo)

	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
		Protocol:       v1.ProtocolTCP,
	}

	makeServiceMap(proxier,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeClusterIP
			svc.Spec.ClusterIP = "10.0.0.200"
			svc.Spec.ExternalIPs = []string{"50.60.70.81"}
			svc.Spec.SessionAffinity = v1.ServiceAffinityClientIP
			svc.Spec.SessionAffinityConfig = &v1.SessionAffinityConfig{
				ClientIP: &v1.ClientIPConfig{
					TimeoutSeconds: ptr.To[int32](v1.DefaultClientIPServiceAffinitySeconds),
				},
			}
			svc.Spec.Ports = []v1.ServicePort{{
				Name:       svcPortName.Port,
				Port:       int32(80),
				Protocol:   v1.ProtocolTCP,
				TargetPort: intstr.FromInt32(8080),
			}}
		}),
	)
	proxier.setInitialized(true)
	proxier.syncProxyRules()

	svc := proxier.svcPortMap[svcPortName]
	svcInfo, ok := svc.(*serviceInfo)
	assert.True(t, ok, "Failed to cast serviceInfo %q", svcPortName.String())
	success := proxier.manageClusterIPLoadbalancer("10.0.0.100", svcInfo, endpoints, queriedLBs)
	assert.True(t, success, "Expected success true, got false")
	assert.Equal(t, svcInfo.hnsID, "LBID-1", "Expected hnsID to be LBID-1, but got: %v", svcInfo.hnsID)
	// Verify LoadBalancer creation
	lbGet, lbGetErr := proxier.hcn.GetLoadBalancerByID("LBID-1")
	assert.NoError(t, lbGetErr, "Expected no error getting LoadBalancer by ID")
	assert.NotNil(t, lbGet, "Expected LoadBalancer to be found")

	// Update without modify support
	endpoints = createTestEndpoints([]string{"10.0.0.10", "10.0.0.11", "10.0.0.12"}, 8080)
	success = proxier.manageClusterIPLoadbalancer("10.0.0.100", svcInfo, endpoints, queriedLBs)
	assert.True(t, success, "Expected success true, got false")
	assert.Equal(t, svcInfo.hnsID, "LBID-2", "Expected hnsID to be LBID-2, but got: %v", svcInfo.hnsID)
	// Verify that the old load balancer has been deleted. This is necessary because updating a load balancer
	// is handled by deleting the existing one and creating a new one, if direct modification is not supported.
	lbGet, lbGetErr = proxier.hcn.GetLoadBalancerByID("LBID-1")
	assert.Error(t, lbGetErr, "Expected error getting LoadBalancer by ID")
	assert.Nil(t, lbGet, "Expected LoadBalancer to be not found")
	// Verify new load balancer with ID LBID-2 exists
	lbGet, lbGetErr = proxier.hcn.GetLoadBalancerByID("LBID-2")
	assert.NoError(t, lbGetErr, "Expected no error getting LoadBalancer by ID")
	assert.NotNil(t, lbGet, "Expected LoadBalancer to be found")

	// Update with modify support
	proxier.supportedFeatures.ModifyLoadbalancer = true
	endpoints = createTestEndpoints([]string{"10.0.0.10", "10.0.0.11", "10.0.0.13"}, 8080)
	success = proxier.manageClusterIPLoadbalancer("10.0.0.100", svcInfo, endpoints, queriedLBs)
	assert.True(t, success, "Expected success true, got false")
	assert.Equal(t, svcInfo.hnsID, "LBID-2", "Expected hnsID to be LBID-2, but got: %v", svcInfo.hnsID)
	// Verify new load balancer with ID LBID-2 exists
	lbGet, lbGetErr = proxier.hcn.GetLoadBalancerByID("LBID-2")
	assert.NoError(t, lbGetErr, "Expected no error getting LoadBalancer by ID")
	assert.NotNil(t, lbGet, "Expected LoadBalancer to be found")
}

func TestManageNodePortLoadbalancer(t *testing.T) {
	proxier := NewFakeProxier(t, testNodeName, netutils.ParseIPSloppy("10.0.0.100"), NETWORK_TYPE_OVERLAY, true)
	endpoints := createTestEndpoints([]string{"10.0.0.10", "10.0.0.11"}, 8080)
	queriedLBs := make(map[loadBalancerIdentifier]*loadBalancerInfo)
	endpointsAvailableForLB := true

	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
		Protocol:       v1.ProtocolTCP,
	}

	makeServiceMap(proxier,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeNodePort
			svc.Spec.ClusterIP = "10.0.0.200"
			svc.Spec.ExternalIPs = []string{"50.60.70.81"}
			svc.Spec.SessionAffinity = v1.ServiceAffinityClientIP
			svc.Spec.SessionAffinityConfig = &v1.SessionAffinityConfig{
				ClientIP: &v1.ClientIPConfig{
					TimeoutSeconds: ptr.To[int32](v1.DefaultClientIPServiceAffinitySeconds),
				},
			}
			svc.Spec.Ports = []v1.ServicePort{{
				Name:       svcPortName.Port,
				Port:       int32(80),
				Protocol:   v1.ProtocolTCP,
				NodePort:   int32(3001),
				TargetPort: intstr.FromInt32(8080),
			}}
		}),
	)
	proxier.setInitialized(true)
	proxier.syncProxyRules()

	svc := proxier.svcPortMap[svcPortName]
	svcInfo, ok := svc.(*serviceInfo)
	assert.True(t, ok, "Failed to cast serviceInfo %q", svcPortName.String())
	success := proxier.manageNodePortLoadbalancer("10.0.0.100", svcInfo, endpoints, queriedLBs, endpointsAvailableForLB)
	assert.True(t, success, "Expected success true, got false")
	assert.Equal(t, svcInfo.nodePorthnsID, "LBID-1", "Expected hnsID to be LBID-1, but got: %v", svcInfo.hnsID)
	// Verify LoadBalancer creation
	lbGet, lbGetErr := proxier.hcn.GetLoadBalancerByID("LBID-1")
	assert.NoError(t, lbGetErr, "Expected no error getting LoadBalancer by ID")
	assert.NotNil(t, lbGet, "Expected LoadBalancer to be found")

	// Update without modify support
	endpoints = createTestEndpoints([]string{"10.0.0.10", "10.0.0.11", "10.0.0.12"}, 8080)
	success = proxier.manageNodePortLoadbalancer("10.0.0.100", svcInfo, endpoints, queriedLBs, endpointsAvailableForLB)
	assert.True(t, success, "Expected success true, got false")
	assert.Equal(t, svcInfo.nodePorthnsID, "LBID-2", "Expected hnsID to be LBID-2, but got: %v", svcInfo.hnsID)
	// Verify that the old load balancer has been deleted. This is necessary because updating a load balancer
	// is handled by deleting the existing one and creating a new one, if direct modification is not supported.
	lbGet, lbGetErr = proxier.hcn.GetLoadBalancerByID("LBID-1")
	assert.Error(t, lbGetErr, "Expected error getting LoadBalancer by ID")
	assert.Nil(t, lbGet, "Expected LoadBalancer to be not found")
	// Verify new load balancer with ID LBID-2 exists
	lbGet, lbGetErr = proxier.hcn.GetLoadBalancerByID("LBID-2")
	assert.NoError(t, lbGetErr, "Expected no error getting LoadBalancer by ID")
	assert.NotNil(t, lbGet, "Expected LoadBalancer to be found")

	// Update with modify support
	proxier.supportedFeatures.ModifyLoadbalancer = true
	endpoints = createTestEndpoints([]string{"10.0.0.10", "10.0.0.11", "10.0.0.13"}, 8080)
	success = proxier.manageNodePortLoadbalancer("10.0.0.100", svcInfo, endpoints, queriedLBs, endpointsAvailableForLB)
	assert.True(t, success, "Expected success true, got false")
	assert.Equal(t, svcInfo.nodePorthnsID, "LBID-2", "Expected hnsID to be LBID-2, but got: %v", svcInfo.hnsID)
	// Verify new load balancer with ID LBID-2 exists
	lbGet, lbGetErr = proxier.hcn.GetLoadBalancerByID("LBID-2")
	assert.NoError(t, lbGetErr, "Expected no error getting LoadBalancer by ID")
	assert.NotNil(t, lbGet, "Expected LoadBalancer to be found")
}

func TestManageExternalIPLoadbalancers(t *testing.T) {
	proxier := NewFakeProxier(t, testNodeName, netutils.ParseIPSloppy("10.0.0.100"), NETWORK_TYPE_OVERLAY, true)
	endpoints := createTestEndpoints([]string{"10.0.0.10", "10.0.0.11"}, 8080)
	queriedLBs := make(map[loadBalancerIdentifier]*loadBalancerInfo)
	endpointsAvailableForLB := true

	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
		Protocol:       v1.ProtocolTCP,
	}

	makeServiceMap(proxier,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeLoadBalancer
			svc.Spec.ClusterIP = "10.0.0.200"
			svc.Spec.ExternalIPs = []string{"50.60.70.81"}
			svc.Spec.SessionAffinity = v1.ServiceAffinityClientIP
			svc.Spec.SessionAffinityConfig = &v1.SessionAffinityConfig{
				ClientIP: &v1.ClientIPConfig{
					TimeoutSeconds: ptr.To[int32](v1.DefaultClientIPServiceAffinitySeconds),
				},
			}
			svc.Spec.Ports = []v1.ServicePort{{
				Name:       svcPortName.Port,
				Port:       int32(80),
				Protocol:   v1.ProtocolTCP,
				TargetPort: intstr.FromInt32(8080),
			}}
		}),
	)
	proxier.setInitialized(true)
	proxier.syncProxyRules()

	svc := proxier.svcPortMap[svcPortName]
	svcInfo, ok := svc.(*serviceInfo)
	assert.True(t, ok, "Failed to cast serviceInfo %q", svcPortName.String())
	success := proxier.manageExternalIPLoadbalancers("10.0.0.100", svcInfo, endpoints, queriedLBs, endpointsAvailableForLB)
	assert.True(t, success, "Expected success true, got false")
	assert.Equal(t, svcInfo.externalIPs[0].hnsID, "LBID-1", "Expected hnsID to be LBID-1, but got: %v", svcInfo.hnsID)
	// Verify LoadBalancer creation
	lbGet, lbGetErr := proxier.hcn.GetLoadBalancerByID("LBID-1")
	assert.NoError(t, lbGetErr, "Expected no error getting LoadBalancer by ID")
	assert.NotNil(t, lbGet, "Expected LoadBalancer to be found")

	// Update without modify support
	endpoints = createTestEndpoints([]string{"10.0.0.10", "10.0.0.11", "10.0.0.12"}, 8080)
	success = proxier.manageExternalIPLoadbalancers("10.0.0.100", svcInfo, endpoints, queriedLBs, endpointsAvailableForLB)
	assert.True(t, success, "Expected success true, got false")
	assert.Equal(t, svcInfo.externalIPs[0].hnsID, "LBID-2", "Expected hnsID to be LBID-2, but got: %v", svcInfo.hnsID)
	// Verify that the old load balancer has been deleted. This is necessary because updating a load balancer
	// is handled by deleting the existing one and creating a new one, if direct modification is not supported.
	lbGet, lbGetErr = proxier.hcn.GetLoadBalancerByID("LBID-1")
	assert.Error(t, lbGetErr, "Expected error getting LoadBalancer by ID")
	assert.Nil(t, lbGet, "Expected LoadBalancer to be not found")
	// Verify new load balancer with ID LBID-2 exists
	lbGet, lbGetErr = proxier.hcn.GetLoadBalancerByID("LBID-2")
	assert.NoError(t, lbGetErr, "Expected no error getting LoadBalancer by ID")
	assert.NotNil(t, lbGet, "Expected LoadBalancer to be found")
}

func TestManageIngressIPLoadbalancers(t *testing.T) {
	proxier := NewFakeProxier(t, testNodeName, netutils.ParseIPSloppy("10.0.0.100"), NETWORK_TYPE_OVERLAY, true)
	proxier.rootHnsEndpointName = mockhcn.DefaultRootEndpointName
	endpoints := createTestEndpoints([]string{"10.0.0.10", "10.0.0.11"}, 8080)
	queriedLBs := make(map[loadBalancerIdentifier]*loadBalancerInfo)
	endpointsAvailableForLB := true

	svcPortName := proxy.ServicePortName{
		NamespacedName: makeNSN("ns1", "svc1"),
		Port:           "p80",
		Protocol:       v1.ProtocolTCP,
	}

	makeServiceMap(proxier,
		makeTestService(svcPortName.Namespace, svcPortName.Name, func(svc *v1.Service) {
			svc.Spec.Type = v1.ServiceTypeLoadBalancer
			svc.Spec.ClusterIP = "10.0.0.200"
			svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{IP: "50.60.70.81"}}
			svc.Spec.SessionAffinity = v1.ServiceAffinityClientIP
			svc.Spec.SessionAffinityConfig = &v1.SessionAffinityConfig{
				ClientIP: &v1.ClientIPConfig{
					TimeoutSeconds: ptr.To[int32](v1.DefaultClientIPServiceAffinitySeconds),
				},
			}
			svc.Spec.Ports = []v1.ServicePort{{
				Name:       svcPortName.Port,
				Port:       int32(80),
				Protocol:   v1.ProtocolTCP,
				TargetPort: intstr.FromInt32(8080),
			}}
		}),
	)
	proxier.setInitialized(true)
	proxier.syncProxyRules()

	svc := proxier.svcPortMap[svcPortName]
	svcInfo, ok := svc.(*serviceInfo)
	assert.True(t, ok, "Failed to cast serviceInfo %q", svcPortName.String())
	success := proxier.manageIngressIPLoadbalancers("10.0.0.100", svcInfo, endpoints, queriedLBs, endpointsAvailableForLB)
	assert.True(t, success, "Expected success true, got false")
	assert.Equal(t, svcInfo.loadBalancerIngressIPs[0].hnsID, "LBID-1", "Expected hnsID to be LBID-1, but got: %v", svcInfo.hnsID)
	assert.Equal(t, svcInfo.loadBalancerIngressIPs[0].healthCheckHnsID, "LBID-2", "Expected healthCheckHnsID to be LBID-2, but got: %v", svcInfo.loadBalancerIngressIPs[0].healthCheckHnsID)
	// Verify LoadBalancer creation
	lbGet, lbGetErr := proxier.hcn.GetLoadBalancerByID("LBID-1")
	assert.NoError(t, lbGetErr, "Expected no error getting LoadBalancer by ID")
	assert.NotNil(t, lbGet, "Expected LoadBalancer to be found")

	// Update without modify support
	endpoints = createTestEndpoints([]string{"10.0.0.10", "10.0.0.11", "10.0.0.12"}, 8080)
	success = proxier.manageIngressIPLoadbalancers("10.0.0.100", svcInfo, endpoints, queriedLBs, endpointsAvailableForLB)
	assert.True(t, success, "Expected success true, got false")
	assert.Equal(t, svcInfo.loadBalancerIngressIPs[0].hnsID, "LBID-3", "Expected hnsID to be LBID-3, but got: %v", svcInfo.hnsID)
	// Since gateway endpoints are not changing, the health check loadbalancer won't get deleted and healthCheckHnsID will remain the same.
	assert.Equal(t, svcInfo.loadBalancerIngressIPs[0].healthCheckHnsID, "LBID-2", "Expected healthCheckHnsID to be LBID-2, but got: %v", svcInfo.loadBalancerIngressIPs[0].healthCheckHnsID)
	// Verify that the old load balancer has been deleted. This is necessary because updating a load balancer
	// is handled by deleting the existing one and creating a new one, if direct modification is not supported.
	lbGet, lbGetErr = proxier.hcn.GetLoadBalancerByID("LBID-1")
	assert.Error(t, lbGetErr, "Expected error getting LoadBalancer by ID")
	assert.Nil(t, lbGet, "Expected LoadBalancer to be not found")
	// Verify new load balancer with ID LBID-2 exists
	lbGet, lbGetErr = proxier.hcn.GetLoadBalancerByID("LBID-3")
	assert.NoError(t, lbGetErr, "Expected no error getting LoadBalancer by ID")
	assert.NotNil(t, lbGet, "Expected LoadBalancer to be found")
}
