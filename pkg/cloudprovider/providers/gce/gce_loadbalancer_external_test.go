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

package gce

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	computealpha "google.golang.org/api/compute/v0.alpha"
	compute "google.golang.org/api/compute/v1"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/mock"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
)

func TestEnsureStaticIP(t *testing.T) {
	fcas := NewFakeCloudAddressService()
	ipName := "some-static-ip"
	serviceName := ""
	region := "us-central1"

	// First ensure call
	ip, existed, err := ensureStaticIP(fcas, ipName, serviceName, region, "", NetworkTierDefault)
	if err != nil || existed || ip == "" {
		t.Fatalf(`ensureStaticIP(%v, %v, %v, %v, "") = %v, %v, %v; want valid ip, false, nil`, fcas, ipName, serviceName, region, ip, existed, err)
	}

	// Second ensure call
	var ipPrime string
	ipPrime, existed, err = ensureStaticIP(fcas, ipName, serviceName, region, ip, NetworkTierDefault)
	if err != nil || !existed || ip != ipPrime {
		t.Fatalf(`ensureStaticIP(%v, %v, %v, %v, %v) = %v, %v, %v; want %v, true, nil`, fcas, ipName, serviceName, region, ip, ipPrime, existed, err, ip)
	}
}

func TestEnsureStaticIPWithTier(t *testing.T) {
	s := NewFakeCloudAddressService()
	serviceName := ""
	region := "us-east1"

	for desc, tc := range map[string]struct {
		name     string
		netTier  NetworkTier
		expected string
	}{
		"Premium (default)": {
			name:     "foo-1",
			netTier:  NetworkTierPremium,
			expected: "PREMIUM",
		},
		"Standard": {
			name:     "foo-2",
			netTier:  NetworkTierStandard,
			expected: "STANDARD",
		},
	} {
		t.Run(desc, func(t *testing.T) {
			ip, existed, err := ensureStaticIP(s, tc.name, serviceName, region, "", tc.netTier)
			assert.NoError(t, err)
			assert.False(t, existed)
			assert.NotEqual(t, "", ip)
			// Get the Address from the fake address service and verify that the tier
			// is set correctly.
			alphaAddr, err := s.GetAlphaRegionAddress(tc.name, region)
			require.NoError(t, err)
			assert.Equal(t, tc.expected, alphaAddr.NetworkTier)
		})
	}
}

func TestVerifyRequestedIP(t *testing.T) {
	region := "test-region"
	lbRef := "test-lb"
	s := NewFakeCloudAddressService()

	for desc, tc := range map[string]struct {
		requestedIP     string
		fwdRuleIP       string
		netTier         NetworkTier
		addrList        []*computealpha.Address
		expectErr       bool
		expectUserOwned bool
	}{
		"requested IP exists": {
			requestedIP:     "1.1.1.1",
			netTier:         NetworkTierPremium,
			addrList:        []*computealpha.Address{{Name: "foo", Address: "1.1.1.1", NetworkTier: "PREMIUM"}},
			expectErr:       false,
			expectUserOwned: true,
		},
		"requested IP is not static, but is in use by the fwd rule": {
			requestedIP: "1.1.1.1",
			fwdRuleIP:   "1.1.1.1",
			netTier:     NetworkTierPremium,
			expectErr:   false,
		},
		"requested IP is not static and is not used by the fwd rule": {
			requestedIP: "1.1.1.1",
			fwdRuleIP:   "2.2.2.2",
			netTier:     NetworkTierPremium,
			expectErr:   true,
		},
		"no requested IP": {
			netTier:   NetworkTierPremium,
			expectErr: false,
		},
		"requested IP exists, but network tier does not match": {
			requestedIP: "1.1.1.1",
			netTier:     NetworkTierStandard,
			addrList:    []*computealpha.Address{{Name: "foo", Address: "1.1.1.1", NetworkTier: "PREMIUM"}},
			expectErr:   true,
		},
	} {
		t.Run(desc, func(t *testing.T) {
			s.SetRegionalAddresses(region, tc.addrList)
			isUserOwnedIP, err := verifyUserRequestedIP(s, region, tc.requestedIP, tc.fwdRuleIP, lbRef, tc.netTier)
			assert.Equal(t, tc.expectErr, err != nil, fmt.Sprintf("err: %v", err))
			assert.Equal(t, tc.expectUserOwned, isUserOwnedIP)
		})
	}
}

func TestCreateForwardingRuleWithTier(t *testing.T) {
	s := NewFakeCloudForwardingRuleService()
	// Common variables among the tests.
	ports := []v1.ServicePort{{Name: "foo", Protocol: v1.ProtocolTCP, Port: int32(123)}}
	region := "test-region"
	target := "test-target-pool"
	svcName := "foo-svc"

	for desc, tc := range map[string]struct {
		netTier      NetworkTier
		expectedRule *computealpha.ForwardingRule
	}{
		"Premium tier": {
			netTier: NetworkTierPremium,
			expectedRule: &computealpha.ForwardingRule{
				Name:        "lb-1",
				Description: `{"kubernetes.io/service-name":"foo-svc"}`,
				IPAddress:   "1.1.1.1",
				IPProtocol:  "TCP",
				PortRange:   "123-123",
				Target:      target,
				NetworkTier: "PREMIUM",
			},
		},
		"Standard tier": {
			netTier: NetworkTierStandard,
			expectedRule: &computealpha.ForwardingRule{
				Name:        "lb-2",
				Description: `{"kubernetes.io/service-name":"foo-svc"}`,
				IPAddress:   "2.2.2.2",
				IPProtocol:  "TCP",
				PortRange:   "123-123",
				Target:      target,
				NetworkTier: "STANDARD",
			},
		},
	} {
		t.Run(desc, func(t *testing.T) {
			lbName := tc.expectedRule.Name
			ipAddr := tc.expectedRule.IPAddress

			err := createForwardingRule(s, lbName, svcName, region, ipAddr, target, ports, tc.netTier)
			assert.NoError(t, err)

			alphaRule, err := s.GetAlphaRegionForwardingRule(lbName, region)
			assert.NoError(t, err)
			assert.Equal(t, tc.expectedRule, alphaRule)
		})
	}
}

func TestDeleteAddressWithWrongTier(t *testing.T) {
	region := "test-region"
	lbRef := "test-lb"
	s := NewFakeCloudAddressService()

	for desc, tc := range map[string]struct {
		addrName     string
		netTier      NetworkTier
		addrList     []*computealpha.Address
		expectDelete bool
	}{
		"Network tiers (premium) match; do nothing": {
			addrName: "foo1",
			netTier:  NetworkTierPremium,
			addrList: []*computealpha.Address{{Name: "foo1", Address: "1.1.1.1", NetworkTier: "PREMIUM"}},
		},
		"Network tiers (standard) match; do nothing": {
			addrName: "foo2",
			netTier:  NetworkTierStandard,
			addrList: []*computealpha.Address{{Name: "foo2", Address: "1.1.1.2", NetworkTier: "STANDARD"}},
		},
		"Wrong network tier (standard); delete address": {
			addrName:     "foo3",
			netTier:      NetworkTierPremium,
			addrList:     []*computealpha.Address{{Name: "foo3", Address: "1.1.1.3", NetworkTier: "STANDARD"}},
			expectDelete: true,
		},
		"Wrong network tier (preimium); delete address": {
			addrName:     "foo4",
			netTier:      NetworkTierStandard,
			addrList:     []*computealpha.Address{{Name: "foo4", Address: "1.1.1.4", NetworkTier: "PREMIUM"}},
			expectDelete: true,
		},
	} {
		t.Run(desc, func(t *testing.T) {
			s.SetRegionalAddresses(region, tc.addrList)
			// Sanity check to ensure we inject the right address.
			_, err := s.GetRegionAddress(tc.addrName, region)
			require.NoError(t, err)

			err = deleteAddressWithWrongTier(s, region, tc.addrName, lbRef, tc.netTier)
			assert.NoError(t, err)
			// Check whether the address still exists.
			_, err = s.GetRegionAddress(tc.addrName, region)
			if tc.expectDelete {
				assert.True(t, isNotFound(err))
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

const (
	gceProjectId = "test-project"
	gceRegion    = "test-region"
	zoneName     = "zone1"
	nodeName     = "test-node-1"
	clusterName  = "Test Cluster Name"
	clusterID    = "test-cluster-id"
)

var apiService = &v1.Service{
	Spec: v1.ServiceSpec{
		SessionAffinity: v1.ServiceAffinityClientIP,
		Type:            v1.ServiceTypeClusterIP,
		Ports:           []v1.ServicePort{{Protocol: v1.ProtocolTCP, Port: int32(123)}},
	},
}

func fakeGCECloud() (*GCECloud, error) {
	client, err := newOauthClient(nil)
	if err != nil {
		return nil, err
	}

	service, err := compute.New(client)
	if err != nil {
		return nil, err
	}

	fakeManager := newFakeManager(gceProjectId, gceRegion)

	alphaFeatureGate, err := NewAlphaFeatureGate([]string{})
	if err != nil {
		return nil, err
	}

	cloud := cloud.NewMockGCE()
	cloud.MockTargetPools.AddInstanceHook = mock.AddInstanceHook
	zonesWithNodes := createNodeZones([]string{zoneName})

	gce := GCECloud{
		region:             gceRegion,
		service:            service,
		manager:            fakeManager,
		managedZones:       []string{zoneName},
		projectID:          gceProjectId,
		AlphaFeatureGate:   alphaFeatureGate,
		nodeZones:          zonesWithNodes,
		nodeInformerSynced: func() bool { return true },
		c:                  cloud,
	}

	return &gce, nil
}

func createExternalLoadBalancer(gce *GCECloud) (*v1.LoadBalancerStatus, error) {
	err := gce.InsertInstance(
		gceProjectId,
		zoneName,
		&compute.Instance{
			Name: nodeName,
			Tags: &compute.Tags{
				Items: []string{nodeName},
			},
		},
	)

	if err != nil {
		return nil, err
	}

	nodes := []*v1.Node{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: nodeName,
				Labels: map[string]string{
					kubeletapis.LabelHostname: nodeName,
				},
			},
			Status: v1.NodeStatus{
				NodeInfo: v1.NodeSystemInfo{
					KubeProxyVersion: "v1.7.2",
				},
			},
		},
	}

	status, err := gce.ensureExternalLoadBalancer(
		clusterName,
		clusterID,
		apiService,
		nil,
		nodes,
	)

	return status, err
}

func TestEnsureExternalLoadBalancer(t *testing.T) {
	gce, err := fakeGCECloud()
	require.NoError(t, err)

	status, err := createExternalLoadBalancer(gce)
	assert.NoError(t, err)
	assert.NotEmpty(t, status.Ingress)

	lbName := cloudprovider.GetLoadBalancerName(apiService)

	// Check that Firewall is created
	firewall, err := gce.GetFirewall(MakeFirewallName(lbName))
	require.NoError(t, err)
	assert.Equal(t, []string{nodeName}, firewall.TargetTags)
	assert.Equal(t, []string{"0.0.0.0/0"}, firewall.SourceRanges)

	// Check that TargetPool is Created
	pool, err := gce.GetTargetPool(lbName, gceRegion)
	require.NoError(t, err)
	assert.Equal(t, lbName, pool.Name)
	assert.NotEmpty(t, pool.HealthChecks)
	assert.Equal(t, 1, len(pool.Instances))

	// Check that HealthCheck is created
	hcName := MakeNodesHealthCheckName(clusterID)
	healthcheck, err := gce.GetHttpHealthCheck(hcName)
	require.NoError(t, err)
	assert.Equal(t, hcName, healthcheck.Name)

	// Check that ForwardingRule is created
	fwdRule, err := gce.GetRegionForwardingRule(lbName, gceRegion)
	require.NoError(t, err)
	assert.Equal(t, lbName, fwdRule.Name)
	assert.Equal(t, "TCP", fwdRule.IPProtocol)
	assert.Equal(t, "123-123", fwdRule.PortRange)
}

func TestUpdateExternalLoadBalancer(t *testing.T) {
	gce, err := fakeGCECloud()
	require.NoError(t, err)

	_, err = createExternalLoadBalancer(gce)
	assert.NoError(t, err)

	newNodeName := "test-node-2"
	gce.InsertInstance(
		gceProjectId,
		zoneName,
		&compute.Instance{
			Name: newNodeName,
			Tags: &compute.Tags{
				Items: []string{newNodeName},
			},
		},
	)

	newNodes := []*v1.Node{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: nodeName,
				Labels: map[string]string{
					kubeletapis.LabelHostname: nodeName,
				},
			},
			Status: v1.NodeStatus{
				NodeInfo: v1.NodeSystemInfo{
					KubeProxyVersion: "v1.7.2",
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: newNodeName,
				Labels: map[string]string{
					kubeletapis.LabelHostname: newNodeName,
				},
			},
			Status: v1.NodeStatus{
				NodeInfo: v1.NodeSystemInfo{
					KubeProxyVersion: "v1.7.2",
				},
			},
		},
	}

	err = gce.updateExternalLoadBalancer(clusterName, apiService, newNodes)
	assert.NoError(t, err)

	lbName := cloudprovider.GetLoadBalancerName(apiService)

	// Check that TargetPool is updated with the new node
	pool, err := gce.GetTargetPool(lbName, gceRegion)
	require.NoError(t, err)

	assert.Equal(
		t,
		[]string{
			fmt.Sprintf("/zones/%s/instances/%s", zoneName, nodeName),
			fmt.Sprintf("/zones/%s/instances/%s", zoneName, newNodeName),
		},
		pool.Instances,
	)
}

func TestEnsureExternalLoadBalancerDeleted(t *testing.T) {
	gce, err := fakeGCECloud()
	require.NoError(t, err)

	_, err = createExternalLoadBalancer(gce)
	assert.NoError(t, err)

	err = gce.ensureExternalLoadBalancerDeleted(clusterName, clusterID, apiService)
	assert.NoError(t, err)

	lbName := cloudprovider.GetLoadBalancerName(apiService)

	// Check that Firewall is deleted
	firewall, err := gce.GetFirewall(MakeFirewallName(lbName))
	require.Error(t, err)
	assert.Nil(t, firewall)

	// Check that TargetPool is deleted
	pool, err := gce.GetTargetPool(lbName, gceRegion)
	require.Error(t, err)
	assert.Nil(t, pool)

	// Check that HealthCheck is deleted
	hcName := MakeNodesHealthCheckName(clusterID)
	healthcheck, err := gce.GetHttpHealthCheck(hcName)
	require.Error(t, err)
	assert.Nil(t, healthcheck)

	// Check forwarding rule is deleted
	fwdRule, err := gce.GetRegionForwardingRule(lbName, gceRegion)
	require.Error(t, err)
	assert.Nil(t, fwdRule)
}
