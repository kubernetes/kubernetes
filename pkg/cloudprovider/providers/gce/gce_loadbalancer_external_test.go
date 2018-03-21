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

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud"
)

func TestEnsureStaticIP(t *testing.T) {
	gce, err := fakeGCECloud(DefaultTestClusterValues())
	require.NoError(t, err)

	ipName := "some-static-ip"
	serviceName := "some-service"

	// First ensure call
	ip, existed, err := ensureStaticIP(gce, ipName, serviceName, gce.region, "", cloud.NetworkTierDefault)
	if err != nil || existed {
		t.Fatalf(`ensureStaticIP(%v, %v, %v, %v, "") = %v, %v, %v; want valid ip, false, nil`, gce, ipName, serviceName, gce.region, ip, existed, err)
	}

	// Second ensure call
	var ipPrime string
	ipPrime, existed, err = ensureStaticIP(gce, ipName, serviceName, gce.region, ip, cloud.NetworkTierDefault)
	if err != nil || !existed || ip != ipPrime {
		t.Fatalf(`ensureStaticIP(%v, %v, %v, %v, %v) = %v, %v, %v; want %v, true, nil`, gce, ipName, serviceName, gce.region, ip, ipPrime, existed, err, ip)
	}
}

func TestEnsureStaticIPWithTier(t *testing.T) {
	s, err := fakeGCECloud(DefaultTestClusterValues())
	require.NoError(t, err)

	serviceName := "some-service"

	for desc, tc := range map[string]struct {
		name     string
		netTier  cloud.NetworkTier
		expected string
	}{
		"Premium (default)": {
			name:     "foo-1",
			netTier:  cloud.NetworkTierPremium,
			expected: "PREMIUM",
		},
		"Standard": {
			name:     "foo-2",
			netTier:  cloud.NetworkTierStandard,
			expected: "STANDARD",
		},
	} {
		t.Run(desc, func(t *testing.T) {
			ip, existed, err := ensureStaticIP(s, tc.name, serviceName, s.region, "", tc.netTier)
			assert.NoError(t, err)
			assert.False(t, existed)
			assert.NotEqual(t, ip, "")
			// Get the Address from the fake address service and verify that the tier
			// is set correctly.
			alphaAddr, err := s.GetAlphaRegionAddress(tc.name, s.region)
			require.NoError(t, err)
			assert.Equal(t, tc.expected, alphaAddr.NetworkTier)
		})
	}
}

func TestVerifyRequestedIP(t *testing.T) {
	lbRef := "test-lb"

	for desc, tc := range map[string]struct {
		requestedIP     string
		fwdRuleIP       string
		netTier         cloud.NetworkTier
		addrList        []*computealpha.Address
		expectErr       bool
		expectUserOwned bool
	}{
		"requested IP exists": {
			requestedIP:     "1.1.1.1",
			netTier:         cloud.NetworkTierPremium,
			addrList:        []*computealpha.Address{{Name: "foo", Address: "1.1.1.1", NetworkTier: "PREMIUM"}},
			expectErr:       false,
			expectUserOwned: true,
		},
		"requested IP is not static, but is in use by the fwd rule": {
			requestedIP: "1.1.1.1",
			fwdRuleIP:   "1.1.1.1",
			netTier:     cloud.NetworkTierPremium,
			expectErr:   false,
		},
		"requested IP is not static and is not used by the fwd rule": {
			requestedIP: "1.1.1.1",
			fwdRuleIP:   "2.2.2.2",
			netTier:     cloud.NetworkTierPremium,
			expectErr:   true,
		},
		"no requested IP": {
			netTier:   cloud.NetworkTierPremium,
			expectErr: false,
		},
		"requested IP exists, but network tier does not match": {
			requestedIP: "1.1.1.1",
			netTier:     cloud.NetworkTierStandard,
			addrList:    []*computealpha.Address{{Name: "foo", Address: "1.1.1.1", NetworkTier: "PREMIUM"}},
			expectErr:   true,
		},
	} {
		t.Run(desc, func(t *testing.T) {
			s, err := fakeGCECloud(DefaultTestClusterValues())
			require.NoError(t, err)

			for _, addr := range tc.addrList {
				s.ReserveAlphaRegionAddress(addr, s.region)
			}
			isUserOwnedIP, err := verifyUserRequestedIP(s, s.region, tc.requestedIP, tc.fwdRuleIP, lbRef, tc.netTier)
			assert.Equal(t, tc.expectErr, err != nil, fmt.Sprintf("err: %v", err))
			assert.Equal(t, tc.expectUserOwned, isUserOwnedIP)
		})
	}
}

func TestCreateForwardingRuleWithTier(t *testing.T) {
	// Common variables among the tests.
	ports := []v1.ServicePort{{Name: "foo", Protocol: v1.ProtocolTCP, Port: int32(123)}}
	target := "test-target-pool"
	vals := DefaultTestClusterValues()
	serviceName := "foo-svc"

	baseLinkUrl := "https://www.googleapis.com/compute/%v/projects/%v/regions/%v/forwardingRules/%v"

	for desc, tc := range map[string]struct {
		netTier      cloud.NetworkTier
		expectedRule *computealpha.ForwardingRule
	}{
		"Premium tier": {
			netTier: cloud.NetworkTierPremium,
			expectedRule: &computealpha.ForwardingRule{
				Name:        "lb-1",
				Description: `{"kubernetes.io/service-name":"foo-svc"}`,
				IPAddress:   "1.1.1.1",
				IPProtocol:  "TCP",
				PortRange:   "123-123",
				Target:      target,
				NetworkTier: "PREMIUM",
				SelfLink:    fmt.Sprintf(baseLinkUrl, "v1", vals.ProjectID, vals.Region, "lb-1"),
			},
		},
		"Standard tier": {
			netTier: cloud.NetworkTierStandard,
			expectedRule: &computealpha.ForwardingRule{
				Name:        "lb-2",
				Description: `{"kubernetes.io/service-name":"foo-svc"}`,
				IPAddress:   "2.2.2.2",
				IPProtocol:  "TCP",
				PortRange:   "123-123",
				Target:      target,
				NetworkTier: "STANDARD",
				SelfLink:    fmt.Sprintf(baseLinkUrl, "alpha", vals.ProjectID, vals.Region, "lb-2"),
			},
		},
	} {
		t.Run(desc, func(t *testing.T) {
			s, err := fakeGCECloud(vals)
			require.NoError(t, err)

			lbName := tc.expectedRule.Name
			ipAddr := tc.expectedRule.IPAddress

			err = createForwardingRule(s, lbName, serviceName, s.region, ipAddr, target, ports, tc.netTier)
			assert.NoError(t, err)

			alphaRule, err := s.GetAlphaRegionForwardingRule(lbName, s.region)
			assert.NoError(t, err)
			assert.Equal(t, tc.expectedRule, alphaRule)
		})
	}
}

func TestDeleteAddressWithWrongTier(t *testing.T) {
	lbRef := "test-lb"

	s, err := fakeGCECloud(DefaultTestClusterValues())
	require.NoError(t, err)

	// Enable the cloud.NetworkTiers feature
	s.AlphaFeatureGate.features[AlphaFeatureNetworkTiers] = true

	for desc, tc := range map[string]struct {
		addrName     string
		netTier      cloud.NetworkTier
		addrList     []*computealpha.Address
		expectDelete bool
	}{
		"Network tiers (premium) match; do nothing": {
			addrName: "foo1",
			netTier:  cloud.NetworkTierPremium,
			addrList: []*computealpha.Address{{Name: "foo1", Address: "1.1.1.1", NetworkTier: "PREMIUM"}},
		},
		"Network tiers (standard) match; do nothing": {
			addrName: "foo2",
			netTier:  cloud.NetworkTierStandard,
			addrList: []*computealpha.Address{{Name: "foo2", Address: "1.1.1.2", NetworkTier: "STANDARD"}},
		},
		"Wrong network tier (standard); delete address": {
			addrName:     "foo3",
			netTier:      cloud.NetworkTierPremium,
			addrList:     []*computealpha.Address{{Name: "foo3", Address: "1.1.1.3", NetworkTier: "STANDARD"}},
			expectDelete: true,
		},
		"Wrong network tier (premium); delete address": {
			addrName:     "foo4",
			netTier:      cloud.NetworkTierStandard,
			addrList:     []*computealpha.Address{{Name: "foo4", Address: "1.1.1.4", NetworkTier: "PREMIUM"}},
			expectDelete: true,
		},
	} {
		t.Run(desc, func(t *testing.T) {
			for _, addr := range tc.addrList {
				s.ReserveAlphaRegionAddress(addr, s.region)
			}

			// Sanity check to ensure we inject the right address.
			_, err = s.GetRegionAddress(tc.addrName, s.region)
			require.NoError(t, err)

			err = deleteAddressWithWrongTier(s, s.region, tc.addrName, lbRef, tc.netTier)
			assert.NoError(t, err)
			// Check whether the address still exists.
			_, err = s.GetRegionAddress(tc.addrName, s.region)
			if tc.expectDelete {
				assert.True(t, isNotFound(err))
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func createExternalLoadBalancer(gce *GCECloud, nodeNames []string, clusterName, clusterID, zoneName string) (*v1.LoadBalancerStatus, error) {
	nodes, err := createAndInsertNodes(gce, nodeNames, zoneName)
	if err != nil {
		return nil, err
	}

	return gce.ensureExternalLoadBalancer(
		clusterName,
		clusterID,
		fakeApiService,
		nil,
		nodes,
	)
}

func TestEnsureExternalLoadBalancer(t *testing.T) {
	vals := DefaultTestClusterValues()
	nodeName := "test-node-1"

	gce, err := fakeGCECloud(vals)
	require.NoError(t, err)

	status, err := createExternalLoadBalancer(gce, []string{nodeName}, vals.ClusterName, vals.ClusterID, vals.ZoneName)
	assert.NoError(t, err)
	assert.NotEmpty(t, status.Ingress)

	lbName := cloudprovider.GetLoadBalancerName(fakeApiService)
	hcName := MakeNodesHealthCheckName(vals.ClusterID)

	// Check that Firewalls are created for the LoadBalancer and the HealthCheck
	fwNames := []string{
		MakeFirewallName(lbName),
		MakeHealthCheckFirewallName(vals.ClusterID, hcName, true),
	}

	for _, fwName := range fwNames {
		firewall, err := gce.GetFirewall(fwName)
		require.NoError(t, err)
		assert.Equal(t, []string{nodeName}, firewall.TargetTags)
		assert.NotEmpty(t, firewall.SourceRanges)
	}

	// Check that TargetPool is Created
	pool, err := gce.GetTargetPool(lbName, gce.region)
	require.NoError(t, err)
	assert.Equal(t, lbName, pool.Name)
	assert.NotEmpty(t, pool.HealthChecks)
	assert.Equal(t, 1, len(pool.Instances))

	// Check that HealthCheck is created
	healthcheck, err := gce.GetHttpHealthCheck(hcName)
	require.NoError(t, err)
	assert.Equal(t, hcName, healthcheck.Name)

	// Check that ForwardingRule is created
	fwdRule, err := gce.GetRegionForwardingRule(lbName, gce.region)
	require.NoError(t, err)
	assert.Equal(t, lbName, fwdRule.Name)
	assert.Equal(t, "TCP", fwdRule.IPProtocol)
	assert.Equal(t, "123-123", fwdRule.PortRange)
}

func TestUpdateExternalLoadBalancer(t *testing.T) {
	vals := DefaultTestClusterValues()
	nodeName := "test-node-1"

	gce, err := fakeGCECloud((DefaultTestClusterValues()))
	require.NoError(t, err)

	_, err = createExternalLoadBalancer(gce, []string{nodeName}, vals.ClusterName, vals.ClusterID, vals.ZoneName)
	assert.NoError(t, err)

	newNodeName := "test-node-2"
	newNodes, err := createAndInsertNodes(gce, []string{nodeName, newNodeName}, vals.ZoneName)
	assert.NoError(t, err)

	// Add the new node, then check that it is properly added to the TargetPool
	err = gce.updateExternalLoadBalancer("", fakeApiService, newNodes)
	assert.NoError(t, err)

	lbName := cloudprovider.GetLoadBalancerName(fakeApiService)

	pool, err := gce.GetTargetPool(lbName, gce.region)
	require.NoError(t, err)

	// TODO: when testify is updated to v1.2.0+, use ElementsMatch instead
	assert.Contains(
		t,
		pool.Instances,
		fmt.Sprintf("/zones/%s/instances/%s", vals.ZoneName, nodeName),
	)

	assert.Contains(
		t,
		pool.Instances,
		fmt.Sprintf("/zones/%s/instances/%s", vals.ZoneName, newNodeName),
	)

	newNodes, err = createAndInsertNodes(gce, []string{nodeName}, vals.ZoneName)
	assert.NoError(t, err)

	// Remove the new node by calling updateExternalLoadBalancer with a list
	// only containing the old node, and test that the TargetPool no longer
	// contains the new node.
	err = gce.updateExternalLoadBalancer(vals.ClusterName, fakeApiService, newNodes)
	assert.NoError(t, err)

	pool, err = gce.GetTargetPool(lbName, gce.region)
	require.NoError(t, err)

	assert.Equal(
		t,
		[]string{fmt.Sprintf("/zones/%s/instances/%s", vals.ZoneName, nodeName)},
		pool.Instances,
	)
}

func TestEnsureExternalLoadBalancerDeleted(t *testing.T) {
	vals := DefaultTestClusterValues()
	gce, err := fakeGCECloud(vals)
	require.NoError(t, err)

	_, err = createExternalLoadBalancer(gce, []string{"test-node-1"}, vals.ClusterName, vals.ClusterID, vals.ZoneName)
	assert.NoError(t, err)

	err = gce.ensureExternalLoadBalancerDeleted(vals.ClusterName, vals.ClusterID, fakeApiService)
	assert.NoError(t, err)

	lbName := cloudprovider.GetLoadBalancerName(fakeApiService)
	hcName := MakeNodesHealthCheckName(vals.ClusterID)

	// Check that Firewalls are deleted for the LoadBalancer and the HealthCheck
	fwNames := []string{
		MakeFirewallName(lbName),
		MakeHealthCheckFirewallName(vals.ClusterID, hcName, true),
	}

	for _, fwName := range fwNames {
		firewall, err := gce.GetFirewall(fwName)
		require.Error(t, err)
		assert.Nil(t, firewall)
	}

	// Check that TargetPool is deleted
	pool, err := gce.GetTargetPool(lbName, gce.region)
	require.Error(t, err)
	assert.Nil(t, pool)

	// Check that HealthCheck is deleted
	healthcheck, err := gce.GetHttpHealthCheck(hcName)
	require.Error(t, err)
	assert.Nil(t, healthcheck)

	// Check forwarding rule is deleted
	fwdRule, err := gce.GetRegionForwardingRule(lbName, gce.region)
	require.Error(t, err)
	assert.Nil(t, fwdRule)
}

func TestLoadBalancerWrongTierResourceDeletion(t *testing.T) {
	vals := DefaultTestClusterValues()
	gce, err := fakeGCECloud(vals)
	require.NoError(t, err)

	// Enable the cloud.NetworkTiers feature
	gce.AlphaFeatureGate.features[AlphaFeatureNetworkTiers] = true
	fakeApiService.Annotations = map[string]string{NetworkTierAnnotationKey: "Premium"}

	// cloud.NetworkTier defaults to Premium
	desiredTier, err := gce.getServiceNetworkTier(fakeApiService)
	require.NoError(t, err)
	assert.Equal(t, cloud.NetworkTierPremium, desiredTier)

	lbName := cloudprovider.GetLoadBalancerName(fakeApiService)
	serviceName := types.NamespacedName{Namespace: fakeApiService.Namespace, Name: fakeApiService.Name}

	// create ForwardingRule and Address with the wrong tier
	err = createForwardingRule(
		gce,
		lbName,
		serviceName.String(),
		gce.region,
		"",
		gce.targetPoolURL(lbName),
		fakeApiService.Spec.Ports,
		cloud.NetworkTierStandard,
	)
	require.NoError(t, err)

	addressObj := &computealpha.Address{
		Name:        lbName,
		Description: serviceName.String(),
		NetworkTier: cloud.NetworkTierStandard.ToGCEValue(),
	}

	err = gce.ReserveAlphaRegionAddress(addressObj, gce.region)
	require.NoError(t, err)

	_, err = createExternalLoadBalancer(gce, []string{"test-node-1"}, vals.ClusterName, vals.ClusterID, vals.ZoneName)
	require.NoError(t, err)

	// Expect forwarding rule tier to not be Standard
	tier, err := gce.getNetworkTierFromForwardingRule(lbName, gce.region)
	assert.NoError(t, err)
	assert.Equal(t, cloud.NetworkTierDefault.ToGCEValue(), tier)

	// Expect address to be deleted
	_, err = gce.GetRegionAddress(lbName, gce.region)
	assert.True(t, isNotFound(err))
}
