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
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	computealpha "google.golang.org/api/compute/v0.alpha"
	compute "google.golang.org/api/compute/v1"

	ga "google.golang.org/api/compute/v1"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/meta"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/mock"
	netsets "k8s.io/kubernetes/pkg/util/net/sets"
)

func TestEnsureStaticIP(t *testing.T) {
	t.Parallel()

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
	t.Parallel()

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
	t.Parallel()

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
	t.Parallel()

	// Common variables among the tests.
	ports := []v1.ServicePort{{Name: "foo", Protocol: v1.ProtocolTCP, Port: int32(123)}}
	target := "test-target-pool"
	vals := DefaultTestClusterValues()
	serviceName := "foo-svc"

	baseLinkURL := "https://www.googleapis.com/compute/%v/projects/%v/regions/%v/forwardingRules/%v"

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
				SelfLink:    fmt.Sprintf(baseLinkURL, "v1", vals.ProjectID, vals.Region, "lb-1"),
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
				SelfLink:    fmt.Sprintf(baseLinkURL, "alpha", vals.ProjectID, vals.Region, "lb-2"),
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
	t.Parallel()

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

func createExternalLoadBalancer(gce *Cloud, svc *v1.Service, nodeNames []string, clusterName, clusterID, zoneName string) (*v1.LoadBalancerStatus, error) {
	nodes, err := createAndInsertNodes(gce, nodeNames, zoneName)
	if err != nil {
		return nil, err
	}

	return gce.ensureExternalLoadBalancer(
		clusterName,
		clusterID,
		svc,
		nil,
		nodes,
	)
}

func TestEnsureExternalLoadBalancer(t *testing.T) {
	t.Parallel()

	vals := DefaultTestClusterValues()
	nodeNames := []string{"test-node-1"}

	gce, err := fakeGCECloud(vals)
	require.NoError(t, err)

	svc := fakeLoadbalancerService("")
	status, err := createExternalLoadBalancer(gce, svc, nodeNames, vals.ClusterName, vals.ClusterID, vals.ZoneName)
	assert.NoError(t, err)
	assert.NotEmpty(t, status.Ingress)

	assertExternalLbResources(t, gce, svc, vals, nodeNames)
}

func TestUpdateExternalLoadBalancer(t *testing.T) {
	t.Parallel()

	vals := DefaultTestClusterValues()
	nodeName := "test-node-1"

	gce, err := fakeGCECloud((DefaultTestClusterValues()))
	require.NoError(t, err)

	svc := fakeLoadbalancerService("")
	_, err = createExternalLoadBalancer(gce, svc, []string{nodeName}, vals.ClusterName, vals.ClusterID, vals.ZoneName)
	assert.NoError(t, err)

	newNodeName := "test-node-2"
	newNodes, err := createAndInsertNodes(gce, []string{nodeName, newNodeName}, vals.ZoneName)
	assert.NoError(t, err)

	// Add the new node, then check that it is properly added to the TargetPool
	err = gce.updateExternalLoadBalancer("", svc, newNodes)
	assert.NoError(t, err)

	lbName := gce.GetLoadBalancerName(context.TODO(), "", svc)

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
	err = gce.updateExternalLoadBalancer(vals.ClusterName, svc, newNodes)
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
	t.Parallel()

	vals := DefaultTestClusterValues()
	gce, err := fakeGCECloud(vals)
	require.NoError(t, err)

	svc := fakeLoadbalancerService("")
	_, err = createExternalLoadBalancer(gce, svc, []string{"test-node-1"}, vals.ClusterName, vals.ClusterID, vals.ZoneName)
	assert.NoError(t, err)

	err = gce.ensureExternalLoadBalancerDeleted(vals.ClusterName, vals.ClusterID, svc)
	assert.NoError(t, err)

	assertExternalLbResourcesDeleted(t, gce, svc, vals, true)
}

func TestLoadBalancerWrongTierResourceDeletion(t *testing.T) {
	t.Parallel()

	vals := DefaultTestClusterValues()
	gce, err := fakeGCECloud(vals)
	require.NoError(t, err)

	// Enable the cloud.NetworkTiers feature
	gce.AlphaFeatureGate.features[AlphaFeatureNetworkTiers] = true
	svc := fakeLoadbalancerService("")
	svc.Annotations = map[string]string{NetworkTierAnnotationKey: "Premium"}

	// cloud.NetworkTier defaults to Premium
	desiredTier, err := gce.getServiceNetworkTier(svc)
	require.NoError(t, err)
	assert.Equal(t, cloud.NetworkTierPremium, desiredTier)

	lbName := gce.GetLoadBalancerName(context.TODO(), "", svc)
	serviceName := types.NamespacedName{Namespace: svc.Namespace, Name: svc.Name}

	// create ForwardingRule and Address with the wrong tier
	err = createForwardingRule(
		gce,
		lbName,
		serviceName.String(),
		gce.region,
		"",
		gce.targetPoolURL(lbName),
		svc.Spec.Ports,
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

	_, err = createExternalLoadBalancer(gce, svc, []string{"test-node-1"}, vals.ClusterName, vals.ClusterID, vals.ZoneName)
	require.NoError(t, err)

	// Expect forwarding rule tier to not be Standard
	tier, err := gce.getNetworkTierFromForwardingRule(lbName, gce.region)
	assert.NoError(t, err)
	assert.Equal(t, cloud.NetworkTierDefault.ToGCEValue(), tier)

	// Expect address to be deleted
	_, err = gce.GetRegionAddress(lbName, gce.region)
	assert.True(t, isNotFound(err))
}

func TestEnsureExternalLoadBalancerFailsIfInvalidNetworkTier(t *testing.T) {
	t.Parallel()

	vals := DefaultTestClusterValues()
	gce, err := fakeGCECloud(DefaultTestClusterValues())
	require.NoError(t, err)
	nodeNames := []string{"test-node-1"}

	nodes, err := createAndInsertNodes(gce, nodeNames, vals.ZoneName)
	require.NoError(t, err)

	// Enable the cloud.NetworkTiers feature
	gce.AlphaFeatureGate.features[AlphaFeatureNetworkTiers] = true
	svc := fakeLoadbalancerService("")
	svc.Annotations = map[string]string{NetworkTierAnnotationKey: wrongTier}

	_, err = gce.ensureExternalLoadBalancer(vals.ClusterName, vals.ClusterID, svc, nil, nodes)
	require.Error(t, err)
	assert.EqualError(t, err, errStrUnsupportedTier)
}

func TestEnsureExternalLoadBalancerFailsWithNoNodes(t *testing.T) {
	t.Parallel()

	vals := DefaultTestClusterValues()
	gce, err := fakeGCECloud(DefaultTestClusterValues())
	require.NoError(t, err)

	svc := fakeLoadbalancerService("")
	_, err = gce.ensureExternalLoadBalancer(vals.ClusterName, vals.ClusterID, svc, nil, []*v1.Node{})
	require.Error(t, err)
	assert.EqualError(t, err, errStrLbNoHosts)
}

func TestForwardingRuleNeedsUpdate(t *testing.T) {
	t.Parallel()

	vals := DefaultTestClusterValues()
	gce, err := fakeGCECloud(DefaultTestClusterValues())
	require.NoError(t, err)
	status, err := createExternalLoadBalancer(gce, fakeLoadbalancerService(""), []string{"test-node-1"}, vals.ClusterName, vals.ClusterID, vals.ZoneName)
	require.NotNil(t, status)
	require.NoError(t, err)

	svc := fakeLoadbalancerService("")
	lbName := gce.GetLoadBalancerName(context.TODO(), "", svc)
	ipAddr := status.Ingress[0].IP

	lbIP := svc.Spec.LoadBalancerIP
	wrongPorts := []v1.ServicePort{svc.Spec.Ports[0]}
	wrongPorts[0].Port = wrongPorts[0].Port + 1

	wrongProtocolPorts := []v1.ServicePort{svc.Spec.Ports[0]}
	wrongProtocolPorts[0].Protocol = v1.ProtocolUDP

	for desc, tc := range map[string]struct {
		lbIP         string
		ports        []v1.ServicePort
		exists       bool
		needsUpdate  bool
		expectIPAddr string
		expectError  bool
	}{
		"When the loadBalancerIP does not equal the FwdRule IP address.": {
			lbIP:         "1.2.3.4",
			ports:        svc.Spec.Ports,
			exists:       true,
			needsUpdate:  true,
			expectIPAddr: ipAddr,
			expectError:  false,
		},
		"When loadBalancerPortRange returns an error.": {
			lbIP:         lbIP,
			ports:        []v1.ServicePort{},
			exists:       true,
			needsUpdate:  false,
			expectIPAddr: "",
			expectError:  true,
		},
		"When portRange not equals to the forwardingRule port range.": {
			lbIP:         lbIP,
			ports:        wrongPorts,
			exists:       true,
			needsUpdate:  true,
			expectIPAddr: ipAddr,
			expectError:  false,
		},
		"When the ports protocol does not equal the ForwardingRuel IP Protocol.": {
			lbIP:         lbIP,
			ports:        wrongProtocolPorts,
			exists:       true,
			needsUpdate:  true,
			expectIPAddr: ipAddr,
			expectError:  false,
		},
		"When basic workflow.": {
			lbIP:         lbIP,
			ports:        svc.Spec.Ports,
			exists:       true,
			needsUpdate:  false,
			expectIPAddr: ipAddr,
			expectError:  false,
		},
	} {
		t.Run(desc, func(t *testing.T) {
			exists, needsUpdate, ipAddress, err := gce.forwardingRuleNeedsUpdate(lbName, vals.Region, tc.lbIP, tc.ports)
			assert.Equal(t, tc.exists, exists, "'exists' didn't return as expected "+desc)
			assert.Equal(t, tc.needsUpdate, needsUpdate, "'needsUpdate' didn't return as expected "+desc)
			assert.Equal(t, tc.expectIPAddr, ipAddress, "'ipAddress' didn't return as expected "+desc)
			if tc.expectError {
				assert.Error(t, err, "Should returns an error "+desc)
			} else {
				assert.NoError(t, err, "Should not returns an error "+desc)
			}
		})
	}
}

func TestTargetPoolNeedsRecreation(t *testing.T) {
	t.Parallel()

	vals := DefaultTestClusterValues()
	gce, err := fakeGCECloud(DefaultTestClusterValues())
	require.NoError(t, err)

	svc := fakeLoadbalancerService("")
	serviceName := svc.ObjectMeta.Name
	lbName := gce.GetLoadBalancerName(context.TODO(), "", svc)
	nodes, err := createAndInsertNodes(gce, []string{"test-node-1"}, vals.ZoneName)
	require.NoError(t, err)
	hostNames := nodeNames(nodes)
	hosts, err := gce.getInstancesByNames(hostNames)

	var instances []string
	for _, host := range hosts {
		instances = append(instances, host.makeComparableHostPath())
	}
	pool := &compute.TargetPool{
		Name:            lbName,
		Description:     fmt.Sprintf(`{"kubernetes.io/service-name":"%s"}`, serviceName),
		Instances:       instances,
		SessionAffinity: translateAffinityType(v1.ServiceAffinityNone),
	}
	err = gce.CreateTargetPool(pool, vals.Region)
	require.NoError(t, err)

	c := gce.c.(*cloud.MockGCE)
	c.MockTargetPools.GetHook = mock.GetTargetPoolInternalErrHook
	exists, needsRecreation, err := gce.targetPoolNeedsRecreation(lbName, vals.Region, v1.ServiceAffinityNone)
	assert.True(t, exists)
	assert.False(t, needsRecreation)
	require.NotNil(t, err)
	assert.True(t, strings.HasPrefix(err.Error(), errPrefixGetTargetPool))
	c.MockTargetPools.GetHook = nil

	exists, needsRecreation, err = gce.targetPoolNeedsRecreation(lbName, vals.Region, v1.ServiceAffinityClientIP)
	assert.True(t, exists)
	assert.True(t, needsRecreation)
	assert.NoError(t, err)

	exists, needsRecreation, err = gce.targetPoolNeedsRecreation(lbName, vals.Region, v1.ServiceAffinityNone)
	assert.True(t, exists)
	assert.False(t, needsRecreation)
	assert.NoError(t, err)
}

func TestFirewallNeedsUpdate(t *testing.T) {
	t.Parallel()

	vals := DefaultTestClusterValues()
	gce, err := fakeGCECloud(DefaultTestClusterValues())
	require.NoError(t, err)
	svc := fakeLoadbalancerService("")
	status, err := createExternalLoadBalancer(gce, svc, []string{"test-node-1"}, vals.ClusterName, vals.ClusterID, vals.ZoneName)
	require.NotNil(t, status)
	require.NoError(t, err)
	svcName := "/" + svc.ObjectMeta.Name
	region := vals.Region

	ipAddr := status.Ingress[0].IP
	lbName := gce.GetLoadBalancerName(context.TODO(), "", svc)

	ipnet, err := netsets.ParseIPNets("0.0.0.0/0")
	require.NoError(t, err)

	wrongIpnet, err := netsets.ParseIPNets("1.0.0.0/10")
	require.NoError(t, err)

	fw, err := gce.GetFirewall(MakeFirewallName(lbName))
	require.NoError(t, err)

	for desc, tc := range map[string]struct {
		lbName       string
		ipAddr       string
		ports        []v1.ServicePort
		ipnet        netsets.IPNet
		fwIPProtocol string
		getHook      func(context.Context, *meta.Key, *cloud.MockFirewalls) (bool, *ga.Firewall, error)
		sourceRange  string
		exists       bool
		needsUpdate  bool
		hasErr       bool
	}{
		"When response is a Non-400 HTTP error.": {
			lbName:       lbName,
			ipAddr:       ipAddr,
			ports:        svc.Spec.Ports,
			ipnet:        ipnet,
			fwIPProtocol: "tcp",
			getHook:      mock.GetFirewallsUnauthorizedErrHook,
			sourceRange:  fw.SourceRanges[0],
			exists:       false,
			needsUpdate:  false,
			hasErr:       true,
		},
		"When given a wrong description.": {
			lbName:       lbName,
			ipAddr:       "",
			ports:        svc.Spec.Ports,
			ipnet:        ipnet,
			fwIPProtocol: "tcp",
			getHook:      nil,
			sourceRange:  fw.SourceRanges[0],
			exists:       true,
			needsUpdate:  true,
			hasErr:       false,
		},
		"When IPProtocol doesn't match.": {
			lbName:       lbName,
			ipAddr:       ipAddr,
			ports:        svc.Spec.Ports,
			ipnet:        ipnet,
			fwIPProtocol: "usps",
			getHook:      nil,
			sourceRange:  fw.SourceRanges[0],
			exists:       true,
			needsUpdate:  true,
			hasErr:       false,
		},
		"When the ports don't match.": {
			lbName:       lbName,
			ipAddr:       ipAddr,
			ports:        []v1.ServicePort{{Protocol: v1.ProtocolTCP, Port: int32(666)}},
			ipnet:        ipnet,
			fwIPProtocol: "tcp",
			getHook:      nil,
			sourceRange:  fw.SourceRanges[0],
			exists:       true,
			needsUpdate:  true,
			hasErr:       false,
		},
		"When parseIPNets returns an error.": {
			lbName:       lbName,
			ipAddr:       ipAddr,
			ports:        svc.Spec.Ports,
			ipnet:        ipnet,
			fwIPProtocol: "tcp",
			getHook:      nil,
			sourceRange:  "badSourceRange",
			exists:       true,
			needsUpdate:  true,
			hasErr:       false,
		},
		"When the source ranges are not equal.": {
			lbName:       lbName,
			ipAddr:       ipAddr,
			ports:        svc.Spec.Ports,
			ipnet:        wrongIpnet,
			fwIPProtocol: "tcp",
			getHook:      nil,
			sourceRange:  fw.SourceRanges[0],
			exists:       true,
			needsUpdate:  true,
			hasErr:       false,
		},
		"When basic flow without exceptions.": {
			lbName:       lbName,
			ipAddr:       ipAddr,
			ports:        svc.Spec.Ports,
			ipnet:        ipnet,
			fwIPProtocol: "tcp",
			getHook:      nil,
			sourceRange:  fw.SourceRanges[0],
			exists:       true,
			needsUpdate:  false,
			hasErr:       false,
		},
	} {
		t.Run(desc, func(t *testing.T) {
			fw, err = gce.GetFirewall(MakeFirewallName(tc.lbName))
			fw.Allowed[0].IPProtocol = tc.fwIPProtocol
			fw, err = gce.GetFirewall(MakeFirewallName(tc.lbName))
			require.Equal(t, fw.Allowed[0].IPProtocol, tc.fwIPProtocol)

			trueSourceRange := fw.SourceRanges[0]
			fw.SourceRanges[0] = tc.sourceRange
			fw, err = gce.GetFirewall(MakeFirewallName(lbName))
			require.Equal(t, fw.SourceRanges[0], tc.sourceRange)

			c := gce.c.(*cloud.MockGCE)
			c.MockFirewalls.GetHook = tc.getHook

			exists, needsUpdate, err := gce.firewallNeedsUpdate(
				tc.lbName,
				svcName,
				region,
				tc.ipAddr,
				tc.ports,
				tc.ipnet)

			assert.Equal(t, tc.exists, exists, "'exists' didn't return as expected "+desc)
			assert.Equal(t, tc.needsUpdate, needsUpdate, "'needsUpdate' didn't return as expected "+desc)
			if tc.hasErr {
				assert.Error(t, err, "Should returns an error "+desc)
			} else {
				assert.NoError(t, err, "Should not returns an error "+desc)
			}

			c.MockFirewalls.GetHook = nil

			fw.Allowed[0].IPProtocol = "tcp"
			fw.SourceRanges[0] = trueSourceRange
			fw, err = gce.GetFirewall(MakeFirewallName(tc.lbName))
			require.Equal(t, fw.Allowed[0].IPProtocol, "tcp")
			require.Equal(t, fw.SourceRanges[0], trueSourceRange)

		})
	}
}

func TestDeleteWrongNetworkTieredResourcesSucceedsWhenNotFound(t *testing.T) {
	t.Parallel()

	gce, err := fakeGCECloud(DefaultTestClusterValues())
	require.NoError(t, err)

	gce.AlphaFeatureGate.features[AlphaFeatureNetworkTiers] = true
	assert.Nil(t, gce.deleteWrongNetworkTieredResources("Wrong_LB_Name", "", cloud.NetworkTier("")))
}

func TestEnsureTargetPoolAndHealthCheck(t *testing.T) {
	t.Parallel()

	vals := DefaultTestClusterValues()
	gce, err := fakeGCECloud(DefaultTestClusterValues())
	require.NoError(t, err)

	nodes, err := createAndInsertNodes(gce, []string{"test-node-1"}, vals.ZoneName)
	require.NoError(t, err)
	svc := fakeLoadbalancerService("")
	status, err := gce.ensureExternalLoadBalancer(
		vals.ClusterName,
		vals.ClusterID,
		svc,
		nil,
		nodes,
	)
	require.NotNil(t, status)
	require.NoError(t, err)

	hostNames := nodeNames(nodes)
	hosts, err := gce.getInstancesByNames(hostNames)
	clusterID := vals.ClusterID

	ipAddr := status.Ingress[0].IP
	lbName := gce.GetLoadBalancerName(context.TODO(), "", svc)
	region := vals.Region

	hcToCreate := makeHTTPHealthCheck(MakeNodesHealthCheckName(clusterID), GetNodesHealthCheckPath(), GetNodesHealthCheckPort())
	hcToDelete := makeHTTPHealthCheck(MakeNodesHealthCheckName(clusterID), GetNodesHealthCheckPath(), GetNodesHealthCheckPort())

	// Apply a tag on the target pool. By verifying the change of the tag, target pool update can be ensured.
	tag := "A Tag"
	pool, err := gce.GetTargetPool(lbName, region)
	pool.CreationTimestamp = tag
	pool, err = gce.GetTargetPool(lbName, region)
	require.Equal(t, tag, pool.CreationTimestamp)
	err = gce.ensureTargetPoolAndHealthCheck(true, true, svc, lbName, clusterID, ipAddr, hosts, hcToCreate, hcToDelete)
	assert.NoError(t, err)
	pool, err = gce.GetTargetPool(lbName, region)
	assert.NotEqual(t, pool.CreationTimestamp, tag)

	pool, err = gce.GetTargetPool(lbName, region)
	assert.Equal(t, 1, len(pool.Instances))
	var manyNodeName [maxTargetPoolCreateInstances + 1]string
	for i := 0; i < maxTargetPoolCreateInstances+1; i++ {
		manyNodeName[i] = fmt.Sprintf("testnode_%d", i)
	}
	manyNodes, err := createAndInsertNodes(gce, manyNodeName[:], vals.ZoneName)
	require.NoError(t, err)
	manyHostNames := nodeNames(manyNodes)
	manyHosts, err := gce.getInstancesByNames(manyHostNames)
	err = gce.ensureTargetPoolAndHealthCheck(true, true, svc, lbName, clusterID, ipAddr, manyHosts, hcToCreate, hcToDelete)
	assert.NoError(t, err)

	pool, err = gce.GetTargetPool(lbName, region)
	assert.Equal(t, maxTargetPoolCreateInstances+1, len(pool.Instances))

	err = gce.ensureTargetPoolAndHealthCheck(true, false, svc, lbName, clusterID, ipAddr, hosts, hcToCreate, hcToDelete)
	assert.NoError(t, err)
	pool, err = gce.GetTargetPool(lbName, region)
	assert.Equal(t, 1, len(pool.Instances))
}

func TestCreateAndUpdateFirewallSucceedsOnXPN(t *testing.T) {
	t.Parallel()

	gce, err := fakeGCECloud(DefaultTestClusterValues())
	require.NoError(t, err)
	vals := DefaultTestClusterValues()

	c := gce.c.(*cloud.MockGCE)
	c.MockFirewalls.InsertHook = mock.InsertFirewallsUnauthorizedErrHook
	c.MockFirewalls.UpdateHook = mock.UpdateFirewallsUnauthorizedErrHook
	gce.onXPN = true
	require.True(t, gce.OnXPN())

	recorder := record.NewFakeRecorder(1024)
	gce.eventRecorder = recorder

	svc := fakeLoadbalancerService("")
	nodes, err := createAndInsertNodes(gce, []string{"test-node-1"}, vals.ZoneName)
	require.NoError(t, err)
	hostNames := nodeNames(nodes)
	hosts, err := gce.getInstancesByNames(hostNames)
	require.NoError(t, err)
	ipnet, err := netsets.ParseIPNets("10.0.0.0/20")
	require.NoError(t, err)
	gce.createFirewall(
		svc,
		gce.GetLoadBalancerName(context.TODO(), "", svc),
		gce.region,
		"A sad little firewall",
		ipnet,
		svc.Spec.Ports,
		hosts)
	require.Nil(t, err)

	msg := fmt.Sprintf("%s %s %s", v1.EventTypeNormal, eventReasonManualChange, eventMsgFirewallChange)
	checkEvent(t, recorder, msg, true)

	gce.updateFirewall(
		svc,
		gce.GetLoadBalancerName(context.TODO(), "", svc),
		gce.region,
		"A sad little firewall",
		ipnet,
		svc.Spec.Ports,
		hosts)
	require.Nil(t, err)

	msg = fmt.Sprintf("%s %s %s", v1.EventTypeNormal, eventReasonManualChange, eventMsgFirewallChange)
	checkEvent(t, recorder, msg, true)
}

func TestEnsureExternalLoadBalancerDeletedSucceedsOnXPN(t *testing.T) {
	t.Parallel()

	vals := DefaultTestClusterValues()
	gce, err := fakeGCECloud(DefaultTestClusterValues())
	require.NoError(t, err)

	_, err = createExternalLoadBalancer(gce, fakeLoadbalancerService(""), []string{"test-node-1"}, vals.ClusterName, vals.ClusterID, vals.ZoneName)
	require.NoError(t, err)

	c := gce.c.(*cloud.MockGCE)
	c.MockFirewalls.DeleteHook = mock.DeleteFirewallsUnauthorizedErrHook
	gce.onXPN = true
	require.True(t, gce.OnXPN())

	recorder := record.NewFakeRecorder(1024)
	gce.eventRecorder = recorder

	svc := fakeLoadbalancerService("")
	err = gce.ensureExternalLoadBalancerDeleted(vals.ClusterName, vals.ClusterID, svc)
	require.NoError(t, err)

	msg := fmt.Sprintf("%s %s %s", v1.EventTypeNormal, eventReasonManualChange, eventMsgFirewallChange)
	checkEvent(t, recorder, msg, true)
}

type EnsureELBParams struct {
	clusterName     string
	clusterID       string
	service         *v1.Service
	existingFwdRule *compute.ForwardingRule
	nodes           []*v1.Node
}

// newEnsureELBParams is the constructor of EnsureELBParams.
func newEnsureELBParams(nodes []*v1.Node, svc *v1.Service) *EnsureELBParams {
	vals := DefaultTestClusterValues()
	return &EnsureELBParams{
		vals.ClusterName,
		vals.ClusterID,
		svc,
		nil,
		nodes,
	}
}

// TestEnsureExternalLoadBalancerErrors tests the function
// ensureExternalLoadBalancer, making sure the system won't panic when
// exceptions raised by gce.
func TestEnsureExternalLoadBalancerErrors(t *testing.T) {
	t.Parallel()

	vals := DefaultTestClusterValues()
	var params *EnsureELBParams

	for desc, tc := range map[string]struct {
		adjustParams func(*EnsureELBParams)
		injectMock   func(*cloud.MockGCE)
	}{
		"No hosts provided": {
			adjustParams: func(params *EnsureELBParams) {
				params.nodes = []*v1.Node{}
			},
		},
		"Invalid node provided": {
			adjustParams: func(params *EnsureELBParams) {
				params.nodes = []*v1.Node{{}}
			},
		},
		"Get forwarding rules failed": {
			injectMock: func(c *cloud.MockGCE) {
				c.MockForwardingRules.GetHook = mock.GetForwardingRulesInternalErrHook
			},
		},
		"Get addresses failed": {
			injectMock: func(c *cloud.MockGCE) {
				c.MockAddresses.GetHook = mock.GetAddressesInternalErrHook
			},
		},
		"Bad load balancer source range provided": {
			adjustParams: func(params *EnsureELBParams) {
				params.service.Spec.LoadBalancerSourceRanges = []string{"BadSourceRange"}
			},
		},
		"Get firewall failed": {
			injectMock: func(c *cloud.MockGCE) {
				c.MockFirewalls.GetHook = mock.GetFirewallsUnauthorizedErrHook
			},
		},
		"Create firewall failed": {
			injectMock: func(c *cloud.MockGCE) {
				c.MockFirewalls.InsertHook = mock.InsertFirewallsUnauthorizedErrHook
			},
		},
		"Get target pool failed": {
			injectMock: func(c *cloud.MockGCE) {
				c.MockTargetPools.GetHook = mock.GetTargetPoolInternalErrHook
			},
		},
		"Get HTTP health checks failed": {
			injectMock: func(c *cloud.MockGCE) {
				c.MockHttpHealthChecks.GetHook = mock.GetHTTPHealthChecksInternalErrHook
			},
		},
		"Create target pools failed": {
			injectMock: func(c *cloud.MockGCE) {
				c.MockTargetPools.InsertHook = mock.InsertTargetPoolsInternalErrHook
			},
		},
		"Create forwarding rules failed": {
			injectMock: func(c *cloud.MockGCE) {
				c.MockForwardingRules.InsertHook = mock.InsertForwardingRulesInternalErrHook
			},
		},
	} {
		t.Run(desc, func(t *testing.T) {
			gce, err := fakeGCECloud(DefaultTestClusterValues())
			nodes, err := createAndInsertNodes(gce, []string{"test-node-1"}, vals.ZoneName)
			require.NoError(t, err)
			svc := fakeLoadbalancerService("")
			params = newEnsureELBParams(nodes, svc)
			if tc.adjustParams != nil {
				tc.adjustParams(params)
			}
			if tc.injectMock != nil {
				tc.injectMock(gce.c.(*cloud.MockGCE))
			}
			status, err := gce.ensureExternalLoadBalancer(
				params.clusterName,
				params.clusterID,
				params.service,
				params.existingFwdRule,
				params.nodes,
			)
			assert.Error(t, err, "Should return an error when "+desc)
			assert.Nil(t, status, "Should not return a status when "+desc)
		})
	}
}

func TestExternalLoadBalancerEnsureHttpHealthCheck(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		desc      string
		modifier  func(*compute.HttpHealthCheck) *compute.HttpHealthCheck
		wantEqual bool
	}{
		{"should ensure HC", func(_ *compute.HttpHealthCheck) *compute.HttpHealthCheck { return nil }, false},
		{
			"should reconcile HC interval",
			func(hc *compute.HttpHealthCheck) *compute.HttpHealthCheck {
				hc.CheckIntervalSec = gceHcCheckIntervalSeconds - 1
				return hc
			},
			false,
		},
		{
			"should allow HC to be configurable to bigger intervals",
			func(hc *compute.HttpHealthCheck) *compute.HttpHealthCheck {
				hc.CheckIntervalSec = gceHcCheckIntervalSeconds * 10
				return hc
			},
			true,
		},
		{
			"should allow HC to accept bigger intervals while applying default value to small thresholds",
			func(hc *compute.HttpHealthCheck) *compute.HttpHealthCheck {
				hc.CheckIntervalSec = gceHcCheckIntervalSeconds * 10
				hc.UnhealthyThreshold = gceHcUnhealthyThreshold - 1
				return hc
			},
			false,
		},
	} {
		t.Run(tc.desc, func(t *testing.T) {

			gce, err := fakeGCECloud(DefaultTestClusterValues())
			require.NoError(t, err)
			c := gce.c.(*cloud.MockGCE)
			c.MockHttpHealthChecks.UpdateHook = func(ctx context.Context, key *meta.Key, obj *ga.HttpHealthCheck, m *cloud.MockHttpHealthChecks) error {
				m.Objects[*key] = &cloud.MockHttpHealthChecksObj{Obj: obj}
				return nil
			}

			hcName, hcPath, hcPort := "test-hc", "/healthz", int32(12345)
			existingHC := makeHTTPHealthCheck(hcName, hcPath, hcPort)
			existingHC = tc.modifier(existingHC)
			if existingHC != nil {
				if err := gce.CreateHTTPHealthCheck(existingHC); err != nil {
					t.Fatalf("gce.CreateHttpHealthCheck(%#v) = %v; want err = nil", existingHC, err)
				}
			}
			if _, err := gce.ensureHTTPHealthCheck(hcName, hcPath, hcPort); err != nil {
				t.Fatalf("gce.ensureHttpHealthCheck(%q, %q, %v) = _, %d; want err = nil", hcName, hcPath, hcPort, err)
			}
			if hc, err := gce.GetHTTPHealthCheck(hcName); err != nil {
				t.Fatalf("gce.GetHttpHealthCheck(%q) = _, %d; want err = nil", hcName, err)
			} else {
				if tc.wantEqual {
					assert.Equal(t, hc, existingHC)
				} else {
					assert.NotEqual(t, hc, existingHC)
				}
			}
		})
	}

}

func TestMergeHttpHealthChecks(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		desc                   string
		checkIntervalSec       int64
		timeoutSec             int64
		healthyThreshold       int64
		unhealthyThreshold     int64
		wantCheckIntervalSec   int64
		wantTimeoutSec         int64
		wantHealthyThreshold   int64
		wantUnhealthyThreshold int64
	}{
		{"unchanged", gceHcCheckIntervalSeconds, gceHcTimeoutSeconds, gceHcHealthyThreshold, gceHcUnhealthyThreshold, gceHcCheckIntervalSeconds, gceHcTimeoutSeconds, gceHcHealthyThreshold, gceHcUnhealthyThreshold},
		{"interval - too small - should reconcile", gceHcCheckIntervalSeconds - 1, gceHcTimeoutSeconds, gceHcHealthyThreshold, gceHcUnhealthyThreshold, gceHcCheckIntervalSeconds, gceHcTimeoutSeconds, gceHcHealthyThreshold, gceHcUnhealthyThreshold},
		{"timeout - too small - should reconcile", gceHcCheckIntervalSeconds, gceHcTimeoutSeconds - 1, gceHcHealthyThreshold, gceHcUnhealthyThreshold, gceHcCheckIntervalSeconds, gceHcTimeoutSeconds, gceHcHealthyThreshold, gceHcUnhealthyThreshold},
		{"healthy threshold - too small - should reconcile", gceHcCheckIntervalSeconds, gceHcTimeoutSeconds, gceHcHealthyThreshold - 1, gceHcUnhealthyThreshold, gceHcCheckIntervalSeconds, gceHcTimeoutSeconds, gceHcHealthyThreshold, gceHcUnhealthyThreshold},
		{"unhealthy threshold - too small - should reconcile", gceHcCheckIntervalSeconds, gceHcTimeoutSeconds, gceHcHealthyThreshold, gceHcUnhealthyThreshold - 1, gceHcCheckIntervalSeconds, gceHcTimeoutSeconds, gceHcHealthyThreshold, gceHcUnhealthyThreshold},
		{"interval - user configured - should keep", gceHcCheckIntervalSeconds + 1, gceHcTimeoutSeconds, gceHcHealthyThreshold, gceHcUnhealthyThreshold, gceHcCheckIntervalSeconds + 1, gceHcTimeoutSeconds, gceHcHealthyThreshold, gceHcUnhealthyThreshold},
		{"timeout - user configured - should keep", gceHcCheckIntervalSeconds, gceHcTimeoutSeconds + 1, gceHcHealthyThreshold, gceHcUnhealthyThreshold, gceHcCheckIntervalSeconds, gceHcTimeoutSeconds + 1, gceHcHealthyThreshold, gceHcUnhealthyThreshold},
		{"healthy threshold - user configured - should keep", gceHcCheckIntervalSeconds, gceHcTimeoutSeconds, gceHcHealthyThreshold + 1, gceHcUnhealthyThreshold, gceHcCheckIntervalSeconds, gceHcTimeoutSeconds, gceHcHealthyThreshold + 1, gceHcUnhealthyThreshold},
		{"unhealthy threshold - user configured - should keep", gceHcCheckIntervalSeconds, gceHcTimeoutSeconds, gceHcHealthyThreshold, gceHcUnhealthyThreshold + 1, gceHcCheckIntervalSeconds, gceHcTimeoutSeconds, gceHcHealthyThreshold, gceHcUnhealthyThreshold + 1},
	} {
		t.Run(tc.desc, func(t *testing.T) {
			wantHC := makeHTTPHealthCheck("hc", "/", 12345)
			hc := &compute.HttpHealthCheck{
				CheckIntervalSec:   tc.checkIntervalSec,
				TimeoutSec:         tc.timeoutSec,
				HealthyThreshold:   tc.healthyThreshold,
				UnhealthyThreshold: tc.unhealthyThreshold,
			}
			mergeHTTPHealthChecks(hc, wantHC)
			if wantHC.CheckIntervalSec != tc.wantCheckIntervalSec {
				t.Errorf("wantHC.CheckIntervalSec = %d; want %d", wantHC.CheckIntervalSec, tc.checkIntervalSec)
			}
			if wantHC.TimeoutSec != tc.wantTimeoutSec {
				t.Errorf("wantHC.TimeoutSec = %d; want %d", wantHC.TimeoutSec, tc.timeoutSec)
			}
			if wantHC.HealthyThreshold != tc.wantHealthyThreshold {
				t.Errorf("wantHC.HealthyThreshold = %d; want %d", wantHC.HealthyThreshold, tc.healthyThreshold)
			}
			if wantHC.UnhealthyThreshold != tc.wantUnhealthyThreshold {
				t.Errorf("wantHC.UnhealthyThreshold = %d; want %d", wantHC.UnhealthyThreshold, tc.unhealthyThreshold)
			}
		})
	}
}

func TestNeedToUpdateHttpHealthChecks(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		desc        string
		modifier    func(*compute.HttpHealthCheck)
		wantChanged bool
	}{
		{"unchanged", nil, false},
		{"desc does not match", func(hc *compute.HttpHealthCheck) { hc.Description = "bad-desc" }, true},
		{"port does not match", func(hc *compute.HttpHealthCheck) { hc.Port = 54321 }, true},
		{"requestPath does not match", func(hc *compute.HttpHealthCheck) { hc.RequestPath = "/anotherone" }, true},
		{"interval needs update", func(hc *compute.HttpHealthCheck) { hc.CheckIntervalSec = gceHcCheckIntervalSeconds - 1 }, true},
		{"timeout needs update", func(hc *compute.HttpHealthCheck) { hc.TimeoutSec = gceHcTimeoutSeconds - 1 }, true},
		{"healthy threshold needs update", func(hc *compute.HttpHealthCheck) { hc.HealthyThreshold = gceHcHealthyThreshold - 1 }, true},
		{"unhealthy threshold needs update", func(hc *compute.HttpHealthCheck) { hc.UnhealthyThreshold = gceHcUnhealthyThreshold - 1 }, true},
		{"interval does not need update", func(hc *compute.HttpHealthCheck) { hc.CheckIntervalSec = gceHcCheckIntervalSeconds + 1 }, false},
		{"timeout does not need update", func(hc *compute.HttpHealthCheck) { hc.TimeoutSec = gceHcTimeoutSeconds + 1 }, false},
		{"healthy threshold does not need update", func(hc *compute.HttpHealthCheck) { hc.HealthyThreshold = gceHcHealthyThreshold + 1 }, false},
		{"unhealthy threshold does not need update", func(hc *compute.HttpHealthCheck) { hc.UnhealthyThreshold = gceHcUnhealthyThreshold + 1 }, false},
	} {
		t.Run(tc.desc, func(t *testing.T) {
			hc := makeHTTPHealthCheck("hc", "/", 12345)
			wantHC := makeHTTPHealthCheck("hc", "/", 12345)
			if tc.modifier != nil {
				tc.modifier(hc)
			}
			if gotChanged := needToUpdateHTTPHealthChecks(hc, wantHC); gotChanged != tc.wantChanged {
				t.Errorf("needToUpdateHTTPHealthChecks(%#v, %#v) = %t; want changed = %t", hc, wantHC, gotChanged, tc.wantChanged)
			}
		})
	}
}
