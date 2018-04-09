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
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	compute "google.golang.org/api/compute/v1"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	v1_service "k8s.io/kubernetes/pkg/api/v1/service"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud"
)

func createInternalLoadBalancer(gce *GCECloud, existingFwdRule *compute.ForwardingRule, nodeNames []string, clusterName, clusterID, zoneName string) (*v1.LoadBalancerStatus, error) {
	nodes, err := createAndInsertNodes(gce, nodeNames, zoneName)
	if err != nil {
		return nil, err
	}

	return gce.ensureInternalLoadBalancer(
		clusterName,
		clusterID,
		fakeApiService,
		existingFwdRule,
		nodes,
	)
}

func TestEnsureInternalBackendServiceUpdates(t *testing.T) {
	vals := DefaultTestClusterValues()
	nodeNames := []string{"test-node-1"}

	gce, err := fakeGCECloud(vals)
	require.NoError(t, err)

	lbName := cloudprovider.GetLoadBalancerName(fakeApiService)
	nodes, err := createAndInsertNodes(gce, nodeNames, vals.ZoneName)
	igName := makeInstanceGroupName(vals.ClusterID)
	igLinks, err := gce.ensureInternalInstanceGroups(igName, nodes)
	require.NoError(t, err)

	sharedBackend := shareBackendService(fakeApiService)
	bsName := makeBackendServiceName(lbName, vals.ClusterID, sharedBackend, cloud.SchemeInternal, "TCP", fakeApiService.Spec.SessionAffinity)
	err = gce.ensureInternalBackendService(bsName, "description", fakeApiService.Spec.SessionAffinity, cloud.SchemeInternal, "TCP", igLinks, "")
	require.NoError(t, err)

	// Update the Internal Backend Service with a new ServiceAffinity
	err = gce.ensureInternalBackendService(bsName, "description", v1.ServiceAffinityNone, cloud.SchemeInternal, "TCP", igLinks, "")
	require.NoError(t, err)

	bs, err := gce.GetRegionBackendService(bsName, gce.region)
	assert.NoError(t, err)
	assert.Equal(t, bs.SessionAffinity, strings.ToUpper(string(v1.ServiceAffinityNone)))
}

func TestEnsureInternalBackendServiceGroups(t *testing.T) {
	vals := DefaultTestClusterValues()
	nodeNames := []string{"test-node-1"}

	gce, err := fakeGCECloud(vals)
	require.NoError(t, err)

	lbName := cloudprovider.GetLoadBalancerName(fakeApiService)
	nodes, err := createAndInsertNodes(gce, nodeNames, vals.ZoneName)
	igName := makeInstanceGroupName(vals.ClusterID)
	igLinks, err := gce.ensureInternalInstanceGroups(igName, nodes)
	require.NoError(t, err)

	sharedBackend := shareBackendService(fakeApiService)
	bsName := makeBackendServiceName(lbName, vals.ClusterID, sharedBackend, cloud.SchemeInternal, "TCP", fakeApiService.Spec.SessionAffinity)
	err = gce.ensureInternalBackendService(bsName, "description", fakeApiService.Spec.SessionAffinity, cloud.SchemeInternal, "TCP", igLinks, "")
	require.NoError(t, err)

	// Update the BackendService with new Instances
	newNodeNames := []string{"new-test-node-1", "new-test-node-2"}
	err = gce.ensureInternalBackendServiceGroups(bsName, newNodeNames)
	assert.NoError(t, err)

	bs, err := gce.GetRegionBackendService(bsName, gce.region)
	assert.NoError(t, err)

	// Check that the instances are updated
	newNodes, err := createAndInsertNodes(gce, newNodeNames, vals.ZoneName)
	newIgLinks, err := gce.ensureInternalInstanceGroups(igName, newNodes)
	backends := backendsFromGroupLinks(newIgLinks)
	assert.Equal(t, bs.Backends, backends)
}

func TestEnsureInternalLoadBalancer(t *testing.T) {
	vals := DefaultTestClusterValues()
	nodeName := "test-node-1"

	gce, err := fakeGCECloud(vals)
	require.NoError(t, err)

	status, err := createInternalLoadBalancer(gce, nil, []string{nodeName}, vals.ClusterName, vals.ClusterID, vals.ZoneName)
	assert.NoError(t, err)
	assert.NotEmpty(t, status.Ingress)

	lbName := cloudprovider.GetLoadBalancerName(fakeApiService)

	// Check that Instance Group is created
	igName := makeInstanceGroupName(vals.ClusterID)
	ig, err := gce.GetInstanceGroup(igName, vals.ZoneName)
	assert.NoError(t, err)
	assert.Equal(t, igName, ig.Name)

	// Check that Firewalls are created for the LoadBalancer and the HealthCheck
	fwNames := []string{
		lbName,
		makeHealthCheckFirewallName(lbName, vals.ClusterID, true),
	}

	for _, fwName := range fwNames {
		firewall, err := gce.GetFirewall(fwName)
		require.NoError(t, err)
		assert.Equal(t, []string{nodeName}, firewall.TargetTags)
		assert.NotEmpty(t, firewall.SourceRanges)
	}

	// Check that HealthCheck is created
	sharedHealthCheck := !v1_service.RequestsOnlyLocalTraffic(fakeApiService)
	hcName := makeHealthCheckName(lbName, vals.ClusterID, sharedHealthCheck)
	healthcheck, err := gce.GetHealthCheck(hcName)
	require.NoError(t, err)
	assert.Equal(t, hcName, healthcheck.Name)

	// Check that BackendService exists
	sharedBackend := shareBackendService(fakeApiService)
	backendServiceName := makeBackendServiceName(lbName, vals.ClusterID, sharedBackend, cloud.SchemeInternal, "TCP", fakeApiService.Spec.SessionAffinity)
	backendServiceLink := gce.getBackendServiceLink(backendServiceName)

	bs, err := gce.GetRegionBackendService(backendServiceName, gce.region)
	require.NoError(t, err)
	assert.Equal(t, "TCP", bs.Protocol)
	assert.Equal(
		t,
		[]string{healthcheck.SelfLink},
		bs.HealthChecks,
	)

	// Check that ForwardingRule is created
	fwdRule, err := gce.GetRegionForwardingRule(lbName, gce.region)
	require.NoError(t, err)
	assert.Equal(t, lbName, fwdRule.Name)
	assert.Equal(t, "TCP", fwdRule.IPProtocol)
	assert.Equal(t, backendServiceLink, fwdRule.BackendService)
	// if no Subnetwork specified, defaults to the GCE NetworkURL
	assert.Equal(t, gce.NetworkURL(), fwdRule.Subnetwork)
}

func TestEnsureInternalLoadBalancerWithExistingResources(t *testing.T) {
	vals := DefaultTestClusterValues()
	nodeNames := []string{"test-node-1"}

	gce, err := fakeGCECloud(vals)
	require.NoError(t, err)

	// Create the expected resources necessary for an Internal Load Balancer
	nm := types.NamespacedName{Name: fakeApiService.Name, Namespace: fakeApiService.Namespace}
	lbName := cloudprovider.GetLoadBalancerName(fakeApiService)

	sharedHealthCheck := !v1_service.RequestsOnlyLocalTraffic(fakeApiService)
	hcName := makeHealthCheckName(lbName, vals.ClusterID, sharedHealthCheck)
	hcPath, hcPort := GetNodesHealthCheckPath(), GetNodesHealthCheckPort()
	existingHC := newInternalLBHealthCheck(hcName, nm, sharedHealthCheck, hcPath, hcPort)
	err = gce.CreateHealthCheck(existingHC)
	require.NoError(t, err)

	nodes, err := createAndInsertNodes(gce, nodeNames, vals.ZoneName)
	igName := makeInstanceGroupName(vals.ClusterID)
	igLinks, err := gce.ensureInternalInstanceGroups(igName, nodes)
	require.NoError(t, err)

	sharedBackend := shareBackendService(fakeApiService)
	bsDescription := makeBackendServiceDescription(nm, sharedBackend)
	bsName := makeBackendServiceName(lbName, vals.ClusterID, sharedBackend, cloud.SchemeInternal, "TCP", fakeApiService.Spec.SessionAffinity)
	err = gce.ensureInternalBackendService(bsName, bsDescription, fakeApiService.Spec.SessionAffinity, cloud.SchemeInternal, "TCP", igLinks, existingHC.SelfLink)
	require.NoError(t, err)

	_, err = createInternalLoadBalancer(gce, nil, nodeNames, vals.ClusterName, vals.ClusterID, vals.ZoneName)
	assert.NoError(t, err)
}

func TestEnsureInternalLoadBalancerClearPreviousResources(t *testing.T) {
	vals := DefaultTestClusterValues()
	gce, err := fakeGCECloud(vals)
	require.NoError(t, err)

	lbName := cloudprovider.GetLoadBalancerName(fakeApiService)

	// Create a ForwardingRule that's missing an IP address
	existingFwdRule := &compute.ForwardingRule{
		Name:                lbName,
		IPAddress:           "",
		Ports:               []string{"123"},
		IPProtocol:          "TCP",
		LoadBalancingScheme: string(cloud.SchemeInternal),
	}
	gce.CreateRegionForwardingRule(existingFwdRule, gce.region)

	// Create a Firewall that's missing a Description
	existingFirewall := &compute.Firewall{
		Name:    lbName,
		Network: gce.networkURL,
		Allowed: []*compute.FirewallAllowed{
			{
				IPProtocol: "tcp",
				Ports:      []string{"123"},
			},
		},
	}
	gce.CreateFirewall(existingFirewall)

	sharedHealthCheck := !v1_service.RequestsOnlyLocalTraffic(fakeApiService)
	hcName := makeHealthCheckName(lbName, vals.ClusterID, sharedHealthCheck)
	hcPath, hcPort := GetNodesHealthCheckPath(), GetNodesHealthCheckPort()
	nm := types.NamespacedName{Name: fakeApiService.Name, Namespace: fakeApiService.Namespace}

	// Create a healthcheck with an incorrect threshold
	existingHC := newInternalLBHealthCheck(hcName, nm, sharedHealthCheck, hcPath, hcPort)
	existingHC.HealthyThreshold = gceHcHealthyThreshold * 10
	gce.CreateHealthCheck(existingHC)

	// Create a backend Service that's missing Description and Backends
	sharedBackend := shareBackendService(fakeApiService)
	backendServiceName := makeBackendServiceName(lbName, vals.ClusterID, sharedBackend, cloud.SchemeInternal, "TCP", fakeApiService.Spec.SessionAffinity)
	existingBS := &compute.BackendService{
		Name:                lbName,
		Protocol:            "TCP",
		HealthChecks:        []string{existingHC.SelfLink},
		SessionAffinity:     translateAffinityType(fakeApiService.Spec.SessionAffinity),
		LoadBalancingScheme: string(cloud.SchemeInternal),
	}

	gce.CreateRegionBackendService(existingBS, gce.region)
	existingFwdRule.BackendService = existingBS.Name

	_, err = createInternalLoadBalancer(gce, existingFwdRule, []string{"test-node-1"}, vals.ClusterName, vals.ClusterID, vals.ZoneName)
	assert.NoError(t, err)

	// Expect new resources with the correct attributes to be created
	rule, _ := gce.GetRegionForwardingRule(lbName, gce.region)
	assert.NotEqual(t, existingFwdRule, rule)

	firewall, err := gce.GetFirewall(lbName)
	require.NoError(t, err)
	assert.NotEqual(t, firewall, existingFirewall)

	healthcheck, err := gce.GetHealthCheck(hcName)
	require.NoError(t, err)
	assert.NotEqual(t, healthcheck, existingHC)

	bs, err := gce.GetRegionBackendService(backendServiceName, gce.region)
	require.NoError(t, err)
	assert.NotEqual(t, bs, existingBS)
}

func TestUpdateInternalLoadBalancerBackendServices(t *testing.T) {
	vals := DefaultTestClusterValues()
	nodeName := "test-node-1"

	gce, err := fakeGCECloud(vals)
	require.NoError(t, err)

	_, err = createInternalLoadBalancer(gce, nil, []string{"test-node-1"}, vals.ClusterName, vals.ClusterID, vals.ZoneName)
	assert.NoError(t, err)

	// BackendService exists prior to updateInternalLoadBalancer call, but has
	// incorrect (missing) attributes.
	// ensureInternalBackendServiceGroups is called and creates the correct
	// BackendService
	lbName := cloudprovider.GetLoadBalancerName(fakeApiService)
	sharedBackend := shareBackendService(fakeApiService)
	backendServiceName := makeBackendServiceName(lbName, vals.ClusterID, sharedBackend, cloud.SchemeInternal, "TCP", fakeApiService.Spec.SessionAffinity)
	existingBS := &compute.BackendService{
		Name:                backendServiceName,
		Protocol:            "TCP",
		SessionAffinity:     translateAffinityType(fakeApiService.Spec.SessionAffinity),
		LoadBalancingScheme: string(cloud.SchemeInternal),
	}

	gce.CreateRegionBackendService(existingBS, gce.region)

	nodes, err := createAndInsertNodes(gce, []string{nodeName}, vals.ZoneName)
	require.NoError(t, err)

	err = gce.updateInternalLoadBalancer(vals.ClusterName, vals.ClusterID, fakeApiService, nodes)
	assert.NoError(t, err)

	bs, err := gce.GetRegionBackendService(backendServiceName, gce.region)
	require.NoError(t, err)

	// Check that the new BackendService has the correct attributes
	url_base := fmt.Sprintf("https://www.googleapis.com/compute/v1/projects/%s", vals.ProjectID)

	assert.NotEqual(t, existingBS, bs)
	assert.Equal(
		t,
		bs.SelfLink,
		fmt.Sprintf("%s/regions/%s/backendServices/%s", url_base, vals.Region, bs.Name),
	)
	assert.Equal(t, bs.Description, `{"kubernetes.io/service-name":"/"}`)
	assert.Equal(
		t,
		bs.HealthChecks,
		[]string{fmt.Sprintf("%s/healthChecks/k8s-%s-node", url_base, vals.ClusterID)},
	)
}

func TestUpdateInternalLoadBalancerNodes(t *testing.T) {
	vals := DefaultTestClusterValues()
	gce, err := fakeGCECloud(vals)
	require.NoError(t, err)

	_, err = createInternalLoadBalancer(gce, nil, []string{"test-node-1"}, vals.ClusterName, vals.ClusterID, vals.ZoneName)
	assert.NoError(t, err)

	// Remove the old Node and insert a new Node.
	newNodeName := "test-node-2"
	newNodes, err := createAndInsertNodes(gce, []string{newNodeName}, vals.ZoneName)
	require.NoError(t, err)

	err = gce.updateInternalLoadBalancer(vals.ClusterName, vals.ClusterID, fakeApiService, newNodes)
	assert.NoError(t, err)

	// Expect node 1 to be deleted and node 2 to still exist
	igName := makeInstanceGroupName(vals.ClusterID)
	instances, err := gce.ListInstancesInInstanceGroup(igName, vals.ZoneName, "ALL")
	require.NoError(t, err)

	assert.Equal(t, 1, len(instances))
	assert.Contains(
		t,
		instances[0].Instance,
		fmt.Sprintf("projects/%s/zones/%s/instances/%s", vals.ProjectID, vals.ZoneName, newNodeName),
	)
}

func TestEnsureInternalLoadBalancerDeleted(t *testing.T) {
	vals := DefaultTestClusterValues()
	gce, err := fakeGCECloud(vals)
	require.NoError(t, err)

	_, err = createInternalLoadBalancer(gce, nil, []string{"test-node-1"}, vals.ClusterName, vals.ClusterID, vals.ZoneName)
	assert.NoError(t, err)

	err = gce.ensureInternalLoadBalancerDeleted(vals.ClusterName, vals.ClusterID, fakeApiService)
	assert.NoError(t, err)

	lbName := cloudprovider.GetLoadBalancerName(fakeApiService)
	sharedHealthCheck := !v1_service.RequestsOnlyLocalTraffic(fakeApiService)
	hcName := makeHealthCheckName(lbName, vals.ClusterID, sharedHealthCheck)

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

	// Check that Instance Group is deleted
	igName := makeInstanceGroupName(vals.ClusterID)
	ig, err := gce.GetInstanceGroup(igName, vals.ZoneName)
	assert.Error(t, err)
	assert.Nil(t, ig)

	// Check that HealthCheck is deleted
	healthcheck, err := gce.GetHealthCheck(hcName)
	require.Error(t, err)
	assert.Nil(t, healthcheck)

	// Check forwarding rule is deleted
	fwdRule, err := gce.GetRegionForwardingRule(lbName, gce.region)
	require.Error(t, err)
	assert.Nil(t, fwdRule)
}

func TestEnsureInternalLoadBalancerDeletedTwiceDoesNotError(t *testing.T) {
	vals := DefaultTestClusterValues()
	gce, err := fakeGCECloud(vals)
	require.NoError(t, err)

	_, err = createInternalLoadBalancer(gce, nil, []string{"test-node-1"}, vals.ClusterName, vals.ClusterID, vals.ZoneName)
	assert.NoError(t, err)

	err = gce.ensureInternalLoadBalancerDeleted(vals.ClusterName, vals.ClusterID, fakeApiService)
	assert.NoError(t, err)

	// Deleting the loadbalancer and resources again should not cause an error.
	err = gce.ensureInternalLoadBalancerDeleted(vals.ClusterName, vals.ClusterID, fakeApiService)
	assert.NoError(t, err)
}
