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

func TestEnsureInternalLoadBalancer(t *testing.T) {
	projectID := "test-project"
	region := "us-central1"
	zoneName := "us-central1-b"
	clusterName := "Test Cluster"
	clusterID := "test-cluster-id"
	nodeName := "test-node-1"

	gce, err := fakeGCECloud(projectID, region, zoneName)
	require.NoError(t, err)

	status, err := createInternalLoadBalancer(gce, nil, []string{nodeName}, clusterName, clusterID, zoneName)
	assert.NoError(t, err)
	assert.NotEmpty(t, status.Ingress)

	lbName := cloudprovider.GetLoadBalancerName(fakeApiService)

	// Check that Instance Group is created
	igName := makeInstanceGroupName(clusterID)
	ig, err := gce.GetInstanceGroup(igName, zoneName)
	assert.NoError(t, err)
	assert.Equal(t, igName, ig.Name)

	// Check that Firewalls are created for the LoadBalancer and the HealthCheck
	fwNames := []string{
		lbName,
		makeHealthCheckFirewallName(lbName, clusterID, true),
	}

	for _, fwName := range fwNames {
		firewall, err := gce.GetFirewall(fwName)
		require.NoError(t, err)
		assert.Equal(t, []string{nodeName}, firewall.TargetTags)
		assert.NotEmpty(t, firewall.SourceRanges)
	}

	// Check that HealthCheck is created
	sharedHealthCheck := !v1_service.RequestsOnlyLocalTraffic(fakeApiService)
	hcName := makeHealthCheckName(lbName, clusterID, sharedHealthCheck)
	healthcheck, err := gce.GetHealthCheck(hcName)
	require.NoError(t, err)
	assert.Equal(t, hcName, healthcheck.Name)

	// Check that BackendService exists
	sharedBackend := shareBackendService(fakeApiService)
	backendServiceName := makeBackendServiceName(lbName, clusterID, sharedBackend, cloud.SchemeInternal, "TCP", fakeApiService.Spec.SessionAffinity)
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
	fwdRule, err := gce.GetRegionForwardingRule(lbName, region)
	require.NoError(t, err)
	assert.Equal(t, lbName, fwdRule.Name)
	assert.Equal(t, "TCP", fwdRule.IPProtocol)
	assert.Equal(t, backendServiceLink, fwdRule.BackendService)
}

func TestEnsureInternalLoadBalancerDeleteWrongResources(t *testing.T) {
	projectID := "test-project"
	region := "us-central1"
	zoneName := "us-central1-b"
	clusterName := "Test Cluster"
	clusterID := "test-cluster-id"

	gce, err := fakeGCECloud(projectID, region, zoneName)
	require.NoError(t, err)

	lbName := cloudprovider.GetLoadBalancerName(fakeApiService)

	// Create a ForwardingRule that's missing an IP address and BackendService
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
	hcName := makeHealthCheckName(lbName, clusterID, sharedHealthCheck)
	hcPath, hcPort := GetNodesHealthCheckPath(), GetNodesHealthCheckPort()
	nm := types.NamespacedName{Name: fakeApiService.Name, Namespace: fakeApiService.Namespace}

	// Create a healthcheck with an incorrect threshold
	existingHC := newInternalLBHealthCheck(hcName, nm, sharedHealthCheck, hcPath, hcPort)
	existingHC.HealthyThreshold = gceHcHealthyThreshold * 10
	gce.CreateHealthCheck(existingHC)

	// Create a backend Service that's missing Description and Backends
	sharedBackend := shareBackendService(fakeApiService)
	backendServiceName := makeBackendServiceName(lbName, clusterID, sharedBackend, cloud.SchemeInternal, "TCP", fakeApiService.Spec.SessionAffinity)
	existingBS := &compute.BackendService{
		Name:                backendServiceName,
		Protocol:            "TCP",
		HealthChecks:        []string{existingHC.SelfLink},
		SessionAffinity:     translateAffinityType(fakeApiService.Spec.SessionAffinity),
		LoadBalancingScheme: string(cloud.SchemeInternal),
	}

	gce.CreateRegionBackendService(existingBS, gce.region)

	_, err = createInternalLoadBalancer(gce, existingFwdRule, []string{"test-node-1"}, clusterName, clusterID, zoneName)
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
	projectID := "test-project"
	region := "us-central1"
	zoneName := "us-central1-b"
	clusterName := "Test Cluster Name"
	clusterID := "test-cluster-id"

	nodeName := "test-node-1"

	gce, err := fakeGCECloud(projectID, region, zoneName)
	require.NoError(t, err)

	_, err = createInternalLoadBalancer(gce, nil, []string{"test-node-1"}, clusterName, clusterID, zoneName)
	assert.NoError(t, err)

	// BackendService exists prior to updateInternalLoadBalancer call, but has
	// incorrect (missing) attributes.
	// ensureInternalBackendServiceGroups is called and creates the correct
	// BackendService
	lbName := cloudprovider.GetLoadBalancerName(fakeApiService)
	sharedBackend := shareBackendService(fakeApiService)
	backendServiceName := makeBackendServiceName(lbName, clusterID, sharedBackend, cloud.SchemeInternal, "TCP", fakeApiService.Spec.SessionAffinity)
	existingBS := &compute.BackendService{
		Name:                backendServiceName,
		Protocol:            "TCP",
		SessionAffinity:     translateAffinityType(fakeApiService.Spec.SessionAffinity),
		LoadBalancingScheme: string(cloud.SchemeInternal),
	}

	gce.CreateRegionBackendService(existingBS, gce.region)

	nodes, err := createAndInsertNodes(gce, []string{nodeName}, zoneName)
	require.NoError(t, err)

	err = gce.updateInternalLoadBalancer(clusterName, clusterID, fakeApiService, nodes)
	assert.NoError(t, err)

	bs, err := gce.GetRegionBackendService(backendServiceName, gce.region)
	require.NoError(t, err)

	// Check that the new BackendService has the correct attributes
	assert.NotEqual(t, existingBS, bs)
	assert.NotEmpty(t, bs.SelfLink)
	assert.NotEmpty(t, bs.Description)
	assert.NotEmpty(t, bs.HealthChecks)
}

func TestUpdateInternalLoadBalancerNodes(t *testing.T) {
	projectID := "test-project"
	region := "us-central1"
	zoneName := "us-central1-b"
	clusterName := "Test Cluster Name"
	clusterID := "test-cluster-id"

	gce, err := fakeGCECloud(projectID, region, zoneName)
	require.NoError(t, err)

	_, err = createInternalLoadBalancer(gce, nil, []string{"test-node-1"}, clusterName, clusterID, zoneName)
	assert.NoError(t, err)

	// Remove the old Node and insert a new Node.
	newNodeName := "test-node-2"
	newNodes, err := createAndInsertNodes(gce, []string{newNodeName}, zoneName)
	require.NoError(t, err)

	err = gce.updateInternalLoadBalancer(clusterName, clusterID, fakeApiService, newNodes)
	assert.NoError(t, err)

	// Expect node 1 to be deleted and node 2 to still exist
	igName := makeInstanceGroupName(clusterID)
	instances, err := gce.ListInstancesInInstanceGroup(igName, zoneName, "ALL")
	require.NoError(t, err)

	assert.Equal(t, 1, len(instances))
	assert.Equal(
		t,
		"https://www.googleapis.com/compute/v1/projects/test-project/zones/us-central1-b/instances/test-node-2",
		instances[0].Instance,
	)
}

func TestEnsureInternalLoadBalancerDeleted(t *testing.T) {
	projectID := "test-project"
	region := "us-central1"
	zoneName := "us-central1-b"
	clusterName := "Test Cluster Name"
	clusterID := "test-cluster-id"

	gce, err := fakeGCECloud(projectID, region, zoneName)
	require.NoError(t, err)

	_, err = createInternalLoadBalancer(gce, nil, []string{"test-node-1"}, clusterName, clusterID, zoneName)
	assert.NoError(t, err)

	err = gce.ensureInternalLoadBalancerDeleted(clusterName, clusterID, fakeApiService)
	assert.NoError(t, err)

	lbName := cloudprovider.GetLoadBalancerName(fakeApiService)
	sharedHealthCheck := !v1_service.RequestsOnlyLocalTraffic(fakeApiService)
	hcName := makeHealthCheckName(lbName, clusterID, sharedHealthCheck)

	// Check that Firewalls are deleted for the LoadBalancer and the HealthCheck
	fwNames := []string{
		MakeFirewallName(lbName),
		MakeHealthCheckFirewallName(clusterID, hcName, true),
	}

	for _, fwName := range fwNames {
		firewall, err := gce.GetFirewall(fwName)
		require.Error(t, err)
		assert.Nil(t, firewall)
	}

	// Check that Instance Group is deleted
	igName := makeInstanceGroupName(clusterID)
	ig, err := gce.GetInstanceGroup(igName, zoneName)
	assert.Error(t, err)
	assert.Nil(t, ig)

	// Check that HealthCheck is deleted
	healthcheck, err := gce.GetHealthCheck(hcName)
	require.Error(t, err)
	assert.Nil(t, healthcheck)

	// Check forwarding rule is deleted
	fwdRule, err := gce.GetRegionForwardingRule(lbName, region)
	require.Error(t, err)
	assert.Nil(t, fwdRule)
}

func TestEnsureInternalLoadBalancerDeletedTwiceDoesNotError(t *testing.T) {
	projectID := "test-project"
	region := "us-central1"
	zoneName := "us-central1-b"
	clusterName := "Test Cluster Name"
	clusterID := "test-cluster-id"

	gce, err := fakeGCECloud(projectID, region, zoneName)
	require.NoError(t, err)

	_, err = createInternalLoadBalancer(gce, nil, []string{"test-node-1"}, clusterName, clusterID, zoneName)
	assert.NoError(t, err)

	err = gce.ensureInternalLoadBalancerDeleted(clusterName, clusterID, fakeApiService)
	assert.NoError(t, err)

	// Deleting the loadbalancer and resources again should not cause an error.
	err = gce.ensureInternalLoadBalancerDeleted(clusterName, clusterID, fakeApiService)
	assert.NoError(t, err)
}
