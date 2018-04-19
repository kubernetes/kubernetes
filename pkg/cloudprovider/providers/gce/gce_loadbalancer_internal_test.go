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

func createInternalLoadBalancer(gce *GCECloud, apiService *v1.Service, existingFwdRule *compute.ForwardingRule, nodeNames []string, clusterName, clusterID, zoneName string) (*v1.LoadBalancerStatus, error) {
	nodes, err := createAndInsertNodes(gce, nodeNames, zoneName)
	if err != nil {
		return nil, err
	}

	return gce.ensureInternalLoadBalancer(
		clusterName,
		clusterID,
		apiService,
		existingFwdRule,
		nodes,
	)
}

func TestEnsureInternalBackendServiceUpdates(t *testing.T) {
	t.Parallel()

	vals := DefaultTestClusterValues()
	nodeNames := []string{"test-node-1"}

	gce, err := fakeGCECloud(vals)
	require.NoError(t, err)

	apiService := fakeLoadbalancerService(string(LBTypeInternal))
	lbName := cloudprovider.GetLoadBalancerName(apiService)
	nodes, err := createAndInsertNodes(gce, nodeNames, vals.ZoneName)
	igName := makeInstanceGroupName(vals.ClusterID)
	igLinks, err := gce.ensureInternalInstanceGroups(igName, nodes)
	require.NoError(t, err)

	sharedBackend := shareBackendService(apiService)
	bsName := makeBackendServiceName(lbName, vals.ClusterID, sharedBackend, cloud.SchemeInternal, "TCP", apiService.Spec.SessionAffinity)
	err = gce.ensureInternalBackendService(bsName, "description", apiService.Spec.SessionAffinity, cloud.SchemeInternal, "TCP", igLinks, "")
	require.NoError(t, err)

	// Update the Internal Backend Service with a new ServiceAffinity
	err = gce.ensureInternalBackendService(bsName, "description", v1.ServiceAffinityNone, cloud.SchemeInternal, "TCP", igLinks, "")
	require.NoError(t, err)

	bs, err := gce.GetRegionBackendService(bsName, gce.region)
	assert.NoError(t, err)
	assert.Equal(t, bs.SessionAffinity, strings.ToUpper(string(v1.ServiceAffinityNone)))
}

func TestEnsureInternalLoadBalancer(t *testing.T) {
	t.Parallel()

	vals := DefaultTestClusterValues()
	nodeNames := []string{"test-node-1"}

	gce, err := fakeGCECloud(vals)
	require.NoError(t, err)

	apiService := fakeLoadbalancerService(string(LBTypeInternal))
	status, err := createInternalLoadBalancer(gce, apiService, nil, nodeNames, vals.ClusterName, vals.ClusterID, vals.ZoneName)
	assert.NoError(t, err)
	assert.NotEmpty(t, status.Ingress)
	assertInternalLbResources(t, gce, apiService, vals, nodeNames)
}

func TestEnsureInternalLoadBalancerWithExistingResources(t *testing.T) {
	t.Parallel()

	vals := DefaultTestClusterValues()
	nodeNames := []string{"test-node-1"}

	gce, err := fakeGCECloud(vals)
	require.NoError(t, err)

	apiService := fakeLoadbalancerService(string(LBTypeInternal))
	// Create the expected resources necessary for an Internal Load Balancer
	nm := types.NamespacedName{Name: apiService.Name, Namespace: apiService.Namespace}
	lbName := cloudprovider.GetLoadBalancerName(apiService)

	sharedHealthCheck := !v1_service.RequestsOnlyLocalTraffic(apiService)
	hcName := makeHealthCheckName(lbName, vals.ClusterID, sharedHealthCheck)
	hcPath, hcPort := GetNodesHealthCheckPath(), GetNodesHealthCheckPort()
	existingHC := newInternalLBHealthCheck(hcName, nm, sharedHealthCheck, hcPath, hcPort)
	err = gce.CreateHealthCheck(existingHC)
	require.NoError(t, err)

	nodes, err := createAndInsertNodes(gce, nodeNames, vals.ZoneName)
	igName := makeInstanceGroupName(vals.ClusterID)
	igLinks, err := gce.ensureInternalInstanceGroups(igName, nodes)
	require.NoError(t, err)

	sharedBackend := shareBackendService(apiService)
	bsDescription := makeBackendServiceDescription(nm, sharedBackend)
	bsName := makeBackendServiceName(lbName, vals.ClusterID, sharedBackend, cloud.SchemeInternal, "TCP", apiService.Spec.SessionAffinity)
	err = gce.ensureInternalBackendService(bsName, bsDescription, apiService.Spec.SessionAffinity, cloud.SchemeInternal, "TCP", igLinks, existingHC.SelfLink)
	require.NoError(t, err)

	_, err = createInternalLoadBalancer(gce, apiService, nil, nodeNames, vals.ClusterName, vals.ClusterID, vals.ZoneName)
	assert.NoError(t, err)
}

func TestEnsureInternalLoadBalancerClearPreviousResources(t *testing.T) {
	t.Parallel()

	vals := DefaultTestClusterValues()
	gce, err := fakeGCECloud(vals)
	require.NoError(t, err)

	apiService := fakeLoadbalancerService(string(LBTypeInternal))
	lbName := cloudprovider.GetLoadBalancerName(apiService)

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

	sharedHealthCheck := !v1_service.RequestsOnlyLocalTraffic(apiService)
	hcName := makeHealthCheckName(lbName, vals.ClusterID, sharedHealthCheck)
	hcPath, hcPort := GetNodesHealthCheckPath(), GetNodesHealthCheckPort()
	nm := types.NamespacedName{Name: apiService.Name, Namespace: apiService.Namespace}

	// Create a healthcheck with an incorrect threshold
	existingHC := newInternalLBHealthCheck(hcName, nm, sharedHealthCheck, hcPath, hcPort)
	existingHC.HealthyThreshold = gceHcHealthyThreshold * 10
	gce.CreateHealthCheck(existingHC)

	// Create a backend Service that's missing Description and Backends
	sharedBackend := shareBackendService(apiService)
	backendServiceName := makeBackendServiceName(lbName, vals.ClusterID, sharedBackend, cloud.SchemeInternal, "TCP", apiService.Spec.SessionAffinity)
	existingBS := &compute.BackendService{
		Name:                lbName,
		Protocol:            "TCP",
		HealthChecks:        []string{existingHC.SelfLink},
		SessionAffinity:     translateAffinityType(apiService.Spec.SessionAffinity),
		LoadBalancingScheme: string(cloud.SchemeInternal),
	}

	gce.CreateRegionBackendService(existingBS, gce.region)
	existingFwdRule.BackendService = existingBS.Name

	_, err = createInternalLoadBalancer(gce, apiService, existingFwdRule, []string{"test-node-1"}, vals.ClusterName, vals.ClusterID, vals.ZoneName)
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
	t.Parallel()

	vals := DefaultTestClusterValues()
	nodeName := "test-node-1"

	gce, err := fakeGCECloud(vals)
	require.NoError(t, err)

	apiService := fakeLoadbalancerService(string(LBTypeInternal))
	_, err = createInternalLoadBalancer(gce, apiService, nil, []string{"test-node-1"}, vals.ClusterName, vals.ClusterID, vals.ZoneName)
	assert.NoError(t, err)

	// BackendService exists prior to updateInternalLoadBalancer call, but has
	// incorrect (missing) attributes.
	// ensureInternalBackendServiceGroups is called and creates the correct
	// BackendService
	lbName := cloudprovider.GetLoadBalancerName(apiService)
	sharedBackend := shareBackendService(apiService)
	backendServiceName := makeBackendServiceName(lbName, vals.ClusterID, sharedBackend, cloud.SchemeInternal, "TCP", apiService.Spec.SessionAffinity)
	existingBS := &compute.BackendService{
		Name:                backendServiceName,
		Protocol:            "TCP",
		SessionAffinity:     translateAffinityType(apiService.Spec.SessionAffinity),
		LoadBalancingScheme: string(cloud.SchemeInternal),
	}

	gce.CreateRegionBackendService(existingBS, gce.region)

	nodes, err := createAndInsertNodes(gce, []string{nodeName}, vals.ZoneName)
	require.NoError(t, err)

	err = gce.updateInternalLoadBalancer(vals.ClusterName, vals.ClusterID, apiService, nodes)
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
	t.Parallel()

	vals := DefaultTestClusterValues()
	gce, err := fakeGCECloud(vals)
	require.NoError(t, err)
	node1Name := []string{"test-node-1"}

	apiService := fakeLoadbalancerService(string(LBTypeInternal))
	nodes, err := createAndInsertNodes(gce, node1Name, vals.ZoneName)
	require.NoError(t, err)

	_, err = gce.ensureInternalLoadBalancer(vals.ClusterName, vals.ClusterID, apiService, nil, nodes)
	assert.NoError(t, err)

	// Replace the node in initial zone; add new node in a new zone.
	node2Name, node3Name := "test-node-2", "test-node-3"
	newNodesZoneA, err := createAndInsertNodes(gce, []string{node2Name}, vals.ZoneName)
	require.NoError(t, err)
	newNodesZoneB, err := createAndInsertNodes(gce, []string{node3Name}, vals.SecondaryZoneName)
	require.NoError(t, err)

	nodes = append(newNodesZoneA, newNodesZoneB...)
	err = gce.updateInternalLoadBalancer(vals.ClusterName, vals.ClusterID, apiService, nodes)
	assert.NoError(t, err)

	lbName := cloudprovider.GetLoadBalancerName(apiService)
	sharedBackend := shareBackendService(apiService)
	backendServiceName := makeBackendServiceName(lbName, vals.ClusterID, sharedBackend, cloud.SchemeInternal, "TCP", apiService.Spec.SessionAffinity)
	bs, err := gce.GetRegionBackendService(backendServiceName, gce.region)
	require.NoError(t, err)
	assert.Equal(t, 2, len(bs.Backends), "Want two backends referencing two instances groups")

	for _, zone := range []string{vals.ZoneName, vals.SecondaryZoneName} {
		var found bool
		for _, be := range bs.Backends {
			if strings.Contains(be.Group, zone) {
				found = true
				break
			}
		}
		assert.True(t, found, "Expected list of backends to have zone %q", zone)
	}

	// Expect initial zone to have test-node-2
	igName := makeInstanceGroupName(vals.ClusterID)
	instances, err := gce.ListInstancesInInstanceGroup(igName, vals.ZoneName, "ALL")
	require.NoError(t, err)
	assert.Equal(t, 1, len(instances))
	assert.Contains(
		t,
		instances[0].Instance,
		fmt.Sprintf("projects/%s/zones/%s/instances/%s", vals.ProjectID, vals.ZoneName, node2Name),
	)

	// Expect initial zone to have test-node-3
	instances, err = gce.ListInstancesInInstanceGroup(igName, vals.SecondaryZoneName, "ALL")
	require.NoError(t, err)
	assert.Equal(t, 1, len(instances))
	assert.Contains(
		t,
		instances[0].Instance,
		fmt.Sprintf("projects/%s/zones/%s/instances/%s", vals.ProjectID, vals.SecondaryZoneName, node3Name),
	)
}

func TestEnsureInternalLoadBalancerDeleted(t *testing.T) {
	t.Parallel()

	vals := DefaultTestClusterValues()
	gce, err := fakeGCECloud(vals)
	require.NoError(t, err)

	apiService := fakeLoadbalancerService(string(LBTypeInternal))
	_, err = createInternalLoadBalancer(gce, apiService, nil, []string{"test-node-1"}, vals.ClusterName, vals.ClusterID, vals.ZoneName)
	assert.NoError(t, err)

	err = gce.ensureInternalLoadBalancerDeleted(vals.ClusterName, vals.ClusterID, apiService)
	assert.NoError(t, err)

	assertInternalLbResourcesDeleted(t, gce, apiService, vals, true)
}

func TestEnsureInternalLoadBalancerDeletedTwiceDoesNotError(t *testing.T) {
	t.Parallel()

	vals := DefaultTestClusterValues()
	gce, err := fakeGCECloud(vals)
	require.NoError(t, err)

	apiService := fakeLoadbalancerService(string(LBTypeInternal))
	_, err = createInternalLoadBalancer(gce, apiService, nil, []string{"test-node-1"}, vals.ClusterName, vals.ClusterID, vals.ZoneName)
	assert.NoError(t, err)

	err = gce.ensureInternalLoadBalancerDeleted(vals.ClusterName, vals.ClusterID, apiService)
	assert.NoError(t, err)

	// Deleting the loadbalancer and resources again should not cause an error.
	err = gce.ensureInternalLoadBalancerDeleted(vals.ClusterName, vals.ClusterID, apiService)
	assert.NoError(t, err)
	assertInternalLbResourcesDeleted(t, gce, apiService, vals, true)
}
