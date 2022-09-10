//go:build !providerless
// +build !providerless

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

// This file contains shared functions and variables to set up for tests for
// ExternalLoadBalancer and InternalLoadBalancers. It currently cannot live in a
// separate package from GCE because then it would cause a circular import.

package gce

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/api/compute/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/record"
	servicehelpers "k8s.io/cloud-provider/service/helpers"
)

// TODO(yankaiz): Create shared error types for both test/non-test codes.
const (
	eventReasonManualChange = "LoadBalancerManualChange"
	errPrefixGetTargetPool  = "error getting load balancer's target pool:"
	wrongTier               = "SupremeLuxury"
	errStrUnsupportedTier   = "unsupported network tier: \"" + wrongTier + "\""
	fakeSvcName             = "fakesvc"
)

func fakeLoadbalancerService(lbType string) *v1.Service {
	return fakeLoadbalancerServiceHelper(lbType, ServiceAnnotationLoadBalancerType)
}

func fakeLoadBalancerServiceDeprecatedAnnotation(lbType string) *v1.Service {
	return fakeLoadbalancerServiceHelper(lbType, deprecatedServiceAnnotationLoadBalancerType)
}

func fakeLoadbalancerServiceHelper(lbType string, annotationKey string) *v1.Service {
	return &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:        fakeSvcName,
			Annotations: map[string]string{annotationKey: lbType},
		},
		Spec: v1.ServiceSpec{
			SessionAffinity: v1.ServiceAffinityClientIP,
			Type:            v1.ServiceTypeLoadBalancer,
			Ports:           []v1.ServicePort{{Protocol: v1.ProtocolTCP, Port: int32(123)}},
		},
	}
}

var (
	FirewallChangeMsg = fmt.Sprintf("%s %s %s", v1.EventTypeNormal, eventReasonManualChange, eventMsgFirewallChange)
)

func createAndInsertNodes(gce *Cloud, nodeNames []string, zoneName string) ([]*v1.Node, error) {
	nodes := []*v1.Node{}

	for _, name := range nodeNames {
		// Inserting the same node name twice causes an error - here we check if
		// the instance exists already before insertion.
		// TestUpdateExternalLoadBalancer inserts a new node, and relies on an older
		// node to already have been inserted.
		instance, _ := gce.getInstanceByName(name)

		if instance == nil {
			err := gce.InsertInstance(
				gce.ProjectID(),
				zoneName,
				&compute.Instance{
					Name: name,
					Tags: &compute.Tags{
						Items: []string{name},
					},
					// add Instance.Zone, otherwise InstanceID() won't return a right instanceID.
					Zone: zoneName,
				},
			)
			if err != nil {
				return nodes, err
			}
		}

		nodes = append(
			nodes,
			&v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: name,
					Labels: map[string]string{
						v1.LabelHostname:              name,
						v1.LabelFailureDomainBetaZone: zoneName,
					},
				},
				Status: v1.NodeStatus{
					NodeInfo: v1.NodeSystemInfo{
						KubeProxyVersion: "v1.7.2",
					},
				},
			},
		)

	}

	return nodes, nil
}

func assertExternalLbResources(t *testing.T, gce *Cloud, apiService *v1.Service, vals TestClusterValues, nodeNames []string) {
	lbName := gce.GetLoadBalancerName(context.TODO(), "", apiService)
	hcName := MakeNodesHealthCheckName(vals.ClusterID)

	// Check that Firewalls are created for the LoadBalancer and the HealthCheck
	fwNames := []string{
		MakeFirewallName(lbName), // Firewalls for external LBs are prefixed with k8s-fw-
		MakeHealthCheckFirewallName(vals.ClusterID, hcName, true),
	}

	for _, fwName := range fwNames {
		firewall, err := gce.GetFirewall(fwName)
		if err != nil {
			t.Fatal(err)
		}
		if diff := cmp.Diff(nodeNames, firewall.TargetTags); diff != "" {
			t.Errorf("want node names diff (-want +got): %s", diff)
		}
		if len(firewall.SourceRanges) == 0 {
			t.Error("SourceRanges is empty")
		}
	}

	// Check that TargetPool is Created
	pool, err := gce.GetTargetPool(lbName, gce.region)
	if err != nil {
		t.Fatal(err)
	}
	if lbName != pool.Name {
		t.Errorf("want: %s, got: %s", lbName, pool.Name)
	}
	if len(pool.HealthChecks) == 0 {
		t.Error("HealthChecks is empty")
	}
	if len(pool.Instances) != 1 {
		t.Errorf("want: %d, got: %d", 1, len(pool.Instances))
	}

	// Check that HealthCheck is created
	healthcheck, err := gce.GetHTTPHealthCheck(hcName)
	if err != nil {
		t.Fatal(err)
	}
	if hcName != healthcheck.Name {
		t.Errorf("want: %s, got: %s", hcName, healthcheck.Name)
	}

	// Check that ForwardingRule is created
	fwdRule, err := gce.GetRegionForwardingRule(lbName, gce.region)
	if err != nil {
		t.Fatal(err)
	}
	if lbName != fwdRule.Name {
		t.Errorf("want: %s, got: %s", lbName, fwdRule.Name)
	}
	if "TCP" != fwdRule.IPProtocol {
		t.Errorf("want: %s, got: %s", lbName, fwdRule.IPProtocol)
	}
	if "123-123" != fwdRule.PortRange {
		t.Errorf("want: %s, got: %s", lbName, fwdRule.PortRange)
	}
}

func assertExternalLbResourcesDeleted(t *testing.T, gce *Cloud, apiService *v1.Service, vals TestClusterValues, firewallsDeleted bool) {
	lbName := gce.GetLoadBalancerName(context.TODO(), "", apiService)
	hcName := MakeNodesHealthCheckName(vals.ClusterID)

	if firewallsDeleted {
		// Check that Firewalls are deleted for the LoadBalancer and the HealthCheck
		fwNames := []string{
			MakeFirewallName(lbName),
			MakeHealthCheckFirewallName(vals.ClusterID, hcName, true),
		}

		for _, fwName := range fwNames {
			firewall, err := gce.GetFirewall(fwName)
			if err == nil {
				t.Fatal(err)
			}
			if firewall != nil {
				t.Error("firewall is not nil")
			}
		}

		// Check forwarding rule is deleted
		fwdRule, err := gce.GetRegionForwardingRule(lbName, gce.region)
		if err == nil {
			t.Fatal(err)
		}
		if fwdRule != nil {
			t.Error("firewall is not nil")
		}
	}

	// Check that TargetPool is deleted
	pool, err := gce.GetTargetPool(lbName, gce.region)
	if err == nil {
		t.Fatal(err)
	}
	if pool != nil {
		t.Error("pool is not nil")
	}

	// Check that HealthCheck is deleted
	healthcheck, err := gce.GetHTTPHealthCheck(hcName)
	if err == nil {
		t.Fatal(err)
	}
	if healthcheck != nil {
		t.Error("healthcheck is not nil")
	}
}

func assertInternalLbResources(t *testing.T, gce *Cloud, apiService *v1.Service, vals TestClusterValues, nodeNames []string) {
	lbName := gce.GetLoadBalancerName(context.TODO(), "", apiService)

	// Check that Instance Group is created
	igName := makeInstanceGroupName(vals.ClusterID)
	ig, err := gce.GetInstanceGroup(igName, vals.ZoneName)
	if err != nil {
		t.Fatal(err)
	}
	if igName != ig.Name {
		t.Errorf("want: %s, got: %s", igName, ig.Name)
	}

	// Check that Firewalls are created for the LoadBalancer and the HealthCheck
	fwNames := []string{
		MakeFirewallName(lbName),
		makeHealthCheckFirewallName(lbName, vals.ClusterID, true),
	}

	for _, fwName := range fwNames {
		firewall, err := gce.GetFirewall(fwName)
		if err != nil {
			t.Fatal(err)
		}
		if diff := cmp.Diff(nodeNames, firewall.TargetTags); diff != "" {
			t.Errorf("want node names diff (-want +got): %s", diff)
		}
		if len(firewall.SourceRanges) == 0 {
			t.Error("SourceRanges is empty")
		}
	}

	// Check that HealthCheck is created
	sharedHealthCheck := !servicehelpers.RequestsOnlyLocalTraffic(apiService)
	hcName := makeHealthCheckName(lbName, vals.ClusterID, sharedHealthCheck)
	healthcheck, err := gce.GetHealthCheck(hcName)
	if err != nil {
		t.Fatal(err)
	}
	if hcName != healthcheck.Name {
		t.Errorf("want: %s, got: %s", hcName, healthcheck.Name)
	}

	// Check that BackendService exists
	sharedBackend := shareBackendService(apiService)
	backendServiceName := makeBackendServiceName(lbName, vals.ClusterID, sharedBackend, cloud.SchemeInternal, "TCP", apiService.Spec.SessionAffinity)
	backendServiceLink := gce.getBackendServiceLink(backendServiceName)

	bs, err := gce.GetRegionBackendService(backendServiceName, gce.region)
	if err != nil {
		t.Fatal(err)
	}
	if "TCP" != bs.Protocol {
		t.Errorf("want: %s, got: %s", "TCP", bs.Protocol)
	}
	if diff := cmp.Diff([]string{healthcheck.SelfLink}, bs.HealthChecks); diff != "" {
		t.Errorf("want health checks diff (-want +got): %s", diff)
	}

	// Check that ForwardingRule is created
	fwdRule, err := gce.GetRegionForwardingRule(lbName, gce.region)
	if err != nil {
		t.Fatal(err)
	}
	if lbName != fwdRule.Name {
		t.Errorf("want: %s, got: %s", lbName, fwdRule.Name)
	}
	if "TCP" != fwdRule.IPProtocol {
		t.Errorf("want: %s, got: %s", "TCP", fwdRule.IPProtocol)
	}
	if backendServiceLink != fwdRule.BackendService {
		t.Errorf("want: %s, got: %s", backendServiceLink, fwdRule.BackendService)
	}
	// if no Subnetwork specified, defaults to the GCE NetworkURL
	if gce.NetworkURL() != fwdRule.Subnetwork {
		t.Errorf("want: %s, got: %s", gce.NetworkURL(), fwdRule.Subnetwork)
	}

	// Check that the IP address has been released. IP is only reserved until ensure function exits.
	ip, err := gce.GetRegionAddress(lbName, gce.region)
	if err == nil {
		t.Fatal(err)
	}
	if ip != nil {
		t.Error("ip is not nil")
	}
}

func assertInternalLbResourcesDeleted(t *testing.T, gce *Cloud, apiService *v1.Service, vals TestClusterValues, firewallsDeleted bool) {
	lbName := gce.GetLoadBalancerName(context.TODO(), "", apiService)
	sharedHealthCheck := !servicehelpers.RequestsOnlyLocalTraffic(apiService)
	hcName := makeHealthCheckName(lbName, vals.ClusterID, sharedHealthCheck)

	// ensureExternalLoadBalancer and ensureInternalLoadBalancer both create
	// Firewalls with the same name.
	if firewallsDeleted {
		// Check that Firewalls are deleted for the LoadBalancer and the HealthCheck
		fwNames := []string{
			MakeFirewallName(lbName),
			MakeHealthCheckFirewallName(vals.ClusterID, hcName, true),
		}

		for _, fwName := range fwNames {
			firewall, err := gce.GetFirewall(fwName)
			if err == nil {
				t.Fatal(err)
			}
			if firewall != nil {
				t.Error("firewall is not nil")
			}
		}

		// Check forwarding rule is deleted
		fwdRule, err := gce.GetRegionForwardingRule(lbName, gce.region)
		if err == nil {
			t.Fatal(err)
		}
		if fwdRule != nil {
			t.Error("fwdRule is not nil")
		}
	}

	// Check that Instance Group is deleted
	igName := makeInstanceGroupName(vals.ClusterID)
	ig, err := gce.GetInstanceGroup(igName, vals.ZoneName)
	if err == nil {
		t.Fatal(err)
	}
	if ig != nil {
		t.Error("ig is not nil")
	}

	// Check that HealthCheck is deleted
	healthcheck, err := gce.GetHealthCheck(hcName)
	if err == nil {
		t.Fatal(err)
	}
	if healthcheck != nil {
		t.Error("healthcheck is not nil")
	}

	// Check that the IP address has been released
	ip, err := gce.GetRegionAddress(lbName, gce.region)
	if err == nil {
		t.Fatal(err)
	}
	if ip != nil {
		t.Error("ip is not nil")
	}
}

func checkEvent(t *testing.T, recorder *record.FakeRecorder, expected string, shouldMatch bool) bool {
	select {
	case received := <-recorder.Events:
		if strings.HasPrefix(received, expected) != shouldMatch {
			t.Errorf(received)
			if shouldMatch {
				t.Errorf("Should receive message \"%v\" but got \"%v\".", expected, received)
			} else {
				t.Errorf("Unexpected event \"%v\".", received)
			}
		}
		return false
	case <-time.After(2 * time.Second):
		if shouldMatch {
			t.Errorf("Should receive message \"%v\" but got timed out.", expected)
		}
		return true
	}
}
