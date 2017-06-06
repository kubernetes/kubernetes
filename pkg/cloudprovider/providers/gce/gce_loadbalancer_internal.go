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
	"strconv"
	"strings"

	"github.com/golang/glog"
	compute "google.golang.org/api/compute/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/api/v1"
	v1_service "k8s.io/kubernetes/pkg/api/v1/service"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

const (
	allInstances = "ALL"
)

type lbBalancingMode string

func (gce *GCECloud) ensureInternalLoadBalancer(clusterName, clusterID string, svc *v1.Service, existingFwdRule *compute.ForwardingRule, nodes []*v1.Node) (*v1.LoadBalancerStatus, error) {
	nm := types.NamespacedName{Name: svc.Name, Namespace: svc.Namespace}
	ports, protocol := getPortsAndProtocol(svc.Spec.Ports)
	scheme := schemeInternal
	loadBalancerName := cloudprovider.GetLoadBalancerName(svc)
	shared := !v1_service.RequestsOnlyLocalTraffic(svc)
	backendServiceName := makeBackendServiceName(loadBalancerName, clusterID, shared, scheme, protocol, svc.Spec.SessionAffinity)
	backendServiceLink := gce.getBackendServiceLink(backendServiceName)

	// Ensure instance groups exist and nodes are assigned to groups
	igName := makeInstanceGroupName(clusterID)
	igLinks, err := gce.ensureInternalInstanceGroups(igName, nodes)
	if err != nil {
		return nil, err
	}

	// Lock the sharedResourceLock to prevent any deletions of shared resources while assembling shared resources here
	gce.sharedResourceLock.Lock()
	defer gce.sharedResourceLock.Unlock()

	// Ensure health check for backend service is created
	// By default, use the node health check endpoint
	hcName := makeHealthCheckName(loadBalancerName, clusterID, shared)
	hcPath, hcPort := GetNodesHealthCheckPath(), GetNodesHealthCheckPort()
	if !shared {
		// Service requires a special health check, retrieve the OnlyLocal port & path
		hcPath, hcPort = v1_service.GetServiceHealthCheckPathPort(svc)
	}
	hc, err := gce.ensureInternalHealthCheck(hcName, nm, shared, hcPath, hcPort)
	if err != nil {
		return nil, err
	}

	// Ensure firewall rules if necessary
	if gce.OnXPN() {
		glog.V(2).Infof("ensureInternalLoadBalancer: cluster is on a cross-project network (XPN) network project %v, compute project %v - skipping firewall creation", gce.networkProjectID, gce.projectID)
	} else {
		if err = gce.ensureInternalFirewalls(loadBalancerName, clusterID, nm, svc, strconv.Itoa(int(hcPort)), shared, nodes); err != nil {
			return nil, err
		}
	}

	expectedFwdRule := &compute.ForwardingRule{
		Name:                loadBalancerName,
		Description:         fmt.Sprintf(`{"kubernetes.io/service-name":"%s"}`, nm.String()),
		IPAddress:           svc.Spec.LoadBalancerIP,
		BackendService:      backendServiceLink,
		Ports:               ports,
		IPProtocol:          string(protocol),
		LoadBalancingScheme: string(scheme),
	}

	// Specify subnetwork if network type is manual
	if len(gce.subnetworkURL) > 0 {
		expectedFwdRule.Subnetwork = gce.subnetworkURL
	} else {
		expectedFwdRule.Network = gce.networkURL
	}

	// Delete the previous internal load balancer if necessary
	fwdRuleDeleted, err := gce.clearExistingInternalLB(loadBalancerName, existingFwdRule, expectedFwdRule, backendServiceName)
	if err != nil {
		return nil, err
	}

	bsDescription := makeBackendServiceDescription(nm, shared)
	err = gce.ensureInternalBackendService(backendServiceName, bsDescription, svc.Spec.SessionAffinity, scheme, protocol, igLinks, hc.SelfLink)
	if err != nil {
		return nil, err
	}

	// If we previously deleted the forwarding rule or it never existed, finally create it.
	if fwdRuleDeleted || existingFwdRule == nil {
		glog.V(2).Infof("ensureInternalLoadBalancer(%v(%v)): creating forwarding rule", loadBalancerName, svc.Name)
		if err = gce.CreateRegionForwardingRule(expectedFwdRule, gce.region); err != nil {
			return nil, err
		}
	}

	// Get the most recent forwarding rule for the new address.
	existingFwdRule, err = gce.GetRegionForwardingRule(loadBalancerName, gce.region)
	if err != nil {
		return nil, err
	}

	status := &v1.LoadBalancerStatus{}
	status.Ingress = []v1.LoadBalancerIngress{{IP: existingFwdRule.IPAddress}}
	return status, nil
}

func (gce *GCECloud) clearExistingInternalLB(loadBalancerName string, existingFwdRule, expectedFwdRule *compute.ForwardingRule, expectedBSName string) (fwdRuleDeleted bool, err error) {
	if existingFwdRule == nil {
		return false, nil
	}

	if !fwdRuleEqual(existingFwdRule, expectedFwdRule) {
		glog.V(2).Infof("clearExistingInternalLB(%v: deleting existing forwarding rule with IP address %v", loadBalancerName, existingFwdRule.IPAddress)
		if err = gce.DeleteRegionForwardingRule(loadBalancerName, gce.region); err != nil && !isNotFound(err) {
			return false, err
		}
		fwdRuleDeleted = true
	}

	existingBSName := getNameFromLink(existingFwdRule.BackendService)
	bs, err := gce.GetRegionBackendService(existingBSName, gce.region)
	if err == nil {
		if bs.Name != expectedBSName {
			glog.V(2).Infof("clearExistingInternalLB(%v): expected backend service %q does not match actual %q - deleting backend service & healthcheck & firewall", loadBalancerName, expectedBSName, bs.Name)
			// Delete the backend service as well in case it's switching between shared, nonshared, tcp, udp.
			var existingHCName string
			if len(bs.HealthChecks) == 1 {
				existingHCName = getNameFromLink(bs.HealthChecks[0])
			}
			if err = gce.teardownInternalBackendResources(existingBSName, existingHCName); err != nil {
				glog.Warningf("clearExistingInternalLB: could not delete old resources: %v", err)
			} else {
				glog.V(2).Infof("clearExistingInternalLB: done deleting old resources")
			}
		}
	} else {
		glog.Warningf("clearExistingInternalLB(%v): failed to retrieve existing backend service %v", loadBalancerName, existingBSName)
	}
	return fwdRuleDeleted, nil
}

func (gce *GCECloud) updateInternalLoadBalancer(clusterName, clusterID string, svc *v1.Service, nodes []*v1.Node) error {
	gce.sharedResourceLock.Lock()
	defer gce.sharedResourceLock.Unlock()

	igName := makeInstanceGroupName(clusterID)
	igLinks, err := gce.ensureInternalInstanceGroups(igName, nodes)
	if err != nil {
		return err
	}

	// Generate the backend service name
	_, protocol := getPortsAndProtocol(svc.Spec.Ports)
	scheme := schemeInternal
	loadBalancerName := cloudprovider.GetLoadBalancerName(svc)
	shared := !v1_service.RequestsOnlyLocalTraffic(svc)
	backendServiceName := makeBackendServiceName(loadBalancerName, clusterID, shared, scheme, protocol, svc.Spec.SessionAffinity)
	// Ensure the backend service has the proper backend instance-group links
	return gce.ensureInternalBackendServiceGroups(backendServiceName, igLinks)
}

func (gce *GCECloud) ensureInternalLoadBalancerDeleted(clusterName, clusterID string, svc *v1.Service) error {
	loadBalancerName := cloudprovider.GetLoadBalancerName(svc)
	_, protocol := getPortsAndProtocol(svc.Spec.Ports)
	scheme := schemeInternal
	shared := !v1_service.RequestsOnlyLocalTraffic(svc)
	var err error

	gce.sharedResourceLock.Lock()
	defer gce.sharedResourceLock.Unlock()

	glog.V(2).Infof("ensureInternalLoadBalancerDeleted(%v): deleting region internal forwarding rule", loadBalancerName)
	if err = gce.DeleteRegionForwardingRule(loadBalancerName, gce.region); err != nil && !isNotFound(err) {
		return err
	}

	backendServiceName := makeBackendServiceName(loadBalancerName, clusterID, shared, scheme, protocol, svc.Spec.SessionAffinity)
	glog.V(2).Infof("ensureInternalLoadBalancerDeleted(%v): deleting region backend service %v", loadBalancerName, backendServiceName)
	if err = gce.DeleteRegionBackendService(backendServiceName, gce.region); err != nil && !isNotFoundOrInUse(err) {
		return err
	}

	// Only delete the health check & health check firewall if they aren't being used by another LB.  If we get
	// an ResourceInuseBy error, then we can skip deleting the firewall.
	hcInUse := false
	hcName := makeHealthCheckName(loadBalancerName, clusterID, shared)
	glog.V(2).Infof("ensureInternalLoadBalancerDeleted(%v): deleting health check %v", loadBalancerName, hcName)
	if err = gce.DeleteHealthCheck(hcName); err != nil && !isNotFoundOrInUse(err) {
		return err
	} else if isInUsedByError(err) {
		glog.V(2).Infof("ensureInternalLoadBalancerDeleted(%v): healthcheck %v still in use", loadBalancerName, hcName)
		hcInUse = true
	}

	glog.V(2).Infof("ensureInternalLoadBalancerDeleted(%v): deleting firewall for traffic", loadBalancerName)
	if err = gce.DeleteFirewall(loadBalancerName); err != nil {
		return err
	}

	if hcInUse {
		glog.V(2).Infof("ensureInternalLoadBalancerDeleted(%v): skipping firewall for healthcheck", loadBalancerName)
	} else {
		glog.V(2).Infof("ensureInternalLoadBalancerDeleted(%v): deleting firewall for healthcheck", loadBalancerName)
		fwHCName := makeHealthCheckFirewallkName(loadBalancerName, clusterID, shared)
		if err = gce.DeleteFirewall(fwHCName); err != nil && !isInUsedByError(err) {
			return err
		}
	}

	// Try deleting instance groups - expect ResourceInuse error if needed by other LBs
	igName := makeInstanceGroupName(clusterID)
	if err = gce.ensureInternalInstanceGroupsDeleted(igName); err != nil && !isInUsedByError(err) {
		return err
	}

	return nil
}

func (gce *GCECloud) teardownInternalBackendResources(bsName, hcName string) error {
	if err := gce.DeleteRegionBackendService(bsName, gce.region); err != nil {
		if isNotFound(err) {
			glog.V(2).Infof("backend service already deleted: %v, err: %v", bsName, err)
		} else if err != nil && isInUsedByError(err) {
			glog.V(2).Infof("backend service in use: %v, err: %v", bsName, err)
		} else {
			return fmt.Errorf("failed to delete backend service: %v, err: %v", bsName, err)
		}
	}

	if hcName == "" {
		return nil
	}
	hcInUse := false
	if err := gce.DeleteHealthCheck(hcName); err != nil {
		if isNotFound(err) {
			glog.V(2).Infof("health check already deleted: %v, err: %v", hcName, err)
		} else if err != nil && isInUsedByError(err) {
			hcInUse = true
			glog.V(2).Infof("health check in use: %v, err: %v", hcName, err)
		} else {
			return fmt.Errorf("failed to delete health check: %v, err: %v", hcName, err)
		}
	}

	hcFirewallName := makeHealthCheckFirewallkNameFromHC(hcName)
	if hcInUse {
		glog.V(2).Infof("skipping deletion of health check firewall: %v", hcFirewallName)
		return nil
	}

	if err := gce.DeleteFirewall(hcFirewallName); err != nil && !isNotFound(err) {
		return fmt.Errorf("failed to delete health check firewall: %v, err: %v", hcFirewallName, err)
	}
	return nil
}

func (gce *GCECloud) ensureInternalFirewall(fwName, fwDesc string, sourceRanges []string, ports []string, protocol v1.Protocol, nodes []*v1.Node) error {
	glog.V(2).Infof("ensureInternalFirewall(%v): checking existing firewall", fwName)
	targetTags, err := gce.GetNodeTags(nodeNames(nodes))
	if err != nil {
		return err
	}

	existingFirewall, err := gce.GetFirewall(fwName)
	if err != nil && !isNotFound(err) {
		return err
	}

	expectedFirewall := &compute.Firewall{
		Name:         fwName,
		Description:  fwDesc,
		Network:      gce.networkURL,
		SourceRanges: sourceRanges,
		TargetTags:   targetTags,
		Allowed: []*compute.FirewallAllowed{
			{
				IPProtocol: strings.ToLower(string(protocol)),
				Ports:      ports,
			},
		},
	}

	if existingFirewall == nil {
		glog.V(2).Infof("ensureInternalFirewall(%v): creating firewall", fwName)
		return gce.CreateFirewall(expectedFirewall)
	}

	if firewallRuleEqual(expectedFirewall, existingFirewall) {
		return nil
	}

	glog.V(2).Infof("ensureInternalFirewall(%v): updating firewall", fwName)
	return gce.UpdateFirewall(expectedFirewall)
}

func (gce *GCECloud) ensureInternalFirewalls(loadBalancerName, clusterID string, nm types.NamespacedName, svc *v1.Service, healthCheckPort string, shared bool, nodes []*v1.Node) error {
	// First firewall is for ingress traffic
	fwDesc := makeFirewallDescription(nm.String(), svc.Spec.LoadBalancerIP)
	ports, protocol := getPortsAndProtocol(svc.Spec.Ports)
	sourceRanges, err := v1_service.GetLoadBalancerSourceRanges(svc)
	if err != nil {
		return err
	}
	err = gce.ensureInternalFirewall(loadBalancerName, fwDesc, sourceRanges.StringSlice(), ports, protocol, nodes)
	if err != nil {
		return err
	}

	// Second firewall is for health checking nodes / services
	fwHCName := makeHealthCheckFirewallkName(loadBalancerName, clusterID, shared)
	hcSrcRanges := LoadBalancerSrcRanges()
	return gce.ensureInternalFirewall(fwHCName, "", hcSrcRanges, []string{healthCheckPort}, v1.ProtocolTCP, nodes)
}

func (gce *GCECloud) ensureInternalHealthCheck(name string, svcName types.NamespacedName, shared bool, path string, port int32) (*compute.HealthCheck, error) {
	glog.V(2).Infof("ensureInternalHealthCheck(%v, %v, %v): checking existing health check", name, path, port)
	expectedHC := newInternalLBHealthCheck(name, svcName, shared, path, port)

	hc, err := gce.GetHealthCheck(name)
	if err != nil && !isNotFound(err) {
		return nil, err
	}

	if hc == nil {
		glog.V(2).Infof("ensureInternalHealthCheck: did not find health check %v, creating one with port %v path %v", name, port, path)
		if err = gce.CreateHealthCheck(expectedHC); err != nil {
			return nil, err
		}
		hc, err = gce.GetHealthCheck(name)
		if err != nil {
			glog.Errorf("Failed to get http health check %v", err)
			return nil, err
		}
		glog.V(2).Infof("ensureInternalHealthCheck: created health check %v", name)
		return hc, nil
	}

	if healthChecksEqual(expectedHC, hc) {
		return hc, nil
	}

	glog.V(2).Infof("ensureInternalHealthCheck: health check %v exists but parameters have drifted - updating...", name)
	if err := gce.UpdateHealthCheck(expectedHC); err != nil {
		glog.Warningf("Failed to reconcile http health check %v parameters", name)
		return nil, err
	}
	glog.V(2).Infof("ensureInternalHealthCheck: corrected health check %v parameters successful", name)
	return hc, nil
}

func (gce *GCECloud) ensureInternalInstanceGroup(name, zone string, nodes []*v1.Node) (string, error) {
	glog.V(2).Infof("ensureInternalInstanceGroup(%v, %v): checking group that it contains %v nodes", name, zone, len(nodes))
	ig, err := gce.GetInstanceGroup(name, zone)
	if err != nil && !isNotFound(err) {
		return "", err
	}

	kubeNodes := sets.NewString()
	for _, n := range nodes {
		kubeNodes.Insert(n.Name)
	}

	gceNodes := sets.NewString()
	if ig == nil {
		glog.V(2).Infof("ensureInternalInstanceGroup(%v, %v): creating instance group", name, zone)
		ig, err = gce.CreateInstanceGroup(name, zone)
		if err != nil {
			return "", err
		}
	} else {
		instances, err := gce.ListInstancesInInstanceGroup(name, zone, allInstances)
		if err != nil {
			return "", err
		}

		for _, ins := range instances.Items {
			parts := strings.Split(ins.Instance, "/")
			gceNodes.Insert(parts[len(parts)-1])
		}
	}

	removeNodes := gceNodes.Difference(kubeNodes).List()
	addNodes := kubeNodes.Difference(gceNodes).List()

	if len(removeNodes) != 0 {
		glog.V(2).Infof("ensureInternalInstanceGroup(%v, %v): removing nodes: %v", name, zone, removeNodes)
		instanceRefs := gce.ToInstanceReferences(zone, removeNodes)
		// Possible we'll receive 404's here if the instance was deleted before getting to this point.
		if err = gce.RemoveInstancesFromInstanceGroup(name, zone, instanceRefs); err != nil && !isNotFound(err) {
			return "", err
		}
	}

	if len(addNodes) != 0 {
		glog.V(2).Infof("ensureInternalInstanceGroup(%v, %v): adding nodes: %v", name, zone, addNodes)
		instanceRefs := gce.ToInstanceReferences(zone, addNodes)
		if err = gce.AddInstancesToInstanceGroup(name, zone, instanceRefs); err != nil {
			return "", err
		}
	}

	return ig.SelfLink, nil
}

// ensureInternalInstanceGroups generates an unmanaged instance group for every zone
// where a K8s node exists. It also ensures that each node belongs to an instance group
func (gce *GCECloud) ensureInternalInstanceGroups(name string, nodes []*v1.Node) ([]string, error) {
	zonedNodes := splitNodesByZone(nodes)
	glog.V(2).Infof("ensureInternalInstanceGroups(%v): %d nodes over %d zones in region %v", name, len(nodes), len(zonedNodes), gce.region)
	var igLinks []string
	for zone, nodes := range zonedNodes {
		igLink, err := gce.ensureInternalInstanceGroup(name, zone, nodes)
		if err != nil {
			return []string{}, err
		}
		igLinks = append(igLinks, igLink)
	}

	return igLinks, nil
}

func (gce *GCECloud) ensureInternalInstanceGroupsDeleted(name string) error {
	// List of nodes isn't available here - fetch all zones in region and try deleting this cluster's ig
	zones, err := gce.ListZonesInRegion(gce.region)
	if err != nil {
		return err
	}

	glog.V(2).Infof("ensureInternalInstanceGroupsDeleted(%v): deleting instance group in all %d zones", name, len(zones))
	for _, z := range zones {
		if err := gce.DeleteInstanceGroup(name, z.Name); err != nil && !isNotFound(err) {
			return err
		}
	}
	return nil
}

func (gce *GCECloud) ensureInternalBackendService(name, description string, affinityType v1.ServiceAffinity, scheme lbScheme, protocol v1.Protocol, igLinks []string, hcLink string) error {
	glog.V(2).Infof("ensureInternalBackendService(%v, %v, %v): checking existing backend service with %d groups", name, scheme, protocol, len(igLinks))
	bs, err := gce.GetRegionBackendService(name, gce.region)
	if err != nil && !isNotFound(err) {
		return err
	}

	backends := backendsFromGroupLinks(igLinks)
	expectedBS := &compute.BackendService{
		Name:                name,
		Protocol:            string(protocol),
		Description:         description,
		HealthChecks:        []string{hcLink},
		Backends:            backends,
		SessionAffinity:     translateAffinityType(affinityType),
		LoadBalancingScheme: string(scheme),
	}

	// Create backend service if none was found
	if bs == nil {
		glog.V(2).Infof("ensureInternalBackendService: creating backend service %v", name)
		err := gce.CreateRegionBackendService(expectedBS, gce.region)
		if err != nil {
			return err
		}
		glog.V(2).Infof("ensureInternalBackendService: created backend service %v successfully", name)
		return nil
	}
	// Check existing backend service
	existingIGLinks := sets.NewString()
	for _, be := range bs.Backends {
		existingIGLinks.Insert(be.Group)
	}

	if backendSvcEqual(expectedBS, bs) {
		return nil
	}

	glog.V(2).Infof("ensureInternalBackendService: updating backend service %v", name)
	if err := gce.UpdateRegionBackendService(expectedBS, gce.region); err != nil {
		return err
	}
	glog.V(2).Infof("ensureInternalBackendService: updated backend service %v successfully", name)
	return nil
}

func (gce *GCECloud) ensureInternalBackendServiceGroups(name string, igLinks []string) error {
	glog.V(2).Infof("ensureInternalBackendServiceGroups(%v): checking existing backend service's groups", name)
	bs, err := gce.GetRegionBackendService(name, gce.region)
	if err != nil {
		return err
	}

	backends := backendsFromGroupLinks(igLinks)
	if backendsListEqual(bs.Backends, backends) {
		return nil
	}

	glog.V(2).Infof("ensureInternalBackendServiceGroups: updating backend service %v", name)
	bs.Backends = backends
	if err := gce.UpdateRegionBackendService(bs, gce.region); err != nil {
		return err
	}
	glog.V(2).Infof("ensureInternalBackendServiceGroups: updated backend service %v successfully", name)
	return nil
}

func backendsFromGroupLinks(igLinks []string) []*compute.Backend {
	var backends []*compute.Backend
	for _, igLink := range igLinks {
		backends = append(backends, &compute.Backend{
			Group: igLink,
		})
	}
	return backends
}

func newInternalLBHealthCheck(name string, svcName types.NamespacedName, shared bool, path string, port int32) *compute.HealthCheck {
	httpSettings := compute.HTTPHealthCheck{
		Port:        int64(port),
		RequestPath: path,
	}
	desc := ""
	if !shared {
		desc = makeHealthCheckDescription(svcName.String())
	}
	return &compute.HealthCheck{
		Name:               name,
		CheckIntervalSec:   gceHcCheckIntervalSeconds,
		TimeoutSec:         gceHcTimeoutSeconds,
		HealthyThreshold:   gceHcHealthyThreshold,
		UnhealthyThreshold: gceHcUnhealthyThreshold,
		HttpHealthCheck:    &httpSettings,
		Type:               "HTTP",
		Description:        desc,
	}
}

func firewallRuleEqual(a, b *compute.Firewall) bool {
	return a.Description == b.Description &&
		len(a.Allowed) == 1 && len(a.Allowed) == len(b.Allowed) &&
		a.Allowed[0].IPProtocol == b.Allowed[0].IPProtocol &&
		equalStringSets(a.Allowed[0].Ports, b.Allowed[0].Ports) &&
		equalStringSets(a.SourceRanges, b.SourceRanges) &&
		equalStringSets(a.TargetTags, b.TargetTags)
}

func healthChecksEqual(a, b *compute.HealthCheck) bool {
	return a.HttpHealthCheck != nil && b.HttpHealthCheck != nil &&
		a.HttpHealthCheck.Port == b.HttpHealthCheck.Port &&
		a.HttpHealthCheck.RequestPath == b.HttpHealthCheck.RequestPath &&
		a.Description == b.Description &&
		a.CheckIntervalSec == b.CheckIntervalSec &&
		a.TimeoutSec == b.TimeoutSec &&
		a.UnhealthyThreshold == b.UnhealthyThreshold &&
		a.HealthyThreshold == b.HealthyThreshold
}

// backendsListEqual asserts that backend lists are equal by instance group link only
func backendsListEqual(a, b []*compute.Backend) bool {
	if len(a) != len(b) {
		return false
	}
	if len(a) == 0 {
		return true
	}

	aSet := sets.NewString()
	for _, v := range a {
		aSet.Insert(v.Group)
	}
	bSet := sets.NewString()
	for _, v := range b {
		bSet.Insert(v.Group)
	}

	return aSet.Equal(bSet)
}

func backendSvcEqual(a, b *compute.BackendService) bool {
	return a.Protocol == b.Protocol &&
		a.Description == b.Description &&
		a.SessionAffinity == b.SessionAffinity &&
		a.LoadBalancingScheme == b.LoadBalancingScheme &&
		equalStringSets(a.HealthChecks, b.HealthChecks) &&
		backendsListEqual(a.Backends, b.Backends)
}

func fwdRuleEqual(a, b *compute.ForwardingRule) bool {
	return (a.IPAddress == "" || b.IPAddress == "" || a.IPAddress == b.IPAddress) &&
		a.IPProtocol == b.IPProtocol &&
		a.LoadBalancingScheme == b.LoadBalancingScheme &&
		equalStringSets(a.Ports, b.Ports) &&
		a.BackendService == b.BackendService
}

func getPortsAndProtocol(svcPorts []v1.ServicePort) (ports []string, protocol v1.Protocol) {
	if len(svcPorts) == 0 {
		return []string{}, v1.ProtocolUDP
	}

	// GCP doesn't support multiple protocols for a single load balancer
	protocol = svcPorts[0].Protocol
	for _, p := range svcPorts {
		ports = append(ports, strconv.Itoa(int(p.Port)))
	}
	return ports, protocol
}

func (gce *GCECloud) getBackendServiceLink(name string) string {
	return fmt.Sprintf("https://www.googleapis.com/compute/v1/projects/%s/regions/%s/backendServices/%s", gce.projectID, gce.region, name)
}

func getNameFromLink(link string) string {
	fields := strings.Split(link, "/")
	return fields[len(fields)-1]
}
