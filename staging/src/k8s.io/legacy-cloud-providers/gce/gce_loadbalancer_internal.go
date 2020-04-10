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
	"strconv"
	"strings"

	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud"
	compute "google.golang.org/api/compute/v1"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	servicehelpers "k8s.io/cloud-provider/service/helpers"
	"k8s.io/klog"
)

const (
	allInstances = "ALL"
	// maxInstancesPerInstanceGroup defines maximum number of VMs per InstanceGroup.
	maxInstancesPerInstanceGroup = 1000
)

func usesNEG(service *v1.Service) bool {
	_, ok := service.GetAnnotations()[NEGAnnotation]
	return ok
}

func (g *Cloud) ensureInternalLoadBalancer(clusterName, clusterID string, svc *v1.Service, existingFwdRule *compute.ForwardingRule, nodes []*v1.Node) (*v1.LoadBalancerStatus, error) {
	if usesNEG(svc) {
		// Do not manage loadBalancer for services using NEGs
		return &svc.Status.LoadBalancer, nil
	}
	nm := types.NamespacedName{Name: svc.Name, Namespace: svc.Namespace}
	ports, protocol := getPortsAndProtocol(svc.Spec.Ports)
	if protocol != v1.ProtocolTCP && protocol != v1.ProtocolUDP {
		return nil, fmt.Errorf("Invalid protocol %s, only TCP and UDP are supported", string(protocol))
	}
	scheme := cloud.SchemeInternal
	loadBalancerName := g.GetLoadBalancerName(context.TODO(), clusterName, svc)
	sharedBackend := shareBackendService(svc)
	backendServiceName := makeBackendServiceName(loadBalancerName, clusterID, sharedBackend, scheme, protocol, svc.Spec.SessionAffinity)
	backendServiceLink := g.getBackendServiceLink(backendServiceName)

	// Ensure instance groups exist and nodes are assigned to groups
	igName := makeInstanceGroupName(clusterID)
	igLinks, err := g.ensureInternalInstanceGroups(igName, nodes)
	if err != nil {
		return nil, err
	}

	// Get existing backend service (if exists)
	var existingBackendService *compute.BackendService
	if existingFwdRule != nil && existingFwdRule.BackendService != "" {
		existingBSName := getNameFromLink(existingFwdRule.BackendService)
		if existingBackendService, err = g.GetRegionBackendService(existingBSName, g.region); err != nil && !isNotFound(err) {
			return nil, err
		}
	}

	// Lock the sharedResourceLock to prevent any deletions of shared resources while assembling shared resources here
	g.sharedResourceLock.Lock()
	defer g.sharedResourceLock.Unlock()

	// Ensure health check exists before creating the backend service. The health check is shared
	// if externalTrafficPolicy=Cluster.
	sharedHealthCheck := !servicehelpers.RequestsOnlyLocalTraffic(svc)
	hcName := makeHealthCheckName(loadBalancerName, clusterID, sharedHealthCheck)
	hcPath, hcPort := GetNodesHealthCheckPath(), GetNodesHealthCheckPort()
	if !sharedHealthCheck {
		// Service requires a special health check, retrieve the OnlyLocal port & path
		hcPath, hcPort = servicehelpers.GetServiceHealthCheckPathPort(svc)
	}
	hc, err := g.ensureInternalHealthCheck(hcName, nm, sharedHealthCheck, hcPath, hcPort)
	if err != nil {
		return nil, err
	}

	// Determine IP which will be used for this LB. If no forwarding rule has been established
	// or specified in the Service spec, then requestedIP = "".
	requestedIP := determineRequestedIP(svc, existingFwdRule)
	ipToUse := requestedIP

	// If the ILB already exists, continue using the subnet that it's already using.
	// This is to support existing ILBs that were setup using the wrong subnet.
	subnetworkURL := g.SubnetworkURL()
	if existingFwdRule != nil && existingFwdRule.Subnetwork != "" {
		// external LBs have an empty Subnetwork field.
		subnetworkURL = existingFwdRule.Subnetwork
	}

	var addrMgr *addressManager
	// If the network is not a legacy network, use the address manager
	if !g.IsLegacyNetwork() {
		addrMgr = newAddressManager(g, nm.String(), g.Region(), subnetworkURL, loadBalancerName, requestedIP, cloud.SchemeInternal)
		ipToUse, err = addrMgr.HoldAddress()
		if err != nil {
			return nil, err
		}
		klog.V(2).Infof("ensureInternalLoadBalancer(%v): reserved IP %q for the forwarding rule", loadBalancerName, ipToUse)
	}

	// Ensure firewall rules if necessary
	if err = g.ensureInternalFirewalls(loadBalancerName, ipToUse, clusterID, nm, svc, strconv.Itoa(int(hcPort)), sharedHealthCheck, nodes); err != nil {
		return nil, err
	}

	expectedFwdRule := &compute.ForwardingRule{
		Name:                loadBalancerName,
		Description:         fmt.Sprintf(`{"kubernetes.io/service-name":"%s"}`, nm.String()),
		IPAddress:           ipToUse,
		BackendService:      backendServiceLink,
		Ports:               ports,
		IPProtocol:          string(protocol),
		LoadBalancingScheme: string(scheme),
	}

	// Given that CreateGCECloud will attempt to determine the subnet based off the network,
	// the subnetwork should rarely be unknown.
	if subnetworkURL != "" {
		expectedFwdRule.Subnetwork = subnetworkURL
	} else {
		expectedFwdRule.Network = g.networkURL
	}

	fwdRuleDeleted := false
	if existingFwdRule != nil && !fwdRuleEqual(existingFwdRule, expectedFwdRule) {
		klog.V(2).Infof("ensureInternalLoadBalancer(%v): deleting existing forwarding rule with IP address %v", loadBalancerName, existingFwdRule.IPAddress)
		if err = ignoreNotFound(g.DeleteRegionForwardingRule(loadBalancerName, g.region)); err != nil {
			return nil, err
		}
		fwdRuleDeleted = true
	}

	bsDescription := makeBackendServiceDescription(nm, sharedBackend)
	err = g.ensureInternalBackendService(backendServiceName, bsDescription, svc.Spec.SessionAffinity, scheme, protocol, igLinks, hc.SelfLink)
	if err != nil {
		return nil, err
	}

	// If we previously deleted the forwarding rule or it never existed, finally create it.
	if fwdRuleDeleted || existingFwdRule == nil {
		klog.V(2).Infof("ensureInternalLoadBalancer(%v): creating forwarding rule", loadBalancerName)
		if err = g.CreateRegionForwardingRule(expectedFwdRule, g.region); err != nil {
			return nil, err
		}
		klog.V(2).Infof("ensureInternalLoadBalancer(%v): created forwarding rule", loadBalancerName)
	}

	// Delete the previous internal load balancer resources if necessary
	if existingBackendService != nil {
		g.clearPreviousInternalResources(svc, loadBalancerName, existingBackendService, backendServiceName, hcName)
	}

	if addrMgr != nil {
		// Now that the controller knows the forwarding rule exists, we can release the address.
		if err := addrMgr.ReleaseAddress(); err != nil {
			klog.Errorf("ensureInternalLoadBalancer: failed to release address reservation, possibly causing an orphan: %v", err)
		}
	}

	// Get the most recent forwarding rule for the address.
	updatedFwdRule, err := g.GetRegionForwardingRule(loadBalancerName, g.region)
	if err != nil {
		return nil, err
	}

	status := &v1.LoadBalancerStatus{}
	status.Ingress = []v1.LoadBalancerIngress{{IP: updatedFwdRule.IPAddress}}
	return status, nil
}

func (g *Cloud) clearPreviousInternalResources(svc *v1.Service, loadBalancerName string, existingBackendService *compute.BackendService, expectedBSName, expectedHCName string) {
	// If a new backend service was created, delete the old one.
	if existingBackendService.Name != expectedBSName {
		klog.V(2).Infof("clearPreviousInternalResources(%v): expected backend service %q does not match previous %q - deleting backend service", loadBalancerName, expectedBSName, existingBackendService.Name)
		if err := g.teardownInternalBackendService(existingBackendService.Name); err != nil && !isNotFound(err) {
			klog.Warningf("clearPreviousInternalResources: could not delete old backend service: %v, err: %v", existingBackendService.Name, err)
		}
	}

	// If a new health check was created, delete the old one.
	if len(existingBackendService.HealthChecks) == 1 {
		existingHCName := getNameFromLink(existingBackendService.HealthChecks[0])
		if existingHCName != expectedHCName {
			klog.V(2).Infof("clearPreviousInternalResources(%v): expected health check %q does not match previous %q - deleting health check", loadBalancerName, expectedHCName, existingHCName)
			if err := g.teardownInternalHealthCheckAndFirewall(svc, existingHCName); err != nil {
				klog.Warningf("clearPreviousInternalResources: could not delete existing healthcheck: %v, err: %v", existingHCName, err)
			}
		}
	} else if len(existingBackendService.HealthChecks) > 1 {
		klog.Warningf("clearPreviousInternalResources(%v): more than one health check on the backend service %v, %v", loadBalancerName, existingBackendService.Name, existingBackendService.HealthChecks)
	}
}

// updateInternalLoadBalancer is called when the list of nodes has changed. Therefore, only the instance groups
// and possibly the backend service need to be updated.
func (g *Cloud) updateInternalLoadBalancer(clusterName, clusterID string, svc *v1.Service, nodes []*v1.Node) error {
	if usesNEG(svc) {
		// Do not manage loadBalancer for services using NEGs
		return nil
	}
	g.sharedResourceLock.Lock()
	defer g.sharedResourceLock.Unlock()

	igName := makeInstanceGroupName(clusterID)
	igLinks, err := g.ensureInternalInstanceGroups(igName, nodes)
	if err != nil {
		return err
	}

	// Generate the backend service name
	_, protocol := getPortsAndProtocol(svc.Spec.Ports)
	scheme := cloud.SchemeInternal
	loadBalancerName := g.GetLoadBalancerName(context.TODO(), clusterName, svc)
	backendServiceName := makeBackendServiceName(loadBalancerName, clusterID, shareBackendService(svc), scheme, protocol, svc.Spec.SessionAffinity)
	// Ensure the backend service has the proper backend/instance-group links
	return g.ensureInternalBackendServiceGroups(backendServiceName, igLinks)
}

func (g *Cloud) ensureInternalLoadBalancerDeleted(clusterName, clusterID string, svc *v1.Service) error {
	loadBalancerName := g.GetLoadBalancerName(context.TODO(), clusterName, svc)
	_, protocol := getPortsAndProtocol(svc.Spec.Ports)
	scheme := cloud.SchemeInternal
	sharedBackend := shareBackendService(svc)
	sharedHealthCheck := !servicehelpers.RequestsOnlyLocalTraffic(svc)

	g.sharedResourceLock.Lock()
	defer g.sharedResourceLock.Unlock()

	klog.V(2).Infof("ensureInternalLoadBalancerDeleted(%v): attempting delete of region internal address", loadBalancerName)
	ensureAddressDeleted(g, loadBalancerName, g.region)

	klog.V(2).Infof("ensureInternalLoadBalancerDeleted(%v): deleting region internal forwarding rule", loadBalancerName)
	if err := ignoreNotFound(g.DeleteRegionForwardingRule(loadBalancerName, g.region)); err != nil {
		return err
	}

	backendServiceName := makeBackendServiceName(loadBalancerName, clusterID, sharedBackend, scheme, protocol, svc.Spec.SessionAffinity)
	klog.V(2).Infof("ensureInternalLoadBalancerDeleted(%v): deleting region backend service %v", loadBalancerName, backendServiceName)
	if err := g.teardownInternalBackendService(backendServiceName); err != nil {
		return err
	}

	deleteFunc := func(fwName string) error {
		if err := ignoreNotFound(g.DeleteFirewall(fwName)); err != nil {
			if isForbidden(err) && g.OnXPN() {
				klog.V(2).Infof("ensureInternalLoadBalancerDeleted(%v): could not delete traffic firewall on XPN cluster. Raising event.", loadBalancerName)
				g.raiseFirewallChangeNeededEvent(svc, FirewallToGCloudDeleteCmd(fwName, g.NetworkProjectID()))
				return nil
			}
			return err
		}
		return nil
	}
	fwName := MakeFirewallName(loadBalancerName)
	klog.V(2).Infof("ensureInternalLoadBalancerDeleted(%v): deleting firewall %s for traffic",
		loadBalancerName, fwName)
	if err := deleteFunc(fwName); err != nil {
		return err
	}
	klog.V(2).Infof("ensureInternalLoadBalancerDeleted(%v): deleting legacy name firewall for traffic", loadBalancerName)
	if err := deleteFunc(loadBalancerName); err != nil {
		return err
	}

	hcName := makeHealthCheckName(loadBalancerName, clusterID, sharedHealthCheck)
	klog.V(2).Infof("ensureInternalLoadBalancerDeleted(%v): deleting health check %v and its firewall", loadBalancerName, hcName)
	if err := g.teardownInternalHealthCheckAndFirewall(svc, hcName); err != nil {
		return err
	}

	// Try deleting instance groups - expect ResourceInuse error if needed by other LBs
	igName := makeInstanceGroupName(clusterID)
	if err := g.ensureInternalInstanceGroupsDeleted(igName); err != nil && !isInUsedByError(err) {
		return err
	}

	return nil
}

func (g *Cloud) teardownInternalBackendService(bsName string) error {
	if err := g.DeleteRegionBackendService(bsName, g.region); err != nil {
		if isNotFound(err) {
			klog.V(2).Infof("teardownInternalBackendService(%v): backend service already deleted. err: %v", bsName, err)
			return nil
		} else if isInUsedByError(err) {
			klog.V(2).Infof("teardownInternalBackendService(%v): backend service in use.", bsName)
			return nil
		} else {
			return fmt.Errorf("failed to delete backend service: %v, err: %v", bsName, err)
		}
	}
	klog.V(2).Infof("teardownInternalBackendService(%v): backend service deleted", bsName)
	return nil
}

func (g *Cloud) teardownInternalHealthCheckAndFirewall(svc *v1.Service, hcName string) error {
	if err := g.DeleteHealthCheck(hcName); err != nil {
		if isNotFound(err) {
			klog.V(2).Infof("teardownInternalHealthCheckAndFirewall(%v): health check does not exist.", hcName)
			// Purposely do not early return - double check the firewall does not exist
		} else if isInUsedByError(err) {
			klog.V(2).Infof("teardownInternalHealthCheckAndFirewall(%v): health check in use.", hcName)
			return nil
		} else {
			return fmt.Errorf("failed to delete health check: %v, err: %v", hcName, err)
		}
	}
	klog.V(2).Infof("teardownInternalHealthCheckAndFirewall(%v): health check deleted", hcName)

	hcFirewallName := makeHealthCheckFirewallNameFromHC(hcName)
	if err := ignoreNotFound(g.DeleteFirewall(hcFirewallName)); err != nil {
		if isForbidden(err) && g.OnXPN() {
			klog.V(2).Infof("teardownInternalHealthCheckAndFirewall(%v): could not delete health check traffic firewall on XPN cluster. Raising Event.", hcName)
			g.raiseFirewallChangeNeededEvent(svc, FirewallToGCloudDeleteCmd(hcFirewallName, g.NetworkProjectID()))
			return nil
		}

		return fmt.Errorf("failed to delete health check firewall: %v, err: %v", hcFirewallName, err)
	}
	klog.V(2).Infof("teardownInternalHealthCheckAndFirewall(%v): health check firewall deleted", hcFirewallName)
	return nil
}

func (g *Cloud) ensureInternalFirewall(svc *v1.Service, fwName, fwDesc string, sourceRanges []string, ports []string, protocol v1.Protocol, nodes []*v1.Node, legacyFwName string) error {
	klog.V(2).Infof("ensureInternalFirewall(%v): checking existing firewall", fwName)
	targetTags, err := g.GetNodeTags(nodeNames(nodes))
	if err != nil {
		return err
	}

	existingFirewall, err := g.GetFirewall(fwName)
	if err != nil && !isNotFound(err) {
		return err
	}
	// TODO(84821) Remove legacyFwName logic after 3 releases, so there would have been atleast 2 master upgrades that would
	// have triggered service sync and deletion of the legacy rules.
	if legacyFwName != "" {
		// Check for firewall named with the legacy naming scheme and delete if found.
		legacyFirewall, err := g.GetFirewall(legacyFwName)
		if err != nil && !isNotFound(err) {
			return err
		}
		if legacyFirewall != nil && existingFirewall != nil {
			// Delete the legacyFirewall rule if the new one was already created. If not, it will be deleted in the
			// next sync or when the service is deleted.
			defer func() {
				err = g.DeleteFirewall(legacyFwName)
				if err != nil {
					klog.Errorf("Failed to delete legacy firewall %s for service %s/%s, err %v",
						legacyFwName, svc.Namespace, svc.Name, err)
				} else {
					klog.V(2).Infof("Successfully deleted legacy firewall %s for service %s/%s",
						legacyFwName, svc.Namespace, svc.Name)
				}
			}()
		}
	}

	expectedFirewall := &compute.Firewall{
		Name:         fwName,
		Description:  fwDesc,
		Network:      g.networkURL,
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
		klog.V(2).Infof("ensureInternalFirewall(%v): creating firewall", fwName)
		err = g.CreateFirewall(expectedFirewall)
		if err != nil && isForbidden(err) && g.OnXPN() {
			klog.V(2).Infof("ensureInternalFirewall(%v): do not have permission to create firewall rule (on XPN). Raising event.", fwName)
			g.raiseFirewallChangeNeededEvent(svc, FirewallToGCloudCreateCmd(expectedFirewall, g.NetworkProjectID()))
			return nil
		}
		return err
	}

	if firewallRuleEqual(expectedFirewall, existingFirewall) {
		return nil
	}

	klog.V(2).Infof("ensureInternalFirewall(%v): updating firewall", fwName)
	err = g.UpdateFirewall(expectedFirewall)
	if err != nil && isForbidden(err) && g.OnXPN() {
		klog.V(2).Infof("ensureInternalFirewall(%v): do not have permission to update firewall rule (on XPN). Raising event.", fwName)
		g.raiseFirewallChangeNeededEvent(svc, FirewallToGCloudUpdateCmd(expectedFirewall, g.NetworkProjectID()))
		return nil
	}
	return err
}

func (g *Cloud) ensureInternalFirewalls(loadBalancerName, ipAddress, clusterID string, nm types.NamespacedName, svc *v1.Service, healthCheckPort string, sharedHealthCheck bool, nodes []*v1.Node) error {
	// First firewall is for ingress traffic
	fwDesc := makeFirewallDescription(nm.String(), ipAddress)
	ports, protocol := getPortsAndProtocol(svc.Spec.Ports)
	sourceRanges, err := servicehelpers.GetLoadBalancerSourceRanges(svc)
	if err != nil {
		return err
	}
	err = g.ensureInternalFirewall(svc, MakeFirewallName(loadBalancerName), fwDesc, sourceRanges.StringSlice(), ports, protocol, nodes, loadBalancerName)
	if err != nil {
		return err
	}

	// Second firewall is for health checking nodes / services
	fwHCName := makeHealthCheckFirewallName(loadBalancerName, clusterID, sharedHealthCheck)
	hcSrcRanges := LoadBalancerSrcRanges()
	return g.ensureInternalFirewall(svc, fwHCName, "", hcSrcRanges, []string{healthCheckPort}, v1.ProtocolTCP, nodes, "")
}

func (g *Cloud) ensureInternalHealthCheck(name string, svcName types.NamespacedName, shared bool, path string, port int32) (*compute.HealthCheck, error) {
	klog.V(2).Infof("ensureInternalHealthCheck(%v, %v, %v): checking existing health check", name, path, port)
	expectedHC := newInternalLBHealthCheck(name, svcName, shared, path, port)

	hc, err := g.GetHealthCheck(name)
	if err != nil && !isNotFound(err) {
		return nil, err
	}

	if hc == nil {
		klog.V(2).Infof("ensureInternalHealthCheck: did not find health check %v, creating one with port %v path %v", name, port, path)
		if err = g.CreateHealthCheck(expectedHC); err != nil {
			return nil, err
		}
		hc, err = g.GetHealthCheck(name)
		if err != nil {
			klog.Errorf("Failed to get http health check %v", err)
			return nil, err
		}
		klog.V(2).Infof("ensureInternalHealthCheck: created health check %v", name)
		return hc, nil
	}

	if needToUpdateHealthChecks(hc, expectedHC) {
		klog.V(2).Infof("ensureInternalHealthCheck: health check %v exists but parameters have drifted - updating...", name)
		expectedHC = mergeHealthChecks(hc, expectedHC)
		if err := g.UpdateHealthCheck(expectedHC); err != nil {
			klog.Warningf("Failed to reconcile http health check %v parameters", name)
			return nil, err
		}
		klog.V(2).Infof("ensureInternalHealthCheck: corrected health check %v parameters successful", name)
		hc, err = g.GetHealthCheck(name)
		if err != nil {
			return nil, err
		}
	}
	return hc, nil
}

func (g *Cloud) ensureInternalInstanceGroup(name, zone string, nodes []*v1.Node) (string, error) {
	klog.V(2).Infof("ensureInternalInstanceGroup(%v, %v): checking group that it contains %v nodes", name, zone, len(nodes))
	ig, err := g.GetInstanceGroup(name, zone)
	if err != nil && !isNotFound(err) {
		return "", err
	}

	kubeNodes := sets.NewString()
	for _, n := range nodes {
		kubeNodes.Insert(n.Name)
	}

	// Individual InstanceGroup has a limit for 1000 instances in it.
	// As a result, it's not possible to add more to it.
	// Given that the long-term fix (AlphaFeatureILBSubsets) is already in-progress,
	// to stop the bleeding we now simply cut down the contents to first 1000
	// instances in the alphabetical order. Since there is a limitation for
	// 250 backend VMs for ILB, this isn't making things worse.
	if len(kubeNodes) > maxInstancesPerInstanceGroup {
		klog.Warningf("Limiting number of VMs for InstanceGroup %s to %d", name, maxInstancesPerInstanceGroup)
		kubeNodes = sets.NewString(kubeNodes.List()[:maxInstancesPerInstanceGroup]...)
	}

	gceNodes := sets.NewString()
	if ig == nil {
		klog.V(2).Infof("ensureInternalInstanceGroup(%v, %v): creating instance group", name, zone)
		newIG := &compute.InstanceGroup{Name: name}
		if err = g.CreateInstanceGroup(newIG, zone); err != nil {
			return "", err
		}

		ig, err = g.GetInstanceGroup(name, zone)
		if err != nil {
			return "", err
		}
	} else {
		instances, err := g.ListInstancesInInstanceGroup(name, zone, allInstances)
		if err != nil {
			return "", err
		}

		for _, ins := range instances {
			parts := strings.Split(ins.Instance, "/")
			gceNodes.Insert(parts[len(parts)-1])
		}
	}

	removeNodes := gceNodes.Difference(kubeNodes).List()
	addNodes := kubeNodes.Difference(gceNodes).List()

	if len(removeNodes) != 0 {
		klog.V(2).Infof("ensureInternalInstanceGroup(%v, %v): removing nodes: %v", name, zone, removeNodes)
		instanceRefs := g.ToInstanceReferences(zone, removeNodes)
		// Possible we'll receive 404's here if the instance was deleted before getting to this point.
		if err = g.RemoveInstancesFromInstanceGroup(name, zone, instanceRefs); err != nil && !isNotFound(err) {
			return "", err
		}
	}

	if len(addNodes) != 0 {
		klog.V(2).Infof("ensureInternalInstanceGroup(%v, %v): adding nodes: %v", name, zone, addNodes)
		instanceRefs := g.ToInstanceReferences(zone, addNodes)
		if err = g.AddInstancesToInstanceGroup(name, zone, instanceRefs); err != nil {
			return "", err
		}
	}

	return ig.SelfLink, nil
}

// ensureInternalInstanceGroups generates an unmanaged instance group for every zone
// where a K8s node exists. It also ensures that each node belongs to an instance group
func (g *Cloud) ensureInternalInstanceGroups(name string, nodes []*v1.Node) ([]string, error) {
	zonedNodes := splitNodesByZone(nodes)
	klog.V(2).Infof("ensureInternalInstanceGroups(%v): %d nodes over %d zones in region %v", name, len(nodes), len(zonedNodes), g.region)
	var igLinks []string
	for zone, nodes := range zonedNodes {
		igLink, err := g.ensureInternalInstanceGroup(name, zone, nodes)
		if err != nil {
			return []string{}, err
		}
		igLinks = append(igLinks, igLink)
	}

	return igLinks, nil
}

func (g *Cloud) ensureInternalInstanceGroupsDeleted(name string) error {
	// List of nodes isn't available here - fetch all zones in region and try deleting this cluster's ig
	zones, err := g.ListZonesInRegion(g.region)
	if err != nil {
		return err
	}

	klog.V(2).Infof("ensureInternalInstanceGroupsDeleted(%v): attempting delete instance group in all %d zones", name, len(zones))
	for _, z := range zones {
		if err := g.DeleteInstanceGroup(name, z.Name); err != nil && !isNotFoundOrInUse(err) {
			return err
		}
	}
	return nil
}

func (g *Cloud) ensureInternalBackendService(name, description string, affinityType v1.ServiceAffinity, scheme cloud.LbScheme, protocol v1.Protocol, igLinks []string, hcLink string) error {
	klog.V(2).Infof("ensureInternalBackendService(%v, %v, %v): checking existing backend service with %d groups", name, scheme, protocol, len(igLinks))
	bs, err := g.GetRegionBackendService(name, g.region)
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
		klog.V(2).Infof("ensureInternalBackendService: creating backend service %v", name)
		err := g.CreateRegionBackendService(expectedBS, g.region)
		if err != nil {
			return err
		}
		klog.V(2).Infof("ensureInternalBackendService: created backend service %v successfully", name)
		return nil
	}

	if backendSvcEqual(expectedBS, bs) {
		return nil
	}

	klog.V(2).Infof("ensureInternalBackendService: updating backend service %v", name)
	// Set fingerprint for optimistic locking
	expectedBS.Fingerprint = bs.Fingerprint
	if err := g.UpdateRegionBackendService(expectedBS, g.region); err != nil {
		return err
	}
	klog.V(2).Infof("ensureInternalBackendService: updated backend service %v successfully", name)
	return nil
}

// ensureInternalBackendServiceGroups updates backend services if their list of backend instance groups is incorrect.
func (g *Cloud) ensureInternalBackendServiceGroups(name string, igLinks []string) error {
	klog.V(2).Infof("ensureInternalBackendServiceGroups(%v): checking existing backend service's groups", name)
	bs, err := g.GetRegionBackendService(name, g.region)
	if err != nil {
		return err
	}

	backends := backendsFromGroupLinks(igLinks)
	if backendsListEqual(bs.Backends, backends) {
		return nil
	}

	// Set the backend service's backends to the updated list.
	bs.Backends = backends

	klog.V(2).Infof("ensureInternalBackendServiceGroups: updating backend service %v", name)
	if err := g.UpdateRegionBackendService(bs, g.region); err != nil {
		return err
	}
	klog.V(2).Infof("ensureInternalBackendServiceGroups: updated backend service %v successfully", name)
	return nil
}

func shareBackendService(svc *v1.Service) bool {
	return GetLoadBalancerAnnotationBackendShare(svc) && !servicehelpers.RequestsOnlyLocalTraffic(svc)
}

func backendsFromGroupLinks(igLinks []string) (backends []*compute.Backend) {
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

// mergeHealthChecks reconciles HealthCheck configures to be no smaller than
// the default values.
// E.g. old health check interval is 2s, new default is 8.
// The HC interval will be reconciled to 8 seconds.
// If the existing health check is larger than the default interval,
// the configuration will be kept.
func mergeHealthChecks(hc, newHC *compute.HealthCheck) *compute.HealthCheck {
	if hc.CheckIntervalSec > newHC.CheckIntervalSec {
		newHC.CheckIntervalSec = hc.CheckIntervalSec
	}
	if hc.TimeoutSec > newHC.TimeoutSec {
		newHC.TimeoutSec = hc.TimeoutSec
	}
	if hc.UnhealthyThreshold > newHC.UnhealthyThreshold {
		newHC.UnhealthyThreshold = hc.UnhealthyThreshold
	}
	if hc.HealthyThreshold > newHC.HealthyThreshold {
		newHC.HealthyThreshold = hc.HealthyThreshold
	}
	return newHC
}

// needToUpdateHealthChecks checks whether the healthcheck needs to be updated.
func needToUpdateHealthChecks(hc, newHC *compute.HealthCheck) bool {
	if hc.HttpHealthCheck == nil || newHC.HttpHealthCheck == nil {
		return true
	}
	changed := hc.HttpHealthCheck.Port != newHC.HttpHealthCheck.Port || hc.HttpHealthCheck.RequestPath != newHC.HttpHealthCheck.RequestPath || hc.Description != newHC.Description
	changed = changed || hc.CheckIntervalSec < newHC.CheckIntervalSec || hc.TimeoutSec < newHC.TimeoutSec
	changed = changed || hc.UnhealthyThreshold < newHC.UnhealthyThreshold || hc.HealthyThreshold < newHC.HealthyThreshold
	return changed
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

func (g *Cloud) getBackendServiceLink(name string) string {
	return g.service.BasePath + strings.Join([]string{g.projectID, "regions", g.region, "backendServices", name}, "/")
}

func getNameFromLink(link string) string {
	if link == "" {
		return ""
	}

	fields := strings.Split(link, "/")
	return fields[len(fields)-1]
}

func determineRequestedIP(svc *v1.Service, fwdRule *compute.ForwardingRule) string {
	if svc.Spec.LoadBalancerIP != "" {
		return svc.Spec.LoadBalancerIP
	}

	if fwdRule != nil {
		return fwdRule.IPAddress
	}

	return ""
}
