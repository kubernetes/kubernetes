/*
Copyright 2014 The Kubernetes Authors.

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

package storage

import (
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	apiservice "k8s.io/kubernetes/pkg/api/service"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
	"k8s.io/kubernetes/pkg/registry/core/service/portallocator"
	netutils "k8s.io/utils/net"
)

// RESTAllocStuff is a temporary struct to facilitate the flattening of service
// REST layers.  It will be cleaned up over a series of commits.
type RESTAllocStuff struct {
	serviceIPAllocatorsByFamily map[api.IPFamily]ipallocator.Interface
	defaultServiceIPFamily      api.IPFamily // --service-cluster-ip-range[0]
	serviceNodePorts            portallocator.Interface
}

// ServiceNodePort includes protocol and port number of a service NodePort.
type ServiceNodePort struct {
	// The IP protocol for this port. Supports "TCP" and "UDP".
	Protocol api.Protocol

	// The port on each node on which this service is exposed.
	// Default is to auto-allocate a port if the ServiceType of this Service requires one.
	NodePort int32
}

// This is a trasitionary function to facilitate service REST flattening.
func makeAlloc(defaultFamily api.IPFamily, ipAllocs map[api.IPFamily]ipallocator.Interface, portAlloc portallocator.Interface) RESTAllocStuff {
	return RESTAllocStuff{
		defaultServiceIPFamily:      defaultFamily,
		serviceIPAllocatorsByFamily: ipAllocs,
		serviceNodePorts:            portAlloc,
	}
}

func (al *RESTAllocStuff) allocateCreate(service *api.Service, dryRun bool) (transaction, error) {
	result := metaTransaction{}

	// Ensure IP family fields are correctly initialized.  We do it here, since
	// we want this to be visible even when dryRun == true.
	if err := al.initIPFamilyFields(nil, service); err != nil {
		return nil, err
	}

	// Allocate ClusterIPs
	//TODO(thockin): validation should not pass with empty clusterIP, but it
	//does (and is tested!).  Fixing that all is a big PR and will have to
	//happen later.
	if txn, err := al.txnAllocClusterIPs(service, dryRun); err != nil {
		result.Revert()
		return nil, err
	} else {
		result = append(result, txn)
	}

	// Allocate ports
	if txn, err := al.txnAllocNodePorts(service, dryRun); err != nil {
		result.Revert()
		return nil, err
	} else {
		result = append(result, txn)
	}

	return result, nil
}

func (al *RESTAllocStuff) releaseAllocatedResources(svc *api.Service) {
	al.releaseServiceClusterIPs(svc)

	for _, nodePort := range collectServiceNodePorts(svc) {
		err := al.serviceNodePorts.Release(nodePort)
		if err != nil {
			// these should be caught by an eventual reconciliation / restart
			utilruntime.HandleError(fmt.Errorf("Error releasing service %s node port %d: %v", svc.Name, nodePort, err))
		}
	}

	if apiservice.NeedsHealthCheck(svc) {
		nodePort := svc.Spec.HealthCheckNodePort
		if nodePort > 0 {
			err := al.serviceNodePorts.Release(int(nodePort))
			if err != nil {
				// these should be caught by an eventual reconciliation / restart
				utilruntime.HandleError(fmt.Errorf("Error releasing service %s health check node port %d: %v", svc.Name, nodePort, err))
			}
		}
	}
}

func shouldAllocateNodePorts(service *api.Service) bool {
	if service.Spec.Type == api.ServiceTypeNodePort {
		return true
	}
	if service.Spec.Type == api.ServiceTypeLoadBalancer {
		if utilfeature.DefaultFeatureGate.Enabled(features.ServiceLBNodePortControl) {
			return *service.Spec.AllocateLoadBalancerNodePorts
		}
		return true
	}
	return false
}

// healthCheckNodePortUpdate handles HealthCheckNodePort allocation/release
// and adjusts HealthCheckNodePort during service update if needed.
func (al *RESTAllocStuff) healthCheckNodePortUpdate(oldService, service *api.Service, nodePortOp *portallocator.PortAllocationOperation) (bool, error) {
	neededHealthCheckNodePort := apiservice.NeedsHealthCheck(oldService)
	oldHealthCheckNodePort := oldService.Spec.HealthCheckNodePort

	needsHealthCheckNodePort := apiservice.NeedsHealthCheck(service)

	switch {
	// Case 1: Transition from don't need HealthCheckNodePort to needs HealthCheckNodePort.
	// Allocate a health check node port or attempt to reserve the user-specified one if provided.
	// Insert health check node port into the service's HealthCheckNodePort field if needed.
	case !neededHealthCheckNodePort && needsHealthCheckNodePort:
		klog.Infof("Transition to LoadBalancer type service with ExternalTrafficPolicy=Local")
		if err := allocateHealthCheckNodePort(service, nodePortOp); err != nil {
			return false, errors.NewInternalError(err)
		}

	// Case 2: Transition from needs HealthCheckNodePort to don't need HealthCheckNodePort.
	// Free the existing healthCheckNodePort and clear the HealthCheckNodePort field.
	case neededHealthCheckNodePort && !needsHealthCheckNodePort:
		klog.Infof("Transition to non LoadBalancer type service or LoadBalancer type service with ExternalTrafficPolicy=Global")
		klog.V(4).Infof("Releasing healthCheckNodePort: %d", oldHealthCheckNodePort)
		nodePortOp.ReleaseDeferred(int(oldHealthCheckNodePort))
	}
	return true, nil
}

func (al *RESTAllocStuff) allocateUpdate(service, oldService *api.Service, dryRun bool) (transaction, error) {
	result := metaTransaction{}

	// Ensure IP family fields are correctly initialized.  We do it here, since
	// we want this to be visible even when dryRun == true.
	if err := al.initIPFamilyFields(oldService, service); err != nil {
		return nil, err
	}

	// Allocate ClusterIPs
	//TODO(thockin): validation should not pass with empty clusterIP, but it
	//does (and is tested!).  Fixing that all is a big PR and will have to
	//happen later.
	if txn, err := al.txnUpdateClusterIPs(service, oldService, dryRun); err != nil {
		result.Revert()
		return nil, err
	} else {
		result = append(result, txn)
	}

	// Allocate ports
	if txn, err := al.txnUpdateNodePorts(service, oldService, dryRun); err != nil {
		result.Revert()
		return nil, err
	} else {
		result = append(result, txn)
	}

	return result, nil
}

func (al *RESTAllocStuff) txnUpdateNodePorts(service, oldService *api.Service, dryRun bool) (transaction, error) {
	// The allocator tracks dry-run-ness internally.
	nodePortOp := portallocator.StartOperation(al.serviceNodePorts, dryRun)

	txn := callbackTransaction{
		commit: func() {
			nodePortOp.Commit()
			// We don't NEED to call Finish() here, but for that package says
			// to, so for future-safety, we will.
			nodePortOp.Finish()
		},
		revert: func() {
			// Weirdly named but this will revert if commit wasn't called
			nodePortOp.Finish()
		},
	}

	// Update service from NodePort or LoadBalancer to ExternalName or ClusterIP, should release NodePort if exists.
	if (oldService.Spec.Type == api.ServiceTypeNodePort || oldService.Spec.Type == api.ServiceTypeLoadBalancer) &&
		(service.Spec.Type == api.ServiceTypeExternalName || service.Spec.Type == api.ServiceTypeClusterIP) {
		releaseNodePorts(oldService, nodePortOp)
	}

	// Update service from any type to NodePort or LoadBalancer, should update NodePort.
	if service.Spec.Type == api.ServiceTypeNodePort || service.Spec.Type == api.ServiceTypeLoadBalancer {
		if err := updateNodePorts(oldService, service, nodePortOp); err != nil {
			txn.Revert()
			return nil, err
		}
	}

	// Handle ExternalTraffic related updates.
	success, err := al.healthCheckNodePortUpdate(oldService, service, nodePortOp)
	if !success || err != nil {
		txn.Revert()
		return nil, err
	}

	return txn, nil
}

func (al *RESTAllocStuff) allocIPs(service *api.Service, toAlloc map[api.IPFamily]string, dryRun bool) (map[api.IPFamily]string, error) {
	allocated := make(map[api.IPFamily]string)

	for family, ip := range toAlloc {
		allocator := al.serviceIPAllocatorsByFamily[family] // should always be there, as we pre validate
		if dryRun {
			allocator = allocator.DryRun()
		}
		if ip == "" {
			allocatedIP, err := allocator.AllocateNext()
			if err != nil {
				return allocated, errors.NewInternalError(fmt.Errorf("failed to allocate a serviceIP: %v", err))
			}
			allocated[family] = allocatedIP.String()
		} else {
			parsedIP := netutils.ParseIPSloppy(ip)
			if err := allocator.Allocate(parsedIP); err != nil {
				el := field.ErrorList{field.Invalid(field.NewPath("spec", "clusterIPs"), service.Spec.ClusterIPs, fmt.Sprintf("failed to allocate IP %v: %v", ip, err))}
				return allocated, errors.NewInvalid(api.Kind("Service"), service.Name, el)
			}
			allocated[family] = ip
		}
	}
	return allocated, nil
}

// releases clusterIPs per family
func (al *RESTAllocStuff) releaseIPs(toRelease map[api.IPFamily]string) (map[api.IPFamily]string, error) {
	if toRelease == nil {
		return nil, nil
	}

	released := make(map[api.IPFamily]string)
	for family, ip := range toRelease {
		allocator, ok := al.serviceIPAllocatorsByFamily[family]
		if !ok {
			// cluster was configured for dual stack, then single stack
			klog.V(4).Infof("delete service. Not releasing ClusterIP:%v because IPFamily:%v is no longer configured on server", ip, family)
			continue
		}

		parsedIP := netutils.ParseIPSloppy(ip)
		if err := allocator.Release(parsedIP); err != nil {
			return released, err
		}
		released[family] = ip
	}

	return released, nil
}

// standard allocator for dualstackgate==Off, hard wired dependency
// and ignores policy, families and clusterIPs
func (al *RESTAllocStuff) allocClusterIP(service *api.Service, dryRun bool) (map[api.IPFamily]string, error) {
	toAlloc := make(map[api.IPFamily]string)

	// get clusterIP.. empty string if user did not specify an ip
	toAlloc[al.defaultServiceIPFamily] = service.Spec.ClusterIP
	// alloc
	allocated, err := al.allocIPs(service, toAlloc, dryRun)

	// set
	if err == nil {
		service.Spec.ClusterIP = allocated[al.defaultServiceIPFamily]
		service.Spec.ClusterIPs = []string{allocated[al.defaultServiceIPFamily]}
	}

	return allocated, err
}

func (al *RESTAllocStuff) txnAllocClusterIPs(service *api.Service, dryRun bool) (transaction, error) {
	// clusterIPs that were allocated may need to be released in case of
	// failure at a higher level.
	toReleaseClusterIPs, err := al.allocClusterIPs(service, dryRun)
	if err != nil {
		return nil, err
	}

	txn := callbackTransaction{
		revert: func() {
			if dryRun {
				return
			}
			released, err := al.releaseIPs(toReleaseClusterIPs)
			if err != nil {
				klog.Warningf("failed to release clusterIPs for failed new service:%v allocated:%v released:%v error:%v",
					service.Name, toReleaseClusterIPs, released, err)
			}
		},
	}
	return txn, nil
}

// allocates ClusterIPs for a service
func (al *RESTAllocStuff) allocClusterIPs(service *api.Service, dryRun bool) (map[api.IPFamily]string, error) {
	// external name don't get ClusterIPs
	if service.Spec.Type == api.ServiceTypeExternalName {
		return nil, nil
	}

	// headless don't get ClusterIPs
	if len(service.Spec.ClusterIPs) > 0 && service.Spec.ClusterIPs[0] == api.ClusterIPNone {
		return nil, nil
	}

	if !utilfeature.DefaultFeatureGate.Enabled(features.IPv6DualStack) {
		return al.allocClusterIP(service, dryRun)
	}

	toAlloc := make(map[api.IPFamily]string)
	// at this stage, the only fact we know is that service has correct ip families
	// assigned to it. It may have partial assigned ClusterIPs (Upgrade to dual stack)
	// may have no ips at all. The below loop is meant to fix this
	// (we also know that this cluster has these families)

	// if there is no slice to work with
	if service.Spec.ClusterIPs == nil {
		service.Spec.ClusterIPs = make([]string, 0, len(service.Spec.IPFamilies))
	}

	for i, ipFamily := range service.Spec.IPFamilies {
		if i > (len(service.Spec.ClusterIPs) - 1) {
			service.Spec.ClusterIPs = append(service.Spec.ClusterIPs, "" /* just a marker */)
		}

		toAlloc[ipFamily] = service.Spec.ClusterIPs[i]
	}

	// allocate
	allocated, err := al.allocIPs(service, toAlloc, dryRun)

	// set if successful
	if err == nil {
		for family, ip := range allocated {
			for i, check := range service.Spec.IPFamilies {
				if family == check {
					service.Spec.ClusterIPs[i] = ip
					// while we technically don't need to do that testing rest does not
					// go through conversion logic but goes through validation *sigh*.
					// so we set ClusterIP here as well
					// because the testing code expects valid (as they are output-ed from conversion)
					// as it patches fields
					if i == 0 {
						service.Spec.ClusterIP = ip
					}
				}
			}
		}
	}

	return allocated, err
}

func (al *RESTAllocStuff) txnUpdateClusterIPs(service *api.Service, oldService *api.Service, dryRun bool) (transaction, error) {
	allocated, released, err := al.handleClusterIPsForUpdatedService(oldService, service, dryRun)
	if err != nil {
		return nil, err
	}

	// on failure: Any newly allocated IP must be released back
	// on failure: Any previously allocated IP that would have been released,
	//             must *not* be released
	// on success: Any previously allocated IP that should be released, will be
	//             released
	txn := callbackTransaction{
		commit: func() {
			if dryRun {
				return
			}
			if actuallyReleased, err := al.releaseIPs(released); err != nil {
				klog.V(4).Infof("service %v/%v failed to clean up after failed service update error:%v. ShouldRelease/Released:%v/%v",
					service.Namespace, service.Name, err, released, actuallyReleased)
			}
		},
		revert: func() {
			if dryRun {
				return
			}
			if actuallyReleased, err := al.releaseIPs(allocated); err != nil {
				klog.V(4).Infof("service %v/%v failed to clean up after failed service update error:%v. Allocated/Released:%v/%v",
					service.Namespace, service.Name, err, allocated, actuallyReleased)
			}
		},
	}
	return txn, nil
}

// handles type change/upgrade/downgrade change type for an update service
// this func does not perform actual release of clusterIPs. it returns
// a map[family]ip for the caller to release when everything else has
// executed successfully
func (al *RESTAllocStuff) handleClusterIPsForUpdatedService(oldService *api.Service, service *api.Service, dryRun bool) (allocated map[api.IPFamily]string, toRelease map[api.IPFamily]string, err error) {

	// We don't want to auto-upgrade (add an IP) or downgrade (remove an IP)
	// PreferDualStack services following a cluster change to/from
	// dual-stackness.
	//
	// That means a PreferDualStack service will only be upgraded/downgraded
	// when:
	// - changing ipFamilyPolicy to "RequireDualStack" or "SingleStack" AND
	// - adding or removing a secondary clusterIP or ipFamily
	if isMatchingPreferDualStackClusterIPFields(oldService, service) {
		return allocated, toRelease, nil // nothing more to do.
	}

	// use cases:
	// A: service changing types from ExternalName TO ClusterIP types ==> allocate all new
	// B: service changing types from ClusterIP types TO ExternalName ==> release all allocated
	// C: Service upgrading to dual stack  ==> partial allocation
	// D: service downgrading from dual stack ==> partial release

	// CASE A:
	// Update service from ExternalName to non-ExternalName, should initialize ClusterIP.
	if oldService.Spec.Type == api.ServiceTypeExternalName && service.Spec.Type != api.ServiceTypeExternalName {
		allocated, err := al.allocClusterIPs(service, dryRun)
		return allocated, nil, err
	}

	// CASE B:

	// if headless service then we bail out early (no clusterIPs management needed)
	if len(oldService.Spec.ClusterIPs) > 0 && oldService.Spec.ClusterIPs[0] == api.ClusterIPNone {
		return nil, nil, nil
	}

	// Update service from non-ExternalName to ExternalName, should release ClusterIP if exists.
	if oldService.Spec.Type != api.ServiceTypeExternalName && service.Spec.Type == api.ServiceTypeExternalName {
		toRelease = make(map[api.IPFamily]string)
		if !utilfeature.DefaultFeatureGate.Enabled(features.IPv6DualStack) {
			// for non dual stack enabled cluster we use clusterIPs
			toRelease[al.defaultServiceIPFamily] = oldService.Spec.ClusterIP
		} else {
			// dual stack is enabled, collect ClusterIPs by families
			for i, family := range oldService.Spec.IPFamilies {
				toRelease[family] = oldService.Spec.ClusterIPs[i]
			}
		}

		return nil, toRelease, nil
	}

	// upgrade and downgrade are specific to dualstack
	if !utilfeature.DefaultFeatureGate.Enabled(features.IPv6DualStack) {
		return nil, nil, nil
	}

	upgraded := len(oldService.Spec.IPFamilies) == 1 && len(service.Spec.IPFamilies) == 2
	downgraded := len(oldService.Spec.IPFamilies) == 2 && len(service.Spec.IPFamilies) == 1

	// CASE C:
	if upgraded {
		toAllocate := make(map[api.IPFamily]string)
		// if secondary ip was named, just get it. if not add a marker
		if len(service.Spec.ClusterIPs) < 2 {
			service.Spec.ClusterIPs = append(service.Spec.ClusterIPs, "" /* marker */)
		}

		toAllocate[service.Spec.IPFamilies[1]] = service.Spec.ClusterIPs[1]

		// allocate
		allocated, err := al.allocIPs(service, toAllocate, dryRun)
		// set if successful
		if err == nil {
			service.Spec.ClusterIPs[1] = allocated[service.Spec.IPFamilies[1]]
		}

		return allocated, nil, err
	}

	// CASE D:
	if downgraded {
		toRelease = make(map[api.IPFamily]string)
		toRelease[oldService.Spec.IPFamilies[1]] = oldService.Spec.ClusterIPs[1]
		// note: we don't release clusterIP, this is left to clean up in the action itself
		return nil, toRelease, err
	}
	// it was not an upgrade nor downgrade
	return nil, nil, nil
}

// for pre dual stack (gate == off). Hardwired to ClusterIP and ignores all new fields
func (al *RESTAllocStuff) releaseServiceClusterIP(service *api.Service) (released map[api.IPFamily]string, err error) {
	toRelease := make(map[api.IPFamily]string)

	// we need to do that to handle cases where allocator is no longer configured on
	// cluster
	if netutils.IsIPv6String(service.Spec.ClusterIP) {
		toRelease[api.IPv6Protocol] = service.Spec.ClusterIP
	} else {
		toRelease[api.IPv4Protocol] = service.Spec.ClusterIP
	}

	return al.releaseIPs(toRelease)
}

// releases allocated ClusterIPs for service that is about to be deleted
func (al *RESTAllocStuff) releaseServiceClusterIPs(service *api.Service) (released map[api.IPFamily]string, err error) {
	// external name don't get ClusterIPs
	if service.Spec.Type == api.ServiceTypeExternalName {
		return nil, nil
	}

	// headless don't get ClusterIPs
	if len(service.Spec.ClusterIPs) > 0 && service.Spec.ClusterIPs[0] == api.ClusterIPNone {
		return nil, nil
	}

	if !utilfeature.DefaultFeatureGate.Enabled(features.IPv6DualStack) {
		return al.releaseServiceClusterIP(service)
	}

	toRelease := make(map[api.IPFamily]string)
	for _, ip := range service.Spec.ClusterIPs {
		if netutils.IsIPv6String(ip) {
			toRelease[api.IPv6Protocol] = ip
		} else {
			toRelease[api.IPv4Protocol] = ip
		}
	}
	return al.releaseIPs(toRelease)
}

// tests if two preferred dual-stack service have matching ClusterIPFields
// assumption: old service is a valid, default service (e.g., loaded from store)
func isMatchingPreferDualStackClusterIPFields(oldService, service *api.Service) bool {
	if oldService == nil {
		return false
	}

	if service.Spec.IPFamilyPolicy == nil {
		return false
	}

	// if type mutated then it is an update
	// that needs to run through the entire process.
	if oldService.Spec.Type != service.Spec.Type {
		return false
	}
	// both must be type that gets an IP assigned
	if service.Spec.Type != api.ServiceTypeClusterIP &&
		service.Spec.Type != api.ServiceTypeNodePort &&
		service.Spec.Type != api.ServiceTypeLoadBalancer {
		return false
	}

	// both must be of IPFamilyPolicy==PreferDualStack
	if service.Spec.IPFamilyPolicy != nil && *(service.Spec.IPFamilyPolicy) != api.IPFamilyPolicyPreferDualStack {
		return false
	}

	if oldService.Spec.IPFamilyPolicy != nil && *(oldService.Spec.IPFamilyPolicy) != api.IPFamilyPolicyPreferDualStack {
		return false
	}

	if !sameClusterIPs(oldService, service) {
		return false
	}

	if !sameIPFamilies(oldService, service) {
		return false
	}

	// they match on
	// Policy: preferDualStack
	// ClusterIPs
	// IPFamilies
	return true
}

func sameClusterIPs(lhs, rhs *api.Service) bool {
	if len(rhs.Spec.ClusterIPs) != len(lhs.Spec.ClusterIPs) {
		return false
	}

	for i, ip := range rhs.Spec.ClusterIPs {
		if lhs.Spec.ClusterIPs[i] != ip {
			return false
		}
	}

	return true
}

func reducedClusterIPs(before, after *api.Service) bool {
	if len(after.Spec.ClusterIPs) == 0 { // Not specified
		return false
	}
	return len(after.Spec.ClusterIPs) < len(before.Spec.ClusterIPs)
}

func sameIPFamilies(lhs, rhs *api.Service) bool {
	if len(rhs.Spec.IPFamilies) != len(lhs.Spec.IPFamilies) {
		return false
	}

	for i, family := range rhs.Spec.IPFamilies {
		if lhs.Spec.IPFamilies[i] != family {
			return false
		}
	}

	return true
}

func reducedIPFamilies(before, after *api.Service) bool {
	if len(after.Spec.IPFamilies) == 0 { // Not specified
		return false
	}
	return len(after.Spec.IPFamilies) < len(before.Spec.IPFamilies)
}

// Helper to get the IP family of a given IP.
func familyOf(ip string) api.IPFamily {
	if netutils.IsIPv4String(ip) {
		return api.IPv4Protocol
	}
	if netutils.IsIPv6String(ip) {
		return api.IPv6Protocol
	}
	return api.IPFamily("unknown")
}

// Helper to avoid nil-checks all over.  Callers of this need to be checking
// for an exact value.
func getIPFamilyPolicy(svc *api.Service) api.IPFamilyPolicyType {
	if svc.Spec.IPFamilyPolicy == nil {
		return "" // callers need to handle this
	}
	return *svc.Spec.IPFamilyPolicy
}

// attempts to default service ip families according to cluster configuration
// while ensuring that provided families are configured on cluster.
func (al *RESTAllocStuff) initIPFamilyFields(oldService, service *api.Service) error {
	// can not do anything here
	if service.Spec.Type == api.ServiceTypeExternalName {
		return nil
	}

	// gate off. We don't need to validate or default new fields
	// we totally depend on existing validation in apis/validation
	if !utilfeature.DefaultFeatureGate.Enabled(features.IPv6DualStack) {
		return nil
	}

	// We don't want to auto-upgrade (add an IP) or downgrade (remove an IP)
	// PreferDualStack services following a cluster change to/from
	// dual-stackness.
	//
	// That means a PreferDualStack service will only be upgraded/downgraded
	// when:
	// - changing ipFamilyPolicy to "RequireDualStack" or "SingleStack" AND
	// - adding or removing a secondary clusterIP or ipFamily
	if isMatchingPreferDualStackClusterIPFields(oldService, service) {
		return nil // nothing more to do.
	}

	// If the user didn't specify ipFamilyPolicy, we can infer a default.  We
	// don't want a static default because we want to make sure that we never
	// change between single- and dual-stack modes with explicit direction, as
	// provided by ipFamilyPolicy.  Consider these cases:
	//   * Create (POST): If they didn't specify a policy we can assume it's
	//     always SingleStack.
	//   * Update (PUT): If they didn't specify a policy we need to adopt the
	//     policy from before.  This is better than always assuming SingleStack
	//     because a PUT that changes clusterIPs from 2 to 1 value but doesn't
	//     specify ipFamily would work.
	//   * Update (PATCH): If they didn't specify a policy it will adopt the
	//     policy from before.
	if service.Spec.IPFamilyPolicy == nil {
		if oldService != nil && oldService.Spec.IPFamilyPolicy != nil {
			// Update from an object with policy, use the old policy
			service.Spec.IPFamilyPolicy = oldService.Spec.IPFamilyPolicy
		} else if service.Spec.ClusterIP == api.ClusterIPNone && len(service.Spec.Selector) == 0 {
			// Special-case: headless + selectorless defaults to dual.
			requireDualStack := api.IPFamilyPolicyRequireDualStack
			service.Spec.IPFamilyPolicy = &requireDualStack
		} else {
			// create or update from an object without policy (e.g.
			// ExternalName) to one that needs policy
			singleStack := api.IPFamilyPolicySingleStack
			service.Spec.IPFamilyPolicy = &singleStack
		}
	}
	// Henceforth we can assume ipFamilyPolicy is set.

	// Do some loose pre-validation of the input.  This makes it easier in the
	// rest of allocation code to not have to consider corner cases.
	// TODO(thockin): when we tighten validation (e.g. to require IPs) we will
	// need a "strict" and a "loose" form of this.
	if el := validation.ValidateServiceClusterIPsRelatedFields(service); len(el) != 0 {
		return errors.NewInvalid(api.Kind("Service"), service.Name, el)
	}

	//TODO(thockin): Move this logic to validation?
	el := make(field.ErrorList, 0)

	// Update-only prep work.
	if oldService != nil {
		if getIPFamilyPolicy(service) == api.IPFamilyPolicySingleStack {
			// As long as ClusterIPs and IPFamilies have not changed, setting
			// the policy to single-stack is clear intent.
			// ClusterIPs[0] is immutable, so it is safe to keep.
			if sameClusterIPs(oldService, service) && len(service.Spec.ClusterIPs) > 1 {
				service.Spec.ClusterIPs = service.Spec.ClusterIPs[0:1]
			}
			if sameIPFamilies(oldService, service) && len(service.Spec.IPFamilies) > 1 {
				service.Spec.IPFamilies = service.Spec.IPFamilies[0:1]
			}
		} else {
			// If the policy is anything but single-stack AND they reduced these
			// fields, it's an error.  They need to specify policy.
			if reducedClusterIPs(oldService, service) {
				el = append(el, field.Invalid(field.NewPath("spec", "ipFamilyPolicy"), service.Spec.IPFamilyPolicy,
					"must be 'SingleStack' to release the secondary cluster IP"))
			}
			if reducedIPFamilies(oldService, service) {
				el = append(el, field.Invalid(field.NewPath("spec", "ipFamilyPolicy"), service.Spec.IPFamilyPolicy,
					"must be 'SingleStack' to release the secondary IP family"))
			}
		}
	}

	// Make sure ipFamilyPolicy makes sense for the provided ipFamilies and
	// clusterIPs.  Further checks happen below - after the special cases.
	if getIPFamilyPolicy(service) == api.IPFamilyPolicySingleStack {
		if len(service.Spec.ClusterIPs) == 2 {
			el = append(el, field.Invalid(field.NewPath("spec", "ipFamilyPolicy"), service.Spec.IPFamilyPolicy,
				"must be 'RequireDualStack' or 'PreferDualStack' when multiple cluster IPs are specified"))
		}
		if len(service.Spec.IPFamilies) == 2 {
			el = append(el, field.Invalid(field.NewPath("spec", "ipFamilyPolicy"), service.Spec.IPFamilyPolicy,
				"must be 'RequireDualStack' or 'PreferDualStack' when multiple IP families are specified"))
		}
	}

	// Infer IPFamilies[] from ClusterIPs[].  Further checks happen below,
	// after the special cases.
	for i, ip := range service.Spec.ClusterIPs {
		if ip == api.ClusterIPNone {
			break
		}

		// We previously validated that IPs are well-formed and that if an
		// ipFamilies[] entry exists it matches the IP.
		fam := familyOf(ip)

		// If the corresponding family is not specified, add it.
		if i >= len(service.Spec.IPFamilies) {
			// Families are checked more later, but this is a better error in
			// this specific case (indicating the user-provided IP, rather
			// than than the auto-assigned family).
			if _, found := al.serviceIPAllocatorsByFamily[fam]; !found {
				el = append(el, field.Invalid(field.NewPath("spec", "clusterIPs").Index(i), service.Spec.ClusterIPs,
					fmt.Sprintf("%s is not configured on this cluster", fam)))
			} else {
				// OK to infer.
				service.Spec.IPFamilies = append(service.Spec.IPFamilies, fam)
			}
		}
	}

	// If we have validation errors, bail out now so we don't make them worse.
	if len(el) > 0 {
		return errors.NewInvalid(api.Kind("Service"), service.Name, el)
	}

	// Special-case: headless + selectorless.  This has to happen before other
	// checks because it explicitly allows combinations of inputs that would
	// otherwise be errors.
	if service.Spec.ClusterIP == api.ClusterIPNone && len(service.Spec.Selector) == 0 {
		// If IPFamilies was not set by the user, start with the default
		// family.
		if len(service.Spec.IPFamilies) == 0 {
			service.Spec.IPFamilies = []api.IPFamily{al.defaultServiceIPFamily}
		}

		// this follows headful services. With one exception on a single stack
		// cluster the user is allowed to create headless services that has multi families
		// the validation allows it
		if len(service.Spec.IPFamilies) < 2 {
			if *(service.Spec.IPFamilyPolicy) != api.IPFamilyPolicySingleStack {
				// add the alt ipfamily
				if service.Spec.IPFamilies[0] == api.IPv4Protocol {
					service.Spec.IPFamilies = append(service.Spec.IPFamilies, api.IPv6Protocol)
				} else {
					service.Spec.IPFamilies = append(service.Spec.IPFamilies, api.IPv4Protocol)
				}
			}
		}

		// nothing more needed here
		return nil
	}

	//
	// Everything below this MUST happen *after* the above special cases.
	//

	// Demanding dual-stack on a non dual-stack cluster.
	if getIPFamilyPolicy(service) == api.IPFamilyPolicyRequireDualStack {
		if len(al.serviceIPAllocatorsByFamily) < 2 {
			el = append(el, field.Invalid(field.NewPath("spec", "ipFamilyPolicy"), service.Spec.IPFamilyPolicy,
				"this cluster is not configured for dual-stack services"))
		}
	}

	// If there is a family requested then it has to be configured on cluster.
	for i, ipFamily := range service.Spec.IPFamilies {
		if _, found := al.serviceIPAllocatorsByFamily[ipFamily]; !found {
			el = append(el, field.Invalid(field.NewPath("spec", "ipFamilies").Index(i), ipFamily, "not configured on this cluster"))
		}
	}

	// If we have validation errors, don't bother with the rest.
	if len(el) > 0 {
		return errors.NewInvalid(api.Kind("Service"), service.Name, el)
	}

	// nil families, gets cluster default
	if len(service.Spec.IPFamilies) == 0 {
		service.Spec.IPFamilies = []api.IPFamily{al.defaultServiceIPFamily}
	}

	// If this service is looking for dual-stack and this cluster does have two
	// families, append the missing family.
	if *(service.Spec.IPFamilyPolicy) != api.IPFamilyPolicySingleStack &&
		len(service.Spec.IPFamilies) == 1 &&
		len(al.serviceIPAllocatorsByFamily) == 2 {

		if service.Spec.IPFamilies[0] == api.IPv4Protocol {
			service.Spec.IPFamilies = append(service.Spec.IPFamilies, api.IPv6Protocol)
		} else if service.Spec.IPFamilies[0] == api.IPv6Protocol {
			service.Spec.IPFamilies = append(service.Spec.IPFamilies, api.IPv4Protocol)
		}
	}

	return nil
}

func isValidAddress(ctx context.Context, addr *api.EndpointAddress, pods rest.Getter) error {
	if addr.TargetRef == nil {
		return fmt.Errorf("Address has no target ref, skipping: %v", addr)
	}
	if genericapirequest.NamespaceValue(ctx) != addr.TargetRef.Namespace {
		return fmt.Errorf("Address namespace doesn't match context namespace")
	}
	obj, err := pods.Get(ctx, addr.TargetRef.Name, &metav1.GetOptions{})
	if err != nil {
		return err
	}
	pod, ok := obj.(*api.Pod)
	if !ok {
		return fmt.Errorf("failed to cast to pod: %v", obj)
	}
	if pod == nil {
		return fmt.Errorf("pod is missing, skipping (%s/%s)", addr.TargetRef.Namespace, addr.TargetRef.Name)
	}
	for _, podIP := range pod.Status.PodIPs {
		if podIP.IP == addr.IP {
			return nil
		}
	}
	return fmt.Errorf("pod ip(s) doesn't match endpoint ip, skipping: %v vs %s (%s/%s)", pod.Status.PodIPs, addr.IP, addr.TargetRef.Namespace, addr.TargetRef.Name)
}

// This is O(N), but we expect haystack to be small;
// so small that we expect a linear search to be faster
func containsNumber(haystack []int, needle int) bool {
	for _, v := range haystack {
		if v == needle {
			return true
		}
	}
	return false
}

// This is O(N), but we expect serviceNodePorts to be small;
// so small that we expect a linear search to be faster
func containsNodePort(serviceNodePorts []ServiceNodePort, serviceNodePort ServiceNodePort) bool {
	for _, snp := range serviceNodePorts {
		if snp == serviceNodePort {
			return true
		}
	}
	return false
}

// Loop through the service ports list, find one with the same port number and
// NodePort specified, return this NodePort otherwise return 0.
func findRequestedNodePort(port int, servicePorts []api.ServicePort) int {
	for i := range servicePorts {
		servicePort := servicePorts[i]
		if port == int(servicePort.Port) && servicePort.NodePort != 0 {
			return int(servicePort.NodePort)
		}
	}
	return 0
}

// allocateHealthCheckNodePort allocates health check node port to service.
func allocateHealthCheckNodePort(service *api.Service, nodePortOp *portallocator.PortAllocationOperation) error {
	healthCheckNodePort := service.Spec.HealthCheckNodePort
	if healthCheckNodePort != 0 {
		// If the request has a health check nodePort in mind, attempt to reserve it.
		err := nodePortOp.Allocate(int(healthCheckNodePort))
		if err != nil {
			return fmt.Errorf("failed to allocate requested HealthCheck NodePort %v: %v",
				healthCheckNodePort, err)
		}
		klog.V(4).Infof("Reserved user requested healthCheckNodePort: %d", healthCheckNodePort)
	} else {
		// If the request has no health check nodePort specified, allocate any.
		healthCheckNodePort, err := nodePortOp.AllocateNext()
		if err != nil {
			return fmt.Errorf("failed to allocate a HealthCheck NodePort %v: %v", healthCheckNodePort, err)
		}
		service.Spec.HealthCheckNodePort = int32(healthCheckNodePort)
		klog.V(4).Infof("Reserved allocated healthCheckNodePort: %d", healthCheckNodePort)
	}
	return nil
}

func (al *RESTAllocStuff) txnAllocNodePorts(service *api.Service, dryRun bool) (transaction, error) {
	// The allocator tracks dry-run-ness internally.
	nodePortOp := portallocator.StartOperation(al.serviceNodePorts, dryRun)

	txn := callbackTransaction{
		commit: func() {
			nodePortOp.Commit()
			// We don't NEED to call Finish() here, but for that package says
			// to, so for future-safety, we will.
			nodePortOp.Finish()
		},
		revert: func() {
			// Weirdly named but this will revert if commit wasn't called
			nodePortOp.Finish()
		},
	}

	// Allocate NodePorts, if needed.
	if service.Spec.Type == api.ServiceTypeNodePort || service.Spec.Type == api.ServiceTypeLoadBalancer {
		if err := initNodePorts(service, nodePortOp); err != nil {
			txn.Revert()
			return nil, err
		}
	}

	// Handle ExternalTraffic related fields during service creation.
	if apiservice.NeedsHealthCheck(service) {
		if err := allocateHealthCheckNodePort(service, nodePortOp); err != nil {
			txn.Revert()
			return nil, errors.NewInternalError(err)
		}
	}

	return txn, nil
}

func initNodePorts(service *api.Service, nodePortOp *portallocator.PortAllocationOperation) error {
	svcPortToNodePort := map[int]int{}
	for i := range service.Spec.Ports {
		servicePort := &service.Spec.Ports[i]
		if servicePort.NodePort == 0 && !shouldAllocateNodePorts(service) {
			// Don't allocate new ports, but do respect specific requests.
			continue
		}
		allocatedNodePort := svcPortToNodePort[int(servicePort.Port)]
		if allocatedNodePort == 0 {
			// This will only scan forward in the service.Spec.Ports list because any matches
			// before the current port would have been found in svcPortToNodePort. This is really
			// looking for any user provided values.
			np := findRequestedNodePort(int(servicePort.Port), service.Spec.Ports)
			if np != 0 {
				err := nodePortOp.Allocate(np)
				if err != nil {
					// TODO: when validation becomes versioned, this gets more complicated.
					el := field.ErrorList{field.Invalid(field.NewPath("spec", "ports").Index(i).Child("nodePort"), np, err.Error())}
					return errors.NewInvalid(api.Kind("Service"), service.Name, el)
				}
				servicePort.NodePort = int32(np)
				svcPortToNodePort[int(servicePort.Port)] = np
			} else {
				nodePort, err := nodePortOp.AllocateNext()
				if err != nil {
					// TODO: what error should be returned here?  It's not a
					// field-level validation failure (the field is valid), and it's
					// not really an internal error.
					return errors.NewInternalError(fmt.Errorf("failed to allocate a nodePort: %v", err))
				}
				servicePort.NodePort = int32(nodePort)
				svcPortToNodePort[int(servicePort.Port)] = nodePort
			}
		} else if int(servicePort.NodePort) != allocatedNodePort {
			// TODO(xiangpengzhao): do we need to allocate a new NodePort in this case?
			// Note: the current implementation is better, because it saves a NodePort.
			if servicePort.NodePort == 0 {
				servicePort.NodePort = int32(allocatedNodePort)
			} else {
				err := nodePortOp.Allocate(int(servicePort.NodePort))
				if err != nil {
					// TODO: when validation becomes versioned, this gets more complicated.
					el := field.ErrorList{field.Invalid(field.NewPath("spec", "ports").Index(i).Child("nodePort"), servicePort.NodePort, err.Error())}
					return errors.NewInvalid(api.Kind("Service"), service.Name, el)
				}
			}
		}
	}

	return nil
}

func updateNodePorts(oldService, newService *api.Service, nodePortOp *portallocator.PortAllocationOperation) error {
	oldNodePortsNumbers := collectServiceNodePorts(oldService)
	newNodePorts := []ServiceNodePort{}
	portAllocated := map[int]bool{}

	for i := range newService.Spec.Ports {
		servicePort := &newService.Spec.Ports[i]
		if servicePort.NodePort == 0 && !shouldAllocateNodePorts(newService) {
			// Don't allocate new ports, but do respect specific requests.
			continue
		}
		nodePort := ServiceNodePort{Protocol: servicePort.Protocol, NodePort: servicePort.NodePort}
		if nodePort.NodePort != 0 {
			if !containsNumber(oldNodePortsNumbers, int(nodePort.NodePort)) && !portAllocated[int(nodePort.NodePort)] {
				err := nodePortOp.Allocate(int(nodePort.NodePort))
				if err != nil {
					el := field.ErrorList{field.Invalid(field.NewPath("spec", "ports").Index(i).Child("nodePort"), nodePort.NodePort, err.Error())}
					return errors.NewInvalid(api.Kind("Service"), newService.Name, el)
				}
				portAllocated[int(nodePort.NodePort)] = true
			}
		} else {
			nodePortNumber, err := nodePortOp.AllocateNext()
			if err != nil {
				// TODO: what error should be returned here?  It's not a
				// field-level validation failure (the field is valid), and it's
				// not really an internal error.
				return errors.NewInternalError(fmt.Errorf("failed to allocate a nodePort: %v", err))
			}
			servicePort.NodePort = int32(nodePortNumber)
			nodePort.NodePort = servicePort.NodePort
		}
		if containsNodePort(newNodePorts, nodePort) {
			return fmt.Errorf("duplicate nodePort: %v", nodePort)
		}
		newNodePorts = append(newNodePorts, nodePort)
	}

	newNodePortsNumbers := collectServiceNodePorts(newService)

	// The comparison loops are O(N^2), but we don't expect N to be huge
	// (there's a hard-limit at 2^16, because they're ports; and even 4 ports would be a lot)
	for _, oldNodePortNumber := range oldNodePortsNumbers {
		if containsNumber(newNodePortsNumbers, oldNodePortNumber) {
			continue
		}
		nodePortOp.ReleaseDeferred(int(oldNodePortNumber))
	}

	return nil
}

func releaseNodePorts(service *api.Service, nodePortOp *portallocator.PortAllocationOperation) {
	nodePorts := collectServiceNodePorts(service)

	for _, nodePort := range nodePorts {
		nodePortOp.ReleaseDeferred(nodePort)
	}
}

func collectServiceNodePorts(service *api.Service) []int {
	servicePorts := []int{}
	for i := range service.Spec.Ports {
		servicePort := &service.Spec.Ports[i]
		if servicePort.NodePort != 0 {
			servicePorts = append(servicePorts, int(servicePort.NodePort))
		}
	}
	return servicePorts
}
