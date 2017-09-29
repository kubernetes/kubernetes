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
	"net/http"
	"strconv"
	"strings"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	apiservice "k8s.io/kubernetes/pkg/api/v1/service"
	"k8s.io/kubernetes/pkg/cloudprovider"
	netsets "k8s.io/kubernetes/pkg/util/net/sets"

	"github.com/golang/glog"
	computealpha "google.golang.org/api/compute/v0.alpha"
	compute "google.golang.org/api/compute/v1"
)

// ensureExternalLoadBalancer is the external implementation of LoadBalancer.EnsureLoadBalancer.
// Our load balancers in GCE consist of four separate GCE resources - a static
// IP address, a firewall rule, a target pool, and a forwarding rule. This
// function has to manage all of them.
//
// Due to an interesting series of design decisions, this handles both creating
// new load balancers and updating existing load balancers, recognizing when
// each is needed.
func (gce *GCECloud) ensureExternalLoadBalancer(clusterName, clusterID string, apiService *v1.Service, existingFwdRule *compute.ForwardingRule, nodes []*v1.Node) (*v1.LoadBalancerStatus, error) {
	if len(nodes) == 0 {
		return nil, fmt.Errorf("Cannot EnsureLoadBalancer() with no hosts")
	}

	hostNames := nodeNames(nodes)
	supportsNodesHealthCheck := supportsNodesHealthCheck(nodes)
	hosts, err := gce.getInstancesByNames(hostNames)
	if err != nil {
		return nil, err
	}

	loadBalancerName := cloudprovider.GetLoadBalancerName(apiService)
	requestedIP := apiService.Spec.LoadBalancerIP
	ports := apiService.Spec.Ports
	portStr := []string{}
	for _, p := range apiService.Spec.Ports {
		portStr = append(portStr, fmt.Sprintf("%s/%d", p.Protocol, p.Port))
	}

	affinityType := apiService.Spec.SessionAffinity

	serviceName := types.NamespacedName{Namespace: apiService.Namespace, Name: apiService.Name}
	glog.V(2).Infof("EnsureLoadBalancer(%v, %v, %v, %v, %v, %v, %v)",
		loadBalancerName, gce.region, requestedIP, portStr, hostNames, serviceName, apiService.Annotations)

	lbRefStr := fmt.Sprintf("%v(%v)", loadBalancerName, serviceName)
	// Check the current and the desired network tiers. If they do not match,
	// tear down the existing resources with the wrong tier.
	netTier, err := gce.getServiceNetworkTier(apiService)
	if err != nil {
		glog.Errorf("EnsureLoadBalancer(%s): failed to get the desired network tier: %v", lbRefStr, err)
		return nil, err
	}
	glog.V(4).Infof("EnsureLoadBalancer(%s): desired network tier %q ", lbRefStr, netTier)
	if gce.AlphaFeatureGate.Enabled(AlphaFeatureNetworkTiers) {
		gce.deleteWrongNetworkTieredResources(loadBalancerName, lbRefStr, netTier)
	}

	// Check if the forwarding rule exists, and if so, what its IP is.
	fwdRuleExists, fwdRuleNeedsUpdate, fwdRuleIP, err := gce.forwardingRuleNeedsUpdate(loadBalancerName, gce.region, requestedIP, ports)
	if err != nil {
		return nil, err
	}
	if !fwdRuleExists {
		glog.V(2).Infof("Forwarding rule %v for Service %v/%v doesn't exist",
			loadBalancerName, apiService.Namespace, apiService.Name)
	}

	// Make sure we know which IP address will be used and have properly reserved
	// it as static before moving forward with the rest of our operations.
	//
	// We use static IP addresses when updating a load balancer to ensure that we
	// can replace the load balancer's other components without changing the
	// address its service is reachable on. We do it this way rather than always
	// keeping the static IP around even though this is more complicated because
	// it makes it less likely that we'll run into quota issues. Only 7 static
	// IP addresses are allowed per region by default.
	//
	// We could let an IP be allocated for us when the forwarding rule is created,
	// but we need the IP to set up the firewall rule, and we want to keep the
	// forwarding rule creation as the last thing that needs to be done in this
	// function in order to maintain the invariant that "if the forwarding rule
	// exists, the LB has been fully created".
	ipAddressToUse := ""

	// Through this process we try to keep track of whether it is safe to
	// release the IP that was allocated.  If the user specifically asked for
	// an IP, we assume they are managing it themselves.  Otherwise, we will
	// release the IP in case of early-terminating failure or upon successful
	// creating of the LB.
	// TODO(#36535): boil this logic down into a set of component functions
	// and key the flag values off of errors returned.
	isUserOwnedIP := false // if this is set, we never release the IP
	isSafeToReleaseIP := false
	defer func() {
		if isUserOwnedIP {
			return
		}
		if isSafeToReleaseIP {
			if err := gce.DeleteRegionAddress(loadBalancerName, gce.region); err != nil && !isNotFound(err) {
				glog.Errorf("Failed to release static IP %s for load balancer (%v(%v), %v): %v", ipAddressToUse, loadBalancerName, serviceName, gce.region, err)
			} else if isNotFound(err) {
				glog.V(2).Infof("EnsureLoadBalancer(%v(%v)): address %s is not reserved.", loadBalancerName, serviceName, ipAddressToUse)
			} else {
				glog.V(2).Infof("EnsureLoadBalancer(%v(%v)): released static IP %s", loadBalancerName, serviceName, ipAddressToUse)
			}
		} else {
			glog.Warningf("orphaning static IP %s during update of load balancer (%v(%v), %v): %v", ipAddressToUse, loadBalancerName, serviceName, gce.region, err)
		}
	}()

	if requestedIP != "" {
		// If user requests a specific IP address, verify first. No mutation to
		// the GCE resources will be performed in the verification process.
		isUserOwnedIP, err = verifyUserRequestedIP(gce, gce.region, requestedIP, fwdRuleIP, lbRefStr, netTier)
		if err != nil {
			return nil, err
		}
		ipAddressToUse = requestedIP
	}

	if !isUserOwnedIP {
		// If we are not using the user-owned IP, either promote the
		// emphemeral IP used by the fwd rule, or create a new static IP.
		ipAddr, existed, err := ensureStaticIP(gce, loadBalancerName, serviceName.String(), gce.region, fwdRuleIP, netTier)
		if err != nil {
			return nil, fmt.Errorf("failed to ensure a static IP for the LB: %v", err)
		}
		glog.V(4).Infof("EnsureLoadBalancer(%s): ensured IP address %s (tier: %s)", lbRefStr, ipAddr, netTier)
		// If the IP was not owned by the user, but it already existed, it
		// could indicate that the previous update cycle failed. We can use
		// this IP and try to run through the process again, but we should
		// not release the IP unless it is explicitly flagged as OK.
		isSafeToReleaseIP = !existed
		ipAddressToUse = ipAddr
	}

	// Deal with the firewall next. The reason we do this here rather than last
	// is because the forwarding rule is used as the indicator that the load
	// balancer is fully created - it's what getLoadBalancer checks for.
	// Check if user specified the allow source range
	sourceRanges, err := apiservice.GetLoadBalancerSourceRanges(apiService)
	if err != nil {
		return nil, err
	}

	firewallExists, firewallNeedsUpdate, err := gce.firewallNeedsUpdate(loadBalancerName, serviceName.String(), gce.region, ipAddressToUse, ports, sourceRanges)
	if err != nil {
		return nil, err
	}

	if firewallNeedsUpdate {
		desc := makeFirewallDescription(serviceName.String(), ipAddressToUse)
		// Unlike forwarding rules and target pools, firewalls can be updated
		// without needing to be deleted and recreated.
		if firewallExists {
			glog.Infof("EnsureLoadBalancer(%v(%v)): updating firewall", loadBalancerName, serviceName)
			if err := gce.updateFirewall(apiService, MakeFirewallName(loadBalancerName), gce.region, desc, sourceRanges, ports, hosts); err != nil {
				return nil, err
			}
			glog.Infof("EnsureLoadBalancer(%v(%v)): updated firewall", loadBalancerName, serviceName)
		} else {
			glog.Infof("EnsureLoadBalancer(%v(%v)): creating firewall", loadBalancerName, serviceName)
			if err := gce.createFirewall(apiService, MakeFirewallName(loadBalancerName), gce.region, desc, sourceRanges, ports, hosts); err != nil {
				return nil, err
			}
			glog.Infof("EnsureLoadBalancer(%v(%v)): created firewall", loadBalancerName, serviceName)
		}
	}

	tpExists, tpNeedsUpdate, err := gce.targetPoolNeedsUpdate(loadBalancerName, gce.region, affinityType)
	if err != nil {
		return nil, err
	}
	if !tpExists {
		glog.Infof("Target pool %v for Service %v/%v doesn't exist", loadBalancerName, apiService.Namespace, apiService.Name)
	}

	// Check which health check needs to create and which health check needs to delete.
	// Health check management is coupled with target pool operation to prevent leaking.
	var hcToCreate, hcToDelete *compute.HttpHealthCheck
	hcLocalTrafficExisting, err := gce.GetHttpHealthCheck(loadBalancerName)
	if err != nil && !isHTTPErrorCode(err, http.StatusNotFound) {
		return nil, fmt.Errorf("error checking HTTP health check %s: %v", loadBalancerName, err)
	}
	if path, healthCheckNodePort := apiservice.GetServiceHealthCheckPathPort(apiService); path != "" {
		glog.V(4).Infof("service %v (%v) needs local traffic health checks on: %d%s)", apiService.Name, loadBalancerName, healthCheckNodePort, path)
		if hcLocalTrafficExisting == nil {
			// This logic exists to detect a transition for non-OnlyLocal to OnlyLocal service
			// turn on the tpNeedsUpdate flag to delete/recreate fwdrule/tpool updating the
			// target pool to use local traffic health check.
			glog.V(2).Infof("Updating from nodes health checks to local traffic health checks for service %v LB %v", apiService.Name, loadBalancerName)
			if supportsNodesHealthCheck {
				hcToDelete = makeHttpHealthCheck(MakeNodesHealthCheckName(clusterID), GetNodesHealthCheckPath(), GetNodesHealthCheckPort())
			}
			tpNeedsUpdate = true
		}
		hcToCreate = makeHttpHealthCheck(loadBalancerName, path, healthCheckNodePort)
	} else {
		glog.V(4).Infof("Service %v needs nodes health checks.", apiService.Name)
		if hcLocalTrafficExisting != nil {
			// This logic exists to detect a transition from OnlyLocal to non-OnlyLocal service
			// and turn on the tpNeedsUpdate flag to delete/recreate fwdrule/tpool updating the
			// target pool to use nodes health check.
			glog.V(2).Infof("Updating from local traffic health checks to nodes health checks for service %v LB %v", apiService.Name, loadBalancerName)
			hcToDelete = hcLocalTrafficExisting
			tpNeedsUpdate = true
		}
		if supportsNodesHealthCheck {
			hcToCreate = makeHttpHealthCheck(MakeNodesHealthCheckName(clusterID), GetNodesHealthCheckPath(), GetNodesHealthCheckPort())
		}
	}
	// Now we get to some slightly more interesting logic.
	// First, neither target pools nor forwarding rules can be updated in place -
	// they have to be deleted and recreated.
	// Second, forwarding rules are layered on top of target pools in that you
	// can't delete a target pool that's currently in use by a forwarding rule.
	// Thus, we have to tear down the forwarding rule if either it or the target
	// pool needs to be updated.
	if fwdRuleExists && (fwdRuleNeedsUpdate || tpNeedsUpdate) {
		// Begin critical section. If we have to delete the forwarding rule,
		// and something should fail before we recreate it, don't release the
		// IP.  That way we can come back to it later.
		isSafeToReleaseIP = false
		if err := gce.DeleteRegionForwardingRule(loadBalancerName, gce.region); err != nil && !isNotFound(err) {
			return nil, fmt.Errorf("failed to delete existing forwarding rule %s for load balancer update: %v", loadBalancerName, err)
		}
		glog.Infof("EnsureLoadBalancer(%v(%v)): deleted forwarding rule", loadBalancerName, serviceName)
	}
	if tpExists && tpNeedsUpdate {
		// Pass healthchecks to DeleteExternalTargetPoolAndChecks to cleanup health checks after cleaning up the target pool itself.
		var hcNames []string
		if hcToDelete != nil {
			hcNames = append(hcNames, hcToDelete.Name)
		}
		if err := gce.DeleteExternalTargetPoolAndChecks(apiService, loadBalancerName, gce.region, clusterID, hcNames...); err != nil {
			return nil, fmt.Errorf("failed to delete existing target pool %s for load balancer update: %v", loadBalancerName, err)
		}
		glog.Infof("EnsureLoadBalancer(%v(%v)): deleted target pool", loadBalancerName, serviceName)
	}

	// Once we've deleted the resources (if necessary), build them back up (or for
	// the first time if they're new).
	if tpNeedsUpdate {
		createInstances := hosts
		if len(hosts) > maxTargetPoolCreateInstances {
			createInstances = createInstances[:maxTargetPoolCreateInstances]
		}
		// Pass healthchecks to createTargetPool which needs them as health check links in the target pool
		if err := gce.createTargetPool(apiService, loadBalancerName, serviceName.String(), ipAddressToUse, gce.region, clusterID, createInstances, affinityType, hcToCreate); err != nil {
			return nil, fmt.Errorf("failed to create target pool %s: %v", loadBalancerName, err)
		}
		if hcToCreate != nil {
			glog.Infof("EnsureLoadBalancer(%v(%v)): created health checks %v for target pool", loadBalancerName, serviceName, hcToCreate.Name)
		}
		if len(hosts) <= maxTargetPoolCreateInstances {
			glog.Infof("EnsureLoadBalancer(%v(%v)): created target pool", loadBalancerName, serviceName)
		} else {
			glog.Infof("EnsureLoadBalancer(%v(%v)): created initial target pool (now updating with %d hosts)", loadBalancerName, serviceName, len(hosts)-maxTargetPoolCreateInstances)

			created := sets.NewString()
			for _, host := range createInstances {
				created.Insert(host.makeComparableHostPath())
			}
			if err := gce.updateTargetPool(loadBalancerName, created, hosts); err != nil {
				return nil, fmt.Errorf("failed to update target pool %s: %v", loadBalancerName, err)
			}
			glog.Infof("EnsureLoadBalancer(%v(%v)): updated target pool (with %d hosts)", loadBalancerName, serviceName, len(hosts)-maxTargetPoolCreateInstances)
		}
	}
	if tpNeedsUpdate || fwdRuleNeedsUpdate {
		glog.Infof("EnsureLoadBalancer(%v(%v)): creating forwarding rule, IP %s (tier: %s)", loadBalancerName, serviceName, ipAddressToUse, netTier)
		if err := createForwardingRule(gce, loadBalancerName, serviceName.String(), gce.region, ipAddressToUse, gce.targetPoolURL(loadBalancerName), ports, netTier); err != nil {
			return nil, fmt.Errorf("failed to create forwarding rule %s: %v", loadBalancerName, err)
		}
		// End critical section.  It is safe to release the static IP (which
		// just demotes it to ephemeral) now that it is attached.  In the case
		// of a user-requested IP, the "is user-owned" flag will be set,
		// preventing it from actually being released.
		isSafeToReleaseIP = true
		glog.Infof("EnsureLoadBalancer(%v(%v)): created forwarding rule, IP %s", loadBalancerName, serviceName, ipAddressToUse)
	}

	status := &v1.LoadBalancerStatus{}
	status.Ingress = []v1.LoadBalancerIngress{{IP: ipAddressToUse}}

	return status, nil
}

// updateExternalLoadBalancer is the external implementation of LoadBalancer.UpdateLoadBalancer.
func (gce *GCECloud) updateExternalLoadBalancer(clusterName string, service *v1.Service, nodes []*v1.Node) error {
	hosts, err := gce.getInstancesByNames(nodeNames(nodes))
	if err != nil {
		return err
	}

	loadBalancerName := cloudprovider.GetLoadBalancerName(service)
	pool, err := gce.service.TargetPools.Get(gce.projectID, gce.region, loadBalancerName).Do()
	if err != nil {
		return err
	}
	existing := sets.NewString()
	for _, instance := range pool.Instances {
		existing.Insert(hostURLToComparablePath(instance))
	}

	return gce.updateTargetPool(loadBalancerName, existing, hosts)
}

// ensureExternalLoadBalancerDeleted is the external implementation of LoadBalancer.EnsureLoadBalancerDeleted
func (gce *GCECloud) ensureExternalLoadBalancerDeleted(clusterName, clusterID string, service *v1.Service) error {
	loadBalancerName := cloudprovider.GetLoadBalancerName(service)

	var hcNames []string
	if path, _ := apiservice.GetServiceHealthCheckPathPort(service); path != "" {
		hcToDelete, err := gce.GetHttpHealthCheck(loadBalancerName)
		if err != nil && !isHTTPErrorCode(err, http.StatusNotFound) {
			glog.Infof("Failed to retrieve health check %v:%v", loadBalancerName, err)
			return err
		}
		hcNames = append(hcNames, hcToDelete.Name)
	} else {
		// EnsureLoadBalancerDeleted() could be triggered by changing service from
		// LoadBalancer type to others. In this case we have no idea whether it was
		// using local traffic health check or nodes health check. Attempt to delete
		// both to prevent leaking.
		hcNames = append(hcNames, loadBalancerName)
		hcNames = append(hcNames, MakeNodesHealthCheckName(clusterID))
	}

	errs := utilerrors.AggregateGoroutines(
		func() error {
			fwName := MakeFirewallName(loadBalancerName)
			err := ignoreNotFound(gce.DeleteFirewall(fwName))
			if isForbidden(err) && gce.OnXPN() {
				glog.V(4).Infof("ensureExternalLoadBalancerDeleted(%v): do not have permission to delete firewall rule (on XPN). Raising event.", loadBalancerName)
				gce.raiseFirewallChangeNeededEvent(service, FirewallToGCloudDeleteCmd(fwName, gce.NetworkProjectID()))
				return nil
			}
			return err
		},
		// Even though we don't hold on to static IPs for load balancers, it's
		// possible that EnsureLoadBalancer left one around in a failed
		// creation/update attempt, so make sure we clean it up here just in case.
		func() error { return ignoreNotFound(gce.DeleteRegionAddress(loadBalancerName, gce.region)) },
		func() error {
			// The forwarding rule must be deleted before either the target pool can,
			// unfortunately, so we have to do these two serially.
			if err := ignoreNotFound(gce.DeleteRegionForwardingRule(loadBalancerName, gce.region)); err != nil {
				return err
			}
			if err := gce.DeleteExternalTargetPoolAndChecks(service, loadBalancerName, gce.region, clusterID, hcNames...); err != nil {
				return err
			}
			return nil
		},
	)
	if errs != nil {
		return utilerrors.Flatten(errs)
	}
	return nil
}

func (gce *GCECloud) DeleteExternalTargetPoolAndChecks(service *v1.Service, name, region, clusterID string, hcNames ...string) error {
	if err := gce.DeleteTargetPool(name, region); err != nil && isHTTPErrorCode(err, http.StatusNotFound) {
		glog.Infof("Target pool %s already deleted. Continuing to delete other resources.", name)
	} else if err != nil {
		glog.Warningf("Failed to delete target pool %s, got error %s.", name, err.Error())
		return err
	}

	// Deletion of health checks is allowed only after the TargetPool reference is deleted
	for _, hcName := range hcNames {
		if err := func() error {
			// Check whether it is nodes health check, which has different name from the load-balancer.
			isNodesHealthCheck := hcName != name
			if isNodesHealthCheck {
				// Lock to prevent deleting necessary nodes health check before it gets attached
				// to target pool.
				gce.sharedResourceLock.Lock()
				defer gce.sharedResourceLock.Unlock()
			}
			glog.Infof("Deleting health check %v", hcName)
			if err := gce.DeleteHttpHealthCheck(hcName); err != nil {
				// Delete nodes health checks will fail if any other target pool is using it.
				if isInUsedByError(err) {
					glog.V(4).Infof("Health check %v is in used: %v.", hcName, err)
					return nil
				} else if !isHTTPErrorCode(err, http.StatusNotFound) {
					glog.Warningf("Failed to delete health check %v: %v", hcName, err)
					return err
				}
				// StatusNotFound could happen when:
				// - This is the first attempt but we pass in a healthcheck that is already deleted
				//   to prevent leaking.
				// - This is the first attempt but user manually deleted the heathcheck.
				// - This is a retry and in previous round we failed to delete the healthcheck firewall
				//   after deleted the healthcheck.
				// We continue to delete the healthcheck firewall to prevent leaking.
				glog.V(4).Infof("Health check %v is already deleted.", hcName)
			}
			// If health check is deleted without error, it means no load-balancer is using it.
			// So we should delete the health check firewall as well.
			fwName := MakeHealthCheckFirewallName(clusterID, hcName, isNodesHealthCheck)
			glog.Infof("Deleting firewall %v.", fwName)
			if err := ignoreNotFound(gce.DeleteFirewall(fwName)); err != nil {
				if isForbidden(err) && gce.OnXPN() {
					glog.V(4).Infof("DeleteExternalTargetPoolAndChecks(%v): do not have permission to delete firewall rule (on XPN). Raising event.", hcName)
					gce.raiseFirewallChangeNeededEvent(service, FirewallToGCloudDeleteCmd(fwName, gce.NetworkProjectID()))
					return nil
				}
				return err
			}
			return nil
		}(); err != nil {
			return err
		}
	}

	return nil
}

// verifyUserRequestedIP checks the user-provided IP to see whether it meets
// all the expected attributes for the load balancer, and returns an error if
// the verification failed. It also returns a boolean to indicate whether the
// IP address is considered owned by the user (i.e., not managed by the
// controller.
func verifyUserRequestedIP(s CloudAddressService, region, requestedIP, fwdRuleIP, lbRef string, desiredNetTier NetworkTier) (isUserOwnedIP bool, err error) {
	if requestedIP == "" {
		return false, nil
	}
	// If a specific IP address has been requested, we have to respect the
	// user's request and use that IP. If the forwarding rule was already using
	// a different IP, it will be harmlessly abandoned because it was only an
	// ephemeral IP (or it was a different static IP owned by the user, in which
	// case we shouldn't delete it anyway).
	existingAddress, err := s.GetRegionAddressByIP(region, requestedIP)
	if err != nil && !isNotFound(err) {
		glog.Errorf("verifyUserRequestedIP: failed to check whether the requested IP %q for LB %s exists: %v", requestedIP, lbRef, err)
		return false, err
	}
	if err == nil {
		// The requested IP is a static IP, owned and managed by the user.

		// Check if the network tier of the static IP matches the desired
		// network tier.
		netTierStr, err := s.getNetworkTierFromAddress(existingAddress.Name, region)
		if err != nil {
			return false, fmt.Errorf("failed to check the network tier of the IP %q: %v", requestedIP, err)
		}
		netTier := NetworkTierGCEValueToType(netTierStr)
		if netTier != desiredNetTier {
			glog.Errorf("verifyUserRequestedIP: requested static IP %q (name: %s) for LB %s has network tier %s, need %s.", requestedIP, existingAddress.Name, lbRef, netTier, desiredNetTier)
			return false, fmt.Errorf("requrested IP %q belongs to the %s network tier; expected %s", requestedIP, netTier, desiredNetTier)
		}
		glog.V(4).Infof("verifyUserRequestedIP: the requested static IP %q (name: %s, tier: %s) for LB %s exists.", requestedIP, existingAddress.Name, netTier, lbRef)
		return true, nil
	}
	if requestedIP == fwdRuleIP {
		// The requested IP is not a static IP, but is currently assigned
		// to this forwarding rule, so we can just use it.
		glog.V(4).Infof("verifyUserRequestedIP: the requested IP %q is not static, but is currently in use by for LB %s", requestedIP, lbRef)
		return false, nil
	}
	// The requested IP is not static and it is not assigned to the
	// current forwarding rule.  It might be attached to a different
	// rule or it might not be part of this project at all.  Either
	// way, we can't use it.
	glog.Errorf("verifyUserRequestedIP: requested IP %q for LB %s is neither static nor assigned to the LB", requestedIP, lbRef)
	return false, fmt.Errorf("requested ip %q is neither static nor assigned to the LB", requestedIP)
}

func (gce *GCECloud) createTargetPool(svc *v1.Service, name, serviceName, ipAddress, region, clusterID string, hosts []*gceInstance, affinityType v1.ServiceAffinity, hc *compute.HttpHealthCheck) error {
	// health check management is coupled with targetPools to prevent leaks. A
	// target pool is the only thing that requires a health check, so we delete
	// associated checks on teardown, and ensure checks on setup.
	hcLinks := []string{}
	if hc != nil {
		// Check whether it is nodes health check, which has different name from the load-balancer.
		isNodesHealthCheck := hc.Name != name
		if isNodesHealthCheck {
			// Lock to prevent necessary nodes health check / firewall gets deleted.
			gce.sharedResourceLock.Lock()
			defer gce.sharedResourceLock.Unlock()
		}

		if err := gce.ensureHttpHealthCheckFirewall(svc, serviceName, ipAddress, region, clusterID, hosts, hc.Name, int32(hc.Port), isNodesHealthCheck); err != nil {
			return err
		}
		var err error
		hcRequestPath, hcPort := hc.RequestPath, hc.Port
		if hc, err = gce.ensureHttpHealthCheck(hc.Name, hc.RequestPath, int32(hc.Port)); err != nil || hc == nil {
			return fmt.Errorf("Failed to ensure health check for %v port %d path %v: %v", name, hcPort, hcRequestPath, err)
		}
		hcLinks = append(hcLinks, hc.SelfLink)
	}

	var instances []string
	for _, host := range hosts {
		instances = append(instances, makeHostURL(gce.service.BasePath, gce.projectID, host.Zone, host.Name))
	}
	glog.Infof("Creating targetpool %v with %d healthchecks", name, len(hcLinks))
	pool := &compute.TargetPool{
		Name:            name,
		Description:     fmt.Sprintf(`{"kubernetes.io/service-name":"%s"}`, serviceName),
		Instances:       instances,
		SessionAffinity: translateAffinityType(affinityType),
		HealthChecks:    hcLinks,
	}

	if err := gce.CreateTargetPool(pool, region); err != nil && !isHTTPErrorCode(err, http.StatusConflict) {
		return err
	}
	return nil
}

func (gce *GCECloud) updateTargetPool(loadBalancerName string, existing sets.String, hosts []*gceInstance) error {
	var toAdd []*compute.InstanceReference
	var toRemove []*compute.InstanceReference
	for _, host := range hosts {
		link := host.makeComparableHostPath()
		if !existing.Has(link) {
			toAdd = append(toAdd, &compute.InstanceReference{Instance: link})
		}
		existing.Delete(link)
	}
	for link := range existing {
		toRemove = append(toRemove, &compute.InstanceReference{Instance: link})
	}

	if len(toAdd) > 0 {
		if err := gce.AddInstancesToTargetPool(loadBalancerName, gce.region, toAdd); err != nil {
			return err
		}
	}

	if len(toRemove) > 0 {
		if err := gce.RemoveInstancesFromTargetPool(loadBalancerName, gce.region, toRemove); err != nil {
			return err
		}
	}

	// Try to verify that the correct number of nodes are now in the target pool.
	// We've been bitten by a bug here before (#11327) where all nodes were
	// accidentally removed and want to make similar problems easier to notice.
	updatedPool, err := gce.GetTargetPool(loadBalancerName, gce.region)
	if err != nil {
		return err
	}
	if len(updatedPool.Instances) != len(hosts) {
		glog.Errorf("Unexpected number of instances (%d) in target pool %s after updating (expected %d). Instances in updated pool: %s",
			len(updatedPool.Instances), loadBalancerName, len(hosts), strings.Join(updatedPool.Instances, ","))
		return fmt.Errorf("Unexpected number of instances (%d) in target pool %s after update (expected %d)", len(updatedPool.Instances), loadBalancerName, len(hosts))
	}
	return nil
}

func (gce *GCECloud) targetPoolURL(name string) string {
	return gce.service.BasePath + strings.Join([]string{gce.projectID, "regions", gce.region, "targetPools", name}, "/")
}

func makeHttpHealthCheck(name, path string, port int32) *compute.HttpHealthCheck {
	return &compute.HttpHealthCheck{
		Name:               name,
		Port:               int64(port),
		RequestPath:        path,
		Host:               "",
		Description:        makeHealthCheckDescription(name),
		CheckIntervalSec:   gceHcCheckIntervalSeconds,
		TimeoutSec:         gceHcTimeoutSeconds,
		HealthyThreshold:   gceHcHealthyThreshold,
		UnhealthyThreshold: gceHcUnhealthyThreshold,
	}
}

func (gce *GCECloud) ensureHttpHealthCheck(name, path string, port int32) (hc *compute.HttpHealthCheck, err error) {
	newHC := makeHttpHealthCheck(name, path, port)
	hc, err = gce.GetHttpHealthCheck(name)
	if hc == nil || err != nil && isHTTPErrorCode(err, http.StatusNotFound) {
		glog.Infof("Did not find health check %v, creating port %v path %v", name, port, path)
		if err = gce.CreateHttpHealthCheck(newHC); err != nil {
			return nil, err
		}
		hc, err = gce.GetHttpHealthCheck(name)
		if err != nil {
			glog.Errorf("Failed to get http health check %v", err)
			return nil, err
		}
		glog.Infof("Created HTTP health check %v healthCheckNodePort: %d", name, port)
		return hc, nil
	}
	// Validate health check fields
	glog.V(4).Infof("Checking http health check params %s", name)
	drift := hc.Port != int64(port) || hc.RequestPath != path || hc.Description != makeHealthCheckDescription(name)
	drift = drift || hc.CheckIntervalSec != gceHcCheckIntervalSeconds || hc.TimeoutSec != gceHcTimeoutSeconds
	drift = drift || hc.UnhealthyThreshold != gceHcUnhealthyThreshold || hc.HealthyThreshold != gceHcHealthyThreshold
	if drift {
		glog.Warningf("Health check %v exists but parameters have drifted - updating...", name)
		if err := gce.UpdateHttpHealthCheck(newHC); err != nil {
			glog.Warningf("Failed to reconcile http health check %v parameters", name)
			return nil, err
		}
		glog.V(4).Infof("Corrected health check %v parameters successful", name)
	}
	return hc, nil
}

// Passing nil for requested IP is perfectly fine - it just means that no specific
// IP is being requested.
// Returns whether the forwarding rule exists, whether it needs to be updated,
// what its IP address is (if it exists), and any error we encountered.
func (gce *GCECloud) forwardingRuleNeedsUpdate(name, region string, loadBalancerIP string, ports []v1.ServicePort) (exists bool, needsUpdate bool, ipAddress string, err error) {
	fwd, err := gce.service.ForwardingRules.Get(gce.projectID, region, name).Do()
	if err != nil {
		if isHTTPErrorCode(err, http.StatusNotFound) {
			return false, true, "", nil
		}
		// Err on the side of caution in case of errors. Caller should notice the error and retry.
		// We never want to end up recreating resources because gce api flaked.
		return true, false, "", fmt.Errorf("error getting load balancer's forwarding rule: %v", err)
	}
	// If the user asks for a specific static ip through the Service spec,
	// check that we're actually using it.
	// TODO: we report loadbalancer IP through status, so we want to verify if
	// that matches the forwarding rule as well.
	if loadBalancerIP != "" && loadBalancerIP != fwd.IPAddress {
		glog.Infof("LoadBalancer ip for forwarding rule %v was expected to be %v, but was actually %v", fwd.Name, fwd.IPAddress, loadBalancerIP)
		return true, true, fwd.IPAddress, nil
	}
	portRange, err := loadBalancerPortRange(ports)
	if err != nil {
		// Err on the side of caution in case of errors. Caller should notice the error and retry.
		// We never want to end up recreating resources because gce api flaked.
		return true, false, "", err
	}
	if portRange != fwd.PortRange {
		glog.Infof("LoadBalancer port range for forwarding rule %v was expected to be %v, but was actually %v", fwd.Name, fwd.PortRange, portRange)
		return true, true, fwd.IPAddress, nil
	}
	// The service controller verified all the protocols match on the ports, just check the first one
	if string(ports[0].Protocol) != fwd.IPProtocol {
		glog.Infof("LoadBalancer protocol for forwarding rule %v was expected to be %v, but was actually %v", fwd.Name, fwd.IPProtocol, string(ports[0].Protocol))
		return true, true, fwd.IPAddress, nil
	}

	return true, false, fwd.IPAddress, nil
}

// Doesn't check whether the hosts have changed, since host updating is handled
// separately.
func (gce *GCECloud) targetPoolNeedsUpdate(name, region string, affinityType v1.ServiceAffinity) (exists bool, needsUpdate bool, err error) {
	tp, err := gce.service.TargetPools.Get(gce.projectID, region, name).Do()
	if err != nil {
		if isHTTPErrorCode(err, http.StatusNotFound) {
			return false, true, nil
		}
		// Err on the side of caution in case of errors. Caller should notice the error and retry.
		// We never want to end up recreating resources because gce api flaked.
		return true, false, fmt.Errorf("error getting load balancer's target pool: %v", err)
	}
	// TODO: If the user modifies their Service's session affinity, it *should*
	// reflect in the associated target pool. However, currently not setting the
	// session affinity on a target pool defaults it to the empty string while
	// not setting in on a Service defaults it to None. There is a lack of
	// documentation around the default setting for the target pool, so if we
	// find it's the undocumented empty string, don't blindly recreate the
	// target pool (which results in downtime). Fix this when we have formally
	// defined the defaults on either side.
	if tp.SessionAffinity != "" && translateAffinityType(affinityType) != tp.SessionAffinity {
		glog.Infof("LoadBalancer target pool %v changed affinity from %v to %v", name, tp.SessionAffinity, affinityType)
		return true, true, nil
	}
	return true, false, nil
}

func (h *gceInstance) makeComparableHostPath() string {
	return fmt.Sprintf("/zones/%s/instances/%s", h.Zone, h.Name)
}

func nodeNames(nodes []*v1.Node) []string {
	ret := make([]string, len(nodes))
	for i, node := range nodes {
		ret[i] = node.Name
	}
	return ret
}

func makeHostURL(projectsApiEndpoint, projectID, zone, host string) string {
	host = canonicalizeInstanceName(host)
	return projectsApiEndpoint + strings.Join([]string{projectID, "zones", zone, "instances", host}, "/")
}

func hostURLToComparablePath(hostURL string) string {
	idx := strings.Index(hostURL, "/zones/")
	if idx < 0 {
		return ""
	}
	return hostURL[idx:]
}

func loadBalancerPortRange(ports []v1.ServicePort) (string, error) {
	if len(ports) == 0 {
		return "", fmt.Errorf("no ports specified for GCE load balancer")
	}

	// The service controller verified all the protocols match on the ports, just check and use the first one
	if ports[0].Protocol != v1.ProtocolTCP && ports[0].Protocol != v1.ProtocolUDP {
		return "", fmt.Errorf("Invalid protocol %s, only TCP and UDP are supported", string(ports[0].Protocol))
	}

	minPort := int32(65536)
	maxPort := int32(0)
	for i := range ports {
		if ports[i].Port < minPort {
			minPort = ports[i].Port
		}
		if ports[i].Port > maxPort {
			maxPort = ports[i].Port
		}
	}
	return fmt.Sprintf("%d-%d", minPort, maxPort), nil
}

// translate from what K8s supports to what the cloud provider supports for session affinity.
func translateAffinityType(affinityType v1.ServiceAffinity) string {
	switch affinityType {
	case v1.ServiceAffinityClientIP:
		return gceAffinityTypeClientIP
	case v1.ServiceAffinityNone:
		return gceAffinityTypeNone
	default:
		glog.Errorf("Unexpected affinity type: %v", affinityType)
		return gceAffinityTypeNone
	}
}

func (gce *GCECloud) firewallNeedsUpdate(name, serviceName, region, ipAddress string, ports []v1.ServicePort, sourceRanges netsets.IPNet) (exists bool, needsUpdate bool, err error) {
	fw, err := gce.service.Firewalls.Get(gce.NetworkProjectID(), MakeFirewallName(name)).Do()
	if err != nil {
		if isHTTPErrorCode(err, http.StatusNotFound) {
			return false, true, nil
		}
		return false, false, fmt.Errorf("error getting load balancer's firewall: %v", err)
	}
	if fw.Description != makeFirewallDescription(serviceName, ipAddress) {
		return true, true, nil
	}
	if len(fw.Allowed) != 1 || (fw.Allowed[0].IPProtocol != "tcp" && fw.Allowed[0].IPProtocol != "udp") {
		return true, true, nil
	}
	// Make sure the allowed ports match.
	allowedPorts := make([]string, len(ports))
	for ix := range ports {
		allowedPorts[ix] = strconv.Itoa(int(ports[ix].Port))
	}
	if !equalStringSets(allowedPorts, fw.Allowed[0].Ports) {
		return true, true, nil
	}
	// The service controller already verified that the protocol matches on all ports, no need to check.

	actualSourceRanges, err := netsets.ParseIPNets(fw.SourceRanges...)
	if err != nil {
		// This really shouldn't happen... GCE has returned something unexpected
		glog.Warningf("Error parsing firewall SourceRanges: %v", fw.SourceRanges)
		// We don't return the error, because we can hopefully recover from this by reconfiguring the firewall
		return true, true, nil
	}

	if !sourceRanges.Equal(actualSourceRanges) {
		return true, true, nil
	}
	return true, false, nil
}

func (gce *GCECloud) ensureHttpHealthCheckFirewall(svc *v1.Service, serviceName, ipAddress, region, clusterID string, hosts []*gceInstance, hcName string, hcPort int32, isNodesHealthCheck bool) error {
	// Prepare the firewall params for creating / checking.
	desc := fmt.Sprintf(`{"kubernetes.io/cluster-id":"%s"}`, clusterID)
	if !isNodesHealthCheck {
		desc = makeFirewallDescription(serviceName, ipAddress)
	}
	sourceRanges := lbSrcRngsFlag.ipn
	ports := []v1.ServicePort{{Protocol: "tcp", Port: hcPort}}

	fwName := MakeHealthCheckFirewallName(clusterID, hcName, isNodesHealthCheck)
	fw, err := gce.service.Firewalls.Get(gce.NetworkProjectID(), fwName).Do()
	if err != nil {
		if !isHTTPErrorCode(err, http.StatusNotFound) {
			return fmt.Errorf("error getting firewall for health checks: %v", err)
		}
		glog.Infof("Creating firewall %v for health checks.", fwName)
		if err := gce.createFirewall(svc, fwName, region, desc, sourceRanges, ports, hosts); err != nil {
			return err
		}
		glog.Infof("Created firewall %v for health checks.", fwName)
		return nil
	}
	// Validate firewall fields.
	if fw.Description != desc ||
		len(fw.Allowed) != 1 ||
		fw.Allowed[0].IPProtocol != string(ports[0].Protocol) ||
		!equalStringSets(fw.Allowed[0].Ports, []string{strconv.Itoa(int(ports[0].Port))}) ||
		!equalStringSets(fw.SourceRanges, sourceRanges.StringSlice()) {
		glog.Warningf("Firewall %v exists but parameters have drifted - updating...", fwName)
		if err := gce.updateFirewall(svc, fwName, region, desc, sourceRanges, ports, hosts); err != nil {
			glog.Warningf("Failed to reconcile firewall %v parameters.", fwName)
			return err
		}
		glog.V(4).Infof("Corrected firewall %v parameters successful", fwName)
	}
	return nil
}

func createForwardingRule(s CloudForwardingRuleService, name, serviceName, region, ipAddress, target string, ports []v1.ServicePort, netTier NetworkTier) error {
	portRange, err := loadBalancerPortRange(ports)
	if err != nil {
		return err
	}
	desc := makeServiceDescription(serviceName)
	ipProtocol := string(ports[0].Protocol)

	switch netTier {
	case NetworkTierPremium:
		rule := &compute.ForwardingRule{
			Name:        name,
			Description: desc,
			IPAddress:   ipAddress,
			IPProtocol:  ipProtocol,
			PortRange:   portRange,
			Target:      target,
		}
		err = s.CreateRegionForwardingRule(rule, region)
	default:
		rule := &computealpha.ForwardingRule{
			Name:        name,
			Description: desc,
			IPAddress:   ipAddress,
			IPProtocol:  ipProtocol,
			PortRange:   portRange,
			Target:      target,
			NetworkTier: netTier.ToGCEValue(),
		}
		err = s.CreateAlphaRegionForwardingRule(rule, region)
	}

	if err != nil && !isHTTPErrorCode(err, http.StatusConflict) {
		return err
	}

	return nil
}

func (gce *GCECloud) createFirewall(svc *v1.Service, name, region, desc string, sourceRanges netsets.IPNet, ports []v1.ServicePort, hosts []*gceInstance) error {
	firewall, err := gce.firewallObject(name, region, desc, sourceRanges, ports, hosts)
	if err != nil {
		return err
	}
	if err = gce.CreateFirewall(firewall); err != nil {
		if isHTTPErrorCode(err, http.StatusConflict) {
			return nil
		} else if isForbidden(err) && gce.OnXPN() {
			glog.V(4).Infof("createFirewall(%v): do not have permission to create firewall rule (on XPN). Raising event.", firewall.Name)
			gce.raiseFirewallChangeNeededEvent(svc, FirewallToGCloudCreateCmd(firewall, gce.NetworkProjectID()))
			return nil
		}
		return err
	}
	return nil
}

func (gce *GCECloud) updateFirewall(svc *v1.Service, name, region, desc string, sourceRanges netsets.IPNet, ports []v1.ServicePort, hosts []*gceInstance) error {
	firewall, err := gce.firewallObject(name, region, desc, sourceRanges, ports, hosts)
	if err != nil {
		return err
	}

	if err = gce.UpdateFirewall(firewall); err != nil {
		if isHTTPErrorCode(err, http.StatusConflict) {
			return nil
		} else if isForbidden(err) && gce.OnXPN() {
			glog.V(4).Infof("updateFirewall(%v): do not have permission to update firewall rule (on XPN). Raising event.", firewall.Name)
			gce.raiseFirewallChangeNeededEvent(svc, FirewallToGCloudUpdateCmd(firewall, gce.NetworkProjectID()))
			return nil
		}
		return err
	}
	return nil
}

func (gce *GCECloud) firewallObject(name, region, desc string, sourceRanges netsets.IPNet, ports []v1.ServicePort, hosts []*gceInstance) (*compute.Firewall, error) {
	allowedPorts := make([]string, len(ports))
	for ix := range ports {
		allowedPorts[ix] = strconv.Itoa(int(ports[ix].Port))
	}
	// If the node tags to be used for this cluster have been predefined in the
	// provider config, just use them. Otherwise, invoke computeHostTags method to get the tags.
	hostTags := gce.nodeTags
	if len(hostTags) == 0 {
		var err error
		if hostTags, err = gce.computeHostTags(hosts); err != nil {
			return nil, fmt.Errorf("No node tags supplied and also failed to parse the given lists of hosts for tags. Abort creating firewall rule.")
		}
	}

	firewall := &compute.Firewall{
		Name:         name,
		Description:  desc,
		Network:      gce.networkURL,
		SourceRanges: sourceRanges.StringSlice(),
		TargetTags:   hostTags,
		Allowed: []*compute.FirewallAllowed{
			{
				// TODO: Make this more generic. Currently this method is only
				// used to create firewall rules for loadbalancers, which have
				// exactly one protocol, so we can never end up with a list of
				// mixed TCP and UDP ports. It should be possible to use a
				// single firewall rule for both a TCP and UDP lb.
				IPProtocol: strings.ToLower(string(ports[0].Protocol)),
				Ports:      allowedPorts,
			},
		},
	}
	return firewall, nil
}

func ensureStaticIP(s CloudAddressService, name, serviceName, region, existingIP string, netTier NetworkTier) (ipAddress string, existing bool, err error) {
	// If the address doesn't exist, this will create it.
	// If the existingIP exists but is ephemeral, this will promote it to static.
	// If the address already exists, this will harmlessly return a StatusConflict
	// and we'll grab the IP before returning.
	existed := false
	desc := makeServiceDescription(serviceName)

	var creationErr error
	switch netTier {
	case NetworkTierPremium:
		addressObj := &compute.Address{
			Name:        name,
			Description: desc,
		}
		if existingIP != "" {
			addressObj.Address = existingIP
		}
		creationErr = s.ReserveRegionAddress(addressObj, region)
	default:
		addressObj := &computealpha.Address{
			Name:        name,
			Description: desc,
			NetworkTier: netTier.ToGCEValue(),
		}
		if existingIP != "" {
			addressObj.Address = existingIP
		}
		creationErr = s.ReserveAlphaRegionAddress(addressObj, region)
	}

	if creationErr != nil {
		// GCE returns StatusConflict if the name conflicts; it returns
		// StatusBadRequest if the IP conflicts.
		if !isHTTPErrorCode(creationErr, http.StatusConflict) && !isHTTPErrorCode(creationErr, http.StatusBadRequest) {
			return "", false, fmt.Errorf("error creating gce static IP address: %v", creationErr)
		}
		existed = true
	}

	addr, err := s.GetRegionAddress(name, region)
	if err != nil {
		return "", false, fmt.Errorf("error getting static IP address: %v", err)
	}

	return addr.Address, existed, nil
}

func (gce *GCECloud) getServiceNetworkTier(svc *v1.Service) (NetworkTier, error) {
	if !gce.AlphaFeatureGate.Enabled(AlphaFeatureNetworkTiers) {
		return NetworkTierDefault, nil
	}
	tier, err := GetServiceNetworkTier(svc)
	if err != nil {
		// Returns an error if the annotation is invalid.
		return NetworkTier(""), err
	}
	return tier, nil
}

func (gce *GCECloud) deleteWrongNetworkTieredResources(lbName, lbRef string, desiredNetTier NetworkTier) error {
	logPrefix := fmt.Sprintf("deleteWrongNetworkTieredResources:(%s)", lbRef)
	if err := deleteFWDRuleWithWrongTier(gce, gce.region, lbName, logPrefix, desiredNetTier); err != nil {
		return err
	}
	if err := deleteAddressWithWrongTier(gce, gce.region, lbName, logPrefix, desiredNetTier); err != nil {
		return err
	}
	return nil
}

// deleteFWDRuleWithWrongTier checks the network tier of existing forwarding
// rule and delete the rule if the tier does not matched the desired tier.
func deleteFWDRuleWithWrongTier(s CloudForwardingRuleService, region, name, logPrefix string, desiredNetTier NetworkTier) error {
	tierStr, err := s.getNetworkTierFromForwardingRule(name, region)
	if isNotFound(err) {
		return nil
	} else if err != nil {
		return err
	}
	existingTier := NetworkTierGCEValueToType(tierStr)
	if existingTier == desiredNetTier {
		return nil
	}
	glog.V(2).Infof("%s: Network tiers do not match; existing forwarding rule: %q, desired: %q. Deleting the forwarding rule",
		logPrefix, existingTier, desiredNetTier)
	err = s.DeleteRegionForwardingRule(name, region)
	return ignoreNotFound(err)
}

// deleteAddressWithWrongTier checks the network tier of existing address
// and delete the address if the tier does not matched the desired tier.
func deleteAddressWithWrongTier(s CloudAddressService, region, name, logPrefix string, desiredNetTier NetworkTier) error {
	// We only check the IP address matching the reserved name that the
	// controller assigned to the LB. We make the assumption that an address of
	// such name is owned by the controller and is safe to release. Whether an
	// IP is owned by the user is not clearly defined in the current code, and
	// this assumption may not match some of the existing logic in the code.
	// However, this is okay since network tiering is still Alpha and will be
	// properly gated.
	// TODO(#51665): Re-evaluate the "ownership" of the IP address to ensure
	// we don't release IP unintentionally.
	tierStr, err := s.getNetworkTierFromAddress(name, region)
	if isNotFound(err) {
		return nil
	} else if err != nil {
		return err
	}
	existingTier := NetworkTierGCEValueToType(tierStr)
	if existingTier == desiredNetTier {
		return nil
	}
	glog.V(2).Infof("%s: Network tiers do not match; existing address: %q, desired: %q. Deleting the address",
		logPrefix, existingTier, desiredNetTier)
	err = s.DeleteRegionAddress(name, region)
	return ignoreNotFound(err)
}
