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
	"net/http"
	"strconv"
	"strings"

	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	servicehelpers "k8s.io/cloud-provider/service/helpers"
	utilnet "k8s.io/utils/net"

	computealpha "google.golang.org/api/compute/v0.alpha"
	compute "google.golang.org/api/compute/v1"
	"k8s.io/klog"
)

const (
	errStrLbNoHosts = "cannot EnsureLoadBalancer() with no hosts"
)

// ensureExternalLoadBalancer is the external implementation of LoadBalancer.EnsureLoadBalancer.
// Our load balancers in GCE consist of four separate GCE resources - a static
// IP address, a firewall rule, a target pool, and a forwarding rule. This
// function has to manage all of them.
//
// Due to an interesting series of design decisions, this handles both creating
// new load balancers and updating existing load balancers, recognizing when
// each is needed.
func (g *Cloud) ensureExternalLoadBalancer(clusterName string, clusterID string, apiService *v1.Service, existingFwdRule *compute.ForwardingRule, nodes []*v1.Node) (*v1.LoadBalancerStatus, error) {
	if len(nodes) == 0 {
		return nil, fmt.Errorf(errStrLbNoHosts)
	}

	hostNames := nodeNames(nodes)
	supportsNodesHealthCheck := supportsNodesHealthCheck(nodes)
	hosts, err := g.getInstancesByNames(hostNames)
	if err != nil {
		return nil, err
	}

	loadBalancerName := g.GetLoadBalancerName(context.TODO(), clusterName, apiService)
	requestedIP := apiService.Spec.LoadBalancerIP
	ports := apiService.Spec.Ports
	portStr := []string{}
	for _, p := range apiService.Spec.Ports {
		portStr = append(portStr, fmt.Sprintf("%s/%d", p.Protocol, p.Port))
	}

	serviceName := types.NamespacedName{Namespace: apiService.Namespace, Name: apiService.Name}
	lbRefStr := fmt.Sprintf("%v(%v)", loadBalancerName, serviceName)
	klog.V(2).Infof("ensureExternalLoadBalancer(%s, %v, %v, %v, %v, %v)", lbRefStr, g.region, requestedIP, portStr, hostNames, apiService.Annotations)

	// Check the current and the desired network tiers. If they do not match,
	// tear down the existing resources with the wrong tier.
	netTier, err := g.getServiceNetworkTier(apiService)
	if err != nil {
		klog.Errorf("ensureExternalLoadBalancer(%s): Failed to get the desired network tier: %v.", lbRefStr, err)
		return nil, err
	}
	klog.V(4).Infof("ensureExternalLoadBalancer(%s): Desired network tier %q.", lbRefStr, netTier)
	if g.AlphaFeatureGate.Enabled(AlphaFeatureNetworkTiers) {
		g.deleteWrongNetworkTieredResources(loadBalancerName, lbRefStr, netTier)
	}

	// Check if the forwarding rule exists, and if so, what its IP is.
	fwdRuleExists, fwdRuleNeedsUpdate, fwdRuleIP, err := g.forwardingRuleNeedsUpdate(loadBalancerName, g.region, requestedIP, ports)
	if err != nil {
		return nil, err
	}
	if !fwdRuleExists {
		klog.V(2).Infof("ensureExternalLoadBalancer(%s): Forwarding rule %v doesn't exist.", lbRefStr, loadBalancerName)
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
			if err := g.DeleteRegionAddress(loadBalancerName, g.region); err != nil && !isNotFound(err) {
				klog.Errorf("ensureExternalLoadBalancer(%s): Failed to release static IP %s in region %v: %v.", lbRefStr, ipAddressToUse, g.region, err)
			} else if isNotFound(err) {
				klog.V(2).Infof("ensureExternalLoadBalancer(%s): IP address %s is not reserved.", lbRefStr, ipAddressToUse)
			} else {
				klog.Infof("ensureExternalLoadBalancer(%s): Released static IP %s.", lbRefStr, ipAddressToUse)
			}
		} else {
			klog.Warningf("ensureExternalLoadBalancer(%s): Orphaning static IP %s in region %v: %v.", lbRefStr, ipAddressToUse, g.region, err)
		}
	}()

	if requestedIP != "" {
		// If user requests a specific IP address, verify first. No mutation to
		// the GCE resources will be performed in the verification process.
		isUserOwnedIP, err = verifyUserRequestedIP(g, g.region, requestedIP, fwdRuleIP, lbRefStr, netTier)
		if err != nil {
			return nil, err
		}
		ipAddressToUse = requestedIP
	}

	if !isUserOwnedIP {
		// If we are not using the user-owned IP, either promote the
		// emphemeral IP used by the fwd rule, or create a new static IP.
		ipAddr, existed, err := ensureStaticIP(g, loadBalancerName, serviceName.String(), g.region, fwdRuleIP, netTier)
		if err != nil {
			return nil, fmt.Errorf("failed to ensure a static IP for load balancer (%s): %v", lbRefStr, err)
		}
		klog.Infof("ensureExternalLoadBalancer(%s): Ensured IP address %s (tier: %s).", lbRefStr, ipAddr, netTier)
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
	sourceRanges, err := servicehelpers.GetLoadBalancerSourceRanges(apiService)
	if err != nil {
		return nil, err
	}

	firewallExists, firewallNeedsUpdate, err := g.firewallNeedsUpdate(loadBalancerName, serviceName.String(), g.region, ipAddressToUse, ports, sourceRanges)
	if err != nil {
		return nil, err
	}

	if firewallNeedsUpdate {
		desc := makeFirewallDescription(serviceName.String(), ipAddressToUse)
		// Unlike forwarding rules and target pools, firewalls can be updated
		// without needing to be deleted and recreated.
		if firewallExists {
			klog.Infof("ensureExternalLoadBalancer(%s): Updating firewall.", lbRefStr)
			if err := g.updateFirewall(apiService, MakeFirewallName(loadBalancerName), g.region, desc, sourceRanges, ports, hosts); err != nil {
				return nil, err
			}
			klog.Infof("ensureExternalLoadBalancer(%s): Updated firewall.", lbRefStr)
		} else {
			klog.Infof("ensureExternalLoadBalancer(%s): Creating firewall.", lbRefStr)
			if err := g.createFirewall(apiService, MakeFirewallName(loadBalancerName), g.region, desc, sourceRanges, ports, hosts); err != nil {
				return nil, err
			}
			klog.Infof("ensureExternalLoadBalancer(%s): Created firewall.", lbRefStr)
		}
	}

	tpExists, tpNeedsRecreation, err := g.targetPoolNeedsRecreation(loadBalancerName, g.region, apiService.Spec.SessionAffinity)
	if err != nil {
		return nil, err
	}
	if !tpExists {
		klog.Infof("ensureExternalLoadBalancer(%s): Target pool for service doesn't exist.", lbRefStr)
	}

	// Check which health check needs to create and which health check needs to delete.
	// Health check management is coupled with target pool operation to prevent leaking.
	var hcToCreate, hcToDelete *compute.HttpHealthCheck
	hcLocalTrafficExisting, err := g.GetHTTPHealthCheck(loadBalancerName)
	if err != nil && !isHTTPErrorCode(err, http.StatusNotFound) {
		return nil, fmt.Errorf("error checking HTTP health check for load balancer (%s): %v", lbRefStr, err)
	}
	if path, healthCheckNodePort := servicehelpers.GetServiceHealthCheckPathPort(apiService); path != "" {
		klog.V(4).Infof("ensureExternalLoadBalancer(%s): Service needs local traffic health checks on: %d%s.", lbRefStr, healthCheckNodePort, path)
		if hcLocalTrafficExisting == nil {
			// This logic exists to detect a transition for non-OnlyLocal to OnlyLocal service
			// turn on the tpNeedsRecreation flag to delete/recreate fwdrule/tpool updating the
			// target pool to use local traffic health check.
			klog.V(2).Infof("ensureExternalLoadBalancer(%s): Updating from nodes health checks to local traffic health checks.", lbRefStr)
			if supportsNodesHealthCheck {
				hcToDelete = makeHTTPHealthCheck(MakeNodesHealthCheckName(clusterID), GetNodesHealthCheckPath(), GetNodesHealthCheckPort())
			}
			tpNeedsRecreation = true
		}
		hcToCreate = makeHTTPHealthCheck(loadBalancerName, path, healthCheckNodePort)
	} else {
		klog.V(4).Infof("ensureExternalLoadBalancer(%s): Service needs nodes health checks.", lbRefStr)
		if hcLocalTrafficExisting != nil {
			// This logic exists to detect a transition from OnlyLocal to non-OnlyLocal service
			// and turn on the tpNeedsRecreation flag to delete/recreate fwdrule/tpool updating the
			// target pool to use nodes health check.
			klog.V(2).Infof("ensureExternalLoadBalancer(%s): Updating from local traffic health checks to nodes health checks.", lbRefStr)
			hcToDelete = hcLocalTrafficExisting
			tpNeedsRecreation = true
		}
		if supportsNodesHealthCheck {
			hcToCreate = makeHTTPHealthCheck(MakeNodesHealthCheckName(clusterID), GetNodesHealthCheckPath(), GetNodesHealthCheckPort())
		}
	}
	// Now we get to some slightly more interesting logic.
	// First, neither target pools nor forwarding rules can be updated in place -
	// they have to be deleted and recreated.
	// Second, forwarding rules are layered on top of target pools in that you
	// can't delete a target pool that's currently in use by a forwarding rule.
	// Thus, we have to tear down the forwarding rule if either it or the target
	// pool needs to be updated.
	if fwdRuleExists && (fwdRuleNeedsUpdate || tpNeedsRecreation) {
		// Begin critical section. If we have to delete the forwarding rule,
		// and something should fail before we recreate it, don't release the
		// IP.  That way we can come back to it later.
		isSafeToReleaseIP = false
		if err := g.DeleteRegionForwardingRule(loadBalancerName, g.region); err != nil && !isNotFound(err) {
			return nil, fmt.Errorf("failed to delete existing forwarding rule for load balancer (%s) update: %v", lbRefStr, err)
		}
		klog.Infof("ensureExternalLoadBalancer(%s): Deleted forwarding rule.", lbRefStr)
	}

	if err := g.ensureTargetPoolAndHealthCheck(tpExists, tpNeedsRecreation, apiService, loadBalancerName, clusterID, ipAddressToUse, hosts, hcToCreate, hcToDelete); err != nil {
		return nil, err
	}

	if tpNeedsRecreation || fwdRuleNeedsUpdate {
		klog.Infof("ensureExternalLoadBalancer(%s): Creating forwarding rule, IP %s (tier: %s).", lbRefStr, ipAddressToUse, netTier)
		if err := createForwardingRule(g, loadBalancerName, serviceName.String(), g.region, ipAddressToUse, g.targetPoolURL(loadBalancerName), ports, netTier); err != nil {
			return nil, fmt.Errorf("failed to create forwarding rule for load balancer (%s): %v", lbRefStr, err)
		}
		// End critical section.  It is safe to release the static IP (which
		// just demotes it to ephemeral) now that it is attached.  In the case
		// of a user-requested IP, the "is user-owned" flag will be set,
		// preventing it from actually being released.
		isSafeToReleaseIP = true
		klog.Infof("ensureExternalLoadBalancer(%s): Created forwarding rule, IP %s.", lbRefStr, ipAddressToUse)
	}

	status := &v1.LoadBalancerStatus{}
	status.Ingress = []v1.LoadBalancerIngress{{IP: ipAddressToUse}}

	return status, nil
}

// updateExternalLoadBalancer is the external implementation of LoadBalancer.UpdateLoadBalancer.
func (g *Cloud) updateExternalLoadBalancer(clusterName string, service *v1.Service, nodes []*v1.Node) error {
	hosts, err := g.getInstancesByNames(nodeNames(nodes))
	if err != nil {
		return err
	}

	loadBalancerName := g.GetLoadBalancerName(context.TODO(), clusterName, service)
	return g.updateTargetPool(loadBalancerName, hosts)
}

// ensureExternalLoadBalancerDeleted is the external implementation of LoadBalancer.EnsureLoadBalancerDeleted
func (g *Cloud) ensureExternalLoadBalancerDeleted(clusterName, clusterID string, service *v1.Service) error {
	loadBalancerName := g.GetLoadBalancerName(context.TODO(), clusterName, service)
	serviceName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	lbRefStr := fmt.Sprintf("%v(%v)", loadBalancerName, serviceName)

	var hcNames []string
	if path, _ := servicehelpers.GetServiceHealthCheckPathPort(service); path != "" {
		hcToDelete, err := g.GetHTTPHealthCheck(loadBalancerName)
		if err != nil && !isHTTPErrorCode(err, http.StatusNotFound) {
			klog.Infof("ensureExternalLoadBalancerDeleted(%s): Failed to retrieve health check:%v.", lbRefStr, err)
			return err
		}
		// If we got 'StatusNotFound' LB was already deleted and it's safe to ignore.
		if err == nil {
			hcNames = append(hcNames, hcToDelete.Name)
		}
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
			klog.Infof("ensureExternalLoadBalancerDeleted(%s): Deleting firewall rule.", lbRefStr)
			fwName := MakeFirewallName(loadBalancerName)
			err := ignoreNotFound(g.DeleteFirewall(fwName))
			if isForbidden(err) && g.OnXPN() {
				klog.V(4).Infof("ensureExternalLoadBalancerDeleted(%s): Do not have permission to delete firewall rule %v (on XPN). Raising event.", lbRefStr, fwName)
				g.raiseFirewallChangeNeededEvent(service, FirewallToGCloudDeleteCmd(fwName, g.NetworkProjectID()))
				return nil
			}
			return err
		},
		// Even though we don't hold on to static IPs for load balancers, it's
		// possible that EnsureLoadBalancer left one around in a failed
		// creation/update attempt, so make sure we clean it up here just in case.
		func() error {
			klog.Infof("ensureExternalLoadBalancerDeleted(%s): Deleting IP address.", lbRefStr)
			return ignoreNotFound(g.DeleteRegionAddress(loadBalancerName, g.region))
		},
		func() error {
			klog.Infof("ensureExternalLoadBalancerDeleted(%s): Deleting forwarding rule.", lbRefStr)
			// The forwarding rule must be deleted before either the target pool can,
			// unfortunately, so we have to do these two serially.
			if err := ignoreNotFound(g.DeleteRegionForwardingRule(loadBalancerName, g.region)); err != nil {
				return err
			}
			klog.Infof("ensureExternalLoadBalancerDeleted(%s): Deleting target pool.", lbRefStr)
			if err := g.DeleteExternalTargetPoolAndChecks(service, loadBalancerName, g.region, clusterID, hcNames...); err != nil {
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

// DeleteExternalTargetPoolAndChecks Deletes an external load balancer pool and verifies the operation
func (g *Cloud) DeleteExternalTargetPoolAndChecks(service *v1.Service, name, region, clusterID string, hcNames ...string) error {
	serviceName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	lbRefStr := fmt.Sprintf("%v(%v)", name, serviceName)

	if err := g.DeleteTargetPool(name, region); err != nil && isHTTPErrorCode(err, http.StatusNotFound) {
		klog.Infof("DeleteExternalTargetPoolAndChecks(%v): Target pool already deleted. Continuing to delete other resources.", lbRefStr)
	} else if err != nil {
		klog.Warningf("DeleteExternalTargetPoolAndChecks(%v): Failed to delete target pool, got error %s.", lbRefStr, err.Error())
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
				g.sharedResourceLock.Lock()
				defer g.sharedResourceLock.Unlock()
			}
			klog.Infof("DeleteExternalTargetPoolAndChecks(%v): Deleting health check %v.", lbRefStr, hcName)
			if err := g.DeleteHTTPHealthCheck(hcName); err != nil {
				// Delete nodes health checks will fail if any other target pool is using it.
				if isInUsedByError(err) {
					klog.V(4).Infof("DeleteExternalTargetPoolAndChecks(%v): Health check %v is in used: %v.", lbRefStr, hcName, err)
					return nil
				} else if !isHTTPErrorCode(err, http.StatusNotFound) {
					klog.Warningf("DeleteExternalTargetPoolAndChecks(%v): Failed to delete health check %v: %v.", lbRefStr, hcName, err)
					return err
				}
				// StatusNotFound could happen when:
				// - This is the first attempt but we pass in a healthcheck that is already deleted
				//   to prevent leaking.
				// - This is the first attempt but user manually deleted the heathcheck.
				// - This is a retry and in previous round we failed to delete the healthcheck firewall
				//   after deleted the healthcheck.
				// We continue to delete the healthcheck firewall to prevent leaking.
				klog.V(4).Infof("DeleteExternalTargetPoolAndChecks(%v): Health check %v is already deleted.", lbRefStr, hcName)
			}
			// If health check is deleted without error, it means no load-balancer is using it.
			// So we should delete the health check firewall as well.
			fwName := MakeHealthCheckFirewallName(clusterID, hcName, isNodesHealthCheck)
			klog.Infof("DeleteExternalTargetPoolAndChecks(%v): Deleting health check firewall %v.", lbRefStr, fwName)
			if err := ignoreNotFound(g.DeleteFirewall(fwName)); err != nil {
				if isForbidden(err) && g.OnXPN() {
					klog.V(4).Infof("DeleteExternalTargetPoolAndChecks(%v): Do not have permission to delete firewall rule %v (on XPN). Raising event.", lbRefStr, fwName)
					g.raiseFirewallChangeNeededEvent(service, FirewallToGCloudDeleteCmd(fwName, g.NetworkProjectID()))
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
func verifyUserRequestedIP(s CloudAddressService, region, requestedIP, fwdRuleIP, lbRef string, desiredNetTier cloud.NetworkTier) (isUserOwnedIP bool, err error) {
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
		klog.Errorf("verifyUserRequestedIP: failed to check whether the requested IP %q for LB %s exists: %v", requestedIP, lbRef, err)
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
		netTier := cloud.NetworkTierGCEValueToType(netTierStr)
		if netTier != desiredNetTier {
			klog.Errorf("verifyUserRequestedIP: requested static IP %q (name: %s) for LB %s has network tier %s, need %s.", requestedIP, existingAddress.Name, lbRef, netTier, desiredNetTier)
			return false, fmt.Errorf("requrested IP %q belongs to the %s network tier; expected %s", requestedIP, netTier, desiredNetTier)
		}
		klog.V(4).Infof("verifyUserRequestedIP: the requested static IP %q (name: %s, tier: %s) for LB %s exists.", requestedIP, existingAddress.Name, netTier, lbRef)
		return true, nil
	}
	if requestedIP == fwdRuleIP {
		// The requested IP is not a static IP, but is currently assigned
		// to this forwarding rule, so we can just use it.
		klog.V(4).Infof("verifyUserRequestedIP: the requested IP %q is not static, but is currently in use by for LB %s", requestedIP, lbRef)
		return false, nil
	}
	// The requested IP is not static and it is not assigned to the
	// current forwarding rule.  It might be attached to a different
	// rule or it might not be part of this project at all.  Either
	// way, we can't use it.
	klog.Errorf("verifyUserRequestedIP: requested IP %q for LB %s is neither static nor assigned to the LB", requestedIP, lbRef)
	return false, fmt.Errorf("requested ip %q is neither static nor assigned to the LB", requestedIP)
}

func (g *Cloud) ensureTargetPoolAndHealthCheck(tpExists, tpNeedsRecreation bool, svc *v1.Service, loadBalancerName, clusterID, ipAddressToUse string, hosts []*gceInstance, hcToCreate, hcToDelete *compute.HttpHealthCheck) error {
	serviceName := types.NamespacedName{Namespace: svc.Namespace, Name: svc.Name}
	lbRefStr := fmt.Sprintf("%v(%v)", loadBalancerName, serviceName)

	if tpExists && tpNeedsRecreation {
		// Pass healthchecks to DeleteExternalTargetPoolAndChecks to cleanup health checks after cleaning up the target pool itself.
		var hcNames []string
		if hcToDelete != nil {
			hcNames = append(hcNames, hcToDelete.Name)
		}
		if err := g.DeleteExternalTargetPoolAndChecks(svc, loadBalancerName, g.region, clusterID, hcNames...); err != nil {
			return fmt.Errorf("failed to delete existing target pool for load balancer (%s) update: %v", lbRefStr, err)
		}
		klog.Infof("ensureTargetPoolAndHealthCheck(%s): Deleted target pool.", lbRefStr)
	}
	// Once we've deleted the resources (if necessary), build them back up (or for
	// the first time if they're new).
	if tpNeedsRecreation {
		createInstances := hosts
		if len(hosts) > maxTargetPoolCreateInstances {
			createInstances = createInstances[:maxTargetPoolCreateInstances]
		}
		if err := g.createTargetPoolAndHealthCheck(svc, loadBalancerName, serviceName.String(), ipAddressToUse, g.region, clusterID, createInstances, hcToCreate); err != nil {
			return fmt.Errorf("failed to create target pool for load balancer (%s): %v", lbRefStr, err)
		}
		if hcToCreate != nil {
			klog.Infof("ensureTargetPoolAndHealthCheck(%s): Created health checks %v.", lbRefStr, hcToCreate.Name)
		}
		if len(hosts) <= maxTargetPoolCreateInstances {
			klog.Infof("ensureTargetPoolAndHealthCheck(%s): Created target pool.", lbRefStr)
		} else {
			klog.Infof("ensureTargetPoolAndHealthCheck(%s): Created initial target pool (now updating the remaining %d hosts).", lbRefStr, len(hosts)-maxTargetPoolCreateInstances)
			if err := g.updateTargetPool(loadBalancerName, hosts); err != nil {
				return fmt.Errorf("failed to update target pool for load balancer (%s): %v", lbRefStr, err)
			}
			klog.Infof("ensureTargetPoolAndHealthCheck(%s): Updated target pool (with %d hosts).", lbRefStr, len(hosts)-maxTargetPoolCreateInstances)
		}
	} else if tpExists {
		// Ensure hosts are updated even if there is no other changes required on target pool.
		if err := g.updateTargetPool(loadBalancerName, hosts); err != nil {
			return fmt.Errorf("failed to update target pool for load balancer (%s): %v", lbRefStr, err)
		}
		klog.Infof("ensureTargetPoolAndHealthCheck(%s): Updated target pool (with %d hosts).", lbRefStr, len(hosts))
		if hcToCreate != nil {
			if hc, err := g.ensureHTTPHealthCheck(hcToCreate.Name, hcToCreate.RequestPath, int32(hcToCreate.Port)); err != nil || hc == nil {
				return fmt.Errorf("failed to ensure health check for %v port %d path %v: %v", loadBalancerName, hcToCreate.Port, hcToCreate.RequestPath, err)
			}
		}
	} else {
		// Panic worthy.
		klog.Errorf("ensureTargetPoolAndHealthCheck(%s): target pool not exists and doesn't need to be created.", lbRefStr)
	}
	return nil
}

func (g *Cloud) createTargetPoolAndHealthCheck(svc *v1.Service, name, serviceName, ipAddress, region, clusterID string, hosts []*gceInstance, hc *compute.HttpHealthCheck) error {
	// health check management is coupled with targetPools to prevent leaks. A
	// target pool is the only thing that requires a health check, so we delete
	// associated checks on teardown, and ensure checks on setup.
	hcLinks := []string{}
	if hc != nil {
		// Check whether it is nodes health check, which has different name from the load-balancer.
		isNodesHealthCheck := hc.Name != name
		if isNodesHealthCheck {
			// Lock to prevent necessary nodes health check / firewall gets deleted.
			g.sharedResourceLock.Lock()
			defer g.sharedResourceLock.Unlock()
		}

		if err := g.ensureHTTPHealthCheckFirewall(svc, serviceName, ipAddress, region, clusterID, hosts, hc.Name, int32(hc.Port), isNodesHealthCheck); err != nil {
			return err
		}
		var err error
		hcRequestPath, hcPort := hc.RequestPath, hc.Port
		if hc, err = g.ensureHTTPHealthCheck(hc.Name, hc.RequestPath, int32(hc.Port)); err != nil || hc == nil {
			return fmt.Errorf("failed to ensure health check for %v port %d path %v: %v", name, hcPort, hcRequestPath, err)
		}
		hcLinks = append(hcLinks, hc.SelfLink)
	}

	var instances []string
	for _, host := range hosts {
		instances = append(instances, host.makeComparableHostPath())
	}
	klog.Infof("Creating targetpool %v with %d healthchecks", name, len(hcLinks))
	pool := &compute.TargetPool{
		Name:            name,
		Description:     fmt.Sprintf(`{"kubernetes.io/service-name":"%s"}`, serviceName),
		Instances:       instances,
		SessionAffinity: translateAffinityType(svc.Spec.SessionAffinity),
		HealthChecks:    hcLinks,
	}

	if err := g.CreateTargetPool(pool, region); err != nil && !isHTTPErrorCode(err, http.StatusConflict) {
		return err
	}
	return nil
}

func (g *Cloud) updateTargetPool(loadBalancerName string, hosts []*gceInstance) error {
	pool, err := g.GetTargetPool(loadBalancerName, g.region)
	if err != nil {
		return err
	}
	existing := sets.NewString()
	for _, instance := range pool.Instances {
		existing.Insert(hostURLToComparablePath(instance))
	}

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
		if err := g.AddInstancesToTargetPool(loadBalancerName, g.region, toAdd); err != nil {
			return err
		}
	}

	if len(toRemove) > 0 {
		if err := g.RemoveInstancesFromTargetPool(loadBalancerName, g.region, toRemove); err != nil {
			return err
		}
	}

	// Try to verify that the correct number of nodes are now in the target pool.
	// We've been bitten by a bug here before (#11327) where all nodes were
	// accidentally removed and want to make similar problems easier to notice.
	updatedPool, err := g.GetTargetPool(loadBalancerName, g.region)
	if err != nil {
		return err
	}
	if len(updatedPool.Instances) != len(hosts) {
		klog.Errorf("Unexpected number of instances (%d) in target pool %s after updating (expected %d). Instances in updated pool: %s",
			len(updatedPool.Instances), loadBalancerName, len(hosts), strings.Join(updatedPool.Instances, ","))
		return fmt.Errorf("unexpected number of instances (%d) in target pool %s after update (expected %d)", len(updatedPool.Instances), loadBalancerName, len(hosts))
	}
	return nil
}

func (g *Cloud) targetPoolURL(name string) string {
	return g.service.BasePath + strings.Join([]string{g.projectID, "regions", g.region, "targetPools", name}, "/")
}

func makeHTTPHealthCheck(name, path string, port int32) *compute.HttpHealthCheck {
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

// mergeHTTPHealthChecks reconciles HttpHealthCheck configures to be no smaller
// than the default values.
// E.g. old health check interval is 2s, new default is 8.
// The HC interval will be reconciled to 8 seconds.
// If the existing health check is larger than the default interval,
// the configuration will be kept.
func mergeHTTPHealthChecks(hc, newHC *compute.HttpHealthCheck) {
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
}

// needToUpdateHTTPHealthChecks checks whether the http healthcheck needs to be
// updated.
func needToUpdateHTTPHealthChecks(hc, newHC *compute.HttpHealthCheck) bool {
	switch {
	case
		hc.Port != newHC.Port,
		hc.RequestPath != newHC.RequestPath,
		hc.Description != newHC.Description,
		hc.CheckIntervalSec < newHC.CheckIntervalSec,
		hc.TimeoutSec < newHC.TimeoutSec,
		hc.UnhealthyThreshold < newHC.UnhealthyThreshold,
		hc.HealthyThreshold < newHC.HealthyThreshold:
		return true
	}
	return false
}

func (g *Cloud) ensureHTTPHealthCheck(name, path string, port int32) (hc *compute.HttpHealthCheck, err error) {
	newHC := makeHTTPHealthCheck(name, path, port)
	hc, err = g.GetHTTPHealthCheck(name)
	if hc == nil || err != nil && isHTTPErrorCode(err, http.StatusNotFound) {
		klog.Infof("Did not find health check %v, creating port %v path %v", name, port, path)
		if err = g.CreateHTTPHealthCheck(newHC); err != nil {
			return nil, err
		}
		hc, err = g.GetHTTPHealthCheck(name)
		if err != nil {
			klog.Errorf("Failed to get http health check %v", err)
			return nil, err
		}
		klog.Infof("Created HTTP health check %v healthCheckNodePort: %d", name, port)
		return hc, nil
	}
	// Validate health check fields
	klog.V(4).Infof("Checking http health check params %s", name)
	if needToUpdateHTTPHealthChecks(hc, newHC) {
		klog.Warningf("Health check %v exists but parameters have drifted - updating...", name)
		mergeHTTPHealthChecks(hc, newHC)
		if err := g.UpdateHTTPHealthCheck(newHC); err != nil {
			klog.Warningf("Failed to reconcile http health check %v parameters", name)
			return nil, err
		}
		klog.V(4).Infof("Corrected health check %v parameters successful", name)
		hc, err = g.GetHTTPHealthCheck(name)
		if err != nil {
			return nil, err
		}
	}
	return hc, nil
}

// Passing nil for requested IP is perfectly fine - it just means that no specific
// IP is being requested.
// Returns whether the forwarding rule exists, whether it needs to be updated,
// what its IP address is (if it exists), and any error we encountered.
func (g *Cloud) forwardingRuleNeedsUpdate(name, region string, loadBalancerIP string, ports []v1.ServicePort) (exists bool, needsUpdate bool, ipAddress string, err error) {
	fwd, err := g.GetRegionForwardingRule(name, region)
	if err != nil {
		if isHTTPErrorCode(err, http.StatusNotFound) {
			return false, true, "", nil
		}
		// Err on the side of caution in case of errors. Caller should notice the error and retry.
		// We never want to end up recreating resources because g api flaked.
		return true, false, "", fmt.Errorf("error getting load balancer's forwarding rule: %v", err)
	}
	// If the user asks for a specific static ip through the Service spec,
	// check that we're actually using it.
	// TODO: we report loadbalancer IP through status, so we want to verify if
	// that matches the forwarding rule as well.
	if loadBalancerIP != "" && loadBalancerIP != fwd.IPAddress {
		klog.Infof("LoadBalancer ip for forwarding rule %v was expected to be %v, but was actually %v", fwd.Name, fwd.IPAddress, loadBalancerIP)
		return true, true, fwd.IPAddress, nil
	}
	portRange, err := loadBalancerPortRange(ports)
	if err != nil {
		// Err on the side of caution in case of errors. Caller should notice the error and retry.
		// We never want to end up recreating resources because g api flaked.
		return true, false, "", err
	}
	if portRange != fwd.PortRange {
		klog.Infof("LoadBalancer port range for forwarding rule %v was expected to be %v, but was actually %v", fwd.Name, fwd.PortRange, portRange)
		return true, true, fwd.IPAddress, nil
	}
	// The service controller verified all the protocols match on the ports, just check the first one
	if string(ports[0].Protocol) != fwd.IPProtocol {
		klog.Infof("LoadBalancer protocol for forwarding rule %v was expected to be %v, but was actually %v", fwd.Name, fwd.IPProtocol, string(ports[0].Protocol))
		return true, true, fwd.IPAddress, nil
	}

	return true, false, fwd.IPAddress, nil
}

// Doesn't check whether the hosts have changed, since host updating is handled
// separately.
func (g *Cloud) targetPoolNeedsRecreation(name, region string, affinityType v1.ServiceAffinity) (exists bool, needsRecreation bool, err error) {
	tp, err := g.GetTargetPool(name, region)
	if err != nil {
		if isHTTPErrorCode(err, http.StatusNotFound) {
			return false, true, nil
		}
		// Err on the side of caution in case of errors. Caller should notice the error and retry.
		// We never want to end up recreating resources because g api flaked.
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
		klog.Infof("LoadBalancer target pool %v changed affinity from %v to %v", name, tp.SessionAffinity, affinityType)
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
		return "", fmt.Errorf("invalid protocol %s, only TCP and UDP are supported", string(ports[0].Protocol))
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
		klog.Errorf("Unexpected affinity type: %v", affinityType)
		return gceAffinityTypeNone
	}
}

func (g *Cloud) firewallNeedsUpdate(name, serviceName, region, ipAddress string, ports []v1.ServicePort, sourceRanges utilnet.IPNetSet) (exists bool, needsUpdate bool, err error) {
	fw, err := g.GetFirewall(MakeFirewallName(name))
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

	actualSourceRanges, err := utilnet.ParseIPNets(fw.SourceRanges...)
	if err != nil {
		// This really shouldn't happen... GCE has returned something unexpected
		klog.Warningf("Error parsing firewall SourceRanges: %v", fw.SourceRanges)
		// We don't return the error, because we can hopefully recover from this by reconfiguring the firewall
		return true, true, nil
	}

	if !sourceRanges.Equal(actualSourceRanges) {
		return true, true, nil
	}
	return true, false, nil
}

func (g *Cloud) ensureHTTPHealthCheckFirewall(svc *v1.Service, serviceName, ipAddress, region, clusterID string, hosts []*gceInstance, hcName string, hcPort int32, isNodesHealthCheck bool) error {
	// Prepare the firewall params for creating / checking.
	desc := fmt.Sprintf(`{"kubernetes.io/cluster-id":"%s"}`, clusterID)
	if !isNodesHealthCheck {
		desc = makeFirewallDescription(serviceName, ipAddress)
	}
	sourceRanges := lbSrcRngsFlag.ipn
	ports := []v1.ServicePort{{Protocol: "tcp", Port: hcPort}}

	fwName := MakeHealthCheckFirewallName(clusterID, hcName, isNodesHealthCheck)
	fw, err := g.GetFirewall(fwName)
	if err != nil {
		if !isHTTPErrorCode(err, http.StatusNotFound) {
			return fmt.Errorf("error getting firewall for health checks: %v", err)
		}
		klog.Infof("Creating firewall %v for health checks.", fwName)
		if err := g.createFirewall(svc, fwName, region, desc, sourceRanges, ports, hosts); err != nil {
			return err
		}
		klog.Infof("Created firewall %v for health checks.", fwName)
		return nil
	}
	// Validate firewall fields.
	if fw.Description != desc ||
		len(fw.Allowed) != 1 ||
		fw.Allowed[0].IPProtocol != string(ports[0].Protocol) ||
		!equalStringSets(fw.Allowed[0].Ports, []string{strconv.Itoa(int(ports[0].Port))}) ||
		!equalStringSets(fw.SourceRanges, sourceRanges.StringSlice()) {
		klog.Warningf("Firewall %v exists but parameters have drifted - updating...", fwName)
		if err := g.updateFirewall(svc, fwName, region, desc, sourceRanges, ports, hosts); err != nil {
			klog.Warningf("Failed to reconcile firewall %v parameters.", fwName)
			return err
		}
		klog.V(4).Infof("Corrected firewall %v parameters successful", fwName)
	}
	return nil
}

func createForwardingRule(s CloudForwardingRuleService, name, serviceName, region, ipAddress, target string, ports []v1.ServicePort, netTier cloud.NetworkTier) error {
	portRange, err := loadBalancerPortRange(ports)
	if err != nil {
		return err
	}
	desc := makeServiceDescription(serviceName)
	ipProtocol := string(ports[0].Protocol)

	switch netTier {
	case cloud.NetworkTierPremium:
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

func (g *Cloud) createFirewall(svc *v1.Service, name, region, desc string, sourceRanges utilnet.IPNetSet, ports []v1.ServicePort, hosts []*gceInstance) error {
	firewall, err := g.firewallObject(name, region, desc, sourceRanges, ports, hosts)
	if err != nil {
		return err
	}
	if err = g.CreateFirewall(firewall); err != nil {
		if isHTTPErrorCode(err, http.StatusConflict) {
			return nil
		} else if isForbidden(err) && g.OnXPN() {
			klog.V(4).Infof("createFirewall(%v): do not have permission to create firewall rule (on XPN). Raising event.", firewall.Name)
			g.raiseFirewallChangeNeededEvent(svc, FirewallToGCloudCreateCmd(firewall, g.NetworkProjectID()))
			return nil
		}
		return err
	}
	return nil
}

func (g *Cloud) updateFirewall(svc *v1.Service, name, region, desc string, sourceRanges utilnet.IPNetSet, ports []v1.ServicePort, hosts []*gceInstance) error {
	firewall, err := g.firewallObject(name, region, desc, sourceRanges, ports, hosts)
	if err != nil {
		return err
	}

	if err = g.UpdateFirewall(firewall); err != nil {
		if isHTTPErrorCode(err, http.StatusConflict) {
			return nil
		} else if isForbidden(err) && g.OnXPN() {
			klog.V(4).Infof("updateFirewall(%v): do not have permission to update firewall rule (on XPN). Raising event.", firewall.Name)
			g.raiseFirewallChangeNeededEvent(svc, FirewallToGCloudUpdateCmd(firewall, g.NetworkProjectID()))
			return nil
		}
		return err
	}
	return nil
}

func (g *Cloud) firewallObject(name, region, desc string, sourceRanges utilnet.IPNetSet, ports []v1.ServicePort, hosts []*gceInstance) (*compute.Firewall, error) {
	allowedPorts := make([]string, len(ports))
	for ix := range ports {
		allowedPorts[ix] = strconv.Itoa(int(ports[ix].Port))
	}
	// If the node tags to be used for this cluster have been predefined in the
	// provider config, just use them. Otherwise, invoke computeHostTags method to get the tags.
	hostTags := g.nodeTags
	if len(hostTags) == 0 {
		var err error
		if hostTags, err = g.computeHostTags(hosts); err != nil {
			return nil, fmt.Errorf("no node tags supplied and also failed to parse the given lists of hosts for tags. Abort creating firewall rule")
		}
	}

	firewall := &compute.Firewall{
		Name:         name,
		Description:  desc,
		Network:      g.networkURL,
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

func ensureStaticIP(s CloudAddressService, name, serviceName, region, existingIP string, netTier cloud.NetworkTier) (ipAddress string, existing bool, err error) {
	// If the address doesn't exist, this will create it.
	// If the existingIP exists but is ephemeral, this will promote it to static.
	// If the address already exists, this will harmlessly return a StatusConflict
	// and we'll grab the IP before returning.
	existed := false
	desc := makeServiceDescription(serviceName)

	var creationErr error
	switch netTier {
	case cloud.NetworkTierPremium:
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

func (g *Cloud) getServiceNetworkTier(svc *v1.Service) (cloud.NetworkTier, error) {
	if !g.AlphaFeatureGate.Enabled(AlphaFeatureNetworkTiers) {
		return cloud.NetworkTierDefault, nil
	}
	tier, err := GetServiceNetworkTier(svc)
	if err != nil {
		// Returns an error if the annotation is invalid.
		return cloud.NetworkTier(""), err
	}
	return tier, nil
}

func (g *Cloud) deleteWrongNetworkTieredResources(lbName, lbRef string, desiredNetTier cloud.NetworkTier) error {
	logPrefix := fmt.Sprintf("deleteWrongNetworkTieredResources:(%s)", lbRef)
	if err := deleteFWDRuleWithWrongTier(g, g.region, lbName, logPrefix, desiredNetTier); err != nil {
		return err
	}
	if err := deleteAddressWithWrongTier(g, g.region, lbName, logPrefix, desiredNetTier); err != nil {
		return err
	}
	return nil
}

// deleteFWDRuleWithWrongTier checks the network tier of existing forwarding
// rule and delete the rule if the tier does not matched the desired tier.
func deleteFWDRuleWithWrongTier(s CloudForwardingRuleService, region, name, logPrefix string, desiredNetTier cloud.NetworkTier) error {
	tierStr, err := s.getNetworkTierFromForwardingRule(name, region)
	if isNotFound(err) {
		return nil
	} else if err != nil {
		return err
	}
	existingTier := cloud.NetworkTierGCEValueToType(tierStr)
	if existingTier == desiredNetTier {
		return nil
	}
	klog.V(2).Infof("%s: Network tiers do not match; existing forwarding rule: %q, desired: %q. Deleting the forwarding rule",
		logPrefix, existingTier, desiredNetTier)
	err = s.DeleteRegionForwardingRule(name, region)
	return ignoreNotFound(err)
}

// deleteAddressWithWrongTier checks the network tier of existing address
// and delete the address if the tier does not matched the desired tier.
func deleteAddressWithWrongTier(s CloudAddressService, region, name, logPrefix string, desiredNetTier cloud.NetworkTier) error {
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
	existingTier := cloud.NetworkTierGCEValueToType(tierStr)
	if existingTier == desiredNetTier {
		return nil
	}
	klog.V(2).Infof("%s: Network tiers do not match; existing address: %q, desired: %q. Deleting the address",
		logPrefix, existingTier, desiredNetTier)
	err = s.DeleteRegionAddress(name, region)
	return ignoreNotFound(err)
}
