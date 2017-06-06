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

	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/api/v1"
	apiservice "k8s.io/kubernetes/pkg/api/v1/service"
	"k8s.io/kubernetes/pkg/cloudprovider"
	netsets "k8s.io/kubernetes/pkg/util/net/sets"

	"github.com/golang/glog"
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
func (gce *GCECloud) ensureExternalLoadBalancer(clusterName string, apiService *v1.Service, existingFwdRule *compute.ForwardingRule, nodes []*v1.Node) (*v1.LoadBalancerStatus, error) {
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
	loadBalancerIP := apiService.Spec.LoadBalancerIP
	ports := apiService.Spec.Ports
	portStr := []string{}
	for _, p := range apiService.Spec.Ports {
		portStr = append(portStr, fmt.Sprintf("%s/%d", p.Protocol, p.Port))
	}

	affinityType := apiService.Spec.SessionAffinity

	serviceName := types.NamespacedName{Namespace: apiService.Namespace, Name: apiService.Name}
	glog.V(2).Infof("EnsureLoadBalancer(%v, %v, %v, %v, %v, %v, %v)",
		loadBalancerName, gce.region, loadBalancerIP, portStr, hostNames, serviceName, apiService.Annotations)

	// Check if the forwarding rule exists, and if so, what its IP is.
	fwdRuleExists, fwdRuleNeedsUpdate, fwdRuleIP, err := gce.forwardingRuleNeedsUpdate(loadBalancerName, gce.region, loadBalancerIP, ports)
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
	ipAddress := ""

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
				glog.Errorf("failed to release static IP %s for load balancer (%v(%v), %v): %v", ipAddress, loadBalancerName, serviceName, gce.region, err)
			} else if isNotFound(err) {
				glog.V(2).Infof("EnsureLoadBalancer(%v(%v)): address %s is not reserved.", loadBalancerName, serviceName, ipAddress)
			} else {
				glog.V(2).Infof("EnsureLoadBalancer(%v(%v)): released static IP %s", loadBalancerName, serviceName, ipAddress)
			}
		} else {
			glog.Warningf("orphaning static IP %s during update of load balancer (%v(%v), %v): %v", ipAddress, loadBalancerName, serviceName, gce.region, err)
		}
	}()

	if loadBalancerIP != "" {
		// If a specific IP address has been requested, we have to respect the
		// user's request and use that IP. If the forwarding rule was already using
		// a different IP, it will be harmlessly abandoned because it was only an
		// ephemeral IP (or it was a different static IP owned by the user, in which
		// case we shouldn't delete it anyway).
		if isStatic, err := gce.projectOwnsStaticIP(loadBalancerName, gce.region, loadBalancerIP); err != nil {
			return nil, fmt.Errorf("failed to test if this GCE project owns the static IP %s: %v", loadBalancerIP, err)
		} else if isStatic {
			// The requested IP is a static IP, owned and managed by the user.
			isUserOwnedIP = true
			isSafeToReleaseIP = false
			ipAddress = loadBalancerIP
			glog.V(4).Infof("EnsureLoadBalancer(%v(%v)): using user-provided static IP %s", loadBalancerName, serviceName, ipAddress)
		} else if loadBalancerIP == fwdRuleIP {
			// The requested IP is not a static IP, but is currently assigned
			// to this forwarding rule, so we can keep it.
			isUserOwnedIP = false
			isSafeToReleaseIP = true
			ipAddress, _, err = gce.ensureStaticIP(loadBalancerName, serviceName.String(), gce.region, fwdRuleIP)
			if err != nil {
				return nil, fmt.Errorf("failed to ensure static IP %s: %v", fwdRuleIP, err)
			}
			glog.V(4).Infof("EnsureLoadBalancer(%v(%v)): using user-provided non-static IP %s", loadBalancerName, serviceName, ipAddress)
		} else {
			// The requested IP is not static and it is not assigned to the
			// current forwarding rule.  It might be attached to a different
			// rule or it might not be part of this project at all.  Either
			// way, we can't use it.
			return nil, fmt.Errorf("requested ip %s is neither static nor assigned to LB %s(%v): %v", loadBalancerIP, loadBalancerName, serviceName, err)
		}
	} else {
		// The user did not request a specific IP.
		isUserOwnedIP = false

		// This will either allocate a new static IP if the forwarding rule didn't
		// already have an IP, or it will promote the forwarding rule's current
		// IP from ephemeral to static, or it will just get the IP if it is
		// already static.
		existed := false
		ipAddress, existed, err = gce.ensureStaticIP(loadBalancerName, serviceName.String(), gce.region, fwdRuleIP)
		if err != nil {
			return nil, fmt.Errorf("failed to ensure static IP %s: %v", fwdRuleIP, err)
		}
		if existed {
			// If the IP was not specifically requested by the user, but it
			// already existed, it seems to be a failed update cycle.  We can
			// use this IP and try to run through the process again, but we
			// should not release the IP unless it is explicitly flagged as OK.
			isSafeToReleaseIP = false
			glog.V(4).Infof("EnsureLoadBalancer(%v(%v)): adopting static IP %s", loadBalancerName, serviceName, ipAddress)
		} else {
			// For total clarity.  The IP did not pre-exist and the user did
			// not ask for a particular one, so we can release the IP in case
			// of failure or success.
			isSafeToReleaseIP = true
			glog.V(4).Infof("EnsureLoadBalancer(%v(%v)): allocated static IP %s", loadBalancerName, serviceName, ipAddress)
		}
	}

	// Deal with the firewall next. The reason we do this here rather than last
	// is because the forwarding rule is used as the indicator that the load
	// balancer is fully created - it's what getLoadBalancer checks for.
	// Check if user specified the allow source range
	sourceRanges, err := apiservice.GetLoadBalancerSourceRanges(apiService)
	if err != nil {
		return nil, err
	}

	firewallExists, firewallNeedsUpdate, err := gce.firewallNeedsUpdate(loadBalancerName, serviceName.String(), gce.region, ipAddress, ports, sourceRanges)
	if err != nil {
		return nil, err
	}

	if firewallNeedsUpdate {
		desc := makeFirewallDescription(serviceName.String(), ipAddress)
		// Unlike forwarding rules and target pools, firewalls can be updated
		// without needing to be deleted and recreated.
		if firewallExists {
			glog.Infof("EnsureLoadBalancer(%v(%v)): updating firewall", loadBalancerName, serviceName)
			if err := gce.updateFirewall(makeFirewallName(loadBalancerName), gce.region, desc, sourceRanges, ports, hosts); err != nil {
				return nil, err
			}
			glog.Infof("EnsureLoadBalancer(%v(%v)): updated firewall", loadBalancerName, serviceName)
		} else {
			glog.Infof("EnsureLoadBalancer(%v(%v)): creating firewall", loadBalancerName, serviceName)
			if err := gce.createFirewall(makeFirewallName(loadBalancerName), gce.region, desc, sourceRanges, ports, hosts); err != nil {
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

	clusterID, err := gce.ClusterID.GetID()
	if err != nil {
		return nil, fmt.Errorf("error getting cluster ID %s: %v", loadBalancerName, err)
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
				hcToDelete = makeHttpHealthCheck(makeNodesHealthCheckName(clusterID), GetNodesHealthCheckPath(), GetNodesHealthCheckPort())
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
			hcToCreate = makeHttpHealthCheck(makeNodesHealthCheckName(clusterID), GetNodesHealthCheckPath(), GetNodesHealthCheckPort())
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
		if err := gce.DeleteExternalTargetPoolAndChecks(loadBalancerName, gce.region, hcNames...); err != nil {
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
		if err := gce.createTargetPool(loadBalancerName, serviceName.String(), ipAddress, gce.region, createInstances, affinityType, hcToCreate); err != nil {
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
		glog.Infof("EnsureLoadBalancer(%v(%v)): creating forwarding rule, IP %s", loadBalancerName, serviceName, ipAddress)
		if err := gce.createForwardingRule(loadBalancerName, serviceName.String(), gce.region, ipAddress, ports); err != nil {
			return nil, fmt.Errorf("failed to create forwarding rule %s: %v", loadBalancerName, err)
		}
		// End critical section.  It is safe to release the static IP (which
		// just demotes it to ephemeral) now that it is attached.  In the case
		// of a user-requested IP, the "is user-owned" flag will be set,
		// preventing it from actually being released.
		isSafeToReleaseIP = true
		glog.Infof("EnsureLoadBalancer(%v(%v)): created forwarding rule, IP %s", loadBalancerName, serviceName, ipAddress)
	}

	status := &v1.LoadBalancerStatus{}
	status.Ingress = []v1.LoadBalancerIngress{{IP: ipAddress}}

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
func (gce *GCECloud) ensureExternalLoadBalancerDeleted(clusterName string, service *v1.Service) error {
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
		clusterID, err := gce.ClusterID.GetID()
		if err != nil {
			return fmt.Errorf("error getting cluster ID %s: %v", loadBalancerName, err)
		}
		// EnsureLoadBalancerDeleted() could be triggered by changing service from
		// LoadBalancer type to others. In this case we have no idea whether it was
		// using local traffic health check or nodes health check. Attempt to delete
		// both to prevent leaking.
		hcNames = append(hcNames, loadBalancerName)
		hcNames = append(hcNames, makeNodesHealthCheckName(clusterID))
	}

	errs := utilerrors.AggregateGoroutines(
		func() error { return ignoreNotFound(gce.DeleteFirewall(makeFirewallName(loadBalancerName))) },
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
			if err := gce.DeleteExternalTargetPoolAndChecks(loadBalancerName, gce.region, hcNames...); err != nil {
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

func (gce *GCECloud) DeleteExternalTargetPoolAndChecks(name, region string, hcNames ...string) error {
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
			clusterID, err := gce.ClusterID.GetID()
			if err != nil {
				return fmt.Errorf("error getting cluster ID: %v", err)
			}
			// If health check is deleted without error, it means no load-balancer is using it.
			// So we should delete the health check firewall as well.
			fwName := MakeHealthCheckFirewallName(clusterID, hcName, isNodesHealthCheck)
			glog.Infof("Deleting firewall %v.", fwName)
			if err := gce.DeleteFirewall(fwName); err != nil {
				if isHTTPErrorCode(err, http.StatusNotFound) {
					glog.V(4).Infof("Firewall %v is already deleted.", fwName)
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

func (gce *GCECloud) createTargetPool(name, serviceName, ipAddress, region string, hosts []*gceInstance, affinityType v1.ServiceAffinity, hc *compute.HttpHealthCheck) error {
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
		if !gce.OnXPN() {
			if err := gce.ensureHttpHealthCheckFirewall(serviceName, ipAddress, region, hosts, hc.Name, int32(hc.Port), isNodesHealthCheck); err != nil {
				return err
			}
		}
		var err error
		if hc, err = gce.ensureHttpHealthCheck(hc.Name, hc.RequestPath, int32(hc.Port)); err != nil || hc == nil {
			return fmt.Errorf("Failed to ensure health check for %v port %d path %v: %v", name, hc.Port, hc.RequestPath, err)
		}
		hcLinks = append(hcLinks, hc.SelfLink)
	}

	var instances []string
	for _, host := range hosts {
		instances = append(instances, makeHostURL(gce.projectID, host.Zone, host.Name))
	}
	glog.Infof("Creating targetpool %v with %d healthchecks", name, len(hcLinks))
	pool := &compute.TargetPool{
		Name:            name,
		Description:     fmt.Sprintf(`{"kubernetes.io/service-name":"%s"}`, serviceName),
		Instances:       instances,
		SessionAffinity: translateAffinityType(affinityType),
		HealthChecks:    hcLinks,
	}

	if _, err := gce.CreateTargetPool(pool, region); err != nil && !isHTTPErrorCode(err, http.StatusConflict) {
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

func (gce *GCECloud) targetPoolURL(name, region string) string {
	return fmt.Sprintf("https://www.googleapis.com/compute/v1/projects/%s/regions/%s/targetPools/%s", gce.projectID, region, name)
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

func makeHostURL(projectID, zone, host string) string {
	host = canonicalizeInstanceName(host)
	return fmt.Sprintf("https://www.googleapis.com/compute/v1/projects/%s/zones/%s/instances/%s",
		projectID, zone, host)
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
	if gce.OnXPN() {
		glog.V(2).Infoln("firewallNeedsUpdate: Cluster is on XPN network - skipping firewall creation")
		return false, false, nil
	}

	fw, err := gce.service.Firewalls.Get(gce.projectID, makeFirewallName(name)).Do()
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

func (gce *GCECloud) ensureHttpHealthCheckFirewall(serviceName, ipAddress, region string, hosts []*gceInstance, hcName string, hcPort int32, isNodesHealthCheck bool) error {
	clusterID, err := gce.ClusterID.GetID()
	if err != nil {
		return fmt.Errorf("error getting cluster ID: %v", err)
	}

	// Prepare the firewall params for creating / checking.
	desc := fmt.Sprintf(`{"kubernetes.io/cluster-id":"%s"}`, clusterID)
	if !isNodesHealthCheck {
		desc = makeFirewallDescription(serviceName, ipAddress)
	}
	sourceRanges := lbSrcRngsFlag.ipn
	ports := []v1.ServicePort{{Protocol: "tcp", Port: hcPort}}

	fwName := MakeHealthCheckFirewallName(clusterID, hcName, isNodesHealthCheck)
	fw, err := gce.service.Firewalls.Get(gce.projectID, fwName).Do()
	if err != nil {
		if !isHTTPErrorCode(err, http.StatusNotFound) {
			return fmt.Errorf("error getting firewall for health checks: %v", err)
		}
		glog.Infof("Creating firewall %v for health checks.", fwName)
		if err := gce.createFirewall(fwName, region, desc, sourceRanges, ports, hosts); err != nil {
			return err
		}
		glog.Infof("Created firewall %v for health checks.", fwName)
		return nil
	}
	// Validate firewall fields.
	if fw.Description != desc ||
		len(fw.Allowed) != 1 ||
		fw.Allowed[0].IPProtocol != string(ports[0].Protocol) ||
		!equalStringSets(fw.Allowed[0].Ports, []string{string(ports[0].Port)}) ||
		!equalStringSets(fw.SourceRanges, sourceRanges.StringSlice()) {
		glog.Warningf("Firewall %v exists but parameters have drifted - updating...", fwName)
		if err := gce.updateFirewall(fwName, region, desc, sourceRanges, ports, hosts); err != nil {
			glog.Warningf("Failed to reconcile firewall %v parameters.", fwName)
			return err
		}
		glog.V(4).Infof("Corrected firewall %v parameters successful", fwName)
	}
	return nil
}

func (gce *GCECloud) createForwardingRule(name, serviceName, region, ipAddress string, ports []v1.ServicePort) error {
	portRange, err := loadBalancerPortRange(ports)
	if err != nil {
		return err
	}
	req := &compute.ForwardingRule{
		Name:        name,
		Description: fmt.Sprintf(`{"kubernetes.io/service-name":"%s"}`, serviceName),
		IPAddress:   ipAddress,
		IPProtocol:  string(ports[0].Protocol),
		PortRange:   portRange,
		Target:      gce.targetPoolURL(name, region),
	}

	if err = gce.CreateRegionForwardingRule(req, region); err != nil && !isHTTPErrorCode(err, http.StatusConflict) {
		return err
	}
	return nil
}

func (gce *GCECloud) createFirewall(name, region, desc string, sourceRanges netsets.IPNet, ports []v1.ServicePort, hosts []*gceInstance) error {
	firewall, err := gce.firewallObject(name, region, desc, sourceRanges, ports, hosts)
	if err != nil {
		return err
	}
	if err = gce.CreateFirewall(firewall); err != nil && !isHTTPErrorCode(err, http.StatusConflict) {
		return err
	}
	return nil
}

func (gce *GCECloud) updateFirewall(name, region, desc string, sourceRanges netsets.IPNet, ports []v1.ServicePort, hosts []*gceInstance) error {
	firewall, err := gce.firewallObject(name, region, desc, sourceRanges, ports, hosts)
	if err != nil {
		return err
	}

	if err = gce.UpdateFirewall(firewall); err != nil && !isHTTPErrorCode(err, http.StatusConflict) {
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

func (gce *GCECloud) projectOwnsStaticIP(name, region string, ipAddress string) (bool, error) {
	pageToken := ""
	page := 0
	for ; page == 0 || (pageToken != "" && page < maxPages); page++ {
		listCall := gce.service.Addresses.List(gce.projectID, region)
		if pageToken != "" {
			listCall = listCall.PageToken(pageToken)
		}
		addresses, err := listCall.Do()
		if err != nil {
			return false, fmt.Errorf("failed to list gce IP addresses: %v", err)
		}
		pageToken = addresses.NextPageToken
		for _, addr := range addresses.Items {
			if addr.Address == ipAddress {
				// This project does own the address, so return success.
				return true, nil
			}
		}
	}
	if page >= maxPages {
		glog.Errorf("projectOwnsStaticIP exceeded maxPages=%d for Addresses.List; truncating.", maxPages)
	}
	return false, nil
}

func (gce *GCECloud) ensureStaticIP(name, serviceName, region, existingIP string) (ipAddress string, created bool, err error) {
	// If the address doesn't exist, this will create it.
	// If the existingIP exists but is ephemeral, this will promote it to static.
	// If the address already exists, this will harmlessly return a StatusConflict
	// and we'll grab the IP before returning.
	existed := false
	addressObj := &compute.Address{
		Name:        name,
		Description: fmt.Sprintf(`{"kubernetes.io/service-name":"%s"}`, serviceName),
	}

	if existingIP != "" {
		addressObj.Address = existingIP
	}

	address, err := gce.ReserveRegionAddress(addressObj, region)
	if err != nil {
		if !isHTTPErrorCode(err, http.StatusConflict) {
			return "", false, fmt.Errorf("error creating gce static IP address: %v", err)
		}
		// StatusConflict == the IP exists already.
		existed = true
	}

	return address.Address, existed, nil
}
