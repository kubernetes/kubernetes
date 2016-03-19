/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"path"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/service"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	netsets "k8s.io/kubernetes/pkg/util/net/sets"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/wait"

	"github.com/golang/glog"
	"github.com/scalingdata/gcfg"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	compute "google.golang.org/api/compute/v1"
	container "google.golang.org/api/container/v1"
	"google.golang.org/api/googleapi"
	"google.golang.org/cloud/compute/metadata"
)

const (
	ProviderName = "gce"

	k8sNodeRouteTag = "k8s-node-route"

	// AffinityTypeNone - no session affinity.
	gceAffinityTypeNone = "NONE"
	// AffinityTypeClientIP - affinity based on Client IP.
	gceAffinityTypeClientIP = "CLIENT_IP"
	// AffinityTypeClientIPProto - affinity based on Client IP and port.
	gceAffinityTypeClientIPProto = "CLIENT_IP_PROTO"

	operationPollInterval        = 3 * time.Second
	operationPollTimeoutDuration = 30 * time.Minute

	// Each page can have 500 results, but we cap how many pages
	// are iterated through to prevent infinite loops if the API
	// were to continuously return a nextPageToken.
	maxPages = 25
)

// GCECloud is an implementation of Interface, LoadBalancer and Instances for Google Compute Engine.
type GCECloud struct {
	service                  *compute.Service
	containerService         *container.Service
	projectID                string
	region                   string
	localZone                string   // The zone in which we are running
	managedZones             []string // List of zones we are spanning (for Ubernetes-Lite, primarily when running on master)
	networkURL               string
	useMetadataServer        bool
	operationPollRateLimiter util.RateLimiter
}

type Config struct {
	Global struct {
		TokenURL    string `gcfg:"token-url"`
		TokenBody   string `gcfg:"token-body"`
		ProjectID   string `gcfg:"project-id"`
		NetworkName string `gcfg:"network-name"`
		Multizone   bool   `gcfg:"multizone"`
	}
}

func init() {
	cloudprovider.RegisterCloudProvider(ProviderName, func(config io.Reader) (cloudprovider.Interface, error) { return newGCECloud(config) })
}

// Raw access to the underlying GCE service, probably should only be used for e2e tests
func (g *GCECloud) GetComputeService() *compute.Service {
	return g.service
}

func getProjectAndZone() (string, string, error) {
	result, err := metadata.Get("instance/zone")
	if err != nil {
		return "", "", err
	}
	parts := strings.Split(result, "/")
	if len(parts) != 4 {
		return "", "", fmt.Errorf("unexpected response: %s", result)
	}
	zone := parts[3]
	projectID, err := metadata.ProjectID()
	if err != nil {
		return "", "", err
	}
	return projectID, zone, nil
}

func getInstanceIDViaMetadata() (string, error) {
	result, err := metadata.Get("instance/hostname")
	if err != nil {
		return "", err
	}
	parts := strings.Split(result, ".")
	if len(parts) == 0 {
		return "", fmt.Errorf("unexpected response: %s", result)
	}
	return parts[0], nil
}

func getCurrentExternalIDViaMetadata() (string, error) {
	externalID, err := metadata.Get("instance/id")
	if err != nil {
		return "", fmt.Errorf("couldn't get external ID: %v", err)
	}
	return externalID, nil
}

func getCurrentMachineTypeViaMetadata() (string, error) {
	mType, err := metadata.Get("instance/machine-type")
	if err != nil {
		return "", fmt.Errorf("couldn't get machine type: %v", err)
	}
	parts := strings.Split(mType, "/")
	if len(parts) != 4 {
		return "", fmt.Errorf("unexpected response for machine type: %s", mType)
	}

	return parts[3], nil
}

func getNetworkNameViaMetadata() (string, error) {
	result, err := metadata.Get("instance/network-interfaces/0/network")
	if err != nil {
		return "", err
	}
	parts := strings.Split(result, "/")
	if len(parts) != 4 {
		return "", fmt.Errorf("unexpected response: %s", result)
	}
	return parts[3], nil
}

func getNetworkNameViaAPICall(svc *compute.Service, projectID string) (string, error) {
	// TODO: use PageToken to list all not just the first 500
	networkList, err := svc.Networks.List(projectID).Do()
	if err != nil {
		return "", err
	}

	if networkList == nil || len(networkList.Items) <= 0 {
		return "", fmt.Errorf("GCE Network List call returned no networks for project %q.", projectID)
	}

	return networkList.Items[0].Name, nil
}

func getZonesForRegion(svc *compute.Service, projectID, region string) ([]string, error) {
	// TODO: use PageToken to list all not just the first 500
	listCall := svc.Zones.List(projectID)

	// Filtering by region doesn't seem to work
	// (tested in https://cloud.google.com/compute/docs/reference/latest/zones/list)
	// listCall = listCall.Filter("region eq " + region)

	res, err := listCall.Do()
	if err != nil {
		return nil, fmt.Errorf("unexpected response listing zones: %v", err)
	}
	zones := []string{}
	for _, zone := range res.Items {
		regionName := lastComponent(zone.Region)
		if regionName == region {
			zones = append(zones, zone.Name)
		}
	}
	return zones, nil
}

// newGCECloud creates a new instance of GCECloud.
func newGCECloud(config io.Reader) (*GCECloud, error) {
	projectID, zone, err := getProjectAndZone()
	if err != nil {
		return nil, err
	}

	region, err := GetGCERegion(zone)
	if err != nil {
		return nil, err
	}

	networkName, err := getNetworkNameViaMetadata()
	if err != nil {
		return nil, err
	}
	networkURL := gceNetworkURL(projectID, networkName)

	// By default, Kubernetes clusters only run against one zone
	managedZones := []string{zone}

	tokenSource := google.ComputeTokenSource("")
	if config != nil {
		var cfg Config
		if err := gcfg.ReadInto(&cfg, config); err != nil {
			glog.Errorf("Couldn't read config: %v", err)
			return nil, err
		}
		if cfg.Global.ProjectID != "" {
			projectID = cfg.Global.ProjectID
		}
		if cfg.Global.NetworkName != "" {
			if strings.Contains(cfg.Global.NetworkName, "/") {
				networkURL = cfg.Global.NetworkName
			} else {
				networkURL = gceNetworkURL(cfg.Global.ProjectID, cfg.Global.NetworkName)
			}
		}
		if cfg.Global.TokenURL != "" {
			tokenSource = newAltTokenSource(cfg.Global.TokenURL, cfg.Global.TokenBody)
		}
		if cfg.Global.Multizone {
			managedZones = nil // Use all zones in region
		}
	}

	return CreateGCECloud(projectID, region, zone, managedZones, networkURL, tokenSource, true /* useMetadataServer */)
}

// Creates a GCECloud object using the specified parameters.
// If no networkUrl is specified, loads networkName via rest call.
// If no tokenSource is specified, uses oauth2.DefaultTokenSource.
// If managedZones is nil / empty all zones in the region will be managed.
func CreateGCECloud(projectID, region, zone string, managedZones []string, networkURL string, tokenSource oauth2.TokenSource, useMetadataServer bool) (*GCECloud, error) {
	if tokenSource == nil {
		var err error
		tokenSource, err = google.DefaultTokenSource(
			oauth2.NoContext,
			compute.CloudPlatformScope,
			compute.ComputeScope)
		glog.Infof("Using DefaultTokenSource %#v", tokenSource)
		if err != nil {
			return nil, err
		}
	} else {
		glog.Infof("Using existing Token Source %#v", tokenSource)
	}

	client := oauth2.NewClient(oauth2.NoContext, tokenSource)
	svc, err := compute.New(client)
	if err != nil {
		return nil, err
	}

	containerSvc, err := container.New(client)
	if err != nil {
		return nil, err
	}

	if networkURL == "" {
		networkName, err := getNetworkNameViaAPICall(svc, projectID)
		if err != nil {
			return nil, err
		}
		networkURL = gceNetworkURL(projectID, networkName)
	}

	if len(managedZones) == 0 {
		managedZones, err = getZonesForRegion(svc, projectID, region)
		if err != nil {
			return nil, err
		}
	}
	if len(managedZones) != 1 {
		glog.Infof("managing multiple zones: %v", managedZones)
	}

	operationPollRateLimiter := util.NewTokenBucketRateLimiter(10, 100) // 10 qps, 100 bucket size.

	return &GCECloud{
		service:                  svc,
		containerService:         containerSvc,
		projectID:                projectID,
		region:                   region,
		localZone:                zone,
		managedZones:             managedZones,
		networkURL:               networkURL,
		useMetadataServer:        useMetadataServer,
		operationPollRateLimiter: operationPollRateLimiter,
	}, nil
}

func (gce *GCECloud) Clusters() (cloudprovider.Clusters, bool) {
	return gce, true
}

// ProviderName returns the cloud provider ID.
func (gce *GCECloud) ProviderName() string {
	return ProviderName
}

// Known-useless DNS search path.
var uselessDNSSearchRE = regexp.MustCompile(`^[0-9]+.google.internal.$`)

// ScrubDNS filters DNS settings for pods.
func (gce *GCECloud) ScrubDNS(nameservers, searches []string) (nsOut, srchOut []string) {
	// GCE has too many search paths by default. Filter the ones we know are useless.
	for _, s := range searches {
		if !uselessDNSSearchRE.MatchString(s) {
			srchOut = append(srchOut, s)
		}
	}
	return nameservers, srchOut
}

// LoadBalancer returns an implementation of LoadBalancer for Google Compute Engine.
func (gce *GCECloud) LoadBalancer() (cloudprovider.LoadBalancer, bool) {
	return gce, true
}

// Instances returns an implementation of Instances for Google Compute Engine.
func (gce *GCECloud) Instances() (cloudprovider.Instances, bool) {
	return gce, true
}

// Zones returns an implementation of Zones for Google Compute Engine.
func (gce *GCECloud) Zones() (cloudprovider.Zones, bool) {
	return gce, true
}

// Routes returns an implementation of Routes for Google Compute Engine.
func (gce *GCECloud) Routes() (cloudprovider.Routes, bool) {
	return gce, true
}

func makeHostURL(projectID, zone, host string) string {
	host = canonicalizeInstanceName(host)
	return fmt.Sprintf("https://www.googleapis.com/compute/v1/projects/%s/zones/%s/instances/%s",
		projectID, zone, host)
}

func (h *gceInstance) makeComparableHostPath() string {
	return fmt.Sprintf("/zones/%s/instances/%s", h.Zone, h.Name)
}

func hostURLToComparablePath(hostURL string) string {
	idx := strings.Index(hostURL, "/zones/")
	if idx < 0 {
		return ""
	}
	return hostURL[idx:]
}

func (gce *GCECloud) targetPoolURL(name, region string) string {
	return fmt.Sprintf("https://www.googleapis.com/compute/v1/projects/%s/regions/%s/targetPools/%s", gce.projectID, region, name)
}

func (gce *GCECloud) waitForOp(op *compute.Operation, getOperation func(operationName string) (*compute.Operation, error)) error {
	if op == nil {
		return fmt.Errorf("operation must not be nil")
	}

	if opIsDone(op) {
		return getErrorFromOp(op)
	}

	opName := op.Name
	return wait.Poll(operationPollInterval, operationPollTimeoutDuration, func() (bool, error) {
		start := time.Now()
		gce.operationPollRateLimiter.Accept()
		duration := time.Now().Sub(start)
		if duration > 5*time.Second {
			glog.Infof("pollOperation: waited %v for %v", duration, opName)
		}
		pollOp, err := getOperation(opName)
		if err != nil {
			glog.Warningf("GCE poll operation %s failed: pollOp: [%v] err: [%v] getErrorFromOp: [%v]", opName, pollOp, err, getErrorFromOp(pollOp))
		}
		return opIsDone(pollOp), getErrorFromOp(pollOp)
	})
}

func opIsDone(op *compute.Operation) bool {
	return op != nil && op.Status == "DONE"
}

func getErrorFromOp(op *compute.Operation) error {
	if op != nil && op.Error != nil && len(op.Error.Errors) > 0 {
		err := &googleapi.Error{
			Code:    int(op.HttpErrorStatusCode),
			Message: op.Error.Errors[0].Message,
		}
		glog.Errorf("GCE operation failed: %v", err)
		return err
	}

	return nil
}

func (gce *GCECloud) waitForGlobalOp(op *compute.Operation) error {
	return gce.waitForOp(op, func(operationName string) (*compute.Operation, error) {
		return gce.service.GlobalOperations.Get(gce.projectID, operationName).Do()
	})
}

func (gce *GCECloud) waitForRegionOp(op *compute.Operation, region string) error {
	return gce.waitForOp(op, func(operationName string) (*compute.Operation, error) {
		return gce.service.RegionOperations.Get(gce.projectID, region, operationName).Do()
	})
}

func (gce *GCECloud) waitForZoneOp(op *compute.Operation, zone string) error {
	return gce.waitForOp(op, func(operationName string) (*compute.Operation, error) {
		return gce.service.ZoneOperations.Get(gce.projectID, zone, operationName).Do()
	})
}

// GetLoadBalancer is an implementation of LoadBalancer.GetLoadBalancer
func (gce *GCECloud) GetLoadBalancer(name, region string) (*api.LoadBalancerStatus, bool, error) {
	fwd, err := gce.service.ForwardingRules.Get(gce.projectID, region, name).Do()
	if err == nil {
		status := &api.LoadBalancerStatus{}
		status.Ingress = []api.LoadBalancerIngress{{IP: fwd.IPAddress}}

		return status, true, nil
	}
	if isHTTPErrorCode(err, http.StatusNotFound) {
		return nil, false, nil
	}
	return nil, false, err
}

func isHTTPErrorCode(err error, code int) bool {
	apiErr, ok := err.(*googleapi.Error)
	return ok && apiErr.Code == code
}

// EnsureLoadBalancer is an implementation of LoadBalancer.EnsureLoadBalancer.
// Our load balancers in GCE consist of four separate GCE resources - a static
// IP address, a firewall rule, a target pool, and a forwarding rule. This
// function has to manage all of them.
// Due to an interesting series of design decisions, this handles both creating
// new load balancers and updating existing load balancers, recognizing when
// each is needed.
func (gce *GCECloud) EnsureLoadBalancer(name, region string, requestedIP net.IP, ports []*api.ServicePort, hostNames []string, svc types.NamespacedName, affinityType api.ServiceAffinity, annotations map[string]string) (*api.LoadBalancerStatus, error) {
	portStr := []string{}
	for _, p := range ports {
		portStr = append(portStr, fmt.Sprintf("%s/%d", p.Protocol, p.Port))
	}
	serviceName := svc.String()
	glog.V(2).Infof("EnsureLoadBalancer(%v, %v, %v, %v, %v, %v, %v)", name, region, requestedIP, portStr, hostNames, serviceName, annotations)

	if len(hostNames) == 0 {
		return nil, fmt.Errorf("Cannot EnsureLoadBalancer() with no hosts")
	}

	hosts, err := gce.getInstancesByNames(hostNames)
	if err != nil {
		return nil, err
	}

	// Check if the forwarding rule exists, and if so, what its IP is.
	fwdRuleExists, fwdRuleNeedsUpdate, fwdRuleIP, err := gce.forwardingRuleNeedsUpdate(name, region, requestedIP, ports)
	if err != nil {
		return nil, err
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
	isUserOwnedIP := false // if this is set, we never release the IP
	isSafeToReleaseIP := false
	defer func() {
		if isUserOwnedIP {
			return
		}
		if isSafeToReleaseIP {
			if err := gce.deleteStaticIP(name, region); err != nil {
				glog.Errorf("failed to release static IP %s for load balancer (%v(%v), %v): %v", ipAddress, name, serviceName, region, err)
			}
			glog.V(2).Infof("EnsureLoadBalancer(%v(%v)): released static IP %s", name, serviceName, ipAddress)
		} else {
			glog.Warningf("orphaning static IP %s during update of load balancer (%v(%v), %v): %v", ipAddress, name, serviceName, region, err)
		}
	}()

	if requestedIP != nil {
		// If a specific IP address has been requested, we have to respect the
		// user's request and use that IP. If the forwarding rule was already using
		// a different IP, it will be harmlessly abandoned because it was only an
		// ephemeral IP (or it was a different static IP owned by the user, in which
		// case we shouldn't delete it anyway).
		if isStatic, err := gce.projectOwnsStaticIP(name, region, requestedIP.String()); err != nil {
			return nil, fmt.Errorf("failed to test if this GCE project owns the static IP %s: %v", requestedIP.String(), err)
		} else if isStatic {
			// The requested IP is a static IP, owned and managed by the user.
			isUserOwnedIP = true
			isSafeToReleaseIP = false
			ipAddress = requestedIP.String()
			glog.V(4).Infof("EnsureLoadBalancer(%v(%v)): using user-provided static IP %s", name, serviceName, ipAddress)
		} else if requestedIP.String() == fwdRuleIP {
			// The requested IP is not a static IP, but is currently assigned
			// to this forwarding rule, so we can keep it.
			isUserOwnedIP = false
			isSafeToReleaseIP = true
			ipAddress, _, err = gce.ensureStaticIP(name, serviceName, region, fwdRuleIP)
			if err != nil {
				return nil, fmt.Errorf("failed to ensure static IP %s: %v", fwdRuleIP, err)
			}
			glog.V(4).Infof("EnsureLoadBalancer(%v(%v)): using user-provided non-static IP %s", name, serviceName, ipAddress)
		} else {
			// The requested IP is not static and it is not assigned to the
			// current forwarding rule.  It might be attached to a different
			// rule or it might not be part of this project at all.  Either
			// way, we can't use it.
			return nil, fmt.Errorf("requested ip %s is neither static nor assigned to LB %s(%v): %v", requestedIP.String(), name, serviceName, err)
		}
	} else {
		// The user did not request a specific IP.
		isUserOwnedIP = false

		// This will either allocate a new static IP if the forwarding rule didn't
		// already have an IP, or it will promote the forwarding rule's current
		// IP from ephemeral to static, or it will just get the IP if it is
		// already static.
		existed := false
		ipAddress, existed, err = gce.ensureStaticIP(name, serviceName, region, fwdRuleIP)
		if err != nil {
			return nil, fmt.Errorf("failed to ensure static IP %s: %v", fwdRuleIP, err)
		}
		if existed {
			// If the IP was not specifically requested by the user, but it
			// already existed, it seems to be a failed update cycle.  We can
			// use this IP and try to run through the process again, but we
			// should not release the IP unless it is explicitly flagged as OK.
			isSafeToReleaseIP = false
			glog.V(4).Infof("EnsureLoadBalancer(%v(%v)): adopting static IP %s", name, serviceName, ipAddress)
		} else {
			// For total clarity.  The IP did not pre-exist and the user did
			// not ask for a particular one, so we can release the IP in case
			// of failure or success.
			isSafeToReleaseIP = true
			glog.V(4).Infof("EnsureLoadBalancer(%v(%v)): allocated static IP %s", name, serviceName, ipAddress)
		}
	}

	// Deal with the firewall next. The reason we do this here rather than last
	// is because the forwarding rule is used as the indicator that the load
	// balancer is fully created - it's what getLoadBalancer checks for.
	// Check if user specified the allow source range
	sourceRanges, err := service.GetLoadBalancerSourceRanges(annotations)
	if err != nil {
		return nil, err
	}

	firewallExists, firewallNeedsUpdate, err := gce.firewallNeedsUpdate(name, serviceName, region, ipAddress, ports, sourceRanges)
	if err != nil {
		return nil, err
	}

	if firewallNeedsUpdate {
		desc := makeFirewallDescription(serviceName, ipAddress)
		// Unlike forwarding rules and target pools, firewalls can be updated
		// without needing to be deleted and recreated.
		if firewallExists {
			if err := gce.updateFirewall(name, region, desc, sourceRanges, ports, hosts); err != nil {
				return nil, err
			}
			glog.V(4).Infof("EnsureLoadBalancer(%v(%v)): updated firewall", name, serviceName)
		} else {
			if err := gce.createFirewall(name, region, desc, sourceRanges, ports, hosts); err != nil {
				return nil, err
			}
			glog.V(4).Infof("EnsureLoadBalancer(%v(%v)): created firewall", name, serviceName)
		}
	}

	tpExists, tpNeedsUpdate, err := gce.targetPoolNeedsUpdate(name, region, affinityType)
	if err != nil {
		return nil, err
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
		if err := gce.deleteForwardingRule(name, region); err != nil {
			return nil, fmt.Errorf("failed to delete existing forwarding rule %s for load balancer update: %v", name, err)
		}
		glog.V(4).Infof("EnsureLoadBalancer(%v(%v)): deleted forwarding rule", name, serviceName)
	}
	if tpExists && tpNeedsUpdate {
		if err := gce.deleteTargetPool(name, region); err != nil {
			return nil, fmt.Errorf("failed to delete existing target pool %s for load balancer update: %v", name, err)
		}
		glog.V(4).Infof("EnsureLoadBalancer(%v(%v)): deleted target pool", name, serviceName)
	}

	// Once we've deleted the resources (if necessary), build them back up (or for
	// the first time if they're new).
	if tpNeedsUpdate {
		if err := gce.createTargetPool(name, serviceName, region, hosts, affinityType); err != nil {
			return nil, fmt.Errorf("failed to create target pool %s: %v", name, err)
		}
		glog.V(4).Infof("EnsureLoadBalancer(%v(%v)): created target pool", name, serviceName)
	}
	if tpNeedsUpdate || fwdRuleNeedsUpdate {
		if err := gce.createForwardingRule(name, serviceName, region, ipAddress, ports); err != nil {
			return nil, fmt.Errorf("failed to create forwarding rule %s: %v", name, err)
		}
		// End critical section.  It is safe to release the static IP (which
		// just demotes it to ephemeral) now that it is attached.  In the case
		// of a user-requested IP, the "is user-owned" flag will be set,
		// preventing it from actually being released.
		isSafeToReleaseIP = true
		glog.V(4).Infof("EnsureLoadBalancer(%v(%v)): created forwarding rule, IP %s", name, serviceName, ipAddress)
	}

	status := &api.LoadBalancerStatus{}
	status.Ingress = []api.LoadBalancerIngress{{IP: ipAddress}}
	return status, nil
}

// Passing nil for requested IP is perfectly fine - it just means that no specific
// IP is being requested.
// Returns whether the forwarding rule exists, whether it needs to be updated,
// what its IP address is (if it exists), and any error we encountered.
func (gce *GCECloud) forwardingRuleNeedsUpdate(name, region string, requestedIP net.IP, ports []*api.ServicePort) (exists bool, needsUpdate bool, ipAddress string, err error) {
	fwd, err := gce.service.ForwardingRules.Get(gce.projectID, region, name).Do()
	if err != nil {
		if isHTTPErrorCode(err, http.StatusNotFound) {
			return false, true, "", nil
		}
		return false, false, "", fmt.Errorf("error getting load balancer's forwarding rule: %v", err)
	}
	if requestedIP != nil && requestedIP.String() != fwd.IPAddress {
		return true, true, fwd.IPAddress, nil
	}
	portRange, err := loadBalancerPortRange(ports)
	if err != nil {
		return false, false, "", err
	}
	if portRange != fwd.PortRange {
		return true, true, fwd.IPAddress, nil
	}
	// The service controller verified all the protocols match on the ports, just check the first one
	if string(ports[0].Protocol) != fwd.IPProtocol {
		return true, true, fwd.IPAddress, nil
	}

	return true, false, fwd.IPAddress, nil
}

func loadBalancerPortRange(ports []*api.ServicePort) (string, error) {
	if len(ports) == 0 {
		return "", fmt.Errorf("no ports specified for GCE load balancer")
	}

	// The service controller verified all the protocols match on the ports, just check and use the first one
	if ports[0].Protocol != api.ProtocolTCP && ports[0].Protocol != api.ProtocolUDP {
		return "", fmt.Errorf("Invalid protocol %s, only TCP and UDP are supported", string(ports[0].Protocol))
	}

	minPort := 65536
	maxPort := 0
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

// Doesn't check whether the hosts have changed, since host updating is handled
// separately.
func (gce *GCECloud) targetPoolNeedsUpdate(name, region string, affinityType api.ServiceAffinity) (exists bool, needsUpdate bool, err error) {
	tp, err := gce.service.TargetPools.Get(gce.projectID, region, name).Do()
	if err != nil {
		if isHTTPErrorCode(err, http.StatusNotFound) {
			return false, true, nil
		}
		return false, false, fmt.Errorf("error getting load balancer's target pool: %v", err)
	}
	if translateAffinityType(affinityType) != tp.SessionAffinity {
		return true, true, nil
	}
	return true, false, nil
}

// translate from what K8s supports to what the cloud provider supports for session affinity.
func translateAffinityType(affinityType api.ServiceAffinity) string {
	switch affinityType {
	case api.ServiceAffinityClientIP:
		return gceAffinityTypeClientIP
	case api.ServiceAffinityNone:
		return gceAffinityTypeNone
	default:
		glog.Errorf("Unexpected affinity type: %v", affinityType)
		return gceAffinityTypeNone
	}
}

func (gce *GCECloud) firewallNeedsUpdate(name, serviceName, region, ipAddress string, ports []*api.ServicePort, sourceRanges netsets.IPNet) (exists bool, needsUpdate bool, err error) {
	fw, err := gce.service.Firewalls.Get(gce.projectID, makeFirewallName(name)).Do()
	if err != nil {
		if isHTTPErrorCode(err, http.StatusNotFound) {
			return false, true, nil
		}
		return false, false, fmt.Errorf("error getting load balancer's target pool: %v", err)
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
		allowedPorts[ix] = strconv.Itoa(ports[ix].Port)
	}
	if !slicesEqual(allowedPorts, fw.Allowed[0].Ports) {
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

func makeFirewallName(name string) string {
	return fmt.Sprintf("k8s-fw-%s", name)
}

func makeFirewallDescription(serviceName, ipAddress string) string {
	return fmt.Sprintf(`{"kubernetes.io/service-ip":"%s", "kubernetes.io/service-name":"%s"}`,
		ipAddress, serviceName)
}

func slicesEqual(x, y []string) bool {
	if len(x) != len(y) {
		return false
	}
	sort.Strings(x)
	sort.Strings(y)
	for i := range x {
		if x[i] != y[i] {
			return false
		}
	}
	return true
}

func (gce *GCECloud) createForwardingRule(name, serviceName, region, ipAddress string, ports []*api.ServicePort) error {
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

	op, err := gce.service.ForwardingRules.Insert(gce.projectID, region, req).Do()
	if err != nil && !isHTTPErrorCode(err, http.StatusConflict) {
		return err
	}
	if op != nil {
		err = gce.waitForRegionOp(op, region)
		if err != nil && !isHTTPErrorCode(err, http.StatusConflict) {
			return err
		}
	}
	return nil
}

func (gce *GCECloud) createTargetPool(name, serviceName, region string, hosts []*gceInstance, affinityType api.ServiceAffinity) error {
	var instances []string
	for _, host := range hosts {
		instances = append(instances, makeHostURL(gce.projectID, host.Zone, host.Name))
	}
	pool := &compute.TargetPool{
		Name:            name,
		Description:     fmt.Sprintf(`{"kubernetes.io/service-name":"%s"}`, serviceName),
		Instances:       instances,
		SessionAffinity: translateAffinityType(affinityType),
	}
	op, err := gce.service.TargetPools.Insert(gce.projectID, region, pool).Do()
	if err != nil && !isHTTPErrorCode(err, http.StatusConflict) {
		return err
	}
	if op != nil {
		err = gce.waitForRegionOp(op, region)
		if err != nil && !isHTTPErrorCode(err, http.StatusConflict) {
			return err
		}
	}
	return nil
}

func (gce *GCECloud) createFirewall(name, region, desc string, sourceRanges netsets.IPNet, ports []*api.ServicePort, hosts []*gceInstance) error {
	firewall, err := gce.firewallObject(name, region, desc, sourceRanges, ports, hosts)
	if err != nil {
		return err
	}
	op, err := gce.service.Firewalls.Insert(gce.projectID, firewall).Do()
	if err != nil && !isHTTPErrorCode(err, http.StatusConflict) {
		return err
	}
	if op != nil {
		err = gce.waitForGlobalOp(op)
		if err != nil && !isHTTPErrorCode(err, http.StatusConflict) {
			return err
		}
	}
	return nil
}

func (gce *GCECloud) updateFirewall(name, region, desc string, sourceRanges netsets.IPNet, ports []*api.ServicePort, hosts []*gceInstance) error {
	firewall, err := gce.firewallObject(name, region, desc, sourceRanges, ports, hosts)
	if err != nil {
		return err
	}
	op, err := gce.service.Firewalls.Update(gce.projectID, makeFirewallName(name), firewall).Do()
	if err != nil && !isHTTPErrorCode(err, http.StatusConflict) {
		return err
	}
	if op != nil {
		err = gce.waitForGlobalOp(op)
		if err != nil {
			return err
		}
	}
	return nil
}

func (gce *GCECloud) firewallObject(name, region, desc string, sourceRanges netsets.IPNet, ports []*api.ServicePort, hosts []*gceInstance) (*compute.Firewall, error) {
	allowedPorts := make([]string, len(ports))
	for ix := range ports {
		allowedPorts[ix] = strconv.Itoa(ports[ix].Port)
	}
	hostTags, err := gce.computeHostTags(hosts)
	if err != nil {
		return nil, err
	}
	firewall := &compute.Firewall{
		Name:         makeFirewallName(name),
		Description:  desc,
		Network:      gce.networkURL,
		SourceRanges: sourceRanges.StringSlice(),
		TargetTags:   hostTags,
		Allowed: []*compute.FirewallAllowed{
			{
				IPProtocol: strings.ToLower(string(ports[0].Protocol)),
				Ports:      allowedPorts,
			},
		},
	}
	return firewall, nil
}

// We grab all tags from all instances being added to the pool.
// * The longest tag that is a prefix of the instance name is used
// * If any instance has a prefix tag, all instances must
// * If no instances have a prefix tag, no tags are used
func (gce *GCECloud) computeHostTags(hosts []*gceInstance) ([]string, error) {
	// TODO: We could store the tags in gceInstance, so we could have already fetched it
	hostNamesByZone := make(map[string][]string)
	for _, host := range hosts {
		hostNamesByZone[host.Zone] = append(hostNamesByZone[host.Zone], host.Name)
	}

	tags := sets.NewString()

	for zone, hostNames := range hostNamesByZone {
		pageToken := ""
		page := 0
		for ; page == 0 || (pageToken != "" && page < maxPages); page++ {
			listCall := gce.service.Instances.List(gce.projectID, zone)

			// Add the filter for hosts
			listCall = listCall.Filter("name eq (" + strings.Join(hostNames, "|") + ")")

			// Add the fields we want
			listCall = listCall.Fields("items(name,tags)")

			if pageToken != "" {
				listCall = listCall.PageToken(pageToken)
			}

			res, err := listCall.Do()
			if err != nil {
				return nil, err
			}
			pageToken = res.NextPageToken
			for _, instance := range res.Items {
				longest_tag := ""
				for _, tag := range instance.Tags.Items {
					if strings.HasPrefix(instance.Name, tag) && len(tag) > len(longest_tag) {
						longest_tag = tag
					}
				}
				if len(longest_tag) > 0 {
					tags.Insert(longest_tag)
				} else if len(tags) > 0 {
					return nil, fmt.Errorf("Some, but not all, instances have prefix tags (%s is missing)", instance.Name)
				}
			}
		}
		if page >= maxPages {
			glog.Errorf("computeHostTags exceeded maxPages=%d for Instances.List: truncating.", maxPages)
		}
	}

	if len(tags) == 0 {
		glog.V(2).Info("No instances had tags, creating rule without target tags")
	}

	return tags.List(), nil
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
	op, err := gce.service.Addresses.Insert(gce.projectID, region, addressObj).Do()
	if err != nil {
		if !isHTTPErrorCode(err, http.StatusConflict) {
			return "", false, fmt.Errorf("error creating gce static IP address: %v", err)
		}
		// StatusConflict == the IP exists already.
		existed = true
	}
	if op != nil {
		err := gce.waitForRegionOp(op, region)
		if err != nil {
			if !isHTTPErrorCode(err, http.StatusConflict) {
				return "", false, fmt.Errorf("error waiting for gce static IP address to be created: %v", err)
			}
			// StatusConflict == the IP exists already.
			existed = true
		}
	}

	// We have to get the address to know which IP was allocated for us.
	address, err := gce.service.Addresses.Get(gce.projectID, region, name).Do()
	if err != nil {
		return "", false, fmt.Errorf("error re-getting gce static IP address: %v", err)
	}
	return address.Address, existed, nil
}

// UpdateLoadBalancer is an implementation of LoadBalancer.UpdateLoadBalancer.
func (gce *GCECloud) UpdateLoadBalancer(name, region string, hostNames []string) error {
	hosts, err := gce.getInstancesByNames(hostNames)
	if err != nil {
		return err
	}

	pool, err := gce.service.TargetPools.Get(gce.projectID, region, name).Do()
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
		add := &compute.TargetPoolsAddInstanceRequest{Instances: toAdd}
		op, err := gce.service.TargetPools.AddInstance(gce.projectID, region, name, add).Do()
		if err != nil {
			return err
		}
		if err := gce.waitForRegionOp(op, region); err != nil {
			return err
		}
	}

	if len(toRemove) > 0 {
		rm := &compute.TargetPoolsRemoveInstanceRequest{Instances: toRemove}
		op, err := gce.service.TargetPools.RemoveInstance(gce.projectID, region, name, rm).Do()
		if err != nil {
			return err
		}
		if err := gce.waitForRegionOp(op, region); err != nil {
			return err
		}
	}

	// Try to verify that the correct number of nodes are now in the target pool.
	// We've been bitten by a bug here before (#11327) where all nodes were
	// accidentally removed and want to make similar problems easier to notice.
	updatedPool, err := gce.service.TargetPools.Get(gce.projectID, region, name).Do()
	if err != nil {
		return err
	}
	if len(updatedPool.Instances) != len(hosts) {
		glog.Errorf("Unexpected number of instances (%d) in target pool %s after updating (expected %d). Instances in updated pool: %s",
			len(updatedPool.Instances), name, len(hosts), strings.Join(updatedPool.Instances, ","))
		return fmt.Errorf("Unexpected number of instances (%d) in target pool %s after update (expected %d)", len(updatedPool.Instances), name, len(hosts))
	}
	return nil
}

// EnsureLoadBalancerDeleted is an implementation of LoadBalancer.EnsureLoadBalancerDeleted.
func (gce *GCECloud) EnsureLoadBalancerDeleted(name, region string) error {
	glog.V(2).Infof("EnsureLoadBalancerDeleted(%v, %v", name, region)
	err := utilerrors.AggregateGoroutines(
		func() error { return gce.deleteFirewall(name, region) },
		// Even though we don't hold on to static IPs for load balancers, it's
		// possible that EnsureLoadBalancer left one around in a failed
		// creation/update attempt, so make sure we clean it up here just in case.
		func() error { return gce.deleteStaticIP(name, region) },
		func() error {
			// The forwarding rule must be deleted before either the target pool can,
			// unfortunately, so we have to do these two serially.
			if err := gce.deleteForwardingRule(name, region); err != nil {
				return err
			}
			if err := gce.deleteTargetPool(name, region); err != nil {
				return err
			}
			return nil
		},
	)
	if err != nil {
		return utilerrors.Flatten(err)
	}
	return nil
}

func (gce *GCECloud) deleteForwardingRule(name, region string) error {
	op, err := gce.service.ForwardingRules.Delete(gce.projectID, region, name).Do()
	if err != nil && isHTTPErrorCode(err, http.StatusNotFound) {
		glog.Infof("Forwarding rule %s already deleted. Continuing to delete other resources.", name)
	} else if err != nil {
		glog.Warningf("Failed to delete forwarding rule %s: got error %s.", name, err.Error())
		return err
	} else {
		if err := gce.waitForRegionOp(op, region); err != nil {
			glog.Warningf("Failed waiting for forwarding rule %s to be deleted: got error %s.", name, err.Error())
			return err
		}
	}
	return nil
}

func (gce *GCECloud) deleteTargetPool(name, region string) error {
	op, err := gce.service.TargetPools.Delete(gce.projectID, region, name).Do()
	if err != nil && isHTTPErrorCode(err, http.StatusNotFound) {
		glog.Infof("Target pool %s already deleted. Continuing to delete other resources.", name)
	} else if err != nil {
		glog.Warningf("Failed to delete target pool %s, got error %s.", name, err.Error())
		return err
	} else {
		if err := gce.waitForRegionOp(op, region); err != nil {
			glog.Warningf("Failed waiting for target pool %s to be deleted: got error %s.", name, err.Error())
			return err
		}
	}
	return nil
}

func (gce *GCECloud) deleteFirewall(name, region string) error {
	fwName := makeFirewallName(name)
	op, err := gce.service.Firewalls.Delete(gce.projectID, fwName).Do()
	if err != nil && isHTTPErrorCode(err, http.StatusNotFound) {
		glog.Infof("Firewall %s already deleted. Continuing to delete other resources.", name)
	} else if err != nil {
		glog.Warningf("Failed to delete firewall %s, got error %v", fwName, err)
		return err
	} else {
		if err := gce.waitForGlobalOp(op); err != nil {
			glog.Warningf("Failed waiting for Firewall %s to be deleted.  Got error: %v", fwName, err)
			return err
		}
	}
	return nil
}

func (gce *GCECloud) deleteStaticIP(name, region string) error {
	op, err := gce.service.Addresses.Delete(gce.projectID, region, name).Do()
	if err != nil && isHTTPErrorCode(err, http.StatusNotFound) {
		glog.Infof("Static IP address %s is not reserved", name)
	} else if err != nil {
		glog.Warningf("Failed to delete static IP address %s, got error %v", name, err)
		return err
	} else {
		if err := gce.waitForRegionOp(op, region); err != nil {
			glog.Warningf("Failed waiting for address %s to be deleted, got error: %v", name, err)
			return err
		}
	}
	return nil
}

// Firewall management: These methods are just passthrough to the existing
// internal firewall creation methods used to manage TCPLoadBalancer.

// GetFirewall returns the Firewall by name.
func (gce *GCECloud) GetFirewall(name string) (*compute.Firewall, error) {
	return gce.service.Firewalls.Get(gce.projectID, name).Do()
}

// CreateFirewall creates the given firewall rule.
func (gce *GCECloud) CreateFirewall(name, desc string, sourceRanges netsets.IPNet, ports []int64, hostNames []string) error {
	region, err := GetGCERegion(gce.localZone)
	if err != nil {
		return err
	}
	// TODO: This completely breaks modularity in the cloudprovider but the methods
	// shared with the TCPLoadBalancer take api.ServicePorts.
	svcPorts := []*api.ServicePort{}
	for _, p := range ports {
		svcPorts = append(svcPorts, &api.ServicePort{Port: int(p)})
	}
	hosts, err := gce.getInstancesByNames(hostNames)
	if err != nil {
		return err
	}
	return gce.createFirewall(name, region, desc, sourceRanges, svcPorts, hosts)
}

// DeleteFirewall deletes the given firewall rule.
func (gce *GCECloud) DeleteFirewall(name string) error {
	region, err := GetGCERegion(gce.localZone)
	if err != nil {
		return err
	}
	return gce.deleteFirewall(name, region)
}

// UpdateFirewall applies the given firewall rule as an update to an existing
// firewall rule with the same name.
func (gce *GCECloud) UpdateFirewall(name, desc string, sourceRanges netsets.IPNet, ports []int64, hostNames []string) error {
	region, err := GetGCERegion(gce.localZone)
	if err != nil {
		return err
	}
	// TODO: This completely breaks modularity in the cloudprovider but the methods
	// shared with the TCPLoadBalancer take api.ServicePorts.
	svcPorts := []*api.ServicePort{}
	for _, p := range ports {
		svcPorts = append(svcPorts, &api.ServicePort{Port: int(p)})
	}
	hosts, err := gce.getInstancesByNames(hostNames)
	if err != nil {
		return err
	}
	return gce.updateFirewall(name, region, desc, sourceRanges, svcPorts, hosts)
}

// Global static IP management

// ReserveGlobalStaticIP creates a global static IP.
// Caller is allocated a random IP if they do not specify an ipAddress. If an
// ipAddress is specified, it must belong to the current project, eg: an
// ephemeral IP associated with a global forwarding rule.
func (gce *GCECloud) ReserveGlobalStaticIP(name, ipAddress string) (address *compute.Address, err error) {
	op, err := gce.service.GlobalAddresses.Insert(gce.projectID, &compute.Address{Name: name, Address: ipAddress}).Do()
	if err != nil {
		return nil, err
	}
	if err := gce.waitForGlobalOp(op); err != nil {
		return nil, err
	}
	// We have to get the address to know which IP was allocated for us.
	return gce.service.GlobalAddresses.Get(gce.projectID, name).Do()
}

// DeleteGlobalStaticIP deletes a global static IP by name.
func (gce *GCECloud) DeleteGlobalStaticIP(name string) error {
	op, err := gce.service.GlobalAddresses.Delete(gce.projectID, name).Do()
	if err != nil {
		return err
	}
	return gce.waitForGlobalOp(op)
}

// GetGlobalStaticIP returns the global static IP by name.
func (gce *GCECloud) GetGlobalStaticIP(name string) (address *compute.Address, err error) {
	return gce.service.GlobalAddresses.Get(gce.projectID, name).Do()
}

// UrlMap management

// GetUrlMap returns the UrlMap by name.
func (gce *GCECloud) GetUrlMap(name string) (*compute.UrlMap, error) {
	return gce.service.UrlMaps.Get(gce.projectID, name).Do()
}

// CreateUrlMap creates an url map, using the given backend service as the default service.
func (gce *GCECloud) CreateUrlMap(backend *compute.BackendService, name string) (*compute.UrlMap, error) {
	urlMap := &compute.UrlMap{
		Name:           name,
		DefaultService: backend.SelfLink,
	}
	op, err := gce.service.UrlMaps.Insert(gce.projectID, urlMap).Do()
	if err != nil {
		return nil, err
	}
	if err = gce.waitForGlobalOp(op); err != nil {
		return nil, err
	}
	return gce.GetUrlMap(name)
}

// UpdateUrlMap applies the given UrlMap as an update, and returns the new UrlMap.
func (gce *GCECloud) UpdateUrlMap(urlMap *compute.UrlMap) (*compute.UrlMap, error) {
	op, err := gce.service.UrlMaps.Update(gce.projectID, urlMap.Name, urlMap).Do()
	if err != nil {
		return nil, err
	}
	if err = gce.waitForGlobalOp(op); err != nil {
		return nil, err
	}
	return gce.service.UrlMaps.Get(gce.projectID, urlMap.Name).Do()
}

// DeleteUrlMap deletes a url map by name.
func (gce *GCECloud) DeleteUrlMap(name string) error {
	op, err := gce.service.UrlMaps.Delete(gce.projectID, name).Do()
	if err != nil {
		if isHTTPErrorCode(err, http.StatusNotFound) {
			return nil
		}
		return err
	}
	return gce.waitForGlobalOp(op)
}

// ListUrlMaps lists all UrlMaps in the project.
func (gce *GCECloud) ListUrlMaps() (*compute.UrlMapList, error) {
	// TODO: use PageToken to list all not just the first 500
	return gce.service.UrlMaps.List(gce.projectID).Do()
}

// TargetHttpProxy management

// GetTargetHttpProxy returns the UrlMap by name.
func (gce *GCECloud) GetTargetHttpProxy(name string) (*compute.TargetHttpProxy, error) {
	return gce.service.TargetHttpProxies.Get(gce.projectID, name).Do()
}

// CreateTargetHttpProxy creates and returns a TargetHttpProxy with the given UrlMap.
func (gce *GCECloud) CreateTargetHttpProxy(urlMap *compute.UrlMap, name string) (*compute.TargetHttpProxy, error) {
	proxy := &compute.TargetHttpProxy{
		Name:   name,
		UrlMap: urlMap.SelfLink,
	}
	op, err := gce.service.TargetHttpProxies.Insert(gce.projectID, proxy).Do()
	if err != nil {
		return nil, err
	}
	if err = gce.waitForGlobalOp(op); err != nil {
		return nil, err
	}
	return gce.GetTargetHttpProxy(name)
}

// SetUrlMapForTargetHttpProxy sets the given UrlMap for the given TargetHttpProxy.
func (gce *GCECloud) SetUrlMapForTargetHttpProxy(proxy *compute.TargetHttpProxy, urlMap *compute.UrlMap) error {
	op, err := gce.service.TargetHttpProxies.SetUrlMap(gce.projectID, proxy.Name, &compute.UrlMapReference{UrlMap: urlMap.SelfLink}).Do()
	if err != nil {
		return err
	}
	return gce.waitForGlobalOp(op)
}

// DeleteTargetHttpProxy deletes the TargetHttpProxy by name.
func (gce *GCECloud) DeleteTargetHttpProxy(name string) error {
	op, err := gce.service.TargetHttpProxies.Delete(gce.projectID, name).Do()
	if err != nil {
		if isHTTPErrorCode(err, http.StatusNotFound) {
			return nil
		}
		return err
	}
	return gce.waitForGlobalOp(op)
}

// ListTargetHttpProxies lists all TargetHttpProxies in the project.
func (gce *GCECloud) ListTargetHttpProxies() (*compute.TargetHttpProxyList, error) {
	// TODO: use PageToken to list all not just the first 500
	return gce.service.TargetHttpProxies.List(gce.projectID).Do()
}

// TargetHttpsProxy management

// GetTargetHttpsProxy returns the UrlMap by name.
func (gce *GCECloud) GetTargetHttpsProxy(name string) (*compute.TargetHttpsProxy, error) {
	return gce.service.TargetHttpsProxies.Get(gce.projectID, name).Do()
}

// CreateTargetHttpsProxy creates and returns a TargetHttpsProxy with the given UrlMap and SslCertificate.
func (gce *GCECloud) CreateTargetHttpsProxy(urlMap *compute.UrlMap, sslCert *compute.SslCertificate, name string) (*compute.TargetHttpsProxy, error) {
	proxy := &compute.TargetHttpsProxy{
		Name:            name,
		UrlMap:          urlMap.SelfLink,
		SslCertificates: []string{sslCert.SelfLink},
	}
	op, err := gce.service.TargetHttpsProxies.Insert(gce.projectID, proxy).Do()
	if err != nil {
		return nil, err
	}
	if err = gce.waitForGlobalOp(op); err != nil {
		return nil, err
	}
	return gce.GetTargetHttpsProxy(name)
}

// SetUrlMapForTargetHttpsProxy sets the given UrlMap for the given TargetHttpsProxy.
func (gce *GCECloud) SetUrlMapForTargetHttpsProxy(proxy *compute.TargetHttpsProxy, urlMap *compute.UrlMap) error {
	op, err := gce.service.TargetHttpsProxies.SetUrlMap(gce.projectID, proxy.Name, &compute.UrlMapReference{UrlMap: urlMap.SelfLink}).Do()
	if err != nil {
		return err
	}
	return gce.waitForGlobalOp(op)
}

// SetSslCertificateForTargetHttpsProxy sets the given SslCertificate for the given TargetHttpsProxy.
func (gce *GCECloud) SetSslCertificateForTargetHttpsProxy(proxy *compute.TargetHttpsProxy, sslCert *compute.SslCertificate) error {
	op, err := gce.service.TargetHttpsProxies.SetSslCertificates(gce.projectID, proxy.Name, &compute.TargetHttpsProxiesSetSslCertificatesRequest{SslCertificates: []string{sslCert.SelfLink}}).Do()
	if err != nil {
		return err
	}
	return gce.waitForGlobalOp(op)
}

// DeleteTargetHttpsProxy deletes the TargetHttpsProxy by name.
func (gce *GCECloud) DeleteTargetHttpsProxy(name string) error {
	op, err := gce.service.TargetHttpsProxies.Delete(gce.projectID, name).Do()
	if err != nil {
		if isHTTPErrorCode(err, http.StatusNotFound) {
			return nil
		}
		return err
	}
	return gce.waitForGlobalOp(op)
}

// ListTargetHttpsProxies lists all TargetHttpsProxies in the project.
func (gce *GCECloud) ListTargetHttpsProxies() (*compute.TargetHttpsProxyList, error) {
	// TODO: use PageToken to list all not just the first 500
	return gce.service.TargetHttpsProxies.List(gce.projectID).Do()
}

// SSL Certificate management

// GetSslCertificate returns the SslCertificate by name.
func (gce *GCECloud) GetSslCertificate(name string) (*compute.SslCertificate, error) {
	return gce.service.SslCertificates.Get(gce.projectID, name).Do()
}

// CreateSslCertificate creates and returns a SslCertificate.
func (gce *GCECloud) CreateSslCertificate(sslCerts *compute.SslCertificate) (*compute.SslCertificate, error) {
	op, err := gce.service.SslCertificates.Insert(gce.projectID, sslCerts).Do()
	if err != nil {
		return nil, err
	}
	if err = gce.waitForGlobalOp(op); err != nil {
		return nil, err
	}
	return gce.GetSslCertificate(sslCerts.Name)
}

// DeleteSslCertificate deletes the SslCertificate by name.
func (gce *GCECloud) DeleteSslCertificate(name string) error {
	op, err := gce.service.SslCertificates.Delete(gce.projectID, name).Do()
	if err != nil {
		if isHTTPErrorCode(err, http.StatusNotFound) {
			return nil
		}
		return err
	}
	return gce.waitForGlobalOp(op)
}

// ListSslCertificates lists all SslCertificates in the project.
func (gce *GCECloud) ListSslCertificates() (*compute.SslCertificateList, error) {
	// TODO: use PageToken to list all not just the first 500
	return gce.service.SslCertificates.List(gce.projectID).Do()
}

// GlobalForwardingRule management

// CreateGlobalForwardingRule creates and returns a GlobalForwardingRule that points to the given TargetHttp(s)Proxy.
// targetProxyLink is the SelfLink of a TargetHttp(s)Proxy.
func (gce *GCECloud) CreateGlobalForwardingRule(targetProxyLink, ip, name, portRange string) (*compute.ForwardingRule, error) {
	rule := &compute.ForwardingRule{
		Name:       name,
		IPAddress:  ip,
		Target:     targetProxyLink,
		PortRange:  portRange,
		IPProtocol: "TCP",
	}
	op, err := gce.service.GlobalForwardingRules.Insert(gce.projectID, rule).Do()
	if err != nil {
		return nil, err
	}
	if err = gce.waitForGlobalOp(op); err != nil {
		return nil, err
	}
	return gce.GetGlobalForwardingRule(name)
}

// SetProxyForGlobalForwardingRule links the given TargetHttp(s)Proxy with the given GlobalForwardingRule.
// targetProxyLink is the SelfLink of a TargetHttp(s)Proxy.
func (gce *GCECloud) SetProxyForGlobalForwardingRule(fw *compute.ForwardingRule, targetProxyLink string) error {
	op, err := gce.service.GlobalForwardingRules.SetTarget(gce.projectID, fw.Name, &compute.TargetReference{Target: targetProxyLink}).Do()
	if err != nil {
		return err
	}
	return gce.waitForGlobalOp(op)
}

// DeleteGlobalForwardingRule deletes the GlobalForwardingRule by name.
func (gce *GCECloud) DeleteGlobalForwardingRule(name string) error {
	op, err := gce.service.GlobalForwardingRules.Delete(gce.projectID, name).Do()
	if err != nil {
		if isHTTPErrorCode(err, http.StatusNotFound) {
			return nil
		}
		return err
	}
	return gce.waitForGlobalOp(op)
}

// GetGlobalForwardingRule returns the GlobalForwardingRule by name.
func (gce *GCECloud) GetGlobalForwardingRule(name string) (*compute.ForwardingRule, error) {
	return gce.service.GlobalForwardingRules.Get(gce.projectID, name).Do()
}

// ListGlobalForwardingRules lists all GlobalForwardingRules in the project.
func (gce *GCECloud) ListGlobalForwardingRules() (*compute.ForwardingRuleList, error) {
	// TODO: use PageToken to list all not just the first 500
	return gce.service.GlobalForwardingRules.List(gce.projectID).Do()
}

// BackendService Management

// GetBackendService retrieves a backend by name.
func (gce *GCECloud) GetBackendService(name string) (*compute.BackendService, error) {
	return gce.service.BackendServices.Get(gce.projectID, name).Do()
}

// UpdateBackendService applies the given BackendService as an update to an existing service.
func (gce *GCECloud) UpdateBackendService(bg *compute.BackendService) error {
	op, err := gce.service.BackendServices.Update(gce.projectID, bg.Name, bg).Do()
	if err != nil {
		return err
	}
	return gce.waitForGlobalOp(op)
}

// DeleteBackendService deletes the given BackendService by name.
func (gce *GCECloud) DeleteBackendService(name string) error {
	op, err := gce.service.BackendServices.Delete(gce.projectID, name).Do()
	if err != nil {
		if isHTTPErrorCode(err, http.StatusNotFound) {
			return nil
		}
		return err
	}
	return gce.waitForGlobalOp(op)
}

// CreateBackendService creates the given BackendService.
func (gce *GCECloud) CreateBackendService(bg *compute.BackendService) error {
	op, err := gce.service.BackendServices.Insert(gce.projectID, bg).Do()
	if err != nil {
		return err
	}
	return gce.waitForGlobalOp(op)
}

// ListBackendServices lists all backend services in the project.
func (gce *GCECloud) ListBackendServices() (*compute.BackendServiceList, error) {
	// TODO: use PageToken to list all not just the first 500
	return gce.service.BackendServices.List(gce.projectID).Do()
}

// GetHealth returns the health of the BackendService identified by the given
// name, in the given instanceGroup. The instanceGroupLink is the fully
// qualified self link of an instance group.
func (gce *GCECloud) GetHealth(name string, instanceGroupLink string) (*compute.BackendServiceGroupHealth, error) {
	groupRef := &compute.ResourceGroupReference{Group: instanceGroupLink}
	return gce.service.BackendServices.GetHealth(gce.projectID, name, groupRef).Do()
}

// Health Checks

// GetHttpHealthCheck returns the given HttpHealthCheck by name.
func (gce *GCECloud) GetHttpHealthCheck(name string) (*compute.HttpHealthCheck, error) {
	return gce.service.HttpHealthChecks.Get(gce.projectID, name).Do()
}

// UpdateHttpHealthCheck applies the given HttpHealthCheck as an update.
func (gce *GCECloud) UpdateHttpHealthCheck(hc *compute.HttpHealthCheck) error {
	op, err := gce.service.HttpHealthChecks.Update(gce.projectID, hc.Name, hc).Do()
	if err != nil {
		return err
	}
	return gce.waitForGlobalOp(op)
}

// DeleteHttpHealthCheck deletes the given HttpHealthCheck by name.
func (gce *GCECloud) DeleteHttpHealthCheck(name string) error {
	op, err := gce.service.HttpHealthChecks.Delete(gce.projectID, name).Do()
	if err != nil {
		if isHTTPErrorCode(err, http.StatusNotFound) {
			return nil
		}
		return err
	}
	return gce.waitForGlobalOp(op)
}

// CreateHttpHealthCheck creates the given HttpHealthCheck.
func (gce *GCECloud) CreateHttpHealthCheck(hc *compute.HttpHealthCheck) error {
	op, err := gce.service.HttpHealthChecks.Insert(gce.projectID, hc).Do()
	if err != nil {
		return err
	}
	return gce.waitForGlobalOp(op)
}

// ListHttpHealthCheck lists all HttpHealthChecks in the project.
func (gce *GCECloud) ListHttpHealthChecks() (*compute.HttpHealthCheckList, error) {
	// TODO: use PageToken to list all not just the first 500
	return gce.service.HttpHealthChecks.List(gce.projectID).Do()
}

// InstanceGroup Management

// CreateInstanceGroup creates an instance group with the given instances. It is the callers responsibility to add named ports.
func (gce *GCECloud) CreateInstanceGroup(name string, zone string) (*compute.InstanceGroup, error) {
	op, err := gce.service.InstanceGroups.Insert(
		gce.projectID, zone, &compute.InstanceGroup{Name: name}).Do()
	if err != nil {
		return nil, err
	}
	if err = gce.waitForZoneOp(op, zone); err != nil {
		return nil, err
	}
	return gce.GetInstanceGroup(name, zone)
}

// DeleteInstanceGroup deletes an instance group.
func (gce *GCECloud) DeleteInstanceGroup(name string, zone string) error {
	op, err := gce.service.InstanceGroups.Delete(
		gce.projectID, zone, name).Do()
	if err != nil {
		return err
	}
	return gce.waitForZoneOp(op, zone)
}

// ListInstanceGroups lists all InstanceGroups in the project and zone.
func (gce *GCECloud) ListInstanceGroups(zone string) (*compute.InstanceGroupList, error) {
	// TODO: use PageToken to list all not just the first 500
	return gce.service.InstanceGroups.List(gce.projectID, zone).Do()
}

// ListInstancesInInstanceGroup lists all the instances in a given instance group and state.
func (gce *GCECloud) ListInstancesInInstanceGroup(name string, zone string, state string) (*compute.InstanceGroupsListInstances, error) {
	// TODO: use PageToken to list all not just the first 500
	return gce.service.InstanceGroups.ListInstances(
		gce.projectID, zone, name,
		&compute.InstanceGroupsListInstancesRequest{InstanceState: state}).Do()
}

// AddInstancesToInstanceGroup adds the given instances to the given instance group.
func (gce *GCECloud) AddInstancesToInstanceGroup(name string, zone string, instanceNames []string) error {
	if len(instanceNames) == 0 {
		return nil
	}
	// Adding the same instance twice will result in a 4xx error
	instances := []*compute.InstanceReference{}
	for _, ins := range instanceNames {
		instances = append(instances, &compute.InstanceReference{Instance: makeHostURL(gce.projectID, zone, ins)})
	}
	op, err := gce.service.InstanceGroups.AddInstances(
		gce.projectID, zone, name,
		&compute.InstanceGroupsAddInstancesRequest{
			Instances: instances,
		}).Do()

	if err != nil {
		return err
	}
	return gce.waitForZoneOp(op, zone)
}

// RemoveInstancesFromInstanceGroup removes the given instances from the instance group.
func (gce *GCECloud) RemoveInstancesFromInstanceGroup(name string, zone string, instanceNames []string) error {
	if len(instanceNames) == 0 {
		return nil
	}
	instances := []*compute.InstanceReference{}
	for _, ins := range instanceNames {
		instanceLink := makeHostURL(gce.projectID, zone, ins)
		instances = append(instances, &compute.InstanceReference{Instance: instanceLink})
	}
	op, err := gce.service.InstanceGroups.RemoveInstances(
		gce.projectID, zone, name,
		&compute.InstanceGroupsRemoveInstancesRequest{
			Instances: instances,
		}).Do()

	if err != nil {
		if isHTTPErrorCode(err, http.StatusNotFound) {
			return nil
		}
		return err
	}
	return gce.waitForZoneOp(op, zone)
}

// AddPortToInstanceGroup adds a port to the given instance group.
func (gce *GCECloud) AddPortToInstanceGroup(ig *compute.InstanceGroup, port int64) (*compute.NamedPort, error) {
	for _, np := range ig.NamedPorts {
		if np.Port == port {
			glog.V(3).Infof("Instance group %v already has named port %+v", ig.Name, np)
			return np, nil
		}
	}
	glog.Infof("Adding port %v to instance group %v with %d ports", port, ig.Name, len(ig.NamedPorts))
	namedPort := compute.NamedPort{Name: fmt.Sprintf("port%v", port), Port: port}
	ig.NamedPorts = append(ig.NamedPorts, &namedPort)

	// setNamedPorts is a zonal endpoint, meaning we invoke it by re-creating a URL like:
	// {project}/zones/{zone}/instanceGroups/{instanceGroup}/setNamedPorts, so the "zone"
	// parameter given to SetNamedPorts must not be the entire zone URL.
	zoneURLParts := strings.Split(ig.Zone, "/")
	zone := zoneURLParts[len(zoneURLParts)-1]

	op, err := gce.service.InstanceGroups.SetNamedPorts(
		gce.projectID, zone, ig.Name,
		&compute.InstanceGroupsSetNamedPortsRequest{
			NamedPorts: ig.NamedPorts}).Do()
	if err != nil {
		return nil, err
	}
	if err = gce.waitForZoneOp(op, zone); err != nil {
		return nil, err
	}
	return &namedPort, nil
}

// GetInstanceGroup returns an instance group by name.
func (gce *GCECloud) GetInstanceGroup(name string, zone string) (*compute.InstanceGroup, error) {
	return gce.service.InstanceGroups.Get(gce.projectID, zone, name).Do()
}

// Take a GCE instance 'hostname' and break it down to something that can be fed
// to the GCE API client library.  Basically this means reducing 'kubernetes-
// minion-2.c.my-proj.internal' to 'kubernetes-minion-2' if necessary.
func canonicalizeInstanceName(name string) string {
	ix := strings.Index(name, ".")
	if ix != -1 {
		name = name[:ix]
	}
	return name
}

// Implementation of Instances.CurrentNodeName
func (gce *GCECloud) CurrentNodeName(hostname string) (string, error) {
	return hostname, nil
}

func (gce *GCECloud) AddSSHKeyToAllInstances(user string, keyData []byte) error {
	return wait.Poll(2*time.Second, 30*time.Second, func() (bool, error) {
		project, err := gce.service.Projects.Get(gce.projectID).Do()
		if err != nil {
			glog.Errorf("Could not get project: %v", err)
			return false, nil
		}
		keyString := fmt.Sprintf("%s:%s %s@%s", user, strings.TrimSpace(string(keyData)), user, user)
		found := false
		for _, item := range project.CommonInstanceMetadata.Items {
			if item.Key == "sshKeys" {
				if strings.Contains(*item.Value, keyString) {
					// We've already added the key
					glog.Info("SSHKey already in project metadata")
					return true, nil
				}
				value := *item.Value + "\n" + keyString
				item.Value = &value
				found = true
				break
			}
		}
		if !found {
			// This is super unlikely, so log.
			glog.Infof("Failed to find sshKeys metadata, creating a new item")
			project.CommonInstanceMetadata.Items = append(project.CommonInstanceMetadata.Items,
				&compute.MetadataItems{
					Key:   "sshKeys",
					Value: &keyString,
				})
		}
		op, err := gce.service.Projects.SetCommonInstanceMetadata(gce.projectID, project.CommonInstanceMetadata).Do()
		if err != nil {
			glog.Errorf("Could not Set Metadata: %v", err)
			return false, nil
		}
		if err := gce.waitForGlobalOp(op); err != nil {
			glog.Errorf("Could not Set Metadata: %v", err)
			return false, nil
		}
		glog.Infof("Successfully added sshKey to project metadata")
		return true, nil
	})
}

// NodeAddresses is an implementation of Instances.NodeAddresses.
func (gce *GCECloud) NodeAddresses(_ string) ([]api.NodeAddress, error) {
	internalIP, err := metadata.Get("instance/network-interfaces/0/ip")
	if err != nil {
		return nil, fmt.Errorf("couldn't get internal IP: %v", err)
	}
	externalIP, err := metadata.Get("instance/network-interfaces/0/access-configs/0/external-ip")
	if err != nil {
		return nil, fmt.Errorf("couldn't get external IP: %v", err)
	}
	return []api.NodeAddress{
		{Type: api.NodeInternalIP, Address: internalIP},
		{Type: api.NodeExternalIP, Address: externalIP},
	}, nil
}

// isCurrentInstance uses metadata server to check if specified instanceID matches current machine's instanceID
func (gce *GCECloud) isCurrentInstance(instanceID string) bool {
	currentInstanceID, err := getInstanceIDViaMetadata()
	if err != nil {
		// Log and swallow error
		glog.Errorf("Failed to fetch instanceID via Metadata: %v", err)
		return false
	}

	return currentInstanceID == canonicalizeInstanceName(instanceID)
}

// ExternalID returns the cloud provider ID of the specified instance (deprecated).
func (gce *GCECloud) ExternalID(instance string) (string, error) {
	if gce.useMetadataServer {
		// Use metadata, if possible, to fetch ID. See issue #12000
		if gce.isCurrentInstance(instance) {
			externalInstanceID, err := getCurrentExternalIDViaMetadata()
			if err == nil {
				return externalInstanceID, nil
			}
		}
	}

	// Fallback to GCE API call if metadata server fails to retrieve ID
	inst, err := gce.getInstanceByName(instance)
	if err != nil {
		return "", err
	}
	return strconv.FormatUint(inst.ID, 10), nil
}

// InstanceID returns the cloud provider ID of the specified instance.
func (gce *GCECloud) InstanceID(instanceName string) (string, error) {
	if gce.useMetadataServer {
		// Use metadata, if possible, to fetch ID. See issue #12000
		if gce.isCurrentInstance(instanceName) {
			projectID, zone, err := getProjectAndZone()
			if err == nil {
				return projectID + "/" + zone + "/" + canonicalizeInstanceName(instanceName), nil
			}
		}
	}
	instance, err := gce.getInstanceByName(instanceName)
	if err != nil {
		return "", err
	}
	return gce.projectID + "/" + instance.Zone + "/" + instance.Name, nil
}

// InstanceType returns the type of the specified instance.
func (gce *GCECloud) InstanceType(instanceName string) (string, error) {
	if gce.useMetadataServer {
		// Use metadata, if possible, to fetch ID. See issue #12000
		if gce.isCurrentInstance(instanceName) {
			mType, err := getCurrentMachineTypeViaMetadata()
			if err == nil {
				return mType, nil
			}
		}
	}
	instance, err := gce.getInstanceByName(instanceName)
	if err != nil {
		return "", err
	}
	return instance.Type, nil
}

// List is an implementation of Instances.List.
func (gce *GCECloud) List(filter string) ([]string, error) {
	var instances []string
	// TODO: Parallelize, although O(zones) so not too bad (N <= 3 typically)
	for _, zone := range gce.managedZones {
		pageToken := ""
		page := 0
		for ; page == 0 || (pageToken != "" && page < maxPages); page++ {
			listCall := gce.service.Instances.List(gce.projectID, zone)
			if len(filter) > 0 {
				listCall = listCall.Filter("name eq " + filter)
			}
			if pageToken != "" {
				listCall = listCall.PageToken(pageToken)
			}
			res, err := listCall.Do()
			if err != nil {
				return nil, err
			}
			pageToken = res.NextPageToken
			for _, instance := range res.Items {
				instances = append(instances, instance.Name)
			}
		}
		if page >= maxPages {
			glog.Errorf("List exceeded maxPages=%d for Instances.List: truncating.", maxPages)
		}
	}
	return instances, nil
}

func getMetadataValue(metadata *compute.Metadata, key string) (string, bool) {
	for _, item := range metadata.Items {
		if item.Key == key {
			return *item.Value, true
		}
	}
	return "", false
}

func truncateClusterName(clusterName string) string {
	if len(clusterName) > 26 {
		return clusterName[:26]
	}
	return clusterName
}

func (gce *GCECloud) ListRoutes(clusterName string) ([]*cloudprovider.Route, error) {
	var routes []*cloudprovider.Route
	pageToken := ""
	page := 0
	for ; page == 0 || (pageToken != "" && page < maxPages); page++ {
		listCall := gce.service.Routes.List(gce.projectID)

		prefix := truncateClusterName(clusterName)
		listCall = listCall.Filter("name eq " + prefix + "-.*")
		if pageToken != "" {
			listCall = listCall.PageToken(pageToken)
		}
		res, err := listCall.Do()
		if err != nil {
			glog.Errorf("Error getting routes from GCE: %v", err)
			return nil, err
		}
		pageToken = res.NextPageToken
		for _, r := range res.Items {
			if r.Network != gce.networkURL {
				continue
			}
			// Not managed if route description != "k8s-node-route"
			if r.Description != k8sNodeRouteTag {
				continue
			}
			// Not managed if route name doesn't start with <clusterName>
			if !strings.HasPrefix(r.Name, prefix) {
				continue
			}

			target := path.Base(r.NextHopInstance)
			routes = append(routes, &cloudprovider.Route{Name: r.Name, TargetInstance: target, DestinationCIDR: r.DestRange})
		}
	}
	if page >= maxPages {
		glog.Errorf("ListRoutes exceeded maxPages=%d for Routes.List; truncating.", maxPages)
	}
	return routes, nil
}

func gceNetworkURL(project, network string) string {
	return fmt.Sprintf("https://www.googleapis.com/compute/v1/projects/%s/global/networks/%s", project, network)
}

func (gce *GCECloud) CreateRoute(clusterName string, nameHint string, route *cloudprovider.Route) error {
	routeName := truncateClusterName(clusterName) + "-" + nameHint

	targetInstance, err := gce.getInstanceByName(route.TargetInstance)
	if err != nil {
		return err
	}
	insertOp, err := gce.service.Routes.Insert(gce.projectID, &compute.Route{
		Name:            routeName,
		DestRange:       route.DestinationCIDR,
		NextHopInstance: fmt.Sprintf("zones/%s/instances/%s", targetInstance.Zone, targetInstance.Name),
		Network:         gce.networkURL,
		Priority:        1000,
		Description:     k8sNodeRouteTag,
	}).Do()
	if err != nil {
		return err
	}
	return gce.waitForGlobalOp(insertOp)
}

func (gce *GCECloud) DeleteRoute(clusterName string, route *cloudprovider.Route) error {
	deleteOp, err := gce.service.Routes.Delete(gce.projectID, route.Name).Do()
	if err != nil {
		return err
	}
	return gce.waitForGlobalOp(deleteOp)
}

func (gce *GCECloud) GetZone() (cloudprovider.Zone, error) {
	return cloudprovider.Zone{
		FailureDomain: gce.localZone,
		Region:        gce.region,
	}, nil
}

// encodeDiskTags encodes requested volume tags into JSON string, as GCE does
// not support tags on GCE PDs and we use Description field as fallback.
func (gce *GCECloud) encodeDiskTags(tags map[string]string) (string, error) {
	if len(tags) == 0 {
		// No tags -> empty JSON
		return "", nil
	}

	enc, err := json.Marshal(tags)
	if err != nil {
		return "", err
	}
	return string(enc), nil
}

// CreateDisk creates a new Persistent Disk, with the specified name & size, in
// the specified zone. It stores specified tags endoced in JSON in Description
// field.
func (gce *GCECloud) CreateDisk(name string, zone string, sizeGb int64, tags map[string]string) error {
	tagsStr, err := gce.encodeDiskTags(tags)
	if err != nil {
		return err
	}

	diskToCreate := &compute.Disk{
		Name:        name,
		SizeGb:      sizeGb,
		Description: tagsStr,
	}

	createOp, err := gce.service.Disks.Insert(gce.projectID, zone, diskToCreate).Do()
	if err != nil {
		return err
	}

	return gce.waitForZoneOp(createOp, zone)
}

func (gce *GCECloud) DeleteDisk(diskToDelete string) error {
	disk, err := gce.getDiskByNameUnknownZone(diskToDelete)
	if err != nil {
		return err
	}

	deleteOp, err := gce.service.Disks.Delete(gce.projectID, disk.Zone, disk.Name).Do()
	if err != nil {
		return err
	}

	return gce.waitForZoneOp(deleteOp, disk.Zone)
}

// Builds the labels that should be automatically added to a PersistentVolume backed by a GCE PD
// Specifically, this builds FailureDomain (zone) and Region labels.
// The PersistentVolumeLabel admission controller calls this and adds the labels when a PV is created.
func (gce *GCECloud) GetAutoLabelsForPD(name string) (map[string]string, error) {
	disk, err := gce.getDiskByNameUnknownZone(name)
	if err != nil {
		return nil, err
	}

	zone := disk.Zone
	region, err := GetGCERegion(zone)
	if err != nil {
		return nil, err
	}

	if zone == "" || region == "" {
		// Unexpected, but sanity-check
		return nil, fmt.Errorf("PD did not have zone/region information: %q", disk.Name)
	}

	labels := make(map[string]string)
	labels[unversioned.LabelZoneFailureDomain] = zone
	labels[unversioned.LabelZoneRegion] = region

	return labels, nil
}

func (gce *GCECloud) AttachDisk(diskName, instanceID string, readOnly bool) error {
	instance, err := gce.getInstanceByName(instanceID)
	if err != nil {
		return fmt.Errorf("error getting instance %q", instanceID)
	}
	disk, err := gce.getDiskByName(diskName, instance.Zone)
	if err != nil {
		return err
	}
	readWrite := "READ_WRITE"
	if readOnly {
		readWrite = "READ_ONLY"
	}
	attachedDisk := gce.convertDiskToAttachedDisk(disk, readWrite)

	attachOp, err := gce.service.Instances.AttachDisk(gce.projectID, disk.Zone, instanceID, attachedDisk).Do()
	if err != nil {
		return err
	}

	return gce.waitForZoneOp(attachOp, disk.Zone)
}

func (gce *GCECloud) DetachDisk(devicePath, instanceID string) error {
	inst, err := gce.getInstanceByName(instanceID)
	if err != nil {
		return fmt.Errorf("error getting instance %q", instanceID)
	}

	detachOp, err := gce.service.Instances.DetachDisk(gce.projectID, inst.Zone, inst.Name, devicePath).Do()
	if err != nil {
		return err
	}

	return gce.waitForZoneOp(detachOp, inst.Zone)
}

func (gce *GCECloud) DiskIsAttached(diskName, instanceID string) (bool, error) {
	instance, err := gce.getInstanceByName(instanceID)
	if err != nil {
		return false, err
	}

	for _, disk := range instance.Disks {
		if disk.DeviceName == diskName {
			// Disk is still attached to node
			return true, nil
		}
	}

	return false, nil
}

// Returns a gceDisk for the disk, if it is found in the specified zone.
// If not found, returns (nil, nil)
func (gce *GCECloud) findDiskByName(diskName string, zone string) (*gceDisk, error) {
	disk, err := gce.service.Disks.Get(gce.projectID, zone, diskName).Do()
	if err == nil {
		d := &gceDisk{
			Zone: lastComponent(disk.Zone),
			Name: disk.Name,
			Kind: disk.Kind,
		}
		return d, nil
	}
	if !isHTTPErrorCode(err, http.StatusNotFound) {
		return nil, err
	}
	return nil, nil
}

// Like findDiskByName, but returns an error if the disk is not found
func (gce *GCECloud) getDiskByName(diskName string, zone string) (*gceDisk, error) {
	disk, err := gce.findDiskByName(diskName, zone)
	if disk == nil && err == nil {
		return nil, fmt.Errorf("GCE persistent disk not found: %q", diskName)
	}
	return disk, err
}

// Scans all managed zones to return the GCE PD
// Prefer getDiskByName, if the zone can be established
func (gce *GCECloud) getDiskByNameUnknownZone(diskName string) (*gceDisk, error) {
	// Note: this is the gotcha right now with GCE PD support:
	// disk names are not unique per-region.
	// (I can create two volumes with name "myvol" in e.g. us-central1-b & us-central1-f)
	// For now, this is simply undefined behvaiour.
	//
	// In future, we will have to require users to qualify their disk
	// "us-central1-a/mydisk".  We could do this for them as part of
	// admission control, but that might be a little weird (values changing
	// on create)

	var found *gceDisk
	for _, zone := range gce.managedZones {
		disk, err := gce.findDiskByName(diskName, zone)
		if err != nil {
			return nil, err
		}
		if found != nil {
			return nil, fmt.Errorf("GCE persistent disk name was found in multiple zones: %q", diskName)
		}
		found = disk
	}
	if found != nil {
		return found, nil
	}
	return nil, fmt.Errorf("GCE persistent disk not found: %q", diskName)
}

// GetGCERegion returns region of the gce zone. Zone names
// are of the form: ${region-name}-${ix}.
// For example "us-central1-b" has a region of "us-central1".
// So we look for the last '-' and trim to just before that.
func GetGCERegion(zone string) (string, error) {
	ix := strings.LastIndex(zone, "-")
	if ix == -1 {
		return "", fmt.Errorf("unexpected zone: %s", zone)
	}
	return zone[:ix], nil
}

// Converts a Disk resource to an AttachedDisk resource.
func (gce *GCECloud) convertDiskToAttachedDisk(disk *gceDisk, readWrite string) *compute.AttachedDisk {
	return &compute.AttachedDisk{
		DeviceName: disk.Name,
		Kind:       disk.Kind,
		Mode:       readWrite,
		Source:     "https://" + path.Join("www.googleapis.com/compute/v1/projects/", gce.projectID, "zones", disk.Zone, "disks", disk.Name),
		Type:       "PERSISTENT",
	}
}

func (gce *GCECloud) listClustersInZone(zone string) ([]string, error) {
	// TODO: use PageToken to list all not just the first 500
	list, err := gce.containerService.Projects.Zones.Clusters.List(gce.projectID, zone).Do()
	if err != nil {
		return nil, err
	}
	result := []string{}
	for _, cluster := range list.Clusters {
		result = append(result, cluster.Name)
	}
	return result, nil
}

func (gce *GCECloud) ListClusters() ([]string, error) {
	allClusters := []string{}

	for _, zone := range gce.managedZones {
		clusters, err := gce.listClustersInZone(zone)
		if err != nil {
			return nil, err
		}
		// TODO: Scoping?  Do we need to qualify the cluster name?
		allClusters = append(allClusters, clusters...)
	}

	return allClusters, nil
}

func (gce *GCECloud) Master(clusterName string) (string, error) {
	return "k8s-" + clusterName + "-master.internal", nil
}

type gceInstance struct {
	Zone  string
	Name  string
	ID    uint64
	Disks []*compute.AttachedDisk
	Type  string
}

type gceDisk struct {
	Zone string
	Name string
	Kind string
}

// Gets the named instances, returning cloudprovider.InstanceNotFound if any instance is not found
func (gce *GCECloud) getInstancesByNames(names []string) ([]*gceInstance, error) {
	instances := make(map[string]*gceInstance)

	for _, name := range names {
		name = canonicalizeInstanceName(name)
		instances[name] = nil
	}

	for _, zone := range gce.managedZones {
		var remaining []string
		for name, instance := range instances {
			if instance == nil {
				remaining = append(remaining, name)
			}
		}

		if len(remaining) == 0 {
			break
		}

		pageToken := ""
		page := 0
		for ; page == 0 || (pageToken != "" && page < maxPages); page++ {
			listCall := gce.service.Instances.List(gce.projectID, zone)

			// Add the filter for hosts
			listCall = listCall.Filter("name eq (" + strings.Join(remaining, "|") + ")")

			listCall = listCall.Fields("items(name,id,disks,machineType)")
			if pageToken != "" {
				listCall.PageToken(pageToken)
			}

			res, err := listCall.Do()
			if err != nil {
				return nil, err
			}
			pageToken = res.NextPageToken
			for _, i := range res.Items {
				name := i.Name
				instance := &gceInstance{
					Zone:  zone,
					Name:  name,
					ID:    i.Id,
					Disks: i.Disks,
					Type:  lastComponent(i.MachineType),
				}
				instances[name] = instance
			}
		}
		if page >= maxPages {
			glog.Errorf("getInstancesByNames exceeded maxPages=%d for Instances.List: truncating.", maxPages)
		}
	}

	instanceArray := make([]*gceInstance, len(names))
	for i, name := range names {
		instance := instances[name]
		if instance == nil {
			glog.Errorf("Failed to retrieve instance: %q", name)
			return nil, cloudprovider.InstanceNotFound
		}
		instanceArray[i] = instances[name]
	}

	return instanceArray, nil
}

// Gets the named instance, returning cloudprovider.InstanceNotFound if the instance is not found
func (gce *GCECloud) getInstanceByName(name string) (*gceInstance, error) {
	// Avoid changing behaviour when not managing multiple zones
	if len(gce.managedZones) == 1 {
		name = canonicalizeInstanceName(name)
		zone := gce.managedZones[0]
		res, err := gce.service.Instances.Get(gce.projectID, zone, name).Do()
		if err != nil {
			glog.Errorf("getInstanceByName/single-zone: failed to get instance %s; err: %v", name, err)
			if isHTTPErrorCode(err, http.StatusNotFound) {
				return nil, cloudprovider.InstanceNotFound
			}
			return nil, err
		}
		return &gceInstance{
			Zone:  lastComponent(res.Zone),
			Name:  res.Name,
			ID:    res.Id,
			Disks: res.Disks,
			Type:  lastComponent(res.MachineType),
		}, nil
	}

	instances, err := gce.getInstancesByNames([]string{name})
	if err != nil {
		glog.Errorf("getInstanceByName/multiple-zones: failed to get instance %s; err: %v", name, err)
		return nil, err
	}
	if len(instances) != 1 || instances[0] == nil {
		// getInstancesByNames not obeying its contract
		return nil, fmt.Errorf("unexpected return value from getInstancesByNames: %v", instances)
	}
	return instances[0], nil
}

// Returns the last component of a URL, i.e. anything after the last slash
// If there is no slash, returns the whole string
func lastComponent(s string) string {
	lastSlash := strings.LastIndex(s, "/")
	if lastSlash != -1 {
		s = s[lastSlash+1:]
	}
	return s
}
