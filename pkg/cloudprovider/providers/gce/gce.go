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

package gce_cloud

import (
	"fmt"
	"io"
	"net"
	"net/http"
	"path"
	"strconv"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/wait"

	"github.com/golang/glog"
	"github.com/scalingdata/gcfg"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	compute "google.golang.org/api/compute/v1"
	container "google.golang.org/api/container/v1beta1"
	"google.golang.org/api/googleapi"
	"google.golang.org/cloud/compute/metadata"
)

const (
	ProviderName = "gce"
)

const k8sNodeRouteTag = "k8s-node-route"

// GCECloud is an implementation of Interface, TCPLoadBalancer and Instances for Google Compute Engine.
type GCECloud struct {
	service          *compute.Service
	containerService *container.Service
	projectID        string
	zone             string
	instanceID       string
	externalID       string
	networkURL       string
}

type Config struct {
	Global struct {
		TokenURL    string `gcfg:"token-url"`
		TokenBody   string `gcfg:"token-body"`
		ProjectID   string `gcfg:"project-id"`
		NetworkName string `gcfg:"network-name"`
	}
}

func init() {
	cloudprovider.RegisterCloudProvider(ProviderName, func(config io.Reader) (cloudprovider.Interface, error) { return newGCECloud(config) })
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

func getInstanceID() (string, error) {
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

func getCurrentExternalID() (string, error) {
	externalID, err := metadata.Get("instance/id")
	if err != nil {
		return "", fmt.Errorf("couldn't get external ID: %v", err)
	}
	return externalID, nil
}

func getNetworkName() (string, error) {
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

// newGCECloud creates a new instance of GCECloud.
func newGCECloud(config io.Reader) (*GCECloud, error) {
	projectID, zone, err := getProjectAndZone()
	if err != nil {
		return nil, err
	}
	// TODO: if we want to use this on a machine that doesn't have the http://metadata server
	// e.g. on a user's machine (not VM) somewhere, we need to have an alternative for
	// instance id lookup.
	instanceID, err := getInstanceID()
	if err != nil {
		return nil, err
	}
	externalID, err := getCurrentExternalID()
	if err != nil {
		return nil, err
	}
	networkName, err := getNetworkName()
	if err != nil {
		return nil, err
	}
	networkURL := gceNetworkURL(projectID, networkName)
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
	return &GCECloud{
		service:          svc,
		containerService: containerSvc,
		projectID:        projectID,
		zone:             zone,
		instanceID:       instanceID,
		externalID:       externalID,
		networkURL:       networkURL,
	}, nil
}

func (gce *GCECloud) Clusters() (cloudprovider.Clusters, bool) {
	return gce, true
}

// ProviderName returns the cloud provider ID.
func (gce *GCECloud) ProviderName() string {
	return ProviderName
}

// TCPLoadBalancer returns an implementation of TCPLoadBalancer for Google Compute Engine.
func (gce *GCECloud) TCPLoadBalancer() (cloudprovider.TCPLoadBalancer, bool) {
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

func makeComparableHostPath(zone, host string) string {
	host = canonicalizeInstanceName(host)
	return fmt.Sprintf("/zones/%s/instances/%s", zone, host)
}

func hostURLToComparablePath(hostURL string) string {
	idx := strings.Index(hostURL, "/zones/")
	if idx < 0 {
		return ""
	}
	return hostURL[idx:]
}

// Session Affinity Type string
type GCEAffinityType string

const (
	// AffinityTypeNone - no session affinity.
	GCEAffinityTypeNone GCEAffinityType = "None"
	// AffinityTypeClientIP is the Client IP based.
	GCEAffinityTypeClientIP GCEAffinityType = "CLIENT_IP"
	// AffinityTypeClientIP is the Client IP based.
	GCEAffinityTypeClientIPProto GCEAffinityType = "CLIENT_IP_PROTO"
)

func (gce *GCECloud) makeTargetPool(name, region string, hosts []string, affinityType GCEAffinityType) error {
	var instances []string
	for _, host := range hosts {
		instances = append(instances, makeHostURL(gce.projectID, gce.zone, host))
	}
	pool := &compute.TargetPool{
		Name:            name,
		Instances:       instances,
		SessionAffinity: string(affinityType),
	}
	op, err := gce.service.TargetPools.Insert(gce.projectID, region, pool).Do()
	if err != nil {
		return err
	}
	if err = gce.waitForRegionOp(op, region); err != nil {
		return err
	}
	return nil
}

func (gce *GCECloud) targetPoolURL(name, region string) string {
	return fmt.Sprintf("https://www.googleapis.com/compute/v1/projects/%s/regions/%s/targetPools/%s", gce.projectID, region, name)
}

func waitForOp(op *compute.Operation, getOperation func() (*compute.Operation, error)) error {
	pollOp := op
	consecPollFails := 0
	for pollOp.Status != "DONE" {
		var err error
		time.Sleep(3 * time.Second)
		pollOp, err = getOperation()
		if err != nil {
			if consecPollFails == 2 {
				// Only bail if we've seen 3 consecutive polling errors.
				return err
			}
			consecPollFails++
		} else {
			consecPollFails = 0
		}
	}
	if pollOp.Error != nil && len(pollOp.Error.Errors) > 0 {
		return &googleapi.Error{
			Code:    int(pollOp.HttpErrorStatusCode),
			Message: pollOp.Error.Errors[0].Message,
		}
	}
	return nil

}

func (gce *GCECloud) waitForGlobalOp(op *compute.Operation) error {
	return waitForOp(op, func() (*compute.Operation, error) {
		return gce.service.GlobalOperations.Get(gce.projectID, op.Name).Do()
	})
}

func (gce *GCECloud) waitForRegionOp(op *compute.Operation, region string) error {
	return waitForOp(op, func() (*compute.Operation, error) {
		return gce.service.RegionOperations.Get(gce.projectID, region, op.Name).Do()
	})
}

func (gce *GCECloud) waitForZoneOp(op *compute.Operation) error {
	return waitForOp(op, func() (*compute.Operation, error) {
		return gce.service.ZoneOperations.Get(gce.projectID, gce.zone, op.Name).Do()
	})
}

// GetTCPLoadBalancer is an implementation of TCPLoadBalancer.GetTCPLoadBalancer
func (gce *GCECloud) GetTCPLoadBalancer(name, region string) (*api.LoadBalancerStatus, bool, error) {
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

// translate from what K8s supports to what the cloud provider supports for session affinity.
func translateAffinityType(affinityType api.ServiceAffinity) GCEAffinityType {
	switch affinityType {
	case api.ServiceAffinityClientIP:
		return GCEAffinityTypeClientIP
	case api.ServiceAffinityNone:
		return GCEAffinityTypeNone
	default:
		glog.Errorf("unexpected affinity type: %v", affinityType)
		return GCEAffinityTypeNone
	}
}

func makeFirewallName(name string) string {
	return fmt.Sprintf("k8s-fw-%s", name)
}

// EnsureTCPLoadBalancer is an implementation of TCPLoadBalancer.EnsureTCPLoadBalancer.
// TODO(a-robinson): Don't just ignore specified IP addresses. Check if they're
// owned by the project and available to be used, and use them if they are.
func (gce *GCECloud) EnsureTCPLoadBalancer(name, region string, loadBalancerIP net.IP, ports []*api.ServicePort, hosts []string, affinityType api.ServiceAffinity) (*api.LoadBalancerStatus, error) {
	if len(hosts) == 0 {
		return nil, fmt.Errorf("Cannot EnsureTCPLoadBalancer() with no hosts")
	}

	glog.V(2).Infof("Checking if load balancer already exists: %s", name)
	_, exists, err := gce.GetTCPLoadBalancer(name, region)
	if err != nil {
		return nil, fmt.Errorf("error checking if GCE load balancer already exists: %v", err)
	}

	// TODO: Implement a more efficient update strategy for common changes than delete & create
	// In particular, if we implement hosts update, we can get rid of UpdateHosts
	if exists {
		err := gce.EnsureTCPLoadBalancerDeleted(name, region)
		if err != nil {
			return nil, fmt.Errorf("error deleting existing GCE load balancer: %v", err)
		}
	}

	err = gce.makeTargetPool(name, region, hosts, translateAffinityType(affinityType))
	if err != nil {
		if !isHTTPErrorCode(err, http.StatusConflict) {
			return nil, err
		}
		glog.Infof("Creating forwarding rule pointing at target pool that already exists: %v", err)
	}

	if len(ports) == 0 {
		return nil, fmt.Errorf("no ports specified for GCE load balancer")
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
	req := &compute.ForwardingRule{
		Name:       name,
		IPProtocol: "TCP",
		PortRange:  fmt.Sprintf("%d-%d", minPort, maxPort),
		Target:     gce.targetPoolURL(name, region),
	}
	if loadBalancerIP != nil {
		req.IPAddress = loadBalancerIP.String()
	}

	op, err := gce.service.ForwardingRules.Insert(gce.projectID, region, req).Do()
	if err != nil && !isHTTPErrorCode(err, http.StatusConflict) {
		return nil, err
	}
	if op != nil {
		err = gce.waitForRegionOp(op, region)
		if err != nil && !isHTTPErrorCode(err, http.StatusConflict) {
			return nil, err
		}
	}
	fwd, err := gce.service.ForwardingRules.Get(gce.projectID, region, name).Do()
	if err != nil {
		return nil, err
	}

	allowedPorts := make([]string, len(ports))
	for ix := range ports {
		allowedPorts[ix] = strconv.Itoa(ports[ix].Port)
	}

	hostTag := gce.computeHostTag(hosts[0])

	firewall := &compute.Firewall{
		Name:         makeFirewallName(name),
		Description:  fmt.Sprintf("KubernetesAutoGenerated_OnlyAllowTrafficForDestinationIP_%s", fwd.IPAddress),
		Network:      gce.networkURL,
		SourceRanges: []string{"0.0.0.0/0"},
		TargetTags:   []string{hostTag},
		Allowed: []*compute.FirewallAllowed{
			{
				IPProtocol: "tcp",
				Ports:      allowedPorts,
			},
		},
	}
	if op, err = gce.service.Firewalls.Insert(gce.projectID, firewall).Do(); err != nil && !isHTTPErrorCode(err, http.StatusConflict) {
		return nil, err
	}
	if err = gce.waitForGlobalOp(op); err != nil && !isHTTPErrorCode(err, http.StatusConflict) {
		return nil, err
	}

	status := &api.LoadBalancerStatus{}
	status.Ingress = []api.LoadBalancerIngress{{IP: fwd.IPAddress}}
	return status, nil
}

// This is kind of hacky, but the managed instance group adds 4 random chars and a hyphen
// to the base name. Older naming schemes put a hyphen and an incrementing index after
// the base name. Thus we pull off the characters after the final dash to support both.
func (gce *GCECloud) computeHostTag(host string) string {
	host = strings.SplitN(host, ".", 2)[0]
	lastHyphen := strings.LastIndex(host, "-")
	if lastHyphen == -1 {
		return host
	}
	return host[:lastHyphen]
}

// UpdateTCPLoadBalancer is an implementation of TCPLoadBalancer.UpdateTCPLoadBalancer.
func (gce *GCECloud) UpdateTCPLoadBalancer(name, region string, hosts []string) error {
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
		link := makeComparableHostPath(gce.zone, host)
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

// EnsureTCPLoadBalancerDeleted is an implementation of TCPLoadBalancer.EnsureTCPLoadBalancerDeleted.
func (gce *GCECloud) EnsureTCPLoadBalancerDeleted(name, region string) error {
	op, err := gce.service.ForwardingRules.Delete(gce.projectID, region, name).Do()
	if err != nil && isHTTPErrorCode(err, http.StatusNotFound) {
		glog.Infof("Forwarding rule %s already deleted. Continuing to delete target pool.", name)
	} else if err != nil {
		glog.Warningf("Failed to delete Forwarding Rules %s: got error %s.", name, err.Error())
		return err
	} else {
		err = gce.waitForRegionOp(op, region)
		if err != nil {
			glog.Warningf("Failed waiting for Forwarding Rule %s to be deleted: got error %s.", name, err.Error())
			return err
		}
	}
	op, err = gce.service.TargetPools.Delete(gce.projectID, region, name).Do()
	if err != nil && isHTTPErrorCode(err, http.StatusNotFound) {
		glog.Infof("Target pool %s already deleted.", name)
		return nil
	} else if err != nil {
		glog.Warningf("Failed to delete Target Pool %s, got error %s.", name, err.Error())
		return err
	}
	err = gce.waitForRegionOp(op, region)
	if err != nil {
		glog.Warningf("Failed waiting for Target Pool %s to be deleted: got error %s.", name, err.Error())
	}
	fwName := makeFirewallName(name)
	op, err = gce.service.Firewalls.Delete(gce.projectID, fwName).Do()
	if err != nil && isHTTPErrorCode(err, http.StatusNotFound) {
		glog.Infof("Firewall doesn't exist, moving on to deleting target pool.")
	} else if err != nil {
		glog.Warningf("Failed to delete firewall %s, got error %v", fwName, err)
		return err
	} else {
		if err = gce.waitForGlobalOp(op); err != nil {
			glog.Warningf("Failed waiting for Firewall %s to be deleted.  Got error: %v", fwName, err)
			return err
		}
	}
	return err
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

// Return the instances matching the relevant name.
func (gce *GCECloud) getInstanceByName(name string) (*compute.Instance, error) {
	name = canonicalizeInstanceName(name)
	res, err := gce.service.Instances.Get(gce.projectID, gce.zone, name).Do()
	if err != nil {
		glog.Errorf("Failed to retrieve TargetInstance resource for instance: %s", name)
		if apiErr, ok := err.(*googleapi.Error); ok && apiErr.Code == http.StatusNotFound {
			return nil, cloudprovider.InstanceNotFound
		}
		return nil, err
	}
	return res, nil
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
				if strings.Contains(item.Value, keyString) {
					// We've already added the key
					glog.Info("SSHKey already in project metadata")
					return true, nil
				}
				item.Value = item.Value + "\n" + keyString
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
					Value: keyString,
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

func (gce *GCECloud) isCurrentInstance(instance string) bool {
	return gce.instanceID == canonicalizeInstanceName(instance)
}

// ExternalID returns the cloud provider ID of the specified instance (deprecated).
func (gce *GCECloud) ExternalID(instance string) (string, error) {
	// if we are asking about the current instance, just go to metadata
	if gce.isCurrentInstance(instance) {
		return gce.externalID, nil
	}
	inst, err := gce.getInstanceByName(instance)
	if err != nil {
		return "", err
	}
	return strconv.FormatUint(inst.Id, 10), nil
}

// InstanceID returns the cloud provider ID of the specified instance.
func (gce *GCECloud) InstanceID(instance string) (string, error) {
	return gce.projectID + "/" + gce.zone + "/" + canonicalizeInstanceName(instance), nil
}

// List is an implementation of Instances.List.
func (gce *GCECloud) List(filter string) ([]string, error) {
	listCall := gce.service.Instances.List(gce.projectID, gce.zone)
	if len(filter) > 0 {
		listCall = listCall.Filter("name eq " + filter)
	}
	res, err := listCall.Do()
	if err != nil {
		return nil, err
	}
	var instances []string
	for _, instance := range res.Items {
		instances = append(instances, instance.Name)
	}
	return instances, nil
}

func getMetadataValue(metadata *compute.Metadata, key string) (string, bool) {
	for _, item := range metadata.Items {
		if item.Key == key {
			return item.Value, true
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
	listCall := gce.service.Routes.List(gce.projectID)

	prefix := truncateClusterName(clusterName)
	listCall = listCall.Filter("name eq " + prefix + "-.*")

	res, err := listCall.Do()
	if err != nil {
		return nil, err
	}
	var routes []*cloudprovider.Route
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
	return routes, nil
}

func gceNetworkURL(project, network string) string {
	return fmt.Sprintf("https://www.googleapis.com/compute/v1/projects/%s/global/networks/%s", project, network)
}

func (gce *GCECloud) CreateRoute(clusterName string, nameHint string, route *cloudprovider.Route) error {
	routeName := truncateClusterName(clusterName) + "-" + nameHint

	instanceName := canonicalizeInstanceName(route.TargetInstance)
	insertOp, err := gce.service.Routes.Insert(gce.projectID, &compute.Route{
		Name:            routeName,
		DestRange:       route.DestinationCIDR,
		NextHopInstance: fmt.Sprintf("zones/%s/instances/%s", gce.zone, instanceName),
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
	region, err := getGceRegion(gce.zone)
	if err != nil {
		return cloudprovider.Zone{}, err
	}
	return cloudprovider.Zone{
		FailureDomain: gce.zone,
		Region:        region,
	}, nil
}

func (gce *GCECloud) AttachDisk(diskName string, readOnly bool) error {
	disk, err := gce.getDisk(diskName)
	if err != nil {
		return err
	}
	readWrite := "READ_WRITE"
	if readOnly {
		readWrite = "READ_ONLY"
	}
	attachedDisk := gce.convertDiskToAttachedDisk(disk, readWrite)
	_, err = gce.service.Instances.AttachDisk(gce.projectID, gce.zone, gce.instanceID, attachedDisk).Do()
	if err != nil {
		// Check if the disk is already attached to this instance.  We do this only
		// in the error case, since it is expected to be exceptional.
		instance, err := gce.service.Instances.Get(gce.projectID, gce.zone, gce.instanceID).Do()
		if err != nil {
			return err
		}
		for _, disk := range instance.Disks {
			if disk.Source == attachedDisk.Source {
				// Disk is already attached, we're good to go.
				return nil
			}
		}

	}
	return err
}

func (gce *GCECloud) DetachDisk(devicePath string) error {
	_, err := gce.service.Instances.DetachDisk(gce.projectID, gce.zone, gce.instanceID, devicePath).Do()
	return err
}

func (gce *GCECloud) getDisk(diskName string) (*compute.Disk, error) {
	return gce.service.Disks.Get(gce.projectID, gce.zone, diskName).Do()
}

// getGceRegion returns region of the gce zone. Zone names
// are of the form: ${region-name}-${ix}.
// For example "us-central1-b" has a region of "us-central1".
// So we look for the last '-' and trim to just before that.
func getGceRegion(zone string) (string, error) {
	ix := strings.LastIndex(zone, "-")
	if ix == -1 {
		return "", fmt.Errorf("unexpected zone: %s", zone)
	}
	return zone[:ix], nil
}

// Converts a Disk resource to an AttachedDisk resource.
func (gce *GCECloud) convertDiskToAttachedDisk(disk *compute.Disk, readWrite string) *compute.AttachedDisk {
	return &compute.AttachedDisk{
		DeviceName: disk.Name,
		Kind:       disk.Kind,
		Mode:       readWrite,
		Source:     "https://" + path.Join("www.googleapis.com/compute/v1/projects/", gce.projectID, "zones", gce.zone, "disks", disk.Name),
		Type:       "PERSISTENT",
	}
}

func (gce *GCECloud) ListClusters() ([]string, error) {
	list, err := gce.containerService.Projects.Clusters.List(gce.projectID).Do()
	if err != nil {
		return nil, err
	}
	result := []string{}
	for _, cluster := range list.Clusters {
		result = append(result, cluster.Name)
	}
	return result, nil
}

func (gce *GCECloud) Master(clusterName string) (string, error) {
	return "k8s-" + clusterName + "-master.internal", nil
}
