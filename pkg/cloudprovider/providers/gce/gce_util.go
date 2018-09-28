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
	"errors"
	"fmt"
	"net"
	"net/http"
	"regexp"
	"sort"
	"strings"
	"sync"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/meta"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/mock"

	"cloud.google.com/go/compute/metadata"
	compute "google.golang.org/api/compute/v1"
	"google.golang.org/api/googleapi"
)

func fakeGCECloud(vals TestClusterValues) (*GCECloud, error) {
	gce := simpleFakeGCECloud(vals)

	gce.AlphaFeatureGate = NewAlphaFeatureGate([]string{})
	gce.nodeInformerSynced = func() bool { return true }

	mockGCE := gce.c.(*cloud.MockGCE)
	mockGCE.MockTargetPools.AddInstanceHook = mock.AddInstanceHook
	mockGCE.MockTargetPools.RemoveInstanceHook = mock.RemoveInstanceHook
	mockGCE.MockForwardingRules.InsertHook = mock.InsertFwdRuleHook
	mockGCE.MockAddresses.InsertHook = mock.InsertAddressHook
	mockGCE.MockAlphaAddresses.InsertHook = mock.InsertAlphaAddressHook
	mockGCE.MockAlphaAddresses.X = mock.AddressAttributes{}
	mockGCE.MockAddresses.X = mock.AddressAttributes{}

	mockGCE.MockInstanceGroups.X = mock.InstanceGroupAttributes{
		InstanceMap: make(map[meta.Key]map[string]*compute.InstanceWithNamedPorts),
		Lock:        &sync.Mutex{},
	}
	mockGCE.MockInstanceGroups.AddInstancesHook = mock.AddInstancesHook
	mockGCE.MockInstanceGroups.RemoveInstancesHook = mock.RemoveInstancesHook
	mockGCE.MockInstanceGroups.ListInstancesHook = mock.ListInstancesHook

	mockGCE.MockRegionBackendServices.UpdateHook = mock.UpdateRegionBackendServiceHook
	mockGCE.MockHealthChecks.UpdateHook = mock.UpdateHealthCheckHook
	mockGCE.MockFirewalls.UpdateHook = mock.UpdateFirewallHook

	keyGA := meta.GlobalKey("key-ga")
	mockGCE.MockZones.Objects[*keyGA] = &cloud.MockZonesObj{
		Obj: &compute.Zone{Name: vals.ZoneName, Region: gce.getRegionLink(vals.Region)},
	}

	return gce, nil
}

type gceInstance struct {
	Zone  string
	Name  string
	ID    uint64
	Disks []*compute.AttachedDisk
	Type  string
}

var (
	autoSubnetIPRange = &net.IPNet{
		IP:   net.ParseIP("10.128.0.0"),
		Mask: net.CIDRMask(9, 32),
	}
)

var providerIdRE = regexp.MustCompile(`^` + ProviderName + `://([^/]+)/([^/]+)/([^/]+)$`)

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

func (gce *GCECloud) raiseFirewallChangeNeededEvent(svc *v1.Service, cmd string) {
	msg := fmt.Sprintf("Firewall change required by network admin: `%v`", cmd)
	if gce.eventRecorder != nil && svc != nil {
		gce.eventRecorder.Event(svc, v1.EventTypeNormal, "LoadBalancerManualChange", msg)
	}
}

// FirewallToGCloudCreateCmd generates a gcloud command to create a firewall with specified params
func FirewallToGCloudCreateCmd(fw *compute.Firewall, projectID string) string {
	args := firewallToGcloudArgs(fw, projectID)
	return fmt.Sprintf("gcloud compute firewall-rules create %v --network %v %v", fw.Name, getNameFromLink(fw.Network), args)
}

// FirewallToGCloudCreateCmd generates a gcloud command to update a firewall to specified params
func FirewallToGCloudUpdateCmd(fw *compute.Firewall, projectID string) string {
	args := firewallToGcloudArgs(fw, projectID)
	return fmt.Sprintf("gcloud compute firewall-rules update %v %v", fw.Name, args)
}

// FirewallToGCloudCreateCmd generates a gcloud command to delete a firewall to specified params
func FirewallToGCloudDeleteCmd(fwName, projectID string) string {
	return fmt.Sprintf("gcloud compute firewall-rules delete %v --project %v", fwName, projectID)
}

func firewallToGcloudArgs(fw *compute.Firewall, projectID string) string {
	var allPorts []string
	for _, a := range fw.Allowed {
		for _, p := range a.Ports {
			allPorts = append(allPorts, fmt.Sprintf("%v:%v", a.IPProtocol, p))
		}
	}

	// Sort all slices to prevent the event from being duped
	sort.Strings(allPorts)
	allow := strings.Join(allPorts, ",")
	sort.Strings(fw.SourceRanges)
	srcRngs := strings.Join(fw.SourceRanges, ",")
	sort.Strings(fw.TargetTags)
	targets := strings.Join(fw.TargetTags, ",")
	return fmt.Sprintf("--description %q --allow %v --source-ranges %v --target-tags %v --project %v", fw.Description, allow, srcRngs, targets, projectID)
}

// Take a GCE instance 'hostname' and break it down to something that can be fed
// to the GCE API client library.  Basically this means reducing 'kubernetes-
// node-2.c.my-proj.internal' to 'kubernetes-node-2' if necessary.
func canonicalizeInstanceName(name string) string {
	ix := strings.Index(name, ".")
	if ix != -1 {
		name = name[:ix]
	}
	return name
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

// mapNodeNameToInstanceName maps a k8s NodeName to a GCE Instance Name
// This is a simple string cast.
func mapNodeNameToInstanceName(nodeName types.NodeName) string {
	return string(nodeName)
}

// mapInstanceToNodeName maps a GCE Instance to a k8s NodeName
func mapInstanceToNodeName(instance *compute.Instance) types.NodeName {
	return types.NodeName(instance.Name)
}

// GetGCERegion returns region of the gce zone. Zone names
// are of the form: ${region-name}-${ix}.
// For example, "us-central1-b" has a region of "us-central1".
// So we look for the last '-' and trim to just before that.
func GetGCERegion(zone string) (string, error) {
	ix := strings.LastIndex(zone, "-")
	if ix == -1 {
		return "", fmt.Errorf("unexpected zone: %s", zone)
	}
	return zone[:ix], nil
}

func isHTTPErrorCode(err error, code int) bool {
	apiErr, ok := err.(*googleapi.Error)
	return ok && apiErr.Code == code
}

func isInUsedByError(err error) bool {
	apiErr, ok := err.(*googleapi.Error)
	if !ok || apiErr.Code != http.StatusBadRequest {
		return false
	}
	return strings.Contains(apiErr.Message, "being used by")
}

// splitProviderID splits a provider's id into core components.
// A providerID is build out of '${ProviderName}://${project-id}/${zone}/${instance-name}'
// See cloudprovider.GetInstanceProviderID.
func splitProviderID(providerID string) (project, zone, instance string, err error) {
	matches := providerIdRE.FindStringSubmatch(providerID)
	if len(matches) != 4 {
		return "", "", "", errors.New("error splitting providerID")
	}
	return matches[1], matches[2], matches[3], nil
}

func equalStringSets(x, y []string) bool {
	if len(x) != len(y) {
		return false
	}
	xString := sets.NewString(x...)
	yString := sets.NewString(y...)
	return xString.Equal(yString)
}

func isNotFound(err error) bool {
	return isHTTPErrorCode(err, http.StatusNotFound)
}

func ignoreNotFound(err error) error {
	if err == nil || isNotFound(err) {
		return nil
	}
	return err
}

func isNotFoundOrInUse(err error) bool {
	return isNotFound(err) || isInUsedByError(err)
}

func isForbidden(err error) bool {
	return isHTTPErrorCode(err, http.StatusForbidden)
}

func makeGoogleAPINotFoundError(message string) error {
	return &googleapi.Error{Code: http.StatusNotFound, Message: message}
}

func makeGoogleAPIError(code int, message string) error {
	return &googleapi.Error{Code: code, Message: message}
}

// TODO(#51665): Remove this once Network Tiers becomes Beta in GCP.
func handleAlphaNetworkTierGetError(err error) (string, error) {
	if isForbidden(err) {
		// Network tier is still an Alpha feature in GCP, and not every project
		// is whitelisted to access the API. If we cannot access the API, just
		// assume the tier is premium.
		return cloud.NetworkTierDefault.ToGCEValue(), nil
	}
	// Can't get the network tier, just return an error.
	return "", err
}

// containsCIDR returns true if outer contains inner.
func containsCIDR(outer, inner *net.IPNet) bool {
	return outer.Contains(firstIPInRange(inner)) && outer.Contains(lastIPInRange(inner))
}

// firstIPInRange returns the first IP in a given IP range.
func firstIPInRange(ipNet *net.IPNet) net.IP {
	return ipNet.IP.Mask(ipNet.Mask)
}

// lastIPInRange returns the last IP in a given IP range.
func lastIPInRange(cidr *net.IPNet) net.IP {
	ip := append([]byte{}, cidr.IP...)
	for i, b := range cidr.Mask {
		ip[i] |= ^b
	}
	return ip
}

// subnetsInCIDR takes a list of subnets for a single region and
// returns subnets which exists in the specified CIDR range.
func subnetsInCIDR(subnets []*compute.Subnetwork, cidr *net.IPNet) ([]*compute.Subnetwork, error) {
	var res []*compute.Subnetwork
	for _, subnet := range subnets {
		_, subnetRange, err := net.ParseCIDR(subnet.IpCidrRange)
		if err != nil {
			return nil, fmt.Errorf("unable to parse CIDR %q for subnet %q: %v", subnet.IpCidrRange, subnet.Name, err)
		}
		if containsCIDR(cidr, subnetRange) {
			res = append(res, subnet)
		}
	}
	return res, nil
}

type netType string

const (
	netTypeLegacy netType = "LEGACY"
	netTypeAuto   netType = "AUTO"
	netTypeCustom netType = "CUSTOM"
)

func typeOfNetwork(network *compute.Network) netType {
	if network.IPv4Range != "" {
		return netTypeLegacy
	}

	if network.AutoCreateSubnetworks {
		return netTypeAuto
	}

	return netTypeCustom
}

func getLocationName(project, zoneOrRegion string) string {
	return fmt.Sprintf("projects/%s/locations/%s", project, zoneOrRegion)
}
