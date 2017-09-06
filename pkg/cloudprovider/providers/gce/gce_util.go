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
	"net/http"
	"regexp"
	"strings"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"

	"cloud.google.com/go/compute/metadata"
	compute "google.golang.org/api/compute/v1"
	"google.golang.org/api/googleapi"
)

type gceInstance struct {
	Zone  string
	Name  string
	ID    uint64
	Disks []*compute.AttachedDisk
	Type  string
}

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
	allow := strings.Join(allPorts, ",")
	srcRngs := strings.Join(fw.SourceRanges, ",")
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
		return NetworkTierDefault.ToGCEValue(), nil
	}
	// Can't get the network tier, just return an error.
	return "", err
}
