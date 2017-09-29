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
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"

	"cloud.google.com/go/compute/metadata"
	"github.com/golang/glog"
	computealpha "google.golang.org/api/compute/v0.alpha"
	computebeta "google.golang.org/api/compute/v0.beta"
	compute "google.golang.org/api/compute/v1"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/cloudprovider"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
)

const (
	defaultZone = ""
)

func newInstancesMetricContext(request, zone string) *metricContext {
	return newGenericMetricContext("instances", request, unusedMetricLabel, zone, computeV1Version)
}

func splitNodesByZone(nodes []*v1.Node) map[string][]*v1.Node {
	zones := make(map[string][]*v1.Node)
	for _, n := range nodes {
		z := getZone(n)
		if z != defaultZone {
			zones[z] = append(zones[z], n)
		}
	}
	return zones
}

func getZone(n *v1.Node) string {
	zone, ok := n.Labels[kubeletapis.LabelZoneFailureDomain]
	if !ok {
		return defaultZone
	}
	return zone
}

// ToInstanceReferences returns instance references by links
func (gce *GCECloud) ToInstanceReferences(zone string, instanceNames []string) (refs []*compute.InstanceReference) {
	for _, ins := range instanceNames {
		instanceLink := makeHostURL(gce.service.BasePath, gce.projectID, zone, ins)
		refs = append(refs, &compute.InstanceReference{Instance: instanceLink})
	}
	return refs
}

// NodeAddresses is an implementation of Instances.NodeAddresses.
func (gce *GCECloud) NodeAddresses(_ types.NodeName) ([]v1.NodeAddress, error) {
	internalIP, err := metadata.Get("instance/network-interfaces/0/ip")
	if err != nil {
		return nil, fmt.Errorf("couldn't get internal IP: %v", err)
	}
	externalIP, err := metadata.Get("instance/network-interfaces/0/access-configs/0/external-ip")
	if err != nil {
		return nil, fmt.Errorf("couldn't get external IP: %v", err)
	}
	return []v1.NodeAddress{
		{Type: v1.NodeInternalIP, Address: internalIP},
		{Type: v1.NodeExternalIP, Address: externalIP},
	}, nil
}

// This method will not be called from the node that is requesting this ID.
// i.e. metadata service and other local methods cannot be used here
func (gce *GCECloud) NodeAddressesByProviderID(providerID string) ([]v1.NodeAddress, error) {
	project, zone, name, err := splitProviderID(providerID)
	if err != nil {
		return []v1.NodeAddress{}, err
	}

	instance, err := gce.service.Instances.Get(project, zone, canonicalizeInstanceName(name)).Do()
	if err != nil {
		return []v1.NodeAddress{}, fmt.Errorf("error while querying for providerID %q: %v", providerID, err)
	}

	if len(instance.NetworkInterfaces) < 1 {
		return []v1.NodeAddress{}, fmt.Errorf("could not find network interfaces for providerID %q", providerID)
	}
	networkInterface := instance.NetworkInterfaces[0]

	nodeAddresses := []v1.NodeAddress{{Type: v1.NodeInternalIP, Address: networkInterface.NetworkIP}}
	for _, config := range networkInterface.AccessConfigs {
		nodeAddresses = append(nodeAddresses, v1.NodeAddress{Type: v1.NodeExternalIP, Address: config.NatIP})
	}

	return nodeAddresses, nil
}

// InstanceTypeByProviderID returns the cloudprovider instance type of the node
// with the specified unique providerID This method will not be called from the
// node that is requesting this ID. i.e. metadata service and other local
// methods cannot be used here
func (gce *GCECloud) InstanceTypeByProviderID(providerID string) (string, error) {
	project, zone, name, err := splitProviderID(providerID)
	if err != nil {
		return "", err
	}
	instance, err := gce.getInstanceFromProjectInZoneByName(project, zone, name)
	if err != nil {
		return "", err
	}
	return instance.Type, nil
}

// ExternalID returns the cloud provider ID of the node with the specified NodeName (deprecated).
func (gce *GCECloud) ExternalID(nodeName types.NodeName) (string, error) {
	instanceName := mapNodeNameToInstanceName(nodeName)
	if gce.useMetadataServer {
		// Use metadata, if possible, to fetch ID. See issue #12000
		if gce.isCurrentInstance(instanceName) {
			externalInstanceID, err := getCurrentExternalIDViaMetadata()
			if err == nil {
				return externalInstanceID, nil
			}
		}
	}

	// Fallback to GCE API call if metadata server fails to retrieve ID
	inst, err := gce.getInstanceByName(instanceName)
	if err != nil {
		return "", err
	}
	return strconv.FormatUint(inst.ID, 10), nil
}

// InstanceExistsByProviderID returns true if the instance with the given provider id still exists and is running.
// If false is returned with no error, the instance will be immediately deleted by the cloud controller manager.
func (gce *GCECloud) InstanceExistsByProviderID(providerID string) (bool, error) {
	return false, cloudprovider.NotImplemented
}

// InstanceID returns the cloud provider ID of the node with the specified NodeName.
func (gce *GCECloud) InstanceID(nodeName types.NodeName) (string, error) {
	instanceName := mapNodeNameToInstanceName(nodeName)
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

// InstanceType returns the type of the specified node with the specified NodeName.
func (gce *GCECloud) InstanceType(nodeName types.NodeName) (string, error) {
	instanceName := mapNodeNameToInstanceName(nodeName)
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

		mc := newInstancesMetricContext("add_ssh_key", "")
		op, err := gce.service.Projects.SetCommonInstanceMetadata(
			gce.projectID, project.CommonInstanceMetadata).Do()

		if err != nil {
			glog.Errorf("Could not Set Metadata: %v", err)
			mc.Observe(err)
			return false, nil
		}

		if err := gce.waitForGlobalOp(op, mc); err != nil {
			glog.Errorf("Could not Set Metadata: %v", err)
			return false, nil
		}

		glog.Infof("Successfully added sshKey to project metadata")
		return true, nil
	})
}

// GetAllZones returns all the zones in which nodes are running
func (gce *GCECloud) GetAllZones() (sets.String, error) {
	// Fast-path for non-multizone
	if len(gce.managedZones) == 1 {
		return sets.NewString(gce.managedZones...), nil
	}

	// TODO: Caching, but this is currently only called when we are creating a volume,
	// which is a relatively infrequent operation, and this is only # zones API calls
	zones := sets.NewString()

	// TODO: Parallelize, although O(zones) so not too bad (N <= 3 typically)
	for _, zone := range gce.managedZones {
		mc := newInstancesMetricContext("list", zone)
		// We only retrieve one page in each zone - we only care about existence
		listCall := gce.service.Instances.List(gce.projectID, zone)

		// No filter: We assume that a zone is either used or unused
		// We could only consider running nodes (like we do in List above),
		// but probably if instances are starting we still want to consider them.
		// I think we should wait until we have a reason to make the
		// call one way or the other; we generally can't guarantee correct
		// volume spreading if the set of zones is changing
		// (and volume spreading is currently only a heuristic).
		// Long term we want to replace GetAllZones (which primarily supports volume
		// spreading) with a scheduler policy that is able to see the global state of
		// volumes and the health of zones.

		// Just a minimal set of fields - we only care about existence
		listCall = listCall.Fields("items(name)")
		res, err := listCall.Do()
		if err != nil {
			return nil, mc.Observe(err)
		}
		mc.Observe(nil)

		if len(res.Items) != 0 {
			zones.Insert(zone)
		}
	}

	return zones, nil
}

// ListInstanceNames returns a string of instance names seperated by spaces.
func (gce *GCECloud) ListInstanceNames(project, zone string) (string, error) {
	res, err := gce.service.Instances.List(project, zone).Fields("items(name)").Do()
	if err != nil {
		return "", err
	}
	var output string
	for _, item := range res.Items {
		output += item.Name + " "
	}
	return output, nil
}

// DeleteInstance deletes an instance specified by project, zone, and name
func (gce *GCECloud) DeleteInstance(project, zone, name string) (*compute.Operation, error) {
	return gce.service.Instances.Delete(project, zone, name).Do()
}

// Implementation of Instances.CurrentNodeName
func (gce *GCECloud) CurrentNodeName(hostname string) (types.NodeName, error) {
	return types.NodeName(hostname), nil
}

// AliasRanges returns a list of CIDR ranges that are assigned to the
// `node` for allocation to pods. Returns a list of the form
// "<ip>/<netmask>".
func (gce *GCECloud) AliasRanges(nodeName types.NodeName) (cidrs []string, err error) {
	var instance *gceInstance
	instance, err = gce.getInstanceByName(mapNodeNameToInstanceName(nodeName))
	if err != nil {
		return
	}

	var res *computebeta.Instance
	res, err = gce.serviceBeta.Instances.Get(
		gce.projectID, instance.Zone, instance.Name).Do()
	if err != nil {
		return
	}

	for _, networkInterface := range res.NetworkInterfaces {
		for _, aliasIpRange := range networkInterface.AliasIpRanges {
			cidrs = append(cidrs, aliasIpRange.IpCidrRange)
		}
	}
	return
}

// AddAliasToInstance adds an alias to the given instance from the named
// secondary range.
func (gce *GCECloud) AddAliasToInstance(nodeName types.NodeName, alias *net.IPNet) error {

	v1instance, err := gce.getInstanceByName(mapNodeNameToInstanceName(nodeName))
	if err != nil {
		return err
	}
	instance, err := gce.serviceAlpha.Instances.Get(gce.projectID, v1instance.Zone, v1instance.Name).Do()
	if err != nil {
		return err
	}

	switch len(instance.NetworkInterfaces) {
	case 0:
		return fmt.Errorf("Instance %q has no network interfaces", nodeName)
	case 1:
	default:
		glog.Warningf("Instance %q has more than one network interface, using only the first (%v)",
			nodeName, instance.NetworkInterfaces)
	}

	iface := instance.NetworkInterfaces[0]
	iface.AliasIpRanges = append(iface.AliasIpRanges, &computealpha.AliasIpRange{
		IpCidrRange:         alias.String(),
		SubnetworkRangeName: gce.secondaryRangeName,
	})

	mc := newInstancesMetricContext("addalias", v1instance.Zone)
	op, err := gce.serviceAlpha.Instances.UpdateNetworkInterface(
		gce.projectID, instance.Zone, instance.Name, iface.Name, iface).Do()
	if err != nil {
		return mc.Observe(err)
	}
	return gce.waitForZoneOp(op, v1instance.Zone, mc)
}

// Gets the named instances, returning cloudprovider.InstanceNotFound if any instance is not found
func (gce *GCECloud) getInstancesByNames(names []string) ([]*gceInstance, error) {
	instances := make(map[string]*gceInstance)
	remaining := len(names)

	nodeInstancePrefix := gce.nodeInstancePrefix
	for _, name := range names {
		name = canonicalizeInstanceName(name)
		if !strings.HasPrefix(name, gce.nodeInstancePrefix) {
			glog.Warningf("instance '%s' does not conform to prefix '%s', removing filter", name, gce.nodeInstancePrefix)
			nodeInstancePrefix = ""
		}
		instances[name] = nil
	}

	for _, zone := range gce.managedZones {
		if remaining == 0 {
			break
		}

		pageToken := ""
		page := 0
		for ; page == 0 || (pageToken != "" && page < maxPages); page++ {
			listCall := gce.service.Instances.List(gce.projectID, zone)

			if nodeInstancePrefix != "" {
				// Add the filter for hosts
				listCall = listCall.Filter("name eq " + nodeInstancePrefix + ".*")
			}

			// TODO(zmerlynn): Internal bug 29524655
			// listCall = listCall.Fields("items(name,id,disks,machineType)")
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
				if _, ok := instances[name]; !ok {
					continue
				}

				instance := &gceInstance{
					Zone:  zone,
					Name:  name,
					ID:    i.Id,
					Disks: i.Disks,
					Type:  lastComponent(i.MachineType),
				}
				instances[name] = instance
				remaining--
			}
		}
		if page >= maxPages {
			glog.Errorf("getInstancesByNames exceeded maxPages=%d for Instances.List: truncating.", maxPages)
		}
	}

	instanceArray := make([]*gceInstance, len(names))
	for i, name := range names {
		name = canonicalizeInstanceName(name)
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
	for _, zone := range gce.managedZones {
		instance, err := gce.getInstanceFromProjectInZoneByName(gce.projectID, zone, name)
		if err != nil {
			if isHTTPErrorCode(err, http.StatusNotFound) {
				continue
			}
			return nil, err
		}
		return instance, nil
	}

	return nil, cloudprovider.InstanceNotFound
}

func (gce *GCECloud) getInstanceFromProjectInZoneByName(project, zone, name string) (*gceInstance, error) {
	name = canonicalizeInstanceName(name)
	mc := newInstancesMetricContext("get", zone)
	res, err := gce.service.Instances.Get(project, zone, name).Do()
	mc.Observe(err)
	if err != nil {
		glog.Errorf("getInstanceFromProjectInZoneByName: failed to get instance %s; err: %v", name, err)
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

// isCurrentInstance uses metadata server to check if specified
// instanceID matches current machine's instanceID
func (gce *GCECloud) isCurrentInstance(instanceID string) bool {
	currentInstanceID, err := getInstanceIDViaMetadata()
	if err != nil {
		// Log and swallow error
		glog.Errorf("Failed to fetch instanceID via Metadata: %v", err)
		return false
	}

	return currentInstanceID == canonicalizeInstanceName(instanceID)
}

// ComputeHostTags grabs all tags from all instances being added to the pool.
// * The longest tag that is a prefix of the instance name is used
// * If any instance has no matching prefix tag, return error
// Invoking this method to get host tags is risky since it depends on the format
// of the host names in the cluster. Only use it as a fallback if gce.nodeTags
// is unspecified
func (gce *GCECloud) computeHostTags(hosts []*gceInstance) ([]string, error) {
	// TODO: We could store the tags in gceInstance, so we could have already fetched it
	hostNamesByZone := make(map[string]map[string]bool) // map of zones -> map of names -> bool (for easy lookup)
	nodeInstancePrefix := gce.nodeInstancePrefix
	for _, host := range hosts {
		if !strings.HasPrefix(host.Name, gce.nodeInstancePrefix) {
			glog.Warningf("instance '%s' does not conform to prefix '%s', ignoring filter", host, gce.nodeInstancePrefix)
			nodeInstancePrefix = ""
		}

		z, ok := hostNamesByZone[host.Zone]
		if !ok {
			z = make(map[string]bool)
			hostNamesByZone[host.Zone] = z
		}
		z[host.Name] = true
	}

	tags := sets.NewString()

	for zone, hostNames := range hostNamesByZone {
		pageToken := ""
		page := 0
		for ; page == 0 || (pageToken != "" && page < maxPages); page++ {
			listCall := gce.service.Instances.List(gce.projectID, zone)

			if nodeInstancePrefix != "" {
				// Add the filter for hosts
				listCall = listCall.Filter("name eq " + nodeInstancePrefix + ".*")
			}

			// Add the fields we want
			// TODO(zmerlynn): Internal bug 29524655
			// listCall = listCall.Fields("items(name,tags)")

			if pageToken != "" {
				listCall = listCall.PageToken(pageToken)
			}

			res, err := listCall.Do()
			if err != nil {
				return nil, err
			}
			pageToken = res.NextPageToken
			for _, instance := range res.Items {
				if !hostNames[instance.Name] {
					continue
				}

				longest_tag := ""
				for _, tag := range instance.Tags.Items {
					if strings.HasPrefix(instance.Name, tag) && len(tag) > len(longest_tag) {
						longest_tag = tag
					}
				}
				if len(longest_tag) > 0 {
					tags.Insert(longest_tag)
				} else {
					return nil, fmt.Errorf("Could not find any tag that is a prefix of instance name for instance %s", instance.Name)
				}
			}
		}
		if page >= maxPages {
			glog.Errorf("computeHostTags exceeded maxPages=%d for Instances.List: truncating.", maxPages)
		}
	}
	if len(tags) == 0 {
		return nil, fmt.Errorf("No instances found")
	}
	return tags.List(), nil
}

// GetNodeTags will first try returning the list of tags specified in GCE cloud Configuration.
// If they weren't provided, it'll compute the host tags with the given hostnames. If the list
// of hostnames has not changed, a cached set of nodetags are returned.
func (gce *GCECloud) GetNodeTags(nodeNames []string) ([]string, error) {
	// If nodeTags were specified through configuration, use them
	if len(gce.nodeTags) > 0 {
		return gce.nodeTags, nil
	}

	gce.computeNodeTagLock.Lock()
	defer gce.computeNodeTagLock.Unlock()

	// Early return if hosts have not changed
	hosts := sets.NewString(nodeNames...)
	if hosts.Equal(gce.lastKnownNodeNames) {
		return gce.lastComputedNodeTags, nil
	}

	// Get GCE instance data by hostname
	instances, err := gce.getInstancesByNames(nodeNames)
	if err != nil {
		return nil, err
	}

	// Determine list of host tags
	tags, err := gce.computeHostTags(instances)
	if err != nil {
		return nil, err
	}

	// Save the list of tags
	gce.lastKnownNodeNames = hosts
	gce.lastComputedNodeTags = tags
	return tags, nil
}
