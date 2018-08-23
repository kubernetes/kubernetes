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
	"net"
	"net/http"
	"strings"
	"time"

	"cloud.google.com/go/compute/metadata"
	"github.com/golang/glog"
	computebeta "google.golang.org/api/compute/v0.beta"
	compute "google.golang.org/api/compute/v1"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/filter"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/meta"
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

func makeHostURL(projectsApiEndpoint, projectID, zone, host string) string {
	host = canonicalizeInstanceName(host)
	return projectsApiEndpoint + strings.Join([]string{projectID, "zones", zone, "instances", host}, "/")
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
func (gce *GCECloud) NodeAddresses(_ context.Context, _ types.NodeName) ([]v1.NodeAddress, error) {
	internalIP, err := metadata.Get("instance/network-interfaces/0/ip")
	if err != nil {
		return nil, fmt.Errorf("couldn't get internal IP: %v", err)
	}
	externalIP, err := metadata.Get("instance/network-interfaces/0/access-configs/0/external-ip")
	if err != nil {
		return nil, fmt.Errorf("couldn't get external IP: %v", err)
	}
	addresses := []v1.NodeAddress{
		{Type: v1.NodeInternalIP, Address: internalIP},
		{Type: v1.NodeExternalIP, Address: externalIP},
	}

	if internalDNSFull, err := metadata.Get("instance/hostname"); err != nil {
		glog.Warningf("couldn't get full internal DNS name: %v", err)
	} else {
		addresses = append(addresses,
			v1.NodeAddress{Type: v1.NodeInternalDNS, Address: internalDNSFull},
			v1.NodeAddress{Type: v1.NodeHostName, Address: internalDNSFull},
		)
	}
	return addresses, nil
}

// NodeAddressesByProviderID will not be called from the node that is requesting this ID.
// i.e. metadata service and other local methods cannot be used here
func (gce *GCECloud) NodeAddressesByProviderID(ctx context.Context, providerID string) ([]v1.NodeAddress, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	_, zone, name, err := splitProviderID(providerID)
	if err != nil {
		return []v1.NodeAddress{}, err
	}

	instance, err := gce.c.Instances().Get(ctx, meta.ZonalKey(canonicalizeInstanceName(name), zone))
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

// instanceByProviderID returns the cloudprovider instance of the node
// with the specified unique providerID
func (gce *GCECloud) instanceByProviderID(providerID string) (*gceInstance, error) {
	project, zone, name, err := splitProviderID(providerID)
	if err != nil {
		return nil, err
	}

	instance, err := gce.getInstanceFromProjectInZoneByName(project, zone, name)
	if err != nil {
		if isHTTPErrorCode(err, http.StatusNotFound) {
			return nil, cloudprovider.InstanceNotFound
		}
		return nil, err
	}

	return instance, nil
}

// InstanceShutdownByProviderID returns true if the instance is in safe state to detach volumes
func (gce *GCECloud) InstanceShutdownByProviderID(ctx context.Context, providerID string) (bool, error) {
	return false, cloudprovider.NotImplemented
}

// InstanceTypeByProviderID returns the cloudprovider instance type of the node
// with the specified unique providerID This method will not be called from the
// node that is requesting this ID. i.e. metadata service and other local
// methods cannot be used here
func (gce *GCECloud) InstanceTypeByProviderID(ctx context.Context, providerID string) (string, error) {
	instance, err := gce.instanceByProviderID(providerID)
	if err != nil {
		return "", err
	}

	return instance.Type, nil
}

// InstanceExistsByProviderID returns true if the instance with the given provider id still exists and is running.
// If false is returned with no error, the instance will be immediately deleted by the cloud controller manager.
func (gce *GCECloud) InstanceExistsByProviderID(ctx context.Context, providerID string) (bool, error) {
	_, err := gce.instanceByProviderID(providerID)
	if err != nil {
		if err == cloudprovider.InstanceNotFound {
			return false, nil
		}
		return false, err
	}

	return true, nil
}

// InstanceID returns the cloud provider ID of the node with the specified NodeName.
func (gce *GCECloud) InstanceID(ctx context.Context, nodeName types.NodeName) (string, error) {
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
func (gce *GCECloud) InstanceType(ctx context.Context, nodeName types.NodeName) (string, error) {
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

func (gce *GCECloud) AddSSHKeyToAllInstances(ctx context.Context, user string, keyData []byte) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	return wait.Poll(2*time.Second, 30*time.Second, func() (bool, error) {
		project, err := gce.c.Projects().Get(ctx, gce.projectID)
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
		err = gce.c.Projects().SetCommonInstanceMetadata(ctx, gce.projectID, project.CommonInstanceMetadata)
		mc.Observe(err)

		if err != nil {
			glog.Errorf("Could not Set Metadata: %v", err)
			return false, nil
		}
		glog.Infof("Successfully added sshKey to project metadata")
		return true, nil
	})
}

// GetAllCurrentZones returns all the zones in which k8s nodes are currently running
func (gce *GCECloud) GetAllCurrentZones() (sets.String, error) {
	if gce.nodeInformerSynced == nil {
		glog.Warningf("GCECloud object does not have informers set, should only happen in E2E binary.")
		return gce.GetAllZonesFromCloudProvider()
	}
	gce.nodeZonesLock.Lock()
	defer gce.nodeZonesLock.Unlock()
	if !gce.nodeInformerSynced() {
		return nil, fmt.Errorf("node informer is not synced when trying to GetAllCurrentZones")
	}
	zones := sets.NewString()
	for zone, nodes := range gce.nodeZones {
		if len(nodes) > 0 {
			zones.Insert(zone)
		}
	}
	return zones, nil
}

// GetAllZonesFromCloudProvider returns all the zones in which nodes are running
// Only use this in E2E tests to get zones, on real clusters this will
// get all zones with compute instances in them even if not k8s instances!!!
// ex. I have k8s nodes in us-central1-c and us-central1-b. I also have
// a non-k8s compute in us-central1-a. This func will return a,b, and c.
//
// TODO: this should be removed from the cloud provider.
func (gce *GCECloud) GetAllZonesFromCloudProvider() (sets.String, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	zones := sets.NewString()
	for _, zone := range gce.managedZones {
		instances, err := gce.c.Instances().List(ctx, zone, filter.None)
		if err != nil {
			return sets.NewString(), err
		}
		if len(instances) > 0 {
			zones.Insert(zone)
		}
	}
	return zones, nil
}

// InsertInstance creates a new instance on GCP
func (gce *GCECloud) InsertInstance(project string, zone string, i *compute.Instance) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newInstancesMetricContext("create", zone)
	return mc.Observe(gce.c.Instances().Insert(ctx, meta.ZonalKey(i.Name, zone), i))
}

// ListInstanceNames returns a string of instance names separated by spaces.
// This method should only be used for e2e testing.
// TODO: remove this method.
func (gce *GCECloud) ListInstanceNames(project, zone string) (string, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	l, err := gce.c.Instances().List(ctx, zone, filter.None)
	if err != nil {
		return "", err
	}
	var names []string
	for _, i := range l {
		names = append(names, i.Name)
	}
	return strings.Join(names, " "), nil
}

// DeleteInstance deletes an instance specified by project, zone, and name
func (gce *GCECloud) DeleteInstance(project, zone, name string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	return gce.c.Instances().Delete(ctx, meta.ZonalKey(name, zone))
}

// Implementation of Instances.CurrentNodeName
func (gce *GCECloud) CurrentNodeName(ctx context.Context, hostname string) (types.NodeName, error) {
	return types.NodeName(hostname), nil
}

// AliasRanges returns a list of CIDR ranges that are assigned to the
// `node` for allocation to pods. Returns a list of the form
// "<ip>/<netmask>".
func (gce *GCECloud) AliasRanges(nodeName types.NodeName) (cidrs []string, err error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	var instance *gceInstance
	instance, err = gce.getInstanceByName(mapNodeNameToInstanceName(nodeName))
	if err != nil {
		return
	}

	var res *computebeta.Instance
	res, err = gce.c.BetaInstances().Get(ctx, meta.ZonalKey(instance.Name, lastComponent(instance.Zone)))
	if err != nil {
		return
	}

	for _, networkInterface := range res.NetworkInterfaces {
		for _, r := range networkInterface.AliasIpRanges {
			cidrs = append(cidrs, r.IpCidrRange)
		}
	}
	return
}

// AddAliasToInstance adds an alias to the given instance from the named
// secondary range.
func (gce *GCECloud) AddAliasToInstance(nodeName types.NodeName, alias *net.IPNet) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	v1instance, err := gce.getInstanceByName(mapNodeNameToInstanceName(nodeName))
	if err != nil {
		return err
	}
	instance, err := gce.c.BetaInstances().Get(ctx, meta.ZonalKey(v1instance.Name, lastComponent(v1instance.Zone)))
	if err != nil {
		return err
	}

	switch len(instance.NetworkInterfaces) {
	case 0:
		return fmt.Errorf("instance %q has no network interfaces", nodeName)
	case 1:
	default:
		glog.Warningf("Instance %q has more than one network interface, using only the first (%v)",
			nodeName, instance.NetworkInterfaces)
	}

	iface := &computebeta.NetworkInterface{}
	iface.Name = instance.NetworkInterfaces[0].Name
	iface.Fingerprint = instance.NetworkInterfaces[0].Fingerprint
	iface.AliasIpRanges = append(iface.AliasIpRanges, &computebeta.AliasIpRange{
		IpCidrRange:         alias.String(),
		SubnetworkRangeName: gce.secondaryRangeName,
	})

	mc := newInstancesMetricContext("add_alias", v1instance.Zone)
	err = gce.c.BetaInstances().UpdateNetworkInterface(ctx, meta.ZonalKey(instance.Name, lastComponent(instance.Zone)), iface.Name, iface)
	return mc.Observe(err)
}

// Gets the named instances, returning cloudprovider.InstanceNotFound if any
// instance is not found
func (gce *GCECloud) getInstancesByNames(names []string) ([]*gceInstance, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	found := map[string]*gceInstance{}
	remaining := len(names)

	nodeInstancePrefix := gce.nodeInstancePrefix
	for _, name := range names {
		name = canonicalizeInstanceName(name)
		if !strings.HasPrefix(name, gce.nodeInstancePrefix) {
			glog.Warningf("Instance %q does not conform to prefix %q, removing filter", name, gce.nodeInstancePrefix)
			nodeInstancePrefix = ""
		}
		found[name] = nil
	}

	for _, zone := range gce.managedZones {
		if remaining == 0 {
			break
		}
		instances, err := gce.c.Instances().List(ctx, zone, filter.Regexp("name", nodeInstancePrefix+".*"))
		if err != nil {
			return nil, err
		}
		for _, inst := range instances {
			if remaining == 0 {
				break
			}
			if _, ok := found[inst.Name]; !ok {
				continue
			}
			if found[inst.Name] != nil {
				glog.Errorf("Instance name %q was duplicated (in zone %q and %q)", inst.Name, zone, found[inst.Name].Zone)
				continue
			}
			found[inst.Name] = &gceInstance{
				Zone:  zone,
				Name:  inst.Name,
				ID:    inst.Id,
				Disks: inst.Disks,
				Type:  lastComponent(inst.MachineType),
			}
			remaining--
		}
	}

	if remaining > 0 {
		var failed []string
		for k := range found {
			if found[k] == nil {
				failed = append(failed, k)
			}
		}
		glog.Errorf("Failed to retrieve instances: %v", failed)
		return nil, cloudprovider.InstanceNotFound
	}

	var ret []*gceInstance
	for _, instance := range found {
		ret = append(ret, instance)
	}

	return ret, nil
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
			glog.Errorf("getInstanceByName: failed to get instance %s in zone %s; err: %v", name, zone, err)
			return nil, err
		}
		return instance, nil
	}

	return nil, cloudprovider.InstanceNotFound
}

func (gce *GCECloud) getInstanceFromProjectInZoneByName(project, zone, name string) (*gceInstance, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	name = canonicalizeInstanceName(name)
	mc := newInstancesMetricContext("get", zone)
	res, err := gce.c.Instances().Get(ctx, meta.ZonalKey(name, zone))
	mc.Observe(err)
	if err != nil {
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
// Invoking this method to get host tags is risky since it depends on the
// format of the host names in the cluster. Only use it as a fallback if
// gce.nodeTags is unspecified
func (gce *GCECloud) computeHostTags(hosts []*gceInstance) ([]string, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	// TODO: We could store the tags in gceInstance, so we could have already fetched it
	hostNamesByZone := make(map[string]map[string]bool) // map of zones -> map of names -> bool (for easy lookup)
	nodeInstancePrefix := gce.nodeInstancePrefix
	for _, host := range hosts {
		if !strings.HasPrefix(host.Name, gce.nodeInstancePrefix) {
			glog.Warningf("instance %v does not conform to prefix '%s', ignoring filter", host, gce.nodeInstancePrefix)
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

	filt := filter.None
	if nodeInstancePrefix != "" {
		filt = filter.Regexp("name", nodeInstancePrefix+".*")
	}
	for zone, hostNames := range hostNamesByZone {
		instances, err := gce.c.Instances().List(ctx, zone, filt)
		if err != nil {
			return nil, err
		}
		for _, instance := range instances {
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
				return nil, fmt.Errorf("could not find any tag that is a prefix of instance name for instance %s", instance.Name)
			}
		}
	}
	if len(tags) == 0 {
		return nil, fmt.Errorf("no instances found")
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
