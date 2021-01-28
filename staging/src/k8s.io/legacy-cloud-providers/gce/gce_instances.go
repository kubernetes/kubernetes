// +build !providerless

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
	computebeta "google.golang.org/api/compute/v0.beta"
	compute "google.golang.org/api/compute/v1"
	"k8s.io/klog/v2"

	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud"
	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/filter"
	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/meta"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	cloudprovider "k8s.io/cloud-provider"
)

const (
	defaultZone                   = ""
	networkInterfaceIP            = "instance/network-interfaces/%s/ip"
	networkInterfaceAccessConfigs = "instance/network-interfaces/%s/access-configs"
	networkInterfaceExternalIP    = "instance/network-interfaces/%s/access-configs/%s/external-ip"
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
	zone, ok := n.Labels[v1.LabelFailureDomainBetaZone]
	if !ok {
		return defaultZone
	}
	return zone
}

func makeHostURL(projectsAPIEndpoint, projectID, zone, host string) string {
	host = canonicalizeInstanceName(host)
	return projectsAPIEndpoint + strings.Join([]string{projectID, "zones", zone, "instances", host}, "/")
}

// ToInstanceReferences returns instance references by links
func (g *Cloud) ToInstanceReferences(zone string, instanceNames []string) (refs []*compute.InstanceReference) {
	for _, ins := range instanceNames {
		instanceLink := makeHostURL(g.service.BasePath, g.projectID, zone, ins)
		refs = append(refs, &compute.InstanceReference{Instance: instanceLink})
	}
	return refs
}

// NodeAddresses is an implementation of Instances.NodeAddresses.
func (g *Cloud) NodeAddresses(ctx context.Context, nodeName types.NodeName) ([]v1.NodeAddress, error) {
	timeoutCtx, cancel := context.WithTimeout(ctx, 1*time.Hour)
	defer cancel()

	instanceName := string(nodeName)

	if g.useMetadataServer {
		// Use metadata server if possible
		if g.isCurrentInstance(instanceName) {

			nics, err := metadata.Get("instance/network-interfaces/")
			if err != nil {
				return nil, fmt.Errorf("couldn't get network interfaces: %v", err)
			}

			nicsArr := strings.Split(nics, "/\n")
			nodeAddresses := []v1.NodeAddress{}

			for _, nic := range nicsArr {

				if nic == "" {
					continue
				}

				internalIP, err := metadata.Get(fmt.Sprintf(networkInterfaceIP, nic))
				if err != nil {
					return nil, fmt.Errorf("couldn't get internal IP: %v", err)
				}
				nodeAddresses = append(nodeAddresses, v1.NodeAddress{Type: v1.NodeInternalIP, Address: internalIP})

				acs, err := metadata.Get(fmt.Sprintf(networkInterfaceAccessConfigs, nic))
				if err != nil {
					return nil, fmt.Errorf("couldn't get access configs: %v", err)
				}

				acsArr := strings.Split(acs, "/\n")

				for _, ac := range acsArr {

					if ac == "" {
						continue
					}

					externalIP, err := metadata.Get(fmt.Sprintf(networkInterfaceExternalIP, nic, ac))
					if err != nil {
						return nil, fmt.Errorf("couldn't get external IP: %v", err)
					}

					if externalIP != "" {
						nodeAddresses = append(nodeAddresses, v1.NodeAddress{Type: v1.NodeExternalIP, Address: externalIP})
					}
				}
			}

			internalDNSFull, err := metadata.Get("instance/hostname")
			if err != nil {
				klog.Warningf("couldn't get full internal DNS name: %v", err)
			} else {
				nodeAddresses = append(nodeAddresses,
					v1.NodeAddress{Type: v1.NodeInternalDNS, Address: internalDNSFull},
					v1.NodeAddress{Type: v1.NodeHostName, Address: internalDNSFull},
				)
			}
			return nodeAddresses, nil
		}
	}

	// Use GCE API
	instanceObj, err := g.getInstanceByName(instanceName)
	if err != nil {
		return nil, fmt.Errorf("couldn't get instance details: %v", err)
	}

	instance, err := g.c.Instances().Get(timeoutCtx, meta.ZonalKey(canonicalizeInstanceName(instanceObj.Name), instanceObj.Zone))
	if err != nil {
		return []v1.NodeAddress{}, fmt.Errorf("error while querying for instance: %v", err)
	}

	return nodeAddressesFromInstance(instance)
}

// NodeAddressesByProviderID will not be called from the node that is requesting this ID.
// i.e. metadata service and other local methods cannot be used here
func (g *Cloud) NodeAddressesByProviderID(ctx context.Context, providerID string) ([]v1.NodeAddress, error) {
	timeoutCtx, cancel := context.WithTimeout(ctx, 1*time.Hour)
	defer cancel()

	_, zone, name, err := splitProviderID(providerID)
	if err != nil {
		return []v1.NodeAddress{}, err
	}

	instance, err := g.c.Instances().Get(timeoutCtx, meta.ZonalKey(canonicalizeInstanceName(name), zone))
	if err != nil {
		return []v1.NodeAddress{}, fmt.Errorf("error while querying for providerID %q: %v", providerID, err)
	}

	return nodeAddressesFromInstance(instance)
}

// instanceByProviderID returns the cloudprovider instance of the node
// with the specified unique providerID
func (g *Cloud) instanceByProviderID(providerID string) (*gceInstance, error) {
	project, zone, name, err := splitProviderID(providerID)
	if err != nil {
		return nil, err
	}

	instance, err := g.getInstanceFromProjectInZoneByName(project, zone, name)
	if err != nil {
		if isHTTPErrorCode(err, http.StatusNotFound) {
			return nil, cloudprovider.InstanceNotFound
		}
		return nil, err
	}

	return instance, nil
}

// InstanceShutdownByProviderID returns true if the instance is in safe state to detach volumes
func (g *Cloud) InstanceShutdownByProviderID(ctx context.Context, providerID string) (bool, error) {
	return false, cloudprovider.NotImplemented
}

// InstanceShutdown returns true if the instance is in safe state to detach volumes
func (g *Cloud) InstanceShutdown(ctx context.Context, node *v1.Node) (bool, error) {
	return false, cloudprovider.NotImplemented
}

func nodeAddressesFromInstance(instance *compute.Instance) ([]v1.NodeAddress, error) {
	if len(instance.NetworkInterfaces) < 1 {
		return nil, fmt.Errorf("could not find network interfaces for instanceID %q", instance.Id)
	}
	nodeAddresses := []v1.NodeAddress{}

	for _, nic := range instance.NetworkInterfaces {
		nodeAddresses = append(nodeAddresses, v1.NodeAddress{Type: v1.NodeInternalIP, Address: nic.NetworkIP})
		for _, config := range nic.AccessConfigs {
			nodeAddresses = append(nodeAddresses, v1.NodeAddress{Type: v1.NodeExternalIP, Address: config.NatIP})
		}
	}

	return nodeAddresses, nil
}

// InstanceTypeByProviderID returns the cloudprovider instance type of the node
// with the specified unique providerID This method will not be called from the
// node that is requesting this ID. i.e. metadata service and other local
// methods cannot be used here
func (g *Cloud) InstanceTypeByProviderID(ctx context.Context, providerID string) (string, error) {
	instance, err := g.instanceByProviderID(providerID)
	if err != nil {
		return "", err
	}

	return instance.Type, nil
}

// InstanceExistsByProviderID returns true if the instance with the given provider id still exists and is running.
// If false is returned with no error, the instance will be immediately deleted by the cloud controller manager.
func (g *Cloud) InstanceExistsByProviderID(ctx context.Context, providerID string) (bool, error) {
	_, err := g.instanceByProviderID(providerID)
	if err != nil {
		if err == cloudprovider.InstanceNotFound {
			return false, nil
		}
		return false, err
	}

	return true, nil
}

// InstanceExists returns true if the instance with the given provider id still exists and is running.
// If false is returned with no error, the instance will be immediately deleted by the cloud controller manager.
func (g *Cloud) InstanceExists(ctx context.Context, node *v1.Node) (bool, error) {
	providerID := node.Spec.ProviderID
	if providerID == "" {
		var err error
		if providerID, err = cloudprovider.GetInstanceProviderID(ctx, g, types.NodeName(node.Name)); err != nil {
			if err == cloudprovider.InstanceNotFound {
				return false, nil
			}
			return false, err
		}
	}
	return g.InstanceExistsByProviderID(ctx, providerID)
}

// InstanceMetadata returns metadata of the specified instance.
func (g *Cloud) InstanceMetadata(ctx context.Context, node *v1.Node) (*cloudprovider.InstanceMetadata, error) {
	timeoutCtx, cancel := context.WithTimeout(ctx, 1*time.Hour)
	defer cancel()

	providerID := node.Spec.ProviderID
	if providerID == "" {
		var err error
		if providerID, err = cloudprovider.GetInstanceProviderID(ctx, g, types.NodeName(node.Name)); err != nil {
			return nil, err
		}
	}

	_, zone, name, err := splitProviderID(providerID)
	if err != nil {
		return nil, err
	}

	region, err := GetGCERegion(zone)
	if err != nil {
		return nil, err
	}

	instance, err := g.c.Instances().Get(timeoutCtx, meta.ZonalKey(canonicalizeInstanceName(name), zone))
	if err != nil {
		return nil, fmt.Errorf("error while querying for providerID %q: %v", providerID, err)
	}

	addresses, err := nodeAddressesFromInstance(instance)
	if err != nil {
		return nil, err
	}

	return &cloudprovider.InstanceMetadata{
		ProviderID:    providerID,
		InstanceType:  lastComponent(instance.MachineType),
		NodeAddresses: addresses,
		Zone:          zone,
		Region:        region,
	}, nil
}

// InstanceID returns the cloud provider ID of the node with the specified NodeName.
func (g *Cloud) InstanceID(ctx context.Context, nodeName types.NodeName) (string, error) {
	instanceName := mapNodeNameToInstanceName(nodeName)
	if g.useMetadataServer {
		// Use metadata, if possible, to fetch ID. See issue #12000
		if g.isCurrentInstance(instanceName) {
			projectID, zone, err := getProjectAndZone()
			if err == nil {
				return projectID + "/" + zone + "/" + canonicalizeInstanceName(instanceName), nil
			}
		}
	}
	instance, err := g.getInstanceByName(instanceName)
	if err != nil {
		return "", err
	}
	return g.projectID + "/" + instance.Zone + "/" + instance.Name, nil
}

// InstanceType returns the type of the specified node with the specified NodeName.
func (g *Cloud) InstanceType(ctx context.Context, nodeName types.NodeName) (string, error) {
	instanceName := mapNodeNameToInstanceName(nodeName)
	if g.useMetadataServer {
		// Use metadata, if possible, to fetch ID. See issue #12000
		if g.isCurrentInstance(instanceName) {
			mType, err := getCurrentMachineTypeViaMetadata()
			if err == nil {
				return mType, nil
			}
		}
	}
	instance, err := g.getInstanceByName(instanceName)
	if err != nil {
		return "", err
	}
	return instance.Type, nil
}

// AddSSHKeyToAllInstances adds an SSH public key as a legal identity for all instances
// expected format for the key is standard ssh-keygen format: <protocol> <blob>
func (g *Cloud) AddSSHKeyToAllInstances(ctx context.Context, user string, keyData []byte) error {
	ctx, cancel := context.WithTimeout(ctx, 1*time.Hour)
	defer cancel()

	return wait.Poll(2*time.Second, 30*time.Second, func() (bool, error) {
		project, err := g.c.Projects().Get(ctx, g.projectID)
		if err != nil {
			klog.Errorf("Could not get project: %v", err)
			return false, nil
		}
		keyString := fmt.Sprintf("%s:%s %s@%s", user, strings.TrimSpace(string(keyData)), user, user)
		found := false
		for _, item := range project.CommonInstanceMetadata.Items {
			if item.Key == "sshKeys" {
				if strings.Contains(*item.Value, keyString) {
					// We've already added the key
					klog.Info("SSHKey already in project metadata")
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
			klog.Infof("Failed to find sshKeys metadata, creating a new item")
			project.CommonInstanceMetadata.Items = append(project.CommonInstanceMetadata.Items,
				&compute.MetadataItems{
					Key:   "sshKeys",
					Value: &keyString,
				})
		}

		mc := newInstancesMetricContext("add_ssh_key", "")
		err = g.c.Projects().SetCommonInstanceMetadata(ctx, g.projectID, project.CommonInstanceMetadata)
		mc.Observe(err)

		if err != nil {
			klog.Errorf("Could not Set Metadata: %v", err)
			return false, nil
		}
		klog.Infof("Successfully added sshKey to project metadata")
		return true, nil
	})
}

// GetAllCurrentZones returns all the zones in which k8s nodes are currently running
func (g *Cloud) GetAllCurrentZones() (sets.String, error) {
	if g.nodeInformerSynced == nil {
		klog.Warning("Cloud object does not have informers set, should only happen in E2E binary.")
		return g.GetAllZonesFromCloudProvider()
	}
	g.nodeZonesLock.Lock()
	defer g.nodeZonesLock.Unlock()
	if !g.nodeInformerSynced() {
		return nil, fmt.Errorf("node informer is not synced when trying to GetAllCurrentZones")
	}
	zones := sets.NewString()
	for zone, nodes := range g.nodeZones {
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
func (g *Cloud) GetAllZonesFromCloudProvider() (sets.String, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	zones := sets.NewString()
	for _, zone := range g.managedZones {
		instances, err := g.c.Instances().List(ctx, zone, filter.None)
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
func (g *Cloud) InsertInstance(project string, zone string, i *compute.Instance) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	mc := newInstancesMetricContext("create", zone)
	return mc.Observe(g.c.Instances().Insert(ctx, meta.ZonalKey(i.Name, zone), i))
}

// ListInstanceNames returns a string of instance names separated by spaces.
// This method should only be used for e2e testing.
// TODO: remove this method.
func (g *Cloud) ListInstanceNames(project, zone string) (string, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	l, err := g.c.Instances().List(ctx, zone, filter.None)
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
func (g *Cloud) DeleteInstance(project, zone, name string) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	return g.c.Instances().Delete(ctx, meta.ZonalKey(name, zone))
}

// CurrentNodeName returns the name of the node we are currently running on
// On most clouds (e.g. GCE) this is the hostname, so we provide the hostname
func (g *Cloud) CurrentNodeName(ctx context.Context, hostname string) (types.NodeName, error) {
	return types.NodeName(hostname), nil
}

// AliasRangesByProviderID returns a list of CIDR ranges that are assigned to the
// `node` for allocation to pods. Returns a list of the form
// "<ip>/<netmask>".
func (g *Cloud) AliasRangesByProviderID(providerID string) (cidrs []string, err error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	_, zone, name, err := splitProviderID(providerID)
	if err != nil {
		return nil, err
	}

	var res *computebeta.Instance
	res, err = g.c.BetaInstances().Get(ctx, meta.ZonalKey(canonicalizeInstanceName(name), zone))
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

// AddAliasToInstanceByProviderID adds an alias to the given instance from the named
// secondary range.
func (g *Cloud) AddAliasToInstanceByProviderID(providerID string, alias *net.IPNet) error {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	_, zone, name, err := splitProviderID(providerID)
	if err != nil {
		return err
	}

	instance, err := g.c.BetaInstances().Get(ctx, meta.ZonalKey(canonicalizeInstanceName(name), zone))
	if err != nil {
		return err
	}

	switch len(instance.NetworkInterfaces) {
	case 0:
		return fmt.Errorf("instance %q has no network interfaces", providerID)
	case 1:
	default:
		klog.Warningf("Instance %q has more than one network interface, using only the first (%v)",
			providerID, instance.NetworkInterfaces)
	}

	iface := &computebeta.NetworkInterface{}
	iface.Name = instance.NetworkInterfaces[0].Name
	iface.Fingerprint = instance.NetworkInterfaces[0].Fingerprint
	iface.AliasIpRanges = append(iface.AliasIpRanges, &computebeta.AliasIpRange{
		IpCidrRange:         alias.String(),
		SubnetworkRangeName: g.secondaryRangeName,
	})

	mc := newInstancesMetricContext("add_alias", zone)
	err = g.c.BetaInstances().UpdateNetworkInterface(ctx, meta.ZonalKey(instance.Name, lastComponent(instance.Zone)), iface.Name, iface)
	return mc.Observe(err)
}

// Gets the named instances, returning cloudprovider.InstanceNotFound if any
// instance is not found
func (g *Cloud) getInstancesByNames(names []string) ([]*gceInstance, error) {
	foundInstances, err := g.getFoundInstanceByNames(names)
	if err != nil {
		return nil, err
	}
	if len(foundInstances) != len(names) {
		if len(foundInstances) == 0 {
			// return error so the TargetPool nodecount does not drop to 0 unexpectedly.
			return nil, cloudprovider.InstanceNotFound
		}
		klog.Warningf("getFoundInstanceByNames - input instances %d, found %d. Continuing LoadBalancer Update", len(names), len(foundInstances))
	}
	return foundInstances, nil
}

// Gets the named instances, returning a list of gceInstances it was able to find from the provided
// list of names.
func (g *Cloud) getFoundInstanceByNames(names []string) ([]*gceInstance, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	found := map[string]*gceInstance{}
	remaining := len(names)

	nodeInstancePrefix := g.nodeInstancePrefix
	for _, name := range names {
		name = canonicalizeInstanceName(name)
		if !strings.HasPrefix(name, g.nodeInstancePrefix) {
			klog.Warningf("Instance %q does not conform to prefix %q, removing filter", name, g.nodeInstancePrefix)
			nodeInstancePrefix = ""
		}
		found[name] = nil
	}

	for _, zone := range g.managedZones {
		if remaining == 0 {
			break
		}
		instances, err := g.c.Instances().List(ctx, zone, filter.Regexp("name", nodeInstancePrefix+".*"))
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
				klog.Errorf("Instance name %q was duplicated (in zone %q and %q)", inst.Name, zone, found[inst.Name].Zone)
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

	var ret []*gceInstance
	var failed []string
	for name, instance := range found {
		if instance != nil {
			ret = append(ret, instance)
		} else {
			failed = append(failed, name)
		}
	}
	if len(failed) > 0 {
		klog.Errorf("Failed to retrieve instances: %v", failed)
	}

	return ret, nil
}

// Gets the named instance, returning cloudprovider.InstanceNotFound if the instance is not found
func (g *Cloud) getInstanceByName(name string) (*gceInstance, error) {
	// Avoid changing behaviour when not managing multiple zones
	for _, zone := range g.managedZones {
		instance, err := g.getInstanceFromProjectInZoneByName(g.projectID, zone, name)
		if err != nil {
			if isHTTPErrorCode(err, http.StatusNotFound) {
				continue
			}
			klog.Errorf("getInstanceByName: failed to get instance %s in zone %s; err: %v", name, zone, err)
			return nil, err
		}
		return instance, nil
	}

	return nil, cloudprovider.InstanceNotFound
}

func (g *Cloud) getInstanceFromProjectInZoneByName(project, zone, name string) (*gceInstance, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	name = canonicalizeInstanceName(name)
	mc := newInstancesMetricContext("get", zone)
	res, err := g.c.Instances().Get(ctx, meta.ZonalKey(name, zone))
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
func (g *Cloud) isCurrentInstance(instanceID string) bool {
	currentInstanceID, err := getInstanceIDViaMetadata()
	if err != nil {
		// Log and swallow error
		klog.Errorf("Failed to fetch instanceID via Metadata: %v", err)
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
func (g *Cloud) computeHostTags(hosts []*gceInstance) ([]string, error) {
	ctx, cancel := cloud.ContextWithCallTimeout()
	defer cancel()

	// TODO: We could store the tags in gceInstance, so we could have already fetched it
	hostNamesByZone := make(map[string]map[string]bool) // map of zones -> map of names -> bool (for easy lookup)
	nodeInstancePrefix := g.nodeInstancePrefix
	for _, host := range hosts {
		if !strings.HasPrefix(host.Name, g.nodeInstancePrefix) {
			klog.Warningf("instance %v does not conform to prefix '%s', ignoring filter", host, g.nodeInstancePrefix)
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
		instances, err := g.c.Instances().List(ctx, zone, filt)
		if err != nil {
			return nil, err
		}
		for _, instance := range instances {
			if !hostNames[instance.Name] {
				continue
			}
			longestTag := ""
			for _, tag := range instance.Tags.Items {
				if strings.HasPrefix(instance.Name, tag) && len(tag) > len(longestTag) {
					longestTag = tag
				}
			}
			if len(longestTag) > 0 {
				tags.Insert(longestTag)
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
func (g *Cloud) GetNodeTags(nodeNames []string) ([]string, error) {
	// If nodeTags were specified through configuration, use them
	if len(g.nodeTags) > 0 {
		return g.nodeTags, nil
	}

	g.computeNodeTagLock.Lock()
	defer g.computeNodeTagLock.Unlock()

	// Early return if hosts have not changed
	hosts := sets.NewString(nodeNames...)
	if hosts.Equal(g.lastKnownNodeNames) {
		return g.lastComputedNodeTags, nil
	}

	// Get GCE instance data by hostname
	instances, err := g.getInstancesByNames(nodeNames)
	if err != nil {
		return nil, err
	}

	// Determine list of host tags
	tags, err := g.computeHostTags(instances)
	if err != nil {
		return nil, err
	}

	// Save the list of tags
	g.lastKnownNodeNames = hosts
	g.lastComputedNodeTags = tags
	return tags, nil
}
