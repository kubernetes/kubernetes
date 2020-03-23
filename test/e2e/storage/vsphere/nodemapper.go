/*
Copyright 2018 The Kubernetes Authors.

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

package vsphere

import (
	"context"
	"errors"
	"strings"
	"sync"

	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vapi/rest"
	"github.com/vmware/govmomi/vapi/tags"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/test/e2e/framework"

	neturl "net/url"
)

// NodeMapper contains information to generate nameToNodeInfo and vcToZoneDatastore maps
type NodeMapper struct {
}

// NodeInfo contains information about vcenter nodes
type NodeInfo struct {
	Name              string
	DataCenterRef     types.ManagedObjectReference
	VirtualMachineRef types.ManagedObjectReference
	HostSystemRef     types.ManagedObjectReference
	VSphere           *VSphere
	Zones             []string
}

const (
	datacenterType             = "Datacenter"
	clusterComputeResourceType = "ClusterComputeResource"
	hostSystemType             = "HostSystem"
)

var (
	nameToNodeInfo        = make(map[string]*NodeInfo)
	vcToZoneDatastoresMap = make(map[string](map[string][]string))
)

// GenerateNodeMap populates node name to node info map
func (nm *NodeMapper) GenerateNodeMap(vSphereInstances map[string]*VSphere, nodeList v1.NodeList) error {
	type VMSearch struct {
		vs         *VSphere
		datacenter *object.Datacenter
	}

	var wg sync.WaitGroup
	var queueChannel []*VMSearch

	var datacenters []*object.Datacenter
	var err error
	for _, vs := range vSphereInstances {

		// Create context
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		if vs.Config.Datacenters == "" {
			datacenters, err = vs.GetAllDatacenter(ctx)
			if err != nil {
				framework.Logf("NodeMapper error: %v", err)
				continue
			}
		} else {
			dcName := strings.Split(vs.Config.Datacenters, ",")
			for _, dc := range dcName {
				dc = strings.TrimSpace(dc)
				if dc == "" {
					continue
				}
				datacenter, err := vs.GetDatacenter(ctx, dc)
				if err != nil {
					framework.Logf("NodeMapper error dc: %s \n err: %v", dc, err)

					continue
				}
				datacenters = append(datacenters, datacenter)
			}
		}

		for _, dc := range datacenters {
			framework.Logf("Search candidates vc=%s and datacenter=%s", vs.Config.Hostname, dc.Name())
			queueChannel = append(queueChannel, &VMSearch{vs: vs, datacenter: dc})
		}
	}

	for _, node := range nodeList.Items {
		n := node
		go func() {
			nodeUUID := getUUIDFromProviderID(n.Spec.ProviderID)
			framework.Logf("Searching for node with UUID: %s", nodeUUID)
			for _, res := range queueChannel {
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				vm, err := res.vs.GetVMByUUID(ctx, nodeUUID, res.datacenter)
				if err != nil {
					framework.Logf("Error %v while looking for node=%s in vc=%s and datacenter=%s",
						err, n.Name, res.vs.Config.Hostname, res.datacenter.Name())
					continue
				}
				if vm != nil {
					hostSystemRef := res.vs.GetHostFromVMReference(ctx, vm.Reference())
					zones := retrieveZoneInformationForNode(n.Name, res.vs, hostSystemRef)
					framework.Logf("Found node %s as vm=%+v placed on host=%+v under zones %s in vc=%s and datacenter=%s",
						n.Name, vm, hostSystemRef, zones, res.vs.Config.Hostname, res.datacenter.Name())
					nodeInfo := &NodeInfo{Name: n.Name, DataCenterRef: res.datacenter.Reference(), VirtualMachineRef: vm.Reference(), HostSystemRef: hostSystemRef, VSphere: res.vs, Zones: zones}
					nm.SetNodeInfo(n.Name, nodeInfo)
					break
				}
			}
			wg.Done()
		}()
		wg.Add(1)
	}
	wg.Wait()

	if len(nameToNodeInfo) != len(nodeList.Items) {
		return errors.New("all nodes not mapped to respective vSphere")
	}
	return nil
}

// Establish rest connection to retrieve tag manager stub
func withTagsClient(ctx context.Context, connection *VSphere, f func(c *rest.Client) error) error {
	c := rest.NewClient(connection.Client.Client)
	user := neturl.UserPassword(connection.Config.Username, connection.Config.Password)
	if err := c.Login(ctx, user); err != nil {
		return err
	}
	defer c.Logout(ctx)
	return f(c)
}

// Iterates over each node and retrieves the zones in which they are placed
func retrieveZoneInformationForNode(nodeName string, connection *VSphere, hostSystemRef types.ManagedObjectReference) []string {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var zones []string
	pc := connection.Client.ServiceContent.PropertyCollector
	withTagsClient(ctx, connection, func(c *rest.Client) error {
		client := tags.NewManager(c)
		// Example result: ["Host", "Cluster", "Datacenter"]
		ancestors, err := mo.Ancestors(ctx, connection.Client, pc, hostSystemRef)
		if err != nil {
			return err
		}

		var validAncestors []mo.ManagedEntity
		// Filter out only Datacenter, ClusterComputeResource and HostSystem type objects. These objects will be
		// in the following order ["Datacenter" < "ClusterComputeResource" < "HostSystem"] so that the highest
		// zone precedence will be received by the HostSystem type.
		for _, ancestor := range ancestors {
			moType := ancestor.ExtensibleManagedObject.Self.Type
			if moType == datacenterType || moType == clusterComputeResourceType || moType == hostSystemType {
				validAncestors = append(validAncestors, ancestor)
			}
		}

		for _, ancestor := range validAncestors {
			var zonesAttachedToObject []string
			tags, err := client.ListAttachedTags(ctx, ancestor)
			if err != nil {
				return err
			}
			for _, value := range tags {
				tag, err := client.GetTag(ctx, value)
				if err != nil {
					return err
				}
				category, err := client.GetCategory(ctx, tag.CategoryID)
				if err != nil {
					return err
				}
				switch {
				case category.Name == "k8s-zone":
					framework.Logf("Found %s associated with %s for %s", tag.Name, ancestor.Name, nodeName)
					zonesAttachedToObject = append(zonesAttachedToObject, tag.Name)
				case category.Name == "k8s-region":
					framework.Logf("Found %s associated with %s for %s", tag.Name, ancestor.Name, nodeName)
				}
			}
			// Overwrite zone information if it exists for this object
			if len(zonesAttachedToObject) != 0 {
				zones = zonesAttachedToObject
			}
		}
		return nil
	})
	return zones
}

// GenerateZoneToDatastoreMap generates a mapping of zone to datastore for easily verifying volume placement
func (nm *NodeMapper) GenerateZoneToDatastoreMap() error {
	// 1. Create zone to hosts map for each VC
	var vcToZoneHostsMap = make(map[string](map[string][]string))
	// 2. Create host to datastores map for each VC
	var vcToHostDatastoresMap = make(map[string](map[string][]string))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	// 3. Populate vcToZoneHostsMap and vcToHostDatastoresMap
	for _, nodeInfo := range nameToNodeInfo {
		vc := nodeInfo.VSphere.Config.Hostname
		host := nodeInfo.HostSystemRef.Value
		for _, zone := range nodeInfo.Zones {
			if vcToZoneHostsMap[vc] == nil {
				vcToZoneHostsMap[vc] = make(map[string][]string)
			}
			// Populating vcToZoneHostsMap using the HostSystemRef and Zone fields from each NodeInfo
			hosts := vcToZoneHostsMap[vc][zone]
			hosts = append(hosts, host)
			vcToZoneHostsMap[vc][zone] = hosts
		}
		if vcToHostDatastoresMap[vc] == nil {
			vcToHostDatastoresMap[vc] = make(map[string][]string)
		}
		datastores := vcToHostDatastoresMap[vc][host]
		// Populating vcToHostDatastoresMap by finding out the datastores mounted on node's host
		datastoreRefs := nodeInfo.VSphere.GetDatastoresMountedOnHost(ctx, nodeInfo.HostSystemRef)
		for _, datastore := range datastoreRefs {
			datastores = append(datastores, datastore.Value)
		}
		vcToHostDatastoresMap[vc][host] = datastores
	}
	// 4, Populate vcToZoneDatastoresMap from vcToZoneHostsMap and vcToHostDatastoresMap
	for vc, zoneToHostsMap := range vcToZoneHostsMap {
		for zone, hosts := range zoneToHostsMap {
			commonDatastores := retrieveCommonDatastoresAmongHosts(hosts, vcToHostDatastoresMap[vc])
			if vcToZoneDatastoresMap[vc] == nil {
				vcToZoneDatastoresMap[vc] = make(map[string][]string)
			}
			vcToZoneDatastoresMap[vc][zone] = commonDatastores
		}
	}
	framework.Logf("Zone to datastores map : %+v", vcToZoneDatastoresMap)
	return nil
}

// retrieveCommonDatastoresAmongHosts retrieves the common datastores from the specified hosts
func retrieveCommonDatastoresAmongHosts(hosts []string, hostToDatastoresMap map[string][]string) []string {
	var datastoreCountMap = make(map[string]int)
	for _, host := range hosts {
		for _, datastore := range hostToDatastoresMap[host] {
			datastoreCountMap[datastore] = datastoreCountMap[datastore] + 1
		}
	}
	var commonDatastores []string
	numHosts := len(hosts)
	for datastore, count := range datastoreCountMap {
		if count == numHosts {
			commonDatastores = append(commonDatastores, datastore)
		}
	}
	return commonDatastores
}

// GetDatastoresInZone returns all the datastores in the specified zone
func (nm *NodeMapper) GetDatastoresInZone(vc string, zone string) []string {
	return vcToZoneDatastoresMap[vc][zone]
}

// GetNodeInfo returns NodeInfo for given nodeName
func (nm *NodeMapper) GetNodeInfo(nodeName string) *NodeInfo {
	return nameToNodeInfo[nodeName]
}

// SetNodeInfo sets NodeInfo for given nodeName. This function is not thread safe. Users need to handle concurrency.
func (nm *NodeMapper) SetNodeInfo(nodeName string, nodeInfo *NodeInfo) {
	nameToNodeInfo[nodeName] = nodeInfo
}
