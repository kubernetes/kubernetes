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

package azure

import (
	"context"
	"strings"
	"sync"
	"time"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-03-01/compute"
	"github.com/Azure/go-autorest/autorest/to"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog"
)

var (
	vmssNameSeparator  = "_"
	vmssCacheSeparator = "#"

	vmssKey                 = "k8svmssKey"
	vmssVirtualMachinesKey  = "k8svmssVirtualMachinesKey"
	availabilitySetNodesKey = "k8sAvailabilitySetNodesKey"

	availabilitySetNodesCacheTTL = 15 * time.Minute
	vmssTTL                      = 10 * time.Minute
	vmssVirtualMachinesTTL       = 10 * time.Minute
)

type vmssVirtualMachinesEntry struct {
	resourceGroup  string
	vmssName       string
	instanceID     string
	virtualMachine *compute.VirtualMachineScaleSetVM
	lastUpdate     time.Time
}

type vmssEntry struct {
	vmss       *compute.VirtualMachineScaleSet
	lastUpdate time.Time
}

func (ss *scaleSet) newVMSSCache() (*timedCache, error) {
	getter := func(key string) (interface{}, error) {
		localCache := &sync.Map{} // [vmssName]*vmssEntry

		allResourceGroups, err := ss.GetResourceGroups()
		if err != nil {
			return nil, err
		}

		for _, resourceGroup := range allResourceGroups.List() {
			allScaleSets, err := ss.VirtualMachineScaleSetsClient.List(context.Background(), resourceGroup)
			if err != nil {
				klog.Errorf("VirtualMachineScaleSetsClient.List failed: %v", err)
				return nil, err
			}

			for i := range allScaleSets {
				scaleSet := allScaleSets[i]
				if scaleSet.Name == nil || *scaleSet.Name == "" {
					klog.Warning("failed to get the name of VMSS")
					continue
				}
				localCache.Store(*scaleSet.Name, &vmssEntry{
					vmss:       &scaleSet,
					lastUpdate: time.Now().UTC(),
				})
			}
		}

		return localCache, nil
	}

	return newTimedcache(vmssTTL, getter)
}

func extractVmssVMName(name string) (string, string, error) {
	split := strings.SplitAfter(name, vmssNameSeparator)
	if len(split) < 2 {
		klog.V(3).Infof("Failed to extract vmssVMName %q", name)
		return "", "", ErrorNotVmssInstance
	}

	ssName := strings.Join(split[0:len(split)-1], "")
	// removing the trailing `vmssNameSeparator` since we used SplitAfter
	ssName = ssName[:len(ssName)-1]
	instanceID := split[len(split)-1]
	return ssName, instanceID, nil
}

func (ss *scaleSet) newVMSSVirtualMachinesCache() (*timedCache, error) {
	getter := func(key string) (interface{}, error) {
		localCache := &sync.Map{} // [nodeName]*vmssVirtualMachinesEntry

		oldCache := make(map[string]vmssVirtualMachinesEntry)

		if ss.vmssVMCache != nil {
			// get old cache before refreshing the cache
			entry, exists, err := ss.vmssVMCache.store.GetByKey(vmssVirtualMachinesKey)
			if err != nil {
				return nil, err
			}
			if exists {
				cached := entry.(*cacheEntry).data
				if cached != nil {
					virtualMachines := cached.(*sync.Map)
					virtualMachines.Range(func(key, value interface{}) bool {
						oldCache[key.(string)] = *value.(*vmssVirtualMachinesEntry)
						return true
					})
				}
			}
		}

		allResourceGroups, err := ss.GetResourceGroups()
		if err != nil {
			return nil, err
		}

		for _, resourceGroup := range allResourceGroups.List() {
			scaleSetNames, err := ss.listScaleSets(resourceGroup)
			if err != nil {
				return nil, err
			}

			for _, ssName := range scaleSetNames {
				vms, err := ss.listScaleSetVMs(ssName, resourceGroup)
				if err != nil {
					return nil, err
				}

				for i := range vms {
					vm := vms[i]
					if vm.OsProfile == nil || vm.OsProfile.ComputerName == nil {
						klog.Warningf("failed to get computerName for vmssVM (%q)", ssName)
						continue
					}

					computerName := strings.ToLower(*vm.OsProfile.ComputerName)
					localCache.Store(computerName, &vmssVirtualMachinesEntry{
						resourceGroup:  resourceGroup,
						vmssName:       ssName,
						instanceID:     to.String(vm.InstanceID),
						virtualMachine: &vm,
						lastUpdate:     time.Now().UTC(),
					})

					if _, exists := oldCache[computerName]; exists {
						delete(oldCache, computerName)
					}
				}
			}

			// add old missing cache data with nil entries to prevent aggressive
			// ARM calls during cache invalidation
			for name, vmEntry := range oldCache {
				// if the nil cache entry has existed for 15 minutes in the cache
				// then it should not be added back to the cache
				if vmEntry.virtualMachine == nil || time.Since(vmEntry.lastUpdate) > 15*time.Minute {
					klog.V(5).Infof("ignoring expired entries from old cache for %s", name)
					continue
				}
				lastUpdate := time.Now().UTC()
				if vmEntry.virtualMachine == nil {
					// if this is already a nil entry then keep the time the nil
					// entry was first created, so we can cleanup unwanted entries
					lastUpdate = vmEntry.lastUpdate
				}

				klog.V(5).Infof("adding old entries to new cache for %s", name)
				localCache.Store(name, &vmssVirtualMachinesEntry{
					resourceGroup:  vmEntry.resourceGroup,
					vmssName:       vmEntry.vmssName,
					instanceID:     vmEntry.instanceID,
					virtualMachine: nil,
					lastUpdate:     lastUpdate,
				})
			}
		}

		return localCache, nil
	}

	return newTimedcache(vmssVirtualMachinesTTL, getter)
}

func (ss *scaleSet) deleteCacheForNode(nodeName string) error {
	cached, err := ss.vmssVMCache.Get(vmssVirtualMachinesKey, cacheReadTypeUnsafe)
	if err != nil {
		klog.Errorf("deleteCacheForNode(%s) failed with error: %v", nodeName, err)
		return err
	}

	virtualMachines := cached.(*sync.Map)
	virtualMachines.Delete(nodeName)
	return nil
}

func (ss *scaleSet) newAvailabilitySetNodesCache() (*timedCache, error) {
	getter := func(key string) (interface{}, error) {
		localCache := sets.NewString()
		resourceGroups, err := ss.GetResourceGroups()
		if err != nil {
			return nil, err
		}

		for _, resourceGroup := range resourceGroups.List() {
			vmList, err := ss.Cloud.ListVirtualMachines(resourceGroup)
			if err != nil {
				return nil, err
			}

			for _, vm := range vmList {
				if vm.Name != nil {
					localCache.Insert(*vm.Name)
				}
			}
		}

		return localCache, nil
	}

	return newTimedcache(availabilitySetNodesCacheTTL, getter)
}

func (ss *scaleSet) isNodeManagedByAvailabilitySet(nodeName string, crt cacheReadType) (bool, error) {
	cached, err := ss.availabilitySetNodesCache.Get(availabilitySetNodesKey, crt)
	if err != nil {
		return false, err
	}

	availabilitySetNodes := cached.(sets.String)
	return availabilitySetNodes.Has(nodeName), nil
}
