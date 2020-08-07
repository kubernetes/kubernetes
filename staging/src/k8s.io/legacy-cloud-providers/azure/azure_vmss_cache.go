// +build !providerless

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
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-07-01/compute"
	"github.com/Azure/go-autorest/autorest/to"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog"
)

var (
	vmssNameSeparator  = "_"
	vmssCacheSeparator = "#"

	vmssKey                 = "k8svmssKey"
	availabilitySetNodesKey = "k8sAvailabilitySetNodesKey"

	availabilitySetNodesCacheTTL = 15 * time.Minute
	vmssTTL                      = 10 * time.Minute
	vmssVirtualMachinesTTL       = 10 * time.Minute

	vmssVirtualMachinesCacheTTLDefaultInSeconds = 600
)

type vmssVirtualMachinesEntry struct {
	resourceGroup  string
	vmssName       string
	instanceID     string
	virtualMachine *compute.VirtualMachineScaleSetVM
	lastUpdate     time.Time
}

type vmssEntry struct {
	vmss          *compute.VirtualMachineScaleSet
	resourceGroup string
	lastUpdate    time.Time
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
					vmss:          &scaleSet,
					resourceGroup: resourceGroup,
					lastUpdate:    time.Now().UTC(),
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

// getVMSSVMCache returns an *timedCache and cache key for a VMSS (creating that cache if new).
func (ss *scaleSet) getVMSSVMCache(resourceGroup, vmssName string) (string, *timedCache, error) {
	cacheKey := strings.ToLower(fmt.Sprintf("%s/%s", resourceGroup, vmssName))
	if entry, ok := ss.vmssVMCache.Load(cacheKey); ok {
		cache := entry.(*timedCache)
		return cacheKey, cache, nil
	}

	cache, err := ss.newVMSSVirtualMachinesCache(resourceGroup, vmssName, cacheKey)
	if err != nil {
		return "", nil, err
	}
	ss.vmssVMCache.Store(cacheKey, cache)
	return cacheKey, cache, nil
}

// gcVMSSVMCache delete stale VMSS VMs caches from deleted VMSSes.
func (ss *scaleSet) gcVMSSVMCache() error {
	cached, err := ss.vmssCache.Get(vmssKey, cacheReadTypeUnsafe)
	if err != nil {
		return err
	}

	vmsses := cached.(*sync.Map)
	removed := map[string]bool{}
	ss.vmssVMCache.Range(func(key, value interface{}) bool {
		cacheKey := key.(string)
		vlistIdx := cacheKey[strings.LastIndex(cacheKey, "/")+1:]
		if _, ok := vmsses.Load(vlistIdx); !ok {
			removed[cacheKey] = true
		}
		return true
	})

	for key := range removed {
		ss.vmssVMCache.Delete(key)
	}

	return nil
}

// newVMSSVirtualMachinesCache instanciates a new VMs cache for VMs belonging to the provided VMSS.
func (ss *scaleSet) newVMSSVirtualMachinesCache(resourceGroupName, vmssName, cacheKey string) (*timedCache, error) {
	getter := func(key string) (interface{}, error) {
		localCache := &sync.Map{} // [nodeName]*vmssVirtualMachinesEntry

		oldCache := make(map[string]vmssVirtualMachinesEntry)

		if vmssCache, ok := ss.vmssVMCache.Load(cacheKey); ok {
			// get old cache before refreshing the cache
			cache := vmssCache.(*timedCache)
			entry, exists, err := cache.store.GetByKey(cacheKey)
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

		vms, err := ss.listScaleSetVMs(vmssName, resourceGroupName)
		if err != nil {
			return nil, err
		}

		for i := range vms {
			vm := vms[i]
			if vm.OsProfile == nil || vm.OsProfile.ComputerName == nil {
				klog.Warningf("failed to get computerName for vmssVM (%q)", vmssName)
				continue
			}

			computerName := strings.ToLower(*vm.OsProfile.ComputerName)
			vmssVMCacheEntry := &vmssVirtualMachinesEntry{
				resourceGroup:  resourceGroupName,
				vmssName:       vmssName,
				instanceID:     to.String(vm.InstanceID),
				virtualMachine: &vm,
				lastUpdate:     time.Now().UTC(),
			}
			// set cache entry to nil when the VM is under deleting.
			if vm.VirtualMachineScaleSetVMProperties != nil &&
				strings.EqualFold(to.String(vm.VirtualMachineScaleSetVMProperties.ProvisioningState), string(compute.ProvisioningStateDeleting)) {
				klog.V(4).Infof("VMSS virtualMachine %q is under deleting, setting its cache to nil", computerName)
				vmssVMCacheEntry.virtualMachine = nil
			}
			localCache.Store(computerName, vmssVMCacheEntry)

			delete(oldCache, computerName)
		}

		// add old missing cache data with nil entries to prevent aggressive
		// ARM calls during cache invalidation
		for name, vmEntry := range oldCache {
			// if the nil cache entry has existed for 15 minutes in the cache
			// then it should not be added back to the cache
			if vmEntry.virtualMachine == nil && time.Since(vmEntry.lastUpdate) > 15*time.Minute {
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

		return localCache, nil
	}

	return newTimedcache(vmssVirtualMachinesTTL, getter)
}

func (ss *scaleSet) deleteCacheForNode(nodeName string) error {
	node, err := ss.getNodeIdentityByNodeName(nodeName, cacheReadTypeUnsafe)
	if err != nil {
		klog.Errorf("deleteCacheForNode(%s) failed with error: %v", nodeName, err)
		return err
	}

	cacheKey, timedcache, err := ss.getVMSSVMCache(node.resourceGroup, node.vmssName)
	if err != nil {
		klog.Errorf("deleteCacheForNode(%s) failed with error: %v", nodeName, err)
		return err
	}

	vmcache, err := timedcache.Get(cacheKey, cacheReadTypeUnsafe)
	if err != nil {
		klog.Errorf("deleteCacheForNode(%s) failed with error: %v", nodeName, err)
		return err
	}
	virtualMachines := vmcache.(*sync.Map)
	virtualMachines.Delete(nodeName)

	if err := ss.gcVMSSVMCache(); err != nil {
		klog.Errorf("deleteCacheForNode(%s) failed to gc stale vmss caches: %v", nodeName, err)
	}

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
