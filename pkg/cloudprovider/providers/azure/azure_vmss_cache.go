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
	"fmt"
	"strings"
	"time"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/util/sets"
)

var (
	vmssNameSeparator = "_"

	nodeNameToScaleSetMappingKey = "k8sNodeNameToScaleSetMappingKey"
	availabilitySetNodesKey      = "k8sAvailabilitySetNodesKey"

	vmssCacheTTL                      = time.Minute
	vmssVMCacheTTL                    = time.Minute
	availabilitySetNodesCacheTTL      = 15 * time.Minute
	nodeNameToScaleSetMappingCacheTTL = 15 * time.Minute
)

// nodeNameToScaleSetMapping maps nodeName to scaleSet name.
// The map is required because vmss nodeName is not equal to its vmName.
type nodeNameToScaleSetMapping map[string]string

func (ss *scaleSet) makeVmssVMName(scaleSetName, instanceID string) string {
	return fmt.Sprintf("%s%s%s", scaleSetName, vmssNameSeparator, instanceID)
}

func extractVmssVMName(name string) (string, string, error) {
	split := strings.SplitAfter(name, vmssNameSeparator)
	if len(split) < 2 {
		glog.Errorf("Failed to extract vmssVMName %q", name)
		return "", "", ErrorNotVmssInstance
	}

	ssName := strings.Join(split[0:len(split)-1], "")
	// removing the trailing `vmssNameSeparator` since we used SplitAfter
	ssName = ssName[:len(ssName)-1]

	instanceID := split[len(split)-1]

	return ssName, instanceID, nil
}

func (ss *scaleSet) newVmssCache() (*timedCache, error) {
	getter := func(key string) (interface{}, error) {
		ctx, cancel := getContextWithCancel()
		defer cancel()
		result, err := ss.VirtualMachineScaleSetsClient.Get(ctx, ss.ResourceGroup, key)
		exists, message, realErr := checkResourceExistsFromError(err)
		if realErr != nil {
			return nil, realErr
		}

		if !exists {
			glog.V(2).Infof("Virtual machine scale set %q not found with message: %q", key, message)
			return nil, nil
		}

		return &result, nil
	}

	return newTimedcache(vmssCacheTTL, getter)
}

func (ss *scaleSet) newNodeNameToScaleSetMappingCache() (*timedCache, error) {
	getter := func(key string) (interface{}, error) {
		scaleSetNames, err := ss.listScaleSets()
		if err != nil {
			return nil, err
		}

		localCache := make(nodeNameToScaleSetMapping)
		for _, ssName := range scaleSetNames {
			vms, err := ss.listScaleSetVMs(ssName)
			if err != nil {
				return nil, err
			}

			for _, vm := range vms {
				if vm.OsProfile == nil || vm.OsProfile.ComputerName == nil {
					glog.Warningf("failed to get computerName for vmssVM (%q)", vm.Name)
					continue
				}

				computerName := strings.ToLower(*vm.OsProfile.ComputerName)
				localCache[computerName] = ssName
			}
		}

		return localCache, nil
	}

	return newTimedcache(nodeNameToScaleSetMappingCacheTTL, getter)
}

func (ss *scaleSet) newAvailabilitySetNodesCache() (*timedCache, error) {
	getter := func(key string) (interface{}, error) {
		vmList, err := ss.Cloud.VirtualMachineClientListWithRetry()
		if err != nil {
			return nil, err
		}

		localCache := sets.NewString()
		for _, vm := range vmList {
			localCache.Insert(*vm.Name)
		}

		return localCache, nil
	}

	return newTimedcache(availabilitySetNodesCacheTTL, getter)
}

func (ss *scaleSet) newVmssVMCache() (*timedCache, error) {
	getter := func(key string) (interface{}, error) {
		// vmssVM name's format is 'scaleSetName_instanceID'
		ssName, instanceID, err := extractVmssVMName(key)
		if err != nil {
			return nil, err
		}

		// Not found, the VM doesn't belong to any known scale sets.
		if ssName == "" {
			return nil, nil
		}

		ctx, cancel := getContextWithCancel()
		defer cancel()
		result, err := ss.VirtualMachineScaleSetVMsClient.Get(ctx, ss.ResourceGroup, ssName, instanceID)
		exists, message, realErr := checkResourceExistsFromError(err)
		if realErr != nil {
			return nil, realErr
		}

		if !exists {
			glog.V(2).Infof("Virtual machine scale set VM %q not found with message: %q", key, message)
			return nil, nil
		}

		return &result, nil
	}

	return newTimedcache(vmssVMCacheTTL, getter)
}

func (ss *scaleSet) getScaleSetNameByNodeName(nodeName string) (string, error) {
	getScaleSetName := func(nodeName string) (string, error) {
		nodeNameMapping, err := ss.nodeNameToScaleSetMappingCache.Get(nodeNameToScaleSetMappingKey)
		if err != nil {
			return "", err
		}

		realMapping := nodeNameMapping.(nodeNameToScaleSetMapping)
		if ssName, ok := realMapping[nodeName]; ok {
			return ssName, nil
		}

		return "", nil
	}

	ssName, err := getScaleSetName(nodeName)
	if err != nil {
		return "", err
	}

	if ssName != "" {
		return ssName, nil
	}

	// ssName is still not found, it is likely that new Nodes are created.
	// Force refresh the cache and try again.
	ss.nodeNameToScaleSetMappingCache.Delete(nodeNameToScaleSetMappingKey)
	return getScaleSetName(nodeName)
}

func (ss *scaleSet) isNodeManagedByAvailabilitySet(nodeName string) (bool, error) {
	cached, err := ss.availabilitySetNodesCache.Get(availabilitySetNodesKey)
	if err != nil {
		return false, err
	}

	availabilitySetNodes := cached.(sets.String)
	return availabilitySetNodes.Has(nodeName), nil
}
