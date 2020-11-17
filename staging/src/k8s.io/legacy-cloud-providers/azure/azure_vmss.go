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

package azure

import (
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-12-01/compute"
	"github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-06-01/network"
	"github.com/Azure/go-autorest/autorest/to"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/klog/v2"
	azcache "k8s.io/legacy-cloud-providers/azure/cache"
	"k8s.io/legacy-cloud-providers/azure/metrics"
	utilnet "k8s.io/utils/net"
)

var (
	// virtualMachineScaleSetsDeallocating indicates VMSS instances are in Deallocating state.
	virtualMachineScaleSetsDeallocating = "Deallocating"

	// ErrorNotVmssInstance indicates an instance is not belonging to any vmss.
	ErrorNotVmssInstance = errors.New("not a vmss instance")

	scaleSetNameRE         = regexp.MustCompile(`.*/subscriptions/(?:.*)/Microsoft.Compute/virtualMachineScaleSets/(.+)/virtualMachines(?:.*)`)
	resourceGroupRE        = regexp.MustCompile(`.*/subscriptions/(?:.*)/resourceGroups/(.+)/providers/Microsoft.Compute/virtualMachineScaleSets/(?:.*)/virtualMachines(?:.*)`)
	vmssMachineIDTemplate  = "/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Compute/virtualMachineScaleSets/%s/virtualMachines/%s"
	vmssIPConfigurationRE  = regexp.MustCompile(`.*/subscriptions/(?:.*)/resourceGroups/(.+)/providers/Microsoft.Compute/virtualMachineScaleSets/(.+)/virtualMachines/(.+)/networkInterfaces(?:.*)`)
	vmssPIPConfigurationRE = regexp.MustCompile(`.*/subscriptions/(?:.*)/resourceGroups/(.+)/providers/Microsoft.Compute/virtualMachineScaleSets/(.+)/virtualMachines/(.+)/networkInterfaces/(.+)/ipConfigurations/(.+)/publicIPAddresses/(.+)`)
	vmssVMProviderIDRE     = regexp.MustCompile(`azure:///subscriptions/(?:.*)/resourceGroups/(.+)/providers/Microsoft.Compute/virtualMachineScaleSets/(.+)/virtualMachines/(?:\d+)`)
)

// vmssMetaInfo contains the metadata for a VMSS.
type vmssMetaInfo struct {
	vmssName      string
	resourceGroup string
}

// nodeIdentity identifies a node within a subscription.
type nodeIdentity struct {
	resourceGroup string
	vmssName      string
	nodeName      string
}

// scaleSet implements VMSet interface for Azure scale set.
type scaleSet struct {
	*Cloud

	// availabilitySet is also required for scaleSet because some instances
	// (e.g. master nodes) may not belong to any scale sets.
	availabilitySet VMSet

	vmssCache                 *azcache.TimedCache
	vmssVMCache               *sync.Map // [resourcegroup/vmssname]*azcache.TimedCache
	availabilitySetNodesCache *azcache.TimedCache
}

// newScaleSet creates a new scaleSet.
func newScaleSet(az *Cloud) (VMSet, error) {
	if az.Config.VmssVirtualMachinesCacheTTLInSeconds == 0 {
		az.Config.VmssVirtualMachinesCacheTTLInSeconds = vmssVirtualMachinesCacheTTLDefaultInSeconds
	}
	ss := &scaleSet{
		Cloud:           az,
		availabilitySet: newAvailabilitySet(az),
		vmssVMCache:     &sync.Map{},
	}

	var err error
	if !ss.DisableAvailabilitySetNodes {
		ss.availabilitySetNodesCache, err = ss.newAvailabilitySetNodesCache()
		if err != nil {
			return nil, err
		}
	}

	ss.vmssCache, err = ss.newVMSSCache()
	if err != nil {
		return nil, err
	}

	return ss, nil
}

func (ss *scaleSet) getVMSS(vmssName string, crt azcache.AzureCacheReadType) (*compute.VirtualMachineScaleSet, error) {
	getter := func(vmssName string) (*compute.VirtualMachineScaleSet, error) {
		cached, err := ss.vmssCache.Get(vmssKey, crt)
		if err != nil {
			return nil, err
		}

		vmsses := cached.(*sync.Map)
		if vmss, ok := vmsses.Load(vmssName); ok {
			result := vmss.(*vmssEntry)
			return result.vmss, nil
		}

		return nil, nil
	}

	vmss, err := getter(vmssName)
	if err != nil {
		return nil, err
	}
	if vmss != nil {
		return vmss, nil
	}

	klog.V(2).Infof("Couldn't find VMSS with name %s, refreshing the cache", vmssName)
	ss.vmssCache.Delete(vmssKey)
	vmss, err = getter(vmssName)
	if err != nil {
		return nil, err
	}

	if vmss == nil {
		return nil, cloudprovider.InstanceNotFound
	}
	return vmss, nil
}

// getVmssVMByNodeIdentity find virtualMachineScaleSetVM by nodeIdentity, using node's parent VMSS cache.
// Returns cloudprovider.InstanceNotFound if the node does not belong to the scale set named in nodeIdentity.
func (ss *scaleSet) getVmssVMByNodeIdentity(node *nodeIdentity, crt azcache.AzureCacheReadType) (string, string, *compute.VirtualMachineScaleSetVM, error) {
	cacheKey, cache, err := ss.getVMSSVMCache(node.resourceGroup, node.vmssName)
	if err != nil {
		return "", "", nil, err
	}

	getter := func(nodeName string, crt azcache.AzureCacheReadType) (string, string, *compute.VirtualMachineScaleSetVM, bool, error) {
		var found bool
		cached, err := cache.Get(cacheKey, crt)
		if err != nil {
			return "", "", nil, found, err
		}

		virtualMachines := cached.(*sync.Map)
		if vm, ok := virtualMachines.Load(nodeName); ok {
			result := vm.(*vmssVirtualMachinesEntry)
			found = true
			return result.vmssName, result.instanceID, result.virtualMachine, found, nil
		}

		return "", "", nil, found, nil
	}

	_, err = getScaleSetVMInstanceID(node.nodeName)
	if err != nil {
		return "", "", nil, err
	}

	vmssName, instanceID, vm, found, err := getter(node.nodeName, crt)
	if err != nil {
		return "", "", nil, err
	}

	if !found {
		klog.V(2).Infof("Couldn't find VMSS VM with nodeName %s, refreshing the cache", node.nodeName)
		vmssName, instanceID, vm, found, err = getter(node.nodeName, azcache.CacheReadTypeForceRefresh)
		if err != nil {
			return "", "", nil, err
		}
	}

	if found && vm != nil {
		return vmssName, instanceID, vm, nil
	}

	if !found || vm == nil {
		return "", "", nil, cloudprovider.InstanceNotFound
	}
	return vmssName, instanceID, vm, nil
}

// getVmssVM gets virtualMachineScaleSetVM by nodeName from cache.
// Returns cloudprovider.InstanceNotFound if nodeName does not belong to any scale set.
func (ss *scaleSet) getVmssVM(nodeName string, crt azcache.AzureCacheReadType) (string, string, *compute.VirtualMachineScaleSetVM, error) {
	node, err := ss.getNodeIdentityByNodeName(nodeName, crt)
	if err != nil {
		return "", "", nil, err
	}

	return ss.getVmssVMByNodeIdentity(node, crt)
}

// GetPowerStatusByNodeName returns the power state of the specified node.
func (ss *scaleSet) GetPowerStatusByNodeName(name string) (powerState string, err error) {
	managedByAS, err := ss.isNodeManagedByAvailabilitySet(name, azcache.CacheReadTypeUnsafe)
	if err != nil {
		klog.Errorf("Failed to check isNodeManagedByAvailabilitySet: %v", err)
		return "", err
	}
	if managedByAS {
		// vm is managed by availability set.
		return ss.availabilitySet.GetPowerStatusByNodeName(name)
	}

	_, _, vm, err := ss.getVmssVM(name, azcache.CacheReadTypeDefault)
	if err != nil {
		return powerState, err
	}

	if vm.InstanceView != nil && vm.InstanceView.Statuses != nil {
		statuses := *vm.InstanceView.Statuses
		for _, status := range statuses {
			state := to.String(status.Code)
			if strings.HasPrefix(state, vmPowerStatePrefix) {
				return strings.TrimPrefix(state, vmPowerStatePrefix), nil
			}
		}
	}

	// vm.InstanceView or vm.InstanceView.Statuses are nil when the VM is under deleting.
	klog.V(3).Infof("InstanceView for node %q is nil, assuming it's stopped", name)
	return vmPowerStateStopped, nil
}

// getCachedVirtualMachineByInstanceID gets scaleSetVMInfo from cache.
// The node must belong to one of scale sets.
func (ss *scaleSet) getVmssVMByInstanceID(resourceGroup, scaleSetName, instanceID string, crt azcache.AzureCacheReadType) (*compute.VirtualMachineScaleSetVM, error) {
	cacheKey, cache, err := ss.getVMSSVMCache(resourceGroup, scaleSetName)
	if err != nil {
		return nil, err
	}

	getter := func(crt azcache.AzureCacheReadType) (vm *compute.VirtualMachineScaleSetVM, found bool, err error) {
		cached, err := cache.Get(cacheKey, crt)
		if err != nil {
			return nil, false, err
		}

		virtualMachines := cached.(*sync.Map)
		virtualMachines.Range(func(key, value interface{}) bool {
			vmEntry := value.(*vmssVirtualMachinesEntry)
			if strings.EqualFold(vmEntry.resourceGroup, resourceGroup) &&
				strings.EqualFold(vmEntry.vmssName, scaleSetName) &&
				strings.EqualFold(vmEntry.instanceID, instanceID) {
				vm = vmEntry.virtualMachine
				found = true
				return false
			}

			return true
		})

		return vm, found, nil
	}

	vm, found, err := getter(crt)
	if err != nil {
		return nil, err
	}
	if !found {
		klog.V(2).Infof("Couldn't find VMSS VM with scaleSetName %q and instanceID %q, refreshing the cache", scaleSetName, instanceID)
		vm, found, err = getter(azcache.CacheReadTypeForceRefresh)
		if err != nil {
			return nil, err
		}
	}
	if found && vm != nil {
		return vm, nil
	}
	if found && vm == nil {
		klog.V(2).Infof("Couldn't find VMSS VM with scaleSetName %q and instanceID %q, refreshing the cache if it is expired", scaleSetName, instanceID)
		vm, found, err = getter(azcache.CacheReadTypeDefault)
		if err != nil {
			return nil, err
		}
	}
	if !found || vm == nil {
		return nil, cloudprovider.InstanceNotFound
	}

	return vm, nil
}

// GetInstanceIDByNodeName gets the cloud provider ID by node name.
// It must return ("", cloudprovider.InstanceNotFound) if the instance does
// not exist or is no longer running.
func (ss *scaleSet) GetInstanceIDByNodeName(name string) (string, error) {
	managedByAS, err := ss.isNodeManagedByAvailabilitySet(name, azcache.CacheReadTypeUnsafe)
	if err != nil {
		klog.Errorf("Failed to check isNodeManagedByAvailabilitySet: %v", err)
		return "", err
	}
	if managedByAS {
		// vm is managed by availability set.
		return ss.availabilitySet.GetInstanceIDByNodeName(name)
	}

	_, _, vm, err := ss.getVmssVM(name, azcache.CacheReadTypeUnsafe)
	if err != nil {
		return "", err
	}

	resourceID := *vm.ID
	convertedResourceID, err := convertResourceGroupNameToLower(resourceID)
	if err != nil {
		klog.Errorf("convertResourceGroupNameToLower failed with error: %v", err)
		return "", err
	}
	return convertedResourceID, nil
}

// GetNodeNameByProviderID gets the node name by provider ID.
// providerID example:
// 	 1. vmas providerID: azure:///subscriptions/subsid/resourceGroups/rg/providers/Microsoft.Compute/virtualMachines/aks-nodepool1-27053986-0
// 	 2. vmss providerID:
//		azure:///subscriptions/subsid/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/aks-agentpool-22126781-vmss/virtualMachines/1
//	    /subscriptions/subsid/resourceGroups/rg/providers/Microsoft.Compute/virtualMachineScaleSets/aks-agentpool-22126781-vmss/virtualMachines/k8s-agentpool-36841236-vmss_1
func (ss *scaleSet) GetNodeNameByProviderID(providerID string) (types.NodeName, error) {
	// NodeName is not part of providerID for vmss instances.
	scaleSetName, err := extractScaleSetNameByProviderID(providerID)
	if err != nil {
		klog.V(4).Infof("Can not extract scale set name from providerID (%s), assuming it is managed by availability set: %v", providerID, err)
		return ss.availabilitySet.GetNodeNameByProviderID(providerID)
	}

	resourceGroup, err := extractResourceGroupByProviderID(providerID)
	if err != nil {
		return "", fmt.Errorf("error of extracting resource group for node %q", providerID)
	}

	instanceID, err := getLastSegment(providerID, "/")
	if err != nil {
		klog.V(4).Infof("Can not extract instanceID from providerID (%s), assuming it is managed by availability set: %v", providerID, err)
		return ss.availabilitySet.GetNodeNameByProviderID(providerID)
	}

	// instanceID contains scaleSetName (returned by disk.ManagedBy), e.g. k8s-agentpool-36841236-vmss_1
	if strings.HasPrefix(strings.ToLower(instanceID), strings.ToLower(scaleSetName)) {
		instanceID, err = getLastSegment(instanceID, "_")
		if err != nil {
			return "", err
		}
	}

	vm, err := ss.getVmssVMByInstanceID(resourceGroup, scaleSetName, instanceID, azcache.CacheReadTypeUnsafe)
	if err != nil {
		return "", err
	}

	if vm.OsProfile != nil && vm.OsProfile.ComputerName != nil {
		nodeName := strings.ToLower(*vm.OsProfile.ComputerName)
		return types.NodeName(nodeName), nil
	}

	return "", nil
}

// GetInstanceTypeByNodeName gets the instance type by node name.
func (ss *scaleSet) GetInstanceTypeByNodeName(name string) (string, error) {
	managedByAS, err := ss.isNodeManagedByAvailabilitySet(name, azcache.CacheReadTypeUnsafe)
	if err != nil {
		klog.Errorf("Failed to check isNodeManagedByAvailabilitySet: %v", err)
		return "", err
	}
	if managedByAS {
		// vm is managed by availability set.
		return ss.availabilitySet.GetInstanceTypeByNodeName(name)
	}

	_, _, vm, err := ss.getVmssVM(name, azcache.CacheReadTypeUnsafe)
	if err != nil {
		return "", err
	}

	if vm.Sku != nil && vm.Sku.Name != nil {
		return *vm.Sku.Name, nil
	}

	return "", nil
}

// GetZoneByNodeName gets availability zone for the specified node. If the node is not running
// with availability zone, then it returns fault domain.
func (ss *scaleSet) GetZoneByNodeName(name string) (cloudprovider.Zone, error) {
	managedByAS, err := ss.isNodeManagedByAvailabilitySet(name, azcache.CacheReadTypeUnsafe)
	if err != nil {
		klog.Errorf("Failed to check isNodeManagedByAvailabilitySet: %v", err)
		return cloudprovider.Zone{}, err
	}
	if managedByAS {
		// vm is managed by availability set.
		return ss.availabilitySet.GetZoneByNodeName(name)
	}

	_, _, vm, err := ss.getVmssVM(name, azcache.CacheReadTypeUnsafe)
	if err != nil {
		return cloudprovider.Zone{}, err
	}

	var failureDomain string
	if vm.Zones != nil && len(*vm.Zones) > 0 {
		// Get availability zone for the node.
		zones := *vm.Zones
		zoneID, err := strconv.Atoi(zones[0])
		if err != nil {
			return cloudprovider.Zone{}, fmt.Errorf("failed to parse zone %q: %v", zones, err)
		}

		failureDomain = ss.makeZone(to.String(vm.Location), zoneID)
	} else if vm.InstanceView != nil && vm.InstanceView.PlatformFaultDomain != nil {
		// Availability zone is not used for the node, falling back to fault domain.
		failureDomain = strconv.Itoa(int(*vm.InstanceView.PlatformFaultDomain))
	} else {
		err = fmt.Errorf("failed to get zone info")
		klog.Errorf("GetZoneByNodeName: got unexpected error %v", err)
		ss.deleteCacheForNode(name)
		return cloudprovider.Zone{}, err
	}

	return cloudprovider.Zone{
		FailureDomain: strings.ToLower(failureDomain),
		Region:        strings.ToLower(to.String(vm.Location)),
	}, nil
}

// GetPrimaryVMSetName returns the VM set name depending on the configured vmType.
// It returns config.PrimaryScaleSetName for vmss and config.PrimaryAvailabilitySetName for standard vmType.
func (ss *scaleSet) GetPrimaryVMSetName() string {
	return ss.Config.PrimaryScaleSetName
}

// GetIPByNodeName gets machine private IP and public IP by node name.
func (ss *scaleSet) GetIPByNodeName(nodeName string) (string, string, error) {
	nic, err := ss.GetPrimaryInterface(nodeName)
	if err != nil {
		klog.Errorf("error: ss.GetIPByNodeName(%s), GetPrimaryInterface(%q), err=%v", nodeName, nodeName, err)
		return "", "", err
	}

	ipConfig, err := getPrimaryIPConfig(nic)
	if err != nil {
		klog.Errorf("error: ss.GetIPByNodeName(%s), getPrimaryIPConfig(%v), err=%v", nodeName, nic, err)
		return "", "", err
	}

	internalIP := *ipConfig.PrivateIPAddress
	publicIP := ""
	if ipConfig.PublicIPAddress != nil && ipConfig.PublicIPAddress.ID != nil {
		pipID := *ipConfig.PublicIPAddress.ID
		matches := vmssPIPConfigurationRE.FindStringSubmatch(pipID)
		if len(matches) == 7 {
			resourceGroupName := matches[1]
			virtualMachineScaleSetName := matches[2]
			virtualMachineIndex := matches[3]
			networkInterfaceName := matches[4]
			IPConfigurationName := matches[5]
			publicIPAddressName := matches[6]
			pip, existsPip, err := ss.getVMSSPublicIPAddress(resourceGroupName, virtualMachineScaleSetName, virtualMachineIndex, networkInterfaceName, IPConfigurationName, publicIPAddressName)
			if err != nil {
				klog.Errorf("ss.getVMSSPublicIPAddress() failed with error: %v", err)
				return "", "", err
			}
			if existsPip && pip.IPAddress != nil {
				publicIP = *pip.IPAddress
			}
		} else {
			klog.Warningf("Failed to get VMSS Public IP with ID %s", pipID)
		}
	}

	return internalIP, publicIP, nil
}

func (ss *scaleSet) getVMSSPublicIPAddress(resourceGroupName string, virtualMachineScaleSetName string, virtualMachineIndex string, networkInterfaceName string, IPConfigurationName string, publicIPAddressName string) (network.PublicIPAddress, bool, error) {
	ctx, cancel := getContextWithCancel()
	defer cancel()

	pip, err := ss.PublicIPAddressesClient.GetVirtualMachineScaleSetPublicIPAddress(ctx, resourceGroupName, virtualMachineScaleSetName, virtualMachineIndex, networkInterfaceName, IPConfigurationName, publicIPAddressName, "")
	exists, rerr := checkResourceExistsFromError(err)
	if rerr != nil {
		return pip, false, rerr.Error()
	}

	if !exists {
		klog.V(2).Infof("Public IP %q not found", publicIPAddressName)
		return pip, false, nil
	}

	return pip, exists, nil
}

// returns a list of private ips assigned to node
// TODO (khenidak): This should read all nics, not just the primary
// allowing users to split ipv4/v6 on multiple nics
func (ss *scaleSet) GetPrivateIPsByNodeName(nodeName string) ([]string, error) {
	ips := make([]string, 0)
	nic, err := ss.GetPrimaryInterface(nodeName)
	if err != nil {
		klog.Errorf("error: ss.GetIPByNodeName(%s), GetPrimaryInterface(%q), err=%v", nodeName, nodeName, err)
		return ips, err
	}

	if nic.IPConfigurations == nil {
		return ips, fmt.Errorf("nic.IPConfigurations for nic (nicname=%q) is nil", *nic.Name)
	}

	for _, ipConfig := range *(nic.IPConfigurations) {
		if ipConfig.PrivateIPAddress != nil {
			ips = append(ips, *(ipConfig.PrivateIPAddress))
		}
	}

	return ips, nil
}

// This returns the full identifier of the primary NIC for the given VM.
func (ss *scaleSet) getPrimaryInterfaceID(machine compute.VirtualMachineScaleSetVM) (string, error) {
	if machine.NetworkProfile == nil || machine.NetworkProfile.NetworkInterfaces == nil {
		return "", fmt.Errorf("failed to find the network interfaces for vm %s", to.String(machine.Name))
	}

	if len(*machine.NetworkProfile.NetworkInterfaces) == 1 {
		return *(*machine.NetworkProfile.NetworkInterfaces)[0].ID, nil
	}

	for _, ref := range *machine.NetworkProfile.NetworkInterfaces {
		if to.Bool(ref.Primary) {
			return *ref.ID, nil
		}
	}

	return "", fmt.Errorf("failed to find a primary nic for the vm. vmname=%q", to.String(machine.Name))
}

// getVmssMachineID returns the full identifier of a vmss virtual machine.
func (az *Cloud) getVmssMachineID(subscriptionID, resourceGroup, scaleSetName, instanceID string) string {
	return fmt.Sprintf(
		vmssMachineIDTemplate,
		subscriptionID,
		strings.ToLower(resourceGroup),
		scaleSetName,
		instanceID)
}

// machineName is composed of computerNamePrefix and 36-based instanceID.
// And instanceID part if in fixed length of 6 characters.
// Refer https://msftstack.wordpress.com/2017/05/10/figuring-out-azure-vm-scale-set-machine-names/.
func getScaleSetVMInstanceID(machineName string) (string, error) {
	nameLength := len(machineName)
	if nameLength < 6 {
		return "", ErrorNotVmssInstance
	}

	instanceID, err := strconv.ParseUint(machineName[nameLength-6:], 36, 64)
	if err != nil {
		return "", ErrorNotVmssInstance
	}

	return fmt.Sprintf("%d", instanceID), nil
}

// extractScaleSetNameByProviderID extracts the scaleset name by vmss node's ProviderID.
func extractScaleSetNameByProviderID(providerID string) (string, error) {
	matches := scaleSetNameRE.FindStringSubmatch(providerID)
	if len(matches) != 2 {
		return "", ErrorNotVmssInstance
	}

	return matches[1], nil
}

// extractResourceGroupByProviderID extracts the resource group name by vmss node's ProviderID.
func extractResourceGroupByProviderID(providerID string) (string, error) {
	matches := resourceGroupRE.FindStringSubmatch(providerID)
	if len(matches) != 2 {
		return "", ErrorNotVmssInstance
	}

	return matches[1], nil
}

// listScaleSets lists all scale sets with orchestrationMode ScaleSetVM.
func (ss *scaleSet) listScaleSets(resourceGroup string) ([]string, error) {
	ctx, cancel := getContextWithCancel()
	defer cancel()

	allScaleSets, rerr := ss.VirtualMachineScaleSetsClient.List(ctx, resourceGroup)
	if rerr != nil {
		klog.Errorf("VirtualMachineScaleSetsClient.List failed: %v", rerr)
		return nil, rerr.Error()
	}

	ssNames := make([]string, 0)
	for _, vmss := range allScaleSets {
		name := *vmss.Name
		if vmss.Sku != nil && to.Int64(vmss.Sku.Capacity) == 0 {
			klog.V(3).Infof("Capacity of VMSS %q is 0, skipping", name)
			continue
		}

		if vmss.VirtualMachineScaleSetProperties == nil || vmss.VirtualMachineScaleSetProperties.VirtualMachineProfile == nil {
			klog.V(3).Infof("VMSS %q orchestrationMode is VirtualMachine, skipping", name)
			continue
		}

		ssNames = append(ssNames, name)
	}

	return ssNames, nil
}

// getNodeIdentityByNodeName use the VMSS cache to find a node's resourcegroup and vmss, returned in a nodeIdentity.
func (ss *scaleSet) getNodeIdentityByNodeName(nodeName string, crt azcache.AzureCacheReadType) (*nodeIdentity, error) {
	getter := func(nodeName string, crt azcache.AzureCacheReadType) (*nodeIdentity, error) {
		node := &nodeIdentity{
			nodeName: nodeName,
		}

		cached, err := ss.vmssCache.Get(vmssKey, crt)
		if err != nil {
			return nil, err
		}

		vmsses := cached.(*sync.Map)
		vmsses.Range(func(key, value interface{}) bool {
			v := value.(*vmssEntry)
			if v.vmss.Name == nil {
				return true
			}

			vmssPrefix := *v.vmss.Name
			if v.vmss.VirtualMachineProfile != nil &&
				v.vmss.VirtualMachineProfile.OsProfile != nil &&
				v.vmss.VirtualMachineProfile.OsProfile.ComputerNamePrefix != nil {
				vmssPrefix = *v.vmss.VirtualMachineProfile.OsProfile.ComputerNamePrefix
			}

			if strings.EqualFold(vmssPrefix, nodeName[:len(nodeName)-6]) {
				node.vmssName = *v.vmss.Name
				node.resourceGroup = v.resourceGroup
				return false
			}

			return true
		})
		return node, nil
	}

	if _, err := getScaleSetVMInstanceID(nodeName); err != nil {
		return nil, err
	}

	node, err := getter(nodeName, crt)
	if err != nil {
		return nil, err
	}
	if node.vmssName != "" {
		return node, nil
	}

	klog.V(2).Infof("Couldn't find VMSS for node %s, refreshing the cache", nodeName)
	node, err = getter(nodeName, azcache.CacheReadTypeForceRefresh)
	if err != nil {
		return nil, err
	}
	if node.vmssName == "" {
		return nil, cloudprovider.InstanceNotFound
	}
	return node, nil
}

// listScaleSetVMs lists VMs belonging to the specified scale set.
func (ss *scaleSet) listScaleSetVMs(scaleSetName, resourceGroup string) ([]compute.VirtualMachineScaleSetVM, error) {
	ctx, cancel := getContextWithCancel()
	defer cancel()

	allVMs, rerr := ss.VirtualMachineScaleSetVMsClient.List(ctx, resourceGroup, scaleSetName, string(compute.InstanceView))
	if rerr != nil {
		klog.Errorf("VirtualMachineScaleSetVMsClient.List failed: %v", rerr)
		if rerr.IsNotFound() {
			return nil, cloudprovider.InstanceNotFound
		}
		return nil, rerr.Error()
	}

	return allVMs, nil
}

// getAgentPoolScaleSets lists the virtual machines for the resource group and then builds
// a list of scale sets that match the nodes available to k8s.
func (ss *scaleSet) getAgentPoolScaleSets(nodes []*v1.Node) (*[]string, error) {
	agentPoolScaleSets := &[]string{}
	for nx := range nodes {
		if isMasterNode(nodes[nx]) {
			continue
		}

		if ss.ShouldNodeExcludedFromLoadBalancer(nodes[nx]) {
			continue
		}

		nodeName := nodes[nx].Name
		ssName, _, _, err := ss.getVmssVM(nodeName, azcache.CacheReadTypeDefault)
		if err != nil {
			return nil, err
		}

		if ssName == "" {
			klog.V(3).Infof("Node %q is not belonging to any known scale sets", nodeName)
			continue
		}

		*agentPoolScaleSets = append(*agentPoolScaleSets, ssName)
	}

	return agentPoolScaleSets, nil
}

// GetVMSetNames selects all possible availability sets or scale sets
// (depending vmType configured) for service load balancer. If the service has
// no loadbalancer mode annotation returns the primary VMSet. If service annotation
// for loadbalancer exists then return the eligible VMSet.
func (ss *scaleSet) GetVMSetNames(service *v1.Service, nodes []*v1.Node) (vmSetNames *[]string, err error) {
	hasMode, isAuto, serviceVMSetNames := getServiceLoadBalancerMode(service)
	useSingleSLB := ss.useStandardLoadBalancer() && !ss.EnableMultipleStandardLoadBalancers
	if !hasMode || useSingleSLB {
		// no mode specified in service annotation or use single SLB mode
		// default to PrimaryScaleSetName
		scaleSetNames := &[]string{ss.Config.PrimaryScaleSetName}
		return scaleSetNames, nil
	}

	scaleSetNames, err := ss.getAgentPoolScaleSets(nodes)
	if err != nil {
		klog.Errorf("ss.GetVMSetNames - getAgentPoolScaleSets failed err=(%v)", err)
		return nil, err
	}
	if len(*scaleSetNames) == 0 {
		klog.Errorf("ss.GetVMSetNames - No scale sets found for nodes in the cluster, node count(%d)", len(nodes))
		return nil, fmt.Errorf("no scale sets found for nodes, node count(%d)", len(nodes))
	}

	// sort the list to have deterministic selection
	sort.Strings(*scaleSetNames)

	if !isAuto {
		if serviceVMSetNames == nil || len(serviceVMSetNames) == 0 {
			return nil, fmt.Errorf("service annotation for LoadBalancerMode is empty, it should have __auto__ or availability sets value")
		}
		// validate scale set exists
		var found bool
		for sasx := range serviceVMSetNames {
			for asx := range *scaleSetNames {
				if strings.EqualFold((*scaleSetNames)[asx], serviceVMSetNames[sasx]) {
					found = true
					serviceVMSetNames[sasx] = (*scaleSetNames)[asx]
					break
				}
			}
			if !found {
				klog.Errorf("ss.GetVMSetNames - scale set (%s) in service annotation not found", serviceVMSetNames[sasx])
				return nil, fmt.Errorf("scale set (%s) - not found", serviceVMSetNames[sasx])
			}
		}
		vmSetNames = &serviceVMSetNames
	}

	return vmSetNames, nil
}

// extractResourceGroupByVMSSNicID extracts the resource group name by vmss nicID.
func extractResourceGroupByVMSSNicID(nicID string) (string, error) {
	matches := vmssIPConfigurationRE.FindStringSubmatch(nicID)
	if len(matches) != 4 {
		return "", fmt.Errorf("error of extracting resourceGroup from nicID %q", nicID)
	}

	return matches[1], nil
}

// GetPrimaryInterface gets machine primary network interface by node name and vmSet.
func (ss *scaleSet) GetPrimaryInterface(nodeName string) (network.Interface, error) {
	managedByAS, err := ss.isNodeManagedByAvailabilitySet(nodeName, azcache.CacheReadTypeDefault)
	if err != nil {
		klog.Errorf("Failed to check isNodeManagedByAvailabilitySet: %v", err)
		return network.Interface{}, err
	}
	if managedByAS {
		// vm is managed by availability set.
		return ss.availabilitySet.GetPrimaryInterface(nodeName)
	}

	ssName, instanceID, vm, err := ss.getVmssVM(nodeName, azcache.CacheReadTypeDefault)
	if err != nil {
		// VM is availability set, but not cached yet in availabilitySetNodesCache.
		if err == ErrorNotVmssInstance {
			return ss.availabilitySet.GetPrimaryInterface(nodeName)
		}

		klog.Errorf("error: ss.GetPrimaryInterface(%s), ss.getVmssVM(%s), err=%v", nodeName, nodeName, err)
		return network.Interface{}, err
	}

	primaryInterfaceID, err := ss.getPrimaryInterfaceID(*vm)
	if err != nil {
		klog.Errorf("error: ss.GetPrimaryInterface(%s), ss.getPrimaryInterfaceID(), err=%v", nodeName, err)
		return network.Interface{}, err
	}

	nicName, err := getLastSegment(primaryInterfaceID, "/")
	if err != nil {
		klog.Errorf("error: ss.GetPrimaryInterface(%s), getLastSegment(%s), err=%v", nodeName, primaryInterfaceID, err)
		return network.Interface{}, err
	}
	resourceGroup, err := extractResourceGroupByVMSSNicID(primaryInterfaceID)
	if err != nil {
		return network.Interface{}, err
	}

	ctx, cancel := getContextWithCancel()
	defer cancel()
	nic, rerr := ss.InterfacesClient.GetVirtualMachineScaleSetNetworkInterface(ctx, resourceGroup, ssName, instanceID, nicName, "")
	if rerr != nil {
		exists, realErr := checkResourceExistsFromError(rerr)
		if realErr != nil {
			klog.Errorf("error: ss.GetPrimaryInterface(%s), ss.GetVirtualMachineScaleSetNetworkInterface.Get(%s, %s, %s), err=%v", nodeName, resourceGroup, ssName, nicName, realErr)
			return network.Interface{}, realErr.Error()
		}

		if !exists {
			return network.Interface{}, cloudprovider.InstanceNotFound
		}
	}

	// Fix interface's location, which is required when updating the interface.
	// TODO: is this a bug of azure SDK?
	if nic.Location == nil || *nic.Location == "" {
		nic.Location = vm.Location
	}

	return nic, nil
}

// getPrimaryNetworkInterfaceConfiguration gets primary network interface configuration for scale set virtual machine.
func (ss *scaleSet) getPrimaryNetworkInterfaceConfiguration(networkConfigurations []compute.VirtualMachineScaleSetNetworkConfiguration, nodeName string) (*compute.VirtualMachineScaleSetNetworkConfiguration, error) {
	if len(networkConfigurations) == 1 {
		return &networkConfigurations[0], nil
	}

	for idx := range networkConfigurations {
		networkConfig := &networkConfigurations[idx]
		if networkConfig.Primary != nil && *networkConfig.Primary == true {
			return networkConfig, nil
		}
	}

	return nil, fmt.Errorf("failed to find a primary network configuration for the scale set VM %q", nodeName)
}

// getPrimaryNetworkInterfaceConfigurationForScaleSet gets primary network interface configuration for scale set.
func (ss *scaleSet) getPrimaryNetworkInterfaceConfigurationForScaleSet(networkConfigurations []compute.VirtualMachineScaleSetNetworkConfiguration, vmssName string) (*compute.VirtualMachineScaleSetNetworkConfiguration, error) {
	if len(networkConfigurations) == 1 {
		return &networkConfigurations[0], nil
	}

	for idx := range networkConfigurations {
		networkConfig := &networkConfigurations[idx]
		if networkConfig.Primary != nil && *networkConfig.Primary == true {
			return networkConfig, nil
		}
	}

	return nil, fmt.Errorf("failed to find a primary network configuration for the scale set %q", vmssName)
}

func getPrimaryIPConfigFromVMSSNetworkConfig(config *compute.VirtualMachineScaleSetNetworkConfiguration) (*compute.VirtualMachineScaleSetIPConfiguration, error) {
	ipConfigurations := *config.IPConfigurations
	if len(ipConfigurations) == 1 {
		return &ipConfigurations[0], nil
	}

	for idx := range ipConfigurations {
		ipConfig := &ipConfigurations[idx]
		if ipConfig.Primary != nil && *ipConfig.Primary == true {
			return ipConfig, nil
		}
	}

	return nil, fmt.Errorf("failed to find a primary IP configuration")
}

func (ss *scaleSet) getConfigForScaleSetByIPFamily(config *compute.VirtualMachineScaleSetNetworkConfiguration, nodeName string, IPv6 bool) (*compute.VirtualMachineScaleSetIPConfiguration, error) {
	ipConfigurations := *config.IPConfigurations

	var ipVersion compute.IPVersion
	if IPv6 {
		ipVersion = compute.IPv6
	} else {
		ipVersion = compute.IPv4
	}
	for idx := range ipConfigurations {
		ipConfig := &ipConfigurations[idx]
		if ipConfig.PrivateIPAddressVersion == ipVersion {
			return ipConfig, nil
		}
	}

	return nil, fmt.Errorf("failed to find a IPconfiguration(IPv6=%v) for the scale set VM %q", IPv6, nodeName)
}

// EnsureHostInPool ensures the given VM's Primary NIC's Primary IP Configuration is
// participating in the specified LoadBalancer Backend Pool, which returns (resourceGroup, vmssName, instanceID, vmssVM, error).
func (ss *scaleSet) EnsureHostInPool(service *v1.Service, nodeName types.NodeName, backendPoolID string, vmSetName string, isInternal bool) (string, string, string, *compute.VirtualMachineScaleSetVM, error) {
	klog.V(3).Infof("ensuring node %q of scaleset %q in LB backendpool %q", nodeName, vmSetName, backendPoolID)
	vmName := mapNodeNameToVMName(nodeName)
	ssName, instanceID, vm, err := ss.getVmssVM(vmName, azcache.CacheReadTypeDefault)
	if err != nil {
		return "", "", "", nil, err
	}

	// Check scale set name:
	// - For basic SKU load balancer, return nil if the node's scale set is mismatched with vmSetName.
	// - For single standard SKU load balancer, backend could belong to multiple VMSS, so we
	//   don't check vmSet for it.
	// - For multiple standard SKU load balancers, the behavior is similar to the basic load balancer
	needCheck := false
	if !ss.useStandardLoadBalancer() {
		// need to check the vmSet name when using the basic LB
		needCheck = true
	} else if ss.EnableMultipleStandardLoadBalancers {
		// need to check the vmSet name when using multiple standard LBs
		needCheck = true
	}
	if vmSetName != "" && needCheck && !strings.EqualFold(vmSetName, ssName) {
		klog.V(3).Infof("EnsureHostInPool skips node %s because it is not in the scaleSet %s", vmName, vmSetName)
		return "", "", "", nil, nil
	}

	// Find primary network interface configuration.
	if vm.NetworkProfileConfiguration.NetworkInterfaceConfigurations == nil {
		klog.V(4).Infof("EnsureHostInPool: cannot obtain the primary network interface configuration, of vm %s, probably because the vm's being deleted", vmName)
		return "", "", "", nil, nil
	}

	networkInterfaceConfigurations := *vm.NetworkProfileConfiguration.NetworkInterfaceConfigurations
	primaryNetworkInterfaceConfiguration, err := ss.getPrimaryNetworkInterfaceConfiguration(networkInterfaceConfigurations, vmName)
	if err != nil {
		return "", "", "", nil, err
	}

	var primaryIPConfiguration *compute.VirtualMachineScaleSetIPConfiguration
	ipv6 := utilnet.IsIPv6String(service.Spec.ClusterIP)
	// Find primary network interface configuration.
	if !ss.Cloud.ipv6DualStackEnabled && !ipv6 {
		// Find primary IP configuration.
		primaryIPConfiguration, err = getPrimaryIPConfigFromVMSSNetworkConfig(primaryNetworkInterfaceConfiguration)
		if err != nil {
			return "", "", "", nil, err
		}
	} else {
		// For IPv6 or dualstack service, we need to pick the right IP configuration based on the cluster ip family
		// IPv6 configuration is only supported as non-primary, so we need to fetch the ip configuration where the
		// privateIPAddressVersion matches the clusterIP family
		primaryIPConfiguration, err = ss.getConfigForScaleSetByIPFamily(primaryNetworkInterfaceConfiguration, vmName, ipv6)
		if err != nil {
			return "", "", "", nil, err
		}
	}

	// Update primary IP configuration's LoadBalancerBackendAddressPools.
	foundPool := false
	newBackendPools := []compute.SubResource{}
	if primaryIPConfiguration.LoadBalancerBackendAddressPools != nil {
		newBackendPools = *primaryIPConfiguration.LoadBalancerBackendAddressPools
	}
	for _, existingPool := range newBackendPools {
		if strings.EqualFold(backendPoolID, *existingPool.ID) {
			foundPool = true
			break
		}
	}

	// The backendPoolID has already been found from existing LoadBalancerBackendAddressPools.
	if foundPool {
		return "", "", "", nil, nil
	}

	if ss.useStandardLoadBalancer() && len(newBackendPools) > 0 {
		// Although standard load balancer supports backends from multiple scale
		// sets, the same network interface couldn't be added to more than one load balancer of
		// the same type. Omit those nodes (e.g. masters) so Azure ARM won't complain
		// about this.
		newBackendPoolsIDs := make([]string, 0, len(newBackendPools))
		for _, pool := range newBackendPools {
			if pool.ID != nil {
				newBackendPoolsIDs = append(newBackendPoolsIDs, *pool.ID)
			}
		}
		isSameLB, oldLBName, err := isBackendPoolOnSameLB(backendPoolID, newBackendPoolsIDs)
		if err != nil {
			return "", "", "", nil, err
		}
		if !isSameLB {
			klog.V(4).Infof("Node %q has already been added to LB %q, omit adding it to a new one", nodeName, oldLBName)
			return "", "", "", nil, nil
		}
	}

	// Compose a new vmssVM with added backendPoolID.
	newBackendPools = append(newBackendPools,
		compute.SubResource{
			ID: to.StringPtr(backendPoolID),
		})
	primaryIPConfiguration.LoadBalancerBackendAddressPools = &newBackendPools
	newVM := &compute.VirtualMachineScaleSetVM{
		Sku:      vm.Sku,
		Location: vm.Location,
		VirtualMachineScaleSetVMProperties: &compute.VirtualMachineScaleSetVMProperties{
			HardwareProfile: vm.HardwareProfile,
			NetworkProfileConfiguration: &compute.VirtualMachineScaleSetVMNetworkProfileConfiguration{
				NetworkInterfaceConfigurations: &networkInterfaceConfigurations,
			},
		},
	}

	// Get the node resource group.
	nodeResourceGroup, err := ss.GetNodeResourceGroup(vmName)
	if err != nil {
		return "", "", "", nil, err
	}

	return nodeResourceGroup, ssName, instanceID, newVM, nil
}

func getVmssAndResourceGroupNameByVMProviderID(providerID string) (string, string, error) {
	matches := vmssVMProviderIDRE.FindStringSubmatch(providerID)
	if len(matches) != 3 {
		return "", "", ErrorNotVmssInstance
	}
	return matches[1], matches[2], nil
}

func (ss *scaleSet) ensureVMSSInPool(service *v1.Service, nodes []*v1.Node, backendPoolID string, vmSetName string) error {
	klog.V(2).Infof("ensureVMSSInPool: ensuring VMSS with backendPoolID %s", backendPoolID)
	vmssNamesMap := make(map[string]bool)

	// the single standard load balancer supports multiple vmss in its backend while
	// multiple standard load balancers and the basic load balancer doesn't
	if ss.useStandardLoadBalancer() && !ss.EnableMultipleStandardLoadBalancers {
		for _, node := range nodes {
			if ss.excludeMasterNodesFromStandardLB() && isMasterNode(node) {
				continue
			}

			if ss.ShouldNodeExcludedFromLoadBalancer(node) {
				klog.V(4).Infof("Excluding unmanaged/external-resource-group node %q", node.Name)
				continue
			}

			// in this scenario the vmSetName is an empty string and the name of vmss should be obtained from the provider IDs of nodes
			resourceGroupName, vmssName, err := getVmssAndResourceGroupNameByVMProviderID(node.Spec.ProviderID)
			if err != nil {
				klog.V(4).Infof("ensureVMSSInPool: found VMAS node %s, will skip checking and continue", node.Name)
				continue
			}
			// only vmsses in the resource group same as it's in azure config are included
			if strings.EqualFold(resourceGroupName, ss.ResourceGroup) {
				vmssNamesMap[vmssName] = true
			}
		}
	} else {
		vmssNamesMap[vmSetName] = true
	}

	klog.V(2).Infof("ensureVMSSInPool begins to update VMSS %v with backendPoolID %s", vmssNamesMap, backendPoolID)
	for vmssName := range vmssNamesMap {
		vmss, err := ss.getVMSS(vmssName, azcache.CacheReadTypeDefault)
		if err != nil {
			return err
		}

		// When vmss is being deleted, CreateOrUpdate API would report "the vmss is being deleted" error.
		// Since it is being deleted, we shouldn't send more CreateOrUpdate requests for it.
		if vmss.ProvisioningState != nil && strings.EqualFold(*vmss.ProvisioningState, virtualMachineScaleSetsDeallocating) {
			klog.V(3).Infof("ensureVMSSInPool: found vmss %s being deleted, skipping", vmssName)
			continue
		}

		if vmss.VirtualMachineProfile.NetworkProfile.NetworkInterfaceConfigurations == nil {
			klog.V(4).Infof("EnsureHostInPool: cannot obtain the primary network interface configuration of vmss %s", vmssName)
			continue
		}
		vmssNIC := *vmss.VirtualMachineProfile.NetworkProfile.NetworkInterfaceConfigurations
		primaryNIC, err := ss.getPrimaryNetworkInterfaceConfigurationForScaleSet(vmssNIC, vmssName)
		if err != nil {
			return err
		}
		var primaryIPConfig *compute.VirtualMachineScaleSetIPConfiguration
		ipv6 := utilnet.IsIPv6String(service.Spec.ClusterIP)
		// Find primary network interface configuration.
		if !ss.Cloud.ipv6DualStackEnabled && !ipv6 {
			// Find primary IP configuration.
			primaryIPConfig, err = getPrimaryIPConfigFromVMSSNetworkConfig(primaryNIC)
			if err != nil {
				return err
			}
		} else {
			primaryIPConfig, err = ss.getConfigForScaleSetByIPFamily(primaryNIC, "", ipv6)
			if err != nil {
				return err
			}
		}

		loadBalancerBackendAddressPools := []compute.SubResource{}
		if primaryIPConfig.LoadBalancerBackendAddressPools != nil {
			loadBalancerBackendAddressPools = *primaryIPConfig.LoadBalancerBackendAddressPools
		}

		var found bool
		for _, loadBalancerBackendAddressPool := range loadBalancerBackendAddressPools {
			if strings.EqualFold(*loadBalancerBackendAddressPool.ID, backendPoolID) {
				found = true
				break
			}
		}
		if found {
			continue
		}

		if ss.useStandardLoadBalancer() && len(loadBalancerBackendAddressPools) > 0 {
			// Although standard load balancer supports backends from multiple scale
			// sets, the same network interface couldn't be added to more than one load balancer of
			// the same type. Omit those nodes (e.g. masters) so Azure ARM won't complain
			// about this.
			newBackendPoolsIDs := make([]string, 0, len(loadBalancerBackendAddressPools))
			for _, pool := range loadBalancerBackendAddressPools {
				if pool.ID != nil {
					newBackendPoolsIDs = append(newBackendPoolsIDs, *pool.ID)
				}
			}
			isSameLB, oldLBName, err := isBackendPoolOnSameLB(backendPoolID, newBackendPoolsIDs)
			if err != nil {
				return err
			}
			if !isSameLB {
				klog.V(4).Infof("VMSS %q has already been added to LB %q, omit adding it to a new one", vmssName, oldLBName)
				return nil
			}
		}

		// Compose a new vmss with added backendPoolID.
		loadBalancerBackendAddressPools = append(loadBalancerBackendAddressPools,
			compute.SubResource{
				ID: to.StringPtr(backendPoolID),
			})
		primaryIPConfig.LoadBalancerBackendAddressPools = &loadBalancerBackendAddressPools
		newVMSS := compute.VirtualMachineScaleSet{
			Sku:      vmss.Sku,
			Location: vmss.Location,
			VirtualMachineScaleSetProperties: &compute.VirtualMachineScaleSetProperties{
				VirtualMachineProfile: &compute.VirtualMachineScaleSetVMProfile{
					NetworkProfile: &compute.VirtualMachineScaleSetNetworkProfile{
						NetworkInterfaceConfigurations: &vmssNIC,
					},
				},
			},
		}

		klog.V(2).Infof("ensureVMSSInPool begins to update vmss(%s) with new backendPoolID %s", vmssName, backendPoolID)
		rerr := ss.CreateOrUpdateVMSS(ss.ResourceGroup, vmssName, newVMSS)
		if rerr != nil {
			klog.Errorf("ensureVMSSInPool CreateOrUpdateVMSS(%s) with new backendPoolID %s, err: %v", vmssName, backendPoolID, err)
			return rerr.Error()
		}
	}
	return nil
}

// EnsureHostsInPool ensures the given Node's primary IP configurations are
// participating in the specified LoadBalancer Backend Pool.
func (ss *scaleSet) EnsureHostsInPool(service *v1.Service, nodes []*v1.Node, backendPoolID string, vmSetName string, isInternal bool) error {
	mc := metrics.NewMetricContext("services", "vmss_ensure_hosts_in_pool", ss.ResourceGroup, ss.SubscriptionID, service.Name)
	isOperationSucceeded := false
	defer func() {
		mc.ObserveOperationWithResult(isOperationSucceeded)
	}()

	hostUpdates := make([]func() error, 0, len(nodes))
	nodeUpdates := make(map[vmssMetaInfo]map[string]compute.VirtualMachineScaleSetVM)
	errors := make([]error, 0)
	for _, node := range nodes {
		localNodeName := node.Name

		if ss.useStandardLoadBalancer() && ss.excludeMasterNodesFromStandardLB() && isMasterNode(node) {
			klog.V(4).Infof("Excluding master node %q from load balancer backendpool %q", localNodeName, backendPoolID)
			continue
		}

		if ss.ShouldNodeExcludedFromLoadBalancer(node) {
			klog.V(4).Infof("Excluding unmanaged/external-resource-group node %q", localNodeName)
			continue
		}

		// Check whether the node is VMAS virtual machine.
		managedByAS, err := ss.isNodeManagedByAvailabilitySet(localNodeName, azcache.CacheReadTypeDefault)
		if err != nil {
			klog.Errorf("Failed to check isNodeManagedByAvailabilitySet(%s): %v", localNodeName, err)
			errors = append(errors, err)
			continue
		}

		if managedByAS {
			// VMAS nodes should also be added to the SLB backends.
			if ss.useStandardLoadBalancer() {
				hostUpdates = append(hostUpdates, func() error {
					_, _, _, _, err := ss.availabilitySet.EnsureHostInPool(service, types.NodeName(localNodeName), backendPoolID, vmSetName, isInternal)
					return err
				})
				continue
			}

			klog.V(3).Infof("EnsureHostsInPool skips node %s because VMAS nodes couldn't be added to basic LB with VMSS backends", localNodeName)
			continue
		}

		nodeResourceGroup, nodeVMSS, nodeInstanceID, nodeVMSSVM, err := ss.EnsureHostInPool(service, types.NodeName(localNodeName), backendPoolID, vmSetName, isInternal)
		if err != nil {
			klog.Errorf("EnsureHostInPool(%s): backendPoolID(%s) - failed to ensure host in pool: %q", getServiceName(service), backendPoolID, err)
			errors = append(errors, err)
			continue
		}

		// No need to update if nodeVMSSVM is nil.
		if nodeVMSSVM == nil {
			continue
		}

		nodeVMSSMetaInfo := vmssMetaInfo{vmssName: nodeVMSS, resourceGroup: nodeResourceGroup}
		if v, ok := nodeUpdates[nodeVMSSMetaInfo]; ok {
			v[nodeInstanceID] = *nodeVMSSVM
		} else {
			nodeUpdates[nodeVMSSMetaInfo] = map[string]compute.VirtualMachineScaleSetVM{
				nodeInstanceID: *nodeVMSSVM,
			}
		}

		// Invalidate the cache since the VMSS VM would be updated.
		defer ss.deleteCacheForNode(localNodeName)
	}

	// Update VMs with best effort that have already been added to nodeUpdates.
	for meta, update := range nodeUpdates {
		// create new instance of meta and update for passing to anonymous function
		meta := meta
		update := update
		hostUpdates = append(hostUpdates, func() error {
			ctx, cancel := getContextWithCancel()
			defer cancel()
			klog.V(2).Infof("EnsureHostInPool begins to UpdateVMs for VMSS(%s, %s) with new backendPoolID %s", meta.resourceGroup, meta.vmssName, backendPoolID)
			rerr := ss.VirtualMachineScaleSetVMsClient.UpdateVMs(ctx, meta.resourceGroup, meta.vmssName, update, "network_update")
			if rerr != nil {
				klog.Errorf("EnsureHostInPool UpdateVMs for VMSS(%s, %s) failed with error %v", meta.resourceGroup, meta.vmssName, rerr.Error())
				return rerr.Error()
			}

			return nil
		})
	}
	errs := utilerrors.AggregateGoroutines(hostUpdates...)
	if errs != nil {
		return utilerrors.Flatten(errs)
	}

	// Fail if there are other errors.
	if len(errors) > 0 {
		return utilerrors.Flatten(utilerrors.NewAggregate(errors))
	}

	// Ensure the backendPoolID is also added on VMSS itself.
	// Refer to issue kubernetes/kubernetes#80365 for detailed information
	err := ss.ensureVMSSInPool(service, nodes, backendPoolID, vmSetName)
	if err != nil {
		return err
	}

	isOperationSucceeded = true
	return nil
}

// ensureBackendPoolDeletedFromNode ensures the loadBalancer backendAddressPools deleted
// from the specified node, which returns (resourceGroup, vmssName, instanceID, vmssVM, error).
func (ss *scaleSet) ensureBackendPoolDeletedFromNode(nodeName, backendPoolID string) (string, string, string, *compute.VirtualMachineScaleSetVM, error) {
	ssName, instanceID, vm, err := ss.getVmssVM(nodeName, azcache.CacheReadTypeDefault)
	if err != nil {
		return "", "", "", nil, err
	}

	// Find primary network interface configuration.
	if vm.NetworkProfileConfiguration.NetworkInterfaceConfigurations == nil {
		klog.V(4).Infof("EnsureHostInPool: cannot obtain the primary network interface configuration, of vm %s, probably because the vm's being deleted", nodeName)
		return "", "", "", nil, nil
	}
	networkInterfaceConfigurations := *vm.NetworkProfileConfiguration.NetworkInterfaceConfigurations
	primaryNetworkInterfaceConfiguration, err := ss.getPrimaryNetworkInterfaceConfiguration(networkInterfaceConfigurations, nodeName)
	if err != nil {
		return "", "", "", nil, err
	}

	// Find primary IP configuration.
	primaryIPConfiguration, err := getPrimaryIPConfigFromVMSSNetworkConfig(primaryNetworkInterfaceConfiguration)
	if err != nil {
		return "", "", "", nil, err
	}
	if primaryIPConfiguration.LoadBalancerBackendAddressPools == nil || len(*primaryIPConfiguration.LoadBalancerBackendAddressPools) == 0 {
		return "", "", "", nil, nil
	}

	// Construct new loadBalancerBackendAddressPools and remove backendAddressPools from primary IP configuration.
	existingBackendPools := *primaryIPConfiguration.LoadBalancerBackendAddressPools
	newBackendPools := []compute.SubResource{}
	foundPool := false
	for i := len(existingBackendPools) - 1; i >= 0; i-- {
		curPool := existingBackendPools[i]
		if strings.EqualFold(backendPoolID, *curPool.ID) {
			klog.V(10).Infof("ensureBackendPoolDeletedFromNode gets unwanted backend pool %q for node %s", backendPoolID, nodeName)
			foundPool = true
			newBackendPools = append(existingBackendPools[:i], existingBackendPools[i+1:]...)
		}
	}

	// Pool not found, assume it has been already removed.
	if !foundPool {
		return "", "", "", nil, nil
	}

	// Compose a new vmssVM with added backendPoolID.
	primaryIPConfiguration.LoadBalancerBackendAddressPools = &newBackendPools
	newVM := &compute.VirtualMachineScaleSetVM{
		Sku:      vm.Sku,
		Location: vm.Location,
		VirtualMachineScaleSetVMProperties: &compute.VirtualMachineScaleSetVMProperties{
			HardwareProfile: vm.HardwareProfile,
			NetworkProfileConfiguration: &compute.VirtualMachineScaleSetVMNetworkProfileConfiguration{
				NetworkInterfaceConfigurations: &networkInterfaceConfigurations,
			},
		},
	}

	// Get the node resource group.
	nodeResourceGroup, err := ss.GetNodeResourceGroup(nodeName)
	if err != nil {
		return "", "", "", nil, err
	}

	return nodeResourceGroup, ssName, instanceID, newVM, nil
}

// GetNodeNameByIPConfigurationID gets the node name and the VMSS name by IP configuration ID.
func (ss *scaleSet) GetNodeNameByIPConfigurationID(ipConfigurationID string) (string, string, error) {
	matches := vmssIPConfigurationRE.FindStringSubmatch(ipConfigurationID)
	if len(matches) != 4 {
		klog.V(4).Infof("Can not extract scale set name from ipConfigurationID (%s), assuming it is managed by availability set", ipConfigurationID)
		return "", "", ErrorNotVmssInstance
	}

	resourceGroup := matches[1]
	scaleSetName := matches[2]
	instanceID := matches[3]
	vm, err := ss.getVmssVMByInstanceID(resourceGroup, scaleSetName, instanceID, azcache.CacheReadTypeUnsafe)
	if err != nil {
		return "", "", err
	}

	if vm.OsProfile != nil && vm.OsProfile.ComputerName != nil {
		return strings.ToLower(*vm.OsProfile.ComputerName), scaleSetName, nil
	}

	return "", "", nil
}

func getScaleSetAndResourceGroupNameByIPConfigurationID(ipConfigurationID string) (string, string, error) {
	matches := vmssIPConfigurationRE.FindStringSubmatch(ipConfigurationID)
	if len(matches) != 4 {
		klog.V(4).Infof("Can not extract scale set name from ipConfigurationID (%s), assuming it is managed by availability set", ipConfigurationID)
		return "", "", ErrorNotVmssInstance
	}

	resourceGroup := matches[1]
	scaleSetName := matches[2]
	return scaleSetName, resourceGroup, nil
}

func (ss *scaleSet) ensureBackendPoolDeletedFromVMSS(service *v1.Service, backendPoolID, vmSetName string, ipConfigurationIDs []string) error {
	vmssNamesMap := make(map[string]bool)

	// the standard load balancer supports multiple vmss in its backend while the basic sku doesn't
	if ss.useStandardLoadBalancer() {
		for _, ipConfigurationID := range ipConfigurationIDs {
			// in this scenario the vmSetName is an empty string and the name of vmss should be obtained from the provider IDs of nodes
			vmssName, resourceGroupName, err := getScaleSetAndResourceGroupNameByIPConfigurationID(ipConfigurationID)
			if err != nil {
				klog.V(4).Infof("ensureBackendPoolDeletedFromVMSS: found VMAS ipcConfigurationID %s, will skip checking and continue", ipConfigurationID)
				continue
			}
			// only vmsses in the resource group same as it's in azure config are included
			if strings.EqualFold(resourceGroupName, ss.ResourceGroup) {
				vmssNamesMap[vmssName] = true
			}
		}
	} else {
		vmssNamesMap[vmSetName] = true
	}

	for vmssName := range vmssNamesMap {
		vmss, err := ss.getVMSS(vmssName, azcache.CacheReadTypeDefault)
		if err != nil {
			return err
		}

		// When vmss is being deleted, CreateOrUpdate API would report "the vmss is being deleted" error.
		// Since it is being deleted, we shouldn't send more CreateOrUpdate requests for it.
		if vmss.ProvisioningState != nil && strings.EqualFold(*vmss.ProvisioningState, virtualMachineScaleSetsDeallocating) {
			klog.V(3).Infof("ensureVMSSInPool: found vmss %s being deleted, skipping", vmssName)
			continue
		}
		if vmss.VirtualMachineProfile.NetworkProfile.NetworkInterfaceConfigurations == nil {
			klog.V(4).Infof("EnsureHostInPool: cannot obtain the primary network interface configuration, of vmss %s", vmssName)
			continue
		}
		vmssNIC := *vmss.VirtualMachineProfile.NetworkProfile.NetworkInterfaceConfigurations
		primaryNIC, err := ss.getPrimaryNetworkInterfaceConfigurationForScaleSet(vmssNIC, vmssName)
		if err != nil {
			return err
		}
		primaryIPConfig, err := getPrimaryIPConfigFromVMSSNetworkConfig(primaryNIC)
		if err != nil {
			return err
		}
		loadBalancerBackendAddressPools := []compute.SubResource{}
		if primaryIPConfig.LoadBalancerBackendAddressPools != nil {
			loadBalancerBackendAddressPools = *primaryIPConfig.LoadBalancerBackendAddressPools
		}

		var found bool
		var newBackendPools []compute.SubResource
		for i := len(loadBalancerBackendAddressPools) - 1; i >= 0; i-- {
			curPool := loadBalancerBackendAddressPools[i]
			if strings.EqualFold(backendPoolID, *curPool.ID) {
				klog.V(10).Infof("ensureBackendPoolDeletedFromVMSS gets unwanted backend pool %q for VMSS %s", backendPoolID, vmssName)
				found = true
				newBackendPools = append(loadBalancerBackendAddressPools[:i], loadBalancerBackendAddressPools[i+1:]...)
			}
		}
		if !found {
			continue
		}

		// Compose a new vmss with added backendPoolID.
		primaryIPConfig.LoadBalancerBackendAddressPools = &newBackendPools
		newVMSS := compute.VirtualMachineScaleSet{
			Sku:      vmss.Sku,
			Location: vmss.Location,
			VirtualMachineScaleSetProperties: &compute.VirtualMachineScaleSetProperties{
				VirtualMachineProfile: &compute.VirtualMachineScaleSetVMProfile{
					NetworkProfile: &compute.VirtualMachineScaleSetNetworkProfile{
						NetworkInterfaceConfigurations: &vmssNIC,
					},
				},
			},
		}

		klog.V(2).Infof("ensureBackendPoolDeletedFromVMSS begins to update vmss(%s) with backendPoolID %s", vmssName, backendPoolID)
		rerr := ss.CreateOrUpdateVMSS(ss.ResourceGroup, vmssName, newVMSS)
		if rerr != nil {
			klog.Errorf("ensureBackendPoolDeletedFromVMSS CreateOrUpdateVMSS(%s) with new backendPoolID %s, err: %v", vmssName, backendPoolID, err)
			return rerr.Error()
		}
	}

	return nil
}

// EnsureBackendPoolDeleted ensures the loadBalancer backendAddressPools deleted from the specified nodes.
func (ss *scaleSet) EnsureBackendPoolDeleted(service *v1.Service, backendPoolID, vmSetName string, backendAddressPools *[]network.BackendAddressPool) error {
	// Returns nil if backend address pools already deleted.
	if backendAddressPools == nil {
		return nil
	}

	mc := metrics.NewMetricContext("services", "vmss_ensure_backend_pool_deleted", ss.ResourceGroup, ss.SubscriptionID, service.Name)
	isOperationSucceeded := false
	defer func() {
		mc.ObserveOperationWithResult(isOperationSucceeded)
	}()

	ipConfigurationIDs := []string{}
	for _, backendPool := range *backendAddressPools {
		if strings.EqualFold(*backendPool.ID, backendPoolID) && backendPool.BackendIPConfigurations != nil {
			for _, ipConf := range *backendPool.BackendIPConfigurations {
				if ipConf.ID == nil {
					continue
				}

				ipConfigurationIDs = append(ipConfigurationIDs, *ipConf.ID)
			}
		}
	}

	hostUpdates := make([]func() error, 0, len(ipConfigurationIDs))
	nodeUpdates := make(map[vmssMetaInfo]map[string]compute.VirtualMachineScaleSetVM)
	errors := make([]error, 0)
	for i := range ipConfigurationIDs {
		ipConfigurationID := ipConfigurationIDs[i]

		var scaleSetName string
		var err error
		if scaleSetName, err = extractScaleSetNameByProviderID(ipConfigurationID); err == nil {
			// Only remove nodes belonging to specified vmSet to basic LB backends.
			if !ss.useStandardLoadBalancer() && !strings.EqualFold(scaleSetName, vmSetName) {
				continue
			}
		}

		nodeName, _, err := ss.GetNodeNameByIPConfigurationID(ipConfigurationID)
		if err != nil {
			if err == ErrorNotVmssInstance { // Do nothing for the VMAS nodes.
				continue
			}

			klog.Errorf("Failed to GetNodeNameByIPConfigurationID(%s): %v", ipConfigurationID, err)
			errors = append(errors, err)
			continue
		}

		nodeResourceGroup, nodeVMSS, nodeInstanceID, nodeVMSSVM, err := ss.ensureBackendPoolDeletedFromNode(nodeName, backendPoolID)
		if err != nil {
			klog.Errorf("EnsureBackendPoolDeleted(%s): backendPoolID(%s) - failed with error %v", getServiceName(service), backendPoolID, err)
			errors = append(errors, err)
			continue
		}

		// No need to update if nodeVMSSVM is nil.
		if nodeVMSSVM == nil {
			continue
		}

		nodeVMSSMetaInfo := vmssMetaInfo{vmssName: nodeVMSS, resourceGroup: nodeResourceGroup}
		if v, ok := nodeUpdates[nodeVMSSMetaInfo]; ok {
			v[nodeInstanceID] = *nodeVMSSVM
		} else {
			nodeUpdates[nodeVMSSMetaInfo] = map[string]compute.VirtualMachineScaleSetVM{
				nodeInstanceID: *nodeVMSSVM,
			}
		}

		// Invalidate the cache since the VMSS VM would be updated.
		defer ss.deleteCacheForNode(nodeName)
	}

	// Update VMs with best effort that have already been added to nodeUpdates.
	for meta, update := range nodeUpdates {
		// create new instance of meta and update for passing to anonymous function
		meta := meta
		update := update
		hostUpdates = append(hostUpdates, func() error {
			ctx, cancel := getContextWithCancel()
			defer cancel()
			klog.V(2).Infof("EnsureBackendPoolDeleted begins to UpdateVMs for VMSS(%s, %s) with backendPoolID %s", meta.resourceGroup, meta.vmssName, backendPoolID)
			rerr := ss.VirtualMachineScaleSetVMsClient.UpdateVMs(ctx, meta.resourceGroup, meta.vmssName, update, "network_update")
			if rerr != nil {
				klog.Errorf("EnsureBackendPoolDeleted UpdateVMs for VMSS(%s, %s) failed with error %v", meta.resourceGroup, meta.vmssName, rerr.Error())
				return rerr.Error()
			}

			return nil
		})
	}
	errs := utilerrors.AggregateGoroutines(hostUpdates...)
	if errs != nil {
		return utilerrors.Flatten(errs)
	}

	// Fail if there are other errors.
	if len(errors) > 0 {
		return utilerrors.Flatten(utilerrors.NewAggregate(errors))
	}

	// Ensure the backendPoolID is also deleted on VMSS itself.
	err := ss.ensureBackendPoolDeletedFromVMSS(service, backendPoolID, vmSetName, ipConfigurationIDs)
	if err != nil {
		return err
	}

	isOperationSucceeded = true
	return nil
}
