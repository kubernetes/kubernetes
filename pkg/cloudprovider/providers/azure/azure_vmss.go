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

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-03-01/compute"
	"github.com/Azure/azure-sdk-for-go/services/network/mgmt/2017-09-01/network"
	"github.com/Azure/go-autorest/autorest/to"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/klog"
)

var (
	// ErrorNotVmssInstance indicates an instance is not belongint to any vmss.
	ErrorNotVmssInstance = errors.New("not a vmss instance")

	scaleSetNameRE         = regexp.MustCompile(`.*/subscriptions/(?:.*)/Microsoft.Compute/virtualMachineScaleSets/(.+)/virtualMachines(?:.*)`)
	resourceGroupRE        = regexp.MustCompile(`.*/subscriptions/(?:.*)/resourceGroups/(.+)/providers/Microsoft.Compute/virtualMachineScaleSets/(?:.*)/virtualMachines(?:.*)`)
	vmssNicResourceGroupRE = regexp.MustCompile(`.*/subscriptions/(?:.*)/resourceGroups/(.+)/providers/Microsoft.Compute/virtualMachineScaleSets/(?:.*)/virtualMachines/(?:.*)/networkInterfaces/(?:.*)`)
	vmssMachineIDTemplate  = "/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Compute/virtualMachineScaleSets/%s/virtualMachines/%s"
	vmssIPConfigurationRE  = regexp.MustCompile(`.*/subscriptions/(?:.*)/resourceGroups/(.+)/providers/Microsoft.Compute/virtualMachineScaleSets/(.+)/virtualMachines/(.+)/networkInterfaces(?:.*)`)
)

// scaleSet implements VMSet interface for Azure scale set.
type scaleSet struct {
	*Cloud

	// availabilitySet is also required for scaleSet because some instances
	// (e.g. master nodes) may not belong to any scale sets.
	availabilitySet VMSet

	vmssCache                      *timedCache
	vmssVMCache                    *timedCache
	nodeNameToScaleSetMappingCache *timedCache
	availabilitySetNodesCache      *timedCache
}

// newScaleSet creates a new scaleSet.
func newScaleSet(az *Cloud) (VMSet, error) {
	var err error
	ss := &scaleSet{
		Cloud:           az,
		availabilitySet: newAvailabilitySet(az),
	}

	ss.nodeNameToScaleSetMappingCache, err = ss.newNodeNameToScaleSetMappingCache()
	if err != nil {
		return nil, err
	}

	ss.availabilitySetNodesCache, err = ss.newAvailabilitySetNodesCache()
	if err != nil {
		return nil, err
	}

	ss.vmssCache, err = ss.newVmssCache()
	if err != nil {
		return nil, err
	}

	ss.vmssVMCache, err = ss.newVmssVMCache()
	if err != nil {
		return nil, err
	}

	return ss, nil
}

// getVmssVM gets virtualMachineScaleSetVM by nodeName from cache.
// It returns cloudprovider.InstanceNotFound if node does not belong to any scale sets.
func (ss *scaleSet) getVmssVM(nodeName string) (ssName, instanceID string, vm compute.VirtualMachineScaleSetVM, err error) {
	instanceID, err = getScaleSetVMInstanceID(nodeName)
	if err != nil {
		return ssName, instanceID, vm, err
	}

	ssName, err = ss.getScaleSetNameByNodeName(nodeName)
	if err != nil {
		return ssName, instanceID, vm, err
	}

	if ssName == "" {
		return "", "", vm, cloudprovider.InstanceNotFound
	}

	resourceGroup, err := ss.GetNodeResourceGroup(nodeName)
	if err != nil {
		return "", "", vm, err
	}

	klog.V(4).Infof("getVmssVM gets scaleSetName (%q) and instanceID (%q) for node %q", ssName, instanceID, nodeName)
	key := buildVmssCacheKey(resourceGroup, ss.makeVmssVMName(ssName, instanceID))
	cachedVM, err := ss.vmssVMCache.Get(key)
	if err != nil {
		return ssName, instanceID, vm, err
	}

	if cachedVM == nil {
		klog.Errorf("Can't find node (%q) in any scale sets", nodeName)
		return ssName, instanceID, vm, cloudprovider.InstanceNotFound
	}

	return ssName, instanceID, *(cachedVM.(*compute.VirtualMachineScaleSetVM)), nil
}

// GetPowerStatusByNodeName returns the power state of the specified node.
func (ss *scaleSet) GetPowerStatusByNodeName(name string) (powerState string, err error) {
	_, _, vm, err := ss.getVmssVM(name)
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

	return "", fmt.Errorf("failed to get power status for node %q", name)
}

// getCachedVirtualMachineByInstanceID gets scaleSetVMInfo from cache.
// The node must belong to one of scale sets.
func (ss *scaleSet) getVmssVMByInstanceID(resourceGroup, scaleSetName, instanceID string) (vm compute.VirtualMachineScaleSetVM, err error) {
	vmName := ss.makeVmssVMName(scaleSetName, instanceID)
	key := buildVmssCacheKey(resourceGroup, vmName)
	cachedVM, err := ss.vmssVMCache.Get(key)
	if err != nil {
		return vm, err
	}

	if cachedVM == nil {
		klog.Errorf("couldn't find vmss virtual machine by scaleSetName (%s) and instanceID (%s)", scaleSetName, instanceID)
		return vm, cloudprovider.InstanceNotFound
	}

	return *(cachedVM.(*compute.VirtualMachineScaleSetVM)), nil
}

// GetInstanceIDByNodeName gets the cloud provider ID by node name.
// It must return ("", cloudprovider.InstanceNotFound) if the instance does
// not exist or is no longer running.
func (ss *scaleSet) GetInstanceIDByNodeName(name string) (string, error) {
	managedByAS, err := ss.isNodeManagedByAvailabilitySet(name)
	if err != nil {
		klog.Errorf("Failed to check isNodeManagedByAvailabilitySet: %v", err)
		return "", err
	}
	if managedByAS {
		// vm is managed by availability set.
		return ss.availabilitySet.GetInstanceIDByNodeName(name)
	}

	_, _, vm, err := ss.getVmssVM(name)
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
func (ss *scaleSet) GetNodeNameByProviderID(providerID string) (types.NodeName, error) {
	// NodeName is not part of providerID for vmss instances.
	scaleSetName, err := extractScaleSetNameByProviderID(providerID)
	if err != nil {
		klog.V(4).Infof("Can not extract scale set name from providerID (%s), assuming it is mananaged by availability set: %v", providerID, err)
		return ss.availabilitySet.GetNodeNameByProviderID(providerID)
	}

	resourceGroup, err := extractResourceGroupByProviderID(providerID)
	if err != nil {
		return "", fmt.Errorf("error of extracting resource group for node %q", providerID)
	}

	instanceID, err := getLastSegment(providerID)
	if err != nil {
		klog.V(4).Infof("Can not extract instanceID from providerID (%s), assuming it is mananaged by availability set: %v", providerID, err)
		return ss.availabilitySet.GetNodeNameByProviderID(providerID)
	}

	vm, err := ss.getVmssVMByInstanceID(resourceGroup, scaleSetName, instanceID)
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
	managedByAS, err := ss.isNodeManagedByAvailabilitySet(name)
	if err != nil {
		klog.Errorf("Failed to check isNodeManagedByAvailabilitySet: %v", err)
		return "", err
	}
	if managedByAS {
		// vm is managed by availability set.
		return ss.availabilitySet.GetInstanceTypeByNodeName(name)
	}

	_, _, vm, err := ss.getVmssVM(name)
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
	managedByAS, err := ss.isNodeManagedByAvailabilitySet(name)
	if err != nil {
		klog.Errorf("Failed to check isNodeManagedByAvailabilitySet: %v", err)
		return cloudprovider.Zone{}, err
	}
	if managedByAS {
		// vm is managed by availability set.
		return ss.availabilitySet.GetZoneByNodeName(name)
	}

	_, _, vm, err := ss.getVmssVM(name)
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

		failureDomain = ss.makeZone(zoneID)
	} else if vm.InstanceView != nil && vm.InstanceView.PlatformFaultDomain != nil {
		// Availability zone is not used for the node, falling back to fault domain.
		failureDomain = strconv.Itoa(int(*vm.InstanceView.PlatformFaultDomain))
	}

	return cloudprovider.Zone{
		FailureDomain: failureDomain,
		Region:        *vm.Location,
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
		pipName, err := getLastSegment(pipID)
		if err != nil {
			return "", "", fmt.Errorf("failed to get publicIP name for node %q with pipID %q", nodeName, pipID)
		}

		resourceGroup, err := ss.GetNodeResourceGroup(nodeName)
		if err != nil {
			return "", "", err
		}

		pip, existsPip, err := ss.getPublicIPAddress(resourceGroup, pipName)
		if err != nil {
			return "", "", err
		}
		if existsPip {
			publicIP = *pip.IPAddress
		}
	}

	return internalIP, publicIP, nil
}

// This returns the full identifier of the primary NIC for the given VM.
func (ss *scaleSet) getPrimaryInterfaceID(machine compute.VirtualMachineScaleSetVM) (string, error) {
	if len(*machine.NetworkProfile.NetworkInterfaces) == 1 {
		return *(*machine.NetworkProfile.NetworkInterfaces)[0].ID, nil
	}

	for _, ref := range *machine.NetworkProfile.NetworkInterfaces {
		if *ref.Primary {
			return *ref.ID, nil
		}
	}

	return "", fmt.Errorf("failed to find a primary nic for the vm. vmname=%q", *machine.Name)
}

// getVmssMachineID returns the full identifier of a vmss virtual machine.
func (az *Cloud) getVmssMachineID(resourceGroup, scaleSetName, instanceID string) string {
	return fmt.Sprintf(
		vmssMachineIDTemplate,
		az.SubscriptionID,
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

// listScaleSets lists all scale sets.
func (ss *scaleSet) listScaleSets(resourceGroup string) ([]string, error) {
	var err error
	ctx, cancel := getContextWithCancel()
	defer cancel()

	allScaleSets, err := ss.VirtualMachineScaleSetsClient.List(ctx, resourceGroup)
	if err != nil {
		klog.Errorf("VirtualMachineScaleSetsClient.List failed: %v", err)
		return nil, err
	}

	ssNames := make([]string, len(allScaleSets))
	for i := range allScaleSets {
		ssNames[i] = *(allScaleSets[i].Name)
	}

	return ssNames, nil
}

// listScaleSetVMs lists VMs belonging to the specified scale set.
func (ss *scaleSet) listScaleSetVMs(scaleSetName, resourceGroup string) ([]compute.VirtualMachineScaleSetVM, error) {
	var err error
	ctx, cancel := getContextWithCancel()
	defer cancel()

	allVMs, err := ss.VirtualMachineScaleSetVMsClient.List(ctx, resourceGroup, scaleSetName, "", "", string(compute.InstanceView))
	if err != nil {
		klog.Errorf("VirtualMachineScaleSetVMsClient.List failed: %v", err)
		return nil, err
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
		ssName, err := ss.getScaleSetNameByNodeName(nodeName)
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
	if !hasMode {
		// no mode specified in service annotation default to PrimaryScaleSetName.
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
		return nil, fmt.Errorf("No scale sets found for nodes, node count(%d)", len(nodes))
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
	matches := vmssNicResourceGroupRE.FindStringSubmatch(nicID)
	if len(matches) != 2 {
		return "", fmt.Errorf("error of extracting resourceGroup from nicID %q", nicID)
	}

	return matches[1], nil
}

// GetPrimaryInterface gets machine primary network interface by node name and vmSet.
func (ss *scaleSet) GetPrimaryInterface(nodeName string) (network.Interface, error) {
	managedByAS, err := ss.isNodeManagedByAvailabilitySet(nodeName)
	if err != nil {
		klog.Errorf("Failed to check isNodeManagedByAvailabilitySet: %v", err)
		return network.Interface{}, err
	}
	if managedByAS {
		// vm is managed by availability set.
		return ss.availabilitySet.GetPrimaryInterface(nodeName)
	}

	ssName, instanceID, vm, err := ss.getVmssVM(nodeName)
	if err != nil {
		// VM is availability set, but not cached yet in availabilitySetNodesCache.
		if err == ErrorNotVmssInstance {
			return ss.availabilitySet.GetPrimaryInterface(nodeName)
		}

		klog.Errorf("error: ss.GetPrimaryInterface(%s), ss.getVmssVM(%s), err=%v", nodeName, nodeName, err)
		return network.Interface{}, err
	}

	primaryInterfaceID, err := ss.getPrimaryInterfaceID(vm)
	if err != nil {
		klog.Errorf("error: ss.GetPrimaryInterface(%s), ss.getPrimaryInterfaceID(), err=%v", nodeName, err)
		return network.Interface{}, err
	}

	nicName, err := getLastSegment(primaryInterfaceID)
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
	nic, err := ss.InterfacesClient.GetVirtualMachineScaleSetNetworkInterface(ctx, resourceGroup, ssName, instanceID, nicName, "")
	if err != nil {
		klog.Errorf("error: ss.GetPrimaryInterface(%s), ss.GetVirtualMachineScaleSetNetworkInterface.Get(%s, %s, %s), err=%v", nodeName, resourceGroup, ssName, nicName, err)
		return network.Interface{}, err
	}

	// Fix interface's location, which is required when updating the interface.
	// TODO: is this a bug of azure SDK?
	if nic.Location == nil || *nic.Location == "" {
		nic.Location = vm.Location
	}

	return nic, nil
}

// getPrimarynetworkInterfaceConfiguration gets primary network interface configuration for scale set virtual machine.
func (ss *scaleSet) getPrimarynetworkInterfaceConfiguration(networkConfigurations []compute.VirtualMachineScaleSetNetworkConfiguration, nodeName string) (*compute.VirtualMachineScaleSetNetworkConfiguration, error) {
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

func (ss *scaleSet) getPrimaryIPConfigForScaleSet(config *compute.VirtualMachineScaleSetNetworkConfiguration, nodeName string) (*compute.VirtualMachineScaleSetIPConfiguration, error) {
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

	return nil, fmt.Errorf("failed to find a primary IP configuration for the scale set VM %q", nodeName)
}

// EnsureHostInPool ensures the given VM's Primary NIC's Primary IP Configuration is
// participating in the specified LoadBalancer Backend Pool.
func (ss *scaleSet) EnsureHostInPool(service *v1.Service, nodeName types.NodeName, backendPoolID string, vmSetName string, isInternal bool) error {
	klog.V(3).Infof("ensuring node %q of scaleset %q in LB backendpool %q", nodeName, vmSetName, backendPoolID)
	vmName := mapNodeNameToVMName(nodeName)
	ssName, instanceID, vm, err := ss.getVmssVM(vmName)
	if err != nil {
		return err
	}

	// Check scale set name:
	// - For basic SKU load balancer, errNotInVMSet should be returned if the node's
	//   scale set is mismatched with vmSetName.
	// - For standard SKU load balancer, backend could belong to multiple VMSS, so we
	//   don't check vmSet for it.
	if vmSetName != "" && !ss.useStandardLoadBalancer() && !strings.EqualFold(vmSetName, ssName) {
		klog.V(3).Infof("EnsureHostInPool skips node %s because it is not in the scaleSet %s", vmName, vmSetName)
		return nil
	}

	// Find primary network interface configuration.
	networkInterfaceConfigurations := *vm.NetworkProfileConfiguration.NetworkInterfaceConfigurations
	primaryNetworkInterfaceConfiguration, err := ss.getPrimarynetworkInterfaceConfiguration(networkInterfaceConfigurations, vmName)
	if err != nil {
		return err
	}

	// Find primary IP configuration.
	primaryIPConfiguration, err := ss.getPrimaryIPConfigForScaleSet(primaryNetworkInterfaceConfiguration, vmName)
	if err != nil {
		return err
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
		return nil
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
			return err
		}
		if !isSameLB {
			klog.V(4).Infof("Node %q has already been added to LB %q, omit adding it to a new one", nodeName, oldLBName)
			return nil
		}
	}

	// Compose a new vmssVM with added backendPoolID.
	newBackendPools = append(newBackendPools,
		compute.SubResource{
			ID: to.StringPtr(backendPoolID),
		})
	primaryIPConfiguration.LoadBalancerBackendAddressPools = &newBackendPools
	newVM := compute.VirtualMachineScaleSetVM{
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
		return err
	}

	// Invalidate the cache since we would update it.
	key := buildVmssCacheKey(nodeResourceGroup, ss.makeVmssVMName(ssName, instanceID))
	defer ss.vmssVMCache.Delete(key)

	// Update vmssVM with backoff.
	ctx, cancel := getContextWithCancel()
	defer cancel()
	klog.V(2).Infof("EnsureHostInPool begins to update vmssVM(%s) with new backendPoolID %s", vmName, backendPoolID)
	resp, err := ss.VirtualMachineScaleSetVMsClient.Update(ctx, nodeResourceGroup, ssName, instanceID, newVM)
	if ss.CloudProviderBackoff && shouldRetryHTTPRequest(resp, err) {
		klog.V(2).Infof("EnsureHostInPool update backing off vmssVM(%s) with new backendPoolID %s, err: %v", vmName, backendPoolID, err)
		retryErr := ss.UpdateVmssVMWithRetry(nodeResourceGroup, ssName, instanceID, newVM)
		if retryErr != nil {
			err = retryErr
			klog.Errorf("EnsureHostInPool update abort backoff vmssVM(%s) with new backendPoolID %s, err: %v", vmName, backendPoolID, err)
		}
	}

	return err
}

// EnsureHostsInPool ensures the given Node's primary IP configurations are
// participating in the specified LoadBalancer Backend Pool.
func (ss *scaleSet) EnsureHostsInPool(service *v1.Service, nodes []*v1.Node, backendPoolID string, vmSetName string, isInternal bool) error {
	hostUpdates := make([]func() error, 0, len(nodes))
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

		f := func() error {
			// VMAS nodes should also be added to the SLB backends.
			if ss.useStandardLoadBalancer() {
				// Check whether the node is VMAS virtual machine.
				managedByAS, err := ss.isNodeManagedByAvailabilitySet(localNodeName)
				if err != nil {
					klog.Errorf("Failed to check isNodeManagedByAvailabilitySet(%s): %v", localNodeName, err)
					return err
				}
				if managedByAS {
					return ss.availabilitySet.EnsureHostInPool(service, types.NodeName(localNodeName), backendPoolID, vmSetName, isInternal)
				}
			}

			err := ss.EnsureHostInPool(service, types.NodeName(localNodeName), backendPoolID, vmSetName, isInternal)
			if err != nil {
				return fmt.Errorf("EnsureHostInPool(%s): backendPoolID(%s) - failed to ensure host in pool: %q", getServiceName(service), backendPoolID, err)
			}
			return nil
		}
		hostUpdates = append(hostUpdates, f)
	}

	errs := utilerrors.AggregateGoroutines(hostUpdates...)
	if errs != nil {
		return utilerrors.Flatten(errs)
	}

	return nil
}

// ensureBackendPoolDeletedFromNode ensures the loadBalancer backendAddressPools deleted from the specified node.
func (ss *scaleSet) ensureBackendPoolDeletedFromNode(service *v1.Service, nodeName, backendPoolID string) error {
	ssName, instanceID, vm, err := ss.getVmssVM(nodeName)
	if err != nil {
		return err
	}

	// Find primary network interface configuration.
	networkInterfaceConfigurations := *vm.NetworkProfileConfiguration.NetworkInterfaceConfigurations
	primaryNetworkInterfaceConfiguration, err := ss.getPrimarynetworkInterfaceConfiguration(networkInterfaceConfigurations, nodeName)
	if err != nil {
		return err
	}

	// Find primary IP configuration.4
	primaryIPConfiguration, err := ss.getPrimaryIPConfigForScaleSet(primaryNetworkInterfaceConfiguration, nodeName)
	if err != nil {
		return err
	}
	if primaryIPConfiguration.LoadBalancerBackendAddressPools == nil || len(*primaryIPConfiguration.LoadBalancerBackendAddressPools) == 0 {
		return nil
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
		return nil
	}

	// Compose a new vmssVM with added backendPoolID.
	primaryIPConfiguration.LoadBalancerBackendAddressPools = &newBackendPools
	newVM := compute.VirtualMachineScaleSetVM{
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
		return err
	}

	// Invalidate the cache since we would update it.
	key := buildVmssCacheKey(nodeResourceGroup, ss.makeVmssVMName(ssName, instanceID))
	defer ss.vmssVMCache.Delete(key)

	// Update vmssVM with backoff.
	ctx, cancel := getContextWithCancel()
	defer cancel()
	klog.V(2).Infof("ensureBackendPoolDeletedFromNode begins to update vmssVM(%s) with backendPoolID %s", nodeName, backendPoolID)
	resp, err := ss.VirtualMachineScaleSetVMsClient.Update(ctx, nodeResourceGroup, ssName, instanceID, newVM)
	if ss.CloudProviderBackoff && shouldRetryHTTPRequest(resp, err) {
		klog.V(2).Infof("ensureBackendPoolDeletedFromNode update backing off vmssVM(%s) with backendPoolID %s, err: %v", nodeName, backendPoolID, err)
		retryErr := ss.UpdateVmssVMWithRetry(nodeResourceGroup, ssName, instanceID, newVM)
		if retryErr != nil {
			err = retryErr
			klog.Errorf("ensureBackendPoolDeletedFromNode update abort backoff vmssVM(%s) with backendPoolID %s, err: %v", nodeName, backendPoolID, err)
		}
	}
	if err != nil {
		klog.Errorf("ensureBackendPoolDeletedFromNode failed to update vmssVM(%s) with backendPoolID %s: %v", nodeName, backendPoolID, err)
	} else {
		klog.V(2).Infof("ensureBackendPoolDeletedFromNode update vmssVM(%s) with backendPoolID %s succeeded", nodeName, backendPoolID)
	}
	return err
}

// getNodeNameByIPConfigurationID gets the node name by IP configuration ID.
func (ss *scaleSet) getNodeNameByIPConfigurationID(ipConfigurationID string) (string, error) {
	matches := vmssIPConfigurationRE.FindStringSubmatch(ipConfigurationID)
	if len(matches) != 4 {
		klog.V(4).Infof("Can not extract scale set name from ipConfigurationID (%s), assuming it is mananaged by availability set", ipConfigurationID)
		return "", ErrorNotVmssInstance
	}

	resourceGroup := matches[1]
	scaleSetName := matches[2]
	instanceID := matches[3]
	vm, err := ss.getVmssVMByInstanceID(resourceGroup, scaleSetName, instanceID)
	if err != nil {
		return "", err
	}

	if vm.OsProfile != nil && vm.OsProfile.ComputerName != nil {
		return strings.ToLower(*vm.OsProfile.ComputerName), nil
	}

	return "", nil
}

// EnsureBackendPoolDeleted ensures the loadBalancer backendAddressPools deleted from the specified nodes.
func (ss *scaleSet) EnsureBackendPoolDeleted(service *v1.Service, backendPoolID, vmSetName string, backendAddressPools *[]network.BackendAddressPool) error {
	// Returns nil if backend address pools already deleted.
	if backendAddressPools == nil {
		return nil
	}

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
	for i := range ipConfigurationIDs {
		ipConfigurationID := ipConfigurationIDs[i]

		f := func() error {
			if scaleSetName, err := extractScaleSetNameByProviderID(ipConfigurationID); err == nil {
				// Only remove nodes belonging to specified vmSet to basic LB backends.
				if !ss.useStandardLoadBalancer() && !strings.EqualFold(scaleSetName, vmSetName) {
					return nil
				}
			}

			nodeName, err := ss.getNodeNameByIPConfigurationID(ipConfigurationID)
			if err != nil {
				if err == ErrorNotVmssInstance { // Do nothing for the VMAS nodes.
					return nil
				}
				klog.Errorf("Failed to getNodeNameByIPConfigurationID(%s): %v", ipConfigurationID, err)
				return err
			}

			err = ss.ensureBackendPoolDeletedFromNode(service, nodeName, backendPoolID)
			if err != nil {
				return fmt.Errorf("failed to ensure backend pool %s deleted from node %s: %v", backendPoolID, nodeName, err)
			}

			return nil
		}
		hostUpdates = append(hostUpdates, f)
	}

	errs := utilerrors.AggregateGoroutines(hostUpdates...)
	if errs != nil {
		return utilerrors.Flatten(errs)
	}

	return nil
}
