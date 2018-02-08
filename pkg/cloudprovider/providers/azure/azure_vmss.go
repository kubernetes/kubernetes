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
	"time"

	"github.com/Azure/azure-sdk-for-go/arm/compute"
	"github.com/Azure/azure-sdk-for-go/arm/network"
	"github.com/Azure/go-autorest/autorest/to"
	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

var (
	// ErrorNotVmssInstance indicates an instance is not belongint to any vmss.
	ErrorNotVmssInstance = errors.New("not a vmss instance")

	scaleSetNameRE = regexp.MustCompile(`.*/subscriptions/(?:.*)/Microsoft.Compute/virtualMachineScaleSets/(.+)/virtualMachines(?:.*)`)
)

// scaleSetVMInfo includes basic information of a virtual machine.
type scaleSetVMInfo struct {
	// The ID of the machine.
	ID string
	// Instance ID of the machine (only for scale sets vm).
	InstanceID string
	// Node name of the machine.
	NodeName string
	// Set name of the machine.
	ScaleSetName string
	// The type of the machine.
	Type string
	// The region of the machine.
	Region string
	// Primary interface ID of the machine.
	PrimaryInterfaceID string
	// Fault domain of the machine.
	FaultDomain string
}

// scaleSet implements VMSet interface for Azure scale set.
type scaleSet struct {
	*Cloud

	// availabilitySet is also required for scaleSet because some instances
	// (e.g. master nodes) may not belong to any scale sets.
	availabilitySet VMSet

	cacheMutex sync.Mutex
	// A local cache of scale sets. The key is scale set name and the value is a
	// list of virtual machines belonging to the scale set.
	cache                     map[string][]scaleSetVMInfo
	availabilitySetNodesCache sets.String
}

// newScaleSet creates a new scaleSet.
func newScaleSet(az *Cloud) VMSet {
	ss := &scaleSet{
		Cloud:                     az,
		availabilitySet:           newAvailabilitySet(az),
		availabilitySetNodesCache: sets.NewString(),
		cache: make(map[string][]scaleSetVMInfo),
	}

	go wait.Until(func() {
		ss.cacheMutex.Lock()
		defer ss.cacheMutex.Unlock()

		if err := ss.updateCache(); err != nil {
			glog.Errorf("updateCache failed: %v", err)
		}
	}, 5*time.Minute, wait.NeverStop)

	return ss
}

// updateCache updates scale sets cache. It should be called within a lock.
func (ss *scaleSet) updateCache() error {
	scaleSetNames, err := ss.listScaleSetsWithRetry()
	if err != nil {
		return err
	}

	localCache := make(map[string][]scaleSetVMInfo)
	for _, scaleSetName := range scaleSetNames {
		if _, ok := localCache[scaleSetName]; !ok {
			localCache[scaleSetName] = make([]scaleSetVMInfo, 0)
		}
		vms, err := ss.listScaleSetVMsWithRetry(scaleSetName)
		if err != nil {
			return err
		}

		for _, vm := range vms {
			nodeName := ""
			if vm.OsProfile != nil && vm.OsProfile.ComputerName != nil {
				nodeName = strings.ToLower(*vm.OsProfile.ComputerName)
			}

			vmSize := ""
			if vm.Sku != nil && vm.Sku.Name != nil {
				vmSize = *vm.Sku.Name
			}

			primaryInterfaceID, err := ss.getPrimaryInterfaceID(vm)
			if err != nil {
				glog.Errorf("getPrimaryInterfaceID for %s failed: %v", nodeName, err)
				return err
			}

			faultDomain := ""
			if vm.InstanceView != nil && vm.InstanceView.PlatformFaultDomain != nil {
				faultDomain = strconv.Itoa(int(*vm.InstanceView.PlatformFaultDomain))
			}

			localCache[scaleSetName] = append(localCache[scaleSetName], scaleSetVMInfo{
				ID:                 *vm.ID,
				Type:               vmSize,
				NodeName:           nodeName,
				FaultDomain:        faultDomain,
				ScaleSetName:       scaleSetName,
				Region:             *vm.Location,
				InstanceID:         *vm.InstanceID,
				PrimaryInterfaceID: primaryInterfaceID,
			})
		}
	}

	// Only update cache after all steps are success.
	ss.cache = localCache

	return nil
}

// getCachedVirtualMachine gets virtualMachine by nodeName from cache.
// It returns cloudprovider.InstanceNotFound if node does not belong to any scale sets.
func (ss *scaleSet) getCachedVirtualMachine(nodeName string) (scaleSetVMInfo, error) {
	ss.cacheMutex.Lock()
	defer ss.cacheMutex.Unlock()

	getVMFromCache := func(nodeName string) (scaleSetVMInfo, bool) {
		glog.V(8).Infof("Getting scaleSetVMInfo for %q from cache %v", nodeName, ss.cache)
		for scaleSetName := range ss.cache {
			for _, vm := range ss.cache[scaleSetName] {
				if vm.NodeName == nodeName {
					return vm, true
				}
			}
		}

		return scaleSetVMInfo{}, false
	}

	vm, found := getVMFromCache(nodeName)
	if found {
		return vm, nil
	}

	// Known node not managed by scale sets.
	if ss.availabilitySetNodesCache.Has(nodeName) {
		glog.V(10).Infof("Found node %q in availabilitySetNodesCache", nodeName)
		return scaleSetVMInfo{}, cloudprovider.InstanceNotFound
	}

	// Update cache and try again.
	glog.V(10).Infof("vmss cache before updateCache: %v", ss.cache)
	if err := ss.updateCache(); err != nil {
		glog.Errorf("updateCache failed with error: %v", err)
		return scaleSetVMInfo{}, err
	}
	glog.V(10).Infof("vmss cache after updateCache: %v", ss.cache)
	vm, found = getVMFromCache(nodeName)
	if found {
		return vm, nil
	}

	// Node still not found, assuming it is not managed by scale sets.
	glog.V(8).Infof("Node %q doesn't belong to any scale sets, adding it to availabilitySetNodesCache", nodeName)
	ss.availabilitySetNodesCache.Insert(nodeName)
	return scaleSetVMInfo{}, cloudprovider.InstanceNotFound
}

// getCachedVirtualMachineByInstanceID gets scaleSetVMInfo from cache.
// The node must belong to one of scale sets.
func (ss *scaleSet) getCachedVirtualMachineByInstanceID(scaleSetName, instanceID string) (scaleSetVMInfo, error) {
	ss.cacheMutex.Lock()
	defer ss.cacheMutex.Unlock()

	getVMByID := func(scaleSetName, instanceID string) (scaleSetVMInfo, bool) {
		glog.V(8).Infof("Getting scaleSetVMInfo with scaleSetName: %q and instanceID %q from cache %v", scaleSetName, instanceID, ss.cache)
		vms, ok := ss.cache[scaleSetName]
		if !ok {
			glog.V(4).Infof("scale set (%s) not found", scaleSetName)
			return scaleSetVMInfo{}, false
		}

		for _, vm := range vms {
			if vm.InstanceID == instanceID {
				glog.V(4).Infof("getCachedVirtualMachineByInstanceID gets vm (%s) by instanceID (%s) within scale set (%s)", vm.NodeName, instanceID, scaleSetName)
				return vm, true
			}
		}

		glog.V(4).Infof("instanceID (%s) not found in scale set (%s)", instanceID, scaleSetName)
		return scaleSetVMInfo{}, false
	}

	vm, found := getVMByID(scaleSetName, instanceID)
	if found {
		return vm, nil
	}

	// Update cache and try again.
	if err := ss.updateCache(); err != nil {
		glog.Errorf("updateCache failed with error: %v", err)
		return scaleSetVMInfo{}, err
	}
	vm, found = getVMByID(scaleSetName, instanceID)
	if found {
		return vm, nil
	}

	return scaleSetVMInfo{}, cloudprovider.InstanceNotFound
}

// GetInstanceIDByNodeName gets the cloud provider ID by node name.
// It must return ("", cloudprovider.InstanceNotFound) if the instance does
// not exist or is no longer running.
func (ss *scaleSet) GetInstanceIDByNodeName(name string) (string, error) {
	vm, err := ss.getCachedVirtualMachine(name)
	if err != nil {
		if err == cloudprovider.InstanceNotFound {
			glog.V(4).Infof("GetInstanceIDByNodeName: node %q is not found in scale sets, assuming it is managed by availability set", name)

			// Retry with standard type because master nodes may not belong to any vmss.
			// TODO: find a better way to identify the type of VM.
			return ss.availabilitySet.GetInstanceIDByNodeName(name)
		}

		return "", err
	}

	return vm.ID, nil
}

// GetNodeNameByProviderID gets the node name by provider ID.
func (ss *scaleSet) GetNodeNameByProviderID(providerID string) (types.NodeName, error) {
	// NodeName is not part of providerID for vmss instances.
	scaleSetName, err := extractScaleSetNameByVMID(providerID)
	if err != nil {
		glog.V(4).Infof("Can not extract scale set name from providerID (%s), assuming it is mananaged by availability set: %v", providerID, err)
		return ss.availabilitySet.GetNodeNameByProviderID(providerID)
	}

	instanceID, err := getLastSegment(providerID)
	if err != nil {
		glog.V(4).Infof("Can not extract instanceID from providerID (%s), assuming it is mananaged by availability set: %v", providerID, err)
		return ss.availabilitySet.GetNodeNameByProviderID(providerID)
	}

	vm, err := ss.getCachedVirtualMachineByInstanceID(scaleSetName, instanceID)
	if err != nil {
		return "", err
	}

	return types.NodeName(vm.NodeName), nil
}

// GetInstanceTypeByNodeName gets the instance type by node name.
func (ss *scaleSet) GetInstanceTypeByNodeName(name string) (string, error) {
	vm, err := ss.getCachedVirtualMachine(name)
	if err != nil {
		if err == cloudprovider.InstanceNotFound {
			glog.V(4).Infof("GetInstanceTypeByNodeName: node %q is not found in scale sets, assuming it is managed by availability set", name)

			// Retry with standard type because master nodes may not belong to any vmss.
			// TODO: find a better way to identify the type of VM.
			return ss.availabilitySet.GetInstanceTypeByNodeName(name)
		}

		return "", err
	}

	return vm.Type, nil
}

// GetZoneByNodeName gets cloudprovider.Zone by node name.
func (ss *scaleSet) GetZoneByNodeName(name string) (cloudprovider.Zone, error) {
	vm, err := ss.getCachedVirtualMachine(name)
	if err != nil {
		if err == cloudprovider.InstanceNotFound {
			glog.V(4).Infof("GetZoneByNodeName: node %q is not found in scale sets, assuming it is managed by availability set", name)
			// Retry with standard type because master nodes may not belong to any vmss.
			// TODO: find a better way to identify the type of VM.
			return ss.availabilitySet.GetZoneByNodeName(name)
		}
		return cloudprovider.Zone{}, err
	}

	return cloudprovider.Zone{
		FailureDomain: vm.FaultDomain,
		Region:        vm.Region,
	}, nil
}

// GetPrimaryVMSetName returns the VM set name depending on the configured vmType.
// It returns config.PrimaryScaleSetName for vmss and config.PrimaryAvailabilitySetName for standard vmType.
func (ss *scaleSet) GetPrimaryVMSetName() string {
	return ss.Config.PrimaryScaleSetName
}

// GetIPByNodeName gets machine IP by node name.
func (ss *scaleSet) GetIPByNodeName(nodeName, vmSetName string) (string, error) {
	nic, err := ss.GetPrimaryInterface(nodeName, vmSetName)
	if err != nil {
		glog.Errorf("error: ss.GetIPByNodeName(%s), GetPrimaryInterface(%q, %q), err=%v", nodeName, nodeName, vmSetName, err)
		return "", err
	}

	ipConfig, err := getPrimaryIPConfig(nic)
	if err != nil {
		glog.Errorf("error: ss.GetIPByNodeName(%s), getPrimaryIPConfig(%v), err=%v", nodeName, nic, err)
		return "", err
	}

	targetIP := *ipConfig.PrivateIPAddress
	return targetIP, nil
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

// extractScaleSetNameByVMID extracts the scaleset name by scaleSetVirtualMachine's ID.
func extractScaleSetNameByVMID(vmID string) (string, error) {
	matches := scaleSetNameRE.FindStringSubmatch(vmID)
	if len(matches) != 2 {
		return "", ErrorNotVmssInstance
	}

	return matches[1], nil
}

// listScaleSetsWithRetry lists scale sets with exponential backoff retry.
func (ss *scaleSet) listScaleSetsWithRetry() ([]string, error) {
	var err error
	var result compute.VirtualMachineScaleSetListResult
	allScaleSets := make([]string, 0)

	backoffError := wait.ExponentialBackoff(ss.requestBackoff(), func() (bool, error) {
		result, err = ss.VirtualMachineScaleSetsClient.List(ss.ResourceGroup)
		if err != nil {
			glog.Errorf("VirtualMachineScaleSetsClient.List for %v failed: %v", ss.ResourceGroup, err)
			return false, err
		}

		return true, nil
	})
	if backoffError != nil {
		return nil, backoffError
	}

	appendResults := (result.Value != nil && len(*result.Value) > 0)
	for appendResults {
		for _, scaleSet := range *result.Value {
			allScaleSets = append(allScaleSets, *scaleSet.Name)
		}
		appendResults = false

		if result.NextLink != nil {
			backoffError := wait.ExponentialBackoff(ss.requestBackoff(), func() (bool, error) {
				result, err = ss.VirtualMachineScaleSetsClient.ListNextResults(ss.ResourceGroup, result)
				if err != nil {
					glog.Errorf("VirtualMachineScaleSetsClient.ListNextResults for %v failed: %v", ss.ResourceGroup, err)
					return false, err
				}

				return true, nil
			})
			if backoffError != nil {
				return nil, backoffError
			}

			appendResults = (result.Value != nil && len(*result.Value) > 0)
		}

	}

	return allScaleSets, nil
}

// listScaleSetVMsWithRetry lists VMs belonging to the specified scale set with exponential backoff retry.
func (ss *scaleSet) listScaleSetVMsWithRetry(scaleSetName string) ([]compute.VirtualMachineScaleSetVM, error) {
	var err error
	var result compute.VirtualMachineScaleSetVMListResult
	allVMs := make([]compute.VirtualMachineScaleSetVM, 0)

	backoffError := wait.ExponentialBackoff(ss.requestBackoff(), func() (bool, error) {
		result, err = ss.VirtualMachineScaleSetVMsClient.List(ss.ResourceGroup, scaleSetName, "", "", string(compute.InstanceView))
		if err != nil {
			glog.Errorf("VirtualMachineScaleSetVMsClient.List for %v failed: %v", scaleSetName, err)
			return false, err
		}

		return true, nil
	})
	if backoffError != nil {
		return nil, backoffError
	}

	appendResults := (result.Value != nil && len(*result.Value) > 0)
	for appendResults {
		allVMs = append(allVMs, *result.Value...)
		appendResults = false

		if result.NextLink != nil {
			backoffError := wait.ExponentialBackoff(ss.requestBackoff(), func() (bool, error) {
				result, err = ss.VirtualMachineScaleSetVMsClient.ListNextResults(ss.ResourceGroup, result)
				if err != nil {
					glog.Errorf("VirtualMachineScaleSetVMsClient.ListNextResults for %v failed: %v", scaleSetName, err)
					return false, err
				}

				return true, nil
			})
			if backoffError != nil {
				return nil, backoffError
			}

			appendResults = (result.Value != nil && len(*result.Value) > 0)
		}

	}

	return allVMs, nil
}

// getAgentPoolScaleSets lists the virtual machines for for the resource group and then builds
// a list of scale sets that match the nodes available to k8s.
func (ss *scaleSet) getAgentPoolScaleSets(nodes []*v1.Node) (*[]string, error) {
	ss.cacheMutex.Lock()
	defer ss.cacheMutex.Unlock()

	// Always update cache to get latest lists of scale sets and virtual machines.
	if err := ss.updateCache(); err != nil {
		return nil, err
	}

	vmNameToScaleSetName := make(map[string]string)
	for scaleSetName := range ss.cache {
		vms := ss.cache[scaleSetName]
		for idx := range vms {
			vm := vms[idx]
			if vm.NodeName != "" {
				vmNameToScaleSetName[vm.NodeName] = scaleSetName
			}
		}
	}

	agentPoolScaleSets := &[]string{}
	availableScaleSetNames := sets.NewString()
	for nx := range nodes {
		if isMasterNode(nodes[nx]) {
			continue
		}

		nodeName := nodes[nx].Name
		ssName, ok := vmNameToScaleSetName[nodeName]
		if !ok {
			// TODO: support master nodes not managed by VMSS.
			glog.Errorf("Node %q is not belonging to any known scale sets", nodeName)
			return nil, fmt.Errorf("node %q is not belonging to any known scale sets", nodeName)
		}

		if availableScaleSetNames.Has(ssName) {
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
		glog.Errorf("ss.GetVMSetNames - getAgentPoolScaleSets failed err=(%v)", err)
		return nil, err
	}
	if len(*scaleSetNames) == 0 {
		glog.Errorf("ss.GetVMSetNames - No scale sets found for nodes in the cluster, node count(%d)", len(nodes))
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
				glog.Errorf("ss.GetVMSetNames - scale set (%s) in service annotation not found", serviceVMSetNames[sasx])
				return nil, fmt.Errorf("scale set (%s) - not found", serviceVMSetNames[sasx])
			}
		}
		vmSetNames = &serviceVMSetNames
	}

	return vmSetNames, nil
}

// GetPrimaryInterface gets machine primary network interface by node name and vmSet.
func (ss *scaleSet) GetPrimaryInterface(nodeName, vmSetName string) (network.Interface, error) {
	vm, err := ss.getCachedVirtualMachine(nodeName)
	if err != nil {
		if err == cloudprovider.InstanceNotFound {
			// Retry with standard type because master nodes may not belong to any vmss.
			// TODO: find a better way to identify the type of VM.
			return ss.availabilitySet.GetPrimaryInterface(nodeName, "")
		}

		glog.Errorf("error: ss.GetPrimaryInterface(%s), ss.getCachedVirtualMachine(%s), err=%v", nodeName, nodeName, err)
		return network.Interface{}, err
	}

	// Check scale set name.
	if vmSetName != "" && !strings.EqualFold(vm.ScaleSetName, vmSetName) {
		return network.Interface{}, errNotInVMSet
	}

	nicName, err := getLastSegment(vm.PrimaryInterfaceID)
	if err != nil {
		glog.Errorf("error: ss.GetPrimaryInterface(%s), getLastSegment(%s), err=%v", nodeName, vm.PrimaryInterfaceID, err)
		return network.Interface{}, err
	}

	nic, err := ss.InterfacesClient.GetVirtualMachineScaleSetNetworkInterface(ss.ResourceGroup, vm.ScaleSetName, vm.InstanceID, nicName, "")
	if err != nil {
		glog.Errorf("error: ss.GetPrimaryInterface(%s), ss.GetVirtualMachineScaleSetNetworkInterface.Get(%s, %s, %s), err=%v", nodeName, ss.ResourceGroup, vm.ScaleSetName, nicName, err)
		return network.Interface{}, err
	}

	// Fix interface's location, which is required when updating the interface.
	// TODO: is this a bug of azure SDK?
	if nic.Location == nil || *nic.Location == "" {
		nic.Location = &vm.Region
	}

	return nic, nil
}

// getScaleSet gets a scale set by name.
func (ss *scaleSet) getScaleSet(name string) (compute.VirtualMachineScaleSet, bool, error) {
	result, err := ss.VirtualMachineScaleSetsClient.Get(ss.ResourceGroup, name)
	exists, realErr := checkResourceExistsFromError(err)
	if realErr != nil {
		return result, false, realErr
	}

	if !exists {
		return result, false, nil
	}

	return result, exists, err
}

// getScaleSetWithRetry gets scale set with exponential backoff retry
func (ss *scaleSet) getScaleSetWithRetry(name string) (compute.VirtualMachineScaleSet, bool, error) {
	var result compute.VirtualMachineScaleSet
	var exists bool

	err := wait.ExponentialBackoff(ss.requestBackoff(), func() (bool, error) {
		var retryErr error
		result, exists, retryErr = ss.getScaleSet(name)
		if retryErr != nil {
			glog.Errorf("backoff: failure, will retry,err=%v", retryErr)
			return false, nil
		}
		glog.V(2).Info("backoff: success")
		return true, nil
	})

	return result, exists, err
}

// getPrimaryNetworkConfiguration gets primary network interface configuration for scale sets.
func (ss *scaleSet) getPrimaryNetworkConfiguration(networkConfigurationList *[]compute.VirtualMachineScaleSetNetworkConfiguration, scaleSetName string) (*compute.VirtualMachineScaleSetNetworkConfiguration, error) {
	networkConfigurations := *networkConfigurationList
	if len(networkConfigurations) == 1 {
		return &networkConfigurations[0], nil
	}

	for idx := range networkConfigurations {
		networkConfig := &networkConfigurations[idx]
		if networkConfig.Primary != nil && *networkConfig.Primary == true {
			return networkConfig, nil
		}
	}

	return nil, fmt.Errorf("failed to find a primary network configuration for the scale set %q", scaleSetName)
}

func (ss *scaleSet) getPrimaryIPConfigForScaleSet(config *compute.VirtualMachineScaleSetNetworkConfiguration, scaleSetName string) (*compute.VirtualMachineScaleSetIPConfiguration, error) {
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

	return nil, fmt.Errorf("failed to find a primary IP configuration for the scale set %q", scaleSetName)
}

// createOrUpdateVMSSWithRetry invokes ss.VirtualMachineScaleSetsClient.CreateOrUpdate with exponential backoff retry.
func (ss *scaleSet) createOrUpdateVMSSWithRetry(virtualMachineScaleSet compute.VirtualMachineScaleSet) error {
	return wait.ExponentialBackoff(ss.requestBackoff(), func() (bool, error) {
		respChan, errChan := ss.VirtualMachineScaleSetsClient.CreateOrUpdate(ss.ResourceGroup, *virtualMachineScaleSet.Name, virtualMachineScaleSet, nil)
		resp := <-respChan
		err := <-errChan
		glog.V(10).Infof("VirtualMachineScaleSetsClient.CreateOrUpdate(%s): end", *virtualMachineScaleSet.Name)
		return processRetryResponse(resp.Response, err)
	})
}

// updateVMSSInstancesWithRetry invokes ss.VirtualMachineScaleSetsClient.UpdateInstances with exponential backoff retry.
func (ss *scaleSet) updateVMSSInstancesWithRetry(scaleSetName string, vmInstanceIDs compute.VirtualMachineScaleSetVMInstanceRequiredIDs) error {
	return wait.ExponentialBackoff(ss.requestBackoff(), func() (bool, error) {
		respChan, errChan := ss.VirtualMachineScaleSetsClient.UpdateInstances(ss.ResourceGroup, scaleSetName, vmInstanceIDs, nil)
		resp := <-respChan
		err := <-errChan
		glog.V(10).Infof("VirtualMachineScaleSetsClient.UpdateInstances(%s): end", scaleSetName)
		return processRetryResponse(resp.Response, err)
	})
}

// EnsureHostsInPool ensures the given Node's primary IP configurations are
// participating in the specified LoadBalancer Backend Pool.
func (ss *scaleSet) EnsureHostsInPool(serviceName string, nodes []*v1.Node, backendPoolID string, vmSetName string) error {
	virtualMachineScaleSet, exists, err := ss.getScaleSetWithRetry(vmSetName)
	if err != nil {
		glog.Errorf("ss.getScaleSetWithRetry(%s) for service %q failed: %v", vmSetName, serviceName, err)
		return err
	}
	if !exists {
		errorMessage := fmt.Errorf("Scale set %q not found", vmSetName)
		glog.Errorf("%v", errorMessage)
		return errorMessage
	}

	// Find primary network interface configuration.
	networkConfigureList := virtualMachineScaleSet.VirtualMachineProfile.NetworkProfile.NetworkInterfaceConfigurations
	primaryNetworkConfiguration, err := ss.getPrimaryNetworkConfiguration(networkConfigureList, vmSetName)
	if err != nil {
		return err
	}

	// Find primary IP configuration.
	primaryIPConfiguration, err := ss.getPrimaryIPConfigForScaleSet(primaryNetworkConfiguration, vmSetName)
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
	if !foundPool {
		newBackendPools = append(newBackendPools,
			compute.SubResource{
				ID: to.StringPtr(backendPoolID),
			})
		primaryIPConfiguration.LoadBalancerBackendAddressPools = &newBackendPools

		glog.V(3).Infof("VirtualMachineScaleSetsClient.CreateOrUpdate for service (%s): scale set (%s) - updating", serviceName, vmSetName)
		respChan, errChan := ss.VirtualMachineScaleSetsClient.CreateOrUpdate(ss.ResourceGroup, vmSetName, virtualMachineScaleSet, nil)
		resp := <-respChan
		err := <-errChan
		glog.V(10).Infof("VirtualMachineScaleSetsClient.CreateOrUpdate(%q): end", vmSetName)
		if ss.CloudProviderBackoff && shouldRetryAPIRequest(resp.Response, err) {
			glog.V(2).Infof("VirtualMachineScaleSetsClient.CreateOrUpdate for service (%s): scale set (%s) - updating, err=%v", serviceName, vmSetName, err)
			retryErr := ss.createOrUpdateVMSSWithRetry(virtualMachineScaleSet)
			if retryErr != nil {
				err = retryErr
				glog.V(2).Infof("VirtualMachineScaleSetsClient.CreateOrUpdate for service (%s) abort backoff: scale set (%s) - updating", serviceName, vmSetName)
			}
		}
		if err != nil {
			return err
		}
	}

	// Construct instanceIDs from nodes.
	instanceIDs := []string{}
	for _, curNode := range nodes {
		curScaleSetName, err := extractScaleSetNameByVMID(curNode.Spec.ExternalID)
		if err != nil {
			glog.V(4).Infof("Node %q is not belonging to any scale sets, omitting it", curNode.Name)
			continue
		}
		if curScaleSetName != vmSetName {
			glog.V(4).Infof("Node %q is not belonging to scale set %q, omitting it", curNode.Name, vmSetName)
			continue
		}

		instanceID, err := getLastSegment(curNode.Spec.ExternalID)
		if err != nil {
			glog.Errorf("Failed to get last segment from %q: %v", curNode.Spec.ExternalID, err)
			return err
		}

		instanceIDs = append(instanceIDs, instanceID)
	}

	// Update instances to latest VMSS model.
	vmInstanceIDs := compute.VirtualMachineScaleSetVMInstanceRequiredIDs{
		InstanceIds: &instanceIDs,
	}
	respChan, errChan := ss.VirtualMachineScaleSetsClient.UpdateInstances(ss.ResourceGroup, vmSetName, vmInstanceIDs, nil)
	resp := <-respChan
	err = <-errChan
	glog.V(10).Infof("VirtualMachineScaleSetsClient.UpdateInstances(%q): end", vmSetName)
	if ss.CloudProviderBackoff && shouldRetryAPIRequest(resp.Response, err) {
		glog.V(2).Infof("VirtualMachineScaleSetsClient.UpdateInstances for service (%s): scale set (%s) - updating, err=%v", serviceName, vmSetName, err)
		retryErr := ss.updateVMSSInstancesWithRetry(vmSetName, vmInstanceIDs)
		if retryErr != nil {
			err = retryErr
			glog.V(2).Infof("VirtualMachineScaleSetsClient.UpdateInstances for service (%s) abort backoff: scale set (%s) - updating", serviceName, vmSetName)
		}
	}
	if err != nil {
		return err
	}

	return nil
}

// EnsureBackendPoolDeleted ensures the loadBalancer backendAddressPools deleted from the specified vmSet.
func (ss *scaleSet) EnsureBackendPoolDeleted(poolID, vmSetName string) error {
	virtualMachineScaleSet, exists, err := ss.getScaleSetWithRetry(vmSetName)
	if err != nil {
		glog.Errorf("ss.EnsureBackendPoolDeleted(%s, %s) getScaleSetWithRetry(%s) failed: %v", poolID, vmSetName, vmSetName, err)
		return err
	}
	if !exists {
		glog.V(2).Infof("ss.EnsureBackendPoolDeleted(%s, %s), scale set %s has already been non-exist", poolID, vmSetName, vmSetName)
		return nil
	}

	// Find primary network interface configuration.
	networkConfigureList := virtualMachineScaleSet.VirtualMachineProfile.NetworkProfile.NetworkInterfaceConfigurations
	primaryNetworkConfiguration, err := ss.getPrimaryNetworkConfiguration(networkConfigureList, vmSetName)
	if err != nil {
		return err
	}

	// Find primary IP configuration.
	primaryIPConfiguration, err := ss.getPrimaryIPConfigForScaleSet(primaryNetworkConfiguration, vmSetName)
	if err != nil {
		return err
	}

	// Construct new loadBalancerBackendAddressPools and remove backendAddressPools from primary IP configuration.
	if primaryIPConfiguration.LoadBalancerBackendAddressPools == nil || len(*primaryIPConfiguration.LoadBalancerBackendAddressPools) == 0 {
		return nil
	}
	existingBackendPools := *primaryIPConfiguration.LoadBalancerBackendAddressPools
	newBackendPools := []compute.SubResource{}
	foundPool := false
	for i := len(existingBackendPools) - 1; i >= 0; i-- {
		curPool := existingBackendPools[i]
		if strings.EqualFold(poolID, *curPool.ID) {
			glog.V(10).Infof("EnsureBackendPoolDeleted gets unwanted backend pool %q for scale set %q", poolID, vmSetName)
			foundPool = true
			newBackendPools = append(existingBackendPools[:i], existingBackendPools[i+1:]...)
		}
	}
	if !foundPool {
		// Pool not found, assume it has been already removed.
		return nil
	}

	// Update scale set with backoff.
	primaryIPConfiguration.LoadBalancerBackendAddressPools = &newBackendPools
	glog.V(3).Infof("VirtualMachineScaleSetsClient.CreateOrUpdate: scale set (%s) - updating", vmSetName)
	respChan, errChan := ss.VirtualMachineScaleSetsClient.CreateOrUpdate(ss.ResourceGroup, vmSetName, virtualMachineScaleSet, nil)
	resp := <-respChan
	err = <-errChan
	glog.V(10).Infof("VirtualMachineScaleSetsClient.CreateOrUpdate(%q): end", vmSetName)
	if ss.CloudProviderBackoff && shouldRetryAPIRequest(resp.Response, err) {
		glog.V(2).Infof("VirtualMachineScaleSetsClient.CreateOrUpdate: scale set (%s) - updating, err=%v", vmSetName, err)
		retryErr := ss.createOrUpdateVMSSWithRetry(virtualMachineScaleSet)
		if retryErr != nil {
			err = retryErr
			glog.V(2).Infof("VirtualMachineScaleSetsClient.CreateOrUpdate abort backoff: scale set (%s) - updating", vmSetName)
		}
	}
	if err != nil {
		return err
	}

	// Update instances to latest VMSS model.
	instanceIDs := []string{"*"}
	vmInstanceIDs := compute.VirtualMachineScaleSetVMInstanceRequiredIDs{
		InstanceIds: &instanceIDs,
	}
	updateRespChan, errChan := ss.VirtualMachineScaleSetsClient.UpdateInstances(ss.ResourceGroup, vmSetName, vmInstanceIDs, nil)
	updateResp := <-updateRespChan
	err = <-errChan
	glog.V(10).Infof("VirtualMachineScaleSetsClient.UpdateInstances(%q): end", vmSetName)
	if ss.CloudProviderBackoff && shouldRetryAPIRequest(updateResp.Response, err) {
		glog.V(2).Infof("VirtualMachineScaleSetsClient.UpdateInstances scale set (%s) - updating, err=%v", vmSetName, err)
		retryErr := ss.updateVMSSInstancesWithRetry(vmSetName, vmInstanceIDs)
		if retryErr != nil {
			err = retryErr
			glog.V(2).Infof("VirtualMachineScaleSetsClient.UpdateInstances abort backoff: scale set (%s) - updating", vmSetName)
		}
	}
	if err != nil {
		return err
	}

	// Update virtualMachineScaleSet again. This is a workaround for removing VMSS reference from LB.
	// TODO: remove this workaround when figuring out the root cause.
	if len(newBackendPools) == 0 {
		glog.V(3).Infof("VirtualMachineScaleSetsClient.CreateOrUpdate: scale set (%s) - updating second time", vmSetName)
		respChan, errChan = ss.VirtualMachineScaleSetsClient.CreateOrUpdate(ss.ResourceGroup, vmSetName, virtualMachineScaleSet, nil)
		resp = <-respChan
		err = <-errChan
		glog.V(10).Infof("VirtualMachineScaleSetsClient.CreateOrUpdate(%q): end", vmSetName)
		if ss.CloudProviderBackoff && shouldRetryAPIRequest(resp.Response, err) {
			glog.V(2).Infof("VirtualMachineScaleSetsClient.CreateOrUpdate: scale set (%s) - updating, err=%v", vmSetName, err)
			retryErr := ss.createOrUpdateVMSSWithRetry(virtualMachineScaleSet)
			if retryErr != nil {
				glog.V(2).Infof("VirtualMachineScaleSetsClient.CreateOrUpdate abort backoff: scale set (%s) - updating", vmSetName)
			}
		}
	}

	return nil
}
