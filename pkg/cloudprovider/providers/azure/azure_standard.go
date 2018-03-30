/*
Copyright 2016 The Kubernetes Authors.

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
	"hash/crc32"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/cloudprovider"

	"github.com/Azure/azure-sdk-for-go/arm/compute"
	"github.com/Azure/azure-sdk-for-go/arm/network"
	"github.com/Azure/go-autorest/autorest/to"
	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
)

const (
	loadBalancerMinimumPriority = 500
	loadBalancerMaximumPriority = 4096

	machineIDTemplate           = "/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Compute/virtualMachines/%s"
	availabilitySetIDTemplate   = "/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Compute/availabilitySets/%s"
	frontendIPConfigIDTemplate  = "/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Network/loadBalancers/%s/frontendIPConfigurations/%s"
	backendPoolIDTemplate       = "/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Network/loadBalancers/%s/backendAddressPools/%s"
	loadBalancerProbeIDTemplate = "/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Network/loadBalancers/%s/probes/%s"

	// InternalLoadBalancerNameSuffix is load balancer posfix
	InternalLoadBalancerNameSuffix = "-internal"

	// nodeLabelRole specifies the role of a node
	nodeLabelRole = "kubernetes.io/role"

	storageAccountNameMaxLength = 24
)

var errNotInVMSet = errors.New("vm is not in the vmset")
var providerIDRE = regexp.MustCompile(`^` + CloudProviderName + `://(?:.*)/Microsoft.Compute/virtualMachines/(.+)$`)

// getStandardMachineID returns the full identifier of a virtual machine.
func (az *Cloud) getStandardMachineID(machineName string) string {
	return fmt.Sprintf(
		machineIDTemplate,
		az.SubscriptionID,
		az.ResourceGroup,
		machineName)
}

// returns the full identifier of an availabilitySet
func (az *Cloud) getAvailabilitySetID(availabilitySetName string) string {
	return fmt.Sprintf(
		availabilitySetIDTemplate,
		az.SubscriptionID,
		az.ResourceGroup,
		availabilitySetName)
}

// returns the full identifier of a loadbalancer frontendipconfiguration.
func (az *Cloud) getFrontendIPConfigID(lbName, backendPoolName string) string {
	return fmt.Sprintf(
		frontendIPConfigIDTemplate,
		az.SubscriptionID,
		az.ResourceGroup,
		lbName,
		backendPoolName)
}

// returns the full identifier of a loadbalancer backendpool.
func (az *Cloud) getBackendPoolID(lbName, backendPoolName string) string {
	return fmt.Sprintf(
		backendPoolIDTemplate,
		az.SubscriptionID,
		az.ResourceGroup,
		lbName,
		backendPoolName)
}

// returns the full identifier of a loadbalancer probe.
func (az *Cloud) getLoadBalancerProbeID(lbName, lbRuleName string) string {
	return fmt.Sprintf(
		loadBalancerProbeIDTemplate,
		az.SubscriptionID,
		az.ResourceGroup,
		lbName,
		lbRuleName)
}

func (az *Cloud) mapLoadBalancerNameToVMSet(lbName string, clusterName string) (vmSetName string) {
	vmSetName = strings.TrimSuffix(lbName, InternalLoadBalancerNameSuffix)
	if strings.EqualFold(clusterName, vmSetName) {
		vmSetName = az.vmSet.GetPrimaryVMSetName()
	}

	return vmSetName
}

// For a load balancer, all frontend ip should reference either a subnet or publicIpAddress.
// Thus Azure do not allow mixed type (public and internal) load balancer.
// So we'd have a separate name for internal load balancer.
// This would be the name for Azure LoadBalancer resource.
func (az *Cloud) getLoadBalancerName(clusterName string, vmSetName string, isInternal bool) string {
	lbNamePrefix := vmSetName
	if strings.EqualFold(vmSetName, az.vmSet.GetPrimaryVMSetName()) {
		lbNamePrefix = clusterName
	}
	if isInternal {
		return fmt.Sprintf("%s%s", lbNamePrefix, InternalLoadBalancerNameSuffix)
	}
	return lbNamePrefix
}

// isMasterNode returns true if the node has a master role label.
// The master role is determined by looking for:
// * a kubernetes.io/role="master" label
func isMasterNode(node *v1.Node) bool {
	if val, ok := node.Labels[nodeLabelRole]; ok && val == "master" {
		return true
	}

	return false
}

// returns the deepest child's identifier from a full identifier string.
func getLastSegment(ID string) (string, error) {
	parts := strings.Split(ID, "/")
	name := parts[len(parts)-1]
	if len(name) == 0 {
		return "", fmt.Errorf("resource name was missing from identifier")
	}

	return name, nil
}

// returns the equivalent LoadBalancerRule, SecurityRule and LoadBalancerProbe
// protocol types for the given Kubernetes protocol type.
func getProtocolsFromKubernetesProtocol(protocol v1.Protocol) (*network.TransportProtocol, *network.SecurityRuleProtocol, *network.ProbeProtocol, error) {
	var transportProto network.TransportProtocol
	var securityProto network.SecurityRuleProtocol
	var probeProto network.ProbeProtocol

	switch protocol {
	case v1.ProtocolTCP:
		transportProto = network.TransportProtocolTCP
		securityProto = network.SecurityRuleProtocolTCP
		probeProto = network.ProbeProtocolTCP
		return &transportProto, &securityProto, &probeProto, nil
	case v1.ProtocolUDP:
		transportProto = network.TransportProtocolUDP
		securityProto = network.SecurityRuleProtocolUDP
		return &transportProto, &securityProto, nil, nil
	default:
		return &transportProto, &securityProto, &probeProto, fmt.Errorf("Only TCP and UDP are supported for Azure LoadBalancers")
	}

}

// This returns the full identifier of the primary NIC for the given VM.
func getPrimaryInterfaceID(machine compute.VirtualMachine) (string, error) {
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

func getPrimaryIPConfig(nic network.Interface) (*network.InterfaceIPConfiguration, error) {
	if nic.IPConfigurations == nil {
		return nil, fmt.Errorf("nic.IPConfigurations for nic (nicname=%q) is nil", *nic.Name)
	}

	if len(*nic.IPConfigurations) == 1 {
		return &((*nic.IPConfigurations)[0]), nil
	}

	for _, ref := range *nic.IPConfigurations {
		if *ref.Primary {
			return &ref, nil
		}
	}

	return nil, fmt.Errorf("failed to determine the primary ipconfig. nicname=%q", *nic.Name)
}

func isInternalLoadBalancer(lb *network.LoadBalancer) bool {
	return strings.HasSuffix(*lb.Name, InternalLoadBalancerNameSuffix)
}

func getBackendPoolName(clusterName string) string {
	return clusterName
}

func getLoadBalancerRuleName(service *v1.Service, port v1.ServicePort, subnetName *string) string {
	if subnetName == nil {
		return fmt.Sprintf("%s-%s-%d", getRulePrefix(service), port.Protocol, port.Port)
	}
	return fmt.Sprintf("%s-%s-%s-%d", getRulePrefix(service), *subnetName, port.Protocol, port.Port)
}

func getSecurityRuleName(service *v1.Service, port v1.ServicePort, sourceAddrPrefix string) string {
	if useSharedSecurityRule(service) {
		safePrefix := strings.Replace(sourceAddrPrefix, "/", "_", -1)
		return fmt.Sprintf("shared-%s-%d-%s", port.Protocol, port.Port, safePrefix)
	}
	safePrefix := strings.Replace(sourceAddrPrefix, "/", "_", -1)
	return fmt.Sprintf("%s-%s-%d-%s", getRulePrefix(service), port.Protocol, port.Port, safePrefix)
}

// This returns a human-readable version of the Service used to tag some resources.
// This is only used for human-readable convenience, and not to filter.
func getServiceName(service *v1.Service) string {
	return fmt.Sprintf("%s/%s", service.Namespace, service.Name)
}

// This returns a prefix for loadbalancer/security rules.
func getRulePrefix(service *v1.Service) string {
	return cloudprovider.GetLoadBalancerName(service)
}

func getPublicIPName(clusterName string, service *v1.Service) string {
	return fmt.Sprintf("%s-%s", clusterName, cloudprovider.GetLoadBalancerName(service))
}

func serviceOwnsRule(service *v1.Service, rule string) bool {
	prefix := getRulePrefix(service)
	return strings.HasPrefix(strings.ToUpper(rule), strings.ToUpper(prefix))
}

func serviceOwnsFrontendIP(fip network.FrontendIPConfiguration, service *v1.Service) bool {
	baseName := cloudprovider.GetLoadBalancerName(service)
	return strings.HasPrefix(*fip.Name, baseName)
}

func getFrontendIPConfigName(service *v1.Service, subnetName *string) string {
	baseName := cloudprovider.GetLoadBalancerName(service)
	if subnetName != nil {
		return fmt.Sprintf("%s-%s", baseName, *subnetName)
	}
	return baseName
}

// This returns the next available rule priority level for a given set of security rules.
func getNextAvailablePriority(rules []network.SecurityRule) (int32, error) {
	var smallest int32 = loadBalancerMinimumPriority
	var spread int32 = 1

outer:
	for smallest < loadBalancerMaximumPriority {
		for _, rule := range rules {
			if *rule.Priority == smallest {
				smallest += spread
				continue outer
			}
		}
		// no one else had it
		return smallest, nil
	}

	return -1, fmt.Errorf("SecurityGroup priorities are exhausted")
}

func (az *Cloud) getIPForMachine(nodeName types.NodeName) (string, string, error) {
	return az.vmSet.GetIPByNodeName(string(nodeName), "")
}

var polyTable = crc32.MakeTable(crc32.Koopman)

//MakeCRC32 : convert string to CRC32 format
func MakeCRC32(str string) string {
	crc := crc32.New(polyTable)
	crc.Write([]byte(str))
	hash := crc.Sum32()
	return strconv.FormatUint(uint64(hash), 10)
}

//ExtractVMData : extract dataDisks, storageProfile from a map struct
func ExtractVMData(vmData map[string]interface{}) (dataDisks []interface{},
	storageProfile map[string]interface{},
	hardwareProfile map[string]interface{}, err error) {
	props, ok := vmData["properties"].(map[string]interface{})
	if !ok {
		return nil, nil, nil, fmt.Errorf("convert vmData(properties) to map error")
	}

	storageProfile, ok = props["storageProfile"].(map[string]interface{})
	if !ok {
		return nil, nil, nil, fmt.Errorf("convert vmData(storageProfile) to map error")
	}

	hardwareProfile, ok = props["hardwareProfile"].(map[string]interface{})
	if !ok {
		return nil, nil, nil, fmt.Errorf("convert vmData(hardwareProfile) to map error")
	}

	dataDisks, ok = storageProfile["dataDisks"].([]interface{})
	if !ok {
		return nil, nil, nil, fmt.Errorf("convert vmData(dataDisks) to map error")
	}
	return dataDisks, storageProfile, hardwareProfile, nil
}

//ExtractDiskData : extract provisioningState, diskState from a map struct
func ExtractDiskData(diskData interface{}) (provisioningState string, diskState string, err error) {
	fragment, ok := diskData.(map[string]interface{})
	if !ok {
		return "", "", fmt.Errorf("convert diskData to map error")
	}

	properties, ok := fragment["properties"].(map[string]interface{})
	if !ok {
		return "", "", fmt.Errorf("convert diskData(properties) to map error")
	}

	provisioningState, ok = properties["provisioningState"].(string) // if there is a disk, provisioningState property will be there
	if ref, ok := properties["diskState"]; ok {
		diskState = ref.(string)
	}
	return provisioningState, diskState, nil
}

// availabilitySet implements VMSet interface for Azure availability sets.
type availabilitySet struct {
	*Cloud
}

// newStandardSet creates a new availabilitySet.
func newAvailabilitySet(az *Cloud) VMSet {
	return &availabilitySet{
		Cloud: az,
	}
}

// GetInstanceIDByNodeName gets the cloud provider ID by node name.
// It must return ("", cloudprovider.InstanceNotFound) if the instance does
// not exist or is no longer running.
func (as *availabilitySet) GetInstanceIDByNodeName(name string) (string, error) {
	var machine compute.VirtualMachine
	var err error

	machine, err = as.getVirtualMachine(types.NodeName(name))
	if err == cloudprovider.InstanceNotFound {
		return "", cloudprovider.InstanceNotFound
	}
	if err != nil {
		if as.CloudProviderBackoff {
			glog.V(2).Infof("InstanceID(%s) backing off", name)
			machine, err = as.GetVirtualMachineWithRetry(types.NodeName(name))
			if err != nil {
				glog.V(2).Infof("InstanceID(%s) abort backoff", name)
				return "", err
			}
		} else {
			return "", err
		}
	}
	return *machine.ID, nil
}

// GetNodeNameByProviderID gets the node name by provider ID.
func (as *availabilitySet) GetNodeNameByProviderID(providerID string) (types.NodeName, error) {
	// NodeName is part of providerID for standard instances.
	matches := providerIDRE.FindStringSubmatch(providerID)
	if len(matches) != 2 {
		return "", errors.New("error splitting providerID")
	}

	return types.NodeName(matches[1]), nil
}

// GetInstanceTypeByNodeName gets the instance type by node name.
func (as *availabilitySet) GetInstanceTypeByNodeName(name string) (string, error) {
	machine, err := as.getVirtualMachine(types.NodeName(name))
	if err != nil {
		glog.Errorf("error: as.GetInstanceTypeByNodeName(%s), as.getVirtualMachine(%s) err=%v", name, name, err)
		return "", err
	}

	return string(machine.HardwareProfile.VMSize), nil
}

// GetZoneByNodeName gets zone from instance view.
func (as *availabilitySet) GetZoneByNodeName(name string) (cloudprovider.Zone, error) {
	vm, err := as.getVirtualMachine(types.NodeName(name))
	if err != nil {
		return cloudprovider.Zone{}, err
	}

	failureDomain := strconv.Itoa(int(*vm.VirtualMachineProperties.InstanceView.PlatformFaultDomain))
	zone := cloudprovider.Zone{
		FailureDomain: failureDomain,
		Region:        *(vm.Location),
	}
	return zone, nil
}

// GetPrimaryVMSetName returns the VM set name depending on the configured vmType.
// It returns config.PrimaryScaleSetName for vmss and config.PrimaryAvailabilitySetName for standard vmType.
func (as *availabilitySet) GetPrimaryVMSetName() string {
	return as.Config.PrimaryAvailabilitySetName
}

// GetIPByNodeName gets machine private IP and public IP by node name.
func (as *availabilitySet) GetIPByNodeName(name, vmSetName string) (string, string, error) {
	nic, err := as.GetPrimaryInterface(name, vmSetName)
	if err != nil {
		return "", "", err
	}

	ipConfig, err := getPrimaryIPConfig(nic)
	if err != nil {
		glog.Errorf("error: as.GetIPByNodeName(%s), getPrimaryIPConfig(%v), err=%v", name, nic, err)
		return "", "", err
	}

	privateIP := *ipConfig.PrivateIPAddress
	publicIP := ""
	if ipConfig.PublicIPAddress != nil && ipConfig.PublicIPAddress.ID != nil {
		pipID := *ipConfig.PublicIPAddress.ID
		pipName, err := getLastSegment(pipID)
		if err != nil {
			return "", "", fmt.Errorf("failed to publicIP name for node %q with pipID %q", name, pipID)
		}
		pip, existsPip, err := as.getPublicIPAddress(as.ResourceGroup, pipName)
		if err != nil {
			return "", "", err
		}
		if existsPip {
			publicIP = *pip.IPAddress
		}
	}

	return privateIP, publicIP, nil
}

// getAgentPoolAvailabiliySets lists the virtual machines for the resource group and then builds
// a list of availability sets that match the nodes available to k8s.
func (as *availabilitySet) getAgentPoolAvailabiliySets(nodes []*v1.Node) (agentPoolAvailabilitySets *[]string, err error) {
	vms, err := as.VirtualMachineClientListWithRetry()
	if err != nil {
		glog.Errorf("as.getNodeAvailabilitySet - VirtualMachineClientListWithRetry failed, err=%v", err)
		return nil, err
	}
	vmNameToAvailabilitySetID := make(map[string]string, len(vms))
	for vmx := range vms {
		vm := vms[vmx]
		if vm.AvailabilitySet != nil {
			vmNameToAvailabilitySetID[*vm.Name] = *vm.AvailabilitySet.ID
		}
	}
	availabilitySetIDs := sets.NewString()
	agentPoolAvailabilitySets = &[]string{}
	for nx := range nodes {
		nodeName := (*nodes[nx]).Name
		if isMasterNode(nodes[nx]) {
			continue
		}
		asID, ok := vmNameToAvailabilitySetID[nodeName]
		if !ok {
			glog.Errorf("as.getNodeAvailabilitySet - Node(%s) has no availability sets", nodeName)
			return nil, fmt.Errorf("Node (%s) - has no availability sets", nodeName)
		}
		if availabilitySetIDs.Has(asID) {
			// already added in the list
			continue
		}
		asName, err := getLastSegment(asID)
		if err != nil {
			glog.Errorf("as.getNodeAvailabilitySet - Node (%s)- getLastSegment(%s), err=%v", nodeName, asID, err)
			return nil, err
		}
		// AvailabilitySet ID is currently upper cased in a indeterministic way
		// We want to keep it lower case, before the ID get fixed
		asName = strings.ToLower(asName)

		*agentPoolAvailabilitySets = append(*agentPoolAvailabilitySets, asName)
	}

	return agentPoolAvailabilitySets, nil
}

// GetVMSetNames selects all possible availability sets or scale sets
// (depending vmType configured) for service load balancer, if the service has
// no loadbalancer mode annotaion returns the primary VMSet. If service annotation
// for loadbalancer exists then return the eligible VMSet.
func (as *availabilitySet) GetVMSetNames(service *v1.Service, nodes []*v1.Node) (availabilitySetNames *[]string, err error) {
	hasMode, isAuto, serviceAvailabilitySetNames := getServiceLoadBalancerMode(service)
	if !hasMode {
		// no mode specified in service annotation default to PrimaryAvailabilitySetName
		availabilitySetNames = &[]string{as.Config.PrimaryAvailabilitySetName}
		return availabilitySetNames, nil
	}
	availabilitySetNames, err = as.getAgentPoolAvailabiliySets(nodes)
	if err != nil {
		glog.Errorf("as.GetVMSetNames - getAgentPoolAvailabiliySets failed err=(%v)", err)
		return nil, err
	}
	if len(*availabilitySetNames) == 0 {
		glog.Errorf("as.GetVMSetNames - No availability sets found for nodes in the cluster, node count(%d)", len(nodes))
		return nil, fmt.Errorf("No availability sets found for nodes, node count(%d)", len(nodes))
	}
	// sort the list to have deterministic selection
	sort.Strings(*availabilitySetNames)
	if !isAuto {
		if serviceAvailabilitySetNames == nil || len(serviceAvailabilitySetNames) == 0 {
			return nil, fmt.Errorf("service annotation for LoadBalancerMode is empty, it should have __auto__ or availability sets value")
		}
		// validate availability set exists
		var found bool
		for sasx := range serviceAvailabilitySetNames {
			for asx := range *availabilitySetNames {
				if strings.EqualFold((*availabilitySetNames)[asx], serviceAvailabilitySetNames[sasx]) {
					found = true
					serviceAvailabilitySetNames[sasx] = (*availabilitySetNames)[asx]
					break
				}
			}
			if !found {
				glog.Errorf("as.GetVMSetNames - Availability set (%s) in service annotation not found", serviceAvailabilitySetNames[sasx])
				return nil, fmt.Errorf("availability set (%s) - not found", serviceAvailabilitySetNames[sasx])
			}
		}
		availabilitySetNames = &serviceAvailabilitySetNames
	}

	return availabilitySetNames, nil
}

// GetPrimaryInterface gets machine primary network interface by node name and vmSet.
func (as *availabilitySet) GetPrimaryInterface(nodeName, vmSetName string) (network.Interface, error) {
	var machine compute.VirtualMachine

	machine, err := as.GetVirtualMachineWithRetry(types.NodeName(nodeName))
	if err != nil {
		glog.V(2).Infof("GetPrimaryInterface(%s, %s) abort backoff", nodeName, vmSetName)
		return network.Interface{}, err
	}

	primaryNicID, err := getPrimaryInterfaceID(machine)
	if err != nil {
		return network.Interface{}, err
	}
	nicName, err := getLastSegment(primaryNicID)
	if err != nil {
		return network.Interface{}, err
	}

	// Check availability set
	if vmSetName != "" {
		expectedAvailabilitySetName := as.getAvailabilitySetID(vmSetName)
		if machine.AvailabilitySet == nil || !strings.EqualFold(*machine.AvailabilitySet.ID, expectedAvailabilitySetName) {
			glog.V(3).Infof(
				"GetPrimaryInterface: nic (%s) is not in the availabilitySet(%s)", nicName, vmSetName)
			return network.Interface{}, errNotInVMSet
		}
	}

	nic, err := as.InterfacesClient.Get(as.ResourceGroup, nicName, "")
	if err != nil {
		return network.Interface{}, err
	}

	return nic, nil
}

// ensureHostInPool ensures the given VM's Primary NIC's Primary IP Configuration is
// participating in the specified LoadBalancer Backend Pool.
func (as *availabilitySet) ensureHostInPool(serviceName string, nodeName types.NodeName, backendPoolID string, vmSetName string) error {
	vmName := mapNodeNameToVMName(nodeName)
	nic, err := as.GetPrimaryInterface(vmName, vmSetName)
	if err != nil {
		if err == errNotInVMSet {
			glog.V(3).Infof("ensureHostInPool skips node %s because it is not in the vmSet %s", nodeName, vmSetName)
			return nil
		}

		glog.Errorf("error: az.ensureHostInPool(%s), az.vmSet.GetPrimaryInterface.Get(%s, %s), err=%v", nodeName, vmName, vmSetName, err)
		return err
	}

	var primaryIPConfig *network.InterfaceIPConfiguration
	primaryIPConfig, err = getPrimaryIPConfig(nic)
	if err != nil {
		return err
	}

	foundPool := false
	newBackendPools := []network.BackendAddressPool{}
	if primaryIPConfig.LoadBalancerBackendAddressPools != nil {
		newBackendPools = *primaryIPConfig.LoadBalancerBackendAddressPools
	}
	for _, existingPool := range newBackendPools {
		if strings.EqualFold(backendPoolID, *existingPool.ID) {
			foundPool = true
			break
		}
	}
	if !foundPool {
		newBackendPools = append(newBackendPools,
			network.BackendAddressPool{
				ID: to.StringPtr(backendPoolID),
			})

		primaryIPConfig.LoadBalancerBackendAddressPools = &newBackendPools

		nicName := *nic.Name
		glog.V(3).Infof("nicupdate(%s): nic(%s) - updating", serviceName, nicName)
		respChan, errChan := as.InterfacesClient.CreateOrUpdate(as.ResourceGroup, *nic.Name, nic, nil)
		resp := <-respChan
		err := <-errChan
		glog.V(10).Infof("InterfacesClient.CreateOrUpdate(%q): end", *nic.Name)
		if as.CloudProviderBackoff && shouldRetryAPIRequest(resp.Response, err) {
			glog.V(2).Infof("nicupdate(%s) backing off: nic(%s) - updating, err=%v", serviceName, nicName, err)
			retryErr := as.CreateOrUpdateInterfaceWithRetry(nic)
			if retryErr != nil {
				err = retryErr
				glog.V(2).Infof("nicupdate(%s) abort backoff: nic(%s) - updating", serviceName, nicName)
			}
		}
		if err != nil {
			return err
		}
	}
	return nil
}

// EnsureHostsInPool ensures the given Node's primary IP configurations are
// participating in the specified LoadBalancer Backend Pool.
func (as *availabilitySet) EnsureHostsInPool(serviceName string, nodes []*v1.Node, backendPoolID string, vmSetName string) error {
	hostUpdates := make([]func() error, len(nodes))
	for i, node := range nodes {
		localNodeName := node.Name
		f := func() error {
			err := as.ensureHostInPool(serviceName, types.NodeName(localNodeName), backendPoolID, vmSetName)
			if err != nil {
				return fmt.Errorf("ensure(%s): backendPoolID(%s) - failed to ensure host in pool: %q", serviceName, backendPoolID, err)
			}
			return nil
		}
		hostUpdates[i] = f
	}

	errs := utilerrors.AggregateGoroutines(hostUpdates...)
	if errs != nil {
		return utilerrors.Flatten(errs)
	}

	return nil
}

// EnsureBackendPoolDeleted ensures the loadBalancer backendAddressPools deleted from the specified vmSet.
func (as *availabilitySet) EnsureBackendPoolDeleted(poolID, vmSetName string) error {
	// Do nothing for availability set.
	return nil
}

// get a storage account by UUID
func generateStorageAccountName(accountNamePrefix string) string {
	uniqueID := strings.Replace(string(uuid.NewUUID()), "-", "", -1)
	accountName := strings.ToLower(accountNamePrefix + uniqueID)
	if len(accountName) > storageAccountNameMaxLength {
		return accountName[:storageAccountNameMaxLength-1]
	}
	return accountName
}
