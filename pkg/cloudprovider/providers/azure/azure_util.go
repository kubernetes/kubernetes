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
	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/types"
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
	loadBalancerRuleIDTemplate  = "/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Network/loadBalancers/%s/loadBalancingRules/%s"
	loadBalancerProbeIDTemplate = "/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Network/loadBalancers/%s/probes/%s"
	securityRuleIDTemplate      = "/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Network/networkSecurityGroups/%s/securityRules/%s"

	// InternalLoadBalancerNameSuffix is load balancer posfix
	InternalLoadBalancerNameSuffix = "-internal"

	// nodeLabelRole specifies the role of a node
	nodeLabelRole = "kubernetes.io/role"

	storageAccountNameMaxLength = 24
)

var providerIDRE = regexp.MustCompile(`^` + CloudProviderName + `://(?:.*)/Microsoft.Compute/virtualMachines/(.+)$`)

// returns the full identifier of a machine
func (az *Cloud) getMachineID(machineName string) string {
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

// returns the full identifier of a loadbalancer rule.
func (az *Cloud) getLoadBalancerRuleID(lbName, lbRuleName string) string {
	return fmt.Sprintf(
		loadBalancerRuleIDTemplate,
		az.SubscriptionID,
		az.ResourceGroup,
		lbName,
		lbRuleName)
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

// returns the full identifier of a network security group security rule.
func (az *Cloud) getSecurityRuleID(securityRuleName string) string {
	return fmt.Sprintf(
		securityRuleIDTemplate,
		az.SubscriptionID,
		az.ResourceGroup,
		az.SecurityGroupName,
		securityRuleName)
}

// returns the full identifier of a publicIPAddress.
func (az *Cloud) getpublicIPAddressID(pipName string) string {
	return fmt.Sprintf(
		publicIPAddressIDTemplate,
		az.SubscriptionID,
		az.ResourceGroup,
		pipName)
}

// getLoadBalancerAvailabilitySetNames selects all possible availability sets for
// service load balancer, if the service has no loadbalancer mode annotaion returns the
// primary availability set if service annotation for loadbalancer availability set
// exists then return the eligible a availability set
func (az *Cloud) getLoadBalancerAvailabilitySetNames(service *v1.Service, nodes []*v1.Node) (availabilitySetNames *[]string, err error) {
	hasMode, isAuto, serviceAvailabilitySetNames := getServiceLoadBalancerMode(service)
	if !hasMode {
		// no mode specified in service annotation default to PrimaryAvailabilitySetName
		availabilitySetNames = &[]string{az.Config.PrimaryAvailabilitySetName}
		return availabilitySetNames, nil
	}
	availabilitySetNames, err = az.getAgentPoolAvailabiliySets(nodes)
	if err != nil {
		glog.Errorf("az.getLoadBalancerAvailabilitySetNames - getAgentPoolAvailabiliySets failed err=(%v)", err)
		return nil, err
	}
	if len(*availabilitySetNames) == 0 {
		glog.Errorf("az.getLoadBalancerAvailabilitySetNames - No availability sets found for nodes in the cluster, node count(%d)", len(nodes))
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
				glog.Errorf("az.getLoadBalancerAvailabilitySetNames - Availability set (%s) in service annotation not found", serviceAvailabilitySetNames[sasx])
				return nil, fmt.Errorf("availability set (%s) - not found", serviceAvailabilitySetNames[sasx])
			}
		}
		availabilitySetNames = &serviceAvailabilitySetNames
	}

	return availabilitySetNames, nil
}

// lists the virtual machines for for the resource group and then builds
// a list of availability sets that match the nodes available to k8s
func (az *Cloud) getAgentPoolAvailabiliySets(nodes []*v1.Node) (agentPoolAvailabilitySets *[]string, err error) {
	vms, err := az.VirtualMachineClientListWithRetry()
	if err != nil {
		glog.Errorf("az.getNodeAvailabilitySet - VirtualMachineClientListWithRetry failed, err=%v", err)
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
			glog.Errorf("az.getNodeAvailabilitySet - Node(%s) has no availability sets", nodeName)
			return nil, fmt.Errorf("Node (%s) - has no availability sets", nodeName)
		}
		if availabilitySetIDs.Has(asID) {
			// already added in the list
			continue
		}
		asName, err := getLastSegment(asID)
		if err != nil {
			glog.Errorf("az.getNodeAvailabilitySet - Node (%s)- getLastSegment(%s), err=%v", nodeName, asID, err)
			return nil, err
		}
		// AvailabilitySet ID is currently upper cased in a indeterministic way
		// We want to keep it lower case, before the ID get fixed
		asName = strings.ToLower(asName)

		*agentPoolAvailabilitySets = append(*agentPoolAvailabilitySets, asName)
	}

	return agentPoolAvailabilitySets, nil
}

func (az *Cloud) mapLoadBalancerNameToAvailabilitySet(lbName string, clusterName string) (availabilitySetName string) {
	availabilitySetName = strings.TrimSuffix(lbName, InternalLoadBalancerNameSuffix)
	if strings.EqualFold(clusterName, availabilitySetName) {
		availabilitySetName = az.Config.PrimaryAvailabilitySetName
	}

	return availabilitySetName
}

// For a load balancer, all frontend ip should reference either a subnet or publicIpAddress.
// Thus Azure do not allow mixed type (public and internal) load balancer.
// So we'd have a separate name for internal load balancer.
// This would be the name for Azure LoadBalancer resource.
func (az *Cloud) getLoadBalancerName(clusterName string, availabilitySetName string, isInternal bool) string {
	lbNamePrefix := availabilitySetName
	if strings.EqualFold(availabilitySetName, az.Config.PrimaryAvailabilitySetName) {
		lbNamePrefix = clusterName
	}
	if isInternal {
		return fmt.Sprintf("%s%s", lbNamePrefix, InternalLoadBalancerNameSuffix)
	}
	return lbNamePrefix
}

// isMasterNode returns returns true is the node has a master role label.
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
	if len(*nic.IPConfigurations) == 1 {
		return &((*nic.IPConfigurations)[0]), nil
	}

	for _, ref := range *nic.IPConfigurations {
		if *ref.Primary {
			return &ref, nil
		}
	}

	return nil, fmt.Errorf("failed to determine the determine primary ipconfig. nicname=%q", *nic.Name)
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
	if az.Config.VMType == vmTypeVMSS {
		ip, publicIP, err := az.getIPForVmssMachine(nodeName)
		if err == cloudprovider.InstanceNotFound || err == ErrorNotVmssInstance {
			return az.getIPForStandardMachine(nodeName)
		}

		return ip, publicIP, err
	}

	return az.getIPForStandardMachine(nodeName)
}

func (az *Cloud) getIPForStandardMachine(nodeName types.NodeName) (string, string, error) {
	az.operationPollRateLimiter.Accept()
	machine, err := az.getVirtualMachine(nodeName)
	if err != nil {
		glog.Errorf("error: az.getIPForMachine(%s), az.getVirtualMachine(%s), err=%v", nodeName, nodeName, err)
		return "", "", err
	}

	nicID, err := getPrimaryInterfaceID(machine)
	if err != nil {
		glog.Errorf("error: az.getIPForMachine(%s), getPrimaryInterfaceID(%v), err=%v", nodeName, machine, err)
		return "", "", err
	}

	nicName, err := getLastSegment(nicID)
	if err != nil {
		glog.Errorf("error: az.getIPForMachine(%s), getLastSegment(%s), err=%v", nodeName, nicID, err)
		return "", "", err
	}

	az.operationPollRateLimiter.Accept()
	glog.V(10).Infof("InterfacesClient.Get(%q): start", nicName)
	nic, err := az.InterfacesClient.Get(az.ResourceGroup, nicName, "")
	glog.V(10).Infof("InterfacesClient.Get(%q): end", nicName)
	if err != nil {
		glog.Errorf("error: az.getIPForMachine(%s), az.InterfacesClient.Get(%s, %s, %s), err=%v", nodeName, az.ResourceGroup, nicName, "", err)
		return "", "", err
	}

	ipConfig, err := getPrimaryIPConfig(nic)
	if err != nil {
		glog.Errorf("error: az.getIPForMachine(%s), getPrimaryIPConfig(%v), err=%v", nodeName, nic, err)
		return "", "", err
	}

	privateIP := *ipConfig.PrivateIPAddress
	publicIP := ""
	if ipConfig.PublicIPAddress != nil && ipConfig.PublicIPAddress.ID != nil {
		pipID := *ipConfig.PublicIPAddress.ID
		pipName, err := getLastSegment(pipID)
		if err != nil {
			return "", "", fmt.Errorf("failed to get publicIP name for node %q with pipID %q", nodeName, pipID)
		}
		pip, existsPip, err := az.getPublicIPAddress(pipName)
		if err != nil {
			return "", "", err
		}
		if existsPip {
			publicIP = *pip.IPAddress
		}
	}

	return privateIP, publicIP, nil
}

// splitProviderID converts a providerID to a NodeName.
func splitProviderID(providerID string) (types.NodeName, error) {
	matches := providerIDRE.FindStringSubmatch(providerID)
	if len(matches) != 2 {
		return "", errors.New("error splitting providerID")
	}
	return types.NodeName(matches[1]), nil
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

// get a storage account by UUID
func generateStorageAccountName(accountNamePrefix string) string {
	uniqueID := strings.Replace(string(uuid.NewUUID()), "-", "", -1)
	accountName := strings.ToLower(accountNamePrefix + uniqueID)
	if len(accountName) > storageAccountNameMaxLength {
		return accountName[:storageAccountNameMaxLength-1]
	}
	return accountName
}
