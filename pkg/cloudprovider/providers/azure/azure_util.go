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
	"strconv"
	"strings"

	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/cloudprovider"

	"github.com/Azure/azure-sdk-for-go/arm/compute"
	"github.com/Azure/azure-sdk-for-go/arm/network"
	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/types"
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

// For a load balancer, all frontend ip should reference either a subnet or publicIpAddress.
// Thus Azure do not allow mixed type (public and internal) load balancer.
// So we'd have a separate name for internal load balancer.
// This would be the name for Azure LoadBalancer resource.
func getLoadBalancerName(clusterName string, isInternal bool) string {
	if isInternal {
		return fmt.Sprintf("%s-internal", clusterName)
	}

	return clusterName
}

func getBackendPoolName(clusterName string) string {
	return clusterName
}

func getLoadBalancerRuleName(service *v1.Service, port v1.ServicePort) string {
	return fmt.Sprintf("%s-%s-%d", getRulePrefix(service), port.Protocol, port.Port)
}

func getSecurityRuleName(service *v1.Service, port v1.ServicePort, sourceAddrPrefix string) string {
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

func getFrontendIPConfigName(service *v1.Service) string {
	return cloudprovider.GetLoadBalancerName(service)
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

func (az *Cloud) getIPForMachine(nodeName types.NodeName) (string, error) {
	az.operationPollRateLimiter.Accept()
	machine, exists, err := az.getVirtualMachine(nodeName)
	if !exists {
		return "", cloudprovider.InstanceNotFound
	}
	if err != nil {
		glog.Errorf("error: az.getIPForMachine(%s), az.getVirtualMachine(%s), err=%v", nodeName, nodeName, err)
		return "", err
	}

	nicID, err := getPrimaryInterfaceID(machine)
	if err != nil {
		glog.Errorf("error: az.getIPForMachine(%s), getPrimaryInterfaceID(%v), err=%v", nodeName, machine, err)
		return "", err
	}

	nicName, err := getLastSegment(nicID)
	if err != nil {
		glog.Errorf("error: az.getIPForMachine(%s), getLastSegment(%s), err=%v", nodeName, nicID, err)
		return "", err
	}

	az.operationPollRateLimiter.Accept()
	glog.V(10).Infof("InterfacesClient.Get(%q): start", nicName)
	nic, err := az.InterfacesClient.Get(az.ResourceGroup, nicName, "")
	glog.V(10).Infof("InterfacesClient.Get(%q): end", nicName)
	if err != nil {
		glog.Errorf("error: az.getIPForMachine(%s), az.InterfacesClient.Get(%s, %s, %s), err=%v", nodeName, az.ResourceGroup, nicName, "", err)
		return "", err
	}

	ipConfig, err := getPrimaryIPConfig(nic)
	if err != nil {
		glog.Errorf("error: az.getIPForMachine(%s), getPrimaryIPConfig(%v), err=%v", nodeName, nic, err)
		return "", err
	}

	targetIP := *ipConfig.PrivateIPAddress
	return targetIP, nil
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
