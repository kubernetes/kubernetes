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
	"fmt"
	"strconv"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	serviceapi "k8s.io/kubernetes/pkg/api/service"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"

	"github.com/Azure/azure-sdk-for-go/arm/network"
	"github.com/Azure/go-autorest/autorest/to"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/types"
)

// GetLoadBalancer returns whether the specified load balancer exists, and
// if so, what its status is.
func (az *Cloud) GetLoadBalancer(clusterName string, service *api.Service) (status *api.LoadBalancerStatus, exists bool, err error) {
	lbName := getLoadBalancerName(clusterName)
	pipName := getPublicIPName(clusterName, service)
	serviceName := getServiceName(service)

	_, existsLb, err := az.getAzureLoadBalancer(lbName)
	if err != nil {
		return nil, false, err
	}
	if !existsLb {
		glog.V(5).Infof("get(%s): lb(%s) - doesn't exist", serviceName, pipName)
		return nil, false, nil
	}

	pip, existsPip, err := az.getPublicIPAddress(pipName)
	if err != nil {
		return nil, false, err
	}
	if !existsPip {
		glog.V(5).Infof("get(%s): pip(%s) - doesn't exist", serviceName, pipName)
		return nil, false, nil
	}

	return &api.LoadBalancerStatus{
		Ingress: []api.LoadBalancerIngress{{IP: *pip.IPAddress}},
	}, true, nil
}

// EnsureLoadBalancer creates a new load balancer 'name', or updates the existing one. Returns the status of the balancer
func (az *Cloud) EnsureLoadBalancer(clusterName string, service *api.Service, nodeNames []string) (*api.LoadBalancerStatus, error) {
	lbName := getLoadBalancerName(clusterName)
	pipName := getPublicIPName(clusterName, service)
	serviceName := getServiceName(service)
	glog.V(2).Infof("ensure(%s): START clusterName=%q lbName=%q", serviceName, clusterName, lbName)

	pip, err := az.ensurePublicIPExists(serviceName, pipName)
	if err != nil {
		return nil, err
	}

	sg, err := az.SecurityGroupsClient.Get(az.ResourceGroup, az.SecurityGroupName, "")
	if err != nil {
		return nil, err
	}
	sg, sgNeedsUpdate, err := az.reconcileSecurityGroup(sg, clusterName, service)
	if err != nil {
		return nil, err
	}
	if sgNeedsUpdate {
		glog.V(3).Infof("ensure(%s): sg(%s) - updating", serviceName, *sg.Name)
		_, err := az.SecurityGroupsClient.CreateOrUpdate(az.ResourceGroup, *sg.Name, sg, nil)
		if err != nil {
			return nil, err
		}
	}

	lb, existsLb, err := az.getAzureLoadBalancer(lbName)
	if err != nil {
		return nil, err
	}
	if !existsLb {
		lb = network.LoadBalancer{
			Name:                         &lbName,
			Location:                     &az.Location,
			LoadBalancerPropertiesFormat: &network.LoadBalancerPropertiesFormat{},
		}
	}

	lb, lbNeedsUpdate, err := az.reconcileLoadBalancer(lb, pip, clusterName, service, nodeNames)
	if err != nil {
		return nil, err
	}
	if !existsLb || lbNeedsUpdate {
		glog.V(3).Infof("ensure(%s): lb(%s) - updating", serviceName, lbName)
		_, err = az.LoadBalancerClient.CreateOrUpdate(az.ResourceGroup, *lb.Name, lb, nil)
		if err != nil {
			return nil, err
		}
	}

	// Add the machines to the backend pool if they're not already
	lbBackendName := getBackendPoolName(clusterName)
	lbBackendPoolID := az.getBackendPoolID(lbName, lbBackendName)
	hostUpdates := make([]func() error, len(nodeNames))
	for i, nodeName := range nodeNames {
		localNodeName := nodeName
		f := func() error {
			err := az.ensureHostInPool(serviceName, types.NodeName(localNodeName), lbBackendPoolID)
			if err != nil {
				return fmt.Errorf("ensure(%s): lb(%s) - failed to ensure host in pool: %q", serviceName, lbName, err)
			}
			return nil
		}
		hostUpdates[i] = f
	}

	errs := utilerrors.AggregateGoroutines(hostUpdates...)
	if errs != nil {
		return nil, utilerrors.Flatten(errs)
	}

	glog.V(2).Infof("ensure(%s): FINISH - %s", serviceName, *pip.IPAddress)
	return &api.LoadBalancerStatus{
		Ingress: []api.LoadBalancerIngress{{IP: *pip.IPAddress}},
	}, nil
}

// UpdateLoadBalancer updates hosts under the specified load balancer.
func (az *Cloud) UpdateLoadBalancer(clusterName string, service *api.Service, nodeNames []string) error {
	_, err := az.EnsureLoadBalancer(clusterName, service, nodeNames)
	return err
}

// EnsureLoadBalancerDeleted deletes the specified load balancer if it
// exists, returning nil if the load balancer specified either didn't exist or
// was successfully deleted.
// This construction is useful because many cloud providers' load balancers
// have multiple underlying components, meaning a Get could say that the LB
// doesn't exist even if some part of it is still laying around.
func (az *Cloud) EnsureLoadBalancerDeleted(clusterName string, service *api.Service) error {
	lbName := getLoadBalancerName(clusterName)
	pipName := getPublicIPName(clusterName, service)
	serviceName := getServiceName(service)

	glog.V(2).Infof("delete(%s): START clusterName=%q lbName=%q", serviceName, clusterName, lbName)

	// reconcile logic is capable of fully reconcile, so we can use this to delete
	service.Spec.Ports = []api.ServicePort{}

	lb, existsLb, err := az.getAzureLoadBalancer(lbName)
	if err != nil {
		return err
	}
	if existsLb {
		lb, lbNeedsUpdate, reconcileErr := az.reconcileLoadBalancer(lb, nil, clusterName, service, []string{})
		if reconcileErr != nil {
			return reconcileErr
		}
		if lbNeedsUpdate {
			if len(*lb.FrontendIPConfigurations) > 0 {
				glog.V(3).Infof("delete(%s): lb(%s) - updating", serviceName, lbName)
				_, err = az.LoadBalancerClient.CreateOrUpdate(az.ResourceGroup, *lb.Name, lb, nil)
				if err != nil {
					return err
				}
			} else {
				glog.V(3).Infof("delete(%s): lb(%s) - deleting; no remaining frontendipconfigs", serviceName, lbName)

				_, err = az.LoadBalancerClient.Delete(az.ResourceGroup, lbName, nil)
				if err != nil {
					return err
				}
			}
		}
	}

	sg, existsSg, err := az.getSecurityGroup()
	if err != nil {
		return err
	}
	if existsSg {
		reconciledSg, sgNeedsUpdate, reconcileErr := az.reconcileSecurityGroup(sg, clusterName, service)
		if reconcileErr != nil {
			return reconcileErr
		}
		if sgNeedsUpdate {
			glog.V(3).Infof("delete(%s): sg(%s) - updating", serviceName, az.SecurityGroupName)
			_, err := az.SecurityGroupsClient.CreateOrUpdate(az.ResourceGroup, *reconciledSg.Name, reconciledSg, nil)
			if err != nil {
				return err
			}
		}
	}

	err = az.ensurePublicIPDeleted(serviceName, pipName)
	if err != nil {
		return err
	}

	glog.V(2).Infof("delete(%s): FINISH", serviceName)
	return nil
}

func (az *Cloud) ensurePublicIPExists(serviceName, pipName string) (*network.PublicIPAddress, error) {
	pip, existsPip, err := az.getPublicIPAddress(pipName)
	if err != nil {
		return nil, err
	}
	if existsPip {
		return &pip, nil
	}

	pip.Name = to.StringPtr(pipName)
	pip.Location = to.StringPtr(az.Location)
	pip.PublicIPAddressPropertiesFormat = &network.PublicIPAddressPropertiesFormat{
		PublicIPAllocationMethod: network.Static,
	}
	pip.Tags = &map[string]*string{"service": &serviceName}

	glog.V(3).Infof("ensure(%s): pip(%s) - creating", serviceName, *pip.Name)
	_, err = az.PublicIPAddressesClient.CreateOrUpdate(az.ResourceGroup, *pip.Name, pip, nil)
	if err != nil {
		return nil, err
	}

	pip, err = az.PublicIPAddressesClient.Get(az.ResourceGroup, *pip.Name, "")
	if err != nil {
		return nil, err
	}

	return &pip, nil

}

func (az *Cloud) ensurePublicIPDeleted(serviceName, pipName string) error {
	_, deleteErr := az.PublicIPAddressesClient.Delete(az.ResourceGroup, pipName, nil)
	_, realErr := checkResourceExistsFromError(deleteErr)
	if realErr != nil {
		return nil
	}
	return nil
}

// This ensures load balancer exists and the frontend ip config is setup.
// This also reconciles the Service's Ports  with the LoadBalancer config.
// This entails adding rules/probes for expected Ports and removing stale rules/ports.
func (az *Cloud) reconcileLoadBalancer(lb network.LoadBalancer, pip *network.PublicIPAddress, clusterName string, service *api.Service, nodeNames []string) (network.LoadBalancer, bool, error) {
	lbName := getLoadBalancerName(clusterName)
	serviceName := getServiceName(service)
	lbFrontendIPConfigName := getFrontendIPConfigName(service)
	lbFrontendIPConfigID := az.getFrontendIPConfigID(lbName, lbFrontendIPConfigName)
	lbBackendPoolName := getBackendPoolName(clusterName)
	lbBackendPoolID := az.getBackendPoolID(lbName, lbBackendPoolName)

	wantLb := len(service.Spec.Ports) > 0
	dirtyLb := false

	// Ensure LoadBalancer's Backend Pool Configuration
	if wantLb {
		if lb.BackendAddressPools == nil ||
			len(*lb.BackendAddressPools) == 0 {
			lb.BackendAddressPools = &[]network.BackendAddressPool{
				{
					Name: to.StringPtr(lbBackendPoolName),
				},
			}
			glog.V(10).Infof("reconcile(%s)(%t): lb backendpool - adding", serviceName, wantLb)
			dirtyLb = true
		} else if len(*lb.BackendAddressPools) != 1 ||
			!strings.EqualFold(*(*lb.BackendAddressPools)[0].Name, lbBackendPoolName) {
			return lb, false, fmt.Errorf("loadbalancer is misconfigured with a different backend pool")
		}
	}

	// Ensure LoadBalancer's Frontend IP Configurations
	dirtyConfigs := false
	newConfigs := []network.FrontendIPConfiguration{}
	if lb.FrontendIPConfigurations != nil {
		newConfigs = *lb.FrontendIPConfigurations
	}
	if !wantLb {
		for i := len(newConfigs) - 1; i >= 0; i-- {
			config := newConfigs[i]
			if strings.EqualFold(*config.Name, lbFrontendIPConfigName) {
				glog.V(3).Infof("reconcile(%s)(%t): lb frontendconfig(%s) - dropping", serviceName, wantLb, lbFrontendIPConfigName)
				newConfigs = append(newConfigs[:i], newConfigs[i+1:]...)
				dirtyConfigs = true
			}
		}
	} else {
		foundConfig := false
		for _, config := range newConfigs {
			if strings.EqualFold(*config.Name, lbFrontendIPConfigName) {
				foundConfig = true
				break
			}
		}
		if !foundConfig {
			newConfigs = append(newConfigs,
				network.FrontendIPConfiguration{
					Name: to.StringPtr(lbFrontendIPConfigName),
					FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
						PublicIPAddress: &network.PublicIPAddress{
							ID: pip.ID,
						},
					},
				})
			glog.V(10).Infof("reconcile(%s)(%t): lb frontendconfig(%s) - adding", serviceName, wantLb, lbFrontendIPConfigName)
			dirtyConfigs = true
		}
	}
	if dirtyConfigs {
		dirtyLb = true
		lb.FrontendIPConfigurations = &newConfigs
	}

	// update probes/rules
	expectedProbes := make([]network.Probe, len(service.Spec.Ports))
	expectedRules := make([]network.LoadBalancingRule, len(service.Spec.Ports))
	for i, port := range service.Spec.Ports {
		lbRuleName := getRuleName(service, port)

		transportProto, _, probeProto, err := getProtocolsFromKubernetesProtocol(port.Protocol)
		if err != nil {
			return lb, false, err
		}

		if serviceapi.NeedsHealthCheck(service) {
			podPresencePath, podPresencePort := serviceapi.GetServiceHealthCheckPathPort(service)

			expectedProbes[i] = network.Probe{
				Name: &lbRuleName,
				ProbePropertiesFormat: &network.ProbePropertiesFormat{
					RequestPath:       to.StringPtr(podPresencePath),
					Protocol:          network.ProbeProtocolHTTP,
					Port:              to.Int32Ptr(podPresencePort),
					IntervalInSeconds: to.Int32Ptr(5),
					NumberOfProbes:    to.Int32Ptr(2),
				},
			}
		} else {
			expectedProbes[i] = network.Probe{
				Name: &lbRuleName,
				ProbePropertiesFormat: &network.ProbePropertiesFormat{
					Protocol:          probeProto,
					Port:              to.Int32Ptr(port.NodePort),
					IntervalInSeconds: to.Int32Ptr(5),
					NumberOfProbes:    to.Int32Ptr(2),
				},
			}
		}

		expectedRules[i] = network.LoadBalancingRule{
			Name: &lbRuleName,
			LoadBalancingRulePropertiesFormat: &network.LoadBalancingRulePropertiesFormat{
				Protocol: transportProto,
				FrontendIPConfiguration: &network.SubResource{
					ID: to.StringPtr(lbFrontendIPConfigID),
				},
				BackendAddressPool: &network.SubResource{
					ID: to.StringPtr(lbBackendPoolID),
				},
				Probe: &network.SubResource{
					ID: to.StringPtr(az.getLoadBalancerProbeID(lbName, lbRuleName)),
				},
				FrontendPort:     to.Int32Ptr(port.Port),
				BackendPort:      to.Int32Ptr(port.Port),
				EnableFloatingIP: to.BoolPtr(true),
			},
		}
	}

	// remove unwanted probes
	dirtyProbes := false
	var updatedProbes []network.Probe
	if lb.Probes != nil {
		updatedProbes = *lb.Probes
	}
	for i := len(updatedProbes) - 1; i >= 0; i-- {
		existingProbe := updatedProbes[i]
		if serviceOwnsRule(service, *existingProbe.Name) {
			glog.V(10).Infof("reconcile(%s)(%t): lb probe(%s) - considering evicting", serviceName, wantLb, *existingProbe.Name)
			keepProbe := false
			if findProbe(expectedProbes, existingProbe) {
				glog.V(10).Infof("reconcile(%s)(%t): lb probe(%s) - keeping", serviceName, wantLb, *existingProbe.Name)
				keepProbe = true
			}
			if !keepProbe {
				updatedProbes = append(updatedProbes[:i], updatedProbes[i+1:]...)
				glog.V(10).Infof("reconcile(%s)(%t): lb probe(%s) - dropping", serviceName, wantLb, *existingProbe.Name)
				dirtyProbes = true
			}
		}
	}
	// add missing, wanted probes
	for _, expectedProbe := range expectedProbes {
		foundProbe := false
		if findProbe(updatedProbes, expectedProbe) {
			glog.V(10).Infof("reconcile(%s)(%t): lb probe(%s) - already exists", serviceName, wantLb, *expectedProbe.Name)
			foundProbe = true
		}
		if !foundProbe {
			glog.V(10).Infof("reconcile(%s)(%t): lb probe(%s) - adding", serviceName, wantLb, *expectedProbe.Name)
			updatedProbes = append(updatedProbes, expectedProbe)
			dirtyProbes = true
		}
	}
	if dirtyProbes {
		dirtyLb = true
		lb.Probes = &updatedProbes
	}

	// update rules
	dirtyRules := false
	var updatedRules []network.LoadBalancingRule
	if lb.LoadBalancingRules != nil {
		updatedRules = *lb.LoadBalancingRules
	}
	// update rules: remove unwanted
	for i := len(updatedRules) - 1; i >= 0; i-- {
		existingRule := updatedRules[i]
		if serviceOwnsRule(service, *existingRule.Name) {
			keepRule := false
			glog.V(10).Infof("reconcile(%s)(%t): lb rule(%s) - considering evicting", serviceName, wantLb, *existingRule.Name)
			if findRule(expectedRules, existingRule) {
				glog.V(10).Infof("reconcile(%s)(%t): lb rule(%s) - keeping", serviceName, wantLb, *existingRule.Name)
				keepRule = true
			}
			if !keepRule {
				glog.V(3).Infof("reconcile(%s)(%t): lb rule(%s) - dropping", serviceName, wantLb, *existingRule.Name)
				updatedRules = append(updatedRules[:i], updatedRules[i+1:]...)
				dirtyRules = true
			}
		}
	}
	// update rules: add needed
	for _, expectedRule := range expectedRules {
		foundRule := false
		if findRule(updatedRules, expectedRule) {
			glog.V(10).Infof("reconcile(%s)(%t): lb rule(%s) - already exists", serviceName, wantLb, *expectedRule.Name)
			foundRule = true
		}
		if !foundRule {
			glog.V(10).Infof("reconcile(%s)(%t): lb rule(%s) adding", serviceName, wantLb, *expectedRule.Name)
			updatedRules = append(updatedRules, expectedRule)
			dirtyRules = true
		}
	}
	if dirtyRules {
		dirtyLb = true
		lb.LoadBalancingRules = &updatedRules
	}

	return lb, dirtyLb, nil
}

// This reconciles the Network Security Group similar to how the LB is reconciled.
// This entails adding required, missing SecurityRules and removing stale rules.
func (az *Cloud) reconcileSecurityGroup(sg network.SecurityGroup, clusterName string, service *api.Service) (network.SecurityGroup, bool, error) {
	serviceName := getServiceName(service)
	wantLb := len(service.Spec.Ports) > 0

	sourceRanges, err := serviceapi.GetLoadBalancerSourceRanges(service)
	if err != nil {
		return sg, false, err
	}
	var sourceAddressPrefixes []string
	if sourceRanges == nil || serviceapi.IsAllowAll(sourceRanges) {
		sourceAddressPrefixes = []string{"Internet"}
	} else {
		for _, ip := range sourceRanges {
			sourceAddressPrefixes = append(sourceAddressPrefixes, ip.String())
		}
	}
	expectedSecurityRules := make([]network.SecurityRule, len(service.Spec.Ports)*len(sourceAddressPrefixes))

	for i, port := range service.Spec.Ports {
		securityRuleName := getRuleName(service, port)
		_, securityProto, _, err := getProtocolsFromKubernetesProtocol(port.Protocol)
		if err != nil {
			return sg, false, err
		}
		for j := range sourceAddressPrefixes {
			ix := i*len(sourceAddressPrefixes) + j
			expectedSecurityRules[ix] = network.SecurityRule{
				Name: to.StringPtr(securityRuleName),
				SecurityRulePropertiesFormat: &network.SecurityRulePropertiesFormat{
					Protocol:                 securityProto,
					SourcePortRange:          to.StringPtr("*"),
					DestinationPortRange:     to.StringPtr(strconv.Itoa(int(port.Port))),
					SourceAddressPrefix:      to.StringPtr(sourceAddressPrefixes[j]),
					DestinationAddressPrefix: to.StringPtr("*"),
					Access:    network.Allow,
					Direction: network.Inbound,
				},
			}
		}
	}

	// update security rules
	dirtySg := false
	var updatedRules []network.SecurityRule
	if sg.SecurityRules != nil {
		updatedRules = *sg.SecurityRules
	}
	// update security rules: remove unwanted
	for i := len(updatedRules) - 1; i >= 0; i-- {
		existingRule := updatedRules[i]
		if serviceOwnsRule(service, *existingRule.Name) {
			glog.V(10).Infof("reconcile(%s)(%t): sg rule(%s) - considering evicting", serviceName, wantLb, *existingRule.Name)
			keepRule := false
			if findSecurityRule(expectedSecurityRules, existingRule) {
				glog.V(10).Infof("reconcile(%s)(%t): sg rule(%s) - keeping", serviceName, wantLb, *existingRule.Name)
				keepRule = true
			}
			if !keepRule {
				glog.V(10).Infof("reconcile(%s)(%t): sg rule(%s) - dropping", serviceName, wantLb, *existingRule.Name)
				updatedRules = append(updatedRules[:i], updatedRules[i+1:]...)
				dirtySg = true
			}
		}
	}
	// update security rules: add needed
	for _, expectedRule := range expectedSecurityRules {
		foundRule := false
		if findSecurityRule(updatedRules, expectedRule) {
			glog.V(10).Infof("reconcile(%s)(%t): sg rule(%s) - already exists", serviceName, wantLb, *expectedRule.Name)
			foundRule = true
		}
		if !foundRule {
			glog.V(10).Infof("reconcile(%s)(%t): sg rule(%s) - adding", serviceName, wantLb, *expectedRule.Name)

			nextAvailablePriority, err := getNextAvailablePriority(updatedRules)
			if err != nil {
				return sg, false, err
			}

			expectedRule.Priority = to.Int32Ptr(nextAvailablePriority)
			updatedRules = append(updatedRules, expectedRule)
			dirtySg = true
		}
	}
	if dirtySg {
		sg.SecurityRules = &updatedRules
	}
	return sg, dirtySg, nil
}

func findProbe(probes []network.Probe, probe network.Probe) bool {
	for _, existingProbe := range probes {
		if strings.EqualFold(*existingProbe.Name, *probe.Name) {
			return true
		}
	}
	return false
}

func findRule(rules []network.LoadBalancingRule, rule network.LoadBalancingRule) bool {
	for _, existingRule := range rules {
		if strings.EqualFold(*existingRule.Name, *rule.Name) {
			return true
		}
	}
	return false
}

func findSecurityRule(rules []network.SecurityRule, rule network.SecurityRule) bool {
	for _, existingRule := range rules {
		if strings.EqualFold(*existingRule.Name, *rule.Name) {
			return true
		}
	}
	return false
}

// This ensures the given VM's Primary NIC's Primary IP Configuration is
// participating in the specified LoadBalancer Backend Pool.
func (az *Cloud) ensureHostInPool(serviceName string, nodeName types.NodeName, backendPoolID string) error {
	vmName := mapNodeNameToVMName(nodeName)
	machine, err := az.VirtualMachinesClient.Get(az.ResourceGroup, vmName, "")
	if err != nil {
		return err
	}

	primaryNicID, err := getPrimaryInterfaceID(machine)
	if err != nil {
		return err
	}
	nicName, err := getLastSegment(primaryNicID)
	if err != nil {
		return err
	}

	// Check availability set
	if az.PrimaryAvailabilitySetName != "" {
		expectedAvailabilitySetName := az.getAvailabilitySetID(az.PrimaryAvailabilitySetName)
		if !strings.EqualFold(*machine.AvailabilitySet.ID, expectedAvailabilitySetName) {
			glog.V(3).Infof(
				"nicupdate(%s): skipping nic (%s) since it is not in the primaryAvailabilitSet(%s)",
				serviceName, nicName, az.PrimaryAvailabilitySetName)
			return nil
		}
	}

	nic, err := az.InterfacesClient.Get(az.ResourceGroup, nicName, "")
	if err != nil {
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

		glog.V(3).Infof("nicupdate(%s): nic(%s) - updating", serviceName, nicName)
		_, err := az.InterfacesClient.CreateOrUpdate(az.ResourceGroup, *nic.Name, nic, nil)
		if err != nil {
			return err
		}
	}
	return nil
}
