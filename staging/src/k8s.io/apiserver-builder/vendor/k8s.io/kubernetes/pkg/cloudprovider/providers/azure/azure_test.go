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
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	serviceapi "k8s.io/kubernetes/pkg/api/v1/service"

	"github.com/Azure/azure-sdk-for-go/arm/compute"
	"github.com/Azure/azure-sdk-for-go/arm/network"
	"github.com/Azure/go-autorest/autorest/to"
)

var testClusterName = "testCluster"

// Test additional of a new service/port.
func TestReconcileLoadBalancerAddPort(t *testing.T) {
	az := getTestCloud()
	svc := getTestService("servicea", 80)
	pip := getTestPublicIP()
	lb := getTestLoadBalancer()
	nodes := []*v1.Node{}

	lb, updated, err := az.reconcileLoadBalancer(lb, &pip, testClusterName, &svc, nodes)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}

	if !updated {
		t.Error("Expected the loadbalancer to need an update")
	}

	// ensure we got a frontend ip configuration
	if len(*lb.FrontendIPConfigurations) != 1 {
		t.Error("Expected the loadbalancer to have a frontend ip configuration")
	}

	validateLoadBalancer(t, lb, svc)
}

func TestReconcileLoadBalancerNodeHealth(t *testing.T) {
	az := getTestCloud()
	svc := getTestService("servicea", 80)
	svc.Annotations = map[string]string{
		serviceapi.BetaAnnotationExternalTraffic:     serviceapi.AnnotationValueExternalTrafficLocal,
		serviceapi.BetaAnnotationHealthCheckNodePort: "32456",
	}
	pip := getTestPublicIP()
	lb := getTestLoadBalancer()

	nodes := []*v1.Node{}

	lb, updated, err := az.reconcileLoadBalancer(lb, &pip, testClusterName, &svc, nodes)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}

	if !updated {
		t.Error("Expected the loadbalancer to need an update")
	}

	// ensure we got a frontend ip configuration
	if len(*lb.FrontendIPConfigurations) != 1 {
		t.Error("Expected the loadbalancer to have a frontend ip configuration")
	}

	validateLoadBalancer(t, lb, svc)
}

// Test removing all services results in removing the frontend ip configuration
func TestReconcileLoadBalancerRemoveAllPortsRemovesFrontendConfig(t *testing.T) {
	az := getTestCloud()
	svc := getTestService("servicea", 80)
	lb := getTestLoadBalancer()
	pip := getTestPublicIP()
	nodes := []*v1.Node{}

	lb, updated, err := az.reconcileLoadBalancer(lb, &pip, testClusterName, &svc, nodes)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}

	svcUpdated := getTestService("servicea")
	lb, updated, err = az.reconcileLoadBalancer(lb, nil, testClusterName, &svcUpdated, nodes)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}

	if !updated {
		t.Error("Expected the loadbalancer to need an update")
	}

	// ensure we abandoned the frontend ip configuration
	if len(*lb.FrontendIPConfigurations) != 0 {
		t.Error("Expected the loadbalancer to have no frontend ip configuration")
	}

	validateLoadBalancer(t, lb, svcUpdated)
}

// Test removal of a port from an existing service.
func TestReconcileLoadBalancerRemovesPort(t *testing.T) {
	az := getTestCloud()
	svc := getTestService("servicea", 80, 443)
	pip := getTestPublicIP()
	nodes := []*v1.Node{}

	existingLoadBalancer := getTestLoadBalancer(svc)

	svcUpdated := getTestService("servicea", 80)
	updatedLoadBalancer, _, err := az.reconcileLoadBalancer(existingLoadBalancer, &pip, testClusterName, &svcUpdated, nodes)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}

	validateLoadBalancer(t, updatedLoadBalancer, svcUpdated)
}

// Test reconciliation of multiple services on same port
func TestReconcileLoadBalancerMultipleServices(t *testing.T) {
	az := getTestCloud()
	svc1 := getTestService("servicea", 80, 443)
	svc2 := getTestService("serviceb", 80)
	pip := getTestPublicIP()
	nodes := []*v1.Node{}

	existingLoadBalancer := getTestLoadBalancer()

	updatedLoadBalancer, _, err := az.reconcileLoadBalancer(existingLoadBalancer, &pip, testClusterName, &svc1, nodes)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}

	updatedLoadBalancer, _, err = az.reconcileLoadBalancer(updatedLoadBalancer, &pip, testClusterName, &svc2, nodes)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}

	validateLoadBalancer(t, updatedLoadBalancer, svc1, svc2)
}

func TestReconcileSecurityGroupNewServiceAddsPort(t *testing.T) {
	az := getTestCloud()
	svc1 := getTestService("serviceea", 80)

	sg := getTestSecurityGroup()

	sg, _, err := az.reconcileSecurityGroup(sg, testClusterName, &svc1)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}

	validateSecurityGroup(t, sg, svc1)
}

func TestReconcileSecurityGroupRemoveServiceRemovesPort(t *testing.T) {
	az := getTestCloud()
	svc := getTestService("servicea", 80, 443)

	sg := getTestSecurityGroup(svc)

	svcUpdated := getTestService("servicea", 80)
	sg, _, err := az.reconcileSecurityGroup(sg, testClusterName, &svcUpdated)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}

	validateSecurityGroup(t, sg, svcUpdated)
}

func TestReconcileSecurityWithSourceRanges(t *testing.T) {
	az := getTestCloud()
	svc := getTestService("servicea", 80, 443)
	svc.Spec.LoadBalancerSourceRanges = []string{
		"192.168.0.1/24",
		"10.0.0.1/32",
	}

	sg := getTestSecurityGroup(svc)
	sg, _, err := az.reconcileSecurityGroup(sg, testClusterName, &svc)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}

	validateSecurityGroup(t, sg, svc)
}

func getTestCloud() *Cloud {
	return &Cloud{
		Config: Config{
			TenantID:          "tenant",
			SubscriptionID:    "subscription",
			ResourceGroup:     "rg",
			Location:          "westus",
			VnetName:          "vnet",
			SubnetName:        "subnet",
			SecurityGroupName: "nsg",
			RouteTableName:    "rt",
		},
	}
}

func getBackendPort(port int32) int32 {
	return port + 10000
}

func getTestPublicIP() network.PublicIPAddress {
	pip := network.PublicIPAddress{}
	pip.ID = to.StringPtr("/this/is/a/public/ip/address/id")
	return pip
}

func getTestService(identifier string, requestedPorts ...int32) v1.Service {
	ports := []v1.ServicePort{}
	for _, port := range requestedPorts {
		ports = append(ports, v1.ServicePort{
			Name:     fmt.Sprintf("port-%d", port),
			Protocol: v1.ProtocolTCP,
			Port:     port,
			NodePort: getBackendPort(port),
		})
	}

	svc := v1.Service{
		Spec: v1.ServiceSpec{
			Type:  v1.ServiceTypeLoadBalancer,
			Ports: ports,
		},
	}
	svc.Name = identifier
	svc.Namespace = "default"
	svc.UID = types.UID(identifier)

	return svc
}

func getTestLoadBalancer(services ...v1.Service) network.LoadBalancer {
	rules := []network.LoadBalancingRule{}
	probes := []network.Probe{}

	for _, service := range services {
		for _, port := range service.Spec.Ports {
			ruleName := getRuleName(&service, port)
			rules = append(rules, network.LoadBalancingRule{
				Name: to.StringPtr(ruleName),
				LoadBalancingRulePropertiesFormat: &network.LoadBalancingRulePropertiesFormat{
					FrontendPort: to.Int32Ptr(port.Port),
					BackendPort:  to.Int32Ptr(port.Port),
				},
			})
			probes = append(probes, network.Probe{
				Name: to.StringPtr(ruleName),
				ProbePropertiesFormat: &network.ProbePropertiesFormat{
					Port: to.Int32Ptr(port.NodePort),
				},
			})
		}
	}

	lb := network.LoadBalancer{
		LoadBalancerPropertiesFormat: &network.LoadBalancerPropertiesFormat{
			LoadBalancingRules: &rules,
			Probes:             &probes,
		},
	}

	return lb
}

func getServiceSourceRanges(service *v1.Service) []string {
	if len(service.Spec.LoadBalancerSourceRanges) == 0 {
		return []string{"Internet"}
	}
	return service.Spec.LoadBalancerSourceRanges
}

func getTestSecurityGroup(services ...v1.Service) network.SecurityGroup {
	rules := []network.SecurityRule{}

	for _, service := range services {
		for _, port := range service.Spec.Ports {
			ruleName := getRuleName(&service, port)

			sources := getServiceSourceRanges(&service)
			for _, src := range sources {
				rules = append(rules, network.SecurityRule{
					Name: to.StringPtr(ruleName),
					SecurityRulePropertiesFormat: &network.SecurityRulePropertiesFormat{
						SourceAddressPrefix:  to.StringPtr(src),
						DestinationPortRange: to.StringPtr(fmt.Sprintf("%d", port.Port)),
					},
				})
			}
		}
	}

	sg := network.SecurityGroup{
		SecurityGroupPropertiesFormat: &network.SecurityGroupPropertiesFormat{
			SecurityRules: &rules,
		},
	}

	return sg
}

func validateLoadBalancer(t *testing.T, loadBalancer network.LoadBalancer, services ...v1.Service) {
	expectedRuleCount := 0
	for _, svc := range services {
		for _, wantedRule := range svc.Spec.Ports {
			expectedRuleCount++
			wantedRuleName := getRuleName(&svc, wantedRule)
			foundRule := false
			for _, actualRule := range *loadBalancer.LoadBalancingRules {
				if strings.EqualFold(*actualRule.Name, wantedRuleName) &&
					*actualRule.FrontendPort == wantedRule.Port &&
					*actualRule.BackendPort == wantedRule.Port {
					foundRule = true
					break
				}
			}
			if !foundRule {
				t.Errorf("Expected load balancer rule but didn't find it: %q", wantedRuleName)
			}

			foundProbe := false
			if serviceapi.NeedsHealthCheck(&svc) {
				path, port := serviceapi.GetServiceHealthCheckPathPort(&svc)
				for _, actualProbe := range *loadBalancer.Probes {
					if strings.EqualFold(*actualProbe.Name, wantedRuleName) &&
						*actualProbe.Port == port &&
						*actualProbe.RequestPath == path &&
						actualProbe.Protocol == network.ProbeProtocolHTTP {
						foundProbe = true
						break
					}
				}
			} else {
				for _, actualProbe := range *loadBalancer.Probes {
					if strings.EqualFold(*actualProbe.Name, wantedRuleName) &&
						*actualProbe.Port == wantedRule.NodePort {
						foundProbe = true
						break
					}
				}
			}
			if !foundProbe {
				for _, actualProbe := range *loadBalancer.Probes {
					t.Logf("Probe: %s %d", *actualProbe.Name, *actualProbe.Port)
				}
				t.Errorf("Expected loadbalancer probe but didn't find it: %q", wantedRuleName)
			}
		}
	}

	lenRules := len(*loadBalancer.LoadBalancingRules)
	if lenRules != expectedRuleCount {
		t.Errorf("Expected the loadbalancer to have %d rules. Found %d.\n%v", expectedRuleCount, lenRules, loadBalancer.LoadBalancingRules)
	}
	lenProbes := len(*loadBalancer.Probes)
	if lenProbes != expectedRuleCount {
		t.Errorf("Expected the loadbalancer to have %d probes. Found %d.", expectedRuleCount, lenProbes)
	}
}

func validateSecurityGroup(t *testing.T, securityGroup network.SecurityGroup, services ...v1.Service) {
	expectedRuleCount := 0
	for _, svc := range services {
		for _, wantedRule := range svc.Spec.Ports {
			sources := getServiceSourceRanges(&svc)

			for _, source := range sources {
				expectedRuleCount++
				wantedRuleName := getRuleName(&svc, wantedRule)
				foundRule := false
				for _, actualRule := range *securityGroup.SecurityRules {
					if strings.EqualFold(*actualRule.Name, wantedRuleName) &&
						*actualRule.SourceAddressPrefix == source &&
						*actualRule.DestinationPortRange == fmt.Sprintf("%d", wantedRule.Port) {
						foundRule = true
						break
					}
				}
				if !foundRule {
					t.Errorf("Expected security group rule but didn't find it: %q", wantedRuleName)
				}
			}
		}
	}

	lenRules := len(*securityGroup.SecurityRules)
	if lenRules != expectedRuleCount {
		t.Errorf("Expected the loadbalancer to have %d rules. Found %d.\n", expectedRuleCount, lenRules)
	}
}

func TestSecurityRulePriorityPicksNextAvailablePriority(t *testing.T) {
	rules := []network.SecurityRule{}

	var expectedPriority int32 = loadBalancerMinimumPriority + 50

	var i int32
	for i = loadBalancerMinimumPriority; i < expectedPriority; i++ {
		rules = append(rules, network.SecurityRule{
			SecurityRulePropertiesFormat: &network.SecurityRulePropertiesFormat{
				Priority: to.Int32Ptr(i),
			},
		})
	}

	priority, err := getNextAvailablePriority(rules)
	if err != nil {
		t.Errorf("Unexpectected error: %q", err)
	}

	if priority != expectedPriority {
		t.Errorf("Expected priority %d. Got priority %d.", expectedPriority, priority)
	}
}

func TestSecurityRulePriorityFailsIfExhausted(t *testing.T) {
	rules := []network.SecurityRule{}

	var i int32
	for i = loadBalancerMinimumPriority; i < loadBalancerMaximumPriority; i++ {
		rules = append(rules, network.SecurityRule{
			SecurityRulePropertiesFormat: &network.SecurityRulePropertiesFormat{
				Priority: to.Int32Ptr(i),
			},
		})
	}

	_, err := getNextAvailablePriority(rules)
	if err == nil {
		t.Error("Expectected an error. There are no priority levels left.")
	}
}

func TestProtocolTranslationTCP(t *testing.T) {
	proto := v1.ProtocolTCP
	transportProto, securityGroupProto, probeProto, err := getProtocolsFromKubernetesProtocol(proto)
	if err != nil {
		t.Error(err)
	}

	if transportProto != network.TransportProtocolTCP {
		t.Errorf("Expected TCP LoadBalancer Rule Protocol. Got %v", transportProto)
	}
	if securityGroupProto != network.TCP {
		t.Errorf("Expected TCP SecurityGroup Protocol. Got %v", transportProto)
	}
	if probeProto != network.ProbeProtocolTCP {
		t.Errorf("Expected TCP LoadBalancer Probe Protocol. Got %v", transportProto)
	}
}

func TestProtocolTranslationUDP(t *testing.T) {
	proto := v1.ProtocolUDP
	_, _, _, err := getProtocolsFromKubernetesProtocol(proto)
	if err == nil {
		t.Error("Expected an error. UDP is unsupported.")
	}
}

// Test Configuration deserialization (json)
func TestNewCloudFromJSON(t *testing.T) {
	config := `{
		"tenantId": "--tenant-id--",
		"subscriptionId": "--subscription-id--",
		"aadClientId": "--aad-client-id--",
		"aadClientSecret": "--aad-client-secret--",
		"resourceGroup": "--resource-group--",
		"location": "--location--",
		"subnetName": "--subnet-name--",
		"securityGroupName": "--security-group-name--",
		"vnetName": "--vnet-name--",
		"routeTableName": "--route-table-name--",
		"primaryAvailabilitySetName": "--primary-availability-set-name--"
	}`
	validateConfig(t, config)
}

// Test Configuration deserialization (yaml)
func TestNewCloudFromYAML(t *testing.T) {
	config := `
tenantId: --tenant-id--
subscriptionId: --subscription-id--
aadClientId: --aad-client-id--
aadClientSecret: --aad-client-secret--
resourceGroup: --resource-group--
location: --location--
subnetName: --subnet-name--
securityGroupName: --security-group-name--
vnetName: --vnet-name--
routeTableName: --route-table-name--
primaryAvailabilitySetName: --primary-availability-set-name--
`
	validateConfig(t, config)
}

func validateConfig(t *testing.T, config string) {
	configReader := strings.NewReader(config)
	cloud, err := NewCloud(configReader)
	if err != nil {
		t.Error(err)
	}

	azureCloud, ok := cloud.(*Cloud)
	if !ok {
		t.Error("NewCloud returned incorrect type")
	}

	if azureCloud.TenantID != "--tenant-id--" {
		t.Errorf("got incorrect value for TenantID")
	}
	if azureCloud.SubscriptionID != "--subscription-id--" {
		t.Errorf("got incorrect value for SubscriptionID")
	}
	if azureCloud.AADClientID != "--aad-client-id--" {
		t.Errorf("got incorrect value for AADClientID")
	}
	if azureCloud.AADClientSecret != "--aad-client-secret--" {
		t.Errorf("got incorrect value for AADClientSecret")
	}
	if azureCloud.ResourceGroup != "--resource-group--" {
		t.Errorf("got incorrect value for ResourceGroup")
	}
	if azureCloud.Location != "--location--" {
		t.Errorf("got incorrect value for Location")
	}
	if azureCloud.SubnetName != "--subnet-name--" {
		t.Errorf("got incorrect value for SubnetName")
	}
	if azureCloud.SecurityGroupName != "--security-group-name--" {
		t.Errorf("got incorrect value for SecurityGroupName")
	}
	if azureCloud.VnetName != "--vnet-name--" {
		t.Errorf("got incorrect value for VnetName")
	}
	if azureCloud.RouteTableName != "--route-table-name--" {
		t.Errorf("got incorrect value for RouteTableName")
	}
	if azureCloud.PrimaryAvailabilitySetName != "--primary-availability-set-name--" {
		t.Errorf("got incorrect value for PrimaryAvailabilitySetName")
	}
}

func TestDecodeInstanceInfo(t *testing.T) {
	response := `{"ID":"_azdev","UD":"0","FD":"99"}`

	faultDomain, err := readFaultDomain(strings.NewReader(response))
	if err != nil {
		t.Error("Unexpected error in ReadFaultDomain")
	}

	if faultDomain == nil {
		t.Error("Fault domain was unexpectedly nil")
	}

	if *faultDomain != "99" {
		t.Error("got incorrect fault domain")
	}
}

func TestFilterNodes(t *testing.T) {
	nodes := []compute.VirtualMachine{
		{Name: to.StringPtr("test")},
		{Name: to.StringPtr("test2")},
		{Name: to.StringPtr("3test")},
	}

	filteredNodes, err := filterNodes(nodes, "^test$")
	if err != nil {
		t.Errorf("Unexpeted error when filtering: %q", err)
	}

	if len(filteredNodes) != 1 {
		t.Error("Got too many nodes after filtering")
	}

	if *filteredNodes[0].Name != "test" {
		t.Error("Get the wrong node after filtering")
	}
}
