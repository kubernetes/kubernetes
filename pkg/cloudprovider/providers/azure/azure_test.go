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

	"github.com/Azure/azure-sdk-for-go/arm/network"
	"github.com/Azure/go-autorest/autorest/to"
)

var testClusterName = "testCluster"

// Test additional of a new service/port.
func TestReconcileLoadBalancerAddPort(t *testing.T) {
	az := getTestCloud()
	svc := getTestService("servicea", v1.ProtocolTCP, 80)
	configProperties := getTestPublicFipConfigurationProperties()
	lb := getTestLoadBalancer()
	nodes := []*v1.Node{}

	svc.Spec.Ports = append(svc.Spec.Ports, v1.ServicePort{
		Name:     fmt.Sprintf("port-udp-%d", 1234),
		Protocol: v1.ProtocolUDP,
		Port:     1234,
		NodePort: getBackendPort(1234),
	})

	lb, updated, err := az.reconcileLoadBalancer(lb, &configProperties, testClusterName, &svc, nodes)
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
	svc := getTestService("servicea", v1.ProtocolTCP, 80)
	svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
	svc.Spec.HealthCheckNodePort = int32(32456)
	configProperties := getTestPublicFipConfigurationProperties()
	lb := getTestLoadBalancer()

	nodes := []*v1.Node{}

	lb, updated, err := az.reconcileLoadBalancer(lb, &configProperties, testClusterName, &svc, nodes)
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
func TestReconcileLoadBalancerRemoveService(t *testing.T) {
	az := getTestCloud()
	svc := getTestService("servicea", v1.ProtocolTCP, 80, 443)
	lb := getTestLoadBalancer()
	configProperties := getTestPublicFipConfigurationProperties()
	nodes := []*v1.Node{}

	lb, updated, err := az.reconcileLoadBalancer(lb, &configProperties, testClusterName, &svc, nodes)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}
	validateLoadBalancer(t, lb, svc)

	lb, updated, err = az.reconcileLoadBalancer(lb, nil, testClusterName, &svc, nodes)
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

	validateLoadBalancer(t, lb)
}

// Test removing all service ports results in removing the frontend ip configuration
func TestReconcileLoadBalancerRemoveAllPortsRemovesFrontendConfig(t *testing.T) {
	az := getTestCloud()
	svc := getTestService("servicea", v1.ProtocolTCP, 80)
	lb := getTestLoadBalancer()
	configProperties := getTestPublicFipConfigurationProperties()
	nodes := []*v1.Node{}

	lb, updated, err := az.reconcileLoadBalancer(lb, &configProperties, testClusterName, &svc, nodes)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}
	validateLoadBalancer(t, lb, svc)

	svcUpdated := getTestService("servicea", v1.ProtocolTCP)
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
	svc := getTestService("servicea", v1.ProtocolTCP, 80, 443)
	configProperties := getTestPublicFipConfigurationProperties()
	nodes := []*v1.Node{}

	existingLoadBalancer := getTestLoadBalancer(svc)

	svcUpdated := getTestService("servicea", v1.ProtocolTCP, 80)
	updatedLoadBalancer, _, err := az.reconcileLoadBalancer(existingLoadBalancer, &configProperties, testClusterName, &svcUpdated, nodes)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}

	validateLoadBalancer(t, updatedLoadBalancer, svcUpdated)
}

// Test reconciliation of multiple services on same port
func TestReconcileLoadBalancerMultipleServices(t *testing.T) {
	az := getTestCloud()
	svc1 := getTestService("servicea", v1.ProtocolTCP, 80, 443)
	svc2 := getTestService("serviceb", v1.ProtocolTCP, 80)
	configProperties := getTestPublicFipConfigurationProperties()
	nodes := []*v1.Node{}

	existingLoadBalancer := getTestLoadBalancer()

	updatedLoadBalancer, _, err := az.reconcileLoadBalancer(existingLoadBalancer, &configProperties, testClusterName, &svc1, nodes)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}

	updatedLoadBalancer, _, err = az.reconcileLoadBalancer(updatedLoadBalancer, &configProperties, testClusterName, &svc2, nodes)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}

	validateLoadBalancer(t, updatedLoadBalancer, svc1, svc2)
}

func TestReconcileSecurityGroupNewServiceAddsPort(t *testing.T) {
	az := getTestCloud()
	svc1 := getTestService("serviceea", v1.ProtocolTCP, 80)

	sg := getTestSecurityGroup()

	sg, _, err := az.reconcileSecurityGroup(sg, testClusterName, &svc1, true)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}

	validateSecurityGroup(t, sg, svc1)
}

func TestReconcileSecurityGroupNewInternalServiceAddsPort(t *testing.T) {
	az := getTestCloud()
	svc1 := getInternalTestService("serviceea", 80)

	sg := getTestSecurityGroup()

	sg, _, err := az.reconcileSecurityGroup(sg, testClusterName, &svc1, true)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}

	validateSecurityGroup(t, sg, svc1)
}

func TestReconcileSecurityGroupRemoveService(t *testing.T) {
	service1 := getTestService("servicea", v1.ProtocolTCP, 81)
	service2 := getTestService("serviceb", v1.ProtocolTCP, 82)

	sg := getTestSecurityGroup(service1, service2)

	validateSecurityGroup(t, sg, service1, service2)
	az := getTestCloud()
	sg, _, err := az.reconcileSecurityGroup(sg, testClusterName, &service1, false)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}

	validateSecurityGroup(t, sg, service2)
}

func TestReconcileSecurityGroupRemoveServiceRemovesPort(t *testing.T) {
	az := getTestCloud()
	svc := getTestService("servicea", v1.ProtocolTCP, 80, 443)

	sg := getTestSecurityGroup(svc)

	svcUpdated := getTestService("servicea", v1.ProtocolTCP, 80)
	sg, _, err := az.reconcileSecurityGroup(sg, testClusterName, &svcUpdated, true)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}

	validateSecurityGroup(t, sg, svcUpdated)
}

func TestReconcileSecurityWithSourceRanges(t *testing.T) {
	az := getTestCloud()
	svc := getTestService("servicea", v1.ProtocolTCP, 80, 443)
	svc.Spec.LoadBalancerSourceRanges = []string{
		"192.168.0.0/24",
		"10.0.0.0/32",
	}

	sg := getTestSecurityGroup(svc)
	sg, _, err := az.reconcileSecurityGroup(sg, testClusterName, &svc, true)
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

func getTestPublicFipConfigurationProperties() network.FrontendIPConfigurationPropertiesFormat {
	return network.FrontendIPConfigurationPropertiesFormat{
		PublicIPAddress: &network.PublicIPAddress{ID: to.StringPtr("/this/is/a/public/ip/address/id")},
	}
}

func getTestService(identifier string, proto v1.Protocol, requestedPorts ...int32) v1.Service {
	ports := []v1.ServicePort{}
	for _, port := range requestedPorts {
		ports = append(ports, v1.ServicePort{
			Name:     fmt.Sprintf("port-tcp-%d", port),
			Protocol: proto,
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
	svc.Annotations = make(map[string]string)

	return svc
}

func getInternalTestService(identifier string, requestedPorts ...int32) v1.Service {
	svc := getTestService(identifier, v1.ProtocolTCP, requestedPorts...)
	svc.Annotations[ServiceAnnotationLoadBalancerInternal] = "true"

	return svc
}

func getTestLoadBalancer(services ...v1.Service) network.LoadBalancer {
	rules := []network.LoadBalancingRule{}
	probes := []network.Probe{}

	for _, service := range services {
		for _, port := range service.Spec.Ports {
			ruleName := getLoadBalancerRuleName(&service, port)
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
		if !requiresInternalLoadBalancer(service) {
			return []string{"Internet"}
		}
	}

	return service.Spec.LoadBalancerSourceRanges
}

func getTestSecurityGroup(services ...v1.Service) network.SecurityGroup {
	rules := []network.SecurityRule{}

	for _, service := range services {
		for _, port := range service.Spec.Ports {
			sources := getServiceSourceRanges(&service)
			for _, src := range sources {
				ruleName := getSecurityRuleName(&service, port, src)
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
	expectedFrontendIPCount := 0
	expectedProbeCount := 0
	for _, svc := range services {
		if len(svc.Spec.Ports) > 0 {
			expectedFrontendIPCount++
		}
		for _, wantedRule := range svc.Spec.Ports {
			expectedRuleCount++
			wantedRuleName := getLoadBalancerRuleName(&svc, wantedRule)
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

			// if UDP rule, there is no probe
			if wantedRule.Protocol == v1.ProtocolUDP {
				continue
			}

			expectedProbeCount++
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

	frontendIPCount := len(*loadBalancer.FrontendIPConfigurations)
	if frontendIPCount != expectedFrontendIPCount {
		t.Errorf("Expected the loadbalancer to have %d frontend IPs. Found %d.\n%v", expectedFrontendIPCount, frontendIPCount, loadBalancer.FrontendIPConfigurations)
	}

	lenRules := len(*loadBalancer.LoadBalancingRules)
	if lenRules != expectedRuleCount {
		t.Errorf("Expected the loadbalancer to have %d rules. Found %d.\n%v", expectedRuleCount, lenRules, loadBalancer.LoadBalancingRules)
	}

	lenProbes := len(*loadBalancer.Probes)
	if lenProbes != expectedProbeCount {
		t.Errorf("Expected the loadbalancer to have %d probes. Found %d.", expectedRuleCount, lenProbes)
	}
}

func validateSecurityGroup(t *testing.T, securityGroup network.SecurityGroup, services ...v1.Service) {
	expectedRuleCount := 0
	for _, svc := range services {
		for _, wantedRule := range svc.Spec.Ports {
			sources := getServiceSourceRanges(&svc)
			for _, source := range sources {
				wantedRuleName := getSecurityRuleName(&svc, wantedRule, source)
				expectedRuleCount++
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

	if *transportProto != network.TransportProtocolTCP {
		t.Errorf("Expected TCP LoadBalancer Rule Protocol. Got %v", transportProto)
	}
	if *securityGroupProto != network.TCP {
		t.Errorf("Expected TCP SecurityGroup Protocol. Got %v", transportProto)
	}
	if *probeProto != network.ProbeProtocolTCP {
		t.Errorf("Expected TCP LoadBalancer Probe Protocol. Got %v", transportProto)
	}
}

func TestProtocolTranslationUDP(t *testing.T) {
	proto := v1.ProtocolUDP
	transportProto, securityGroupProto, probeProto, _ := getProtocolsFromKubernetesProtocol(proto)
	if *transportProto != network.TransportProtocolUDP {
		t.Errorf("Expected UDP LoadBalancer Rule Protocol. Got %v", transportProto)
	}
	if *securityGroupProto != network.UDP {
		t.Errorf("Expected UDP SecurityGroup Protocol. Got %v", transportProto)
	}
	if probeProto != nil {
		t.Errorf("Expected UDP LoadBalancer Probe Protocol. Got %v", transportProto)
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
		"primaryAvailabilitySetName": "--primary-availability-set-name--",
		"cloudProviderBackoff": true,
		"cloudProviderBackoffRetries": 6,
		"cloudProviderBackoffExponent": 1.5,
		"cloudProviderBackoffDuration": 5,
		"cloudProviderBackoffJitter": 1.0,
		"cloudProviderRatelimit": true,
		"cloudProviderRateLimitQPS": 0.5,
		"cloudProviderRateLimitBucket": 5
	}`
	validateConfig(t, config)
}

// Test Backoff and Rate Limit defaults (json)
func TestCloudDefaultConfigFromJSON(t *testing.T) {
	config := `{}`

	validateEmptyConfig(t, config)
}

// Test Backoff and Rate Limit defaults (yaml)
func TestCloudDefaultConfigFromYAML(t *testing.T) {
	config := ``

	validateEmptyConfig(t, config)
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
cloudProviderBackoff: true
cloudProviderBackoffRetries: 6
cloudProviderBackoffExponent: 1.5
cloudProviderBackoffDuration: 5
cloudProviderBackoffJitter: 1.0
cloudProviderRatelimit: true
cloudProviderRateLimitQPS: 0.5
cloudProviderRateLimitBucket: 5
`
	validateConfig(t, config)
}

func validateConfig(t *testing.T, config string) {
	azureCloud := getCloudFromConfig(t, config)

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
	if azureCloud.CloudProviderBackoff != true {
		t.Errorf("got incorrect value for CloudProviderBackoff")
	}
	if azureCloud.CloudProviderBackoffRetries != 6 {
		t.Errorf("got incorrect value for CloudProviderBackoffRetries")
	}
	if azureCloud.CloudProviderBackoffExponent != 1.5 {
		t.Errorf("got incorrect value for CloudProviderBackoffExponent")
	}
	if azureCloud.CloudProviderBackoffDuration != 5 {
		t.Errorf("got incorrect value for CloudProviderBackoffDuration")
	}
	if azureCloud.CloudProviderBackoffJitter != 1.0 {
		t.Errorf("got incorrect value for CloudProviderBackoffJitter")
	}
	if azureCloud.CloudProviderRateLimit != true {
		t.Errorf("got incorrect value for CloudProviderRateLimit")
	}
	if azureCloud.CloudProviderRateLimitQPS != 0.5 {
		t.Errorf("got incorrect value for CloudProviderRateLimitQPS")
	}
	if azureCloud.CloudProviderRateLimitBucket != 5 {
		t.Errorf("got incorrect value for CloudProviderRateLimitBucket")
	}
}

func getCloudFromConfig(t *testing.T, config string) *Cloud {
	configReader := strings.NewReader(config)
	cloud, err := NewCloud(configReader)
	if err != nil {
		t.Error(err)
	}
	azureCloud, ok := cloud.(*Cloud)
	if !ok {
		t.Error("NewCloud returned incorrect type")
	}
	return azureCloud
}

// TODO include checks for other appropriate default config parameters
func validateEmptyConfig(t *testing.T, config string) {
	azureCloud := getCloudFromConfig(t, config)

	// backoff should be disabled by default if not explicitly enabled in config
	if azureCloud.CloudProviderBackoff != false {
		t.Errorf("got incorrect value for CloudProviderBackoff")
	}

	// rate limits should be disabled by default if not explicitly enabled in config
	if azureCloud.CloudProviderRateLimit != false {
		t.Errorf("got incorrect value for CloudProviderRateLimit")
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

func TestSplitProviderID(t *testing.T) {
	providers := []struct {
		providerID string
		name       types.NodeName

		fail bool
	}{
		{
			providerID: CloudProviderName + ":///subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/myResourceGroupName/providers/Microsoft.Compute/virtualMachines/k8s-agent-AAAAAAAA-0",
			name:       "/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/myResourceGroupName/providers/Microsoft.Compute/virtualMachines/k8s-agent-AAAAAAAA-0",
			fail:       false,
		},
		{
			providerID: CloudProviderName + ":/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/myResourceGroupName/providers/Microsoft.Compute/virtualMachines/k8s-agent-AAAAAAAA-0",
			name:       "",
			fail:       true,
		},
		{
			providerID: CloudProviderName + "://",
			name:       "",
			fail:       true,
		},
		{
			providerID: ":///subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/myResourceGroupName/providers/Microsoft.Compute/virtualMachines/k8s-agent-AAAAAAAA-0",
			name:       "",
			fail:       true,
		},
		{
			providerID: "aws:///subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/myResourceGroupName/providers/Microsoft.Compute/virtualMachines/k8s-agent-AAAAAAAA-0",
			name:       "",
			fail:       true,
		},
	}

	for _, test := range providers {
		name, err := splitProviderID(test.providerID)
		if (err != nil) != test.fail {
			t.Errorf("Expected to failt=%t, with pattern %v", test.fail, test)
		}

		if test.fail {
			continue
		}

		if name != test.name {
			t.Errorf("Expected %v, but got %v", test.name, name)
		}

	}
}
