// +build !providerless

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
	"bytes"
	"context"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-07-01/compute"
	"github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-06-01/network"
	"github.com/Azure/go-autorest/autorest/to"
	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/record"
	servicehelpers "k8s.io/cloud-provider/service/helpers"
	"k8s.io/legacy-cloud-providers/azure/auth"
)

var testClusterName = "testCluster"

func TestParseConfig(t *testing.T) {
	azureConfig := `{
		"aadClientCertPassword": "aadClientCertPassword",
		"aadClientCertPath": "aadClientCertPath",
		"aadClientId": "aadClientId",
		"aadClientSecret": "aadClientSecret",
		"cloud":"AzurePublicCloud",
		"cloudProviderBackoff": true,
		"cloudProviderBackoffDuration": 1,
		"cloudProviderBackoffExponent": 1,
		"cloudProviderBackoffJitter": 1,
		"cloudProviderBackoffRetries": 1,
		"cloudProviderRatelimit": true,
		"cloudProviderRateLimitBucket": 1,
		"CloudProviderRateLimitBucketWrite": 1,
		"cloudProviderRateLimitQPS": 1,
		"CloudProviderRateLimitQPSWrite": 1,
		"location": "location",
		"maximumLoadBalancerRuleCount": 1,
		"primaryAvailabilitySetName": "primaryAvailabilitySetName",
		"primaryScaleSetName": "primaryScaleSetName",
		"resourceGroup": "resourceGroup",
		"routeTableName": "routeTableName",
		"routeTableResourceGroup": "routeTableResourceGroup",
		"securityGroupName": "securityGroupName",
		"subnetName": "subnetName",
		"subscriptionId": "subscriptionId",
		"tenantId": "tenantId",
		"useInstanceMetadata": true,
		"useManagedIdentityExtension": true,
		"vnetName": "vnetName",
		"vnetResourceGroup": "vnetResourceGroup",
		vmType: "standard"
	}`
	expected := &Config{
		AzureAuthConfig: auth.AzureAuthConfig{
			AADClientCertPassword:       "aadClientCertPassword",
			AADClientCertPath:           "aadClientCertPath",
			AADClientID:                 "aadClientId",
			AADClientSecret:             "aadClientSecret",
			Cloud:                       "AzurePublicCloud",
			SubscriptionID:              "subscriptionId",
			TenantID:                    "tenantId",
			UseManagedIdentityExtension: true,
		},
		CloudProviderBackoff:              true,
		CloudProviderBackoffDuration:      1,
		CloudProviderBackoffExponent:      1,
		CloudProviderBackoffJitter:        1,
		CloudProviderBackoffRetries:       1,
		CloudProviderRateLimit:            true,
		CloudProviderRateLimitBucket:      1,
		CloudProviderRateLimitBucketWrite: 1,
		CloudProviderRateLimitQPS:         1,
		CloudProviderRateLimitQPSWrite:    1,
		Location:                          "location",
		MaximumLoadBalancerRuleCount:      1,
		PrimaryAvailabilitySetName:        "primaryAvailabilitySetName",
		PrimaryScaleSetName:               "primaryScaleSetName",
		ResourceGroup:                     "resourcegroup",
		RouteTableName:                    "routeTableName",
		RouteTableResourceGroup:           "routeTableResourceGroup",
		SecurityGroupName:                 "securityGroupName",
		SubnetName:                        "subnetName",
		UseInstanceMetadata:               true,
		VMType:                            "standard",
		VnetName:                          "vnetName",
		VnetResourceGroup:                 "vnetResourceGroup",
	}

	buffer := bytes.NewBufferString(azureConfig)
	config, err := parseConfig(buffer)
	assert.NoError(t, err)
	assert.Equal(t, expected, config)
}

// Test flipServiceInternalAnnotation
func TestFlipServiceInternalAnnotation(t *testing.T) {
	svc := getTestService("servicea", v1.ProtocolTCP, nil, 80)
	svcUpdated := flipServiceInternalAnnotation(&svc)
	if !requiresInternalLoadBalancer(svcUpdated) {
		t.Errorf("Expected svc to be an internal service")
	}
	svcUpdated = flipServiceInternalAnnotation(svcUpdated)
	if requiresInternalLoadBalancer(svcUpdated) {
		t.Errorf("Expected svc to be an external service")
	}

	svc2 := getInternalTestService("serviceb", 8081)
	svc2Updated := flipServiceInternalAnnotation(&svc2)
	if requiresInternalLoadBalancer(svc2Updated) {
		t.Errorf("Expected svc to be an external service")
	}

	svc2Updated = flipServiceInternalAnnotation(svc2Updated)
	if !requiresInternalLoadBalancer(svc2Updated) {
		t.Errorf("Expected svc to be an internal service")
	}
}

// Test additional of a new service/port.
func TestAddPort(t *testing.T) {
	az := getTestCloud()
	svc := getTestService("servicea", v1.ProtocolTCP, nil, 80)
	clusterResources := getClusterResources(az, 1, 1)

	svc.Spec.Ports = append(svc.Spec.Ports, v1.ServicePort{
		Name:     fmt.Sprintf("port-udp-%d", 1234),
		Protocol: v1.ProtocolUDP,
		Port:     1234,
		NodePort: getBackendPort(1234),
	})

	lb, err := az.reconcileLoadBalancer(testClusterName, &svc, clusterResources.nodes, true /* wantLb */)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}

	// ensure we got a frontend ip configuration
	if len(*lb.FrontendIPConfigurations) != 1 {
		t.Error("Expected the loadbalancer to have a frontend ip configuration")
	}

	validateLoadBalancer(t, lb, svc)
}

func TestLoadBalancerInternalServiceModeSelection(t *testing.T) {
	testLoadBalancerServiceDefaultModeSelection(t, true)
	testLoadBalancerServiceAutoModeSelection(t, true)
	testLoadBalancerServicesSpecifiedSelection(t, true)
	testLoadBalancerMaxRulesServices(t, true)
	testLoadBalancerServiceAutoModeDeleteSelection(t, true)
}

func TestLoadBalancerExternalServiceModeSelection(t *testing.T) {
	testLoadBalancerServiceDefaultModeSelection(t, false)
	testLoadBalancerServiceAutoModeSelection(t, false)
	testLoadBalancerServicesSpecifiedSelection(t, false)
	testLoadBalancerMaxRulesServices(t, false)
	testLoadBalancerServiceAutoModeDeleteSelection(t, false)
}

func testLoadBalancerServiceDefaultModeSelection(t *testing.T, isInternal bool) {
	az := getTestCloud()
	const vmCount = 8
	const availabilitySetCount = 4
	const serviceCount = 9

	clusterResources := getClusterResources(az, vmCount, availabilitySetCount)
	getTestSecurityGroup(az)

	for index := 1; index <= serviceCount; index++ {
		svcName := fmt.Sprintf("service-%d", index)
		var svc v1.Service
		if isInternal {
			svc = getInternalTestService(svcName, 8081)
			addTestSubnet(t, az, &svc)
		} else {
			svc = getTestService(svcName, v1.ProtocolTCP, nil, 8081)
		}

		lbStatus, err := az.EnsureLoadBalancer(context.TODO(), testClusterName, &svc, clusterResources.nodes)
		if err != nil {
			t.Errorf("Unexpected error: %q", err)
		}
		if lbStatus == nil {
			t.Errorf("Unexpected error: %s", svcName)
		}

		expectedLBName := testClusterName
		if isInternal {
			expectedLBName = testClusterName + "-internal"
		}

		ctx, cancel := getContextWithCancel()
		defer cancel()
		result, _ := az.LoadBalancerClient.List(ctx, az.Config.ResourceGroup)
		lb := result[0]
		lbCount := len(result)
		expectedNumOfLB := 1
		if lbCount != expectedNumOfLB {
			t.Errorf("Unexpected number of LB's: Expected (%d) Found (%d)", expectedNumOfLB, lbCount)
		}

		if !strings.EqualFold(*lb.Name, expectedLBName) {
			t.Errorf("lb name should be the default LB name Extected (%s) Fouund (%s)", expectedLBName, *lb.Name)
		}

		ruleCount := len(*lb.LoadBalancingRules)
		if ruleCount != index {
			t.Errorf("lb rule count should be equal to nuber of services deployed, expected (%d) Found (%d)", index, ruleCount)
		}
	}
}

// Validate even distribution of external services across load balancers
// based on number of availability sets
func testLoadBalancerServiceAutoModeSelection(t *testing.T, isInternal bool) {
	az := getTestCloud()
	const vmCount = 8
	const availabilitySetCount = 4
	const serviceCount = 9

	clusterResources := getClusterResources(az, vmCount, availabilitySetCount)
	getTestSecurityGroup(az)

	for index := 1; index <= serviceCount; index++ {
		svcName := fmt.Sprintf("service-%d", index)
		var svc v1.Service
		if isInternal {
			svc = getInternalTestService(svcName, 8081)
			addTestSubnet(t, az, &svc)
		} else {
			svc = getTestService(svcName, v1.ProtocolTCP, nil, 8081)
		}
		setLoadBalancerAutoModeAnnotation(&svc)
		lbStatus, err := az.EnsureLoadBalancer(context.TODO(), testClusterName, &svc, clusterResources.nodes)
		if err != nil {
			t.Errorf("Unexpected error: %q", err)
		}
		if lbStatus == nil {
			t.Errorf("Unexpected error: %s", svcName)
		}

		// expected is MIN(index, availabilitySetCount)
		expectedNumOfLB := int(math.Min(float64(index), float64(availabilitySetCount)))
		ctx, cancel := getContextWithCancel()
		defer cancel()
		result, _ := az.LoadBalancerClient.List(ctx, az.Config.ResourceGroup)
		lbCount := len(result)
		if lbCount != expectedNumOfLB {
			t.Errorf("Unexpected number of LB's: Expected (%d) Found (%d)", expectedNumOfLB, lbCount)
		}

		maxRules := 0
		minRules := serviceCount
		for _, lb := range result {
			ruleCount := len(*lb.LoadBalancingRules)
			if ruleCount < minRules {
				minRules = ruleCount
			}
			if ruleCount > maxRules {
				maxRules = ruleCount
			}
		}

		delta := maxRules - minRules
		if delta > 1 {
			t.Errorf("Unexpected min or max rule in LB's in resource group: Service Index (%d) Min (%d) Max(%d)", index, minRules, maxRules)
		}
	}
}

// Validate availability set selection of services across load balancers
// based on provided availability sets through service annotation
// The scenario is that there are 4 availability sets in the agent pool but the
// services will be assigned load balancers that are part of the provided availability sets
// specified in service annotation
func testLoadBalancerServicesSpecifiedSelection(t *testing.T, isInternal bool) {
	az := getTestCloud()
	const vmCount = 8
	const availabilitySetCount = 4
	const serviceCount = 9

	clusterResources := getClusterResources(az, vmCount, availabilitySetCount)
	getTestSecurityGroup(az)

	selectedAvailabilitySetName1 := getAvailabilitySetName(az, 1, availabilitySetCount)
	selectedAvailabilitySetName2 := getAvailabilitySetName(az, 2, availabilitySetCount)
	for index := 1; index <= serviceCount; index++ {
		svcName := fmt.Sprintf("service-%d", index)
		var svc v1.Service
		if isInternal {
			svc = getInternalTestService(svcName, 8081)
			addTestSubnet(t, az, &svc)
		} else {
			svc = getTestService(svcName, v1.ProtocolTCP, nil, 8081)
		}
		lbMode := fmt.Sprintf("%s,%s", selectedAvailabilitySetName1, selectedAvailabilitySetName2)
		setLoadBalancerModeAnnotation(&svc, lbMode)

		lbStatus, err := az.EnsureLoadBalancer(context.TODO(), testClusterName, &svc, clusterResources.nodes)
		if err != nil {
			t.Errorf("Unexpected error: %q", err)
		}
		if lbStatus == nil {
			t.Errorf("Unexpected error: %s", svcName)
		}

		// expected is MIN(index, 2)
		expectedNumOfLB := int(math.Min(float64(index), float64(2)))
		ctx, cancel := getContextWithCancel()
		defer cancel()
		result, _ := az.LoadBalancerClient.List(ctx, az.Config.ResourceGroup)
		lbCount := len(result)
		if lbCount != expectedNumOfLB {
			t.Errorf("Unexpected number of LB's: Expected (%d) Found (%d)", expectedNumOfLB, lbCount)
		}
	}
}

func testLoadBalancerMaxRulesServices(t *testing.T, isInternal bool) {
	az := getTestCloud()
	const vmCount = 1
	const availabilitySetCount = 1

	clusterResources := getClusterResources(az, vmCount, availabilitySetCount)
	getTestSecurityGroup(az)

	az.Config.MaximumLoadBalancerRuleCount = 1

	for index := 1; index <= az.Config.MaximumLoadBalancerRuleCount; index++ {
		svcName := fmt.Sprintf("service-%d", index)
		var svc v1.Service
		if isInternal {
			svc = getInternalTestService(svcName, 8081)
			addTestSubnet(t, az, &svc)
		} else {
			svc = getTestService(svcName, v1.ProtocolTCP, nil, 8081)
		}

		lbStatus, err := az.EnsureLoadBalancer(context.TODO(), testClusterName, &svc, clusterResources.nodes)
		if err != nil {
			t.Errorf("Unexpected error: %q", err)
		}
		if lbStatus == nil {
			t.Errorf("Unexpected error: %s", svcName)
		}

		// expected is MIN(index, az.Config.MaximumLoadBalancerRuleCount)
		expectedNumOfLBRules := int(math.Min(float64(index), float64(az.Config.MaximumLoadBalancerRuleCount)))
		ctx, cancel := getContextWithCancel()
		defer cancel()
		result, _ := az.LoadBalancerClient.List(ctx, az.Config.ResourceGroup)
		lbCount := len(result)
		if lbCount != expectedNumOfLBRules {
			t.Errorf("Unexpected number of LB's: Expected (%d) Found (%d)", expectedNumOfLBRules, lbCount)
		}
	}

	// validate adding a new service fails since it will exceed the max limit on LB
	svcName := fmt.Sprintf("service-%d", az.Config.MaximumLoadBalancerRuleCount+1)
	var svc v1.Service
	if isInternal {
		svc = getInternalTestService(svcName, 8081)
		addTestSubnet(t, az, &svc)
	} else {
		svc = getTestService(svcName, v1.ProtocolTCP, nil, 8081)
	}
	_, err := az.EnsureLoadBalancer(context.TODO(), testClusterName, &svc, clusterResources.nodes)
	if err == nil {
		t.Errorf("Expect any new service to fail as max limit in lb has reached")
	} else {
		expectedErrMessageSubString := "all available load balancers have exceeded maximum rule limit"
		if !strings.Contains(err.Error(), expectedErrMessageSubString) {
			t.Errorf("Error message returned is not expected, expected sub string=%s, actual error message=%v", expectedErrMessageSubString, err)
		}
	}
}

// Validate service deletion in lb auto selection mode
func testLoadBalancerServiceAutoModeDeleteSelection(t *testing.T, isInternal bool) {
	az := getTestCloud()
	const vmCount = 8
	const availabilitySetCount = 4
	const serviceCount = 9

	clusterResources := getClusterResources(az, vmCount, availabilitySetCount)
	getTestSecurityGroup(az)

	for index := 1; index <= serviceCount; index++ {
		svcName := fmt.Sprintf("service-%d", index)
		var svc v1.Service
		if isInternal {
			svc = getInternalTestService(svcName, 8081)
			addTestSubnet(t, az, &svc)
		} else {
			svc = getTestService(svcName, v1.ProtocolTCP, nil, 8081)
		}
		setLoadBalancerAutoModeAnnotation(&svc)
		lbStatus, err := az.EnsureLoadBalancer(context.TODO(), testClusterName, &svc, clusterResources.nodes)
		if err != nil {
			t.Errorf("Unexpected error: %q", err)
		}
		if lbStatus == nil {
			t.Errorf("Unexpected error: %s", svcName)
		}
	}

	for index := serviceCount; index >= 1; index-- {
		svcName := fmt.Sprintf("service-%d", index)
		var svc v1.Service
		if isInternal {
			svc = getInternalTestService(svcName, 8081)
			addTestSubnet(t, az, &svc)
		} else {
			svc = getTestService(svcName, v1.ProtocolTCP, nil, 8081)
		}

		setLoadBalancerAutoModeAnnotation(&svc)

		// expected is MIN(index, availabilitySetCount)
		expectedNumOfLB := int(math.Min(float64(index), float64(availabilitySetCount)))
		ctx, cancel := getContextWithCancel()
		defer cancel()
		result, _ := az.LoadBalancerClient.List(ctx, az.Config.ResourceGroup)
		lbCount := len(result)
		if lbCount != expectedNumOfLB {
			t.Errorf("Unexpected number of LB's: Expected (%d) Found (%d)", expectedNumOfLB, lbCount)
		}

		err := az.EnsureLoadBalancerDeleted(context.TODO(), testClusterName, &svc)
		if err != nil {
			t.Errorf("Unexpected error: %q", err)
		}
	}
}

// Test addition of a new service on an internal LB with a subnet.
func TestReconcileLoadBalancerAddServiceOnInternalSubnet(t *testing.T) {
	az := getTestCloud()
	clusterResources := getClusterResources(az, 1, 1)
	svc := getInternalTestService("servicea", 80)
	addTestSubnet(t, az, &svc)

	lb, err := az.reconcileLoadBalancer(testClusterName, &svc, clusterResources.nodes, true /* wantLb */)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}

	// ensure we got a frontend ip configuration
	if len(*lb.FrontendIPConfigurations) != 1 {
		t.Error("Expected the loadbalancer to have a frontend ip configuration")
	}

	validateLoadBalancer(t, lb, svc)
}

func TestReconcileSecurityGroupFromAnyDestinationAddressPrefixToLoadBalancerIP(t *testing.T) {
	az := getTestCloud()
	svc1 := getTestService("serviceea", v1.ProtocolTCP, nil, 80)
	svc1.Spec.LoadBalancerIP = "192.168.0.0"
	sg := getTestSecurityGroup(az)
	// Simulate a pre-Kubernetes 1.8 NSG, where we do not specify the destination address prefix
	sg, err := az.reconcileSecurityGroup(testClusterName, &svc1, to.StringPtr(""), true)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}
	sg, err = az.reconcileSecurityGroup(testClusterName, &svc1, to.StringPtr(svc1.Spec.LoadBalancerIP), true)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}
	validateSecurityGroup(t, sg, svc1)
}

func TestReconcileSecurityGroupDynamicLoadBalancerIP(t *testing.T) {
	az := getTestCloud()
	svc1 := getTestService("servicea", v1.ProtocolTCP, nil, 80)
	svc1.Spec.LoadBalancerIP = ""
	sg := getTestSecurityGroup(az)
	dynamicallyAssignedIP := "192.168.0.0"
	sg, err := az.reconcileSecurityGroup(testClusterName, &svc1, to.StringPtr(dynamicallyAssignedIP), true)
	if err != nil {
		t.Errorf("unexpected error: %q", err)
	}
	validateSecurityGroup(t, sg, svc1)
}

// Test addition of services on an internal LB using both default and explicit subnets.
func TestReconcileLoadBalancerAddServicesOnMultipleSubnets(t *testing.T) {
	az := getTestCloud()
	clusterResources := getClusterResources(az, 1, 1)
	svc1 := getTestService("service1", v1.ProtocolTCP, nil, 8081)
	svc2 := getInternalTestService("service2", 8081)

	// Internal and External service cannot reside on the same LB resource
	addTestSubnet(t, az, &svc2)

	// svc1 is using LB without "-internal" suffix
	lb, err := az.reconcileLoadBalancer(testClusterName, &svc1, clusterResources.nodes, true /* wantLb */)
	if err != nil {
		t.Errorf("Unexpected error reconciling svc1: %q", err)
	}

	// ensure we got a frontend ip configuration for each service
	if len(*lb.FrontendIPConfigurations) != 1 {
		t.Error("Expected the loadbalancer to have 1 frontend ip configurations")
	}

	validateLoadBalancer(t, lb, svc1)

	// svc2 is using LB with "-internal" suffix
	lb, err = az.reconcileLoadBalancer(testClusterName, &svc2, nil, true /* wantLb */)
	if err != nil {
		t.Errorf("Unexpected error reconciling svc2: %q", err)
	}

	// ensure we got a frontend ip configuration for each service
	if len(*lb.FrontendIPConfigurations) != 1 {
		t.Error("Expected the loadbalancer to have 1 frontend ip configurations")
	}

	validateLoadBalancer(t, lb, svc2)
}

// Test moving a service exposure from one subnet to another.
func TestReconcileLoadBalancerEditServiceSubnet(t *testing.T) {
	az := getTestCloud()
	clusterResources := getClusterResources(az, 1, 1)
	svc := getInternalTestService("service1", 8081)
	addTestSubnet(t, az, &svc)

	lb, err := az.reconcileLoadBalancer(testClusterName, &svc, clusterResources.nodes, true /* wantLb */)
	if err != nil {
		t.Errorf("Unexpected error reconciling initial svc: %q", err)
	}

	validateLoadBalancer(t, lb, svc)

	svc.Annotations[ServiceAnnotationLoadBalancerInternalSubnet] = "NewSubnet"
	addTestSubnet(t, az, &svc)

	lb, err = az.reconcileLoadBalancer(testClusterName, &svc, clusterResources.nodes, true /* wantLb */)
	if err != nil {
		t.Errorf("Unexpected error reconciling edits to svc: %q", err)
	}

	// ensure we got a frontend ip configuration for the service
	if len(*lb.FrontendIPConfigurations) != 1 {
		t.Error("Expected the loadbalancer to have 1 frontend ip configuration")
	}

	validateLoadBalancer(t, lb, svc)
}

func TestReconcileLoadBalancerNodeHealth(t *testing.T) {
	az := getTestCloud()
	clusterResources := getClusterResources(az, 1, 1)
	svc := getTestService("servicea", v1.ProtocolTCP, nil, 80)
	svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
	svc.Spec.HealthCheckNodePort = int32(32456)

	lb, err := az.reconcileLoadBalancer(testClusterName, &svc, clusterResources.nodes, true /* wantLb */)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
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
	clusterResources := getClusterResources(az, 1, 1)
	svc := getTestService("servicea", v1.ProtocolTCP, nil, 80, 443)

	lb, err := az.reconcileLoadBalancer(testClusterName, &svc, clusterResources.nodes, true /* wantLb */)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}

	lb, err = az.reconcileLoadBalancer(testClusterName, &svc, clusterResources.nodes, false /* wantLb */)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
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
	clusterResources := getClusterResources(az, 1, 1)
	svc := getTestService("servicea", v1.ProtocolTCP, nil, 80)

	lb, err := az.reconcileLoadBalancer(testClusterName, &svc, clusterResources.nodes, true /* wantLb */)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}
	validateLoadBalancer(t, lb, svc)

	svcUpdated := getTestService("servicea", v1.ProtocolTCP, nil)
	lb, err = az.reconcileLoadBalancer(testClusterName, &svcUpdated, clusterResources.nodes, false /* wantLb*/)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
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
	clusterResources := getClusterResources(az, 1, 1)

	svc := getTestService("servicea", v1.ProtocolTCP, nil, 80, 443)
	lb, err := az.reconcileLoadBalancer(testClusterName, &svc, clusterResources.nodes, true /* wantLb */)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}

	svcUpdated := getTestService("servicea", v1.ProtocolTCP, nil, 80)
	lb, err = az.reconcileLoadBalancer(testClusterName, &svcUpdated, clusterResources.nodes, true /* wantLb */)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}

	validateLoadBalancer(t, lb, svcUpdated)
}

// Test reconciliation of multiple services on same port
func TestReconcileLoadBalancerMultipleServices(t *testing.T) {
	az := getTestCloud()
	clusterResources := getClusterResources(az, 1, 1)
	svc1 := getTestService("servicea", v1.ProtocolTCP, nil, 80, 443)
	svc2 := getTestService("serviceb", v1.ProtocolTCP, nil, 80)

	updatedLoadBalancer, err := az.reconcileLoadBalancer(testClusterName, &svc1, clusterResources.nodes, true /* wantLb */)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}

	updatedLoadBalancer, err = az.reconcileLoadBalancer(testClusterName, &svc2, clusterResources.nodes, true /* wantLb */)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}

	validateLoadBalancer(t, updatedLoadBalancer, svc1, svc2)
}

func findLBRuleForPort(lbRules []network.LoadBalancingRule, port int32) (network.LoadBalancingRule, error) {
	for _, lbRule := range lbRules {
		if *lbRule.FrontendPort == port {
			return lbRule, nil
		}
	}
	return network.LoadBalancingRule{}, fmt.Errorf("Expected LB rule with port %d but none found", port)
}

func TestServiceDefaultsToNoSessionPersistence(t *testing.T) {
	az := getTestCloud()
	svc := getTestService("service-sa-omitted", v1.ProtocolTCP, nil, 7170)
	clusterResources := getClusterResources(az, 1, 1)

	lb, err := az.reconcileLoadBalancer(testClusterName, &svc, clusterResources.nodes, true /* wantLb */)
	if err != nil {
		t.Errorf("Unexpected error reconciling svc1: %q", err)
	}
	validateLoadBalancer(t, lb, svc)
	lbRule, err := findLBRuleForPort(*lb.LoadBalancingRules, 7170)
	if err != nil {
		t.Error(err)
	}

	if lbRule.LoadDistribution != network.LoadDistributionDefault {
		t.Errorf("Expected LB rule to have default load distribution but was %s", lbRule.LoadDistribution)
	}
}

func TestServiceRespectsNoSessionAffinity(t *testing.T) {
	az := getTestCloud()
	svc := getTestService("service-sa-none", v1.ProtocolTCP, nil, 7170)
	svc.Spec.SessionAffinity = v1.ServiceAffinityNone
	clusterResources := getClusterResources(az, 1, 1)

	lb, err := az.reconcileLoadBalancer(testClusterName, &svc, clusterResources.nodes, true /* wantLb */)
	if err != nil {
		t.Errorf("Unexpected error reconciling svc1: %q", err)
	}

	validateLoadBalancer(t, lb, svc)

	lbRule, err := findLBRuleForPort(*lb.LoadBalancingRules, 7170)
	if err != nil {
		t.Error(err)
	}

	if lbRule.LoadDistribution != network.LoadDistributionDefault {
		t.Errorf("Expected LB rule to have default load distribution but was %s", lbRule.LoadDistribution)
	}
}

func TestServiceRespectsClientIPSessionAffinity(t *testing.T) {
	az := getTestCloud()
	svc := getTestService("service-sa-clientip", v1.ProtocolTCP, nil, 7170)
	svc.Spec.SessionAffinity = v1.ServiceAffinityClientIP
	clusterResources := getClusterResources(az, 1, 1)

	lb, err := az.reconcileLoadBalancer(testClusterName, &svc, clusterResources.nodes, true /* wantLb */)
	if err != nil {
		t.Errorf("Unexpected error reconciling svc1: %q", err)
	}

	validateLoadBalancer(t, lb, svc)

	lbRule, err := findLBRuleForPort(*lb.LoadBalancingRules, 7170)
	if err != nil {
		t.Error(err)
	}

	if lbRule.LoadDistribution != network.LoadDistributionSourceIP {
		t.Errorf("Expected LB rule to have SourceIP load distribution but was %s", lbRule.LoadDistribution)
	}
}

func TestReconcileSecurityGroupNewServiceAddsPort(t *testing.T) {
	az := getTestCloud()
	getTestSecurityGroup(az)
	svc1 := getTestService("servicea", v1.ProtocolTCP, nil, 80)
	clusterResources := getClusterResources(az, 1, 1)
	lb, _ := az.reconcileLoadBalancer(testClusterName, &svc1, clusterResources.nodes, true)
	lbStatus, _ := az.getServiceLoadBalancerStatus(&svc1, lb)

	sg, err := az.reconcileSecurityGroup(testClusterName, &svc1, &lbStatus.Ingress[0].IP, true /* wantLb */)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}

	validateSecurityGroup(t, sg, svc1)
}

func TestReconcileSecurityGroupNewInternalServiceAddsPort(t *testing.T) {
	az := getTestCloud()
	getTestSecurityGroup(az)
	svc1 := getInternalTestService("serviceea", 80)
	addTestSubnet(t, az, &svc1)
	clusterResources := getClusterResources(az, 1, 1)

	lb, _ := az.reconcileLoadBalancer(testClusterName, &svc1, clusterResources.nodes, true)
	lbStatus, _ := az.getServiceLoadBalancerStatus(&svc1, lb)
	sg, err := az.reconcileSecurityGroup(testClusterName, &svc1, &lbStatus.Ingress[0].IP, true /* wantLb */)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}

	validateSecurityGroup(t, sg, svc1)
}

func TestReconcileSecurityGroupRemoveService(t *testing.T) {
	az := getTestCloud()
	service1 := getTestService("servicea", v1.ProtocolTCP, nil, 81)
	service2 := getTestService("serviceb", v1.ProtocolTCP, nil, 82)
	clusterResources := getClusterResources(az, 1, 1)

	lb, _ := az.reconcileLoadBalancer(testClusterName, &service1, clusterResources.nodes, true)
	az.reconcileLoadBalancer(testClusterName, &service2, clusterResources.nodes, true)

	lbStatus, _ := az.getServiceLoadBalancerStatus(&service1, lb)

	sg := getTestSecurityGroup(az, service1, service2)
	validateSecurityGroup(t, sg, service1, service2)

	sg, err := az.reconcileSecurityGroup(testClusterName, &service1, &lbStatus.Ingress[0].IP, false /* wantLb */)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}

	validateSecurityGroup(t, sg, service2)
}

func TestReconcileSecurityGroupRemoveServiceRemovesPort(t *testing.T) {
	az := getTestCloud()
	svc := getTestService("servicea", v1.ProtocolTCP, nil, 80, 443)
	clusterResources := getClusterResources(az, 1, 1)

	sg := getTestSecurityGroup(az, svc)
	svcUpdated := getTestService("servicea", v1.ProtocolTCP, nil, 80)
	lb, _ := az.reconcileLoadBalancer(testClusterName, &svc, clusterResources.nodes, true)
	lbStatus, _ := az.getServiceLoadBalancerStatus(&svc, lb)

	sg, err := az.reconcileSecurityGroup(testClusterName, &svcUpdated, &lbStatus.Ingress[0].IP, true /* wantLb */)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}

	validateSecurityGroup(t, sg, svcUpdated)
}

func TestReconcileSecurityWithSourceRanges(t *testing.T) {
	az := getTestCloud()
	svc := getTestService("servicea", v1.ProtocolTCP, nil, 80, 443)
	svc.Spec.LoadBalancerSourceRanges = []string{
		"192.168.0.0/24",
		"10.0.0.0/32",
	}
	clusterResources := getClusterResources(az, 1, 1)

	sg := getTestSecurityGroup(az, svc)
	lb, _ := az.reconcileLoadBalancer(testClusterName, &svc, clusterResources.nodes, true)
	lbStatus, _ := az.getServiceLoadBalancerStatus(&svc, lb)

	sg, err := az.reconcileSecurityGroup(testClusterName, &svc, &lbStatus.Ingress[0].IP, true /* wantLb */)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}

	validateSecurityGroup(t, sg, svc)
}

func TestReconcileSecurityGroupEtagMismatch(t *testing.T) {
	az := getTestCloud()

	sg := getTestSecurityGroup(az)
	cachedSG := *sg
	cachedSG.Etag = to.StringPtr("1111111-0000-0000-0000-000000000000")
	az.nsgCache.Set(to.String(sg.Name), &cachedSG)

	svc1 := getTestService("servicea", v1.ProtocolTCP, nil, 80)
	clusterResources := getClusterResources(az, 1, 1)
	lb, _ := az.reconcileLoadBalancer(testClusterName, &svc1, clusterResources.nodes, true)
	lbStatus, _ := az.getServiceLoadBalancerStatus(&svc1, lb)

	newSG, err := az.reconcileSecurityGroup(testClusterName, &svc1, &lbStatus.Ingress[0].IP, true /* wantLb */)
	assert.Nil(t, newSG)
	assert.NotNil(t, err)
	assert.Equal(t, err, errPreconditionFailedEtagMismatch)
}

func TestReconcilePublicIPWithNewService(t *testing.T) {
	az := getTestCloud()
	svc := getTestService("servicea", v1.ProtocolTCP, nil, 80, 443)

	pip, err := az.reconcilePublicIP(testClusterName, &svc, "", true /* wantLb*/)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}
	validatePublicIP(t, pip, &svc, true)

	pip2, err := az.reconcilePublicIP(testClusterName, &svc, "", true /* wantLb */)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}
	validatePublicIP(t, pip2, &svc, true)
	if pip.Name != pip2.Name ||
		pip.PublicIPAddressPropertiesFormat.IPAddress != pip2.PublicIPAddressPropertiesFormat.IPAddress {
		t.Errorf("We should get the exact same public ip resource after a second reconcile")
	}
}

func TestReconcilePublicIPRemoveService(t *testing.T) {
	az := getTestCloud()
	svc := getTestService("servicea", v1.ProtocolTCP, nil, 80, 443)

	pip, err := az.reconcilePublicIP(testClusterName, &svc, "", true /* wantLb*/)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}

	validatePublicIP(t, pip, &svc, true)

	// Remove the service
	pip, err = az.reconcilePublicIP(testClusterName, &svc, "", false /* wantLb */)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}
	validatePublicIP(t, pip, &svc, false)

}

func TestReconcilePublicIPWithInternalService(t *testing.T) {
	az := getTestCloud()
	svc := getInternalTestService("servicea", 80, 443)

	pip, err := az.reconcilePublicIP(testClusterName, &svc, "", true /* wantLb*/)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}

	validatePublicIP(t, pip, &svc, true)
}

func TestReconcilePublicIPWithExternalAndInternalSwitch(t *testing.T) {
	az := getTestCloud()
	svc := getInternalTestService("servicea", 80, 443)

	pip, err := az.reconcilePublicIP(testClusterName, &svc, "", true /* wantLb*/)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}
	validatePublicIP(t, pip, &svc, true)

	// Update to external service
	svcUpdated := getTestService("servicea", v1.ProtocolTCP, nil, 80)
	pip, err = az.reconcilePublicIP(testClusterName, &svcUpdated, "", true /* wantLb*/)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}
	validatePublicIP(t, pip, &svcUpdated, true)

	// Update to internal service again
	pip, err = az.reconcilePublicIP(testClusterName, &svc, "", true /* wantLb*/)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}
	validatePublicIP(t, pip, &svc, true)
}

func getTestCloud() (az *Cloud) {
	az = &Cloud{
		Config: Config{
			AzureAuthConfig: auth.AzureAuthConfig{
				TenantID:       "tenant",
				SubscriptionID: "subscription",
			},
			ResourceGroup:                "rg",
			VnetResourceGroup:            "rg",
			RouteTableResourceGroup:      "rg",
			Location:                     "westus",
			VnetName:                     "vnet",
			SubnetName:                   "subnet",
			SecurityGroupName:            "nsg",
			RouteTableName:               "rt",
			PrimaryAvailabilitySetName:   "as",
			MaximumLoadBalancerRuleCount: 250,
			VMType:                       vmTypeStandard,
		},
		nodeZones:          map[string]sets.String{},
		nodeInformerSynced: func() bool { return true },
		nodeResourceGroups: map[string]string{},
		unmanagedNodes:     sets.NewString(),
		routeCIDRs:         map[string]string{},
		eventRecorder:      &record.FakeRecorder{},
	}
	az.DisksClient = newFakeDisksClient()
	az.InterfacesClient = newFakeAzureInterfacesClient()
	az.LoadBalancerClient = newFakeAzureLBClient()
	az.PublicIPAddressesClient = newFakeAzurePIPClient(az.Config.SubscriptionID)
	az.RoutesClient = newFakeRoutesClient()
	az.RouteTablesClient = newFakeRouteTablesClient()
	az.SecurityGroupsClient = newFakeAzureNSGClient()
	az.SubnetsClient = newFakeAzureSubnetsClient()
	az.VirtualMachineScaleSetsClient = newFakeVirtualMachineScaleSetsClient()
	az.VirtualMachineScaleSetVMsClient = newFakeVirtualMachineScaleSetVMsClient()
	az.VirtualMachinesClient = newFakeAzureVirtualMachinesClient()
	az.vmSet = newAvailabilitySet(az)
	az.vmCache, _ = az.newVMCache()
	az.lbCache, _ = az.newLBCache()
	az.nsgCache, _ = az.newNSGCache()
	az.rtCache, _ = az.newRouteTableCache()
	az.controllerCommon = &controllerCommon{cloud: az}

	return az
}

const networkInterfacesIDTemplate = "/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Network/networkInterfaces/%s"
const primaryIPConfigIDTemplate = "%s/ipConfigurations/ipconfig"

// returns the full identifier of Network Interface.
func getNetworkInterfaceID(subscriptionID string, resourceGroupName, nicName string) string {
	return fmt.Sprintf(
		networkInterfacesIDTemplate,
		subscriptionID,
		resourceGroupName,
		nicName)
}

// returns the full identifier of a private ipconfig of the nic
func getPrimaryIPConfigID(nicID string) string {
	return fmt.Sprintf(
		primaryIPConfigIDTemplate,
		nicID)
}

const TestResourceNameFormat = "%s-%d"
const TestVMResourceBaseName = "vm"
const TestASResourceBaseName = "as"

func getTestResourceName(resourceBaseName string, index int) string {
	return fmt.Sprintf(TestResourceNameFormat, resourceBaseName, index)
}

func getVMName(vmIndex int) string {
	return getTestResourceName(TestVMResourceBaseName, vmIndex)
}

func getAvailabilitySetName(az *Cloud, vmIndex int, numAS int) string {
	asIndex := vmIndex % numAS
	if asIndex == 0 {
		return az.Config.PrimaryAvailabilitySetName
	}

	return getTestResourceName(TestASResourceBaseName, asIndex)
}

// test supporting on 1 nic per vm
// we really dont care about the name of the nic
// just using the vm name for testing purposes
func getNICName(vmIndex int) string {
	return getVMName(vmIndex)
}

type ClusterResources struct {
	nodes                []*v1.Node
	availabilitySetNames []string
}

func getClusterResources(az *Cloud, vmCount int, availabilitySetCount int) (clusterResources *ClusterResources) {
	if vmCount < availabilitySetCount {
		return nil
	}
	clusterResources = &ClusterResources{}
	clusterResources.nodes = []*v1.Node{}
	clusterResources.availabilitySetNames = []string{}
	for vmIndex := 0; vmIndex < vmCount; vmIndex++ {
		vmName := getVMName(vmIndex)
		asName := getAvailabilitySetName(az, vmIndex, availabilitySetCount)
		clusterResources.availabilitySetNames = append(clusterResources.availabilitySetNames, asName)

		nicName := getNICName(vmIndex)
		nicID := getNetworkInterfaceID(az.Config.SubscriptionID, az.Config.ResourceGroup, nicName)
		primaryIPConfigID := getPrimaryIPConfigID(nicID)
		isPrimary := true
		newNIC := network.Interface{
			ID:   &nicID,
			Name: &nicName,
			InterfacePropertiesFormat: &network.InterfacePropertiesFormat{
				IPConfigurations: &[]network.InterfaceIPConfiguration{
					{
						ID: &primaryIPConfigID,
						InterfaceIPConfigurationPropertiesFormat: &network.InterfaceIPConfigurationPropertiesFormat{
							PrivateIPAddress: &nicName,
							Primary:          &isPrimary,
						},
					},
				},
			},
		}
		ctx, cancel := getContextWithCancel()
		defer cancel()
		az.InterfacesClient.CreateOrUpdate(ctx, az.Config.ResourceGroup, nicName, newNIC)

		// create vm
		asID := az.getAvailabilitySetID(az.Config.ResourceGroup, asName)
		newVM := compute.VirtualMachine{
			Name:     &vmName,
			Location: &az.Config.Location,
			VirtualMachineProperties: &compute.VirtualMachineProperties{
				AvailabilitySet: &compute.SubResource{
					ID: &asID,
				},
				NetworkProfile: &compute.NetworkProfile{
					NetworkInterfaces: &[]compute.NetworkInterfaceReference{
						{
							ID: &nicID,
						},
					},
				},
			},
		}

		vmCtx, vmCancel := getContextWithCancel()
		defer vmCancel()
		_, err := az.VirtualMachinesClient.CreateOrUpdate(vmCtx, az.Config.ResourceGroup, vmName, newVM, "")
		if err != nil {
		}
		// add to kubernetes
		newNode := &v1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: vmName,
				Labels: map[string]string{
					v1.LabelHostname: vmName,
				},
			},
		}
		clusterResources.nodes = append(clusterResources.nodes, newNode)
	}

	return clusterResources
}

func getBackendPort(port int32) int32 {
	return port + 10000
}

func getTestService(identifier string, proto v1.Protocol, annotations map[string]string, requestedPorts ...int32) v1.Service {
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
	if annotations == nil {
		svc.Annotations = make(map[string]string)
	} else {
		svc.Annotations = annotations
	}

	return svc
}

func getInternalTestService(identifier string, requestedPorts ...int32) v1.Service {
	svc := getTestService(identifier, v1.ProtocolTCP, nil, requestedPorts...)
	svc.Annotations[ServiceAnnotationLoadBalancerInternal] = "true"
	return svc
}

func getResourceGroupTestService(identifier, resourceGroup, loadBalancerIP string, requestedPorts ...int32) v1.Service {
	svc := getTestService(identifier, v1.ProtocolTCP, nil, requestedPorts...)
	svc.Spec.LoadBalancerIP = loadBalancerIP
	svc.Annotations[ServiceAnnotationLoadBalancerResourceGroup] = resourceGroup
	return svc
}

func setLoadBalancerModeAnnotation(service *v1.Service, lbMode string) {
	service.Annotations[ServiceAnnotationLoadBalancerMode] = lbMode
}

func setLoadBalancerAutoModeAnnotation(service *v1.Service) {
	setLoadBalancerModeAnnotation(service, ServiceAnnotationLoadBalancerAutoModeValue)
}

func getServiceSourceRanges(service *v1.Service) []string {
	if len(service.Spec.LoadBalancerSourceRanges) == 0 {
		if !requiresInternalLoadBalancer(service) {
			return []string{"Internet"}
		}
	}

	return service.Spec.LoadBalancerSourceRanges
}

func getTestSecurityGroup(az *Cloud, services ...v1.Service) *network.SecurityGroup {
	rules := []network.SecurityRule{}

	for _, service := range services {
		for _, port := range service.Spec.Ports {
			sources := getServiceSourceRanges(&service)
			for _, src := range sources {
				ruleName := az.getSecurityRuleName(&service, port, src)
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
		Name: &az.SecurityGroupName,
		Etag: to.StringPtr("0000000-0000-0000-0000-000000000000"),
		SecurityGroupPropertiesFormat: &network.SecurityGroupPropertiesFormat{
			SecurityRules: &rules,
		},
	}

	ctx, cancel := getContextWithCancel()
	defer cancel()
	az.SecurityGroupsClient.CreateOrUpdate(
		ctx,
		az.ResourceGroup,
		az.SecurityGroupName,
		sg,
		"")

	return &sg
}

func validateLoadBalancer(t *testing.T, loadBalancer *network.LoadBalancer, services ...v1.Service) {
	az := getTestCloud()
	expectedRuleCount := 0
	expectedFrontendIPCount := 0
	expectedProbeCount := 0
	expectedFrontendIPs := []ExpectedFrontendIPInfo{}
	for _, svc := range services {
		if len(svc.Spec.Ports) > 0 {
			expectedFrontendIPCount++
			expectedFrontendIP := ExpectedFrontendIPInfo{
				Name:   az.getFrontendIPConfigName(&svc, subnet(&svc)),
				Subnet: subnet(&svc),
			}
			expectedFrontendIPs = append(expectedFrontendIPs, expectedFrontendIP)
		}
		for _, wantedRule := range svc.Spec.Ports {
			expectedRuleCount++
			wantedRuleName := az.getLoadBalancerRuleName(&svc, wantedRule.Protocol, wantedRule.Port, subnet(&svc))
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
			if servicehelpers.NeedsHealthCheck(&svc) {
				path, port := servicehelpers.GetServiceHealthCheckPathPort(&svc)
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

	frontendIPs := *loadBalancer.FrontendIPConfigurations
	for _, expectedFrontendIP := range expectedFrontendIPs {
		if !expectedFrontendIP.existsIn(frontendIPs) {
			t.Errorf("Expected the loadbalancer to have frontend IP %s/%s. Found %s", expectedFrontendIP.Name, to.String(expectedFrontendIP.Subnet), describeFIPs(frontendIPs))
		}
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

type ExpectedFrontendIPInfo struct {
	Name   string
	Subnet *string
}

func (expected ExpectedFrontendIPInfo) matches(frontendIP network.FrontendIPConfiguration) bool {
	return strings.EqualFold(expected.Name, to.String(frontendIP.Name)) && strings.EqualFold(to.String(expected.Subnet), to.String(subnetName(frontendIP)))
}

func (expected ExpectedFrontendIPInfo) existsIn(frontendIPs []network.FrontendIPConfiguration) bool {
	for _, fip := range frontendIPs {
		if expected.matches(fip) {
			return true
		}
	}
	return false
}

func subnetName(frontendIP network.FrontendIPConfiguration) *string {
	if frontendIP.Subnet != nil {
		return frontendIP.Subnet.Name
	}
	return nil
}

func describeFIPs(frontendIPs []network.FrontendIPConfiguration) string {
	description := ""
	for _, actualFIP := range frontendIPs {
		actualSubnetName := ""
		if actualFIP.Subnet != nil {
			actualSubnetName = to.String(actualFIP.Subnet.Name)
		}
		actualFIPText := fmt.Sprintf("%s/%s ", to.String(actualFIP.Name), actualSubnetName)
		description = description + actualFIPText
	}
	return description
}

func validatePublicIP(t *testing.T, publicIP *network.PublicIPAddress, service *v1.Service, wantLb bool) {
	isInternal := requiresInternalLoadBalancer(service)
	if isInternal || !wantLb {
		if publicIP != nil {
			t.Errorf("Expected publicIP resource to be nil, when it is an internal service or doesn't want LB")
		}
		return
	}

	// For external service
	if publicIP == nil {
		t.Errorf("Expected publicIP resource exists, when it is not an internal service")
	}

	if publicIP.Tags == nil || publicIP.Tags[serviceTagKey] == nil {
		t.Errorf("Expected publicIP resource has tags[%s]", serviceTagKey)
	}

	serviceName := getServiceName(service)
	if serviceName != *(publicIP.Tags[serviceTagKey]) {
		t.Errorf("Expected publicIP resource has matching tags[%s]", serviceTagKey)
	}

	if publicIP.Tags[clusterNameKey] == nil {
		t.Errorf("Expected publicIP resource has tags[%s]", clusterNameKey)
	}

	if *(publicIP.Tags[clusterNameKey]) != testClusterName {
		t.Errorf("Expected publicIP resource has matching tags[%s]", clusterNameKey)
	}

	// We cannot use service.Spec.LoadBalancerIP to compare with
	// Public IP's IPAddress
	// Because service properties are updated outside of cloudprovider code
}

func contains(ruleValues []string, targetValue string) bool {
	for _, ruleValue := range ruleValues {
		if strings.EqualFold(ruleValue, targetValue) {
			return true
		}
	}
	return false
}

func securityRuleMatches(serviceSourceRange string, servicePort v1.ServicePort, serviceIP string, securityRule network.SecurityRule) error {
	ruleSource := securityRule.SourceAddressPrefixes
	if ruleSource == nil || len(*ruleSource) == 0 {
		if securityRule.SourceAddressPrefix == nil {
			ruleSource = &[]string{}
		} else {
			ruleSource = &[]string{*securityRule.SourceAddressPrefix}
		}
	}

	rulePorts := securityRule.DestinationPortRanges
	if rulePorts == nil || len(*rulePorts) == 0 {
		if securityRule.DestinationPortRange == nil {
			rulePorts = &[]string{}
		} else {
			rulePorts = &[]string{*securityRule.DestinationPortRange}
		}
	}

	ruleDestination := securityRule.DestinationAddressPrefixes
	if ruleDestination == nil || len(*ruleDestination) == 0 {
		if securityRule.DestinationAddressPrefix == nil {
			ruleDestination = &[]string{}
		} else {
			ruleDestination = &[]string{*securityRule.DestinationAddressPrefix}
		}
	}

	if !contains(*ruleSource, serviceSourceRange) {
		return fmt.Errorf("Rule does not contain source %s", serviceSourceRange)
	}

	if !contains(*rulePorts, fmt.Sprintf("%d", servicePort.Port)) {
		return fmt.Errorf("Rule does not contain port %d", servicePort.Port)
	}

	if serviceIP != "" && !contains(*ruleDestination, serviceIP) {
		return fmt.Errorf("Rule does not contain destination %s", serviceIP)
	}

	return nil
}

func validateSecurityGroup(t *testing.T, securityGroup *network.SecurityGroup, services ...v1.Service) {
	az := getTestCloud()
	seenRules := make(map[string]string)
	for _, svc := range services {
		for _, wantedRule := range svc.Spec.Ports {
			sources := getServiceSourceRanges(&svc)
			for _, source := range sources {
				wantedRuleName := az.getSecurityRuleName(&svc, wantedRule, source)
				seenRules[wantedRuleName] = wantedRuleName
				foundRule := false
				for _, actualRule := range *securityGroup.SecurityRules {
					if strings.EqualFold(*actualRule.Name, wantedRuleName) {
						err := securityRuleMatches(source, wantedRule, svc.Spec.LoadBalancerIP, actualRule)
						if err != nil {
							t.Errorf("Found matching security rule %q but properties were incorrect: %v", wantedRuleName, err)
						}
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
	expectedRuleCount := len(seenRules)
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
	if *securityGroupProto != network.SecurityRuleProtocolTCP {
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
	if *securityGroupProto != network.SecurityRuleProtocolUDP {
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
		"aadClientCertPath": "--aad-client-cert-path--",
		"aadClientCertPassword": "--aad-client-cert-password--",
		"resourceGroup": "--resource-group--",
		"routeTableResourceGroup": "--route-table-resource-group--",
		"location": "--location--",
		"subnetName": "--subnet-name--",
		"securityGroupName": "--security-group-name--",
		"vnetName": "--vnet-name--",
		"routeTableName": "--route-table-name--",
		"primaryAvailabilitySetName": "--primary-availability-set-name--",
		"cloudProviderBackoff": true,
		"cloudProviderRatelimit": true,
		"cloudProviderRateLimitQPS": 0.5,
		"cloudProviderRateLimitBucket": 5
	}`
	validateConfig(t, config)
}

// Test Backoff and Rate Limit defaults (json)
func TestCloudDefaultConfigFromJSON(t *testing.T) {
	config := `{
                "aadClientId": "--aad-client-id--",
                "aadClientSecret": "--aad-client-secret--"
        }`

	validateEmptyConfig(t, config)
}

// Test Backoff and Rate Limit defaults (yaml)
func TestCloudDefaultConfigFromYAML(t *testing.T) {
	config := `
aadClientId: --aad-client-id--
aadClientSecret: --aad-client-secret--
`
	validateEmptyConfig(t, config)
}

// Test Configuration deserialization (yaml) without
// specific resource group for the route table
func TestNewCloudFromYAML(t *testing.T) {
	config := `
tenantId: --tenant-id--
subscriptionId: --subscription-id--
aadClientId: --aad-client-id--
aadClientSecret: --aad-client-secret--
aadClientCertPath: --aad-client-cert-path--
aadClientCertPassword: --aad-client-cert-password--
resourceGroup: --resource-group--
routeTableResourceGroup: --route-table-resource-group--
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
	if azureCloud.AADClientCertPath != "--aad-client-cert-path--" {
		t.Errorf("got incorrect value for AADClientCertPath")
	}
	if azureCloud.AADClientCertPassword != "--aad-client-cert-password--" {
		t.Errorf("got incorrect value for AADClientCertPassword")
	}
	if azureCloud.ResourceGroup != "--resource-group--" {
		t.Errorf("got incorrect value for ResourceGroup")
	}
	if azureCloud.RouteTableResourceGroup != "--route-table-resource-group--" {
		t.Errorf("got incorrect value for RouteTableResourceGroup")
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
	azureCloud, err := NewCloudWithoutFeatureGates(configReader)
	if err != nil {
		t.Error(err)
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

func TestGetNodeNameByProviderID(t *testing.T) {
	az := getTestCloud()
	providers := []struct {
		providerID string
		name       types.NodeName

		fail bool
	}{
		{
			providerID: CloudProviderName + ":///subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/myResourceGroupName/providers/Microsoft.Compute/virtualMachines/k8s-agent-AAAAAAAA-0",
			name:       "k8s-agent-AAAAAAAA-0",
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
		name, err := az.vmSet.GetNodeNameByProviderID(test.providerID)
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

func addTestSubnet(t *testing.T, az *Cloud, svc *v1.Service) {
	if svc.Annotations[ServiceAnnotationLoadBalancerInternal] != "true" {
		t.Error("Subnet added to non-internal service")
	}
	subName := svc.Annotations[ServiceAnnotationLoadBalancerInternalSubnet]
	if subName == "" {
		subName = az.SubnetName
	}

	subnetID := fmt.Sprintf("/subscriptions/%s/resourceGroups/%s/providers/Microsoft.Network/virtualNetworks/%s/subnets/%s",
		az.SubscriptionID,
		az.VnetResourceGroup,
		az.VnetName,
		subName)

	ctx, cancel := getContextWithCancel()
	defer cancel()
	_, err := az.SubnetsClient.CreateOrUpdate(ctx, az.VnetResourceGroup, az.VnetName, subName,
		network.Subnet{
			ID:   &subnetID,
			Name: &subName,
		})

	if err != nil {
		t.Errorf("Subnet cannot be created or update, %v", err)
	}
	svc.Annotations[ServiceAnnotationLoadBalancerInternalSubnet] = subName
}

func TestIfServiceSpecifiesSharedRuleAndRuleDoesNotExistItIsCreated(t *testing.T) {
	az := getTestCloud()
	svc := getTestService("servicesr", v1.ProtocolTCP, nil, 80)
	svc.Spec.LoadBalancerIP = "192.168.77.88"
	svc.Annotations[ServiceAnnotationSharedSecurityRule] = "true"

	sg := getTestSecurityGroup(az)

	sg, err := az.reconcileSecurityGroup(testClusterName, &svc, to.StringPtr(svc.Spec.LoadBalancerIP), true)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}

	validateSecurityGroup(t, sg, svc)

	expectedRuleName := "shared-TCP-80-Internet"
	_, securityRule, ruleFound := findSecurityRuleByName(*sg.SecurityRules, expectedRuleName)
	if !ruleFound {
		t.Fatalf("Expected security rule %q but it was not present", expectedRuleName)
	}

	err = securityRuleMatches("Internet", v1.ServicePort{Port: 80}, "192.168.77.88", securityRule)
	if err != nil {
		t.Errorf("Shared rule was not updated with new service IP: %v", err)
	}

	if securityRule.Priority == nil {
		t.Errorf("Shared rule %s had no priority", expectedRuleName)
	}

	if securityRule.Access != network.SecurityRuleAccessAllow {
		t.Errorf("Shared rule %s did not have Allow access", expectedRuleName)
	}

	if securityRule.Direction != network.SecurityRuleDirectionInbound {
		t.Errorf("Shared rule %s did not have Inbound direction", expectedRuleName)
	}
}

func TestIfServiceSpecifiesSharedRuleAndRuleExistsThenTheServicesPortAndAddressAreAdded(t *testing.T) {
	az := getTestCloud()
	svc := getTestService("servicesr", v1.ProtocolTCP, nil, 80)
	svc.Spec.LoadBalancerIP = "192.168.77.88"
	svc.Annotations[ServiceAnnotationSharedSecurityRule] = "true"

	expectedRuleName := "shared-TCP-80-Internet"

	sg := getTestSecurityGroup(az)
	sg.SecurityRules = &[]network.SecurityRule{
		{
			Name: &expectedRuleName,
			SecurityRulePropertiesFormat: &network.SecurityRulePropertiesFormat{
				Protocol:                 network.SecurityRuleProtocolTCP,
				SourcePortRange:          to.StringPtr("*"),
				SourceAddressPrefix:      to.StringPtr("Internet"),
				DestinationPortRange:     to.StringPtr("80"),
				DestinationAddressPrefix: to.StringPtr("192.168.33.44"),
				Access:                   network.SecurityRuleAccessAllow,
				Direction:                network.SecurityRuleDirectionInbound,
			},
		},
	}

	sg, err := az.reconcileSecurityGroup(testClusterName, &svc, to.StringPtr(svc.Spec.LoadBalancerIP), true)
	if err != nil {
		t.Errorf("Unexpected error: %q", err)
	}

	validateSecurityGroup(t, sg, svc)

	_, securityRule, ruleFound := findSecurityRuleByName(*sg.SecurityRules, expectedRuleName)
	if !ruleFound {
		t.Fatalf("Expected security rule %q but it was not present", expectedRuleName)
	}

	expectedDestinationIPCount := 2
	if len(*securityRule.DestinationAddressPrefixes) != expectedDestinationIPCount {
		t.Errorf("Shared rule should have had %d destination IP addresses but had %d", expectedDestinationIPCount, len(*securityRule.DestinationAddressPrefixes))
	}

	err = securityRuleMatches("Internet", v1.ServicePort{Port: 80}, "192.168.33.44", securityRule)
	if err != nil {
		t.Errorf("Shared rule no longer matched other service IP: %v", err)
	}

	err = securityRuleMatches("Internet", v1.ServicePort{Port: 80}, "192.168.77.88", securityRule)
	if err != nil {
		t.Errorf("Shared rule was not updated with new service IP: %v", err)
	}
}

func TestIfServicesSpecifySharedRuleButDifferentPortsThenSeparateRulesAreCreated(t *testing.T) {
	az := getTestCloud()

	svc1 := getTestService("servicesr1", v1.ProtocolTCP, nil, 4444)
	svc1.Spec.LoadBalancerIP = "192.168.77.88"
	svc1.Annotations[ServiceAnnotationSharedSecurityRule] = "true"

	svc2 := getTestService("servicesr2", v1.ProtocolTCP, nil, 8888)
	svc2.Spec.LoadBalancerIP = "192.168.33.44"
	svc2.Annotations[ServiceAnnotationSharedSecurityRule] = "true"

	expectedRuleName1 := "shared-TCP-4444-Internet"
	expectedRuleName2 := "shared-TCP-8888-Internet"

	sg := getTestSecurityGroup(az)

	sg, err := az.reconcileSecurityGroup(testClusterName, &svc1, to.StringPtr(svc1.Spec.LoadBalancerIP), true)
	if err != nil {
		t.Errorf("Unexpected error adding svc1: %q", err)
	}

	sg, err = az.reconcileSecurityGroup(testClusterName, &svc2, to.StringPtr(svc2.Spec.LoadBalancerIP), true)
	if err != nil {
		t.Errorf("Unexpected error adding svc2: %q", err)
	}

	validateSecurityGroup(t, sg, svc1, svc2)

	_, securityRule1, rule1Found := findSecurityRuleByName(*sg.SecurityRules, expectedRuleName1)
	if !rule1Found {
		t.Fatalf("Expected security rule %q but it was not present", expectedRuleName1)
	}

	_, securityRule2, rule2Found := findSecurityRuleByName(*sg.SecurityRules, expectedRuleName2)
	if !rule2Found {
		t.Fatalf("Expected security rule %q but it was not present", expectedRuleName2)
	}

	expectedDestinationIPCount1 := 1
	if len(*securityRule1.DestinationAddressPrefixes) != expectedDestinationIPCount1 {
		t.Errorf("Shared rule %s should have had %d destination IP addresses but had %d", expectedRuleName1, expectedDestinationIPCount1, len(*securityRule1.DestinationAddressPrefixes))
	}

	err = securityRuleMatches("Internet", v1.ServicePort{Port: 4444}, "192.168.77.88", securityRule1)
	if err != nil {
		t.Errorf("Shared rule %s did not match service IP: %v", expectedRuleName1, err)
	}

	err = securityRuleMatches("Internet", v1.ServicePort{Port: 8888}, "192.168.33.44", securityRule1)
	if err == nil {
		t.Errorf("Shared rule %s matched wrong service's port and IP", expectedRuleName1)
	}

	expectedDestinationIPCount2 := 1
	if len(*securityRule2.DestinationAddressPrefixes) != expectedDestinationIPCount2 {
		t.Errorf("Shared rule %s should have had %d destination IP addresses but had %d", expectedRuleName2, expectedDestinationIPCount2, len(*securityRule2.DestinationAddressPrefixes))
	}

	err = securityRuleMatches("Internet", v1.ServicePort{Port: 8888}, "192.168.33.44", securityRule2)
	if err != nil {
		t.Errorf("Shared rule %s did not match service IP: %v", expectedRuleName2, err)
	}

	err = securityRuleMatches("Internet", v1.ServicePort{Port: 4444}, "192.168.77.88", securityRule2)
	if err == nil {
		t.Errorf("Shared rule %s matched wrong service's port and IP", expectedRuleName2)
	}
}

func TestIfServicesSpecifySharedRuleButDifferentProtocolsThenSeparateRulesAreCreated(t *testing.T) {
	az := getTestCloud()

	svc1 := getTestService("servicesr1", v1.ProtocolTCP, nil, 4444)
	svc1.Spec.LoadBalancerIP = "192.168.77.88"
	svc1.Annotations[ServiceAnnotationSharedSecurityRule] = "true"

	svc2 := getTestService("servicesr2", v1.ProtocolUDP, nil, 4444)
	svc2.Spec.LoadBalancerIP = "192.168.77.88"
	svc2.Annotations[ServiceAnnotationSharedSecurityRule] = "true"

	expectedRuleName1 := "shared-TCP-4444-Internet"
	expectedRuleName2 := "shared-UDP-4444-Internet"

	sg := getTestSecurityGroup(az)

	sg, err := az.reconcileSecurityGroup(testClusterName, &svc1, to.StringPtr(svc1.Spec.LoadBalancerIP), true)
	if err != nil {
		t.Errorf("Unexpected error adding svc1: %q", err)
	}

	sg, err = az.reconcileSecurityGroup(testClusterName, &svc2, to.StringPtr(svc2.Spec.LoadBalancerIP), true)
	if err != nil {
		t.Errorf("Unexpected error adding svc2: %q", err)
	}

	validateSecurityGroup(t, sg, svc1, svc2)

	_, securityRule1, rule1Found := findSecurityRuleByName(*sg.SecurityRules, expectedRuleName1)
	if !rule1Found {
		t.Fatalf("Expected security rule %q but it was not present", expectedRuleName1)
	}

	_, securityRule2, rule2Found := findSecurityRuleByName(*sg.SecurityRules, expectedRuleName2)
	if !rule2Found {
		t.Fatalf("Expected security rule %q but it was not present", expectedRuleName2)
	}

	expectedDestinationIPCount1 := 1
	if len(*securityRule1.DestinationAddressPrefixes) != expectedDestinationIPCount1 {
		t.Errorf("Shared rule %s should have had %d destination IP addresses but had %d", expectedRuleName1, expectedDestinationIPCount1, len(*securityRule1.DestinationAddressPrefixes))
	}

	err = securityRuleMatches("Internet", v1.ServicePort{Port: 4444}, "192.168.77.88", securityRule1)
	if err != nil {
		t.Errorf("Shared rule %s did not match service IP: %v", expectedRuleName1, err)
	}

	if securityRule1.Protocol != network.SecurityRuleProtocolTCP {
		t.Errorf("Shared rule %s should have been %s but was %s", expectedRuleName1, network.SecurityRuleProtocolTCP, securityRule1.Protocol)
	}

	expectedDestinationIPCount2 := 1
	if len(*securityRule2.DestinationAddressPrefixes) != expectedDestinationIPCount2 {
		t.Errorf("Shared rule %s should have had %d destination IP addresses but had %d", expectedRuleName2, expectedDestinationIPCount2, len(*securityRule2.DestinationAddressPrefixes))
	}

	err = securityRuleMatches("Internet", v1.ServicePort{Port: 4444}, "192.168.77.88", securityRule2)
	if err != nil {
		t.Errorf("Shared rule %s did not match service IP: %v", expectedRuleName2, err)
	}

	if securityRule2.Protocol != network.SecurityRuleProtocolUDP {
		t.Errorf("Shared rule %s should have been %s but was %s", expectedRuleName2, network.SecurityRuleProtocolUDP, securityRule2.Protocol)
	}
}

func TestIfServicesSpecifySharedRuleButDifferentSourceAddressesThenSeparateRulesAreCreated(t *testing.T) {
	az := getTestCloud()

	svc1 := getTestService("servicesr1", v1.ProtocolTCP, nil, 80)
	svc1.Spec.LoadBalancerIP = "192.168.77.88"
	svc1.Spec.LoadBalancerSourceRanges = []string{"192.168.12.0/24"}
	svc1.Annotations[ServiceAnnotationSharedSecurityRule] = "true"

	svc2 := getTestService("servicesr2", v1.ProtocolTCP, nil, 80)
	svc2.Spec.LoadBalancerIP = "192.168.33.44"
	svc2.Spec.LoadBalancerSourceRanges = []string{"192.168.34.0/24"}
	svc2.Annotations[ServiceAnnotationSharedSecurityRule] = "true"

	expectedRuleName1 := "shared-TCP-80-192.168.12.0_24"
	expectedRuleName2 := "shared-TCP-80-192.168.34.0_24"

	sg := getTestSecurityGroup(az)

	sg, err := az.reconcileSecurityGroup(testClusterName, &svc1, to.StringPtr(svc1.Spec.LoadBalancerIP), true)
	if err != nil {
		t.Errorf("Unexpected error adding svc1: %q", err)
	}

	sg, err = az.reconcileSecurityGroup(testClusterName, &svc2, to.StringPtr(svc2.Spec.LoadBalancerIP), true)
	if err != nil {
		t.Errorf("Unexpected error adding svc2: %q", err)
	}

	validateSecurityGroup(t, sg, svc1, svc2)

	_, securityRule1, rule1Found := findSecurityRuleByName(*sg.SecurityRules, expectedRuleName1)
	if !rule1Found {
		t.Fatalf("Expected security rule %q but it was not present", expectedRuleName1)
	}

	_, securityRule2, rule2Found := findSecurityRuleByName(*sg.SecurityRules, expectedRuleName2)
	if !rule2Found {
		t.Fatalf("Expected security rule %q but it was not present", expectedRuleName2)
	}

	expectedDestinationIPCount1 := 1
	if len(*securityRule1.DestinationAddressPrefixes) != expectedDestinationIPCount1 {
		t.Errorf("Shared rule %s should have had %d destination IP addresses but had %d", expectedRuleName1, expectedDestinationIPCount1, len(*securityRule1.DestinationAddressPrefixes))
	}

	err = securityRuleMatches(svc1.Spec.LoadBalancerSourceRanges[0], v1.ServicePort{Port: 80}, "192.168.77.88", securityRule1)
	if err != nil {
		t.Errorf("Shared rule %s did not match service IP: %v", expectedRuleName1, err)
	}

	err = securityRuleMatches(svc2.Spec.LoadBalancerSourceRanges[0], v1.ServicePort{Port: 80}, "192.168.33.44", securityRule1)
	if err == nil {
		t.Errorf("Shared rule %s matched wrong service's port and IP", expectedRuleName1)
	}

	expectedDestinationIPCount2 := 1
	if len(*securityRule2.DestinationAddressPrefixes) != expectedDestinationIPCount2 {
		t.Errorf("Shared rule %s should have had %d destination IP addresses but had %d", expectedRuleName2, expectedDestinationIPCount2, len(*securityRule2.DestinationAddressPrefixes))
	}

	err = securityRuleMatches(svc2.Spec.LoadBalancerSourceRanges[0], v1.ServicePort{Port: 80}, "192.168.33.44", securityRule2)
	if err != nil {
		t.Errorf("Shared rule %s did not match service IP: %v", expectedRuleName2, err)
	}

	err = securityRuleMatches(svc1.Spec.LoadBalancerSourceRanges[0], v1.ServicePort{Port: 80}, "192.168.77.88", securityRule2)
	if err == nil {
		t.Errorf("Shared rule %s matched wrong service's port and IP", expectedRuleName2)
	}
}

func TestIfServicesSpecifySharedRuleButSomeAreOnDifferentPortsThenRulesAreSeparatedOrConsoliatedByPort(t *testing.T) {
	az := getTestCloud()

	svc1 := getTestService("servicesr1", v1.ProtocolTCP, nil, 4444)
	svc1.Spec.LoadBalancerIP = "192.168.77.88"
	svc1.Annotations[ServiceAnnotationSharedSecurityRule] = "true"

	svc2 := getTestService("servicesr2", v1.ProtocolTCP, nil, 8888)
	svc2.Spec.LoadBalancerIP = "192.168.33.44"
	svc2.Annotations[ServiceAnnotationSharedSecurityRule] = "true"

	svc3 := getTestService("servicesr3", v1.ProtocolTCP, nil, 4444)
	svc3.Spec.LoadBalancerIP = "192.168.99.11"
	svc3.Annotations[ServiceAnnotationSharedSecurityRule] = "true"

	expectedRuleName13 := "shared-TCP-4444-Internet"
	expectedRuleName2 := "shared-TCP-8888-Internet"

	sg := getTestSecurityGroup(az)

	sg, err := az.reconcileSecurityGroup(testClusterName, &svc1, to.StringPtr(svc1.Spec.LoadBalancerIP), true)
	if err != nil {
		t.Errorf("Unexpected error adding svc1: %q", err)
	}

	sg, err = az.reconcileSecurityGroup(testClusterName, &svc2, to.StringPtr(svc2.Spec.LoadBalancerIP), true)
	if err != nil {
		t.Errorf("Unexpected error adding svc2: %q", err)
	}

	sg, err = az.reconcileSecurityGroup(testClusterName, &svc3, to.StringPtr(svc3.Spec.LoadBalancerIP), true)
	if err != nil {
		t.Errorf("Unexpected error adding svc3: %q", err)
	}

	validateSecurityGroup(t, sg, svc1, svc2, svc3)

	_, securityRule13, rule13Found := findSecurityRuleByName(*sg.SecurityRules, expectedRuleName13)
	if !rule13Found {
		t.Fatalf("Expected security rule %q but it was not present", expectedRuleName13)
	}

	_, securityRule2, rule2Found := findSecurityRuleByName(*sg.SecurityRules, expectedRuleName2)
	if !rule2Found {
		t.Fatalf("Expected security rule %q but it was not present", expectedRuleName2)
	}

	expectedDestinationIPCount13 := 2
	if len(*securityRule13.DestinationAddressPrefixes) != expectedDestinationIPCount13 {
		t.Errorf("Shared rule %s should have had %d destination IP addresses but had %d", expectedRuleName13, expectedDestinationIPCount13, len(*securityRule13.DestinationAddressPrefixes))
	}

	err = securityRuleMatches("Internet", v1.ServicePort{Port: 4444}, "192.168.77.88", securityRule13)
	if err != nil {
		t.Errorf("Shared rule %s did not match service IP: %v", expectedRuleName13, err)
	}

	err = securityRuleMatches("Internet", v1.ServicePort{Port: 4444}, "192.168.99.11", securityRule13)
	if err != nil {
		t.Errorf("Shared rule %s did not match service IP: %v", expectedRuleName13, err)
	}

	err = securityRuleMatches("Internet", v1.ServicePort{Port: 8888}, "192.168.33.44", securityRule13)
	if err == nil {
		t.Errorf("Shared rule %s matched wrong service's port and IP", expectedRuleName13)
	}

	if securityRule13.Priority == nil {
		t.Errorf("Shared rule %s had no priority", expectedRuleName13)
	}

	if securityRule13.Access != network.SecurityRuleAccessAllow {
		t.Errorf("Shared rule %s did not have Allow access", expectedRuleName13)
	}

	if securityRule13.Direction != network.SecurityRuleDirectionInbound {
		t.Errorf("Shared rule %s did not have Inbound direction", expectedRuleName13)
	}

	expectedDestinationIPCount2 := 1
	if len(*securityRule2.DestinationAddressPrefixes) != expectedDestinationIPCount2 {
		t.Errorf("Shared rule %s should have had %d destination IP addresses but had %d", expectedRuleName2, expectedDestinationIPCount2, len(*securityRule2.DestinationAddressPrefixes))
	}

	err = securityRuleMatches("Internet", v1.ServicePort{Port: 8888}, "192.168.33.44", securityRule2)
	if err != nil {
		t.Errorf("Shared rule %s did not match service IP: %v", expectedRuleName2, err)
	}

	err = securityRuleMatches("Internet", v1.ServicePort{Port: 4444}, "192.168.77.88", securityRule2)
	if err == nil {
		t.Errorf("Shared rule %s matched wrong service's port and IP", expectedRuleName2)
	}

	err = securityRuleMatches("Internet", v1.ServicePort{Port: 4444}, "192.168.99.11", securityRule2)
	if err == nil {
		t.Errorf("Shared rule %s matched wrong service's port and IP", expectedRuleName2)
	}
}

func TestIfServiceSpecifiesSharedRuleAndServiceIsDeletedThenTheServicesPortAndAddressAreRemoved(t *testing.T) {
	az := getTestCloud()

	svc1 := getTestService("servicesr1", v1.ProtocolTCP, nil, 80)
	svc1.Spec.LoadBalancerIP = "192.168.77.88"
	svc1.Annotations[ServiceAnnotationSharedSecurityRule] = "true"

	svc2 := getTestService("servicesr2", v1.ProtocolTCP, nil, 80)
	svc2.Spec.LoadBalancerIP = "192.168.33.44"
	svc2.Annotations[ServiceAnnotationSharedSecurityRule] = "true"

	expectedRuleName := "shared-TCP-80-Internet"

	sg := getTestSecurityGroup(az)

	sg, err := az.reconcileSecurityGroup(testClusterName, &svc1, to.StringPtr(svc1.Spec.LoadBalancerIP), true)
	if err != nil {
		t.Errorf("Unexpected error adding svc1: %q", err)
	}

	sg, err = az.reconcileSecurityGroup(testClusterName, &svc2, to.StringPtr(svc2.Spec.LoadBalancerIP), true)
	if err != nil {
		t.Errorf("Unexpected error adding svc2: %q", err)
	}

	validateSecurityGroup(t, sg, svc1, svc2)

	sg, err = az.reconcileSecurityGroup(testClusterName, &svc1, to.StringPtr(svc1.Spec.LoadBalancerIP), false)
	if err != nil {
		t.Errorf("Unexpected error removing svc1: %q", err)
	}

	validateSecurityGroup(t, sg, svc2)

	_, securityRule, ruleFound := findSecurityRuleByName(*sg.SecurityRules, expectedRuleName)
	if !ruleFound {
		t.Fatalf("Expected security rule %q but it was not present", expectedRuleName)
	}

	expectedDestinationIPCount := 1
	if len(*securityRule.DestinationAddressPrefixes) != expectedDestinationIPCount {
		t.Errorf("Shared rule should have had %d destination IP addresses but had %d", expectedDestinationIPCount, len(*securityRule.DestinationAddressPrefixes))
	}

	err = securityRuleMatches("Internet", v1.ServicePort{Port: 80}, "192.168.33.44", securityRule)
	if err != nil {
		t.Errorf("Shared rule no longer matched other service IP: %v", err)
	}

	err = securityRuleMatches("Internet", v1.ServicePort{Port: 80}, "192.168.77.88", securityRule)
	if err == nil {
		t.Error("Shared rule was not updated to remove deleted service IP")
	}
}

func TestIfSomeServicesShareARuleAndOneIsDeletedItIsRemovedFromTheRightRule(t *testing.T) {
	az := getTestCloud()

	svc1 := getTestService("servicesr1", v1.ProtocolTCP, nil, 4444)
	svc1.Spec.LoadBalancerIP = "192.168.77.88"
	svc1.Annotations[ServiceAnnotationSharedSecurityRule] = "true"

	svc2 := getTestService("servicesr2", v1.ProtocolTCP, nil, 8888)
	svc2.Spec.LoadBalancerIP = "192.168.33.44"
	svc2.Annotations[ServiceAnnotationSharedSecurityRule] = "true"

	svc3 := getTestService("servicesr3", v1.ProtocolTCP, nil, 4444)
	svc3.Spec.LoadBalancerIP = "192.168.99.11"
	svc3.Annotations[ServiceAnnotationSharedSecurityRule] = "true"

	expectedRuleName13 := "shared-TCP-4444-Internet"
	expectedRuleName2 := "shared-TCP-8888-Internet"

	sg := getTestSecurityGroup(az)

	sg, err := az.reconcileSecurityGroup(testClusterName, &svc1, to.StringPtr(svc1.Spec.LoadBalancerIP), true)
	if err != nil {
		t.Errorf("Unexpected error adding svc1: %q", err)
	}

	sg, err = az.reconcileSecurityGroup(testClusterName, &svc2, to.StringPtr(svc2.Spec.LoadBalancerIP), true)
	if err != nil {
		t.Errorf("Unexpected error adding svc2: %q", err)
	}

	sg, err = az.reconcileSecurityGroup(testClusterName, &svc3, to.StringPtr(svc3.Spec.LoadBalancerIP), true)
	if err != nil {
		t.Errorf("Unexpected error adding svc3: %q", err)
	}

	validateSecurityGroup(t, sg, svc1, svc2, svc3)

	sg, err = az.reconcileSecurityGroup(testClusterName, &svc1, to.StringPtr(svc1.Spec.LoadBalancerIP), false)
	if err != nil {
		t.Errorf("Unexpected error removing svc1: %q", err)
	}

	validateSecurityGroup(t, sg, svc2, svc3)

	_, securityRule13, rule13Found := findSecurityRuleByName(*sg.SecurityRules, expectedRuleName13)
	if !rule13Found {
		t.Fatalf("Expected security rule %q but it was not present", expectedRuleName13)
	}

	_, securityRule2, rule2Found := findSecurityRuleByName(*sg.SecurityRules, expectedRuleName2)
	if !rule2Found {
		t.Fatalf("Expected security rule %q but it was not present", expectedRuleName2)
	}

	expectedDestinationIPCount13 := 1
	if len(*securityRule13.DestinationAddressPrefixes) != expectedDestinationIPCount13 {
		t.Errorf("Shared rule %s should have had %d destination IP addresses but had %d", expectedRuleName13, expectedDestinationIPCount13, len(*securityRule13.DestinationAddressPrefixes))
	}

	err = securityRuleMatches("Internet", v1.ServicePort{Port: 4444}, "192.168.77.88", securityRule13)
	if err == nil {
		t.Errorf("Shared rule %s should have had svc1 removed but did not", expectedRuleName13)
	}

	err = securityRuleMatches("Internet", v1.ServicePort{Port: 4444}, "192.168.99.11", securityRule13)
	if err != nil {
		t.Errorf("Shared rule %s did not match service IP: %v", expectedRuleName13, err)
	}

	err = securityRuleMatches("Internet", v1.ServicePort{Port: 8888}, "192.168.33.44", securityRule13)
	if err == nil {
		t.Errorf("Shared rule %s matched wrong service's port and IP", expectedRuleName13)
	}

	if securityRule13.Priority == nil {
		t.Errorf("Shared rule %s had no priority", expectedRuleName13)
	}

	if securityRule13.Access != network.SecurityRuleAccessAllow {
		t.Errorf("Shared rule %s did not have Allow access", expectedRuleName13)
	}

	if securityRule13.Direction != network.SecurityRuleDirectionInbound {
		t.Errorf("Shared rule %s did not have Inbound direction", expectedRuleName13)
	}

	expectedDestinationIPCount2 := 1
	if len(*securityRule2.DestinationAddressPrefixes) != expectedDestinationIPCount2 {
		t.Errorf("Shared rule %s should have had %d destination IP addresses but had %d", expectedRuleName2, expectedDestinationIPCount2, len(*securityRule2.DestinationAddressPrefixes))
	}

	err = securityRuleMatches("Internet", v1.ServicePort{Port: 8888}, "192.168.33.44", securityRule2)
	if err != nil {
		t.Errorf("Shared rule %s did not match service IP: %v", expectedRuleName2, err)
	}

	err = securityRuleMatches("Internet", v1.ServicePort{Port: 4444}, "192.168.77.88", securityRule2)
	if err == nil {
		t.Errorf("Shared rule %s matched wrong service's port and IP", expectedRuleName2)
	}

	err = securityRuleMatches("Internet", v1.ServicePort{Port: 4444}, "192.168.99.11", securityRule2)
	if err == nil {
		t.Errorf("Shared rule %s matched wrong service's port and IP", expectedRuleName2)
	}
}

func TestIfServiceSpecifiesSharedRuleAndLastServiceIsDeletedThenRuleIsDeleted(t *testing.T) {
	az := getTestCloud()

	svc1 := getTestService("servicesr1", v1.ProtocolTCP, nil, 4444)
	svc1.Spec.LoadBalancerIP = "192.168.77.88"
	svc1.Annotations[ServiceAnnotationSharedSecurityRule] = "true"

	svc2 := getTestService("servicesr2", v1.ProtocolTCP, nil, 8888)
	svc2.Spec.LoadBalancerIP = "192.168.33.44"
	svc2.Annotations[ServiceAnnotationSharedSecurityRule] = "true"

	svc3 := getTestService("servicesr3", v1.ProtocolTCP, nil, 4444)
	svc3.Spec.LoadBalancerIP = "192.168.99.11"
	svc3.Annotations[ServiceAnnotationSharedSecurityRule] = "true"

	expectedRuleName13 := "shared-TCP-4444-Internet"
	expectedRuleName2 := "shared-TCP-8888-Internet"

	sg := getTestSecurityGroup(az)

	sg, err := az.reconcileSecurityGroup(testClusterName, &svc1, to.StringPtr(svc1.Spec.LoadBalancerIP), true)
	if err != nil {
		t.Errorf("Unexpected error adding svc1: %q", err)
	}

	sg, err = az.reconcileSecurityGroup(testClusterName, &svc2, to.StringPtr(svc2.Spec.LoadBalancerIP), true)
	if err != nil {
		t.Errorf("Unexpected error adding svc2: %q", err)
	}

	sg, err = az.reconcileSecurityGroup(testClusterName, &svc3, to.StringPtr(svc3.Spec.LoadBalancerIP), true)
	if err != nil {
		t.Errorf("Unexpected error adding svc3: %q", err)
	}

	validateSecurityGroup(t, sg, svc1, svc2, svc3)

	sg, err = az.reconcileSecurityGroup(testClusterName, &svc1, to.StringPtr(svc1.Spec.LoadBalancerIP), false)
	if err != nil {
		t.Errorf("Unexpected error removing svc1: %q", err)
	}

	sg, err = az.reconcileSecurityGroup(testClusterName, &svc3, to.StringPtr(svc3.Spec.LoadBalancerIP), false)
	if err != nil {
		t.Errorf("Unexpected error removing svc3: %q", err)
	}

	validateSecurityGroup(t, sg, svc2)

	_, _, rule13Found := findSecurityRuleByName(*sg.SecurityRules, expectedRuleName13)
	if rule13Found {
		t.Fatalf("Expected security rule %q to have been deleted but it was still present", expectedRuleName13)
	}

	_, securityRule2, rule2Found := findSecurityRuleByName(*sg.SecurityRules, expectedRuleName2)
	if !rule2Found {
		t.Fatalf("Expected security rule %q but it was not present", expectedRuleName2)
	}

	expectedDestinationIPCount2 := 1
	if len(*securityRule2.DestinationAddressPrefixes) != expectedDestinationIPCount2 {
		t.Errorf("Shared rule %s should have had %d destination IP addresses but had %d", expectedRuleName2, expectedDestinationIPCount2, len(*securityRule2.DestinationAddressPrefixes))
	}

	err = securityRuleMatches("Internet", v1.ServicePort{Port: 8888}, "192.168.33.44", securityRule2)
	if err != nil {
		t.Errorf("Shared rule %s did not match service IP: %v", expectedRuleName2, err)
	}

	err = securityRuleMatches("Internet", v1.ServicePort{Port: 4444}, "192.168.77.88", securityRule2)
	if err == nil {
		t.Errorf("Shared rule %s matched wrong service's port and IP", expectedRuleName2)
	}

	err = securityRuleMatches("Internet", v1.ServicePort{Port: 4444}, "192.168.99.11", securityRule2)
	if err == nil {
		t.Errorf("Shared rule %s matched wrong service's port and IP", expectedRuleName2)
	}
}

func TestCanCombineSharedAndPrivateRulesInSameGroup(t *testing.T) {
	az := getTestCloud()

	svc1 := getTestService("servicesr1", v1.ProtocolTCP, nil, 4444)
	svc1.Spec.LoadBalancerIP = "192.168.77.88"
	svc1.Annotations[ServiceAnnotationSharedSecurityRule] = "true"

	svc2 := getTestService("servicesr2", v1.ProtocolTCP, nil, 8888)
	svc2.Spec.LoadBalancerIP = "192.168.33.44"
	svc2.Annotations[ServiceAnnotationSharedSecurityRule] = "true"

	svc3 := getTestService("servicesr3", v1.ProtocolTCP, nil, 4444)
	svc3.Spec.LoadBalancerIP = "192.168.99.11"
	svc3.Annotations[ServiceAnnotationSharedSecurityRule] = "true"

	svc4 := getTestService("servicesr4", v1.ProtocolTCP, nil, 4444)
	svc4.Spec.LoadBalancerIP = "192.168.22.33"
	svc4.Annotations[ServiceAnnotationSharedSecurityRule] = "false"

	svc5 := getTestService("servicesr5", v1.ProtocolTCP, nil, 8888)
	svc5.Spec.LoadBalancerIP = "192.168.22.33"
	svc5.Annotations[ServiceAnnotationSharedSecurityRule] = "false"

	expectedRuleName13 := "shared-TCP-4444-Internet"
	expectedRuleName2 := "shared-TCP-8888-Internet"
	expectedRuleName4 := az.getSecurityRuleName(&svc4, v1.ServicePort{Port: 4444, Protocol: v1.ProtocolTCP}, "Internet")
	expectedRuleName5 := az.getSecurityRuleName(&svc5, v1.ServicePort{Port: 8888, Protocol: v1.ProtocolTCP}, "Internet")

	sg := getTestSecurityGroup(az)

	sg, err := az.reconcileSecurityGroup(testClusterName, &svc1, to.StringPtr(svc1.Spec.LoadBalancerIP), true)
	if err != nil {
		t.Errorf("Unexpected error adding svc1: %q", err)
	}

	sg, err = az.reconcileSecurityGroup(testClusterName, &svc2, to.StringPtr(svc2.Spec.LoadBalancerIP), true)
	if err != nil {
		t.Errorf("Unexpected error adding svc2: %q", err)
	}

	sg, err = az.reconcileSecurityGroup(testClusterName, &svc3, to.StringPtr(svc3.Spec.LoadBalancerIP), true)
	if err != nil {
		t.Errorf("Unexpected error adding svc3: %q", err)
	}

	sg, err = az.reconcileSecurityGroup(testClusterName, &svc4, to.StringPtr(svc4.Spec.LoadBalancerIP), true)
	if err != nil {
		t.Errorf("Unexpected error adding svc4: %q", err)
	}

	sg, err = az.reconcileSecurityGroup(testClusterName, &svc5, to.StringPtr(svc5.Spec.LoadBalancerIP), true)
	if err != nil {
		t.Errorf("Unexpected error adding svc4: %q", err)
	}

	validateSecurityGroup(t, sg, svc1, svc2, svc3, svc4, svc5)

	expectedRuleCount := 4
	if len(*sg.SecurityRules) != expectedRuleCount {
		t.Errorf("Expected security group to have %d rules but it had %d", expectedRuleCount, len(*sg.SecurityRules))
	}

	_, securityRule13, rule13Found := findSecurityRuleByName(*sg.SecurityRules, expectedRuleName13)
	if !rule13Found {
		t.Fatalf("Expected security rule %q but it was not present", expectedRuleName13)
	}

	_, securityRule2, rule2Found := findSecurityRuleByName(*sg.SecurityRules, expectedRuleName2)
	if !rule2Found {
		t.Fatalf("Expected security rule %q but it was not present", expectedRuleName2)
	}

	_, securityRule4, rule4Found := findSecurityRuleByName(*sg.SecurityRules, expectedRuleName4)
	if !rule4Found {
		t.Fatalf("Expected security rule %q but it was not present", expectedRuleName4)
	}

	_, securityRule5, rule5Found := findSecurityRuleByName(*sg.SecurityRules, expectedRuleName5)
	if !rule5Found {
		t.Fatalf("Expected security rule %q but it was not present", expectedRuleName5)
	}

	expectedDestinationIPCount13 := 2
	if len(*securityRule13.DestinationAddressPrefixes) != expectedDestinationIPCount13 {
		t.Errorf("Shared rule %s should have had %d destination IP addresses but had %d", expectedRuleName13, expectedDestinationIPCount13, len(*securityRule13.DestinationAddressPrefixes))
	}

	err = securityRuleMatches("Internet", v1.ServicePort{Port: 4444}, "192.168.77.88", securityRule13)
	if err != nil {
		t.Errorf("Shared rule %s did not match service IP: %v", expectedRuleName13, err)
	}

	err = securityRuleMatches("Internet", v1.ServicePort{Port: 4444}, "192.168.99.11", securityRule13)
	if err != nil {
		t.Errorf("Shared rule %s did not match service IP: %v", expectedRuleName13, err)
	}

	err = securityRuleMatches("Internet", v1.ServicePort{Port: 4444}, "192.168.22.33", securityRule13)
	if err == nil {
		t.Errorf("Shared rule %s matched wrong (unshared) service's port and IP", expectedRuleName13)
	}

	expectedDestinationIPCount2 := 1
	if len(*securityRule2.DestinationAddressPrefixes) != expectedDestinationIPCount2 {
		t.Errorf("Shared rule %s should have had %d destination IP addresses but had %d", expectedRuleName2, expectedDestinationIPCount2, len(*securityRule2.DestinationAddressPrefixes))
	}

	err = securityRuleMatches("Internet", v1.ServicePort{Port: 8888}, "192.168.33.44", securityRule2)
	if err != nil {
		t.Errorf("Shared rule %s did not match service IP: %v", expectedRuleName2, err)
	}

	err = securityRuleMatches("Internet", v1.ServicePort{Port: 8888}, "192.168.22.33", securityRule2)
	if err == nil {
		t.Errorf("Shared rule %s matched wrong (unshared) service's port and IP", expectedRuleName2)
	}

	if securityRule4.DestinationAddressPrefixes != nil {
		t.Errorf("Expected unshared rule %s to use single destination IP address but used collection", expectedRuleName4)
	}

	if securityRule4.DestinationAddressPrefix == nil {
		t.Errorf("Expected unshared rule %s to have a destination IP address", expectedRuleName4)
	} else {
		if !strings.EqualFold(*securityRule4.DestinationAddressPrefix, svc4.Spec.LoadBalancerIP) {
			t.Errorf("Expected unshared rule %s to have a destination %s but had %s", expectedRuleName4, svc4.Spec.LoadBalancerIP, *securityRule4.DestinationAddressPrefix)
		}
	}

	if securityRule5.DestinationAddressPrefixes != nil {
		t.Errorf("Expected unshared rule %s to use single destination IP address but used collection", expectedRuleName5)
	}

	if securityRule5.DestinationAddressPrefix == nil {
		t.Errorf("Expected unshared rule %s to have a destination IP address", expectedRuleName5)
	} else {
		if !strings.EqualFold(*securityRule5.DestinationAddressPrefix, svc5.Spec.LoadBalancerIP) {
			t.Errorf("Expected unshared rule %s to have a destination %s but had %s", expectedRuleName5, svc5.Spec.LoadBalancerIP, *securityRule5.DestinationAddressPrefix)
		}
	}

	sg, err = az.reconcileSecurityGroup(testClusterName, &svc1, to.StringPtr(svc1.Spec.LoadBalancerIP), false)
	if err != nil {
		t.Errorf("Unexpected error removing svc1: %q", err)
	}

	sg, err = az.reconcileSecurityGroup(testClusterName, &svc5, to.StringPtr(svc5.Spec.LoadBalancerIP), false)
	if err != nil {
		t.Errorf("Unexpected error removing svc5: %q", err)
	}

	_, securityRule13, rule13Found = findSecurityRuleByName(*sg.SecurityRules, expectedRuleName13)
	if !rule13Found {
		t.Fatalf("Expected security rule %q but it was not present", expectedRuleName13)
	}

	_, securityRule2, rule2Found = findSecurityRuleByName(*sg.SecurityRules, expectedRuleName2)
	if !rule2Found {
		t.Fatalf("Expected security rule %q but it was not present", expectedRuleName2)
	}

	_, securityRule4, rule4Found = findSecurityRuleByName(*sg.SecurityRules, expectedRuleName4)
	if !rule4Found {
		t.Fatalf("Expected security rule %q but it was not present", expectedRuleName4)
	}

	_, _, rule5Found = findSecurityRuleByName(*sg.SecurityRules, expectedRuleName5)
	if rule5Found {
		t.Fatalf("Expected security rule %q to have been removed but it was not present", expectedRuleName5)
	}

	expectedDestinationIPCount13 = 1
	if len(*securityRule13.DestinationAddressPrefixes) != expectedDestinationIPCount13 {
		t.Errorf("Shared rule %s should have had %d destination IP addresses but had %d", expectedRuleName13, expectedDestinationIPCount13, len(*securityRule13.DestinationAddressPrefixes))
	}
}

// TODO: sanity check if the same IP address incorrectly gets put in twice?
// (shouldn't happen but...)

// func TestIfServiceIsEditedFromOwnRuleToSharedRuleThenOwnRuleIsDeletedAndSharedRuleIsCreated(t *testing.T) {
// 	t.Error()
// }

// func TestIfServiceIsEditedFromSharedRuleToOwnRuleThenItIsRemovedFromSharedRuleAndOwnRuleIsCreated(t *testing.T) {
// 	t.Error()
// }

func TestGetResourceGroupFromDiskURI(t *testing.T) {
	tests := []struct {
		diskURL        string
		expectedResult string
		expectError    bool
	}{
		{
			diskURL:        "/subscriptions/4be8920b-2978-43d7-axyz-04d8549c1d05/resourceGroups/azure-k8s1102/providers/Microsoft.Compute/disks/andy-mghyb1102-dynamic-pvc-f7f014c9-49f4-11e8-ab5c-000d3af7b38e",
			expectedResult: "azure-k8s1102",
			expectError:    false,
		},
		{
			// case insentive check
			diskURL:        "/subscriptions/4be8920b-2978-43d7-axyz-04d8549c1d05/resourcegroups/azure-k8s1102/providers/Microsoft.Compute/disks/andy-mghyb1102-dynamic-pvc-f7f014c9-49f4-11e8-ab5c-000d3af7b38e",
			expectedResult: "azure-k8s1102",
			expectError:    false,
		},
		{
			diskURL:        "/4be8920b-2978-43d7-axyz-04d8549c1d05/resourceGroups/azure-k8s1102/providers/Microsoft.Compute/disks/andy-mghyb1102-dynamic-pvc-f7f014c9-49f4-11e8-ab5c-000d3af7b38e",
			expectedResult: "",
			expectError:    true,
		},
		{
			diskURL:        "",
			expectedResult: "",
			expectError:    true,
		},
	}

	for _, test := range tests {
		result, err := getResourceGroupFromDiskURI(test.diskURL)
		assert.Equal(t, result, test.expectedResult, "Expect result not equal with getResourceGroupFromDiskURI(%s) return: %q, expected: %q",
			test.diskURL, result, test.expectedResult)

		if test.expectError {
			assert.NotNil(t, err, "Expect error during getResourceGroupFromDiskURI(%s)", test.diskURL)
		} else {
			assert.Nil(t, err, "Expect error is nil during getResourceGroupFromDiskURI(%s)", test.diskURL)
		}
	}
}

func TestGetResourceGroups(t *testing.T) {
	tests := []struct {
		name               string
		nodeResourceGroups map[string]string
		expected           sets.String
		informerSynced     bool
		expectError        bool
	}{
		{
			name:               "cloud provider configured RG should be returned by default",
			nodeResourceGroups: map[string]string{},
			informerSynced:     true,
			expected:           sets.NewString("rg"),
		},
		{
			name:               "cloud provider configured RG and node RGs should be returned",
			nodeResourceGroups: map[string]string{"node1": "rg1", "node2": "rg2"},
			informerSynced:     true,
			expected:           sets.NewString("rg", "rg1", "rg2"),
		},
		{
			name:               "error should be returned if informer hasn't synced yet",
			nodeResourceGroups: map[string]string{"node1": "rg1", "node2": "rg2"},
			informerSynced:     false,
			expectError:        true,
		},
	}

	az := getTestCloud()
	for _, test := range tests {
		az.nodeResourceGroups = test.nodeResourceGroups
		if test.informerSynced {
			az.nodeInformerSynced = func() bool { return true }
		} else {
			az.nodeInformerSynced = func() bool { return false }
		}
		actual, err := az.GetResourceGroups()
		if test.expectError {
			assert.NotNil(t, err, test.name)
			continue
		}

		assert.Nil(t, err, test.name)
		assert.Equal(t, test.expected, actual, test.name)
	}
}

func TestGetNodeResourceGroup(t *testing.T) {
	tests := []struct {
		name               string
		nodeResourceGroups map[string]string
		node               string
		expected           string
		informerSynced     bool
		expectError        bool
	}{
		{
			name:               "cloud provider configured RG should be returned by default",
			nodeResourceGroups: map[string]string{},
			informerSynced:     true,
			node:               "node1",
			expected:           "rg",
		},
		{
			name:               "node RGs should be returned",
			nodeResourceGroups: map[string]string{"node1": "rg1", "node2": "rg2"},
			informerSynced:     true,
			node:               "node1",
			expected:           "rg1",
		},
		{
			name:               "error should be returned if informer hasn't synced yet",
			nodeResourceGroups: map[string]string{"node1": "rg1", "node2": "rg2"},
			informerSynced:     false,
			expectError:        true,
		},
	}

	az := getTestCloud()
	for _, test := range tests {
		az.nodeResourceGroups = test.nodeResourceGroups
		if test.informerSynced {
			az.nodeInformerSynced = func() bool { return true }
		} else {
			az.nodeInformerSynced = func() bool { return false }
		}
		actual, err := az.GetNodeResourceGroup(test.node)
		if test.expectError {
			assert.NotNil(t, err, test.name)
			continue
		}

		assert.Nil(t, err, test.name)
		assert.Equal(t, test.expected, actual, test.name)
	}
}
