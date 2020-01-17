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
	"context"
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/Azure/azure-sdk-for-go/services/network/mgmt/2019-06-01/network"
	"github.com/Azure/go-autorest/autorest/to"
	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestFindProbe(t *testing.T) {
	tests := []struct {
		msg           string
		existingProbe []network.Probe
		curProbe      network.Probe
		expected      bool
	}{
		{
			msg:      "empty existing probes should return false",
			expected: false,
		},
		{
			msg: "probe names match while ports unmatch should return false",
			existingProbe: []network.Probe{
				{
					Name: to.StringPtr("httpProbe"),
					ProbePropertiesFormat: &network.ProbePropertiesFormat{
						Port: to.Int32Ptr(1),
					},
				},
			},
			curProbe: network.Probe{
				Name: to.StringPtr("httpProbe"),
				ProbePropertiesFormat: &network.ProbePropertiesFormat{
					Port: to.Int32Ptr(2),
				},
			},
			expected: false,
		},
		{
			msg: "probe ports match while names unmatch should return false",
			existingProbe: []network.Probe{
				{
					Name: to.StringPtr("probe1"),
					ProbePropertiesFormat: &network.ProbePropertiesFormat{
						Port: to.Int32Ptr(1),
					},
				},
			},
			curProbe: network.Probe{
				Name: to.StringPtr("probe2"),
				ProbePropertiesFormat: &network.ProbePropertiesFormat{
					Port: to.Int32Ptr(1),
				},
			},
			expected: false,
		},
		{
			msg: "both probe ports and names match should return true",
			existingProbe: []network.Probe{
				{
					Name: to.StringPtr("matchName"),
					ProbePropertiesFormat: &network.ProbePropertiesFormat{
						Port: to.Int32Ptr(1),
					},
				},
			},
			curProbe: network.Probe{
				Name: to.StringPtr("matchName"),
				ProbePropertiesFormat: &network.ProbePropertiesFormat{
					Port: to.Int32Ptr(1),
				},
			},
			expected: true,
		},
	}

	for i, test := range tests {
		findResult := findProbe(test.existingProbe, test.curProbe)
		assert.Equal(t, test.expected, findResult, fmt.Sprintf("TestCase[%d]: %s", i, test.msg))
	}
}

func TestFindRule(t *testing.T) {
	tests := []struct {
		msg          string
		existingRule []network.LoadBalancingRule
		curRule      network.LoadBalancingRule
		expected     bool
	}{
		{
			msg:      "empty existing rules should return false",
			expected: false,
		},
		{
			msg: "rule names unmatch should return false",
			existingRule: []network.LoadBalancingRule{
				{
					Name: to.StringPtr("httpProbe1"),
					LoadBalancingRulePropertiesFormat: &network.LoadBalancingRulePropertiesFormat{
						FrontendPort: to.Int32Ptr(1),
					},
				},
			},
			curRule: network.LoadBalancingRule{
				Name: to.StringPtr("httpProbe2"),
				LoadBalancingRulePropertiesFormat: &network.LoadBalancingRulePropertiesFormat{
					FrontendPort: to.Int32Ptr(1),
				},
			},
			expected: false,
		},
		{
			msg: "rule names match while frontend ports unmatch should return false",
			existingRule: []network.LoadBalancingRule{
				{
					Name: to.StringPtr("httpProbe"),
					LoadBalancingRulePropertiesFormat: &network.LoadBalancingRulePropertiesFormat{
						FrontendPort: to.Int32Ptr(1),
					},
				},
			},
			curRule: network.LoadBalancingRule{
				Name: to.StringPtr("httpProbe"),
				LoadBalancingRulePropertiesFormat: &network.LoadBalancingRulePropertiesFormat{
					FrontendPort: to.Int32Ptr(2),
				},
			},
			expected: false,
		},
		{
			msg: "rule names match while backend ports unmatch should return false",
			existingRule: []network.LoadBalancingRule{
				{
					Name: to.StringPtr("httpProbe"),
					LoadBalancingRulePropertiesFormat: &network.LoadBalancingRulePropertiesFormat{
						BackendPort: to.Int32Ptr(1),
					},
				},
			},
			curRule: network.LoadBalancingRule{
				Name: to.StringPtr("httpProbe"),
				LoadBalancingRulePropertiesFormat: &network.LoadBalancingRulePropertiesFormat{
					BackendPort: to.Int32Ptr(2),
				},
			},
			expected: false,
		},
		{
			msg: "rule names match while idletimeout unmatch should return false",
			existingRule: []network.LoadBalancingRule{
				{
					Name: to.StringPtr("httpRule"),
					LoadBalancingRulePropertiesFormat: &network.LoadBalancingRulePropertiesFormat{
						IdleTimeoutInMinutes: to.Int32Ptr(1),
					},
				},
			},
			curRule: network.LoadBalancingRule{
				Name: to.StringPtr("httpRule"),
				LoadBalancingRulePropertiesFormat: &network.LoadBalancingRulePropertiesFormat{
					IdleTimeoutInMinutes: to.Int32Ptr(2),
				},
			},
			expected: false,
		},
		{
			msg: "rule names match while idletimeout nil should return true",
			existingRule: []network.LoadBalancingRule{
				{
					Name:                              to.StringPtr("httpRule"),
					LoadBalancingRulePropertiesFormat: &network.LoadBalancingRulePropertiesFormat{},
				},
			},
			curRule: network.LoadBalancingRule{
				Name: to.StringPtr("httpRule"),
				LoadBalancingRulePropertiesFormat: &network.LoadBalancingRulePropertiesFormat{
					IdleTimeoutInMinutes: to.Int32Ptr(2),
				},
			},
			expected: true,
		},
		{
			msg: "rule names match while LoadDistribution unmatch should return false",
			existingRule: []network.LoadBalancingRule{
				{
					Name: to.StringPtr("probe1"),
					LoadBalancingRulePropertiesFormat: &network.LoadBalancingRulePropertiesFormat{
						LoadDistribution: network.LoadDistributionSourceIP,
					},
				},
			},
			curRule: network.LoadBalancingRule{
				Name: to.StringPtr("probe2"),
				LoadBalancingRulePropertiesFormat: &network.LoadBalancingRulePropertiesFormat{
					LoadDistribution: network.LoadDistributionSourceIP,
				},
			},
			expected: false,
		},
		{
			msg: "both rule names and LoadBalancingRulePropertiesFormats match should return true",
			existingRule: []network.LoadBalancingRule{
				{
					Name: to.StringPtr("matchName"),
					LoadBalancingRulePropertiesFormat: &network.LoadBalancingRulePropertiesFormat{
						BackendPort:      to.Int32Ptr(2),
						FrontendPort:     to.Int32Ptr(2),
						LoadDistribution: network.LoadDistributionSourceIP,
					},
				},
			},
			curRule: network.LoadBalancingRule{
				Name: to.StringPtr("matchName"),
				LoadBalancingRulePropertiesFormat: &network.LoadBalancingRulePropertiesFormat{
					BackendPort:      to.Int32Ptr(2),
					FrontendPort:     to.Int32Ptr(2),
					LoadDistribution: network.LoadDistributionSourceIP,
				},
			},
			expected: true,
		},
	}

	for i, test := range tests {
		findResult := findRule(test.existingRule, test.curRule, true)
		assert.Equal(t, test.expected, findResult, fmt.Sprintf("TestCase[%d]: %s", i, test.msg))
	}
}

func TestGetIdleTimeout(t *testing.T) {
	for _, c := range []struct {
		desc        string
		annotations map[string]string
		i           *int32
		err         bool
	}{
		{desc: "no annotation"},
		{desc: "annotation empty value", annotations: map[string]string{ServiceAnnotationLoadBalancerIdleTimeout: ""}, err: true},
		{desc: "annotation not a number", annotations: map[string]string{ServiceAnnotationLoadBalancerIdleTimeout: "cookies"}, err: true},
		{desc: "annotation negative value", annotations: map[string]string{ServiceAnnotationLoadBalancerIdleTimeout: "-6"}, err: true},
		{desc: "annotation zero value", annotations: map[string]string{ServiceAnnotationLoadBalancerIdleTimeout: "0"}, err: true},
		{desc: "annotation too low value", annotations: map[string]string{ServiceAnnotationLoadBalancerIdleTimeout: "3"}, err: true},
		{desc: "annotation too high value", annotations: map[string]string{ServiceAnnotationLoadBalancerIdleTimeout: "31"}, err: true},
		{desc: "annotation good value", annotations: map[string]string{ServiceAnnotationLoadBalancerIdleTimeout: "24"}, i: to.Int32Ptr(24)},
	} {
		t.Run(c.desc, func(t *testing.T) {
			s := &v1.Service{}
			s.Annotations = c.annotations
			i, err := getIdleTimeout(s)

			if !reflect.DeepEqual(c.i, i) {
				t.Fatalf("got unexpected value: %d", to.Int32(i))
			}
			if (err != nil) != c.err {
				t.Fatalf("expected error=%v, got %v", c.err, err)
			}
		})
	}
}

func TestSubnet(t *testing.T) {
	for i, c := range []struct {
		desc     string
		service  *v1.Service
		expected *string
	}{
		{
			desc:     "No annotation should return nil",
			service:  &v1.Service{},
			expected: nil,
		},
		{
			desc: "annotation with subnet but no ILB should return nil",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						ServiceAnnotationLoadBalancerInternalSubnet: "subnet",
					},
				},
			},
			expected: nil,
		},
		{
			desc: "annotation with subnet but ILB=false should return nil",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						ServiceAnnotationLoadBalancerInternalSubnet: "subnet",
						ServiceAnnotationLoadBalancerInternal:       "false",
					},
				},
			},
			expected: nil,
		},
		{
			desc: "annotation with empty subnet should return nil",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						ServiceAnnotationLoadBalancerInternalSubnet: "",
						ServiceAnnotationLoadBalancerInternal:       "true",
					},
				},
			},
			expected: nil,
		},
		{
			desc: "annotation with subnet and ILB should return subnet",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						ServiceAnnotationLoadBalancerInternalSubnet: "subnet",
						ServiceAnnotationLoadBalancerInternal:       "true",
					},
				},
			},
			expected: to.StringPtr("subnet"),
		},
	} {
		real := subnet(c.service)
		assert.Equal(t, c.expected, real, fmt.Sprintf("TestCase[%d]: %s", i, c.desc))
	}
}

func TestEnsureLoadBalancerDeleted(t *testing.T) {
	const vmCount = 8
	const availabilitySetCount = 4

	tests := []struct {
		desc              string
		service           v1.Service
		expectCreateError bool
	}{
		{
			desc:    "external service should be created and deleted successfully",
			service: getTestService("test1", v1.ProtocolTCP, nil, 80),
		},
		{
			desc:    "internal service should be created and deleted successfully",
			service: getInternalTestService("test2", 80),
		},
		{
			desc:    "annotated service with same resourceGroup should be created and deleted successfully",
			service: getResourceGroupTestService("test3", "rg", "", 80),
		},
		{
			desc:              "annotated service with different resourceGroup shouldn't be created but should be deleted successfully",
			service:           getResourceGroupTestService("test4", "random-rg", "1.2.3.4", 80),
			expectCreateError: true,
		},
	}

	az := getTestCloud()
	for i, c := range tests {
		clusterResources := getClusterResources(az, vmCount, availabilitySetCount)
		getTestSecurityGroup(az)
		if c.service.Annotations[ServiceAnnotationLoadBalancerInternal] == "true" {
			addTestSubnet(t, az, &c.service)
		}

		// create the service first.
		lbStatus, err := az.EnsureLoadBalancer(context.TODO(), testClusterName, &c.service, clusterResources.nodes)
		if c.expectCreateError {
			assert.NotNil(t, err, "TestCase[%d]: %s", i, c.desc)
		} else {
			assert.Nil(t, err, "TestCase[%d]: %s", i, c.desc)
			assert.NotNil(t, lbStatus, "TestCase[%d]: %s", i, c.desc)
			result, rerr := az.LoadBalancerClient.List(context.TODO(), az.Config.ResourceGroup)
			assert.Nil(t, rerr, "TestCase[%d]: %s", i, c.desc)
			assert.Equal(t, len(result), 1, "TestCase[%d]: %s", i, c.desc)
			assert.Equal(t, len(*result[0].LoadBalancingRules), 1, "TestCase[%d]: %s", i, c.desc)
		}

		// finally, delete it.
		err = az.EnsureLoadBalancerDeleted(context.TODO(), testClusterName, &c.service)
		assert.Nil(t, err, "TestCase[%d]: %s", i, c.desc)
		result, rerr := az.LoadBalancerClient.List(context.Background(), az.Config.ResourceGroup)
		assert.Nil(t, rerr, "TestCase[%d]: %s", i, c.desc)
		assert.Equal(t, len(result), 0, "TestCase[%d]: %s", i, c.desc)
	}
}

func TestServiceOwnsPublicIP(t *testing.T) {
	tests := []struct {
		desc        string
		pip         *network.PublicIPAddress
		clusterName string
		serviceName string
		expected    bool
	}{
		{
			desc:        "false should be returned when pip is nil",
			clusterName: "kubernetes",
			serviceName: "nginx",
			expected:    false,
		},
		{
			desc: "false should be returned when service name tag doesn't match",
			pip: &network.PublicIPAddress{
				Tags: map[string]*string{
					serviceTagKey: to.StringPtr("nginx"),
				},
			},
			serviceName: "web",
			expected:    false,
		},
		{
			desc: "true should be returned when service name tag matches and cluster name tag is not set",
			pip: &network.PublicIPAddress{
				Tags: map[string]*string{
					serviceTagKey: to.StringPtr("nginx"),
				},
			},
			clusterName: "kubernetes",
			serviceName: "nginx",
			expected:    true,
		},
		{
			desc: "false should be returned when cluster name doesn't match",
			pip: &network.PublicIPAddress{
				Tags: map[string]*string{
					serviceTagKey:  to.StringPtr("nginx"),
					clusterNameKey: to.StringPtr("kubernetes"),
				},
			},
			clusterName: "k8s",
			serviceName: "nginx",
			expected:    false,
		},
		{
			desc: "false should be returned when cluster name matches while service name doesn't match",
			pip: &network.PublicIPAddress{
				Tags: map[string]*string{
					serviceTagKey:  to.StringPtr("web"),
					clusterNameKey: to.StringPtr("kubernetes"),
				},
			},
			clusterName: "kubernetes",
			serviceName: "nginx",
			expected:    false,
		},
		{
			desc: "true should be returned when both service name tag and cluster name match",
			pip: &network.PublicIPAddress{
				Tags: map[string]*string{
					serviceTagKey:  to.StringPtr("nginx"),
					clusterNameKey: to.StringPtr("kubernetes"),
				},
			},
			clusterName: "kubernetes",
			serviceName: "nginx",
			expected:    true,
		},
	}

	for i, c := range tests {
		owns := serviceOwnsPublicIP(c.pip, c.clusterName, c.serviceName)
		assert.Equal(t, owns, c.expected, "TestCase[%d]: %s", i, c.desc)
	}
}

func TestGetPublicIPAddressResourceGroup(t *testing.T) {
	az := getTestCloud()

	for i, c := range []struct {
		desc        string
		annotations map[string]string
		expected    string
	}{
		{
			desc:     "no annotation",
			expected: "rg",
		},
		{
			desc:        "annoation with empty string resource group",
			annotations: map[string]string{ServiceAnnotationLoadBalancerResourceGroup: ""},
			expected:    "rg",
		},
		{
			desc:        "annoation with non-empty resource group ",
			annotations: map[string]string{ServiceAnnotationLoadBalancerResourceGroup: "rg2"},
			expected:    "rg2",
		},
	} {
		t.Run(c.desc, func(t *testing.T) {
			s := &v1.Service{}
			s.Annotations = c.annotations
			real := az.getPublicIPAddressResourceGroup(s)
			assert.Equal(t, c.expected, real, "TestCase[%d]: %s", i, c.desc)
		})
	}
}

func TestGetServiceTags(t *testing.T) {
	tests := []struct {
		desc     string
		service  *v1.Service
		expected []string
	}{
		{
			desc:     "nil should be returned when service is nil",
			service:  nil,
			expected: nil,
		},
		{
			desc:     "nil should be returned when service has no annotations",
			service:  &v1.Service{},
			expected: nil,
		},
		{
			desc: "single tag should be returned when service has set one annotations",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						ServiceAnnotationAllowedServiceTag: "tag1",
					},
				},
			},
			expected: []string{"tag1"},
		},
		{
			desc: "multiple tags should be returned when service has set multi-annotations",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						ServiceAnnotationAllowedServiceTag: "tag1, tag2",
					},
				},
			},
			expected: []string{"tag1", "tag2"},
		},
		{
			desc: "correct tags should be returned when comma or spaces are included in the annotations",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						ServiceAnnotationAllowedServiceTag: ", tag1, ",
					},
				},
			},
			expected: []string{"tag1"},
		},
	}

	for i, c := range tests {
		tags := getServiceTags(c.service)
		assert.Equal(t, tags, c.expected, "TestCase[%d]: %s", i, c.desc)
	}
}

func TestGetServiceLoadBalancer(t *testing.T) {
	testCases := []struct {
		desc           string
		existingLBs    []network.LoadBalancer
		service        v1.Service
		annotations    map[string]string
		sku            string
		wantLB         bool
		expectedLB     *network.LoadBalancer
		expectedStatus *v1.LoadBalancerStatus
		expectedExists bool
		expectedError  bool
	}{
		{
			desc: "getServiceLoadBalancer shall return corresponding lb, status, exists if there are exsisted lbs",
			existingLBs: []network.LoadBalancer{
				{
					Name: to.StringPtr("lb1"),
					LoadBalancerPropertiesFormat: &network.LoadBalancerPropertiesFormat{
						FrontendIPConfigurations: &[]network.FrontendIPConfiguration{
							{
								Name: to.StringPtr("atest1"),
								FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
									PublicIPAddress: &network.PublicIPAddress{ID: to.StringPtr("id1")},
								},
							},
						},
					},
				},
			},
			service: getTestService("test1", v1.ProtocolTCP, nil, 80),
			wantLB:  false,
			expectedLB: &network.LoadBalancer{
				Name: to.StringPtr("lb1"),
				LoadBalancerPropertiesFormat: &network.LoadBalancerPropertiesFormat{
					FrontendIPConfigurations: &[]network.FrontendIPConfiguration{
						{
							Name: to.StringPtr("atest1"),
							FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
								PublicIPAddress: &network.PublicIPAddress{ID: to.StringPtr("id1")},
							},
						},
					},
				},
			},
			expectedStatus: &v1.LoadBalancerStatus{Ingress: []v1.LoadBalancerIngress{{IP: "", Hostname: ""}}},
			expectedExists: true,
			expectedError:  false,
		},
		{
			desc:           "getServiceLoadBalancer shall report error if there're loadbalancer mode annotations on a standard lb",
			service:        getTestService("test1", v1.ProtocolTCP, nil, 80),
			annotations:    map[string]string{ServiceAnnotationLoadBalancerMode: "__auto__"},
			sku:            "standard",
			expectedExists: false,
			expectedError:  true,
		},
		{
			desc: "getServiceLoadBalancer shall select the lb with minimum lb rules if wantLb is true, the sku is " +
				"not standard and there are existing lbs already",
			existingLBs: []network.LoadBalancer{
				{
					Name: to.StringPtr("testCluster"),
					LoadBalancerPropertiesFormat: &network.LoadBalancerPropertiesFormat{
						LoadBalancingRules: &[]network.LoadBalancingRule{
							{Name: to.StringPtr("rule1")},
						},
					},
				},
				{
					Name: to.StringPtr("as-1"),
					LoadBalancerPropertiesFormat: &network.LoadBalancerPropertiesFormat{
						LoadBalancingRules: &[]network.LoadBalancingRule{
							{Name: to.StringPtr("rule1")},
							{Name: to.StringPtr("rule2")},
						},
					},
				},
				{
					Name: to.StringPtr("as-2"),
					LoadBalancerPropertiesFormat: &network.LoadBalancerPropertiesFormat{
						LoadBalancingRules: &[]network.LoadBalancingRule{
							{Name: to.StringPtr("rule1")},
							{Name: to.StringPtr("rule2")},
							{Name: to.StringPtr("rule3")},
						},
					},
				},
			},
			service:     getTestService("test1", v1.ProtocolTCP, nil, 80),
			annotations: map[string]string{ServiceAnnotationLoadBalancerMode: "__auto__"},
			wantLB:      true,
			expectedLB: &network.LoadBalancer{
				Name: to.StringPtr("testCluster"),
				LoadBalancerPropertiesFormat: &network.LoadBalancerPropertiesFormat{
					LoadBalancingRules: &[]network.LoadBalancingRule{
						{Name: to.StringPtr("rule1")},
					},
				},
			},
			expectedExists: false,
			expectedError:  false,
		},
		{
			desc:    "getServiceLoadBalancer shall create a new lb otherwise",
			service: getTestService("test1", v1.ProtocolTCP, nil, 80),
			expectedLB: &network.LoadBalancer{
				Name:                         to.StringPtr("testCluster"),
				Location:                     to.StringPtr("westus"),
				LoadBalancerPropertiesFormat: &network.LoadBalancerPropertiesFormat{},
			},
			expectedExists: false,
			expectedError:  false,
		},
	}

	for i, test := range testCases {
		az := getTestCloud()
		clusterResources := getClusterResources(az, 3, 3)

		for _, existingLB := range test.existingLBs {
			err := az.LoadBalancerClient.CreateOrUpdate(context.TODO(), "rg", *existingLB.Name, existingLB, "")
			if err != nil {
				t.Fatalf("TestCase[%d] meets unexpected error: %v", i, err)
			}
		}
		test.service.Annotations = test.annotations
		az.LoadBalancerSku = test.sku
		lb, status, exists, err := az.getServiceLoadBalancer(&test.service, testClusterName,
			clusterResources.nodes, test.wantLB)
		assert.Equal(t, test.expectedLB, lb, "TestCase[%d]: %s", i, test.desc)
		assert.Equal(t, test.expectedStatus, status, "TestCase[%d]: %s", i, test.desc)
		assert.Equal(t, test.expectedExists, exists, "TestCase[%d]: %s", i, test.desc)
		assert.Equal(t, test.expectedError, err != nil, "TestCase[%d]: %s", i, test.desc)
	}
}

func TestIsFrontendIPChanged(t *testing.T) {
	testCases := []struct {
		desc                   string
		config                 network.FrontendIPConfiguration
		service                v1.Service
		lbFrontendIPConfigName string
		annotations            string
		loadBalancerIP         string
		exsistingSubnet        network.Subnet
		exsistingPIPs          []network.PublicIPAddress
		expectedFlag           bool
		expectedError          bool
	}{
		{
			desc: "isFrontendIPChanged shall return true if config.Name has a prefix of lb's name and " +
				"config.Name != lbFrontendIPConfigName",
			config:                 network.FrontendIPConfiguration{Name: to.StringPtr("atest1-name")},
			service:                getInternalTestService("test1", 80),
			lbFrontendIPConfigName: "configName",
			expectedFlag:           true,
			expectedError:          false,
		},
		{
			desc: "isFrontendIPChanged shall return false if config.Name doesn't have a prefix of lb's name " +
				"and config.Name != lbFrontendIPConfigName",
			config:                 network.FrontendIPConfiguration{Name: to.StringPtr("btest1-name")},
			service:                getInternalTestService("test1", 80),
			lbFrontendIPConfigName: "configName",
			expectedFlag:           false,
			expectedError:          false,
		},
		{
			desc: "isFrontendIPChanged shall return false if the service is internal, no loadBalancerIP is given, " +
				"subnetName == nil and config.PrivateIPAllocationMethod == network.Static",
			config: network.FrontendIPConfiguration{
				Name: to.StringPtr("atest1-name"),
				FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
					PrivateIPAllocationMethod: network.IPAllocationMethod("static"),
				},
			},
			service:       getInternalTestService("test1", 80),
			expectedFlag:  true,
			expectedError: false,
		},
		{
			desc: "isFrontendIPChanged shall return false if the service is internal, no loadBalancerIP is given, " +
				"subnetName == nil and config.PrivateIPAllocationMethod != network.Static",
			config: network.FrontendIPConfiguration{
				Name: to.StringPtr("btest1-name"),
				FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
					PrivateIPAllocationMethod: network.IPAllocationMethod("dynamic"),
				},
			},
			lbFrontendIPConfigName: "btest1-name",
			service:                getInternalTestService("test1", 80),
			expectedFlag:           false,
			expectedError:          false,
		},
		{
			desc: "isFrontendIPChanged shall return true if the service is internal and " +
				"config.Subnet.Name == subnet.Name",
			config: network.FrontendIPConfiguration{
				Name: to.StringPtr("btest1-name"),
				FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
					Subnet: &network.Subnet{Name: to.StringPtr("testSubnet")},
				},
			},
			lbFrontendIPConfigName: "btest1-name",
			service:                getInternalTestService("test1", 80),
			annotations:            "testSubnet",
			exsistingSubnet:        network.Subnet{Name: to.StringPtr("testSubnet1")},
			expectedFlag:           true,
			expectedError:          false,
		},
		{
			desc: "isFrontendIPChanged shall return true if the service is internal, subnet == nil, " +
				"loadBalancerIP != '' and config.PrivateIPAllocationMethod != 'static'",
			config: network.FrontendIPConfiguration{
				Name: to.StringPtr("btest1-name"),
				FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
					PrivateIPAllocationMethod: network.IPAllocationMethod("dynamic"),
				},
			},
			lbFrontendIPConfigName: "btest1-name",
			service:                getInternalTestService("test1", 80),
			loadBalancerIP:         "1.1.1.1",
			expectedFlag:           true,
			expectedError:          false,
		},
		{
			desc: "isFrontendIPChanged shall return true if the service is internal, subnet == nil and " +
				"loadBalancerIP != config.PrivateIPAddress",
			config: network.FrontendIPConfiguration{
				Name: to.StringPtr("btest1-name"),
				FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
					PrivateIPAllocationMethod: network.IPAllocationMethod("static"),
					PrivateIPAddress:          to.StringPtr("1.1.1.2"),
				},
			},
			lbFrontendIPConfigName: "btest1-name",
			service:                getInternalTestService("test1", 80),
			loadBalancerIP:         "1.1.1.1",
			expectedFlag:           true,
			expectedError:          false,
		},
		{
			desc: "isFrontendIPChanged shall return false if no loadbalancerIP is given",
			config: network.FrontendIPConfiguration{
				Name: to.StringPtr("btest1-name"),
				FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
					PrivateIPAllocationMethod: network.IPAllocationMethod("static"),
					PrivateIPAddress:          to.StringPtr("1.1.1.2"),
				},
			},
			lbFrontendIPConfigName: "btest1-name",
			service:                getTestService("test1", v1.ProtocolTCP, nil, 80),
			expectedFlag:           false,
			expectedError:          false,
		},
		{
			desc: "isFrontendIPChanged shall return false if config.PublicIPAddress == nil",
			config: network.FrontendIPConfiguration{
				Name: to.StringPtr("btest1-name"),
				FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
					PublicIPAddress: nil,
				},
			},
			lbFrontendIPConfigName: "btest1-name",
			service:                getTestService("test1", v1.ProtocolTCP, nil, 80),
			loadBalancerIP:         "1.1.1.1",
			exsistingPIPs: []network.PublicIPAddress{
				{
					Name: to.StringPtr("pipName"),
					PublicIPAddressPropertiesFormat: &network.PublicIPAddressPropertiesFormat{
						IPAddress: to.StringPtr("1.1.1.1"),
					},
				},
			},
			expectedFlag:  false,
			expectedError: false,
		},
		{
			desc: "isFrontendIPChanged shall return false if pip.ID == config.PublicIPAddress.ID",
			config: network.FrontendIPConfiguration{
				Name: to.StringPtr("btest1-name"),
				FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
					PublicIPAddress: &network.PublicIPAddress{ID: to.StringPtr("/subscriptions/subscription" +
						"/resourceGroups/rg/providers/Microsoft.Network/publicIPAddresses/pipName")},
				},
			},
			lbFrontendIPConfigName: "btest1-name",
			service:                getTestService("test1", v1.ProtocolTCP, nil, 80),
			loadBalancerIP:         "1.1.1.1",
			exsistingPIPs: []network.PublicIPAddress{
				{
					Name: to.StringPtr("pipName"),
					PublicIPAddressPropertiesFormat: &network.PublicIPAddressPropertiesFormat{
						IPAddress: to.StringPtr("1.1.1.1"),
					},
				},
			},
			expectedFlag:  false,
			expectedError: false,
		},
		{
			desc: "isFrontendIPChanged shall return true if pip.ID != config.PublicIPAddress.ID",
			config: network.FrontendIPConfiguration{
				Name: to.StringPtr("btest1-name"),
				FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
					PublicIPAddress: &network.PublicIPAddress{ID: to.StringPtr("/subscriptions/subscription" +
						"/resourceGroups/rg/providers/Microsoft.Network/publicIPAddresses/pipName1")},
				},
			},
			lbFrontendIPConfigName: "btest1-name",
			service:                getTestService("test1", v1.ProtocolTCP, nil, 80),
			loadBalancerIP:         "1.1.1.1",
			exsistingPIPs: []network.PublicIPAddress{
				{
					Name: to.StringPtr("pipName"),
					PublicIPAddressPropertiesFormat: &network.PublicIPAddressPropertiesFormat{
						IPAddress: to.StringPtr("1.1.1.1"),
					},
				},
			},
			expectedFlag:  true,
			expectedError: false,
		},
	}

	for i, test := range testCases {
		az := getTestCloud()
		err := az.SubnetsClient.CreateOrUpdate(context.TODO(), "rg", "vnet", "testSubnet", test.exsistingSubnet)
		if err != nil {
			t.Fatalf("TestCase[%d] meets unexpected error: %v", i, err)
		}
		for _, existingPIP := range test.exsistingPIPs {
			err := az.PublicIPAddressesClient.CreateOrUpdate(context.TODO(), "rg", "pipName", existingPIP)
			if err != nil {
				t.Fatalf("TestCase[%d] meets unexpected error: %v", i, err)
			}
		}
		test.service.Spec.LoadBalancerIP = test.loadBalancerIP
		test.service.Annotations[ServiceAnnotationLoadBalancerInternalSubnet] = test.annotations
		flag, rerr := az.isFrontendIPChanged("testCluster", test.config,
			&test.service, test.lbFrontendIPConfigName)
		assert.Equal(t, test.expectedFlag, flag, "TestCase[%d]: %s", i, test.desc)
		assert.Equal(t, test.expectedError, rerr != nil, "TestCase[%d]: %s", i, test.desc)
	}
}

func TestDeterminePublicIPName(t *testing.T) {
	testCases := []struct {
		desc           string
		loadBalancerIP string
		exsistingPIPs  []network.PublicIPAddress
		expectedIP     string
		expectedError  bool
	}{
		{
			desc: "determinePublicIpName shall get public IP from az.getPublicIPName if no specific " +
				"loadBalancerIP is given",
			expectedIP:    "testCluster-atest1",
			expectedError: false,
		},
		{
			desc:           "determinePublicIpName shall report error if loadBalancerIP is not in the resource group",
			loadBalancerIP: "1.2.3.4",
			expectedIP:     "",
			expectedError:  true,
		},
		{
			desc: "determinePublicIpName shall return loadBalancerIP in service.Spec if it's in the " +
				"resource group",
			loadBalancerIP: "1.2.3.4",
			exsistingPIPs: []network.PublicIPAddress{
				{
					Name: to.StringPtr("pipName"),
					PublicIPAddressPropertiesFormat: &network.PublicIPAddressPropertiesFormat{
						IPAddress: to.StringPtr("1.2.3.4"),
					},
				},
			},
			expectedIP:    "pipName",
			expectedError: false,
		},
	}
	for i, test := range testCases {
		az := getTestCloud()
		service := getTestService("test1", v1.ProtocolTCP, nil, 80)
		service.Spec.LoadBalancerIP = test.loadBalancerIP
		for _, existingPIP := range test.exsistingPIPs {
			err := az.PublicIPAddressesClient.CreateOrUpdate(context.TODO(), "rg", "test", existingPIP)
			if err != nil {
				t.Fatalf("TestCase[%d] meets unexpected error: %v", i, err)
			}
		}
		ip, _, err := az.determinePublicIPName("testCluster", &service)
		assert.Equal(t, test.expectedIP, ip, "TestCase[%d]: %s", i, test.desc)
		assert.Equal(t, test.expectedError, err != nil, "TestCase[%d]: %s", i, test.desc)
	}
}

func TestReconcileLoadBalancerRule(t *testing.T) {
	testCases := []struct {
		desc            string
		service         v1.Service
		loadBalancerSku string
		wantLb          bool
		expectedProbes  []network.Probe
		expectedRules   []network.LoadBalancingRule
		expectedErr     error
	}{
		{
			desc:    "reconcileLoadBalancerRule shall return nil if wantLb is false",
			service: getTestService("test1", v1.ProtocolTCP, nil, 80),
			wantLb:  false,
		},
		{
			desc:    "reconcileLoadBalancerRule shall return corresponding probe and lbRule(blb)",
			service: getTestService("test1", v1.ProtocolTCP, map[string]string{"service.beta.kubernetes.io/azure-load-balancer-disable-tcp-reset": "true"}, 80),
			wantLb:  true,
			expectedProbes: []network.Probe{
				{
					Name: to.StringPtr("atest1-TCP-80"),
					ProbePropertiesFormat: &network.ProbePropertiesFormat{
						Protocol:          network.ProbeProtocol("Tcp"),
						Port:              to.Int32Ptr(10080),
						IntervalInSeconds: to.Int32Ptr(5),
						NumberOfProbes:    to.Int32Ptr(2),
					},
				},
			},
			expectedRules: []network.LoadBalancingRule{
				{
					Name: to.StringPtr("atest1-TCP-80"),
					LoadBalancingRulePropertiesFormat: &network.LoadBalancingRulePropertiesFormat{
						Protocol: network.TransportProtocol("Tcp"),
						FrontendIPConfiguration: &network.SubResource{
							ID: to.StringPtr("frontendIPConfigID"),
						},
						BackendAddressPool: &network.SubResource{
							ID: to.StringPtr("backendPoolID"),
						},
						LoadDistribution:     "Default",
						FrontendPort:         to.Int32Ptr(80),
						BackendPort:          to.Int32Ptr(80),
						EnableFloatingIP:     to.BoolPtr(true),
						DisableOutboundSnat:  to.BoolPtr(false),
						IdleTimeoutInMinutes: to.Int32Ptr(0),
						Probe: &network.SubResource{
							ID: to.StringPtr("/subscriptions/subscription/resourceGroups/rg/providers/" +
								"Microsoft.Network/loadBalancers/lbname/probes/atest1-TCP-80"),
						},
						EnableTCPReset: nil,
					},
				},
			},
		},
		{
			desc:            "reconcileLoadBalancerRule shall return corresponding probe and lbRule (slb without tcp reset)",
			service:         getTestService("test1", v1.ProtocolTCP, map[string]string{"service.beta.kubernetes.io/azure-load-balancer-disable-tcp-reset": "True"}, 80),
			loadBalancerSku: "standard",
			wantLb:          true,
			expectedProbes: []network.Probe{
				{
					Name: to.StringPtr("atest1-TCP-80"),
					ProbePropertiesFormat: &network.ProbePropertiesFormat{
						Protocol:          network.ProbeProtocol("Tcp"),
						Port:              to.Int32Ptr(10080),
						IntervalInSeconds: to.Int32Ptr(5),
						NumberOfProbes:    to.Int32Ptr(2),
					},
				},
			},
			expectedRules: []network.LoadBalancingRule{
				{
					Name: to.StringPtr("atest1-TCP-80"),
					LoadBalancingRulePropertiesFormat: &network.LoadBalancingRulePropertiesFormat{
						Protocol: network.TransportProtocol("Tcp"),
						FrontendIPConfiguration: &network.SubResource{
							ID: to.StringPtr("frontendIPConfigID"),
						},
						BackendAddressPool: &network.SubResource{
							ID: to.StringPtr("backendPoolID"),
						},
						LoadDistribution:     "Default",
						FrontendPort:         to.Int32Ptr(80),
						BackendPort:          to.Int32Ptr(80),
						EnableFloatingIP:     to.BoolPtr(true),
						DisableOutboundSnat:  to.BoolPtr(false),
						IdleTimeoutInMinutes: to.Int32Ptr(0),
						Probe: &network.SubResource{
							ID: to.StringPtr("/subscriptions/subscription/resourceGroups/rg/providers/" +
								"Microsoft.Network/loadBalancers/lbname/probes/atest1-TCP-80"),
						},
						EnableTCPReset: to.BoolPtr(false),
					},
				},
			},
		},
		{
			desc:            "reconcileLoadBalancerRule shall return corresponding probe and lbRule(slb with tcp reset)",
			service:         getTestService("test1", v1.ProtocolTCP, nil, 80),
			loadBalancerSku: "standard",
			wantLb:          true,
			expectedProbes: []network.Probe{
				{
					Name: to.StringPtr("atest1-TCP-80"),
					ProbePropertiesFormat: &network.ProbePropertiesFormat{
						Protocol:          network.ProbeProtocol("Tcp"),
						Port:              to.Int32Ptr(10080),
						IntervalInSeconds: to.Int32Ptr(5),
						NumberOfProbes:    to.Int32Ptr(2),
					},
				},
			},
			expectedRules: []network.LoadBalancingRule{
				{
					Name: to.StringPtr("atest1-TCP-80"),
					LoadBalancingRulePropertiesFormat: &network.LoadBalancingRulePropertiesFormat{
						Protocol: network.TransportProtocol("Tcp"),
						FrontendIPConfiguration: &network.SubResource{
							ID: to.StringPtr("frontendIPConfigID"),
						},
						BackendAddressPool: &network.SubResource{
							ID: to.StringPtr("backendPoolID"),
						},
						LoadDistribution:     "Default",
						FrontendPort:         to.Int32Ptr(80),
						BackendPort:          to.Int32Ptr(80),
						EnableFloatingIP:     to.BoolPtr(true),
						DisableOutboundSnat:  to.BoolPtr(false),
						IdleTimeoutInMinutes: to.Int32Ptr(0),
						Probe: &network.SubResource{
							ID: to.StringPtr("/subscriptions/subscription/resourceGroups/rg/providers/" +
								"Microsoft.Network/loadBalancers/lbname/probes/atest1-TCP-80"),
						},
						EnableTCPReset: to.BoolPtr(true),
					},
				},
			},
		},
	}
	for i, test := range testCases {
		az := getTestCloud()
		az.Config.LoadBalancerSku = test.loadBalancerSku
		probe, lbrule, err := az.reconcileLoadBalancerRule(&test.service, test.wantLb,
			"frontendIPConfigID", "backendPoolID", "lbname", to.Int32Ptr(0))

		if test.expectedErr != nil {
			assert.Equal(t, test.expectedErr, err, "TestCase[%d]: %s", i, test.desc)
		} else {
			assert.Equal(t, test.expectedProbes, probe, "TestCase[%d]: %s", i, test.desc)
			assert.Equal(t, test.expectedRules, lbrule, "TestCase[%d]: %s", i, test.desc)
			assert.Nil(t, err)
		}
	}
}

func getTestLoadBalancer(name, rgName, clusterName, identifier *string, service v1.Service, lbSku string) network.LoadBalancer {
	lb := network.LoadBalancer{
		Name: name,
		Sku: &network.LoadBalancerSku{
			Name: network.LoadBalancerSkuName(lbSku),
		},
		LoadBalancerPropertiesFormat: &network.LoadBalancerPropertiesFormat{
			FrontendIPConfigurations: &[]network.FrontendIPConfiguration{
				{
					Name: identifier,
					FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
						PublicIPAddress: &network.PublicIPAddress{ID: to.StringPtr("id1")},
					},
				},
			},
			BackendAddressPools: &[]network.BackendAddressPool{
				{Name: clusterName},
			},
			Probes: &[]network.Probe{
				{
					Name: to.StringPtr(*identifier + "-" + string(service.Spec.Ports[0].Protocol) +
						"-" + strconv.Itoa(int(service.Spec.Ports[0].Port))),
					ProbePropertiesFormat: &network.ProbePropertiesFormat{
						Port: to.Int32Ptr(10080),
					},
				},
			},
			LoadBalancingRules: &[]network.LoadBalancingRule{
				{
					Name: to.StringPtr(*identifier + "-" + string(service.Spec.Ports[0].Protocol) +
						"-" + strconv.Itoa(int(service.Spec.Ports[0].Port))),
					LoadBalancingRulePropertiesFormat: &network.LoadBalancingRulePropertiesFormat{
						Protocol: network.TransportProtocol(strings.Title(
							strings.ToLower(string(service.Spec.Ports[0].Protocol)))),
						FrontendIPConfiguration: &network.SubResource{
							ID: to.StringPtr("/subscriptions/subscription/resourceGroups/" + *rgName + "/providers/" +
								"Microsoft.Network/loadBalancers/" + *name + "/frontendIPConfigurations/atest1"),
						},
						BackendAddressPool: &network.SubResource{
							ID: to.StringPtr("/subscriptions/subscription/resourceGroups/" + *rgName + "/providers/" +
								"Microsoft.Network/loadBalancers/" + *name + "/backendAddressPools/" + *clusterName),
						},
						LoadDistribution: network.LoadDistribution("Default"),
						FrontendPort:     to.Int32Ptr(service.Spec.Ports[0].Port),
						BackendPort:      to.Int32Ptr(service.Spec.Ports[0].Port),
						EnableFloatingIP: to.BoolPtr(true),
						EnableTCPReset:   to.BoolPtr(strings.EqualFold(lbSku, "standard")),
						Probe: &network.SubResource{
							ID: to.StringPtr("/subscriptions/subscription/resourceGroups/" + *rgName + "/providers/Microsoft.Network/loadBalancers/testCluster/probes/atest1-TCP-80"),
						},
					},
				},
			},
		},
	}
	return lb
}

func TestReconcileLoadBalancer(t *testing.T) {
	service1 := getTestService("test1", v1.ProtocolTCP, nil, 80)
	basicLb1 := getTestLoadBalancer(to.StringPtr("lb1"), to.StringPtr("rg"), to.StringPtr("testCluster"), to.StringPtr("atest1"), service1, "Basic")

	service2 := getTestService("test1", v1.ProtocolTCP, nil, 80)
	basicLb2 := getTestLoadBalancer(to.StringPtr("lb1"), to.StringPtr("rg"), to.StringPtr("testCluster"), to.StringPtr("btest1"), service2, "Basic")
	basicLb2.Name = to.StringPtr("testCluster")
	basicLb2.FrontendIPConfigurations = &[]network.FrontendIPConfiguration{
		{
			Name: to.StringPtr("btest1"),
			FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
				PublicIPAddress: &network.PublicIPAddress{ID: to.StringPtr("id1")},
			},
		},
	}

	service3 := getTestService("test1", v1.ProtocolTCP, nil, 80)
	modifiedLb1 := getTestLoadBalancer(to.StringPtr("testCluster"), to.StringPtr("rg"), to.StringPtr("testCluster"), to.StringPtr("atest1"), service3, "Basic")
	modifiedLb1.FrontendIPConfigurations = &[]network.FrontendIPConfiguration{
		{
			Name: to.StringPtr("atest1"),
			FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
				PublicIPAddress: &network.PublicIPAddress{ID: to.StringPtr("id1")},
			},
		},
		{
			Name: to.StringPtr("btest1"),
			FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
				PublicIPAddress: &network.PublicIPAddress{ID: to.StringPtr("id1")},
			},
		},
	}
	modifiedLb1.Probes = &[]network.Probe{
		{
			Name: to.StringPtr("atest1-" + string(service3.Spec.Ports[0].Protocol) +
				"-" + strconv.Itoa(int(service3.Spec.Ports[0].Port))),
			ProbePropertiesFormat: &network.ProbePropertiesFormat{
				Port: to.Int32Ptr(10080),
			},
		},
		{
			Name: to.StringPtr("atest1-" + string(service3.Spec.Ports[0].Protocol) +
				"-" + strconv.Itoa(int(service3.Spec.Ports[0].Port))),
			ProbePropertiesFormat: &network.ProbePropertiesFormat{
				Port: to.Int32Ptr(10081),
			},
		},
	}
	expectedLb1 := getTestLoadBalancer(to.StringPtr("testCluster"), to.StringPtr("rg"), to.StringPtr("testCluster"), to.StringPtr("atest1"), service3, "Basic")
	(*expectedLb1.LoadBalancerPropertiesFormat.LoadBalancingRules)[0].DisableOutboundSnat = to.BoolPtr(false)
	(*expectedLb1.LoadBalancerPropertiesFormat.LoadBalancingRules)[0].EnableTCPReset = nil
	expectedLb1.FrontendIPConfigurations = &[]network.FrontendIPConfiguration{
		{
			Name: to.StringPtr("btest1"),
			FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
				PublicIPAddress: &network.PublicIPAddress{ID: to.StringPtr("id1")},
			},
		},
		{
			Name: to.StringPtr("atest1"),
			FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
				PublicIPAddress: &network.PublicIPAddress{ID: to.StringPtr("/subscriptions/subscription/" +
					"resourceGroups/rg/providers/Microsoft.Network/publicIPAddresses/pipName")},
			},
		},
	}

	service4 := getTestService("test1", v1.ProtocolTCP, map[string]string{"service.beta.kubernetes.io/azure-load-balancer-disable-tcp-reset": "true"}, 80)
	existingSLB := getTestLoadBalancer(to.StringPtr("testCluster"), to.StringPtr("rg"), to.StringPtr("testCluster"), to.StringPtr("atest1"), service4, "Standard")
	existingSLB.FrontendIPConfigurations = &[]network.FrontendIPConfiguration{
		{
			Name: to.StringPtr("atest1"),
			FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
				PublicIPAddress: &network.PublicIPAddress{ID: to.StringPtr("id1")},
			},
		},
		{
			Name: to.StringPtr("btest1"),
			FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
				PublicIPAddress: &network.PublicIPAddress{ID: to.StringPtr("id1")},
			},
		},
	}
	existingSLB.Probes = &[]network.Probe{
		{
			Name: to.StringPtr("atest1-" + string(service4.Spec.Ports[0].Protocol) +
				"-" + strconv.Itoa(int(service4.Spec.Ports[0].Port))),
			ProbePropertiesFormat: &network.ProbePropertiesFormat{
				Port: to.Int32Ptr(10080),
			},
		},
		{
			Name: to.StringPtr("atest1-" + string(service4.Spec.Ports[0].Protocol) +
				"-" + strconv.Itoa(int(service4.Spec.Ports[0].Port))),
			ProbePropertiesFormat: &network.ProbePropertiesFormat{
				Port: to.Int32Ptr(10081),
			},
		},
	}

	expectedSLb := getTestLoadBalancer(to.StringPtr("testCluster"), to.StringPtr("rg"), to.StringPtr("testCluster"), to.StringPtr("atest1"), service4, "Standard")
	(*expectedSLb.LoadBalancerPropertiesFormat.LoadBalancingRules)[0].DisableOutboundSnat = to.BoolPtr(true)
	(*expectedSLb.LoadBalancerPropertiesFormat.LoadBalancingRules)[0].EnableTCPReset = to.BoolPtr(false)
	expectedSLb.FrontendIPConfigurations = &[]network.FrontendIPConfiguration{
		{
			Name: to.StringPtr("btest1"),
			FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
				PublicIPAddress: &network.PublicIPAddress{ID: to.StringPtr("id1")},
			},
		},
		{
			Name: to.StringPtr("atest1"),
			FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
				PublicIPAddress: &network.PublicIPAddress{ID: to.StringPtr("/subscriptions/subscription/" +
					"resourceGroups/rg/providers/Microsoft.Network/publicIPAddresses/pipName")},
			},
		},
	}

	service5 := getTestService("test1", v1.ProtocolTCP, nil, 80)
	slb5 := getTestLoadBalancer(to.StringPtr("testCluster"), to.StringPtr("rg"), to.StringPtr("testCluster"), to.StringPtr("atest1"), service5, "Standard")
	slb5.FrontendIPConfigurations = &[]network.FrontendIPConfiguration{
		{
			Name: to.StringPtr("atest1"),
			FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
				PublicIPAddress: &network.PublicIPAddress{ID: to.StringPtr("id1")},
			},
		},
		{
			Name: to.StringPtr("btest1"),
			FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
				PublicIPAddress: &network.PublicIPAddress{ID: to.StringPtr("id1")},
			},
		},
	}
	slb5.Probes = &[]network.Probe{
		{
			Name: to.StringPtr("atest1-" + string(service4.Spec.Ports[0].Protocol) +
				"-" + strconv.Itoa(int(service4.Spec.Ports[0].Port))),
			ProbePropertiesFormat: &network.ProbePropertiesFormat{
				Port: to.Int32Ptr(10080),
			},
		},
		{
			Name: to.StringPtr("atest1-" + string(service4.Spec.Ports[0].Protocol) +
				"-" + strconv.Itoa(int(service4.Spec.Ports[0].Port))),
			ProbePropertiesFormat: &network.ProbePropertiesFormat{
				Port: to.Int32Ptr(10081),
			},
		},
	}

	//change to false to test that reconcilication will fix it
	(*slb5.LoadBalancerPropertiesFormat.LoadBalancingRules)[0].EnableTCPReset = to.BoolPtr(false)

	expectedSLb5 := getTestLoadBalancer(to.StringPtr("testCluster"), to.StringPtr("rg"), to.StringPtr("testCluster"), to.StringPtr("atest1"), service5, "Standard")
	(*expectedSLb5.LoadBalancerPropertiesFormat.LoadBalancingRules)[0].DisableOutboundSnat = to.BoolPtr(true)
	expectedSLb5.FrontendIPConfigurations = &[]network.FrontendIPConfiguration{
		{
			Name: to.StringPtr("btest1"),
			FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
				PublicIPAddress: &network.PublicIPAddress{ID: to.StringPtr("id1")},
			},
		},
		{
			Name: to.StringPtr("atest1"),
			FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
				PublicIPAddress: &network.PublicIPAddress{ID: to.StringPtr("/subscriptions/subscription/" +
					"resourceGroups/rg/providers/Microsoft.Network/publicIPAddresses/pipName")},
			},
		},
	}

	service6 := getTestService("test1", v1.ProtocolUDP, nil, 80)
	lb6 := getTestLoadBalancer(to.StringPtr("testCluster"), to.StringPtr("rg"), to.StringPtr("testCluster"), to.StringPtr("atest1"), service6, "basic")
	lb6.FrontendIPConfigurations = &[]network.FrontendIPConfiguration{}
	lb6.Probes = &[]network.Probe{}
	expectedLB6 := getTestLoadBalancer(to.StringPtr("testCluster"), to.StringPtr("rg"), to.StringPtr("testCluster"), to.StringPtr("atest1"), service6, "basic")
	expectedLB6.Probes = &[]network.Probe{}
	(*expectedLB6.LoadBalancerPropertiesFormat.LoadBalancingRules)[0].Probe = nil
	(*expectedLB6.LoadBalancerPropertiesFormat.LoadBalancingRules)[0].EnableTCPReset = nil
	(*expectedLB6.LoadBalancerPropertiesFormat.LoadBalancingRules)[0].DisableOutboundSnat = to.BoolPtr(false)
	expectedLB6.FrontendIPConfigurations = &[]network.FrontendIPConfiguration{
		{
			Name: to.StringPtr("atest1"),
			FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
				PublicIPAddress: &network.PublicIPAddress{ID: to.StringPtr("/subscriptions/subscription/" +
					"resourceGroups/rg/providers/Microsoft.Network/publicIPAddresses/pipName")},
			},
		},
	}

	service7 := getTestService("test1", v1.ProtocolUDP, nil, 80)
	service7.Spec.HealthCheckNodePort = 10081
	service7.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
	lb7 := getTestLoadBalancer(to.StringPtr("testCluster"), to.StringPtr("rg"), to.StringPtr("testCluster"), to.StringPtr("atest1"), service7, "basic")
	lb7.FrontendIPConfigurations = &[]network.FrontendIPConfiguration{}
	lb7.Probes = &[]network.Probe{}
	expectedLB7 := getTestLoadBalancer(to.StringPtr("testCluster"), to.StringPtr("rg"), to.StringPtr("testCluster"), to.StringPtr("atest1"), service7, "basic")
	(*expectedLB7.LoadBalancerPropertiesFormat.LoadBalancingRules)[0].Probe = &network.SubResource{
		ID: to.StringPtr("/subscriptions/subscription/resourceGroups/rg/providers/Microsoft.Network/loadBalancers/testCluster/probes/atest1-UDP-80"),
	}
	(*expectedLB7.LoadBalancerPropertiesFormat.LoadBalancingRules)[0].EnableTCPReset = nil
	(*expectedLB7.LoadBalancerPropertiesFormat.LoadBalancingRules)[0].DisableOutboundSnat = to.BoolPtr(false)
	expectedLB7.FrontendIPConfigurations = &[]network.FrontendIPConfiguration{
		{
			Name: to.StringPtr("atest1"),
			FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
				PublicIPAddress: &network.PublicIPAddress{ID: to.StringPtr("/subscriptions/subscription/" +
					"resourceGroups/rg/providers/Microsoft.Network/publicIPAddresses/pipName")},
			},
		},
	}
	expectedLB7.Probes = &[]network.Probe{
		{
			Name: to.StringPtr("atest1-" + string(service7.Spec.Ports[0].Protocol) +
				"-" + strconv.Itoa(int(service7.Spec.Ports[0].Port))),
			ProbePropertiesFormat: &network.ProbePropertiesFormat{
				Port:              to.Int32Ptr(10081),
				RequestPath:       to.StringPtr("/healthz"),
				Protocol:          network.ProbeProtocolHTTP,
				IntervalInSeconds: to.Int32Ptr(5),
				NumberOfProbes:    to.Int32Ptr(2),
			},
		},
	}

	service8 := getTestService("test1", v1.ProtocolTCP, nil, 80)
	lb8 := getTestLoadBalancer(to.StringPtr("testCluster"), to.StringPtr("anotherRG"), to.StringPtr("testCluster"), to.StringPtr("atest1"), service8, "Standard")
	lb8.FrontendIPConfigurations = &[]network.FrontendIPConfiguration{}
	lb8.Probes = &[]network.Probe{}
	expectedLB8 := getTestLoadBalancer(to.StringPtr("testCluster"), to.StringPtr("anotherRG"), to.StringPtr("testCluster"), to.StringPtr("atest1"), service8, "Standard")
	(*expectedLB8.LoadBalancerPropertiesFormat.LoadBalancingRules)[0].DisableOutboundSnat = to.BoolPtr(false)
	expectedLB8.FrontendIPConfigurations = &[]network.FrontendIPConfiguration{
		{
			Name: to.StringPtr("atest1"),
			FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
				PublicIPAddress: &network.PublicIPAddress{ID: to.StringPtr("/subscriptions/subscription/" +
					"resourceGroups/rg/providers/Microsoft.Network/publicIPAddresses/pipName")},
			},
		},
	}
	expectedLB8.Probes = &[]network.Probe{
		{
			Name: to.StringPtr("atest1-" + string(service8.Spec.Ports[0].Protocol) +
				"-" + strconv.Itoa(int(service7.Spec.Ports[0].Port))),
			ProbePropertiesFormat: &network.ProbePropertiesFormat{
				Port:              to.Int32Ptr(10080),
				Protocol:          network.ProbeProtocolTCP,
				IntervalInSeconds: to.Int32Ptr(5),
				NumberOfProbes:    to.Int32Ptr(2),
			},
		},
	}

	testCases := []struct {
		desc                      string
		service                   v1.Service
		loadBalancerSku           string
		preConfigLBType           string
		loadBalancerResourceGroup string
		disableOutboundSnat       *bool
		wantLb                    bool
		existingLB                network.LoadBalancer
		expectedLB                network.LoadBalancer
		expectedError             error
	}{
		{
			desc: "reconcileLoadBalancer shall return the lb deeply equal to the existingLB if there's no " +
				"modification needed when wantLb == true",
			loadBalancerSku: "basic",
			service:         service1,
			existingLB:      basicLb1,
			wantLb:          true,
			expectedLB:      basicLb1,
			expectedError:   nil,
		},
		{
			desc: "reconcileLoadBalancer shall return the lb deeply equal to the existingLB if there's no " +
				"modification needed when wantLb == false",
			loadBalancerSku: "basic",
			service:         service2,
			existingLB:      basicLb2,
			wantLb:          false,
			expectedLB:      basicLb2,
			expectedError:   nil,
		},
		{
			desc:            "reconcileLoadBalancer shall remove and reconstruct the correspoind field of lb",
			loadBalancerSku: "basic",
			service:         service3,
			existingLB:      modifiedLb1,
			wantLb:          true,
			expectedLB:      expectedLb1,
			expectedError:   nil,
		},
		{
			desc:            "reconcileLoadBalancer shall not raise an error",
			loadBalancerSku: "basic",
			service:         service3,
			existingLB:      modifiedLb1,
			preConfigLBType: "external",
			wantLb:          true,
			expectedLB:      expectedLb1,
			expectedError:   nil,
		},
		{
			desc:                "reconcileLoadBalancer shall remove and reconstruct the correspoind field of lb and set enableTcpReset to false in lbRule",
			loadBalancerSku:     "standard",
			service:             service4,
			disableOutboundSnat: to.BoolPtr(true),
			existingLB:          existingSLB,
			wantLb:              true,
			expectedLB:          expectedSLb,
			expectedError:       nil,
		},
		{
			desc:                "reconcileLoadBalancer shall remove and reconstruct the correspoind field of lb and set enableTcpReset to true in lbRule",
			loadBalancerSku:     "standard",
			service:             service5,
			disableOutboundSnat: to.BoolPtr(true),
			existingLB:          slb5,
			wantLb:              true,
			expectedLB:          expectedSLb5,
			expectedError:       nil,
		},
		{
			desc:            "reconcileLoadBalancer shall reconcile UDP services",
			loadBalancerSku: "basic",
			service:         service6,
			existingLB:      lb6,
			wantLb:          true,
			expectedLB:      expectedLB6,
			expectedError:   nil,
		},
		{
			desc:            "reconcileLoadBalancer shall reconcile probes for local traffic policy UDP services",
			loadBalancerSku: "basic",
			service:         service7,
			existingLB:      lb7,
			wantLb:          true,
			expectedLB:      expectedLB7,
			expectedError:   nil,
		},
		{
			desc:                      "reconcileLoadBalancer in other resource group",
			loadBalancerSku:           "standard",
			loadBalancerResourceGroup: "anotherRG",
			service:                   service8,
			existingLB:                lb8,
			wantLb:                    true,
			expectedLB:                expectedLB8,
			expectedError:             nil,
		},
	}

	for i, test := range testCases {
		az := getTestCloud()
		az.Config.LoadBalancerSku = test.loadBalancerSku
		az.DisableOutboundSNAT = test.disableOutboundSnat
		if test.preConfigLBType != "" {
			az.Config.PreConfiguredBackendPoolLoadBalancerTypes = test.preConfigLBType
		}
		az.LoadBalancerResourceGroup = test.loadBalancerResourceGroup

		clusterResources := getClusterResources(az, 3, 3)
		test.service.Spec.LoadBalancerIP = "1.2.3.4"

		err := az.PublicIPAddressesClient.CreateOrUpdate(context.TODO(), "rg", "pipName", network.PublicIPAddress{
			Name: to.StringPtr("pipName"),
			PublicIPAddressPropertiesFormat: &network.PublicIPAddressPropertiesFormat{
				IPAddress: to.StringPtr("1.2.3.4"),
			},
		})
		if err != nil {
			t.Fatalf("TestCase[%d] meets unexpected error: %v", i, err)
		}

		err = az.LoadBalancerClient.CreateOrUpdate(context.TODO(), az.getLoadBalancerResourceGroup(), "lb1", test.existingLB, "")
		if err != nil {
			t.Fatalf("TestCase[%d] meets unexpected error: %v", i, err)
		}

		lb, rerr := az.reconcileLoadBalancer("testCluster", &test.service, clusterResources.nodes, test.wantLb)
		assert.Equal(t, test.expectedError, rerr, "TestCase[%d]: %s", i, test.desc)

		if test.expectedError == nil {
			assert.Equal(t, &test.expectedLB, lb, "TestCase[%d]: %s", i, test.desc)
		}
	}
}

func TestGetServiceLoadBalancerStatus(t *testing.T) {
	az := getTestCloud()
	service := getTestService("test1", v1.ProtocolTCP, nil, 80)
	internalService := getInternalTestService("test1", 80)

	PIPClient := newFakeAzurePIPClient(az.Config.SubscriptionID)
	PIPClient.setFakeStore(map[string]map[string]network.PublicIPAddress{
		"rg": {"id1": network.PublicIPAddress{
			PublicIPAddressPropertiesFormat: &network.PublicIPAddressPropertiesFormat{
				IPAddress: to.StringPtr("1.2.3.4"),
			},
		}},
	})
	az.PublicIPAddressesClient = PIPClient

	lb1 := getTestLoadBalancer(to.StringPtr("lb1"), to.StringPtr("rg"), to.StringPtr("testCluster"),
		to.StringPtr("test1"), internalService, "Basic")
	lb1.FrontendIPConfigurations = nil
	lb2 := getTestLoadBalancer(to.StringPtr("lb2"), to.StringPtr("rg"), to.StringPtr("testCluster"),
		to.StringPtr("test1"), internalService, "Basic")
	lb2.FrontendIPConfigurations = &[]network.FrontendIPConfiguration{
		{
			Name: to.StringPtr("atest1"),
			FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
				PublicIPAddress:  &network.PublicIPAddress{ID: to.StringPtr("id1")},
				PrivateIPAddress: to.StringPtr("private"),
			},
		},
	}
	lb3 := getTestLoadBalancer(to.StringPtr("lb3"), to.StringPtr("rg"), to.StringPtr("testCluster"),
		to.StringPtr("test1"), internalService, "Basic")
	lb3.FrontendIPConfigurations = &[]network.FrontendIPConfiguration{
		{
			Name: to.StringPtr("btest1"),
			FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
				PublicIPAddress:  &network.PublicIPAddress{ID: to.StringPtr("id1")},
				PrivateIPAddress: to.StringPtr("private"),
			},
		},
	}
	lb4 := getTestLoadBalancer(to.StringPtr("lb4"), to.StringPtr("rg"), to.StringPtr("testCluster"),
		to.StringPtr("test1"), service, "Basic")
	lb4.FrontendIPConfigurations = &[]network.FrontendIPConfiguration{
		{
			Name: to.StringPtr("atest1"),
			FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
				PublicIPAddress:  &network.PublicIPAddress{ID: nil},
				PrivateIPAddress: to.StringPtr("private"),
			},
		},
	}
	lb5 := getTestLoadBalancer(to.StringPtr("lb5"), to.StringPtr("rg"), to.StringPtr("testCluster"),
		to.StringPtr("test1"), service, "Basic")
	lb5.FrontendIPConfigurations = &[]network.FrontendIPConfiguration{
		{
			Name: to.StringPtr("atest1"),
			FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
				PublicIPAddress:  nil,
				PrivateIPAddress: to.StringPtr("private"),
			},
		},
	}
	lb6 := getTestLoadBalancer(to.StringPtr("lb6"), to.StringPtr("rg"), to.StringPtr("testCluster"),
		to.StringPtr("test1"), service, "Basic")
	lb6.FrontendIPConfigurations = &[]network.FrontendIPConfiguration{
		{
			Name: to.StringPtr("atest1"),
			FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
				PublicIPAddress:  &network.PublicIPAddress{ID: to.StringPtr("illegal/id/")},
				PrivateIPAddress: to.StringPtr("private"),
			},
		},
	}

	testCases := []struct {
		desc           string
		service        *v1.Service
		lb             *network.LoadBalancer
		expectedStatus *v1.LoadBalancerStatus
		expectedError  bool
	}{
		{
			desc:    "getServiceLoadBalancer shall return nil if no lb is given",
			service: &service,
			lb:      nil,
		},
		{
			desc:    "getServiceLoadBalancerStatus shall return nil if given lb has no front ip config",
			service: &service,
			lb:      &lb1,
		},
		{
			desc:           "getServiceLoadBalancerStatus shall return private ip if service is internal",
			service:        &internalService,
			lb:             &lb2,
			expectedStatus: &v1.LoadBalancerStatus{Ingress: []v1.LoadBalancerIngress{{IP: "private"}}},
		},
		{
			desc: "getServiceLoadBalancerStatus shall return nil if lb.FrontendIPConfigurations.name != " +
				"az.getFrontendIPConfigName(service)",
			service: &internalService,
			lb:      &lb3,
		},
		{
			desc: "getServiceLoadBalancerStatus shall report error if the id of lb's " +
				"public ip address cannot be read",
			service:       &service,
			lb:            &lb4,
			expectedError: true,
		},
		{
			desc:          "getServiceLoadBalancerStatus shall report error if lb's public ip address cannot be read",
			service:       &service,
			lb:            &lb5,
			expectedError: true,
		},
		{
			desc:          "getServiceLoadBalancerStatus shall report error if id of lb's public ip address is illegal",
			service:       &service,
			lb:            &lb6,
			expectedError: true,
		},
		{
			desc: "getServiceLoadBalancerStatus shall return the corresponding " +
				"lb status if everything is good",
			service:        &service,
			lb:             &lb2,
			expectedStatus: &v1.LoadBalancerStatus{Ingress: []v1.LoadBalancerIngress{{IP: "1.2.3.4"}}},
		},
	}

	for i, test := range testCases {
		status, err := az.getServiceLoadBalancerStatus(test.service, test.lb)
		assert.Equal(t, test.expectedStatus, status, "TestCase[%d]: %s", i, test.desc)
		assert.Equal(t, test.expectedError, err != nil, "TestCase[%d]: %s", i, test.desc)
	}
}

func TestReconcileSecurityGroup(t *testing.T) {
	testCases := []struct {
		desc          string
		service       v1.Service
		lbIP          *string
		wantLb        bool
		existingSgs   map[string]network.SecurityGroup
		expectedSg    *network.SecurityGroup
		expectedError bool
	}{
		{
			desc: "reconcileSecurityGroup shall report error if the sg is shared and no ports in service",
			service: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						ServiceAnnotationSharedSecurityRule: "true",
					},
				},
			},
			expectedError: true,
		},
		{
			desc:          "reconcileSecurityGroup shall report error if no such sg can be found",
			service:       getTestService("test1", v1.ProtocolTCP, nil, 80),
			expectedError: true,
		},
		{
			desc:          "reconcileSecurityGroup shall report error if wantLb is true and lbIP is nil",
			service:       getTestService("test1", v1.ProtocolTCP, nil, 80),
			wantLb:        true,
			existingSgs:   map[string]network.SecurityGroup{"nsg": {}},
			expectedError: true,
		},
		{
			desc:        "reconcileSecurityGroup shall remain the existingSgs intact if nothing needs to be modified",
			service:     getTestService("test1", v1.ProtocolTCP, nil, 80),
			existingSgs: map[string]network.SecurityGroup{"nsg": {}},
			expectedSg:  &network.SecurityGroup{},
		},
		{
			desc:    "reconcileSecurityGroup shall delete unwanted sgs and create needed ones",
			service: getTestService("test1", v1.ProtocolTCP, nil, 80),
			existingSgs: map[string]network.SecurityGroup{"nsg": {
				Name: to.StringPtr("nsg"),
				SecurityGroupPropertiesFormat: &network.SecurityGroupPropertiesFormat{
					SecurityRules: &[]network.SecurityRule{
						{
							Name: to.StringPtr("atest1-toBeDeleted"),
							SecurityRulePropertiesFormat: &network.SecurityRulePropertiesFormat{
								SourceAddressPrefix:      to.StringPtr("prefix"),
								SourcePortRange:          to.StringPtr("range"),
								DestinationAddressPrefix: to.StringPtr("desPrefix"),
								DestinationPortRange:     to.StringPtr("desRange"),
							},
						},
					},
				},
			}},
			lbIP:   to.StringPtr("1.1.1.1"),
			wantLb: true,
			expectedSg: &network.SecurityGroup{
				Name: to.StringPtr("nsg"),
				SecurityGroupPropertiesFormat: &network.SecurityGroupPropertiesFormat{
					SecurityRules: &[]network.SecurityRule{
						{
							Name: to.StringPtr("atest1-TCP-80-Internet"),
							SecurityRulePropertiesFormat: &network.SecurityRulePropertiesFormat{
								Protocol:                 network.SecurityRuleProtocol("Tcp"),
								SourcePortRange:          to.StringPtr("*"),
								DestinationPortRange:     to.StringPtr("80"),
								SourceAddressPrefix:      to.StringPtr("Internet"),
								DestinationAddressPrefix: to.StringPtr("1.1.1.1"),
								Access:                   network.SecurityRuleAccess("Allow"),
								Priority:                 to.Int32Ptr(500),
								Direction:                network.SecurityRuleDirection("Inbound"),
							},
						},
					},
				},
			},
		},
	}

	for i, test := range testCases {
		az := getTestCloud()
		for name, sg := range test.existingSgs {
			err := az.SecurityGroupsClient.CreateOrUpdate(context.TODO(), "rg", name, sg, "")
			if err != nil {
				t.Fatalf("TestCase[%d] meets unexpected error: %v", i, err)
			}
		}
		sg, err := az.reconcileSecurityGroup("testCluster", &test.service, test.lbIP, test.wantLb)
		assert.Equal(t, test.expectedSg, sg, "TestCase[%d]: %s", i, test.desc)
		assert.Equal(t, test.expectedError, err != nil, "TestCase[%d]: %s", i, test.desc)
	}
}

func TestSafeDeletePublicIP(t *testing.T) {
	testCases := []struct {
		desc          string
		pip           *network.PublicIPAddress
		lb            *network.LoadBalancer
		expectedError bool
	}{
		{
			desc: "safeDeletePublicIP shall delete corresponding ip configurations and lb rules",
			pip: &network.PublicIPAddress{
				Name: to.StringPtr("pip1"),
				PublicIPAddressPropertiesFormat: &network.PublicIPAddressPropertiesFormat{
					IPConfiguration: &network.IPConfiguration{
						ID: to.StringPtr("id1"),
					},
				},
			},
			lb: &network.LoadBalancer{
				Name: to.StringPtr("lb1"),
				LoadBalancerPropertiesFormat: &network.LoadBalancerPropertiesFormat{
					FrontendIPConfigurations: &[]network.FrontendIPConfiguration{
						{
							ID: to.StringPtr("id1"),
							FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
								LoadBalancingRules: &[]network.SubResource{{ID: to.StringPtr("rules1")}},
							},
						},
					},
					LoadBalancingRules: &[]network.LoadBalancingRule{{ID: to.StringPtr("rules1")}},
				},
			},
		},
	}

	for i, test := range testCases {
		az := getTestCloud()
		err := az.PublicIPAddressesClient.CreateOrUpdate(context.TODO(), "rg", "pip1", network.PublicIPAddress{
			Name: to.StringPtr("pip1"),
			PublicIPAddressPropertiesFormat: &network.PublicIPAddressPropertiesFormat{
				IPConfiguration: &network.IPConfiguration{
					ID: to.StringPtr("id1"),
				},
			},
		})
		if err != nil {
			t.Fatalf("TestCase[%d] meets unexpected error: %v", i, err)
		}
		service := getTestService("test1", v1.ProtocolTCP, nil, 80)
		rerr := az.safeDeletePublicIP(&service, "rg", test.pip, test.lb)
		assert.Equal(t, 0, len(*test.lb.FrontendIPConfigurations), "TestCase[%d]: %s", i, test.desc)
		assert.Equal(t, 0, len(*test.lb.LoadBalancingRules), "TestCase[%d]: %s", i, test.desc)
		assert.Equal(t, test.expectedError, rerr != nil, "TestCase[%d]: %s", i, test.desc)
	}
}

func TestReconcilePublicIP(t *testing.T) {
	testCases := []struct {
		desc          string
		wantLb        bool
		annotations   map[string]string
		existingPIPs  []network.PublicIPAddress
		expectedID    string
		expectedPIP   *network.PublicIPAddress
		expectedError bool
	}{
		{
			desc:   "reconcilePublicIP shall return nil if there's no pip in service",
			wantLb: false,
		},
		{
			desc:   "reconcilePublicIP shall return nil if no pip is owned by service",
			wantLb: false,
			existingPIPs: []network.PublicIPAddress{
				{
					Name: to.StringPtr("pip1"),
				},
			},
		},
		{
			desc:   "reconcilePublicIP shall delete unwanted pips and create a new one",
			wantLb: true,
			existingPIPs: []network.PublicIPAddress{
				{
					Name: to.StringPtr("pip1"),
					Tags: map[string]*string{"service": to.StringPtr("default/test1")},
				},
			},
			expectedID: "/subscriptions/subscription/resourceGroups/rg/providers/" +
				"Microsoft.Network/publicIPAddresses/testCluster-atest1",
		},
		{
			desc:        "reconcilePublicIP shall report error if the given PIP name doesn't exist in the resource group",
			wantLb:      true,
			annotations: map[string]string{ServiceAnnotationPIPName: "testPIP"},
			existingPIPs: []network.PublicIPAddress{
				{
					Name: to.StringPtr("pip1"),
					Tags: map[string]*string{"service": to.StringPtr("default/test1")},
				},
				{
					Name: to.StringPtr("pip2"),
					Tags: map[string]*string{"service": to.StringPtr("default/test1")},
				},
			},
			expectedError: true,
		},
		{
			desc:        "reconcilePublicIP shall delete unwanted PIP when given the name of desired PIP",
			wantLb:      true,
			annotations: map[string]string{ServiceAnnotationPIPName: "testPIP"},
			existingPIPs: []network.PublicIPAddress{
				{
					Name: to.StringPtr("pip1"),
					Tags: map[string]*string{"service": to.StringPtr("default/test1")},
				},
				{
					Name: to.StringPtr("pip2"),
					Tags: map[string]*string{"service": to.StringPtr("default/test1")},
				},
				{
					Name: to.StringPtr("testPIP"),
					Tags: map[string]*string{"service": to.StringPtr("default/test1")},
				},
			},
			expectedPIP: &network.PublicIPAddress{
				ID:   to.StringPtr("/subscriptions/subscription/resourceGroups/rg/providers/Microsoft.Network/publicIPAddresses/testPIP"),
				Name: to.StringPtr("testPIP"),
				Tags: map[string]*string{"service": to.StringPtr("default/test1")},
			},
		},
		{
			desc:        "reconcilePublicIP shall find the PIP by given name and shall not delete the PIP which is not owned by service",
			wantLb:      true,
			annotations: map[string]string{ServiceAnnotationPIPName: "testPIP"},
			existingPIPs: []network.PublicIPAddress{
				{
					Name: to.StringPtr("pip1"),
				},
				{
					Name: to.StringPtr("pip2"),
					Tags: map[string]*string{"service": to.StringPtr("default/test1")},
				},
				{
					Name: to.StringPtr("testPIP"),
				},
			},
			expectedPIP: &network.PublicIPAddress{
				ID:   to.StringPtr("/subscriptions/subscription/resourceGroups/rg/providers/Microsoft.Network/publicIPAddresses/testPIP"),
				Name: to.StringPtr("testPIP"),
			},
		},
	}

	for i, test := range testCases {
		az := getTestCloud()
		service := getTestService("test1", v1.ProtocolTCP, nil, 80)
		service.Annotations = test.annotations
		for _, pip := range test.existingPIPs {
			err := az.PublicIPAddressesClient.CreateOrUpdate(context.TODO(), "rg", to.String(pip.Name), pip)
			if err != nil {
				t.Fatalf("TestCase[%d] meets unexpected error: %v", i, err)
			}
		}
		pip, err := az.reconcilePublicIP("testCluster", &service, "", test.wantLb)
		if test.expectedID != "" {
			assert.Equal(t, test.expectedID, to.String(pip.ID), "TestCase[%d]: %s", i, test.desc)
		} else if test.expectedPIP != nil && test.expectedPIP.Name != nil {
			assert.Equal(t, *test.expectedPIP.Name, *pip.Name, "TestCase[%d]: %s", i, test.desc)
		}
		assert.Equal(t, test.expectedError, err != nil, "TestCase[%d]: %s", i, test.desc)
	}
}

func TestEnsurePublicIPExists(t *testing.T) {
	testCases := []struct {
		desc                    string
		existingPIPs            []network.PublicIPAddress
		inputDNSLabel           string
		foundDNSLabelAnnotation bool
		expectedPIP             *network.PublicIPAddress
		expectedID              string
		expectedError           bool
	}{
		{
			desc:         "ensurePublicIPExists shall return existed PIP if there is any",
			existingPIPs: []network.PublicIPAddress{{Name: to.StringPtr("pip1")}},
			expectedPIP: &network.PublicIPAddress{
				Name: to.StringPtr("pip1"),
				ID: to.StringPtr("/subscriptions/subscription/resourceGroups/rg" +
					"/providers/Microsoft.Network/publicIPAddresses/pip1"),
			},
		},
		{
			desc: "ensurePublicIPExists shall create a new pip if there is no existed pip",
			expectedID: "/subscriptions/subscription/resourceGroups/rg/providers/" +
				"Microsoft.Network/publicIPAddresses/pip1",
		},
		{
			desc:                    "ensurePublicIPExists shall update existed PIP's dns label",
			inputDNSLabel:           "newdns",
			foundDNSLabelAnnotation: true,
			existingPIPs: []network.PublicIPAddress{{
				Name: to.StringPtr("pip1"),
				PublicIPAddressPropertiesFormat: &network.PublicIPAddressPropertiesFormat{
					DNSSettings: &network.PublicIPAddressDNSSettings{
						DomainNameLabel: to.StringPtr("previousdns"),
					},
				},
			}},
			expectedPIP: &network.PublicIPAddress{
				Name: to.StringPtr("pip1"),
				ID: to.StringPtr("/subscriptions/subscription/resourceGroups/rg" +
					"/providers/Microsoft.Network/publicIPAddresses/pip1"),
				PublicIPAddressPropertiesFormat: &network.PublicIPAddressPropertiesFormat{
					DNSSettings: &network.PublicIPAddressDNSSettings{
						DomainNameLabel: to.StringPtr("newdns"),
					},
				},
			},
		},
		{
			desc:                    "ensurePublicIPExists shall delete DNS from PIP if DNS label is set empty",
			foundDNSLabelAnnotation: true,
			existingPIPs: []network.PublicIPAddress{{
				Name: to.StringPtr("pip1"),
				PublicIPAddressPropertiesFormat: &network.PublicIPAddressPropertiesFormat{
					DNSSettings: &network.PublicIPAddressDNSSettings{
						DomainNameLabel: to.StringPtr("previousdns"),
					},
				},
			}},
			expectedPIP: &network.PublicIPAddress{
				Name: to.StringPtr("pip1"),
				ID: to.StringPtr("/subscriptions/subscription/resourceGroups/rg" +
					"/providers/Microsoft.Network/publicIPAddresses/pip1"),
				PublicIPAddressPropertiesFormat: &network.PublicIPAddressPropertiesFormat{
					DNSSettings: nil,
				},
			},
		},
		{
			desc:                    "ensurePublicIPExists shall not delete DNS from PIP if DNS label annotation is not set",
			foundDNSLabelAnnotation: false,
			existingPIPs: []network.PublicIPAddress{{
				Name: to.StringPtr("pip1"),
				PublicIPAddressPropertiesFormat: &network.PublicIPAddressPropertiesFormat{
					DNSSettings: &network.PublicIPAddressDNSSettings{
						DomainNameLabel: to.StringPtr("previousdns"),
					},
				},
			}},
			expectedPIP: &network.PublicIPAddress{
				Name: to.StringPtr("pip1"),
				ID: to.StringPtr("/subscriptions/subscription/resourceGroups/rg" +
					"/providers/Microsoft.Network/publicIPAddresses/pip1"),
				PublicIPAddressPropertiesFormat: &network.PublicIPAddressPropertiesFormat{
					DNSSettings: &network.PublicIPAddressDNSSettings{
						DomainNameLabel: to.StringPtr("previousdns"),
					},
				},
			},
		},
	}

	for i, test := range testCases {
		az := getTestCloud()
		service := getTestService("test1", v1.ProtocolTCP, nil, 80)
		for _, pip := range test.existingPIPs {
			err := az.PublicIPAddressesClient.CreateOrUpdate(context.TODO(), "rg", to.String(pip.Name), pip)
			if err != nil {
				t.Fatalf("TestCase[%d] meets unexpected error: %v", i, err)
			}
		}
		pip, err := az.ensurePublicIPExists(&service, "pip1", test.inputDNSLabel, "", false, test.foundDNSLabelAnnotation)
		if test.expectedID != "" {
			assert.Equal(t, test.expectedID, to.String(pip.ID), "TestCase[%d]: %s", i, test.desc)
		} else {
			assert.Equal(t, test.expectedPIP, pip, "TestCase[%d]: %s", i, test.desc)
		}
		assert.Equal(t, test.expectedError, err != nil, "TestCase[%d]: %s", i, test.desc)
	}
}

func TestShouldUpdateLoadBalancer(t *testing.T) {
	testCases := []struct {
		desc                   string
		lbHasDeletionTimestamp bool
		existsLb               bool
		expectedOutput         bool
	}{
		{
			desc:                   "should update a load balancer that does not have a deletion timestamp and exists in Azure",
			lbHasDeletionTimestamp: false,
			existsLb:               true,
			expectedOutput:         true,
		},
		{
			desc:                   "should not update a load balancer that is being deleted / already deleted in K8s",
			lbHasDeletionTimestamp: true,
			existsLb:               true,
			expectedOutput:         false,
		},
		{
			desc:                   "should not update a load balancer that does not exist in Azure",
			lbHasDeletionTimestamp: false,
			existsLb:               false,
			expectedOutput:         false,
		},
		{
			desc:                   "should not update a load balancer that has a deletion timestamp and does not exist in Azure",
			lbHasDeletionTimestamp: true,
			existsLb:               false,
			expectedOutput:         false,
		},
	}

	for i, test := range testCases {
		az := getTestCloud()
		service := getTestService("test1", v1.ProtocolTCP, nil, 80)
		if test.lbHasDeletionTimestamp {
			service.ObjectMeta.DeletionTimestamp = &metav1.Time{Time: time.Now()}
		}
		if test.existsLb {
			lb := network.LoadBalancer{
				Name: to.StringPtr("lb1"),
				LoadBalancerPropertiesFormat: &network.LoadBalancerPropertiesFormat{
					FrontendIPConfigurations: &[]network.FrontendIPConfiguration{
						{
							Name: to.StringPtr("atest1"),
							FrontendIPConfigurationPropertiesFormat: &network.FrontendIPConfigurationPropertiesFormat{
								PublicIPAddress: &network.PublicIPAddress{ID: to.StringPtr("id1")},
							},
						},
					},
				},
			}
			err := az.LoadBalancerClient.CreateOrUpdate(context.TODO(), "rg", *lb.Name, lb, "")
			if err != nil {
				t.Fatalf("TestCase[%d] meets unexpected error: %v", i, err)
			}
		}
		shouldUpdateLoadBalancer := az.shouldUpdateLoadBalancer(testClusterName, &service)
		assert.Equal(t, test.expectedOutput, shouldUpdateLoadBalancer, "TestCase[%d]: %s", i, test.desc)
	}
}

func TestIsBackendPoolPreConfigured(t *testing.T) {
	testCases := []struct {
		desc                                      string
		preConfiguredBackendPoolLoadBalancerTypes string
		isInternalService                         bool
		expectedOutput                            bool
	}{
		{
			desc: "should return true when preConfiguredBackendPoolLoadBalancerTypes is both for any case",
			preConfiguredBackendPoolLoadBalancerTypes: "all",
			isInternalService:                         true,
			expectedOutput:                            true,
		},
		{
			desc: "should return true when preConfiguredBackendPoolLoadBalancerTypes is both for any case",
			preConfiguredBackendPoolLoadBalancerTypes: "all",
			isInternalService:                         false,
			expectedOutput:                            true,
		},
		{
			desc: "should return true when preConfiguredBackendPoolLoadBalancerTypes is external when creating external lb",
			preConfiguredBackendPoolLoadBalancerTypes: "external",
			isInternalService:                         false,
			expectedOutput:                            true,
		},
		{
			desc: "should return false when preConfiguredBackendPoolLoadBalancerTypes is external when creating internal lb",
			preConfiguredBackendPoolLoadBalancerTypes: "external",
			isInternalService:                         true,
			expectedOutput:                            false,
		},
		{
			desc: "should return false when preConfiguredBackendPoolLoadBalancerTypes is internal when creating external lb",
			preConfiguredBackendPoolLoadBalancerTypes: "internal",
			isInternalService:                         false,
			expectedOutput:                            false,
		},
		{
			desc: "should return true when preConfiguredBackendPoolLoadBalancerTypes is internal when creating internal lb",
			preConfiguredBackendPoolLoadBalancerTypes: "internal",
			isInternalService:                         true,
			expectedOutput:                            true,
		},
		{
			desc: "should return false when preConfiguredBackendPoolLoadBalancerTypes is empty for any case",
			preConfiguredBackendPoolLoadBalancerTypes: "",
			isInternalService:                         true,
			expectedOutput:                            false,
		},
		{
			desc: "should return false when preConfiguredBackendPoolLoadBalancerTypes is empty for any case",
			preConfiguredBackendPoolLoadBalancerTypes: "",
			isInternalService:                         false,
			expectedOutput:                            false,
		},
	}

	for i, test := range testCases {
		az := getTestCloud()
		az.Config.PreConfiguredBackendPoolLoadBalancerTypes = test.preConfiguredBackendPoolLoadBalancerTypes
		var service v1.Service
		if test.isInternalService {
			service = getInternalTestService("test", 80)
		} else {
			service = getTestService("test", v1.ProtocolTCP, nil, 80)
		}

		isPreConfigured := az.isBackendPoolPreConfigured(&service)
		assert.Equal(t, test.expectedOutput, isPreConfigured, "TestCase[%d]: %s", i, test.desc)
	}
}
