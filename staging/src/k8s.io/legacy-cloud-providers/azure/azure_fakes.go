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
	"fmt"

	"github.com/golang/mock/gomock"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/record"
	"k8s.io/legacy-cloud-providers/azure/auth"
	"k8s.io/legacy-cloud-providers/azure/clients/diskclient/mockdiskclient"
	"k8s.io/legacy-cloud-providers/azure/clients/interfaceclient/mockinterfaceclient"
	"k8s.io/legacy-cloud-providers/azure/clients/loadbalancerclient/mockloadbalancerclient"
	"k8s.io/legacy-cloud-providers/azure/clients/publicipclient/mockpublicipclient"
	"k8s.io/legacy-cloud-providers/azure/clients/routeclient/mockrouteclient"
	"k8s.io/legacy-cloud-providers/azure/clients/routetableclient/mockroutetableclient"
	"k8s.io/legacy-cloud-providers/azure/clients/securitygroupclient/mocksecuritygroupclient"
	"k8s.io/legacy-cloud-providers/azure/clients/subnetclient/mocksubnetclient"
	"k8s.io/legacy-cloud-providers/azure/clients/vmclient/mockvmclient"
	"k8s.io/legacy-cloud-providers/azure/clients/vmssclient/mockvmssclient"
	"k8s.io/legacy-cloud-providers/azure/clients/vmssvmclient/mockvmssvmclient"
)

var (
	errPreconditionFailedEtagMismatch = fmt.Errorf("PreconditionFailedEtagMismatch")
)

// GetTestCloud returns a fake azure cloud for unit tests in Azure related CSI drivers
func GetTestCloud(ctrl *gomock.Controller) (az *Cloud) {
	az = &Cloud{
		Config: Config{
			AzureAuthConfig: auth.AzureAuthConfig{
				TenantID:       "tenant",
				SubscriptionID: "subscription",
			},
			ResourceGroup:                "rg",
			VnetResourceGroup:            "rg",
			RouteTableResourceGroup:      "rg",
			SecurityGroupResourceGroup:   "rg",
			Location:                     "westus",
			VnetName:                     "vnet",
			SubnetName:                   "subnet",
			SecurityGroupName:            "nsg",
			RouteTableName:               "rt",
			PrimaryAvailabilitySetName:   "as",
			PrimaryScaleSetName:          "vmss",
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
	az.DisksClient = mockdiskclient.NewMockInterface(ctrl)
	az.InterfacesClient = mockinterfaceclient.NewMockInterface(ctrl)
	az.LoadBalancerClient = mockloadbalancerclient.NewMockInterface(ctrl)
	az.PublicIPAddressesClient = mockpublicipclient.NewMockInterface(ctrl)
	az.RoutesClient = mockrouteclient.NewMockInterface(ctrl)
	az.RouteTablesClient = mockroutetableclient.NewMockInterface(ctrl)
	az.SecurityGroupsClient = mocksecuritygroupclient.NewMockInterface(ctrl)
	az.SubnetsClient = mocksubnetclient.NewMockInterface(ctrl)
	az.VirtualMachineScaleSetsClient = mockvmssclient.NewMockInterface(ctrl)
	az.VirtualMachineScaleSetVMsClient = mockvmssvmclient.NewMockInterface(ctrl)
	az.VirtualMachinesClient = mockvmclient.NewMockInterface(ctrl)
	az.vmSet = newAvailabilitySet(az)
	az.vmCache, _ = az.newVMCache()
	az.lbCache, _ = az.newLBCache()
	az.nsgCache, _ = az.newNSGCache()
	az.rtCache, _ = az.newRouteTableCache()

	common := &controllerCommon{cloud: az, resourceGroup: "rg", location: "westus"}
	az.controllerCommon = common
	az.ManagedDiskController = &ManagedDiskController{common: common}

	return az
}
