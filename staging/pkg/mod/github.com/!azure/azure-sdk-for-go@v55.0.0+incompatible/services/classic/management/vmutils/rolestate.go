// +build go1.7

package vmutils

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

import (
	"time"

	"github.com/Azure/azure-sdk-for-go/services/classic/management"
	vm "github.com/Azure/azure-sdk-for-go/services/classic/management/virtualmachine"
)

// WaitForDeploymentPowerState blocks until all role instances in deployment
// reach desired power state.
func WaitForDeploymentPowerState(client management.Client, cloudServiceName, deploymentName string, desiredPowerstate vm.PowerState) error {
	for {
		deployment, err := vm.NewClient(client).GetDeployment(cloudServiceName, deploymentName)
		if err != nil {
			return err
		}
		if allInstancesInPowerState(deployment.RoleInstanceList, desiredPowerstate) {
			return nil
		}
		time.Sleep(2 * time.Second)
	}
}

func allInstancesInPowerState(instances []vm.RoleInstance, desiredPowerstate vm.PowerState) bool {
	for _, r := range instances {
		if r.PowerState != desiredPowerstate {
			return false
		}
	}

	return true
}

// WaitForDeploymentInstanceStatus blocks until all role instances in deployment
// reach desired InstanceStatus.
func WaitForDeploymentInstanceStatus(client management.Client, cloudServiceName, deploymentName string, desiredInstanceStatus vm.InstanceStatus) error {
	for {
		deployment, err := vm.NewClient(client).GetDeployment(cloudServiceName, deploymentName)
		if err != nil {
			return err
		}
		if allInstancesInInstanceStatus(deployment.RoleInstanceList, desiredInstanceStatus) {
			return nil
		}
		time.Sleep(2 * time.Second)
	}
}

func allInstancesInInstanceStatus(instances []vm.RoleInstance, desiredInstancestatus vm.InstanceStatus) bool {
	for _, r := range instances {
		if r.InstanceStatus != desiredInstancestatus {
			return false
		}
	}

	return true
}
