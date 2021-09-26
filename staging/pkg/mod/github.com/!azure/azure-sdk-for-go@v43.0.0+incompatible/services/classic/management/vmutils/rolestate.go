// +build go1.7

package vmutils

// Copyright 2017 Microsoft Corporation
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

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
