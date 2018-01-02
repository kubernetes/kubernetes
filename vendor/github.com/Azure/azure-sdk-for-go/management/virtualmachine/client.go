// +build go1.7

// Package virtualmachine provides a client for Virtual Machines.
package virtualmachine

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
	"encoding/xml"
	"fmt"

	"github.com/Azure/azure-sdk-for-go/management"
)

const (
	azureDeploymentListURL        = "services/hostedservices/%s/deployments"
	azureDeploymentURL            = "services/hostedservices/%s/deployments/%s"
	azureListDeploymentsInSlotURL = "services/hostedservices/%s/deploymentslots/Production"
	deleteAzureDeploymentURL      = "services/hostedservices/%s/deployments/%s?comp=media"
	azureAddRoleURL               = "services/hostedservices/%s/deployments/%s/roles"
	azureRoleURL                  = "services/hostedservices/%s/deployments/%s/roles/%s"
	azureOperationsURL            = "services/hostedservices/%s/deployments/%s/roleinstances/%s/Operations"
	azureRoleSizeListURL          = "rolesizes"

	errParamNotSpecified = "Parameter %s is not specified."
)

//NewClient is used to instantiate a new VirtualMachineClient from an Azure client
func NewClient(client management.Client) VirtualMachineClient {
	return VirtualMachineClient{client: client}
}

// CreateDeploymentOptions can be used to create a customized deployement request
type CreateDeploymentOptions struct {
	DNSServers         []DNSServer
	LoadBalancers      []LoadBalancer
	ReservedIPName     string
	VirtualNetworkName string
}

// CreateDeployment creates a deployment and then creates a virtual machine
// in the deployment based on the specified configuration.
//
// https://msdn.microsoft.com/en-us/library/azure/jj157194.aspx
func (vm VirtualMachineClient) CreateDeployment(
	role Role,
	cloudServiceName string,
	options CreateDeploymentOptions) (management.OperationID, error) {

	req := DeploymentRequest{
		Name:               role.RoleName,
		DeploymentSlot:     "Production",
		Label:              role.RoleName,
		RoleList:           []Role{role},
		DNSServers:         options.DNSServers,
		LoadBalancers:      options.LoadBalancers,
		ReservedIPName:     options.ReservedIPName,
		VirtualNetworkName: options.VirtualNetworkName,
	}

	data, err := xml.Marshal(req)
	if err != nil {
		return "", err
	}

	requestURL := fmt.Sprintf(azureDeploymentListURL, cloudServiceName)
	return vm.client.SendAzurePostRequest(requestURL, data)
}

// GetDeploymentName queries an existing Azure cloud service for the name of the Deployment,
// if any, in its 'Production' slot (the only slot possible). If none exists, it returns empty
// string but no error
//
//https://msdn.microsoft.com/en-us/library/azure/ee460804.aspx
func (vm VirtualMachineClient) GetDeploymentName(cloudServiceName string) (string, error) {
	var deployment DeploymentResponse
	if cloudServiceName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "cloudServiceName")
	}
	requestURL := fmt.Sprintf(azureListDeploymentsInSlotURL, cloudServiceName)
	response, err := vm.client.SendAzureGetRequest(requestURL)
	if err != nil {
		if management.IsResourceNotFoundError(err) {
			return "", nil
		}
		return "", err
	}
	err = xml.Unmarshal(response, &deployment)
	if err != nil {
		return "", err
	}

	return deployment.Name, nil
}

func (vm VirtualMachineClient) GetDeployment(cloudServiceName, deploymentName string) (DeploymentResponse, error) {
	var deployment DeploymentResponse
	if cloudServiceName == "" {
		return deployment, fmt.Errorf(errParamNotSpecified, "cloudServiceName")
	}
	if deploymentName == "" {
		return deployment, fmt.Errorf(errParamNotSpecified, "deploymentName")
	}
	requestURL := fmt.Sprintf(azureDeploymentURL, cloudServiceName, deploymentName)
	response, azureErr := vm.client.SendAzureGetRequest(requestURL)
	if azureErr != nil {
		return deployment, azureErr
	}

	err := xml.Unmarshal(response, &deployment)
	return deployment, err
}

func (vm VirtualMachineClient) DeleteDeployment(cloudServiceName, deploymentName string) (management.OperationID, error) {
	if cloudServiceName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "cloudServiceName")
	}
	if deploymentName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "deploymentName")
	}

	requestURL := fmt.Sprintf(deleteAzureDeploymentURL, cloudServiceName, deploymentName)
	return vm.client.SendAzureDeleteRequest(requestURL)
}

func (vm VirtualMachineClient) GetRole(cloudServiceName, deploymentName, roleName string) (*Role, error) {
	if cloudServiceName == "" {
		return nil, fmt.Errorf(errParamNotSpecified, "cloudServiceName")
	}
	if deploymentName == "" {
		return nil, fmt.Errorf(errParamNotSpecified, "deploymentName")
	}
	if roleName == "" {
		return nil, fmt.Errorf(errParamNotSpecified, "roleName")
	}

	role := new(Role)

	requestURL := fmt.Sprintf(azureRoleURL, cloudServiceName, deploymentName, roleName)
	response, azureErr := vm.client.SendAzureGetRequest(requestURL)
	if azureErr != nil {
		return nil, azureErr
	}

	err := xml.Unmarshal(response, role)
	if err != nil {
		return nil, err
	}

	return role, nil
}

// AddRole adds a Virtual Machine to a deployment of Virtual Machines, where role name = VM name
// See https://msdn.microsoft.com/en-us/library/azure/jj157186.aspx
func (vm VirtualMachineClient) AddRole(cloudServiceName string, deploymentName string, role Role) (management.OperationID, error) {
	if cloudServiceName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "cloudServiceName")
	}
	if deploymentName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "deploymentName")
	}

	data, err := xml.Marshal(PersistentVMRole{Role: role})
	if err != nil {
		return "", err
	}

	requestURL := fmt.Sprintf(azureAddRoleURL, cloudServiceName, deploymentName)
	return vm.client.SendAzurePostRequest(requestURL, data)
}

// UpdateRole updates the configuration of the specified virtual machine
// See https://msdn.microsoft.com/en-us/library/azure/jj157187.aspx
func (vm VirtualMachineClient) UpdateRole(cloudServiceName, deploymentName, roleName string, role Role) (management.OperationID, error) {
	if cloudServiceName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "cloudServiceName")
	}
	if deploymentName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "deploymentName")
	}
	if roleName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "roleName")
	}

	data, err := xml.Marshal(PersistentVMRole{Role: role})
	if err != nil {
		return "", err
	}

	requestURL := fmt.Sprintf(azureRoleURL, cloudServiceName, deploymentName, roleName)
	return vm.client.SendAzurePutRequest(requestURL, "text/xml", data)
}

func (vm VirtualMachineClient) StartRole(cloudServiceName, deploymentName, roleName string) (management.OperationID, error) {
	if cloudServiceName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "cloudServiceName")
	}
	if deploymentName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "deploymentName")
	}
	if roleName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "roleName")
	}

	startRoleOperationBytes, err := xml.Marshal(StartRoleOperation{
		OperationType: "StartRoleOperation",
	})
	if err != nil {
		return "", err
	}

	requestURL := fmt.Sprintf(azureOperationsURL, cloudServiceName, deploymentName, roleName)
	return vm.client.SendAzurePostRequest(requestURL, startRoleOperationBytes)
}

func (vm VirtualMachineClient) ShutdownRole(cloudServiceName, deploymentName, roleName string, postaction PostShutdownAction) (management.OperationID, error) {
	if cloudServiceName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "cloudServiceName")
	}
	if deploymentName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "deploymentName")
	}
	if roleName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "roleName")
	}

	shutdownRoleOperationBytes, err := xml.Marshal(ShutdownRoleOperation{
		OperationType:      "ShutdownRoleOperation",
		PostShutdownAction: postaction,
	})
	if err != nil {
		return "", err
	}

	requestURL := fmt.Sprintf(azureOperationsURL, cloudServiceName, deploymentName, roleName)
	return vm.client.SendAzurePostRequest(requestURL, shutdownRoleOperationBytes)
}

func (vm VirtualMachineClient) RestartRole(cloudServiceName, deploymentName, roleName string) (management.OperationID, error) {
	if cloudServiceName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "cloudServiceName")
	}
	if deploymentName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "deploymentName")
	}
	if roleName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "roleName")
	}

	restartRoleOperationBytes, err := xml.Marshal(RestartRoleOperation{
		OperationType: "RestartRoleOperation",
	})
	if err != nil {
		return "", err
	}

	requestURL := fmt.Sprintf(azureOperationsURL, cloudServiceName, deploymentName, roleName)
	return vm.client.SendAzurePostRequest(requestURL, restartRoleOperationBytes)
}

func (vm VirtualMachineClient) DeleteRole(cloudServiceName, deploymentName, roleName string, deleteVHD bool) (management.OperationID, error) {
	if cloudServiceName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "cloudServiceName")
	}
	if deploymentName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "deploymentName")
	}
	if roleName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "roleName")
	}

	requestURL := fmt.Sprintf(azureRoleURL, cloudServiceName, deploymentName, roleName)
	if deleteVHD {
		requestURL += "?comp=media"
	}
	return vm.client.SendAzureDeleteRequest(requestURL)
}

func (vm VirtualMachineClient) GetRoleSizeList() (RoleSizeList, error) {
	roleSizeList := RoleSizeList{}

	response, err := vm.client.SendAzureGetRequest(azureRoleSizeListURL)
	if err != nil {
		return roleSizeList, err
	}

	err = xml.Unmarshal(response, &roleSizeList)
	return roleSizeList, err
}

// CaptureRole captures a VM role. If reprovisioningConfigurationSet is non-nil,
// the VM role is redeployed after capturing the image, otherwise, the original
// VM role is deleted.
//
// NOTE: an image resulting from this operation shows up in
// osimage.GetImageList() as images with Category "User".
func (vm VirtualMachineClient) CaptureRole(cloudServiceName, deploymentName, roleName, imageName, imageLabel string,
	reprovisioningConfigurationSet *ConfigurationSet) (management.OperationID, error) {
	if cloudServiceName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "cloudServiceName")
	}
	if deploymentName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "deploymentName")
	}
	if roleName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "roleName")
	}

	if reprovisioningConfigurationSet != nil &&
		!(reprovisioningConfigurationSet.ConfigurationSetType == ConfigurationSetTypeLinuxProvisioning ||
			reprovisioningConfigurationSet.ConfigurationSetType == ConfigurationSetTypeWindowsProvisioning) {
		return "", fmt.Errorf("ConfigurationSet type can only be WindowsProvisioningConfiguration or LinuxProvisioningConfiguration")
	}

	operation := CaptureRoleOperation{
		OperationType:             "CaptureRoleOperation",
		PostCaptureAction:         PostCaptureActionReprovision,
		ProvisioningConfiguration: reprovisioningConfigurationSet,
		TargetImageLabel:          imageLabel,
		TargetImageName:           imageName,
	}
	if reprovisioningConfigurationSet == nil {
		operation.PostCaptureAction = PostCaptureActionDelete
	}

	data, err := xml.Marshal(operation)
	if err != nil {
		return "", err
	}

	return vm.client.SendAzurePostRequest(fmt.Sprintf(azureOperationsURL, cloudServiceName, deploymentName, roleName), data)
}
