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

	"github.com/Azure/azure-sdk-for-go/services/classic/management"
)

const (
	azureDeploymentListURL                    = "services/hostedservices/%s/deployments"
	azureDeploymentURL                        = "services/hostedservices/%s/deployments/%s"
	azureUpdateDeploymentURL                  = "services/hostedservices/%s/deployments/%s?comp=%s"
	azureDeploymentSlotSwapURL                = "services/hostedservices/%s"
	azureDeploymentSlotURL                    = "services/hostedservices/%s/deploymentslots/%s"
	azureUpdateDeploymentSlotConfigurationURL = "services/hostedservices/%s/deploymentslots/%s?comp=%s"
	deleteAzureDeploymentURL                  = "services/hostedservices/%s/deployments/%s?comp=media"
	azureDeleteDeploymentBySlotURL            = "services/hostedservices/%s/deploymentslots/%s"
	azureAddRoleURL                           = "services/hostedservices/%s/deployments/%s/roles"
	azureRoleURL                              = "services/hostedservices/%s/deployments/%s/roles/%s"
	azureOperationsURL                        = "services/hostedservices/%s/deployments/%s/roleinstances/%s/Operations"
	azureRoleSizeListURL                      = "rolesizes"

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

// CreateDeploymentFromPackageOptions can be used to create a customized deployement request
type CreateDeploymentFromPackageOptions struct {
	Name                   string
	PackageURL             string
	Label                  string
	Configuration          string
	StartDeployment        bool
	TreatWarningsAsError   bool
	ExtendedProperties     []ExtendedProperty
	ExtensionConfiguration ExtensionConfiguration
}

// CreateDeploymentRequest is the type for creating a deployment of a cloud service package
// in the deployment based on the specified configuration. See
// https://docs.microsoft.com/en-us/rest/api/compute/cloudservices/rest-create-deployment
type CreateDeploymentRequest struct {
	XMLName xml.Name `xml:"http://schemas.microsoft.com/windowsazure CreateDeployment"`
	// Required parameters:
	Name          string ``                 // Specifies the name of the deployment.
	PackageURL    string `xml:"PackageUrl"` // Specifies a URL that refers to the location of the service package in the Blob service. The service package can be located either in a storage account beneath the same subscription or a Shared Access Signature (SAS) URI from any storage account.
	Label         string ``                 // Specifies an identifier for the deployment that is base-64 encoded. The identifier can be up to 100 characters in length. It is recommended that the label be unique within the subscription. The label can be used for your tracking purposes.
	Configuration string ``                 // Specifies the base-64 encoded service configuration file for the deployment.
	// Optional parameters:
	StartDeployment        bool                   ``                                  // Indicates whether to start the deployment immediately after it is created. The default value is false
	TreatWarningsAsError   bool                   ``                                  // Indicates whether to treat package validation warnings as errors. The default value is false. If set to true, the Created Deployment operation fails if there are validation warnings on the service package.
	ExtendedProperties     []ExtendedProperty     `xml:">ExtendedProperty,omitempty"` // Array of ExtendedProprties. Each extended property must have both a defined name and value. You can have a maximum of 25 extended property name and value pairs.
	ExtensionConfiguration ExtensionConfiguration `xml:",omitempty"`
}

// CreateDeploymentFromPackage creates a deployment from a cloud services package (.cspkg) and configuration file (.cscfg)
func (vm VirtualMachineClient) CreateDeploymentFromPackage(
	cloudServiceName string,
	deploymentSlot DeploymentSlot,
	options CreateDeploymentFromPackageOptions) (management.OperationID, error) {

	if cloudServiceName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "cloudServiceName")
	}

	req := CreateDeploymentRequest{
		Name:                   options.Name,
		Label:                  options.Label,
		Configuration:          options.Configuration,
		PackageURL:             options.PackageURL,
		StartDeployment:        options.StartDeployment,
		TreatWarningsAsError:   options.TreatWarningsAsError,
		ExtendedProperties:     options.ExtendedProperties,
		ExtensionConfiguration: options.ExtensionConfiguration,
	}

	data, err := xml.Marshal(req)
	if err != nil {
		return "", err
	}

	requestURL := fmt.Sprintf(azureDeploymentSlotURL, cloudServiceName, deploymentSlot)
	return vm.client.SendAzurePostRequest(requestURL, data)
}

// SwapDeploymentRequest is the type used for specifying information to swap the deployments in
// a cloud service
// https://docs.microsoft.com/en-us/rest/api/compute/cloudservices/rest-swap-deployment
type SwapDeploymentRequest struct {
	XMLName xml.Name `xml:"http://schemas.microsoft.com/windowsazure Swap"`
	// Required parameters:
	Production       string
	SourceDeployment string
}

// SwapDeployment initiates a virtual IP address swap between the staging and production deployment environments for a service.
// If the service is currently running in the staging environment, it will be swapped to the production environment.
// If it is running in the production environment, it will be swapped to staging.
func (vm VirtualMachineClient) SwapDeployment(
	cloudServiceName string) (management.OperationID, error) {

	if cloudServiceName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "cloudServiceName")
	}

	productionDeploymentName, err := vm.GetDeploymentNameForSlot(cloudServiceName, DeploymentSlotProduction)
	if err != nil {
		return "", err
	}

	stagingDeploymentName, err := vm.GetDeploymentNameForSlot(cloudServiceName, DeploymentSlotStaging)
	if err != nil {
		return "", err
	}

	req := SwapDeploymentRequest{
		Production:       productionDeploymentName,
		SourceDeployment: stagingDeploymentName,
	}

	data, err := xml.Marshal(req)
	if err != nil {
		return "", err
	}

	requestURL := fmt.Sprintf(azureDeploymentSlotSwapURL, cloudServiceName)
	return vm.client.SendAzurePostRequest(requestURL, data)
}

// ChangeDeploymentConfigurationRequestOptions can be used to update configuration of a deployment
type ChangeDeploymentConfigurationRequestOptions struct {
	Mode                   UpgradeType
	Configuration          string
	TreatWarningsAsError   bool
	ExtendedProperties     []ExtendedProperty
	ExtensionConfiguration ExtensionConfiguration
}

// ChangeDeploymentConfigurationRequest is the type for changing the configuration of a deployment of a cloud service p
// https://docs.microsoft.com/en-us/rest/api/compute/cloudservices/rest-change-deployment-configuration
type ChangeDeploymentConfigurationRequest struct {
	XMLName xml.Name `xml:"http://schemas.microsoft.com/windowsazure ChangeConfiguration"`
	// Required parameters:
	Configuration string `` // Specifies the base-64 encoded service configuration file for the deployment.
	// Optional parameters:
	Mode                   UpgradeType            ``                                  // Specifies the type of Upgrade (Auto | Manual | Simultaneous) .
	TreatWarningsAsError   bool                   ``                                  // Indicates whether to treat package validation warnings as errors. The default value is false. If set to true, the Created Deployment operation fails if there are validation warnings on the service package.
	ExtendedProperties     []ExtendedProperty     `xml:">ExtendedProperty,omitempty"` // Array of ExtendedProprties. Each extended property must have both a defined name and value. You can have a maximum of 25 extended property name and value pairs.
	ExtensionConfiguration ExtensionConfiguration `xml:",omitempty"`
}

// ChangeDeploymentConfiguration updates the configuration for a deployment from a configuration file (.cscfg)
func (vm VirtualMachineClient) ChangeDeploymentConfiguration(
	cloudServiceName string,
	deploymentSlot DeploymentSlot,
	options ChangeDeploymentConfigurationRequestOptions) (management.OperationID, error) {

	if cloudServiceName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "cloudServiceName")
	}

	req := ChangeDeploymentConfigurationRequest{
		Mode:                   options.Mode,
		Configuration:          options.Configuration,
		TreatWarningsAsError:   options.TreatWarningsAsError,
		ExtendedProperties:     options.ExtendedProperties,
		ExtensionConfiguration: options.ExtensionConfiguration,
	}
	if req.Mode == "" {
		req.Mode = UpgradeTypeAuto
	}

	data, err := xml.Marshal(req)
	if err != nil {
		return "", err
	}

	requestURL := fmt.Sprintf(azureUpdateDeploymentSlotConfigurationURL, cloudServiceName, deploymentSlot, "config")
	return vm.client.SendAzurePostRequest(requestURL, data)
}

// UpdateDeploymentStatusRequest is the type used to make UpdateDeploymentStatus requests
type UpdateDeploymentStatusRequest struct {
	XMLName xml.Name `xml:"http://schemas.microsoft.com/windowsazure UpdateDeploymentStatus"`
	// Required parameters:
	Status string
}

// UpdateDeploymentStatus changes the running status of a deployment. The status of a deployment can be running or suspended.
// https://docs.microsoft.com/en-us/rest/api/compute/cloudservices/rest-update-deployment-status
func (vm VirtualMachineClient) UpdateDeploymentStatus(
	cloudServiceName string,
	deploymentSlot DeploymentSlot,
	status string) (management.OperationID, error) {

	if cloudServiceName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "cloudServiceName")
	}

	if status != "Running" && status != "Suspended" {
		return "", fmt.Errorf("Invalid status provided")
	}

	req := UpdateDeploymentStatusRequest{
		Status: status,
	}

	data, err := xml.Marshal(req)
	if err != nil {
		return "", err
	}

	requestURL := fmt.Sprintf(azureUpdateDeploymentSlotConfigurationURL, cloudServiceName, deploymentSlot, "status")
	return vm.client.SendAzurePostRequest(requestURL, data)
}

// UpdateDeploymentStatusByName changes the running status of a deployment. The status of a deployment can be running or suspended.
// https://docs.microsoft.com/en-us/rest/api/compute/cloudservices/rest-update-deployment-status
func (vm VirtualMachineClient) UpdateDeploymentStatusByName(
	cloudServiceName string,
	deploymentName string,
	status string) (management.OperationID, error) {

	if cloudServiceName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "cloudServiceName")
	}

	if status != "Running" && status != "Suspended" {
		return "", fmt.Errorf("Invalid status provided")
	}

	req := UpdateDeploymentStatusRequest{
		Status: status,
	}

	data, err := xml.Marshal(req)
	if err != nil {
		return "", err
	}

	requestURL := fmt.Sprintf(azureUpdateDeploymentURL, cloudServiceName, deploymentName, "status")
	return vm.client.SendAzurePostRequest(requestURL, data)
}

// GetDeploymentName queries an existing Azure cloud service for the name of the Deployment,
// if any, in its 'Production' slot (the only slot possible). If none exists, it returns empty
// string but no error
//
//https://msdn.microsoft.com/en-us/library/azure/ee460804.aspx
func (vm VirtualMachineClient) GetDeploymentName(cloudServiceName string) (string, error) {
	return vm.GetDeploymentNameForSlot(cloudServiceName, DeploymentSlotProduction)
}

// GetDeploymentNameForSlot queries an existing Azure cloud service for the name of the Deployment,
// in a given slot. If none exists, it returns empty
// string but no error
//
//https://msdn.microsoft.com/en-us/library/azure/ee460804.aspx
func (vm VirtualMachineClient) GetDeploymentNameForSlot(cloudServiceName string, deploymentSlot DeploymentSlot) (string, error) {
	var deployment DeploymentResponse
	if cloudServiceName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "cloudServiceName")
	}
	requestURL := fmt.Sprintf(azureDeploymentSlotURL, cloudServiceName, deploymentSlot)
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

// GetDeploymentBySlot used to retrieve deployment events for a single deployment slot (staging or production)
func (vm VirtualMachineClient) GetDeploymentBySlot(cloudServiceName string, deploymentSlot DeploymentSlot) (DeploymentResponse, error) {
	var deployment DeploymentResponse
	if cloudServiceName == "" {
		return deployment, fmt.Errorf(errParamNotSpecified, "cloudServiceName")
	}
	if deploymentSlot == "" {
		return deployment, fmt.Errorf(errParamNotSpecified, "deploymentSlot")
	}
	requestURL := fmt.Sprintf(azureDeploymentSlotURL, cloudServiceName, deploymentSlot)
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

func (vm VirtualMachineClient) DeleteDeploymentBySlot(cloudServiceName string, deploymentSlot DeploymentSlot) (management.OperationID, error) {
	if cloudServiceName == "" {
		return "", fmt.Errorf(errParamNotSpecified, "cloudServiceName")
	}
	if deploymentSlot == "" {
		return "", fmt.Errorf(errParamNotSpecified, "deploymentSlot")
	}

	requestURL := fmt.Sprintf(azureDeleteDeploymentBySlotURL, cloudServiceName, deploymentSlot)
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
