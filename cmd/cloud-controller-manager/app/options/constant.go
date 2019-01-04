/*
Copyright 2018 The Kubernetes Authors.

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

package options

const (
	// cloudControllerManagerUserAgent is the userAgent name when starting cloud-controller managers.
	cloudControllerManagerUserAgent = "cloud-controller-manager"

	// DefaultInsecureCloudControllerManagerPort is the default insecure cloud-controller manager port.
	DefaultInsecureCloudControllerManagerPort = 0

	// CloudControllerManagerServiceController cloud controller manager service controller name.
	CloudControllerManagerServiceController = "service controller"

	// CloudControllerManagerSecureServing cloud controller manager secure serving option value.
	CloudControllerManagerSecureServing = "secure serving"

	// CloudControllerManagerInsecureServing cloud controller manager insecure serving option value.
	CloudControllerManagerInsecureServing = "insecure serving"

	// CloudControllerManagerAuthentication flag sets authentication headers.
	CloudControllerManagerAuthentication = "authentication"

	// CloudControllerManagerAuthorization  flag sets authentication headers.
	CloudControllerManagerAuthorization = "authorization"

	// CloudControllerManagerMisc initiate misc/additional flags.
	CloudControllerManagerMisc = "misc"

	// CloudControllerManagerMaster flag sets the address of the Kubernetes API server.
	CloudControllerManagerMaster = "master"

	// CloudControllerManagerKubeConfig flag sets Path to kubeconfig file with authorization and master location information.
	CloudControllerManagerKubeConfig = "kubeconfig"

	// CloudControllerManagerNodeStatusUpdateFrequency flag sets specifies how often the controller updates nodes' status.
	CloudControllerManagerNodeStatusUpdateFrequency = "node-status-update-frequency"

	// CloudControllerManagerInsecureServingNetworkProtocol defines the default insecure serving network protocol
	CloudControllerManagerInsecureServingNetworkProtocol = "tcp"
)
