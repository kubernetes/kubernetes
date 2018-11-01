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

import (
	"github.com/spf13/pflag"

	kubectrlmgrconfig "k8s.io/kubernetes/pkg/controller/apis/config"
)

// ValidateCloudProviderConfiguration ensures validation of the CloudProviderConfiguration struct.
func ValidateCloudProviderConfiguration(s *kubectrlmgrconfig.CloudProviderConfiguration) []error {
	allErrors := []error{}

	return allErrors
}

// AddCloudProviderConfigurationFlags adds flags related to cloudprovider for controller manager to the specified FlagSet.
func AddCloudProviderConfigurationFlags(s *kubectrlmgrconfig.CloudProviderConfiguration, fs *pflag.FlagSet) {
	fs.StringVar(&s.Name, "cloud-provider", s.Name,
		"The provider for cloud services. Empty string for no provider.")

	fs.StringVar(&s.CloudConfigFile, "cloud-config", s.CloudConfigFile,
		"The path to the cloud provider configuration file. Empty string for no configuration file.")
}
