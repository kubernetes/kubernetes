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

package options

import (
	"github.com/spf13/pflag"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
)

type CloudProviderOptions struct {
	CloudConfigFile string
	CloudProvider   string
}

func NewCloudProviderOptions() *CloudProviderOptions {
	return &CloudProviderOptions{}
}

func (s *CloudProviderOptions) Validate() []error {
	allErrors := []error{}
	return allErrors
}

func (s *CloudProviderOptions) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&s.CloudProvider, "cloud-provider", s.CloudProvider,
		"The provider for cloud services. Empty string for no provider.")

	fs.StringVar(&s.CloudConfigFile, "cloud-config", s.CloudConfigFile,
		"The path to the cloud provider configuration file. Empty string for no configuration file.")
}

func (s *CloudProviderOptions) ApplyTo(c **CloudProviderOptions, cfg *componentconfig.KubeControllerManagerConfiguration) error {
	if s == nil {
		return nil
	}

	*c = &CloudProviderOptions{
		CloudProvider:   s.CloudProvider,
		CloudConfigFile: s.CloudConfigFile,
	}
	// sync back to component config
	// TODO: find more elegant way than synching back the values.
	cfg.CloudProvider = s.CloudProvider
	cfg.CloudConfigFile = s.CloudConfigFile

	return nil
}
