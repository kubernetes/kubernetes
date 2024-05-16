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
	"fmt"

	"github.com/spf13/pflag"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/kubernetes/pkg/features"
)

// CloudProviderOptions contains cloud provider config
type CloudProviderOptions struct {
	CloudConfigFile string
	CloudProvider   string
}

// NewCloudProviderOptions creates a default CloudProviderOptions
func NewCloudProviderOptions() *CloudProviderOptions {
	return &CloudProviderOptions{}
}

// Validate checks invalid config
func (opts *CloudProviderOptions) Validate() []error {
	var errs []error

	switch {
	case opts.CloudProvider == "":
	case opts.CloudProvider == "external":
		if !utilfeature.DefaultFeatureGate.Enabled(features.DisableCloudProviders) {
			errs = append(errs, fmt.Errorf("when using --cloud-provider set to '%s', "+
				"please set DisableCloudProviders feature to true", opts.CloudProvider))
		}
		if !utilfeature.DefaultFeatureGate.Enabled(features.DisableKubeletCloudCredentialProviders) {
			errs = append(errs, fmt.Errorf("when using --cloud-provider set to '%s', "+
				"please set DisableKubeletCloudCredentialProviders feature to true", opts.CloudProvider))
		}
	case cloudprovider.IsDeprecatedInternal(opts.CloudProvider):
		if utilfeature.DefaultFeatureGate.Enabled(features.DisableCloudProviders) {
			errs = append(errs, fmt.Errorf("when using --cloud-provider set to '%s', "+
				"please set DisableCloudProviders feature to false", opts.CloudProvider))
		}
		if utilfeature.DefaultFeatureGate.Enabled(features.DisableKubeletCloudCredentialProviders) {
			errs = append(errs, fmt.Errorf("when using --cloud-provider set to '%s', "+
				"please set DisableKubeletCloudCredentialProviders feature to false", opts.CloudProvider))
		}
	default:
		errs = append(errs, fmt.Errorf("unknown --cloud-provider: %s", opts.CloudProvider))
	}

	return errs
}

// AddFlags returns flags of cloud provider for a API Server
func (s *CloudProviderOptions) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&s.CloudProvider, "cloud-provider", s.CloudProvider,
		"The provider for cloud services. Empty string for no provider.")
	fs.MarkDeprecated("cloud-provider", "will be removed in a future version") // nolint: errcheck
	fs.StringVar(&s.CloudConfigFile, "cloud-config", s.CloudConfigFile,
		"The path to the cloud provider configuration file. Empty string for no configuration file.")
	fs.MarkDeprecated("cloud-config", "will be removed in a future version") // nolint: errcheck
}
