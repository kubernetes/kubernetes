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
	"os"

	"github.com/spf13/pflag"

	genericoptions "k8s.io/apiserver/pkg/server/options"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/cloudprovider"
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

func (s *CloudProviderOptions) DefaultExternalHost(genericoptions *genericoptions.ServerRunOptions) error {
	if len(genericoptions.ExternalHost) != 0 {
		return nil
	}

	// TODO: extend for other providers
	if s.CloudProvider == "gce" || s.CloudProvider == "aws" {
		cloud, err := cloudprovider.InitCloudProvider(s.CloudProvider, s.CloudConfigFile)
		if err != nil {
			return fmt.Errorf("%q cloud provider could not be initialized: %v", s.CloudProvider, err)
		}
		instances, supported := cloud.Instances()
		if !supported {
			return fmt.Errorf("%q cloud provider has no instances", s.CloudProvider)
		}
		hostname, err := os.Hostname()
		if err != nil {
			return fmt.Errorf("failed to get hostname: %v", err)
		}
		nodeName, err := instances.CurrentNodeName(hostname)
		if err != nil {
			return fmt.Errorf("failed to get NodeName from %q cloud provider: %v", s.CloudProvider, err)
		}
		addrs, err := instances.NodeAddresses(nodeName)
		if err != nil {
			return fmt.Errorf("failed to get external host address from %q cloud provider: %v", s.CloudProvider, err)
		} else {
			for _, addr := range addrs {
				if addr.Type == v1.NodeExternalIP {
					genericoptions.ExternalHost = addr.Address
				}
			}
		}
	}

	return nil
}
