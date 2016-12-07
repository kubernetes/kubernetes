/*
Copyright 2016 The Kubernetes Authors.

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

package flags

import (
	"fmt"

	"github.com/spf13/pflag"
)

var cloudproviders = []string{
	"aws",
	"azure",
	"cloudstack",
	"gce",
	"mesos",
	"openstack",
	"ovirt",
	"photon",
	"rackspace",
	"vsphere",
}

func NewCloudProviderFlag(provider *string) pflag.Value {
	return &cloudProviderValue{provider: provider}
}

type cloudProviderValue struct {
	provider *string
}

func (c *cloudProviderValue) String() string {
	return *c.provider
}

func (c *cloudProviderValue) Set(s string) error {
	if ValidateCloudProvider(s) {
		*c.provider = s
		return nil
	}

	return fmt.Errorf("cloud provider %q is not supported, you can use any of %v", s, cloudproviders)
}

func (c *cloudProviderValue) Type() string {
	return "cloudprovider"
}

func ValidateCloudProvider(provider string) bool {
	for _, supported := range cloudproviders {
		if provider == supported {
			return true
		}
	}
	return false
}
