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

// DeprecatedControllerOptions holds the DeprecatedController options, those option are deprecated.
// TODO remove these fields once the deprecated flags are removed.
type DeprecatedControllerOptions struct {
	*kubectrlmgrconfig.DeprecatedControllerConfiguration
}

// AddFlags adds flags related to DeprecatedController for controller manager to the specified FlagSet.
func (o *DeprecatedControllerOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}
}

// ApplyTo fills up DeprecatedController config with options.
func (o *DeprecatedControllerOptions) ApplyTo(cfg *kubectrlmgrconfig.DeprecatedControllerConfiguration) error {
	if o == nil {
		return nil
	}

	return nil
}

// Validate checks validation of DeprecatedControllerOptions.
func (o *DeprecatedControllerOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}
	return errs
}
