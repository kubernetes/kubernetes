/*
Copyright The Kubernetes Authors.

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

	evictionrequestconfig "k8s.io/kubernetes/pkg/controller/evictionrequest/config"
)

// EvictionRequestControllerOptions holds the EvictionRequestController options.
type EvictionRequestControllerOptions struct {
	*evictionrequestconfig.EvictionRequestControllerConfiguration
}

// AddFlags adds flags related to EvictionRequestController for controller manager to the specified FlagSet.
func (o *EvictionRequestControllerOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.Int32Var(&o.ConcurrentEvictionRequestSyncs, "concurrent-eviction-request-syncs", o.ConcurrentEvictionRequestSyncs, "The number of eviction requests that are allowed to sync concurrently.")
}

// ApplyTo fills up EvictionRequestController config with options.
func (o *EvictionRequestControllerOptions) ApplyTo(cfg *evictionrequestconfig.EvictionRequestControllerConfiguration) error {
	if o == nil {
		return nil
	}

	cfg.ConcurrentEvictionRequestSyncs = o.ConcurrentEvictionRequestSyncs

	return nil
}

// Validate checks validation of EvictionRequestControllerOptions.
func (o *EvictionRequestControllerOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}
	return errs
}
