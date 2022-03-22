/*
Copyright 2022 The Kubernetes Authors.

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

	resourceclaimconfig "k8s.io/kubernetes/pkg/controller/resourceclaim/config"
)

// ResourceClaimControllerOptions holds the ResourceClaimController options.
type ResourceClaimControllerOptions struct {
	*resourceclaimconfig.ResourceClaimControllerConfiguration
}

// AddFlags adds flags related to ResourceClaimController for controller manager to the specified FlagSet.
func (o *ResourceClaimControllerOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.Int32Var(&o.ConcurrentResourceClaimSyncs, "concurrent-resource-claim-syncs", o.ConcurrentResourceClaimSyncs, "The number of ResourceClaim syncing operations that will be done concurrently. Larger number = faster ResourceClaim updating, but more CPU (and network) load")
}

// ApplyTo fills up ResourceClaimController config with options.
func (o *ResourceClaimControllerOptions) ApplyTo(cfg *resourceclaimconfig.ResourceClaimControllerConfiguration) error {
	if o == nil {
		return nil
	}

	cfg.ConcurrentResourceClaimSyncs = o.ConcurrentResourceClaimSyncs

	return nil
}

// Validate checks validation of ResourceClaimControllerOptions.
func (o *ResourceClaimControllerOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}
	if o.ConcurrentResourceClaimSyncs < 1 {
		errs = append(errs, fmt.Errorf("concurrent-resource-claim-syncs must be greater than 0, but got %d", o.ConcurrentResourceClaimSyncs))
	}
	return errs
}
