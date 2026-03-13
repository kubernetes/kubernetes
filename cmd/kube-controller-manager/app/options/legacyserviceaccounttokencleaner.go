/*
Copyright 2023 The Kubernetes Authors.

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

	serviceaccountconfig "k8s.io/kubernetes/pkg/controller/serviceaccount/config"
)

// LegacySATokenCleanerOptions holds the LegacySATokenCleaner options.
type LegacySATokenCleanerOptions struct {
	*serviceaccountconfig.LegacySATokenCleanerConfiguration
}

// AddFlags adds flags related to LegacySATokenCleaner for controller manager to the specified FlagSet
func (o *LegacySATokenCleanerOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.DurationVar(&o.CleanUpPeriod.Duration, "legacy-service-account-token-clean-up-period", o.CleanUpPeriod.Duration, "The period of time since the last usage of an legacy service account token before it can be deleted.")
}

// ApplyTo fills up LegacySATokenCleaner config with options.
func (o *LegacySATokenCleanerOptions) ApplyTo(cfg *serviceaccountconfig.LegacySATokenCleanerConfiguration) error {
	if o == nil {
		return nil
	}

	cfg.CleanUpPeriod = o.CleanUpPeriod

	return nil
}

// Validate checks validation of LegacySATokenCleanerOptions.
func (o *LegacySATokenCleanerOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}
	return errs
}
