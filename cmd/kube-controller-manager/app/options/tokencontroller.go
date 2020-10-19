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

	serviceaccountconfig "k8s.io/kubernetes/pkg/controller/serviceaccount/config"
)

// TokenControllerOptions holds the TokenController options.
type TokenControllerOptions struct {
	*serviceaccountconfig.TokenControllerConfiguration
}

// AddFlags adds flags related to TokenController for controller manager to the specified FlagSet
func (o *TokenControllerOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.BoolVar(&o.RedactSystemTokens, "redact-system-tokens", false, "Redact tokens for secrets in the system namespace")
}

// ApplyTo fills up TokenController config with options.
func (o *TokenControllerOptions) ApplyTo(cfg *serviceaccountconfig.TokenControllerConfiguration) error {
	if o == nil {
		return nil
	}

	cfg.RedactSystemTokens = o.RedactSystemTokens

	return nil
}

// Validate checks validation of TokenControllerOptions.
func (o *TokenControllerOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}
	return errs
}
