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
	"k8s.io/kubernetes/pkg/apis/componentconfig"
)

// SAControllerOptions holds the ServiceAccountController options.
type SAControllerOptions struct {
	ServiceAccountKeyFile  string
	ConcurrentSATokenSyncs int32
	RootCAFile             string
}

// AddFlags adds flags related to ServiceAccountController for controller manager to the specified FlagSet
func (o *SAControllerOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.StringVar(&o.ServiceAccountKeyFile, "service-account-private-key-file", o.ServiceAccountKeyFile, "Filename containing a PEM-encoded private RSA or ECDSA key used to sign service account tokens.")
	fs.Int32Var(&o.ConcurrentSATokenSyncs, "concurrent-serviceaccount-token-syncs", o.ConcurrentSATokenSyncs, "The number of service account token objects that are allowed to sync concurrently. Larger number = more responsive token generation, but more CPU (and network) load")
	fs.StringVar(&o.RootCAFile, "root-ca-file", o.RootCAFile, "If set, this root certificate authority will be included in service account's token secret. This must be a valid PEM-encoded CA bundle.")
}

// ApplyTo fills up ServiceAccountController config with options.
func (o *SAControllerOptions) ApplyTo(cfg *componentconfig.SAControllerConfiguration) error {
	if o == nil {
		return nil
	}

	cfg.ServiceAccountKeyFile = o.ServiceAccountKeyFile
	cfg.ConcurrentSATokenSyncs = o.ConcurrentSATokenSyncs
	cfg.RootCAFile = o.RootCAFile

	return nil
}

// Validate checks validation of ServiceAccountControllerOptions.
func (o *SAControllerOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}
	return errs
}
