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

// DeprecatedControllerOptions holds the DeprecatedController options, those option are deprecated.
type DeprecatedControllerOptions struct {
	DeletingPodsQPS    float32
	DeletingPodsBurst  int32
	RegisterRetryCount int32
}

// AddFlags adds flags related to DeprecatedController for controller manager to the specified FlagSet.
func (o *DeprecatedControllerOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.Float32Var(&o.DeletingPodsQPS, "deleting-pods-qps", 0.1, "Number of nodes per second on which pods are deleted in case of node failure.")
	fs.MarkDeprecated("deleting-pods-qps", "This flag is currently no-op and will be deleted.")
	fs.Int32Var(&o.DeletingPodsBurst, "deleting-pods-burst", 0, "Number of nodes on which pods are bursty deleted in case of node failure. For more details look into RateLimiter.")
	fs.MarkDeprecated("deleting-pods-burst", "This flag is currently no-op and will be deleted.")
	fs.Int32Var(&o.RegisterRetryCount, "register-retry-count", o.RegisterRetryCount, ""+
		"The number of retries for initial node registration.  Retry interval equals node-sync-period.")
	fs.MarkDeprecated("register-retry-count", "This flag is currently no-op and will be deleted.")
}

// ApplyTo fills up DeprecatedController config with options.
func (o *DeprecatedControllerOptions) ApplyTo(cfg *componentconfig.DeprecatedControllerConfiguration) error {
	if o == nil {
		return nil
	}

	cfg.DeletingPodsQps = o.DeletingPodsQPS
	cfg.DeletingPodsBurst = o.DeletingPodsBurst
	cfg.RegisterRetryCount = o.RegisterRetryCount

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
