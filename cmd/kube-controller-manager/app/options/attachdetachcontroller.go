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

	attachdetachconfig "k8s.io/kubernetes/pkg/controller/volume/attachdetach/config"
)

// AttachDetachControllerOptions holds the AttachDetachController options.
type AttachDetachControllerOptions struct {
	*attachdetachconfig.AttachDetachControllerConfiguration
}

// AddFlags adds flags related to AttachDetachController for controller manager to the specified FlagSet.
func (o *AttachDetachControllerOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.BoolVar(&o.DisableAttachDetachReconcilerSync, "disable-attach-detach-reconcile-sync", false, "Disable volume attach detach reconciler sync. Disabling this may cause volumes to be mismatched with pods. Use wisely.")
	fs.DurationVar(&o.ReconcilerSyncLoopPeriod.Duration, "attach-detach-reconcile-sync-period", o.ReconcilerSyncLoopPeriod.Duration, "The reconciler sync wait time between volume attach detach. This duration must be larger than one second, and increasing this value from the default may allow for volumes to be mismatched with pods.")
}

// ApplyTo fills up AttachDetachController config with options.
func (o *AttachDetachControllerOptions) ApplyTo(cfg *attachdetachconfig.AttachDetachControllerConfiguration) error {
	if o == nil {
		return nil
	}

	cfg.DisableAttachDetachReconcilerSync = o.DisableAttachDetachReconcilerSync
	cfg.ReconcilerSyncLoopPeriod = o.ReconcilerSyncLoopPeriod

	return nil
}

// Validate checks validation of AttachDetachControllerOptions.
func (o *AttachDetachControllerOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}
	return errs
}
