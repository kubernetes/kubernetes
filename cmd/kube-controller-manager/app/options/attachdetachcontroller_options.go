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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericcontrollermanager "k8s.io/kubernetes/cmd/controller-manager/app"
)

// AttachDetachControllerOptions is part of context object for the controller manager.
type AttachDetachControllerOptions struct {
	ReconcilerSyncLoopPeriod          metav1.Duration
	DisableAttachDetachReconcilerSync bool
}

// AddFlags adds flags related to debugging for controller manager to the specified FlagSet
func (o *AttachDetachControllerOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.BoolVar(&o.DisableAttachDetachReconcilerSync, "disable-attach-detach-reconcile-sync", false, "Disable volume attach detach reconciler sync. Disabling this may cause volumes to be mismatched with pods. Use wisely.")
	fs.DurationVar(&o.ReconcilerSyncLoopPeriod.Duration, "attach-detach-reconcile-sync-period", o.ReconcilerSyncLoopPeriod.Duration, "The reconciler sync wait time between volume attach detach. This duration must be larger than one second, and increasing this value from the default may allow for volumes to be mismatched with pods.")
}

// ApplyTo fills up parts of controller manager config with options.
func (o *AttachDetachControllerOptions) ApplyTo(c *genericcontrollermanager.Config) error {
	if o == nil {
		return nil
	}

	c.ComponentConfig.AttachDetachControllerConfig.DisableAttachDetachReconcilerSync = o.DisableAttachDetachReconcilerSync
	c.ComponentConfig.AttachDetachControllerConfig.ReconcilerSyncLoopPeriod.Duration = o.ReconcilerSyncLoopPeriod.Duration

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
