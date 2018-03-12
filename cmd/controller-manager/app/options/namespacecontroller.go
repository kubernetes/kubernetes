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
	"k8s.io/kubernetes/pkg/apis/componentconfig"
)

// NamespaceControllerOptions holds the NamespaceController options.
type NamespaceControllerOptions struct {
	NamespaceSyncPeriod      metav1.Duration
	ConcurrentNamespaceSyncs int32
}

// AddFlags adds flags related to NamespaceController for controller manager to the specified FlagSet.
func (o *NamespaceControllerOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.DurationVar(&o.NamespaceSyncPeriod.Duration, "namespace-sync-period", o.NamespaceSyncPeriod.Duration, "The period for syncing namespace life-cycle updates")
	fs.Int32Var(&o.ConcurrentNamespaceSyncs, "concurrent-namespace-syncs", o.ConcurrentNamespaceSyncs, "The number of namespace objects that are allowed to sync concurrently. Larger number = more responsive namespace termination, but more CPU (and network) load")
}

// ApplyTo fills up NamespaceController config with options.
func (o *NamespaceControllerOptions) ApplyTo(cfg *componentconfig.NamespaceControllerConfiguration) error {
	if o == nil {
		return nil
	}

	cfg.NamespaceSyncPeriod = o.NamespaceSyncPeriod
	cfg.ConcurrentNamespaceSyncs = o.ConcurrentNamespaceSyncs

	return nil
}

// Validate checks validation of NamespaceControllerOptions.
func (o *NamespaceControllerOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}
	return errs
}
