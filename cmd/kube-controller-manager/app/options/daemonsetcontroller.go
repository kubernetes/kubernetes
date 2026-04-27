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
	"fmt"

	"github.com/spf13/pflag"

	daemonconfig "k8s.io/kubernetes/pkg/controller/daemon/config"
)

// DaemonSetControllerOptions holds the DaemonSetController options.
type DaemonSetControllerOptions struct {
	*daemonconfig.DaemonSetControllerConfiguration
}

// AddFlags adds flags related to DaemonSetController for controller manager to the specified FlagSet.
func (o *DaemonSetControllerOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.Int32Var(&o.ConcurrentDaemonSetSyncs, "concurrent-daemonset-syncs", o.ConcurrentDaemonSetSyncs, "The number of daemonset objects that are allowed to sync concurrently. Larger number = more responsive daemonsets, but more CPU (and network) load")
}

// ApplyTo fills up DaemonSetController config with options.
func (o *DaemonSetControllerOptions) ApplyTo(cfg *daemonconfig.DaemonSetControllerConfiguration) error {
	if o == nil {
		return nil
	}

	cfg.ConcurrentDaemonSetSyncs = o.ConcurrentDaemonSetSyncs

	return nil
}

// Validate checks validation of DaemonSetControllerOptions.
func (o *DaemonSetControllerOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}
	if o.ConcurrentDaemonSetSyncs < 1 {
		errs = append(errs, fmt.Errorf("concurrent-daemonset-syncs must be greater than 0, but got %d", o.ConcurrentDaemonSetSyncs))
	}
	return errs
}
