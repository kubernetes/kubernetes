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
	"fmt"

	"github.com/spf13/pflag"

	leaderelectionconfig "k8s.io/kubernetes/pkg/controller/leaderelection/config"
)

// LeaderElectionControllerOptions holds the LeaderElectionController options.
type LeaderElectionControllerOptions struct {
	*leaderelectionconfig.LeaderElectionControllerConfiguration
}

// AddFlags adds flags related to LeaderElectionController for controller manager to the specified FlagSet.
func (o *LeaderElectionControllerOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.Int32Var(&o.ConcurrentLeaseSyncs, "concurrent-leader-election-lease-syncs", 5, "The number of LeaderElectionController workers that are allowed to sync concurrently.")
}

// ApplyTo fills up LeaderElectionController config with options.
func (o *LeaderElectionControllerOptions) ApplyTo(cfg *leaderelectionconfig.LeaderElectionControllerConfiguration) error {
	if o == nil {
		return nil
	}

	cfg.ConcurrentLeaseSyncs = o.ConcurrentLeaseSyncs

	return nil
}

// Validate checks validation of LeaderElectionControllerOptions.
func (o *LeaderElectionControllerOptions) Validate() []error {
	if o == nil {
		return nil
	}
	var errs []error
	if o.ConcurrentLeaseSyncs <= 0 {
		// omits controller or flag names because the CLI already includes these in the message.
		errs = append(errs, fmt.Errorf("must be positive, got %d", o.ConcurrentLeaseSyncs))
	}
	return errs
}
