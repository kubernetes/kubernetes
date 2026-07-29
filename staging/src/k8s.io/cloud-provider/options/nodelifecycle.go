/*
Copyright The Kubernetes Authors.

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

	nodelifecycleconfig "k8s.io/cloud-provider/controllers/nodelifecycle/config"
	"k8s.io/cloud-provider/names"
)

// NodeLifecycleControllerOptions holds the CloudNodeLifecycleController options.
type NodeLifecycleControllerOptions struct {
	*nodelifecycleconfig.NodeLifecycleControllerConfiguration
}

// AddFlags adds flags related to CloudNodeLifecycleController for controller manager to the specified FlagSet.
func (o *NodeLifecycleControllerOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.Int32Var(&o.ConcurrentNodeLifecycleSyncs, "concurrent-node-lifecycle-syncs", o.ConcurrentNodeLifecycleSyncs,
		fmt.Sprintf("The number of workers for syncing NodeStatus in %s.", names.CloudNodeLifecycleController))
	fs.DurationVar(&o.NodeMonitorPeriod.Duration, "node-monitor-period", o.NodeMonitorPeriod.Duration,
		"The period for syncing NodeStatus in CloudNodeLifecycleController.")
}

// ApplyTo fills up CloudNodeLifecycleController config with options.
func (o *NodeLifecycleControllerOptions) ApplyTo(cfg *nodelifecycleconfig.NodeLifecycleControllerConfiguration) error {
	if o == nil {
		return nil
	}

	cfg.ConcurrentNodeLifecycleSyncs = o.ConcurrentNodeLifecycleSyncs
	cfg.NodeMonitorPeriod = o.NodeMonitorPeriod

	return nil
}

// Validate checks validation of NodeLifecycleControllerOptions.
func (o *NodeLifecycleControllerOptions) Validate() []error {
	if o == nil {
		return nil
	}
	var errs []error

	if o.ConcurrentNodeLifecycleSyncs < 1 {
		errs = append(errs, fmt.Errorf("concurrent-node-lifecycle-syncs must be at least 1, but got %d", o.ConcurrentNodeLifecycleSyncs))
	}

	return errs
}
