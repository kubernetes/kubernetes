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

	devicetaintevictionconfig "k8s.io/kubernetes/pkg/controller/devicetainteviction/config"
)

// DeviceTaintEvictionControllerOptions holds the DeviceTaintEvictionController options.
type DeviceTaintEvictionControllerOptions struct {
	*devicetaintevictionconfig.DeviceTaintEvictionControllerConfiguration
}

// AddFlags adds flags related to DeviceTaintEvictionController for controller manager to the specified FlagSet.
func (o *DeviceTaintEvictionControllerOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.Int32Var(&o.ConcurrentSyncs, "concurrent-device-taint-eviction-syncs", o.ConcurrentSyncs, "The number of operations (evicting pods, updating DeviceTaintRule status) allowed to run concurrently. Greater number = more responsive, but more CPU (and network) load")
}

// ApplyTo fills up DeviceTaintEvictionController config with options.
func (o *DeviceTaintEvictionControllerOptions) ApplyTo(cfg *devicetaintevictionconfig.DeviceTaintEvictionControllerConfiguration) error {
	if o == nil {
		return nil
	}

	cfg.ConcurrentSyncs = o.ConcurrentSyncs

	return nil
}

// Validate checks validation of DeviceTaintEvictionControllerOptions.
func (o *DeviceTaintEvictionControllerOptions) Validate() []error {
	if o == nil {
		return nil
	}

	var errs []error
	if o.ConcurrentSyncs <= 0 {
		errs = append(errs, fmt.Errorf("concurrent-device-taint-eviction-syncs must be greater than zero, got %d", o.ConcurrentSyncs))
	}
	return errs
}
