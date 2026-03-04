/*
Copyright 2021 The Kubernetes Authors.

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

	ephemeralvolumeconfig "k8s.io/kubernetes/pkg/controller/volume/ephemeral/config"
)

// EphemeralVolumeControllerOptions holds the EphemeralVolumeController options.
type EphemeralVolumeControllerOptions struct {
	*ephemeralvolumeconfig.EphemeralVolumeControllerConfiguration
}

// AddFlags adds flags related to EphemeralVolumeController for controller manager to the specified FlagSet.
func (o *EphemeralVolumeControllerOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.Int32Var(&o.ConcurrentEphemeralVolumeSyncs, "concurrent-ephemeralvolume-syncs", o.ConcurrentEphemeralVolumeSyncs, "The number of ephemeral volume syncing operations that will be done concurrently. Larger number = faster ephemeral volume updating, but more CPU (and network) load")
}

// ApplyTo fills up EphemeralVolumeController config with options.
func (o *EphemeralVolumeControllerOptions) ApplyTo(cfg *ephemeralvolumeconfig.EphemeralVolumeControllerConfiguration) error {
	if o == nil {
		return nil
	}

	cfg.ConcurrentEphemeralVolumeSyncs = o.ConcurrentEphemeralVolumeSyncs

	return nil
}

// Validate checks validation of EphemeralVolumeControllerOptions.
func (o *EphemeralVolumeControllerOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}
	if o.ConcurrentEphemeralVolumeSyncs < 1 {
		errs = append(errs, fmt.Errorf("concurrent-ephemeralvolume-syncs must be greater than 0, but got %d", o.ConcurrentEphemeralVolumeSyncs))
	}
	return errs
}
