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

	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
	ttlafterfinishedconfig "k8s.io/kubernetes/pkg/controller/ttlafterfinished/config"
)

// TTLAfterFinishedControllerOptions holds the TTLAfterFinishedController options.
type TTLAfterFinishedControllerOptions struct {
	*ttlafterfinishedconfig.TTLAfterFinishedControllerConfiguration
}

// AddFlags adds flags related to TTLAfterFinishedController for controller manager to the specified FlagSet.
func (o *TTLAfterFinishedControllerOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.Int32Var(&o.ConcurrentTTLSyncs, "concurrent-ttl-after-finished-syncs", o.ConcurrentTTLSyncs, fmt.Sprintf("The number of %s workers that are allowed to sync concurrently.", names.TTLAfterFinishedController))
}

// ApplyTo fills up TTLAfterFinishedController config with options.
func (o *TTLAfterFinishedControllerOptions) ApplyTo(cfg *ttlafterfinishedconfig.TTLAfterFinishedControllerConfiguration) error {
	if o == nil {
		return nil
	}

	cfg.ConcurrentTTLSyncs = o.ConcurrentTTLSyncs

	return nil
}

// Validate checks validation of TTLAfterFinishedControllerOptions.
func (o *TTLAfterFinishedControllerOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}
	return errs
}
