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

	imperativeevictionresponderconfig "k8s.io/kubernetes/pkg/controller/imperativeevictionresponder/config"
)

const (
	minConcurrentImperativeEvictionResponderSyncs = 1
	maxConcurrentImperativeEvictionResponderSyncs = 50
)

// ImperativeEvictionResponderControllerOptions holds the ImperativeEvictionResponderController options.
type ImperativeEvictionResponderControllerOptions struct {
	*imperativeevictionresponderconfig.ImperativeEvictionResponderControllerConfiguration
}

// AddFlags adds flags related to ImperativeEvictionResponderController for controller manager to the specified FlagSet.
func (o *ImperativeEvictionResponderControllerOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.Int32Var(&o.ConcurrentImperativeEvictionResponderSyncs, "concurrent-imperative-eviction-responder-syncs", o.ConcurrentImperativeEvictionResponderSyncs, "The number of eviction object syncing and imperative eviction operations that will be done concurrently. Larger number = bigger throughput of imperative evictions that call the /eviction API and faster Eviction status updating, but more CPU (and network) load. Defaults to 5.")
}

// ApplyTo fills up ImperativeEvictionResponderController config with options.
func (o *ImperativeEvictionResponderControllerOptions) ApplyTo(cfg *imperativeevictionresponderconfig.ImperativeEvictionResponderControllerConfiguration) error {
	if o == nil {
		return nil
	}

	cfg.ConcurrentImperativeEvictionResponderSyncs = o.ConcurrentImperativeEvictionResponderSyncs

	return nil
}

// Validate checks validation of ImperativeEvictionResponderControllerOptions.
func (o *ImperativeEvictionResponderControllerOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}

	if o.ConcurrentImperativeEvictionResponderSyncs < minConcurrentImperativeEvictionResponderSyncs {
		errs = append(errs, fmt.Errorf("concurrent-imperative-eviction-responder-syncs must not be less than %d, but got %d", minConcurrentImperativeEvictionResponderSyncs, o.ConcurrentImperativeEvictionResponderSyncs))
	} else if o.ConcurrentImperativeEvictionResponderSyncs > maxConcurrentImperativeEvictionResponderSyncs {
		errs = append(errs, fmt.Errorf("concurrent-imperative-eviction-responder-syncs must not be more than %d, but got %d", maxConcurrentImperativeEvictionResponderSyncs, o.ConcurrentImperativeEvictionResponderSyncs))
	}

	return errs
}
