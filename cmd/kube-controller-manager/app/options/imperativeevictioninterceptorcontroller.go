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

	imperativeevictioninterceptorconfig "k8s.io/kubernetes/pkg/controller/imperativeevictioninterceptor/config"
)

const (
	minConcurrentImperativeEvictionInterceptorControllerSyncs = 1
	maxConcurrentImperativeEvictionInterceptorControllerSyncs = 50
)

// ImperativeEvictionInterceptorControllerOptions holds the ImperativeEvictionInterceptorController options.
type ImperativeEvictionInterceptorControllerOptions struct {
	*imperativeevictioninterceptorconfig.ImperativeEvictionInterceptorControllerConfiguration
}

// AddFlags adds flags related to ImperativeEvictionInterceptorController for controller manager to the specified FlagSet.
func (o *ImperativeEvictionInterceptorControllerOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.Int32Var(&o.ConcurrentImperativeEvictionInterceptorControllerSyncs, "concurrent-imperative-eviction-interceptor-controller-syncs", o.ConcurrentImperativeEvictionInterceptorControllerSyncs, "The number of eviction request syncing and imperative eviction operations that will be done concurrently. Larger number = bigger throughput of imperative evictions that call the /eviction API and faster EvictionRequest status updating, but more CPU (and network) load. Defaults to 5.")
}

// ApplyTo fills up ImperativeEvictionInterceptorController config with options.
func (o *ImperativeEvictionInterceptorControllerOptions) ApplyTo(cfg *imperativeevictioninterceptorconfig.ImperativeEvictionInterceptorControllerConfiguration) error {
	if o == nil {
		return nil
	}

	cfg.ConcurrentImperativeEvictionInterceptorControllerSyncs = o.ConcurrentImperativeEvictionInterceptorControllerSyncs

	return nil
}

// Validate checks validation of ImperativeEvictionInterceptorControllerOptions.
func (o *ImperativeEvictionInterceptorControllerOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}

	if o.ConcurrentImperativeEvictionInterceptorControllerSyncs < minConcurrentImperativeEvictionInterceptorControllerSyncs {
		errs = append(errs, fmt.Errorf("concurrent-imperative-eviction-interceptor-controller-syncs must not be less than %d, but got %d", minConcurrentImperativeEvictionInterceptorControllerSyncs, o.ConcurrentImperativeEvictionInterceptorControllerSyncs))
	} else if o.ConcurrentImperativeEvictionInterceptorControllerSyncs > maxConcurrentImperativeEvictionInterceptorControllerSyncs {
		errs = append(errs, fmt.Errorf("concurrent-imperative-eviction-interceptor-controller-syncs must not be more than %d, but got %d", maxConcurrentImperativeEvictionInterceptorControllerSyncs, o.ConcurrentImperativeEvictionInterceptorControllerSyncs))
	}

	return errs
}
