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
	"time"

	"github.com/spf13/pflag"

	endpointconfig "k8s.io/kubernetes/pkg/controller/endpoint/config"
)

// EndpointControllerOptions holds the EndPointController options.
type EndpointControllerOptions struct {
	*endpointconfig.EndpointControllerConfiguration
}

// AddFlags adds flags related to EndPointController for controller manager to the specified FlagSet.
func (o *EndpointControllerOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.Int32Var(&o.ConcurrentEndpointSyncs, "concurrent-endpoint-syncs", o.ConcurrentEndpointSyncs, "The number of endpoint syncing operations that will be done concurrently. Larger number = faster endpoint updating, but more CPU (and network) load")
	fs.DurationVar(&o.EndpointUpdatesBatchPeriod.Duration, "endpoint-updates-batch-period", o.EndpointUpdatesBatchPeriod.Duration, "DEPRECATED: The length of endpoint updates batching period. Processing of pod changes will be delayed by this duration to join them with potential upcoming updates and reduce the overall number of endpoints updates. Larger number = higher endpoint programming latency, but lower number of endpoints revision generated. If set, overrides --endpoint-updates-qps.")
	fs.MarkDeprecated("endpoint-updates-batch-period", "Deprecated. Use --endpoint-updates-qps instead.")
	fs.Float64Var(&o.EndpointUpdatesQPS, "endpoint-updates-qps", o.EndpointUpdatesQPS, "Defines a maximum number of pod-triggered endpoint updates.")
	fs.IntVar(&o.EndpointUpdatesBurst, "endpoint-updates-burst", o.EndpointUpdatesBurst, "Defines a number of first pod-triggered endpoint updates that are passed without delay.")
}

// ApplyTo fills up EndPointController config with options.
func (o *EndpointControllerOptions) ApplyTo(cfg *endpointconfig.EndpointControllerConfiguration) error {
	if o == nil {
		return nil
	}

	cfg.ConcurrentEndpointSyncs = o.ConcurrentEndpointSyncs
	var zero time.Duration
	if cfg.EndpointUpdatesBatchPeriod.Duration != zero {
		cfg.EndpointUpdatesQPS = 1.0 / cfg.EndpointUpdatesBatchPeriod.Duration.Seconds()
	} else {
		cfg.EndpointUpdatesQPS = o.EndpointUpdatesQPS
	}
	cfg.EndpointUpdatesBurst = o.EndpointUpdatesBurst

	return nil
}

// Validate checks validation of EndpointControllerOptions.
func (o *EndpointControllerOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}
	return errs
}
