/*
Copyright 2019 The Kubernetes Authors.

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

	endpointsliceconfig "k8s.io/kubernetes/pkg/controller/endpointslice/config"
)

const (
	minConcurrentServiceEndpointSyncs = 1
	maxConcurrentServiceEndpointSyncs = 50
	minMaxEndpointsPerSlice           = 1
	maxMaxEndpointsPerSlice           = 1000
)

// EndpointSliceControllerOptions holds the EndpointSliceController options.
type EndpointSliceControllerOptions struct {
	*endpointsliceconfig.EndpointSliceControllerConfiguration
}

// AddFlags adds flags related to EndpointSliceController for controller manager to the specified FlagSet.
func (o *EndpointSliceControllerOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.Int32Var(&o.ConcurrentServiceEndpointSyncs, "concurrent-service-endpoint-syncs", o.ConcurrentServiceEndpointSyncs, "The number of service endpoint syncing operations that will be done concurrently. Larger number = faster endpoint slice updating, but more CPU (and network) load. Defaults to 5.")
	fs.Int32Var(&o.MaxEndpointsPerSlice, "max-endpoints-per-slice", o.MaxEndpointsPerSlice, "The maximum number of endpoints that will be added to an EndpointSlice. More endpoints per slice will result in less endpoint slices, but larger resources. Defaults to 100.")
}

// ApplyTo fills up EndpointSliceController config with options.
func (o *EndpointSliceControllerOptions) ApplyTo(cfg *endpointsliceconfig.EndpointSliceControllerConfiguration) error {
	if o == nil {
		return nil
	}

	cfg.ConcurrentServiceEndpointSyncs = o.ConcurrentServiceEndpointSyncs
	cfg.MaxEndpointsPerSlice = o.MaxEndpointsPerSlice

	return nil
}

// Validate checks validation of EndpointSliceControllerOptions.
func (o *EndpointSliceControllerOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}

	if o.ConcurrentServiceEndpointSyncs < minConcurrentServiceEndpointSyncs {
		errs = append(errs, fmt.Errorf("concurrent-service-endpoint-syncs must not be less than %d, but got %d", minConcurrentServiceEndpointSyncs, o.ConcurrentServiceEndpointSyncs))
	} else if o.ConcurrentServiceEndpointSyncs > maxConcurrentServiceEndpointSyncs {
		errs = append(errs, fmt.Errorf("concurrent-service-endpoint-syncs must not be more than %d, but got %d", maxConcurrentServiceEndpointSyncs, o.ConcurrentServiceEndpointSyncs))
	}

	if o.MaxEndpointsPerSlice < minMaxEndpointsPerSlice {
		errs = append(errs, fmt.Errorf("max-endpoints-per-slice must not be less than %d, but got %d", minMaxEndpointsPerSlice, o.MaxEndpointsPerSlice))
	} else if o.MaxEndpointsPerSlice > maxMaxEndpointsPerSlice {
		errs = append(errs, fmt.Errorf("max-endpoints-per-slice must not be more than %d, but got %d", maxMaxEndpointsPerSlice, o.MaxEndpointsPerSlice))
	}

	return errs
}
