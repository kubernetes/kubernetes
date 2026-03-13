/*
Copyright 2020 The Kubernetes Authors.

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
	endpointslicemirroringconfig "k8s.io/kubernetes/pkg/controller/endpointslicemirroring/config"
)

const (
	mirroringMinConcurrentServiceEndpointSyncs = 1
	mirroringMaxConcurrentServiceEndpointSyncs = 50
	mirroringMinMaxEndpointsPerSubset          = 1
	mirroringMaxMaxEndpointsPerSubset          = 1000
)

// EndpointSliceMirroringControllerOptions holds the
// EndpointSliceMirroringController options.
type EndpointSliceMirroringControllerOptions struct {
	*endpointslicemirroringconfig.EndpointSliceMirroringControllerConfiguration
}

// AddFlags adds flags related to EndpointSliceMirroringController for
// controller manager to the specified FlagSet.
func (o *EndpointSliceMirroringControllerOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.Int32Var(&o.MirroringConcurrentServiceEndpointSyncs, "mirroring-concurrent-service-endpoint-syncs", o.MirroringConcurrentServiceEndpointSyncs, fmt.Sprintf("The number of service endpoint syncing operations that will be done concurrently by the %s. Larger number = faster endpoint slice updating, but more CPU (and network) load. Defaults to 5.", names.EndpointSliceMirroringController))
	fs.Int32Var(&o.MirroringMaxEndpointsPerSubset, "mirroring-max-endpoints-per-subset", o.MirroringMaxEndpointsPerSubset, fmt.Sprintf("The maximum number of endpoints that will be added to an EndpointSlice by the %s. More endpoints per slice will result in less endpoint slices, but larger resources. Defaults to 100.", names.EndpointSliceMirroringController))
	fs.DurationVar(&o.MirroringEndpointUpdatesBatchPeriod.Duration, "mirroring-endpointslice-updates-batch-period", o.MirroringEndpointUpdatesBatchPeriod.Duration, fmt.Sprintf("The length of EndpointSlice updates batching period for %s. Processing of EndpointSlice changes will be delayed by this duration to join them with potential upcoming updates and reduce the overall number of EndpointSlice updates. Larger number = higher endpoint programming latency, but lower number of endpoints revision generated", names.EndpointSliceMirroringController))
}

// ApplyTo fills up EndpointSliceMirroringController config with options.
func (o *EndpointSliceMirroringControllerOptions) ApplyTo(cfg *endpointslicemirroringconfig.EndpointSliceMirroringControllerConfiguration) error {
	if o == nil {
		return nil
	}

	cfg.MirroringConcurrentServiceEndpointSyncs = o.MirroringConcurrentServiceEndpointSyncs
	cfg.MirroringMaxEndpointsPerSubset = o.MirroringMaxEndpointsPerSubset
	cfg.MirroringEndpointUpdatesBatchPeriod = o.MirroringEndpointUpdatesBatchPeriod

	return nil
}

// Validate checks validation of EndpointSliceMirroringControllerOptions.
func (o *EndpointSliceMirroringControllerOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}

	if o.MirroringConcurrentServiceEndpointSyncs < mirroringMinConcurrentServiceEndpointSyncs {
		errs = append(errs, fmt.Errorf("mirroring-concurrent-service-endpoint-syncs must not be less than %d, but got %d", mirroringMinConcurrentServiceEndpointSyncs, o.MirroringConcurrentServiceEndpointSyncs))
	} else if o.MirroringConcurrentServiceEndpointSyncs > mirroringMaxConcurrentServiceEndpointSyncs {
		errs = append(errs, fmt.Errorf("mirroring-concurrent-service-endpoint-syncs must not be more than %d, but got %d", mirroringMaxConcurrentServiceEndpointSyncs, o.MirroringConcurrentServiceEndpointSyncs))
	}

	if o.MirroringMaxEndpointsPerSubset < mirroringMinMaxEndpointsPerSubset {
		errs = append(errs, fmt.Errorf("mirroring-max-endpoints-per-subset must not be less than %d, but got %d", mirroringMinMaxEndpointsPerSubset, o.MirroringMaxEndpointsPerSubset))
	} else if o.MirroringMaxEndpointsPerSubset > mirroringMaxMaxEndpointsPerSubset {
		errs = append(errs, fmt.Errorf("mirroring-max-endpoints-per-subset must not be more than %d, but got %d", mirroringMaxMaxEndpointsPerSubset, o.MirroringMaxEndpointsPerSubset))
	}

	return errs
}
