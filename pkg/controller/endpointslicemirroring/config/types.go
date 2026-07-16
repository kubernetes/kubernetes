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

package config

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// EndpointSliceMirroringControllerConfiguration contains elements describing
// EndpointSliceMirroringController.
type EndpointSliceMirroringControllerConfiguration struct {
	// mirroringConcurrentServiceEndpointSyncs is the number of service endpoint
	// syncing operations that will be done concurrently. Larger number = faster
	// endpoint slice updating, but more CPU (and network) load.
	MirroringConcurrentServiceEndpointSyncs int32

	// mirroringMaxEndpointsPerSubset is the maximum number of endpoints that
	// will be mirrored to an EndpointSlice for an EndpointSubset.
	MirroringMaxEndpointsPerSubset int32

	// mirroringEndpointUpdatesBatchPeriod can be used to batch EndpointSlice
	// updates. All updates triggered by EndpointSlice changes will be delayed
	// by up to 'mirroringEndpointUpdatesBatchPeriod'. If other addresses in the
	// same Endpoints resource change in that period, they will be batched to a
	// single EndpointSlice update. Default 0 value means that each Endpoints
	// update triggers an EndpointSlice update.
	MirroringEndpointUpdatesBatchPeriod metav1.Duration
}
