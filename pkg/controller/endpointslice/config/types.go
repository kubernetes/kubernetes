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

package config

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// EndpointSliceControllerConfiguration contains elements describing
// EndpointSliceController.
type EndpointSliceControllerConfiguration struct {
	// concurrentServiceEndpointSyncs is the number of service endpoint syncing
	// operations that will be done concurrently. Larger number = faster
	// endpoint slice updating, but more CPU (and network) load.
	ConcurrentServiceEndpointSyncs int32

	// maxEndpointsPerSlice is the maximum number of endpoints that will be
	// added to an EndpointSlice. More endpoints per slice will result in fewer
	// and larger endpoint slices, but larger resources.
	MaxEndpointsPerSlice int32

	// EndpointUpdatesBatchPeriod can be used to batch endpoint updates.
	// All updates of endpoint triggered by pod change will be delayed by up to
	// 'EndpointUpdatesBatchPeriod'. If other pods in the same endpoint change
	// in that period, they will be batched to a single endpoint update.
	// Default 0 value means that each pod update triggers an endpoint update.
	EndpointUpdatesBatchPeriod metav1.Duration
}
