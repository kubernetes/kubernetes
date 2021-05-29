/*
Copyright 2016 The Kubernetes Authors.

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

// Package app implements a server that runs a set of active
// components.  This includes replication controllers, service endpoints and
// nodes.
//
package app

import (
	"net/http"

	endpointslicecontroller "k8s.io/kubernetes/pkg/controller/endpointslice"
	endpointslicemirroringcontroller "k8s.io/kubernetes/pkg/controller/endpointslicemirroring"
)

func startEndpointSliceController(ctx ControllerContext) (http.Handler, bool, error) {
	go endpointslicecontroller.NewController(
		ctx.InformerFactory.Core().V1().Pods(),
		ctx.InformerFactory.Core().V1().Services(),
		ctx.InformerFactory.Core().V1().Nodes(),
		ctx.InformerFactory.Discovery().V1().EndpointSlices(),
		ctx.ComponentConfig.EndpointSliceController.MaxEndpointsPerSlice,
		ctx.ClientBuilder.ClientOrDie("endpointslice-controller"),
		ctx.ComponentConfig.EndpointSliceController.EndpointUpdatesBatchPeriod.Duration,
	).Run(int(ctx.ComponentConfig.EndpointSliceController.ConcurrentServiceEndpointSyncs), ctx.Stop)
	return nil, true, nil
}

func startEndpointSliceMirroringController(ctx ControllerContext) (http.Handler, bool, error) {
	go endpointslicemirroringcontroller.NewController(
		ctx.InformerFactory.Core().V1().Endpoints(),
		ctx.InformerFactory.Discovery().V1().EndpointSlices(),
		ctx.InformerFactory.Core().V1().Services(),
		ctx.ComponentConfig.EndpointSliceMirroringController.MirroringMaxEndpointsPerSubset,
		ctx.ClientBuilder.ClientOrDie("endpointslicemirroring-controller"),
		ctx.ComponentConfig.EndpointSliceMirroringController.MirroringEndpointUpdatesBatchPeriod.Duration,
	).Run(int(ctx.ComponentConfig.EndpointSliceMirroringController.MirroringConcurrentServiceEndpointSyncs), ctx.Stop)
	return nil, true, nil
}
