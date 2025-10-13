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
package app

import (
	"context"

	endpointslicecontroller "k8s.io/kubernetes/pkg/controller/endpointslice"
	endpointslicemirroringcontroller "k8s.io/kubernetes/pkg/controller/endpointslicemirroring"

	"k8s.io/kubernetes/cmd/kube-controller-manager/internal/controller"
	"k8s.io/kubernetes/cmd/kube-controller-manager/internal/controller/run"
	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
)

func newEndpointSliceControllerDescriptor() *controller.Descriptor {
	return &controller.Descriptor{
		Name:        names.EndpointSliceController,
		Aliases:     []string{"endpointslice"},
		Constructor: newEndpointSliceController,
	}
}

func newEndpointSliceController(ctx context.Context, controllerContext controller.Context, controllerName string) (controller.Controller, error) {
	client, err := controllerContext.NewClient("endpointslice-controller")
	if err != nil {
		return nil, err
	}

	esc := endpointslicecontroller.NewController(
		ctx,
		controllerContext.InformerFactory.Core().V1().Pods(),
		controllerContext.InformerFactory.Core().V1().Services(),
		controllerContext.InformerFactory.Core().V1().Nodes(),
		controllerContext.InformerFactory.Discovery().V1().EndpointSlices(),
		controllerContext.ComponentConfig.EndpointSliceController.MaxEndpointsPerSlice,
		client,
		controllerContext.ComponentConfig.EndpointSliceController.EndpointUpdatesBatchPeriod.Duration,
	)
	return run.NewControllerLoop(func(ctx context.Context) {
		esc.Run(ctx, int(controllerContext.ComponentConfig.EndpointSliceController.ConcurrentServiceEndpointSyncs))
	}, controllerName), nil
}

func newEndpointSliceMirroringControllerDescriptor() *controller.Descriptor {
	return &controller.Descriptor{
		Name:        names.EndpointSliceMirroringController,
		Aliases:     []string{"endpointslicemirroring"},
		Constructor: newEndpointSliceMirroringController,
	}
}

func newEndpointSliceMirroringController(ctx context.Context, controllerContext controller.Context, controllerName string) (controller.Controller, error) {
	client, err := controllerContext.NewClient("endpointslicemirroring-controller")
	if err != nil {
		return nil, err
	}

	esmc := endpointslicemirroringcontroller.NewController(
		ctx,
		controllerContext.InformerFactory.Core().V1().Endpoints(),
		controllerContext.InformerFactory.Discovery().V1().EndpointSlices(),
		controllerContext.InformerFactory.Core().V1().Services(),
		controllerContext.ComponentConfig.EndpointSliceMirroringController.MirroringMaxEndpointsPerSubset,
		client,
		controllerContext.ComponentConfig.EndpointSliceMirroringController.MirroringEndpointUpdatesBatchPeriod.Duration,
	)
	return run.NewControllerLoop(func(ctx context.Context) {
		esmc.Run(ctx, int(controllerContext.ComponentConfig.EndpointSliceMirroringController.MirroringConcurrentServiceEndpointSyncs))
	}, controllerName), nil
}
