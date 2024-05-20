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

	"k8s.io/controller-manager/controller"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
	endpointslicecontroller "k8s.io/kubernetes/pkg/controller/endpointslice"
	endpointslicemirroringcontroller "k8s.io/kubernetes/pkg/controller/endpointslicemirroring"
)

func newEndpointSliceControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:     names.EndpointSliceController,
		aliases:  []string{"endpointslice"},
		initFunc: startEndpointSliceController,
	}
}

func startEndpointSliceController(ctx context.Context, controllerContext ControllerContext, controllerName string) (controller.Interface, bool, error) {
	logger := klog.FromContext(ctx)
	go endpointslicecontroller.NewController(
		ctx,
		controllerContext.InformerFactory.Core().V1().Pods(),
		controllerContext.InformerFactory.Core().V1().Services(),
		controllerContext.InformerFactory.Core().V1().Nodes(),
		controllerContext.InformerFactory.Discovery().V1().EndpointSlices(),
		controllerContext.ComponentConfig.EndpointSliceController.MaxEndpointsPerSlice,
		controllerContext.ClientBuilder.ClientOrDie(logger, "endpointslice-controller"),
		controllerContext.ComponentConfig.EndpointSliceController.EndpointUpdatesBatchPeriod.Duration,
	).Run(ctx, int(controllerContext.ComponentConfig.EndpointSliceController.ConcurrentServiceEndpointSyncs))
	return nil, true, nil
}

func newEndpointSliceMirroringControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:     names.EndpointSliceMirroringController,
		aliases:  []string{"endpointslicemirroring"},
		initFunc: startEndpointSliceMirroringController,
	}
}

func startEndpointSliceMirroringController(ctx context.Context, controllerContext ControllerContext, controllerName string) (controller.Interface, bool, error) {
	logger := klog.FromContext(ctx)
	go endpointslicemirroringcontroller.NewController(
		ctx,
		controllerContext.InformerFactory.Core().V1().Endpoints(),
		controllerContext.InformerFactory.Discovery().V1().EndpointSlices(),
		controllerContext.InformerFactory.Core().V1().Services(),
		controllerContext.ComponentConfig.EndpointSliceMirroringController.MirroringMaxEndpointsPerSubset,
		controllerContext.ClientBuilder.ClientOrDie(logger, "endpointslicemirroring-controller"),
		controllerContext.ComponentConfig.EndpointSliceMirroringController.MirroringEndpointUpdatesBatchPeriod.Duration,
	).Run(ctx, int(controllerContext.ComponentConfig.EndpointSliceMirroringController.MirroringConcurrentServiceEndpointSyncs))
	return nil, true, nil
}
