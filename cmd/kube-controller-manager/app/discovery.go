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

	discoveryv1beta1 "k8s.io/api/discovery/v1beta1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog"
	endpointslicecontroller "k8s.io/kubernetes/pkg/controller/endpointslice"
	"k8s.io/kubernetes/pkg/features"
)

func startEndpointSliceController(ctx ControllerContext) (http.Handler, bool, error) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.EndpointSlice) {
		klog.V(4).Infof("Not starting endpointslice-controller since EndpointSlice feature gate is disabled")
		return nil, false, nil
	}

	if !ctx.AvailableResources[discoveryv1beta1.SchemeGroupVersion.WithResource("endpointslices")] {
		klog.Warningf("Not starting endpointslice-controller since discovery.k8s.io/v1beta1 resources are not available")
		return nil, false, nil
	}

	go endpointslicecontroller.NewController(
		ctx.InformerFactory.Core().V1().Pods(),
		ctx.InformerFactory.Core().V1().Services(),
		ctx.InformerFactory.Core().V1().Nodes(),
		ctx.InformerFactory.Discovery().V1beta1().EndpointSlices(),
		ctx.ComponentConfig.EndpointSliceController.MaxEndpointsPerSlice,
		ctx.ClientBuilder.ClientOrDie("endpointslice-controller"),
	).Run(int(ctx.ComponentConfig.EndpointSliceController.ConcurrentServiceEndpointSyncs), ctx.Stop)
	return nil, true, nil
}
