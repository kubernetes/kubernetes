/*
Copyright 2023 The Kubernetes Authors.

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

	"k8s.io/component-base/featuregate"
	"k8s.io/kubernetes/pkg/controller/servicecidrs"
	"k8s.io/kubernetes/pkg/features"

	"k8s.io/kubernetes/cmd/kube-controller-manager/internal/controller"
	"k8s.io/kubernetes/cmd/kube-controller-manager/internal/controller/run"
	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
)

func newServiceCIDRsControllerDescriptor() *controller.Descriptor {
	return &controller.Descriptor{
		Name:        names.ServiceCIDRController,
		Constructor: newServiceCIDRsController,
		RequiredFeatureGates: []featuregate.Feature{
			features.MultiCIDRServiceAllocator,
		},
	}
}

func newServiceCIDRsController(ctx context.Context, controllerContext controller.Context, controllerName string) (controller.Controller, error) {
	client, err := controllerContext.NewClient("service-cidrs-controller")
	if err != nil {
		return nil, err
	}

	// TODO use component config
	scc := servicecidrs.NewController(
		ctx,
		controllerContext.InformerFactory.Networking().V1().ServiceCIDRs(),
		controllerContext.InformerFactory.Networking().V1().IPAddresses(),
		client,
	)
	return run.NewControllerLoop(func(ctx context.Context) {
		scc.Run(ctx, 5)
	}, controllerName), nil
}
