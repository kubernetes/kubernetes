/*
Copyright The Kubernetes Authors.

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

package app

import (
	"context"

	"k8s.io/component-base/featuregate"
	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
	"k8s.io/kubernetes/pkg/controller/imperativeevictioninterceptor"
	"k8s.io/kubernetes/pkg/features"
)

func newImperativeEvictionInterceptorControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:        names.ImperativeEvictionInterceptorController,
		constructor: newImperativeEvictionInterceptorController,
		requiredFeatureGates: []featuregate.Feature{
			features.EvictionRequestAPI,
		},
	}
}

func newImperativeEvictionInterceptorController(ctx context.Context, controllerContext ControllerContext, controllerName string) (Controller, error) {
	client, err := controllerContext.NewClient(controllerName)
	if err != nil {
		return nil, err
	}

	evictionRequestController, err := imperativeevictioninterceptor.NewController(
		ctx,
		controllerName,
		controllerContext.InformerFactory.Coordination().V1alpha1().EvictionRequests(),
		controllerContext.InformerFactory.Core().V1().Pods(),
		client,
	)
	if err != nil {
		return nil, err
	}

	return newControllerLoop(func(ctx context.Context) {
		evictionRequestController.Run(ctx, int(controllerContext.ComponentConfig.ImperativeEvictionInterceptorController.ConcurrentImperativeEvictionInterceptorControllerSyncs))
	}, controllerName), nil
}
