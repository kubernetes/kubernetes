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

	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/scale"
	"k8s.io/controller-manager/controller"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
	"k8s.io/kubernetes/pkg/controller/disruption"
)

func newDisruptionControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:     names.DisruptionController,
		aliases:  []string{"disruption"},
		initFunc: startDisruptionController,
	}
}

func startDisruptionController(ctx context.Context, controllerContext ControllerContext, controllerName string) (controller.Interface, bool, error) {
	logger := klog.FromContext(ctx)
	client := controllerContext.ClientBuilder.ClientOrDie(logger, "disruption-controller")
	config := controllerContext.ClientBuilder.ConfigOrDie(logger, "disruption-controller")
	scaleKindResolver := scale.NewDiscoveryScaleKindResolver(client.Discovery())
	scaleClient, err := scale.NewForConfig(config, controllerContext.RESTMapper, dynamic.LegacyAPIPathResolverFunc, scaleKindResolver)
	if err != nil {
		return nil, false, err
	}

	go disruption.NewDisruptionController(
		ctx,
		controllerContext.InformerFactory.Core().V1().Pods(),
		controllerContext.InformerFactory.Policy().V1().PodDisruptionBudgets(),
		controllerContext.InformerFactory.Core().V1().ReplicationControllers(),
		controllerContext.InformerFactory.Apps().V1().ReplicaSets(),
		controllerContext.InformerFactory.Apps().V1().Deployments(),
		controllerContext.InformerFactory.Apps().V1().StatefulSets(),
		client,
		controllerContext.RESTMapper,
		scaleClient,
		client.Discovery(),
	).Run(ctx)
	return nil, true, nil
}
