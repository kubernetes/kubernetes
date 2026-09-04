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

	"k8s.io/client-go/dynamic"
	"k8s.io/component-base/featuregate"
	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
	"k8s.io/kubernetes/pkg/controller/podcheckpoint"
	"k8s.io/kubernetes/pkg/features"
)

func newPodCheckpointControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:                 names.PodCheckpointController,
		constructor:          newPodCheckpointController,
		requiredFeatureGates: []featuregate.Feature{features.PodLevelCheckpointRestore},
	}
}

func newPodCheckpointController(ctx context.Context, controllerContext ControllerContext, controllerName string) (Controller, error) {
	config, err := controllerContext.NewClientConfig("podcheckpoint-controller")
	if err != nil {
		return nil, err
	}

	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		return nil, err
	}

	pc := podcheckpoint.NewController(dynamicClient, controllerContext.InformerFactory.Core().V1().Pods())

	return newControllerLoop(func(ctx context.Context) {
		pc.Run(ctx, 1)
	}, controllerName), nil
}
