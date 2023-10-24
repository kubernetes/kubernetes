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

package app

import (
	"context"

	"k8s.io/controller-manager/controller"
	"k8s.io/kubernetes/pkg/controller/leaderelection"
)

func startLeaderElectionController(ctx context.Context, controllerContext ControllerContext) (controller.Interface, bool, error) {
	c, err := leaderelection.NewController(
		controllerContext.InformerFactory.Coordination().V1().Leases(),
		controllerContext.ClientBuilder.ClientOrDie("leader-election-controller").CoordinationV1(),
	)
	if err != nil {
		return nil, false, err
	}
	go c.Run(ctx, int(controllerContext.ComponentConfig.LeaderElectionController.ConcurrentLeaseSyncs))
	return nil, true, nil
}
