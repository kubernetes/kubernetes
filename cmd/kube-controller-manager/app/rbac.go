/*
Copyright 2017 The Kubernetes Authors.

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

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/controller-manager/controller"
	"k8s.io/kubernetes/pkg/controller/clusterroleaggregation"
)

func startClusterRoleAggregrationController(ctx context.Context, controllerContext ControllerContext) (controller.Interface, bool, error) {
	if !controllerContext.AvailableResources[schema.GroupVersionResource{Group: "rbac.authorization.k8s.io", Version: "v1", Resource: "clusterroles"}] {
		return nil, false, nil
	}
	go clusterroleaggregation.NewClusterRoleAggregation(
		controllerContext.InformerFactory.Rbac().V1().ClusterRoles(),
		controllerContext.ClientBuilder.ClientOrDie("clusterrole-aggregation-controller").RbacV1(),
	).Run(ctx, 5)
	return nil, true, nil
}
