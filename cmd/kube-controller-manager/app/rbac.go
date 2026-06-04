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

	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
	"k8s.io/kubernetes/pkg/controller/clusterroleaggregation"
)

func newClusterRoleAggregrationControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:        names.ClusterRoleAggregationController,
		aliases:     []string{"clusterrole-aggregation"},
		constructor: newClusterRoleAggregationController,
	}
}

func newClusterRoleAggregationController(ctx context.Context, controllerContext ControllerContext, controllerName string) (Controller, error) {
	client, err := controllerContext.NewClient("clusterrole-aggregation-controller")
	if err != nil {
		return nil, err
	}

	crac := clusterroleaggregation.NewClusterRoleAggregation(
		controllerContext.InformerFactory.Rbac().V1().ClusterRoles(),
		client.RbacV1(),
	)
	return newControllerLoop(func(ctx context.Context) {
		crac.Run(ctx, 5)
	}, controllerName), nil
}
