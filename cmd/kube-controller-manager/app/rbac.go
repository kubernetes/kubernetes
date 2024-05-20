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

	"k8s.io/controller-manager/controller"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
	"k8s.io/kubernetes/pkg/controller/clusterroleaggregation"
)

func newClusterRoleAggregrationControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:     names.ClusterRoleAggregationController,
		aliases:  []string{"clusterrole-aggregation"},
		initFunc: startClusterRoleAggregationController,
	}
}

func startClusterRoleAggregationController(ctx context.Context, controllerContext ControllerContext, controllerName string) (controller.Interface, bool, error) {
	logger := klog.FromContext(ctx)
	go clusterroleaggregation.NewClusterRoleAggregation(
		controllerContext.InformerFactory.Rbac().V1().ClusterRoles(),
		controllerContext.ClientBuilder.ClientOrDie(logger, "clusterrole-aggregation-controller").RbacV1(),
	).Run(ctx, 5)
	return nil, true, nil
}
