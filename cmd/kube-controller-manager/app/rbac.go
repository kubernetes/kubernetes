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
	"net/http"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/controller/clusterroleaggregation"
)

func startClusterRoleAggregrationController(ctx ControllerContext) (http.Handler, bool, error) {
	if !ctx.AvailableResources[schema.GroupVersionResource{Group: "rbac.authorization.k8s.io", Version: "v1", Resource: "clusterroles"}] {
		return nil, false, nil
	}
	go clusterroleaggregation.NewClusterRoleAggregation(
		ctx.InformerFactory.Rbac().V1().ClusterRoles(),
		ctx.ClientBuilder.ClientOrDie("clusterrole-aggregation-controller").RbacV1(),
	).Run(5, ctx.Stop)
	return nil, true, nil
}
