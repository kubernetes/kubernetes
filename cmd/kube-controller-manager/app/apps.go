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
	"k8s.io/apimachinery/pkg/runtime/schema"
	petset "k8s.io/kubernetes/pkg/controller/petset"
)

func startStatefulSetController(ctx ControllerContext) (bool, error) {
	if !ctx.AvailableResources[schema.GroupVersionResource{Group: "apps", Version: "v1beta1", Resource: "statefulsets"}] {
		return false, nil
	}
	resyncPeriod := ResyncPeriod(&ctx.Options)()
	go petset.NewStatefulSetController(
		ctx.InformerFactory.Pods().Informer(),
		ctx.ClientBuilder.ClientOrDie("statefulset-controller"),
		resyncPeriod,
	).Run(1, ctx.Stop)
	return true, nil
}
