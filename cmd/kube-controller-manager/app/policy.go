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
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/scale"
	"k8s.io/kubernetes/pkg/controller/disruption"

	"net/http"

	"k8s.io/klog"
)

func startDisruptionController(ctx ControllerContext) (http.Handler, bool, error) {
	var group = "policy"
	var version = "v1beta1"
	var resource = "poddisruptionbudgets"

	if !ctx.AvailableResources[schema.GroupVersionResource{Group: group, Version: version, Resource: resource}] {
		klog.Infof(
			"Refusing to start disruption because resource %q in group %q is not available.",
			resource, group+"/"+version)
		return nil, false, nil
	}

	client := ctx.ClientBuilder.ClientOrDie("disruption-controller")
	config := ctx.ClientBuilder.ConfigOrDie("disruption-controller")
	scaleKindResolver := scale.NewDiscoveryScaleKindResolver(client.Discovery())
	scaleClient, err := scale.NewForConfig(config, ctx.RESTMapper, dynamic.LegacyAPIPathResolverFunc, scaleKindResolver)
	if err != nil {
		return nil, false, err
	}

	go disruption.NewDisruptionController(
		ctx.InformerFactory.Core().V1().Pods(),
		ctx.InformerFactory.Policy().V1beta1().PodDisruptionBudgets(),
		ctx.InformerFactory.Core().V1().ReplicationControllers(),
		ctx.InformerFactory.Apps().V1().ReplicaSets(),
		ctx.InformerFactory.Apps().V1().Deployments(),
		ctx.InformerFactory.Apps().V1().StatefulSets(),
		client,
		ctx.RESTMapper,
		scaleClient,
	).Run(ctx.Stop)
	return nil, true, nil
}
