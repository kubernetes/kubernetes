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

package app

import (
	"github.com/golang/glog"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	_ "k8s.io/kubernetes/pkg/apis/autoscaling/install"
	hpastorage "k8s.io/kubernetes/pkg/registry/autoscaling/horizontalpodautoscaler/storage"
)

func installAutoscalingAPIs(g *genericapiserver.GenericAPIServer, optsGetter generic.RESTOptionsGetter) {
	hpaStorage, hpaStatusStorage := hpastorage.NewREST(optsGetter)

	autoscalingResources := map[string]rest.Storage{
		"horizontalpodautoscalers":        hpaStorage,
		"horizontalpodautoscalers/status": hpaStatusStorage,
	}
	autoscalingGroupMeta := api.Registry.GroupOrDie(autoscaling.GroupName)
	apiGroupInfo := genericapiserver.APIGroupInfo{
		GroupMeta: *autoscalingGroupMeta,
		VersionedResourcesStorageMap: map[string]map[string]rest.Storage{
			"v1": autoscalingResources,
		},
		OptionsExternalVersion: &api.Registry.GroupOrDie(api.GroupName).GroupVersion,
		Scheme:                 api.Scheme,
		ParameterCodec:         api.ParameterCodec,
		NegotiatedSerializer:   api.Codecs,
	}
	if err := g.InstallAPIGroup(&apiGroupInfo); err != nil {
		glog.Fatalf("Error in registering group versions: %v", err)
	}
}
