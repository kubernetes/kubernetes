/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

// Package install installs the v1 monolithic api, making it available as an
// option to all of the API encoding/decoding machinery.
package install

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/sets"

	// TODO: Remove from here, add to places where this install package is imported.
	_ "k8s.io/kubernetes/pkg/api/v1/install"
)

func init() {
	if err := registered.AnnounceGroup(&registered.GroupMetaFactory{
		GroupName:              "api",
		VersionPreferenceOrder: []string{"v1"},
		ImportPrefix:           "k8s.io/kubernetes/pkg/api",
		RootScopedKinds: sets.NewString(
			"Node",
			"Namespace",
			"PersistentVolume",
			"ComponentStatus",
		),
		IgnoredKinds: sets.NewString(
			"ListOptions",
			"DeleteOptions",
			"Status",
			"PodLogOptions",
			"PodExecOptions",
			"PodAttachOptions",
			"PodProxyOptions",
			"NodeProxyOptions",
			"ServiceProxyOptions",
			"ThirdPartyResource",
			"ThirdPartyResourceData",
			"ThirdPartyResourceList",
		),
		AddInternalObjectsToScheme: func(scheme *runtime.Scheme) {
			scheme.AddGenericConversionFunc(genericConversions)
			api.AddToScheme(scheme)
		},
	}); err != nil {
		panic(err)
	}
	// setup aliases for groups of resources
	// mapper.AddResourceAlias("all", userResources...)
}

// This is a "fast-path" that avoids reflection for common types. It
// focuses on the objects that are converted the most in the cluster.
// TODO: generate one of these for every external API group - this is
// to prove the impact
func genericConversions(objA, objB interface{}, s conversion.Scope) (bool, error) {
	switch a := objA.(type) {
	case *v1.Pod:
		switch b := objB.(type) {
		case *api.Pod:
			return true, v1.Convert_v1_Pod_To_api_Pod(a, b, s)
		}
	case *api.Pod:
		switch b := objB.(type) {
		case *v1.Pod:
			return true, v1.Convert_api_Pod_To_v1_Pod(a, b, s)
		}

	case *v1.Event:
		switch b := objB.(type) {
		case *api.Event:
			return true, v1.Convert_v1_Event_To_api_Event(a, b, s)
		}
	case *api.Event:
		switch b := objB.(type) {
		case *v1.Event:
			return true, v1.Convert_api_Event_To_v1_Event(a, b, s)
		}

	case *v1.ReplicationController:
		switch b := objB.(type) {
		case *api.ReplicationController:
			return true, v1.Convert_v1_ReplicationController_To_api_ReplicationController(a, b, s)
		}
	case *api.ReplicationController:
		switch b := objB.(type) {
		case *v1.ReplicationController:
			return true, v1.Convert_api_ReplicationController_To_v1_ReplicationController(a, b, s)
		}

	case *v1.Node:
		switch b := objB.(type) {
		case *api.Node:
			return true, v1.Convert_v1_Node_To_api_Node(a, b, s)
		}
	case *api.Node:
		switch b := objB.(type) {
		case *v1.Node:
			return true, v1.Convert_api_Node_To_v1_Node(a, b, s)
		}

	case *v1.Namespace:
		switch b := objB.(type) {
		case *api.Namespace:
			return true, v1.Convert_v1_Namespace_To_api_Namespace(a, b, s)
		}
	case *api.Namespace:
		switch b := objB.(type) {
		case *v1.Namespace:
			return true, v1.Convert_api_Namespace_To_v1_Namespace(a, b, s)
		}

	case *v1.Service:
		switch b := objB.(type) {
		case *api.Service:
			return true, v1.Convert_v1_Service_To_api_Service(a, b, s)
		}
	case *api.Service:
		switch b := objB.(type) {
		case *v1.Service:
			return true, v1.Convert_api_Service_To_v1_Service(a, b, s)
		}

	case *v1.Endpoints:
		switch b := objB.(type) {
		case *api.Endpoints:
			return true, v1.Convert_v1_Endpoints_To_api_Endpoints(a, b, s)
		}
	case *api.Endpoints:
		switch b := objB.(type) {
		case *v1.Endpoints:
			return true, v1.Convert_api_Endpoints_To_v1_Endpoints(a, b, s)
		}
	}
	return false, nil
}
