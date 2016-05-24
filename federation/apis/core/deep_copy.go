/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package core

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/runtime"
)

func addDeepCopyFuncs(scheme *runtime.Scheme) {
	if err := scheme.AddGeneratedDeepCopyFuncs(
		api.DeepCopy_api_DeleteOptions,
		api.DeepCopy_api_ExportOptions,
		api.DeepCopy_api_List,
		api.DeepCopy_api_ListOptions,
		api.DeepCopy_api_ObjectMeta,
		api.DeepCopy_api_ObjectReference,
		api.DeepCopy_api_OwnerReference,
		api.DeepCopy_api_Service,
		api.DeepCopy_api_ServiceList,
		api.DeepCopy_api_ServicePort,
		api.DeepCopy_api_ServiceSpec,
		api.DeepCopy_api_ServiceStatus,
	); err != nil {
		// if one of the deep copy functions is malformed, detect it immediately.
		panic(err)
	}
}
