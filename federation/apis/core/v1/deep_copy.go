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

package v1

import (
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/runtime"
)

func addDeepCopyFuncs(scheme *runtime.Scheme) {
	if err := scheme.AddGeneratedDeepCopyFuncs(
		v1.DeepCopy_v1_DeleteOptions,
		v1.DeepCopy_v1_ExportOptions,
		v1.DeepCopy_v1_List,
		v1.DeepCopy_v1_ListOptions,
		v1.DeepCopy_v1_ObjectMeta,
		v1.DeepCopy_v1_ObjectReference,
		v1.DeepCopy_v1_OwnerReference,
		v1.DeepCopy_v1_Service,
		v1.DeepCopy_v1_ServiceList,
		v1.DeepCopy_v1_ServicePort,
		v1.DeepCopy_v1_ServiceSpec,
		v1.DeepCopy_v1_ServiceStatus,
	); err != nil {
		// if one of the deep copy functions is malformed, detect it immediately.
		panic(err)
	}
}
