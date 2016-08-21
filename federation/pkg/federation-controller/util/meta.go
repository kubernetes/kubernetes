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

package util

import (
	"reflect"

	api_v1 "k8s.io/kubernetes/pkg/api/v1"
)

// Copies cluster-independent, user provided data from the given ObjectMeta struct. If in
// the future the ObjectMeta structure is expanded then any field that is not populared
// by the api server should be included here.
func CopyObjectMeta(obj api_v1.ObjectMeta) api_v1.ObjectMeta {
	return api_v1.ObjectMeta{
		Name:        obj.Name,
		Namespace:   obj.Namespace,
		Labels:      obj.Labels,
		Annotations: obj.Annotations,
	}
}

// Checks if cluster-independed, user provided data in two given ObjectMeta are eqaul. If in
// the future the ObjectMeta structure is expanded then any field that is not populared
// by the api server should be included here.
func ObjectMetaEquivalent(a, b api_v1.ObjectMeta) bool {
	if a.Name != b.Name {
		return false
	}
	if a.Namespace != b.Namespace {
		return false
	}
	if !reflect.DeepEqual(a.Labels, b.Labels) {
		return false
	}
	if !reflect.DeepEqual(a.Annotations, b.Annotations) {
		return false
	}
	return true
}
