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
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"
)

// Copies cluster-independent, user provided data from the given ObjectMeta struct. If in
// the future the ObjectMeta structure is expanded then any field that is not populated
// by the api server should be included here.
func copyObjectMeta(obj api_v1.ObjectMeta) api_v1.ObjectMeta {
	return api_v1.ObjectMeta{
		Name:        obj.Name,
		Namespace:   obj.Namespace,
		Labels:      obj.Labels,
		Annotations: obj.Annotations,
	}
}

// Deep copies cluster-independent, user provided data from the given ObjectMeta struct. If in
// the future the ObjectMeta structure is expanded then any field that is not populated
// by the api server should be included here.
func DeepCopyRelevantObjectMeta(obj api_v1.ObjectMeta) api_v1.ObjectMeta {
	copyMeta := copyObjectMeta(obj)
	if obj.Labels != nil {
		copyMeta.Labels = make(map[string]string)
		for key, val := range obj.Labels {
			copyMeta.Labels[key] = val
		}
	}
	if obj.Annotations != nil {
		copyMeta.Annotations = make(map[string]string)
		for key, val := range obj.Annotations {
			copyMeta.Annotations[key] = val
		}
	}
	return copyMeta
}

// Checks if cluster-independent, user provided data in two given ObjectMeta are equal. If in
// the future the ObjectMeta structure is expanded then any field that is not populated
// by the api server should be included here.
func ObjectMetaEquivalent(a, b api_v1.ObjectMeta) bool {
	if a.Name != b.Name {
		return false
	}
	if a.Namespace != b.Namespace {
		return false
	}
	if !reflect.DeepEqual(a.Labels, b.Labels) && (len(a.Labels) != 0 || len(b.Labels) != 0) {
		return false
	}
	if !reflect.DeepEqual(a.Annotations, b.Annotations) && (len(a.Annotations) != 0 || len(b.Annotations) != 0) {
		return false
	}
	return true
}

// Checks if cluster-independent, user provided data in ObjectMeta and Spec in two given top
// level api objects are equivalent.
func ObjectMetaAndSpecEquivalent(a, b runtime.Object) bool {
	objectMetaA := reflect.ValueOf(a).Elem().FieldByName("ObjectMeta").Interface().(api_v1.ObjectMeta)
	objectMetaB := reflect.ValueOf(b).Elem().FieldByName("ObjectMeta").Interface().(api_v1.ObjectMeta)
	specA := reflect.ValueOf(a).Elem().FieldByName("Spec").Interface()
	specB := reflect.ValueOf(b).Elem().FieldByName("Spec").Interface()
	return ObjectMetaEquivalent(objectMetaA, objectMetaB) && reflect.DeepEqual(specA, specB)
}

func DeepCopyApiTypeOrPanic(item interface{}) interface{} {
	result, err := conversion.NewCloner().DeepCopy(item)
	if err != nil {
		panic(err)
	}
	return result
}
