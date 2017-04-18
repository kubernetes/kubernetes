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

package schedulingpolicy

import (
	"reflect"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
)

func mergeAnnotations(obj runtime.Object, annotations map[string]string) {

	if len(annotations) == 0 {
		return
	}

	val := reflect.Indirect(reflect.ValueOf(obj))
	annotationsFld, ok := getAnnotationsField(val)
	if !ok {
		return
	}

	orig := annotationsFld.Interface()
	if orig == nil {
		orig = map[string]string{}
	}

	origMap := orig.(map[string]string)

	for k := range origMap {
		if _, ok := annotations[k]; !ok {
			annotations[k] = origMap[k]
		}
	}

	annotationsFld.Set(reflect.ValueOf(annotations))
}

func getAnnotationsField(val reflect.Value) (reflect.Value, bool) {
	metadataFld, ok := getFieldByName(val, "ObjectMeta")
	if !ok {
		return reflect.Value{}, false
	}
	return getFieldByJSONTag(metadataFld, "annotations")
}

func getFieldByName(obj reflect.Value, field string) (val reflect.Value, ok bool) {

	if obj.Kind() == reflect.Ptr {
		obj = reflect.Indirect(obj)
	}

	val = obj.FieldByName(field)
	if val.IsValid() {
		return val, true
	}

	return reflect.Value{}, false
}

func getFieldByJSONTag(obj reflect.Value, field string) (val reflect.Value, ok bool) {

	tpe := obj.Type()

	if obj.Kind() == reflect.Ptr {
		obj = reflect.Indirect(obj)
		tpe = obj.Type()
	}

	for i := 0; i < tpe.NumField(); i++ {
		fld := tpe.Field(i)
		for _, s := range strings.Split(fld.Tag.Get("json"), ",") {
			if s == field {
				return obj.FieldByName(fld.Name), true
			}
		}
	}

	return reflect.Zero(tpe), false
}
