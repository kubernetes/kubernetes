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
	"fmt"
	"reflect"

	"k8s.io/kubernetes/pkg/controller/framework"
	pkg_runtime "k8s.io/kubernetes/pkg/runtime"
)

// Returns framework.ResourceEventHandlerFuncs that trigger the given function
// on all object changes. Preproc perprocessing function is executed before each trigger and chec.
func NewTriggerOnAllChangesPreproc(triggerFunc func(pkg_runtime.Object), preproc func(obj pkg_runtime.Object)) *framework.ResourceEventHandlerFuncs {
	return &framework.ResourceEventHandlerFuncs{
		DeleteFunc: func(old interface{}) {
			oldObj := old.(pkg_runtime.Object)
			preproc(oldObj)
			triggerFunc(oldObj)
		},
		AddFunc: func(cur interface{}) {
			curObj := cur.(pkg_runtime.Object)
			preproc(curObj)
			triggerFunc(curObj)
		},
		UpdateFunc: func(old, cur interface{}) {
			curObj := cur.(pkg_runtime.Object)
			oldObj := cur.(pkg_runtime.Object)
			preproc(curObj)
			preproc(oldObj)
			if !reflect.DeepEqual(old, cur) {
				triggerFunc(curObj)
			}
		},
	}
}

func NewTriggerOnAllChanges(triggerFunc func(pkg_runtime.Object)) *framework.ResourceEventHandlerFuncs {
	return NewTriggerOnAllChangesPreproc(triggerFunc, func(obj pkg_runtime.Object) {})
}

// Returns framework.ResourceEventHandlerFuncs that trigger the given function
// on object add and delete as well as spec/object meta on update. Preproc preprocessing is executed
// before each trigger and check.
func NewTriggerOnMetaAndSpecChangesPreproc(triggerFunc func(pkg_runtime.Object), preproc func(obj pkg_runtime.Object)) *framework.ResourceEventHandlerFuncs {
	getFieldOrPanic := func(obj interface{}, fieldName string) interface{} {
		val := reflect.ValueOf(obj).Elem().FieldByName(fieldName)
		if val.IsValid() {
			return val.Interface()
		} else {
			panic(fmt.Errorf("field not found: %s", fieldName))
		}
	}
	return &framework.ResourceEventHandlerFuncs{
		DeleteFunc: func(old interface{}) {
			oldObj := old.(pkg_runtime.Object)
			preproc(oldObj)
			triggerFunc(oldObj)
		},
		AddFunc: func(cur interface{}) {
			curObj := cur.(pkg_runtime.Object)
			preproc(curObj)
			triggerFunc(curObj)
		},
		UpdateFunc: func(old, cur interface{}) {
			curObj := cur.(pkg_runtime.Object)
			oldObj := cur.(pkg_runtime.Object)
			preproc(curObj)
			preproc(oldObj)
			if !reflect.DeepEqual(getFieldOrPanic(old, "ObjectMeta"), getFieldOrPanic(cur, "ObjectMeta")) ||
				!reflect.DeepEqual(getFieldOrPanic(old, "Spec"), getFieldOrPanic(cur, "Spec")) {
				triggerFunc(curObj)
			}
		},
	}
}

func TriggerOnMetaAndSpecChanges(triggerFunc func(pkg_runtime.Object)) *framework.ResourceEventHandlerFuncs {
	return NewTriggerOnAllChangesPreproc(triggerFunc, func(obj pkg_runtime.Object) {})
}
