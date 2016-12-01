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

	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/cache"
	pkgruntime "k8s.io/kubernetes/pkg/runtime"
)

// Returns cache.ResourceEventHandlerFuncs that trigger the given function
// on all object changes.
func NewTriggerOnAllChanges(triggerFunc func(pkgruntime.Object)) *cache.ResourceEventHandlerFuncs {
	return &cache.ResourceEventHandlerFuncs{
		DeleteFunc: func(old interface{}) {
			oldObj := old.(pkgruntime.Object)
			triggerFunc(oldObj)
		},
		AddFunc: func(cur interface{}) {
			curObj := cur.(pkgruntime.Object)
			triggerFunc(curObj)
		},
		UpdateFunc: func(old, cur interface{}) {
			curObj := cur.(pkgruntime.Object)
			if !reflect.DeepEqual(old, cur) {
				triggerFunc(curObj)
			}
		},
	}
}

// Returns cache.ResourceEventHandlerFuncs that trigger the given function
// on object add and delete as well as spec/object meta on update.
func NewTriggerOnMetaAndSpecChanges(triggerFunc func(pkgruntime.Object)) *cache.ResourceEventHandlerFuncs {
	getFieldOrPanic := func(obj interface{}, fieldName string) interface{} {
		val := reflect.ValueOf(obj).Elem().FieldByName(fieldName)
		if val.IsValid() {
			return val.Interface()
		} else {
			panic(fmt.Errorf("field not found: %s", fieldName))
		}
	}
	return &cache.ResourceEventHandlerFuncs{
		DeleteFunc: func(old interface{}) {
			oldObj := old.(pkgruntime.Object)
			triggerFunc(oldObj)
		},
		AddFunc: func(cur interface{}) {
			curObj := cur.(pkgruntime.Object)
			triggerFunc(curObj)
		},
		UpdateFunc: func(old, cur interface{}) {
			curObj := cur.(pkgruntime.Object)
			oldMeta := getFieldOrPanic(old, "ObjectMeta").(apiv1.ObjectMeta)
			curMeta := getFieldOrPanic(cur, "ObjectMeta").(apiv1.ObjectMeta)
			if !ObjectMetaEquivalent(oldMeta, curMeta) ||
				!reflect.DeepEqual(getFieldOrPanic(old, "Spec"), getFieldOrPanic(cur, "Spec")) {
				triggerFunc(curObj)
			}
		},
	}
}
