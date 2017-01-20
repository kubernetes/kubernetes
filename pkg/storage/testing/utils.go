/*
Copyright 2015 The Kubernetes Authors.

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

package testing

import (
	"path"

	"golang.org/x/net/context"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
)

// CreateObj will create a single object using the storage interface
func CreateObj(helper storage.Interface, name string, obj, out runtime.Object, ttl uint64) error {
	return helper.Create(context.TODO(), name, obj, out, ttl)
}

//CreateObjList will create a list from the array of objects
func CreateObjList(prefix string, helper storage.Interface, items []runtime.Object) error {
	for i := range items {
		obj := items[i]
		meta, err := meta.Accessor(obj)
		if err != nil {
			return err
		}
		err = CreateObj(helper, path.Join(prefix, meta.GetName()), obj, obj, 0)
		if err != nil {
			return err
		}
		items[i] = obj
	}
	return nil
}

// CreateList will properly create a list using the storage interface
func CreateList(prefix string, helper storage.Interface, list runtime.Object) error {
	items, err := meta.ExtractList(list)
	if err != nil {
		return err
	}
	err = CreateObjList(prefix, helper, items)
	if err != nil {
		return err
	}
	return meta.SetList(list, items)
}
