/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package testclient

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
)

type FakeGenericResources struct {
	Data []byte
	Err  error
	Name string
	Opts api.ListOptions
	Obj  interface{}

	Namespace        string
	Fake             *FakeExperimental
	GroupVersionKind unversioned.GroupVersionKind
}

func (f *FakeGenericResources) List(opts api.ListOptions) ([]byte, error) {
	f.Opts = opts
	return f.Data, f.Err
}

func (f *FakeGenericResources) Get(name string) ([]byte, error) {
	f.Name = name
	return f.Data, f.Err
}

func (f *FakeGenericResources) Create(name string, obj interface{}) ([]byte, error) {
	f.Name = name
	f.Obj = obj
	return f.Data, f.Err
}

func (f *FakeGenericResources) Update(name string, obj interface{}) ([]byte, error) {
	f.Name = name
	f.Obj = obj
	return f.Data, f.Err
}

func (f *FakeGenericResources) Delete(name string) ([]byte, error) {
	f.Name = name
	return f.Data, f.Err
}
