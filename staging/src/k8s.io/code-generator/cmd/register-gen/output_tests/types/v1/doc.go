/*
Copyright 2024 The Kubernetes Authors.

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

// +k8s:register-gen=package

// This is a test package
package v1

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

type Foo struct {
	TypeMeta `json:"inline"`
	X        int `json:"x,omitempty"`
}

func (in *Foo) DeepCopyObject() runtime.Object {
	panic("not implemented")
}

func (in *Foo) GetObjectKind() schema.ObjectKind {
	panic("not implemented")
}

type Bar struct {
	TypeMeta `json:"inline"`
	Y        string `json:"y,omitempty"`
}

func (in *Bar) DeepCopyObject() runtime.Object {
	panic("not implemented")
}

func (in *Bar) GetObjectKind() schema.ObjectKind {
	panic("not implemented")
}

type TypeMeta int
