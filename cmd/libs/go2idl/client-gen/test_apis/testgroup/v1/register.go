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

package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/api/v1"
)

var SchemeGroupVersion = schema.GroupVersion{Group: "testgroup.k8s.io", Version: "v1"}

var (
	SchemeBuilder = runtime.NewSchemeBuilder(addKnownTypes)
	AddToScheme   = SchemeBuilder.AddToScheme
)

// Adds the list of known types to api.Scheme.
func addKnownTypes(scheme *runtime.Scheme) error {
	scheme.AddKnownTypes(SchemeGroupVersion,
		&TestType{},
		&TestTypeList{},
	)

	scheme.AddKnownTypes(SchemeGroupVersion,
		&v1.ListOptions{},
		&v1.DeleteOptions{},
		&metav1.Status{},
		&metav1.ExportOptions{},
	)
	metav1.AddToGroupVersion(scheme, SchemeGroupVersion)
	return nil
}

func (obj *TestType) GetObjectKind() schema.ObjectKind     { return &obj.TypeMeta }
func (obj *TestTypeList) GetObjectKind() schema.ObjectKind { return &obj.TypeMeta }
