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

package test

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/testapigroup"
	"k8s.io/apimachinery/pkg/apis/testapigroup/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	apiserializer "k8s.io/apimachinery/pkg/runtime/serializer"
)

// List holds a list of objects, which may not be known by the server.
type List struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta

	Items []runtime.Object
}

func TestScheme() (*runtime.Scheme, apiserializer.CodecFactory) {
	internalGV := schema.GroupVersion{Group: "", Version: runtime.APIVersionInternal}
	externalGV := schema.GroupVersion{Group: "", Version: "v1"}
	scheme := runtime.NewScheme()

	scheme.AddKnownTypes(internalGV,
		&testapigroup.Carp{},
		&testapigroup.CarpList{},
		&List{},
	)
	scheme.AddKnownTypes(externalGV,
		&v1.Carp{},
		&v1.CarpList{},
		&List{},
	)
	testapigroup.AddToScheme(scheme)
	v1.AddToScheme(scheme)

	codecs := apiserializer.NewCodecFactory(scheme)
	return scheme, codecs
}
