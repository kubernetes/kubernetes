/*
Copyright 2014 The Kubernetes Authors.

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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/plugin/pkg/scheduler/api"
)

// SchemeGroupVersion is group version used to register these objects
// TODO this should be in the "scheduler" group
var SchemeGroupVersion = schema.GroupVersion{Group: "", Version: "v1"}

func init() {
	if err := addKnownTypes(api.Scheme); err != nil {
		// Programmer error.
		panic(err)
	}
}

var (
	SchemeBuilder = runtime.NewSchemeBuilder(addKnownTypes)
	AddToScheme   = SchemeBuilder.AddToScheme
)

func addKnownTypes(scheme *runtime.Scheme) error {
	scheme.AddKnownTypes(SchemeGroupVersion,
		&Policy{},
	)
	return nil
}

func (obj *Policy) GetObjectKind() schema.ObjectKind { return &obj.TypeMeta }
