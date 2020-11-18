/*
Copyright 2019 The Kubernetes Authors.

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

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// GroupName is the "group" that is needed to uniquely identify the API
const GroupName = "controllermanager.config.k8s.io"

// SchemeGroupVersion is group version used to register these objects
var SchemeGroupVersion = schema.GroupVersion{Group: GroupName, Version: "v1alpha1"}

var (
	// SchemeBuilder is the scheme builder with scheme init functions to run for this API package
	SchemeBuilder = runtime.NewSchemeBuilder(addKnownTypes)
	// localSchemeBuilder extends the SchemeBuilder instance with the external types. In this package,
	// defaulting and conversion init funcs are registered as well.
	localSchemeBuilder = &SchemeBuilder
	// AddToScheme is a global function that registers this API group & version to a scheme
	AddToScheme = localSchemeBuilder.AddToScheme
)

// Adds the list of known types to the given scheme.
func addKnownTypes(scheme *runtime.Scheme) error {
	scheme.AddKnownTypes(SchemeGroupVersion,
		&LeaderMigrationConfiguration{},
	)
	metav1.AddToGroupVersion(scheme, SchemeGroupVersion)
	return nil
}
