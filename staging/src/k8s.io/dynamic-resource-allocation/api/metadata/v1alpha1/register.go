/*
Copyright The Kubernetes Authors.

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

// GroupName is the group name for DRA device metadata.
const GroupName = "metadata.resource.k8s.io"

// SchemeGroupVersion is the group version for v1alpha1 metadata types.
var SchemeGroupVersion = schema.GroupVersion{Group: GroupName, Version: "v1alpha1"}

var (
	// localSchemeBuilder is used to add types and conversion functions to the scheme.
	// The generated conversion code references this variable by name.
	localSchemeBuilder = runtime.NewSchemeBuilder(addKnownTypes)

	// AddToScheme applies all the stored functions to the scheme.
	AddToScheme = localSchemeBuilder.AddToScheme
)

func addKnownTypes(scheme *runtime.Scheme) error {
	scheme.AddKnownTypes(SchemeGroupVersion,
		&DeviceMetadata{},
	)
	metav1.AddToGroupVersion(scheme, SchemeGroupVersion)
	return nil
}
