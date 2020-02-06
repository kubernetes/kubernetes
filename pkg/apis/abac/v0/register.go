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

package v0

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/apis/abac"
)

// GroupName is the group name use in this package
const GroupName = "abac.authorization.kubernetes.io"

// SchemeGroupVersion is the API group version used to register abac v0
var SchemeGroupVersion = schema.GroupVersion{Group: GroupName, Version: "v0"}

func init() {
	// TODO: Delete this init function, abac should not have its own scheme.
	utilruntime.Must(addKnownTypes(abac.Scheme))

	utilruntime.Must(RegisterConversions(abac.Scheme))
}

var (
	// SchemeBuilder is the scheme builder with scheme init functions to run for this API package
	// TODO: move SchemeBuilder with zz_generated.deepcopy.go to k8s.io/api.
	SchemeBuilder runtime.SchemeBuilder
	// localSchemeBuilder ïs a pointer to SchemeBuilder instance. Using localSchemeBuilder
	// defaulting and conversion init funcs are registered as well.
	// localSchemeBuilder and AddToScheme will stay in k8s.io/kubernetes.
	localSchemeBuilder = &SchemeBuilder
	// AddToScheme is a common registration function for mapping packaged scoped group & version keys to a scheme
	AddToScheme = localSchemeBuilder.AddToScheme
)

func init() {
	// We only register manually written functions here. The registration of the
	// generated functions takes place in the generated files. The separation
	// makes the code compile even when the generated files are missing.
	localSchemeBuilder.Register(addKnownTypes)
}

func addKnownTypes(scheme *runtime.Scheme) error {
	scheme.AddKnownTypes(SchemeGroupVersion,
		&Policy{},
	)
	return nil
}
