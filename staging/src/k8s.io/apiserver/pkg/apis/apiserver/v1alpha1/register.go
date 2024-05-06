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

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

const GroupName = "apiserver.k8s.io"
const ConfigGroupName = "apiserver.config.k8s.io"

// SchemeGroupVersion is group version used to register these objects
var SchemeGroupVersion = schema.GroupVersion{Group: GroupName, Version: "v1alpha1"}

// ConfigSchemeGroupVersion is group version used to register these objects
var ConfigSchemeGroupVersion = schema.GroupVersion{Group: ConfigGroupName, Version: "v1alpha1"}

var (
	// TODO: move SchemeBuilder with zz_generated.deepcopy.go to k8s.io/api.
	// localSchemeBuilder and AddToScheme will stay in k8s.io/kubernetes.
	SchemeBuilder      runtime.SchemeBuilder
	localSchemeBuilder = &SchemeBuilder
	AddToScheme        = localSchemeBuilder.AddToScheme
)

func init() {
	// We only register manually written functions here. The registration of the
	// generated functions takes place in the generated files. The separation
	// makes the code compile even when the generated files are missing.
	localSchemeBuilder.Register(addKnownTypes, addDefaultingFuncs)
}

// Adds the list of known types to the given scheme.
func addKnownTypes(scheme *runtime.Scheme) error {
	scheme.AddKnownTypes(SchemeGroupVersion,
		&AdmissionConfiguration{},
		&EgressSelectorConfiguration{},
	)
	scheme.AddKnownTypes(ConfigSchemeGroupVersion,
		&AuthenticationConfiguration{},
		&AuthorizationConfiguration{},
		&TracingConfiguration{},
	)
	metav1.AddToGroupVersion(scheme, SchemeGroupVersion)
	metav1.AddToGroupVersion(scheme, ConfigSchemeGroupVersion)
	return nil
}
