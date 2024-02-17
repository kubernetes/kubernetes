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

package v1beta2

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/version"
)

// GroupName is the name of api group
const GroupName = "flowcontrol.apiserver.k8s.io"

// SchemeGroupVersion is group version used to register these objects
var SchemeGroupVersion = schema.GroupVersion{Group: GroupName, Version: "v1beta2"}

// Kind takes an unqualified kind and returns a Group qualified GroupKind
func Kind(kind string) schema.GroupKind {
	return SchemeGroupVersion.WithKind(kind).GroupKind()
}

// Resource takes an unqualified resource and returns a Group qualified GroupResource
func Resource(resource string) schema.GroupResource {
	return SchemeGroupVersion.WithResource(resource).GroupResource()
}

var (
	// SchemeBuilder installs the api group to a scheme
	SchemeBuilder = runtime.NewSchemeBuilder(addKnownTypes)
	// AddToScheme adds api to a scheme
	AddToScheme = SchemeBuilder.AddToScheme
)

// Adds the list of known types to the given scheme.
// Do not remove before feature EmulationVersion graduates. Types used in integration tests TestEnableEmulationVersion.
func addKnownTypes(scheme *runtime.Scheme) error {
	scheme.AddKnownTypes(SchemeGroupVersion,
		&FlowSchema{},
		&FlowSchemaList{},
		&PriorityLevelConfiguration{},
		&PriorityLevelConfigurationList{},
	)
	metav1.AddToGroupVersion(scheme, SchemeGroupVersion)

	// Registers the lifecycle of the group version, which is checked to make sure a gvr is not available before its type is introduced or after it is removed.
	// All individual resource types of this group share the lifecycle of the group version and
	// do not require their own lifecycles to be specified, like: scheme.SetResourceLifecycle(SchemeGroupVersion.WithResource("flowschema"), &FlowSchema{})
	scheme.SetGroupVersionLifecycle(SchemeGroupVersion, schema.APILifecycle{
		IntroducedVersion: version.MajorMinor(1, 23),
		RemovedVersion:    version.MajorMinor(1, 29),
	})

	return nil
}
