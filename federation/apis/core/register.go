/*
Copyright 2016 The Kubernetes Authors.

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

package core

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/serializer"
)

// Scheme is the default instance of runtime.Scheme to which types in the Kubernetes API are already registered.
var Scheme = runtime.NewScheme()

// Codecs provides access to encoding and decoding for the scheme
var Codecs = serializer.NewCodecFactory(Scheme)

// GroupName is the group name use in this package
const GroupName = ""

// SchemeGroupVersion is group version used to register these objects
var SchemeGroupVersion = unversioned.GroupVersion{Group: GroupName, Version: runtime.APIVersionInternal}

// Unversioned is group version for unversioned API objects
// TODO: this should be v1 probably
var Unversioned = unversioned.GroupVersion{Group: "", Version: "v1"}

// ParameterCodec handles versioning of objects that are converted to query parameters.
var ParameterCodec = runtime.NewParameterCodec(Scheme)

// Kind takes an unqualified kind and returns a Group qualified GroupKind
func Kind(kind string) unversioned.GroupKind {
	return SchemeGroupVersion.WithKind(kind).GroupKind()
}

// Resource takes an unqualified resource and returns a Group qualified GroupResource
func Resource(resource string) unversioned.GroupResource {
	return SchemeGroupVersion.WithResource(resource).GroupResource()
}

var (
	SchemeBuilder = runtime.NewSchemeBuilder(addKnownTypes, addDefaultingFuncs, addConversionFuncs)
	AddToScheme   = SchemeBuilder.AddToScheme
)

func addKnownTypes(scheme *runtime.Scheme) error {
	if err := scheme.AddIgnoredConversionType(&unversioned.TypeMeta{}, &unversioned.TypeMeta{}); err != nil {
		return err
	}
	scheme.AddKnownTypes(SchemeGroupVersion,
		&api.ServiceList{},
		&api.Service{},
		&api.Namespace{},
		&api.NamespaceList{},
		&api.ListOptions{},
		&api.DeleteOptions{},
		&api.Secret{},
		&api.SecretList{},
		&api.Event{},
		&api.EventList{},
		&api.ConfigMap{},
		&api.ConfigMapList{},
	)

	// Register Unversioned types under their own special group
	scheme.AddUnversionedTypes(Unversioned,
		&unversioned.ExportOptions{},
		&unversioned.Status{},
		&unversioned.APIVersions{},
		&unversioned.APIGroupList{},
		&unversioned.APIGroup{},
		&unversioned.APIResourceList{},
	)
	return nil
}
