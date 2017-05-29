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
	"os"

	"k8s.io/apimachinery/pkg/apimachinery/announced"
	"k8s.io/apimachinery/pkg/apimachinery/registered"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/kubernetes/pkg/api"
)

// NOTE: the registry, scheme and codecs are created here to allow to install a federation core group
// that is completely independent from the Kubernetes core group. It's only used for the core group itself.
// The other apigroups in the federation apiserver use the Kubernetes registry, scheme and codecs.

// GroupFactoryRegistry is the APIGroupFactoryRegistry (overlaps a bit with Registry, see comments in package for details)
var GroupFactoryRegistry = make(announced.APIGroupFactoryRegistry)

// Registry is an instance of an API registry.  This is an interim step to start removing the idea of a global
// API registry.
var Registry = registered.NewOrDie(os.Getenv("KUBE_API_VERSIONS"))

// Scheme is the default instance of runtime.Scheme to which types in the Kubernetes API are already registered.
var Scheme = runtime.NewScheme()

// Codecs provides access to encoding and decoding for the scheme
var Codecs = serializer.NewCodecFactory(Scheme)

// GroupName is the group name use in this package
const GroupName = ""

// SchemeGroupVersion is group version used to register these objects
var SchemeGroupVersion = schema.GroupVersion{Group: GroupName, Version: runtime.APIVersionInternal}

// Unversioned is group version for unversioned API objects
// TODO: this should be v1 probably
var Unversioned = schema.GroupVersion{Group: "", Version: "v1"}

// ParameterCodec handles versioning of objects that are converted to query parameters.
var ParameterCodec = runtime.NewParameterCodec(Scheme)

// Kind takes an unqualified kind and returns a Group qualified GroupKind
func Kind(kind string) schema.GroupKind {
	return SchemeGroupVersion.WithKind(kind).GroupKind()
}

// Resource takes an unqualified resource and returns a Group qualified GroupResource
func Resource(resource string) schema.GroupResource {
	return SchemeGroupVersion.WithResource(resource).GroupResource()
}

var (
	SchemeBuilder = runtime.NewSchemeBuilder(addKnownTypes)
	AddToScheme   = SchemeBuilder.AddToScheme
)

func addKnownTypes(scheme *runtime.Scheme) error {
	if err := scheme.AddIgnoredConversionType(&metav1.TypeMeta{}, &metav1.TypeMeta{}); err != nil {
		return err
	}
	scheme.AddKnownTypes(SchemeGroupVersion,
		&api.ServiceList{},
		&api.Service{},
		&api.Namespace{},
		&api.NamespaceList{},
		&api.Secret{},
		&api.SecretList{},
		&api.Event{},
		&api.EventList{},
		&api.ConfigMap{},
		&api.ConfigMapList{},
	)

	// Register Unversioned types under their own special group
	scheme.AddUnversionedTypes(Unversioned,
		&metav1.Status{},
		&metav1.APIVersions{},
		&metav1.APIGroupList{},
		&metav1.APIGroup{},
		&metav1.APIResourceList{},
	)
	return nil
}
