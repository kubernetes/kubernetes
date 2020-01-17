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

package internalversion

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1beta1 "k8s.io/apimachinery/pkg/apis/meta/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// GroupName is the group name for this API.
const GroupName = "meta.k8s.io"

var (
	// TODO: move SchemeBuilder with zz_generated.deepcopy.go to k8s.io/api.
	// localSchemeBuilder and AddToScheme will stay in k8s.io/kubernetes.
	SchemeBuilder      runtime.SchemeBuilder
	localSchemeBuilder = &SchemeBuilder
	AddToScheme        = localSchemeBuilder.AddToScheme
)

// SchemeGroupVersion is group version used to register these objects
var SchemeGroupVersion = schema.GroupVersion{Group: GroupName, Version: runtime.APIVersionInternal}

// Kind takes an unqualified kind and returns a Group qualified GroupKind
func Kind(kind string) schema.GroupKind {
	return SchemeGroupVersion.WithKind(kind).GroupKind()
}

// addToGroupVersion registers common meta types into schemas.
func addToGroupVersion(scheme *runtime.Scheme) error {
	if err := scheme.AddIgnoredConversionType(&metav1.TypeMeta{}, &metav1.TypeMeta{}); err != nil {
		return err
	}
	// ListOptions is the only options struct which needs conversion (it exposes labels and fields
	// as selectors for convenience). The other types have only a single representation today.
	scheme.AddKnownTypes(SchemeGroupVersion,
		&ListOptions{},
		&metav1.GetOptions{},
		&metav1.ExportOptions{},
		&metav1.DeleteOptions{},
		&metav1.CreateOptions{},
		&metav1.UpdateOptions{},
	)
	scheme.AddKnownTypes(SchemeGroupVersion,
		&metav1.Table{},
		&metav1.TableOptions{},
		&metav1beta1.PartialObjectMetadata{},
		&metav1beta1.PartialObjectMetadataList{},
	)
	if err := metav1beta1.AddMetaToScheme(scheme); err != nil {
		return err
	}
	if err := metav1.AddMetaToScheme(scheme); err != nil {
		return err
	}
	// Allow delete options to be decoded across all version in this scheme (we may want to be more clever than this)
	scheme.AddUnversionedTypes(SchemeGroupVersion,
		&metav1.DeleteOptions{},
		&metav1.CreateOptions{},
		&metav1.UpdateOptions{})

	metav1.AddToGroupVersion(scheme, metav1.SchemeGroupVersion)
	return nil
}

// Unlike other API groups, meta internal knows about all meta external versions, but keeps
// the logic for conversion private.
func init() {
	localSchemeBuilder.Register(addToGroupVersion)
}
