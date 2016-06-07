/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package rbac

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch/versioned"
)

const GroupName = "rbac.authorization.k8s.io"

// SchemeGroupVersion is group version used to register these objects
var SchemeGroupVersion = unversioned.GroupVersion{Group: GroupName, Version: runtime.APIVersionInternal}

// Kind takes an unqualified kind and returns back a Group qualified GroupKind
func Kind(kind string) unversioned.GroupKind {
	return SchemeGroupVersion.WithKind(kind).GroupKind()
}

// Resource takes an unqualified resource and returns back a Group qualified GroupResource
func Resource(resource string) unversioned.GroupResource {
	return SchemeGroupVersion.WithResource(resource).GroupResource()
}

func AddToScheme(scheme *runtime.Scheme) {
	// Add the API to Scheme.
	addKnownTypes(scheme)
}

// Adds the list of known types to api.Scheme.
func addKnownTypes(scheme *runtime.Scheme) {
	scheme.AddKnownTypes(SchemeGroupVersion,
		&Role{},
		&RoleBinding{},
		&RoleBindingList{},
		&RoleList{},

		&ClusterRole{},
		&ClusterRoleBinding{},
		&ClusterRoleBindingList{},
		&ClusterRoleList{},

		&api.ListOptions{},
		&api.DeleteOptions{},
		&api.ExportOptions{},
	)
	versioned.AddToGroupVersion(scheme, SchemeGroupVersion)
}

func (obj *ClusterRole) GetObjectMeta() meta.Object                       { return &obj.ObjectMeta }
func (obj *ClusterRole) GetObjectKind() unversioned.ObjectKind            { return &obj.TypeMeta }
func (obj *ClusterRoleBinding) GetObjectMeta() meta.Object                { return &obj.ObjectMeta }
func (obj *ClusterRoleBinding) GetObjectKind() unversioned.ObjectKind     { return &obj.TypeMeta }
func (obj *ClusterRoleBindingList) GetObjectKind() unversioned.ObjectKind { return &obj.TypeMeta }
func (obj *ClusterRoleBindingList) GetListMeta() meta.List                { return &obj.ListMeta }
func (obj *ClusterRoleList) GetObjectKind() unversioned.ObjectKind        { return &obj.TypeMeta }
func (obj *ClusterRoleList) GetListMeta() meta.List                       { return &obj.ListMeta }

func (obj *Role) GetObjectMeta() meta.Object                       { return &obj.ObjectMeta }
func (obj *Role) GetObjectKind() unversioned.ObjectKind            { return &obj.TypeMeta }
func (obj *RoleBinding) GetObjectMeta() meta.Object                { return &obj.ObjectMeta }
func (obj *RoleBinding) GetObjectKind() unversioned.ObjectKind     { return &obj.TypeMeta }
func (obj *RoleBindingList) GetObjectKind() unversioned.ObjectKind { return &obj.TypeMeta }
func (obj *RoleBindingList) GetListMeta() meta.List                { return &obj.ListMeta }
func (obj *RoleList) GetObjectKind() unversioned.ObjectKind        { return &obj.TypeMeta }
func (obj *RoleList) GetListMeta() meta.List                       { return &obj.ListMeta }
