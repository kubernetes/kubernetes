/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package api

import (
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
)

// Scheme is the default instance of runtime.Scheme to which types in the Kubernetes API are already registered.
var Scheme = runtime.NewScheme()

// ParameterCodec handles versioning of objects that are converted to query parameters.
var ParameterCodec = runtime.NewParameterCodec(Scheme)

// SchemeGroupVersion is group version used to register these objects
var SchemeGroupVersion = unversioned.GroupVersion{Group: "", Version: runtime.APIVersionInternal}
var Unversioned = unversioned.GroupVersion{Group: "", Version: "v1"}

func init() {
	if err := Scheme.AddIgnoredConversionType(&unversioned.TypeMeta{}, &unversioned.TypeMeta{}); err != nil {
		panic(err)
	}
	Scheme.AddKnownTypes(SchemeGroupVersion,
		&Pod{},
		&PodList{},
		&PodStatusResult{},
		&PodTemplate{},
		&PodTemplateList{},
		&ReplicationControllerList{},
		&ReplicationController{},
		&ServiceList{},
		&Service{},
		&NodeList{},
		&Node{},
		&Endpoints{},
		&EndpointsList{},
		&Binding{},
		&Event{},
		&EventList{},
		&List{},
		&LimitRange{},
		&LimitRangeList{},
		&ResourceQuota{},
		&ResourceQuotaList{},
		&Namespace{},
		&NamespaceList{},
		&ServiceAccount{},
		&ServiceAccountList{},
		&Secret{},
		&SecretList{},
		&PersistentVolume{},
		&PersistentVolumeList{},
		&PersistentVolumeClaim{},
		&PersistentVolumeClaimList{},
		&DeleteOptions{},
		&PodAttachOptions{},
		&PodLogOptions{},
		&PodExecOptions{},
		&PodProxyOptions{},
		&ComponentStatus{},
		&ComponentStatusList{},
		&SerializedReference{},
		&RangeAllocation{},
	)

	// Register Unversioned types under their own special group
	Scheme.AddUnversionedTypes(Unversioned,
		&unversioned.ListOptions{},
		&unversioned.Status{},
		&unversioned.APIVersions{},
		&unversioned.APIGroupList{},
		&unversioned.APIGroup{},
		&unversioned.APIResourceList{},
	)
}

func (obj *Pod) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *PodList) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *PodStatusResult) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *PodTemplate) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *PodTemplateList) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *ReplicationController) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *ReplicationControllerList) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *Service) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *ServiceList) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *Endpoints) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *EndpointsList) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *Node) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *NodeList) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *Binding) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *Event) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *EventList) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *List) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *LimitRange) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *LimitRangeList) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *ResourceQuota) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *ResourceQuotaList) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *Namespace) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *NamespaceList) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *ServiceAccount) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *ServiceAccountList) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *Secret) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *SecretList) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *PersistentVolume) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *PersistentVolumeList) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *PersistentVolumeClaim) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *PersistentVolumeClaimList) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *DeleteOptions) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *ListOptions) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *PodAttachOptions) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *PodLogOptions) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *PodExecOptions) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *PodProxyOptions) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *ComponentStatus) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *ComponentStatusList) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *SerializedReference) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *RangeAllocation) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
