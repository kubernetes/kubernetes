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
	"k8s.io/kubernetes/pkg/api/meta"
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

// Unversiond is group version for unversioned API objects
// TODO: this should be v1 probably
var Unversioned = unversioned.GroupVersion{Group: "", Version: "v1"}

// ParameterCodec handles versioning of objects that are converted to query parameters.
var ParameterCodec = runtime.NewParameterCodec(Scheme)

// Kind takes an unqualified kind and returns back a Group qualified GroupKind
func Kind(kind string) unversioned.GroupKind {
	return SchemeGroupVersion.WithKind(kind).GroupKind()
}

// Resource takes an unqualified resource and returns back a Group qualified GroupResource
func Resource(resource string) unversioned.GroupResource {
	return SchemeGroupVersion.WithResource(resource).GroupResource()
}

func AddToScheme(scheme *runtime.Scheme) {
	if err := Scheme.AddIgnoredConversionType(&unversioned.TypeMeta{}, &unversioned.TypeMeta{}); err != nil {
		panic(err)
	}
	scheme.AddKnownTypes(SchemeGroupVersion,
		&Pod{},
		&PodList{},
		&PodStatusResult{},
		&PodTemplate{},
		&PodTemplateList{},
		&ReplicationControllerList{},
		&ReplicationController{},
		&ServiceList{},
		&Service{},
		&ServiceProxyOptions{},
		&NodeList{},
		&Node{},
		&NodeProxyOptions{},
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
		&ListOptions{},
		&PodAttachOptions{},
		&PodLogOptions{},
		&PodExecOptions{},
		&PodProxyOptions{},
		&ComponentStatus{},
		&ComponentStatusList{},
		&SerializedReference{},
		&RangeAllocation{},
		&ConfigMap{},
		&ConfigMapList{},
	)

	// Register Unversioned types under their own special group
	Scheme.AddUnversionedTypes(Unversioned,
		&unversioned.ExportOptions{},
		&unversioned.Status{},
		&unversioned.APIVersions{},
		&unversioned.APIGroupList{},
		&unversioned.APIGroup{},
		&unversioned.APIResourceList{},
	)
}

func (obj *Pod) GetObjectMeta() meta.Object                                  { return &obj.ObjectMeta }
func (obj *Pod) GetObjectKind() unversioned.ObjectKind                       { return &obj.TypeMeta }
func (obj *PodList) GetObjectKind() unversioned.ObjectKind                   { return &obj.TypeMeta }
func (obj *PodStatusResult) GetObjectMeta() meta.Object                      { return &obj.ObjectMeta }
func (obj *PodStatusResult) GetObjectKind() unversioned.ObjectKind           { return &obj.TypeMeta }
func (obj *PodTemplate) GetObjectMeta() meta.Object                          { return &obj.ObjectMeta }
func (obj *PodTemplate) GetObjectKind() unversioned.ObjectKind               { return &obj.TypeMeta }
func (obj *PodTemplateList) GetObjectKind() unversioned.ObjectKind           { return &obj.TypeMeta }
func (obj *ReplicationController) GetObjectMeta() meta.Object                { return &obj.ObjectMeta }
func (obj *ReplicationController) GetObjectKind() unversioned.ObjectKind     { return &obj.TypeMeta }
func (obj *ReplicationControllerList) GetObjectKind() unversioned.ObjectKind { return &obj.TypeMeta }
func (obj *Service) GetObjectMeta() meta.Object                              { return &obj.ObjectMeta }
func (obj *Service) GetObjectKind() unversioned.ObjectKind                   { return &obj.TypeMeta }
func (obj *ServiceList) GetObjectKind() unversioned.ObjectKind               { return &obj.TypeMeta }
func (obj *Endpoints) GetObjectMeta() meta.Object                            { return &obj.ObjectMeta }
func (obj *Endpoints) GetObjectKind() unversioned.ObjectKind                 { return &obj.TypeMeta }
func (obj *EndpointsList) GetObjectKind() unversioned.ObjectKind             { return &obj.TypeMeta }
func (obj *Node) GetObjectMeta() meta.Object                                 { return &obj.ObjectMeta }
func (obj *Node) GetObjectKind() unversioned.ObjectKind                      { return &obj.TypeMeta }
func (obj *NodeList) GetObjectKind() unversioned.ObjectKind                  { return &obj.TypeMeta }
func (obj *NodeProxyOptions) GetObjectKind() unversioned.ObjectKind          { return &obj.TypeMeta }
func (obj *Binding) GetObjectMeta() meta.Object                              { return &obj.ObjectMeta }
func (obj *Binding) GetObjectKind() unversioned.ObjectKind                   { return &obj.TypeMeta }
func (obj *Event) GetObjectMeta() meta.Object                                { return &obj.ObjectMeta }
func (obj *Event) GetObjectKind() unversioned.ObjectKind                     { return &obj.TypeMeta }
func (obj *EventList) GetObjectKind() unversioned.ObjectKind                 { return &obj.TypeMeta }
func (obj *List) GetObjectKind() unversioned.ObjectKind                      { return &obj.TypeMeta }
func (obj *ListOptions) GetObjectKind() unversioned.ObjectKind               { return &obj.TypeMeta }
func (obj *LimitRange) GetObjectMeta() meta.Object                           { return &obj.ObjectMeta }
func (obj *LimitRange) GetObjectKind() unversioned.ObjectKind                { return &obj.TypeMeta }
func (obj *LimitRangeList) GetObjectKind() unversioned.ObjectKind            { return &obj.TypeMeta }
func (obj *ResourceQuota) GetObjectMeta() meta.Object                        { return &obj.ObjectMeta }
func (obj *ResourceQuota) GetObjectKind() unversioned.ObjectKind             { return &obj.TypeMeta }
func (obj *ResourceQuotaList) GetObjectKind() unversioned.ObjectKind         { return &obj.TypeMeta }
func (obj *Namespace) GetObjectMeta() meta.Object                            { return &obj.ObjectMeta }
func (obj *Namespace) GetObjectKind() unversioned.ObjectKind                 { return &obj.TypeMeta }
func (obj *NamespaceList) GetObjectKind() unversioned.ObjectKind             { return &obj.TypeMeta }
func (obj *ServiceAccount) GetObjectMeta() meta.Object                       { return &obj.ObjectMeta }
func (obj *ServiceAccount) GetObjectKind() unversioned.ObjectKind            { return &obj.TypeMeta }
func (obj *ServiceAccountList) GetObjectKind() unversioned.ObjectKind        { return &obj.TypeMeta }
func (obj *Secret) GetObjectMeta() meta.Object                               { return &obj.ObjectMeta }
func (obj *Secret) GetObjectKind() unversioned.ObjectKind                    { return &obj.TypeMeta }
func (obj *SecretList) GetObjectKind() unversioned.ObjectKind                { return &obj.TypeMeta }
func (obj *PersistentVolume) GetObjectMeta() meta.Object                     { return &obj.ObjectMeta }
func (obj *PersistentVolume) GetObjectKind() unversioned.ObjectKind          { return &obj.TypeMeta }
func (obj *PersistentVolumeList) GetObjectKind() unversioned.ObjectKind      { return &obj.TypeMeta }
func (obj *PersistentVolumeClaim) GetObjectMeta() meta.Object                { return &obj.ObjectMeta }
func (obj *PersistentVolumeClaim) GetObjectKind() unversioned.ObjectKind     { return &obj.TypeMeta }
func (obj *PersistentVolumeClaimList) GetObjectKind() unversioned.ObjectKind { return &obj.TypeMeta }
func (obj *DeleteOptions) GetObjectKind() unversioned.ObjectKind             { return &obj.TypeMeta }
func (obj *PodAttachOptions) GetObjectKind() unversioned.ObjectKind          { return &obj.TypeMeta }
func (obj *PodLogOptions) GetObjectKind() unversioned.ObjectKind             { return &obj.TypeMeta }
func (obj *PodExecOptions) GetObjectKind() unversioned.ObjectKind            { return &obj.TypeMeta }
func (obj *PodProxyOptions) GetObjectKind() unversioned.ObjectKind           { return &obj.TypeMeta }
func (obj *ServiceProxyOptions) GetObjectKind() unversioned.ObjectKind       { return &obj.TypeMeta }
func (obj *ComponentStatus) GetObjectMeta() meta.Object                      { return &obj.ObjectMeta }
func (obj *ComponentStatus) GetObjectKind() unversioned.ObjectKind           { return &obj.TypeMeta }
func (obj *ComponentStatusList) GetObjectKind() unversioned.ObjectKind       { return &obj.TypeMeta }
func (obj *SerializedReference) GetObjectKind() unversioned.ObjectKind       { return &obj.TypeMeta }
func (obj *RangeAllocation) GetObjectMeta() meta.Object                      { return &obj.ObjectMeta }
func (obj *RangeAllocation) GetObjectKind() unversioned.ObjectKind           { return &obj.TypeMeta }
func (obj *ObjectReference) GetObjectKind() unversioned.ObjectKind           { return obj }
func (obj *ExportOptions) GetObjectKind() unversioned.ObjectKind             { return &obj.TypeMeta }
func (obj *ConfigMap) GetObjectMeta() meta.Object                            { return &obj.ObjectMeta }
func (obj *ConfigMap) GetObjectKind() unversioned.ObjectKind                 { return &obj.TypeMeta }
func (obj *ConfigMapList) GetObjectKind() unversioned.ObjectKind             { return &obj.TypeMeta }
