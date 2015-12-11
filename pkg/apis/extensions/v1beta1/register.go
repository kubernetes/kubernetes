/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package v1beta1

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
)

// SchemeGroupVersion is group version used to register these objects
var SchemeGroupVersion = unversioned.GroupVersion{Group: "extensions", Version: "v1beta1"}

var Codec = runtime.CodecFor(api.Scheme, SchemeGroupVersion.String())

func init() {
	addKnownTypes()
	addDefaultingFuncs()
	addConversionFuncs()
}

// Adds the list of known types to api.Scheme.
func addKnownTypes() {
	api.Scheme.AddKnownTypes(SchemeGroupVersion,
		&ClusterAutoscaler{},
		&ClusterAutoscalerList{},
		&Deployment{},
		&DeploymentList{},
		&HorizontalPodAutoscaler{},
		&HorizontalPodAutoscalerList{},
		&Job{},
		&JobList{},
		&ReplicationControllerDummy{},
		&Scale{},
		&ThirdPartyResource{},
		&ThirdPartyResourceList{},
		&DaemonSetList{},
		&DaemonSet{},
		&ThirdPartyResourceData{},
		&ThirdPartyResourceDataList{},
		&Ingress{},
		&IngressList{},
	)

	// Register Unversioned types
	// TODO this should not be done here
	api.Scheme.AddKnownTypes(SchemeGroupVersion, &unversioned.ListOptions{})
}

func (obj *ClusterAutoscaler) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *ClusterAutoscalerList) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *Deployment) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *DeploymentList) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *HorizontalPodAutoscaler) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *HorizontalPodAutoscalerList) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *Job) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *JobList) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *ReplicationControllerDummy) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *Scale) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *ThirdPartyResource) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *ThirdPartyResourceList) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *DaemonSet) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *DaemonSetList) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *ThirdPartyResourceData) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *ThirdPartyResourceDataList) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *Ingress) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *IngressList) SetGroupVersionKind(gvk *unversioned.GroupVersionKind) {
	unversioned.UpdateTypeMeta(&obj.TypeMeta, gvk)
}
