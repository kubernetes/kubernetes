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

package v1

import (
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	versionedwatch "k8s.io/kubernetes/pkg/watch/versioned"
)

// GroupName is the group name use in this package
const GroupName = ""

// SchemeGroupVersion is group version used to register these objects
var SchemeGroupVersion = unversioned.GroupVersion{Group: GroupName, Version: "v1"}

var (
	SchemeBuilder = runtime.NewSchemeBuilder(addKnownTypes, addDefaultingFuncs, addConversionFuncs, addFastPathConversionFuncs)
	AddToScheme   = SchemeBuilder.AddToScheme
)

// Adds the list of known types to api.Scheme.
func addKnownTypes(scheme *runtime.Scheme) error {
	scheme.AddKnownTypes(SchemeGroupVersion,
		&Pod{},
		&PodList{},
		&PodStatusResult{},
		&PodTemplate{},
		&PodTemplateList{},
		&ReplicationController{},
		&ReplicationControllerList{},
		&Service{},
		&ServiceProxyOptions{},
		&ServiceList{},
		&Endpoints{},
		&EndpointsList{},
		&Node{},
		&NodeList{},
		&NodeProxyOptions{},
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
		&Secret{},
		&SecretList{},
		&ServiceAccount{},
		&ServiceAccountList{},
		&PersistentVolume{},
		&PersistentVolumeList{},
		&PersistentVolumeClaim{},
		&PersistentVolumeClaimList{},
		&DeleteOptions{},
		&ExportOptions{},
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

	// Add common types
	scheme.AddKnownTypes(SchemeGroupVersion, &unversioned.Status{})

	// Add the watch version that applies
	versionedwatch.AddToGroupVersion(scheme, SchemeGroupVersion)
	return nil
}
