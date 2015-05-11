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

package v1beta1

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

// Codec encodes internal objects to the v1beta1 scheme
var Codec = runtime.CodecFor(api.Scheme, "v1beta1")

// Dependency does nothing but give a hook for other packages to force a
// compile-time error when this API version is eventually removed.  This is
// useful, for example, to clean up things that are implicitly tied to
// semantics of older APIs.
const Dependency = true

func init() {
	api.Scheme.AddKnownTypes("v1beta1",
		&Pod{},
		&PodStatusResult{},
		&PodList{},
		&ReplicationController{},
		&ReplicationControllerList{},
		&Service{},
		&ServiceList{},
		&Endpoints{},
		&EndpointsList{},
		&Minion{},
		&MinionList{},
		&NodeInfo{},
		&Binding{},
		&Status{},
		&Event{},
		&EventList{},
		&ContainerManifest{},
		&ContainerManifestList{},
		&List{},
		&LimitRange{},
		&LimitRangeList{},
		&ResourceQuota{},
		&ResourceQuotaList{},
		&Namespace{},
		&NamespaceList{},
		&Secret{},
		&SecretList{},
		&PersistentVolume{},
		&PersistentVolumeList{},
		&PersistentVolumeClaim{},
		&PersistentVolumeClaimList{},
		&DeleteOptions{},
		&ListOptions{},
		&PodLogOptions{},
		&PodExecOptions{},
		&PodProxyOptions{},
		&ComponentStatus{},
		&ComponentStatusList{},
		&SerializedReference{},
		&RangeAllocation{},
	)
	// Future names are supported
	api.Scheme.AddKnownTypeWithName("v1beta1", "Node", &Minion{})
	api.Scheme.AddKnownTypeWithName("v1beta1", "NodeList", &MinionList{})
}

func (*Pod) IsAnAPIObject()                       {}
func (*PodStatusResult) IsAnAPIObject()           {}
func (*PodList) IsAnAPIObject()                   {}
func (*ReplicationController) IsAnAPIObject()     {}
func (*ReplicationControllerList) IsAnAPIObject() {}
func (*Service) IsAnAPIObject()                   {}
func (*ServiceList) IsAnAPIObject()               {}
func (*Endpoints) IsAnAPIObject()                 {}
func (*EndpointsList) IsAnAPIObject()             {}
func (*Minion) IsAnAPIObject()                    {}
func (*NodeInfo) IsAnAPIObject()                  {}
func (*MinionList) IsAnAPIObject()                {}
func (*Binding) IsAnAPIObject()                   {}
func (*Status) IsAnAPIObject()                    {}
func (*Event) IsAnAPIObject()                     {}
func (*EventList) IsAnAPIObject()                 {}
func (*ContainerManifest) IsAnAPIObject()         {}
func (*ContainerManifestList) IsAnAPIObject()     {}
func (*List) IsAnAPIObject()                      {}
func (*LimitRange) IsAnAPIObject()                {}
func (*LimitRangeList) IsAnAPIObject()            {}
func (*ResourceQuota) IsAnAPIObject()             {}
func (*ResourceQuotaList) IsAnAPIObject()         {}
func (*Namespace) IsAnAPIObject()                 {}
func (*NamespaceList) IsAnAPIObject()             {}
func (*Secret) IsAnAPIObject()                    {}
func (*SecretList) IsAnAPIObject()                {}
func (*PersistentVolume) IsAnAPIObject()          {}
func (*PersistentVolumeList) IsAnAPIObject()      {}
func (*PersistentVolumeClaim) IsAnAPIObject()     {}
func (*PersistentVolumeClaimList) IsAnAPIObject() {}
func (*DeleteOptions) IsAnAPIObject()             {}
func (*ListOptions) IsAnAPIObject()               {}
func (*PodLogOptions) IsAnAPIObject()             {}
func (*PodExecOptions) IsAnAPIObject()            {}
func (*PodProxyOptions) IsAnAPIObject()           {}
func (*ComponentStatus) IsAnAPIObject()           {}
func (*ComponentStatusList) IsAnAPIObject()       {}
func (*SerializedReference) IsAnAPIObject()       {}
func (*RangeAllocation) IsAnAPIObject()           {}
