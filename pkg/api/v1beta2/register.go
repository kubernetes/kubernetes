/*
Copyright 2014 Google Inc. All rights reserved.

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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

// Codec encodes internal objects to the v1beta2 scheme
var Codec = runtime.CodecFor(api.Scheme, "v1beta2")

func init() {
	api.Scheme.AddKnownTypes("v1beta2",
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
		&Binding{},
		&Status{},
		&Event{},
		&EventList{},
		&ContainerManifest{},
		&ContainerManifestList{},
		&BoundPod{},
		&BoundPods{},
		&List{},
		&LimitRange{},
		&LimitRangeList{},
		&ResourceQuota{},
		&ResourceQuotaList{},
		&ResourceQuotaUsage{},
		&Namespace{},
		&NamespaceList{},
		&Secret{},
		&SecretList{},
		&PersistentVolume{},
		&PersistentVolumeList{},
		&PersistentVolumeClaim{},
		&PersistentVolumeClaimList{},
	)
	// Future names are supported
	api.Scheme.AddKnownTypeWithName("v1beta2", "Node", &Minion{})
	api.Scheme.AddKnownTypeWithName("v1beta2", "NodeList", &MinionList{})
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
func (*MinionList) IsAnAPIObject()                {}
func (*Binding) IsAnAPIObject()                   {}
func (*Status) IsAnAPIObject()                    {}
func (*Event) IsAnAPIObject()                     {}
func (*EventList) IsAnAPIObject()                 {}
func (*ContainerManifest) IsAnAPIObject()         {}
func (*ContainerManifestList) IsAnAPIObject()     {}
func (*BoundPod) IsAnAPIObject()                  {}
func (*BoundPods) IsAnAPIObject()                 {}
func (*List) IsAnAPIObject()                      {}
func (*LimitRange) IsAnAPIObject()                {}
func (*LimitRangeList) IsAnAPIObject()            {}
func (*ResourceQuota) IsAnAPIObject()             {}
func (*ResourceQuotaList) IsAnAPIObject()         {}
func (*ResourceQuotaUsage) IsAnAPIObject()        {}
func (*Namespace) IsAnAPIObject()                 {}
func (*NamespaceList) IsAnAPIObject()             {}
func (*Secret) IsAnAPIObject()                    {}
func (*SecretList) IsAnAPIObject()                {}
func (*PersistentVolume) IsAnAPIObject()          {}
func (*PersistentVolumeList) IsAnAPIObject()      {}
func (*PersistentVolumeClaim) IsAnAPIObject()     {}
func (*PersistentVolumeClaimList) IsAnAPIObject() {}
