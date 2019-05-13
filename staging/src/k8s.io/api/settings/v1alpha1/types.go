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

package v1alpha1

import (
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodPreset is a policy resource that defines additional runtime
// requirements for a Pod.
type PodPreset struct {
	metav1.TypeMeta `json:",inline"`
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// +optional
	Spec PodPresetSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`
}

// PodPresetSpec is a description of a pod preset.
type PodPresetSpec struct {
	// Selector is a label query over a set of resources, in this case pods.
	// Required.
	Selector metav1.LabelSelector `json:"selector,omitempty" protobuf:"bytes,1,opt,name=selector"`

	// Env defines the collection of EnvVar to inject into containers.
	// +optional
	Env []v1.EnvVar `json:"env,omitempty" protobuf:"bytes,2,rep,name=env"`
	// EnvFrom defines the collection of EnvFromSource to inject into containers.
	// +optional
	EnvFrom []v1.EnvFromSource `json:"envFrom,omitempty" protobuf:"bytes,3,rep,name=envFrom"`
	// Volumes defines the collection of Volume to inject into the pod.
	// +optional
	Volumes []v1.Volume `json:"volumes,omitempty" protobuf:"bytes,4,rep,name=volumes"`
	// VolumeMounts defines the collection of VolumeMount to inject into containers.
	// +optional
	VolumeMounts []v1.VolumeMount `json:"volumeMounts,omitempty" protobuf:"bytes,5,rep,name=volumeMounts"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// PodPresetList is a list of PodPreset objects.
type PodPresetList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is a list of schema objects.
	Items []PodPreset `json:"items" protobuf:"bytes,2,rep,name=items"`
}
