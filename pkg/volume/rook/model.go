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

package rook

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const attachmentPluralResources = "volumeattachs"

type VolumeAttach struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata"`
	Spec              VolumeAttachSpec   `json:"spec"`
	Status            VolumeAttachStatus `json:"status,omitempty"`
}

type VolumeAttachSpec struct {
	VolumeID    string `json:"volumeID"`
	VolumeGroup string `json:"volumeGroup"`
	Node        string `json:"node"`
}

type VolumeAttachStatus struct {
	State   VolumeAttachState `json:"state,omitempty"`
	Message string            `json:"message,omitempty"`
}

type VolumeAttachState string

const (
	VolumeAttachStatePending  VolumeAttachState = "Pending"
	VolumeAttachStateAttached VolumeAttachState = "Attached"
	VolumeAttachStateFailed   VolumeAttachState = "Failed"
)

type VolumeAttachList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`
	VolumeAttachs   []VolumeAttach `json:"volumeAttachs"`
}
