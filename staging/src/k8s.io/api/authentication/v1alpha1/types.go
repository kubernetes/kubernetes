/*
Copyright 2022 The Kubernetes Authors.

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
	v1 "k8s.io/api/authentication/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +genclient:onlyVerbs=create
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.26

// SelfSubjectReview contains the user information that the kube-apiserver has about the user making this request.
// When using impersonation, users will receive the user info of the user being impersonated.  If impersonation or
// request header authentication is used, any extra keys will have their case ignored and returned as lowercase.
type SelfSubjectReview struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	// Status is filled in by the server with the user attributes.
	Status SelfSubjectReviewStatus `json:"status,omitempty" protobuf:"bytes,2,opt,name=status"`
}

// SelfSubjectReviewStatus is filled by the kube-apiserver and sent back to a user.
type SelfSubjectReviewStatus struct {
	// User attributes of the user making this request.
	// +optional
	UserInfo v1.UserInfo `json:"userInfo,omitempty" protobuf:"bytes,1,opt,name=userInfo"`
}
