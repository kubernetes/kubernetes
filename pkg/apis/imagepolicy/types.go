/*
Copyright 2016 The Kubernetes Authors.

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

package imagepolicy

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient=true
// +nonNamespaced=true
// +noMethods=true

// ImageReview checks if the set of images in a pod are allowed.
type ImageReview struct {
	metav1.TypeMeta
	metav1.ObjectMeta

	// Spec holds information about the pod being evaluated
	Spec ImageReviewSpec

	// Status is filled in by the backend and indicates whether the pod should be allowed.
	Status ImageReviewStatus
}

// ImageReviewSpec is a description of the pod creation request.
type ImageReviewSpec struct {
	// Containers is a list of a subset of the information in each container of the Pod being created.
	Containers []ImageReviewContainerSpec
	// Annotations is a list of key-value pairs extracted from the Pod's annotations.
	// It only includes keys which match the pattern `*.image-policy.k8s.io/*`.
	// It is up to each webhook backend to determine how to interpret these annotations, if at all.
	Annotations map[string]string
	// Namespace is the namespace the pod is being created in.
	Namespace string
}

// ImageReviewContainerSpec is a description of a container within the pod creation request.
type ImageReviewContainerSpec struct {
	// This can be in the form image:tag or image@SHA:012345679abcdef.
	Image string
	// In future, we may add command line overrides, exec health check command lines, and so on.
}

// ImageReviewStatus is the result of the token authentication request.
type ImageReviewStatus struct {
	// Allowed indicates that all images were allowed to be run.
	Allowed bool
	// Reason should be empty unless Allowed is false in which case it
	// may contain a short description of what is wrong.  Kubernetes
	// may truncate excessively long errors when displaying to the user.
	Reason string
}
