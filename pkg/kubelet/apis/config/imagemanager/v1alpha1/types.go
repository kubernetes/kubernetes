/*
Copyright 2024 The Kubernetes Authors.

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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// ImagePullIntent is a record of the kubelet attempting to pull an image.
//
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type ImagePullIntent struct {
	metav1.TypeMeta

	// Image is the image spec from a Container's `image` field.
	// The filename is a SHA-256 hash of this value. This is to avoid filename-unsafe
	// characters like ':' and '/'.
	Image string `json:"image"`
}

// ImagePullRecord is a record of an image that was pulled by the kubelet.
//
// If there are no records in the `kubernetesSecrets` field and both `nodeWideCredentials`
// and `anonymous` are `false`, credentials must be re-checked the next time an
// image represented by this record is being requested.
//
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type ImagePulledRecord struct {
	metav1.TypeMeta

	// LastUpdatedTime is the time of the last update to this record
	LastUpdatedTime metav1.Time `json:"lastUpdatedTime"`

	// ImageRef is a reference to the image represented by this file as received
	// from the CRI.
	// The filename is a SHA-256 hash of this value. This is to avoid filename-unsafe
	// characters like ':' and '/'.
	ImageRef string `json:"imageRef"`

	// CredentialMapping maps `image` to the set of credentials that it was
	// previously pulled with.
	// `image` in this case is the content of a pod's container `image` field that's
	// got its tag/digest removed.
	//
	// Example:
	//   Container requests the `hello-world:latest@sha256:91fb4b041da273d5a3273b6d587d62d518300a6ad268b28628f74997b93171b2` image:
	//     "credentialMapping": {
	//       "hello-world": { "nodePodsAccessible": true }
	//     }
	CredentialMapping map[string]ImagePullCredentials `json:"credentialMapping,omitempty"`
}

// ImagePullCredentials describe credentials that can be used to pull an image.
type ImagePullCredentials struct {
	// KuberneteSecretCoordinates is an index of coordinates of all the kubernetes
	// secrets that were used to pull the image.
	// +optional
	KubernetesSecrets []ImagePullSecret `json:"kubernetesSecretCoordinates"`

	// NodePodsAccessible is a flag denoting the pull credentials are accessible
	// by all the pods on the node, or that no credentials are needed for the pull.
	//
	// If true, it is mutually exclusive with the `kubernetesSecrets` field.
	// +optional
	NodePodsAccessible bool `json:"nodePodsAccessible,omitempty"`
}

// ImagePullSecret is a representation of a Kubernetes secret object coordinates along
// with a credential hash of the pull secret credentials this object contains.
type ImagePullSecret struct {
	UID       string `json:"uid"`
	Namespace string `json:"namespace"`
	Name      string `json:"name"`

	// CredentialHash is a SHA-256 retrieved by hashing the image pull credentials
	// content of the secret specified by the UID/Namespace/Name coordinates.
	CredentialHash string `json:"credentialHash"`
}
