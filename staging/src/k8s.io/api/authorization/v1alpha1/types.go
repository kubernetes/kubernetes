/*
Copyright The Kubernetes Authors.

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
	admissionv1 "k8s.io/api/admission/v1"
	authorizationv1 "k8s.io/api/authorization/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

// +genclient
// +genclient:nonNamespaced
// +genclient:onlyVerbs=create
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.37

// AuthorizationConditionsReview describes a request to evaluate authorization conditions.
type AuthorizationConditionsReview struct {
	metav1.TypeMeta `json:",inline"`
	// metadata is the standard list metadata.
	// In AuthorizationConditionsReview, it must be an empty struct.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	// +k8s:opaqueType
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// request describes the attributes for the authorization conditions request.
	// +k8s:optional
	// +optional
	Request *AuthorizationConditionsRequest `json:"request,omitempty" protobuf:"bytes,2,opt,name=request"`
	// response describes the attributes for the authorization conditions response.
	// +k8s:optional
	// +optional
	Response *AuthorizationConditionsResponse `json:"response,omitempty" protobuf:"bytes,3,opt,name=response"`
}

// AuthorizationConditionsRequest describes the authorization conditions request.
type AuthorizationConditionsRequest struct {
	// decision contains the conditional decision the authorizer authored at authorization time.
	// +required
	Decision authorizationv1.ConditionsAwareDecision `json:"decision" protobuf:"bytes,1,opt,name=decision"`

	// admissionRequest may contain additional information for evaluating the conditions.
	// +k8s:unionMember
	// +k8s:optional
	// +optional
	AdmissionRequest *admissionv1.AdmissionRequest `json:"admissionRequest,omitempty" protobuf:"bytes,2,opt,name=admissionRequest"`
}

// AuthorizationConditionsResponse describes an authorization conditions response.
type AuthorizationConditionsResponse struct {
	// uid is an identifier for the individual request/response.
	// This must be copied over from the corresponding AuthorizationConditionsRequest.
	// It is possible that the same request content (except uid) is sent to the
	// authorizer multiple times.
	// +k8s:required
	// +required
	UID types.UID `json:"uid" protobuf:"bytes,1,opt,name=uid"`

	// decision contains the authorizer's decision after seeing the data.
	// +required
	Decision authorizationv1.ConditionsAwareDecision `json:"decision" protobuf:"bytes,2,opt,name=decision"`
}
