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

package v1beta1

import (
	fmt "fmt"

	authorizationv1 "k8s.io/api/authorization/v1"
	authorizationv1beta1 "k8s.io/api/authorization/v1beta1"
	conversion "k8s.io/apimachinery/pkg/conversion"
	authorizationv1beta1apiserver "k8s.io/apiserver/pkg/apis/authorization/v1beta1"
	authorization "k8s.io/kubernetes/pkg/apis/authorization"
	authorizationv1internal "k8s.io/kubernetes/pkg/apis/authorization/v1"
)

// NOTE: These conversion implementations must be kept in sync with k8s.io/apiserver/pkg/apis/authorization/v1beta1/conversion.go

func enforceUnconditionalHandledDecisionTypesOnly(ao *authorization.AuthorizationOptions) error {
	// A nil AuthorizationOptions represents the default (unconditional) handled decision types,
	// which is expressible in v1beta1.
	if ao == nil {
		return nil
	}
	// Convert to v1 to make use of the helper functions
	authzOptionsV1 := &authorizationv1.AuthorizationOptions{}
	if err := authorizationv1internal.Convert_authorization_AuthorizationOptions_To_v1_AuthorizationOptions(ao, authzOptionsV1, nil); err != nil {
		return err
	}
	return authorizationv1beta1apiserver.EnforceUnconditionalHandledDecisionTypesOnly(authzOptionsV1)
}

// Convert_authorization_SelfSubjectAccessReviewSpec_To_v1beta1_SelfSubjectAccessReviewSpec explicitly does not propagate the AuthorizationOptions field to the v1beta1
// object; the conversion is thus lossy. However, this is ok, as SelfSubjectAccessReview objects are never stored.
func Convert_authorization_SelfSubjectAccessReviewSpec_To_v1beta1_SelfSubjectAccessReviewSpec(in *authorization.SelfSubjectAccessReviewSpec, out *authorizationv1beta1.SelfSubjectAccessReviewSpec, s conversion.Scope) error {
	if err := enforceUnconditionalHandledDecisionTypesOnly(in.AuthorizationOptions); err != nil {
		return err
	}
	return autoConvert_authorization_SelfSubjectAccessReviewSpec_To_v1beta1_SelfSubjectAccessReviewSpec(in, out, s)
}

// Convert_authorization_SubjectAccessReviewSpec_To_v1beta1_SubjectAccessReviewSpec explicitly does not propagate the AuthorizationOptions field to the v1beta1
// object; the conversion is thus lossy. However, this is ok, as {Local,}SubjectAccessReview objects are never stored.
func Convert_authorization_SubjectAccessReviewSpec_To_v1beta1_SubjectAccessReviewSpec(in *authorization.SubjectAccessReviewSpec, out *authorizationv1beta1.SubjectAccessReviewSpec, s conversion.Scope) error {
	if err := enforceUnconditionalHandledDecisionTypesOnly(in.AuthorizationOptions); err != nil {
		return err
	}
	return autoConvert_authorization_SubjectAccessReviewSpec_To_v1beta1_SubjectAccessReviewSpec(in, out, s)
}

// Convert_authorization_SubjectAccessReviewStatus_To_v1beta1_SubjectAccessReviewStatus explicitly does not propagate the ConditionalDecision field to the v1beta1
// object; the conversion is thus lossy. However, this is ok, as {Local,Self,}SubjectAccessReview objects are never stored.
func Convert_authorization_SubjectAccessReviewStatus_To_v1beta1_SubjectAccessReviewStatus(in *authorization.SubjectAccessReviewStatus, out *authorizationv1beta1.SubjectAccessReviewStatus, s conversion.Scope) error {
	// in.ConditionalDecision != nil implies that the response is conditional; this is not expressible in v1beta1, so fail closed.
	if in.ConditionalDecision != nil {
		return fmt.Errorf("cannot convert SubjectAccessReviewStatus to v1beta1, v1beta1 does not support in.ConditionalDecision, which is non-nil in the input object")
	}

	return autoConvert_authorization_SubjectAccessReviewStatus_To_v1beta1_SubjectAccessReviewStatus(in, out, s)
}
