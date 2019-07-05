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

package request

import (
	admissionv1beta1 "k8s.io/api/admission/v1beta1"
	authenticationv1 "k8s.io/api/authentication/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/generic"
)

// CreateAdmissionReview creates an AdmissionReview for the provided admission.Attributes
func CreateAdmissionReview(versionedAttributes *generic.VersionedAttributes, invocation *generic.WebhookInvocation) admissionv1beta1.AdmissionReview {
	attr := versionedAttributes.Attributes
	gvk := invocation.Kind
	gvr := invocation.Resource
	subresource := invocation.Subresource
	requestGVK := attr.GetKind()
	requestGVR := attr.GetResource()
	requestSubResource := attr.GetSubresource()
	aUserInfo := attr.GetUserInfo()
	userInfo := authenticationv1.UserInfo{
		Extra:    make(map[string]authenticationv1.ExtraValue),
		Groups:   aUserInfo.GetGroups(),
		UID:      aUserInfo.GetUID(),
		Username: aUserInfo.GetName(),
	}
	dryRun := attr.IsDryRun()

	// Convert the extra information in the user object
	for key, val := range aUserInfo.GetExtra() {
		userInfo.Extra[key] = authenticationv1.ExtraValue(val)
	}

	return admissionv1beta1.AdmissionReview{
		Request: &admissionv1beta1.AdmissionRequest{
			UID: uuid.NewUUID(),
			Kind: metav1.GroupVersionKind{
				Group:   gvk.Group,
				Kind:    gvk.Kind,
				Version: gvk.Version,
			},
			Resource: metav1.GroupVersionResource{
				Group:    gvr.Group,
				Resource: gvr.Resource,
				Version:  gvr.Version,
			},
			SubResource: subresource,
			RequestKind: &metav1.GroupVersionKind{
				Group:   requestGVK.Group,
				Kind:    requestGVK.Kind,
				Version: requestGVK.Version,
			},
			RequestResource: &metav1.GroupVersionResource{
				Group:    requestGVR.Group,
				Resource: requestGVR.Resource,
				Version:  requestGVR.Version,
			},
			RequestSubResource: requestSubResource,
			Name:               attr.GetName(),
			Namespace:          attr.GetNamespace(),
			Operation:          admissionv1beta1.Operation(attr.GetOperation()),
			UserInfo:           userInfo,
			Object: runtime.RawExtension{
				Object: versionedAttributes.VersionedObject,
			},
			OldObject: runtime.RawExtension{
				Object: versionedAttributes.VersionedOldObject,
			},
			DryRun: &dryRun,
			Options: runtime.RawExtension{
				Object: attr.GetOperationOptions(),
			},
		},
	}
}
