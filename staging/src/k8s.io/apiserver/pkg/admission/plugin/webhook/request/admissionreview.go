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
	admissionv1alpha1 "k8s.io/api/admission/v1alpha1"
	authenticationv1 "k8s.io/api/authentication/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apiserver/pkg/admission"
)

// CreateAdmissionReview creates an AdmissionReview for the provided admission.Attributes
func CreateAdmissionReview(attr admission.Attributes) admissionv1alpha1.AdmissionReview {
	gvk := attr.GetKind()
	gvr := attr.GetResource()
	aUserInfo := attr.GetUserInfo()
	userInfo := authenticationv1.UserInfo{
		Extra:    make(map[string]authenticationv1.ExtraValue),
		Groups:   aUserInfo.GetGroups(),
		UID:      aUserInfo.GetUID(),
		Username: aUserInfo.GetName(),
	}

	// Convert the extra information in the user object
	for key, val := range aUserInfo.GetExtra() {
		userInfo.Extra[key] = authenticationv1.ExtraValue(val)
	}

	return admissionv1alpha1.AdmissionReview{
		Request: &admissionv1alpha1.AdmissionRequest{
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
			SubResource: attr.GetSubresource(),
			Name:        attr.GetName(),
			Namespace:   attr.GetNamespace(),
			Operation:   admissionv1alpha1.Operation(attr.GetOperation()),
			UserInfo:    userInfo,
			Object: runtime.RawExtension{
				Object: attr.GetObject(),
			},
			OldObject: runtime.RawExtension{
				Object: attr.GetOldObject(),
			},
		},
	}
}
