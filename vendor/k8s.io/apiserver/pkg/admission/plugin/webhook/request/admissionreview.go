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
	"fmt"

	admissionv1 "k8s.io/api/admission/v1"
	admissionv1beta1 "k8s.io/api/admission/v1beta1"
	authenticationv1 "k8s.io/api/authentication/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/generic"
)

// AdmissionResponse contains the fields extracted from an AdmissionReview response
type AdmissionResponse struct {
	AuditAnnotations map[string]string
	Allowed          bool
	Patch            []byte
	PatchType        admissionv1.PatchType
	Result           *metav1.Status
	Warnings         []string
}

// VerifyAdmissionResponse checks the validity of the provided admission review object, and returns the
// audit annotations, whether the response allowed the request, any provided patch/patchType/status,
// or an error if the provided admission review was not valid.
func VerifyAdmissionResponse(uid types.UID, mutating bool, review runtime.Object) (*AdmissionResponse, error) {
	switch r := review.(type) {
	case *admissionv1.AdmissionReview:
		if r.Response == nil {
			return nil, fmt.Errorf("webhook response was absent")
		}

		// Verify UID matches
		if r.Response.UID != uid {
			return nil, fmt.Errorf("expected response.uid=%q, got %q", uid, r.Response.UID)
		}

		// Verify GVK
		v1GVK := admissionv1.SchemeGroupVersion.WithKind("AdmissionReview")
		if r.GroupVersionKind() != v1GVK {
			return nil, fmt.Errorf("expected webhook response of %v, got %v", v1GVK.String(), r.GroupVersionKind().String())
		}

		patch := []byte(nil)
		patchType := admissionv1.PatchType("")

		if mutating {
			// Ensure a mutating webhook provides both patch and patchType together
			if len(r.Response.Patch) > 0 && r.Response.PatchType == nil {
				return nil, fmt.Errorf("webhook returned response.patch but not response.patchType")
			}
			if len(r.Response.Patch) == 0 && r.Response.PatchType != nil {
				return nil, fmt.Errorf("webhook returned response.patchType but not response.patch")
			}
			patch = r.Response.Patch
			if r.Response.PatchType != nil {
				patchType = *r.Response.PatchType
				if len(patchType) == 0 {
					return nil, fmt.Errorf("webhook returned invalid response.patchType of %q", patchType)
				}
			}
		} else {
			// Ensure a validating webhook doesn't return patch or patchType
			if len(r.Response.Patch) > 0 {
				return nil, fmt.Errorf("validating webhook may not return response.patch")
			}
			if r.Response.PatchType != nil {
				return nil, fmt.Errorf("validating webhook may not return response.patchType")
			}
		}

		return &AdmissionResponse{
			AuditAnnotations: r.Response.AuditAnnotations,
			Allowed:          r.Response.Allowed,
			Patch:            patch,
			PatchType:        patchType,
			Result:           r.Response.Result,
			Warnings:         r.Response.Warnings,
		}, nil

	case *admissionv1beta1.AdmissionReview:
		if r.Response == nil {
			return nil, fmt.Errorf("webhook response was absent")
		}

		// Response GVK and response.uid were not verified in v1beta1 handling, allow any

		patch := []byte(nil)
		patchType := admissionv1.PatchType("")
		if mutating {
			patch = r.Response.Patch
			if len(r.Response.Patch) > 0 {
				// patch type was not verified in v1beta1 admissionreview handling. pin to only supported version if a patch is provided.
				patchType = admissionv1.PatchTypeJSONPatch
			}
		}

		return &AdmissionResponse{
			AuditAnnotations: r.Response.AuditAnnotations,
			Allowed:          r.Response.Allowed,
			Patch:            patch,
			PatchType:        patchType,
			Result:           r.Response.Result,
			Warnings:         r.Response.Warnings,
		}, nil

	default:
		return nil, fmt.Errorf("unexpected response type %T", review)
	}
}

// CreateAdmissionObjects returns the unique request uid, the AdmissionReview object to send the webhook and to decode the response into,
// or an error if the webhook does not support receiving any of the admission review versions we know to send
func CreateAdmissionObjects(versionedAttributes *generic.VersionedAttributes, invocation *generic.WebhookInvocation) (uid types.UID, request, response runtime.Object, err error) {
	for _, version := range invocation.Webhook.GetAdmissionReviewVersions() {
		switch version {
		case admissionv1.SchemeGroupVersion.Version:
			uid := types.UID(uuid.NewUUID())
			request := CreateV1AdmissionReview(uid, versionedAttributes, invocation)
			response := &admissionv1.AdmissionReview{}
			return uid, request, response, nil

		case admissionv1beta1.SchemeGroupVersion.Version:
			uid := types.UID(uuid.NewUUID())
			request := CreateV1beta1AdmissionReview(uid, versionedAttributes, invocation)
			response := &admissionv1beta1.AdmissionReview{}
			return uid, request, response, nil

		}
	}
	return "", nil, nil, fmt.Errorf("webhook does not accept known AdmissionReview versions (v1, v1beta1)")
}

// CreateV1AdmissionReview creates an AdmissionReview for the provided admission.Attributes
func CreateV1AdmissionReview(uid types.UID, versionedAttributes *generic.VersionedAttributes, invocation *generic.WebhookInvocation) *admissionv1.AdmissionReview {
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

	return &admissionv1.AdmissionReview{
		Request: &admissionv1.AdmissionRequest{
			UID: uid,
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
			Operation:          admissionv1.Operation(attr.GetOperation()),
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

// CreateV1beta1AdmissionReview creates an AdmissionReview for the provided admission.Attributes
func CreateV1beta1AdmissionReview(uid types.UID, versionedAttributes *generic.VersionedAttributes, invocation *generic.WebhookInvocation) *admissionv1beta1.AdmissionReview {
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

	return &admissionv1beta1.AdmissionReview{
		Request: &admissionv1beta1.AdmissionRequest{
			UID: uid,
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
