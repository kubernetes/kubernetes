/*
Copyright 2019 The Kubernetes Authors.

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
	"reflect"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	admissionv1 "k8s.io/api/admission/v1"
	admissionv1beta1 "k8s.io/api/admission/v1beta1"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	appsv1 "k8s.io/api/apps/v1"
	authenticationv1 "k8s.io/api/authentication/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/webhook"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/generic"
	"k8s.io/apiserver/pkg/authentication/user"
	utilpointer "k8s.io/utils/pointer"
)

func TestVerifyAdmissionResponse(t *testing.T) {
	v1beta1JSONPatch := admissionv1beta1.PatchTypeJSONPatch
	v1JSONPatch := admissionv1.PatchTypeJSONPatch

	emptyv1beta1Patch := admissionv1beta1.PatchType("")
	emptyv1Patch := admissionv1.PatchType("")

	invalidv1beta1Patch := admissionv1beta1.PatchType("Foo")
	invalidv1Patch := admissionv1.PatchType("Foo")

	testcases := []struct {
		name     string
		uid      types.UID
		mutating bool
		review   runtime.Object

		expectAuditAnnotations map[string]string
		expectAllowed          bool
		expectPatch            []byte
		expectPatchType        admissionv1.PatchType
		expectResult           *metav1.Status
		expectErr              string
	}{
		// Allowed validating
		{
			name: "v1beta1 allowed validating",
			uid:  "123",
			review: &admissionv1beta1.AdmissionReview{
				Response: &admissionv1beta1.AdmissionResponse{Allowed: true},
			},
			expectAllowed: true,
		},
		{
			name: "v1 allowed validating",
			uid:  "123",
			review: &admissionv1.AdmissionReview{
				TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
				Response: &admissionv1.AdmissionResponse{UID: "123", Allowed: true},
			},
			expectAllowed: true,
		},
		// Allowed mutating
		{
			name:     "v1beta1 allowed mutating",
			uid:      "123",
			mutating: true,
			review: &admissionv1beta1.AdmissionReview{
				Response: &admissionv1beta1.AdmissionResponse{Allowed: true},
			},
			expectAllowed: true,
		},
		{
			name:     "v1 allowed mutating",
			uid:      "123",
			mutating: true,
			review: &admissionv1.AdmissionReview{
				TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
				Response: &admissionv1.AdmissionResponse{UID: "123", Allowed: true},
			},
			expectAllowed: true,
		},

		// Audit annotations
		{
			name: "v1beta1 auditAnnotations",
			uid:  "123",
			review: &admissionv1beta1.AdmissionReview{
				Response: &admissionv1beta1.AdmissionResponse{
					Allowed:          true,
					AuditAnnotations: map[string]string{"foo": "bar"},
				},
			},
			expectAllowed:          true,
			expectAuditAnnotations: map[string]string{"foo": "bar"},
		},
		{
			name: "v1 auditAnnotations",
			uid:  "123",
			review: &admissionv1.AdmissionReview{
				TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
				Response: &admissionv1.AdmissionResponse{
					UID:              "123",
					Allowed:          true,
					AuditAnnotations: map[string]string{"foo": "bar"},
				},
			},
			expectAllowed:          true,
			expectAuditAnnotations: map[string]string{"foo": "bar"},
		},

		// Patch
		{
			name:     "v1beta1 patch",
			uid:      "123",
			mutating: true,
			review: &admissionv1beta1.AdmissionReview{
				Response: &admissionv1beta1.AdmissionResponse{
					Allowed: true,
					Patch:   []byte(`[{"op":"add","path":"/foo","value":"bar"}]`),
				},
			},
			expectAllowed:   true,
			expectPatch:     []byte(`[{"op":"add","path":"/foo","value":"bar"}]`),
			expectPatchType: "JSONPatch",
		},
		{
			name:     "v1 patch",
			uid:      "123",
			mutating: true,
			review: &admissionv1.AdmissionReview{
				TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
				Response: &admissionv1.AdmissionResponse{
					UID:       "123",
					Allowed:   true,
					Patch:     []byte(`[{"op":"add","path":"/foo","value":"bar"}]`),
					PatchType: &v1JSONPatch,
				},
			},
			expectAllowed:   true,
			expectPatch:     []byte(`[{"op":"add","path":"/foo","value":"bar"}]`),
			expectPatchType: "JSONPatch",
		},

		// Result
		{
			name: "v1beta1 result",
			uid:  "123",
			review: &admissionv1beta1.AdmissionReview{
				Response: &admissionv1beta1.AdmissionResponse{
					Allowed: false,
					Result:  &metav1.Status{Status: "Failure", Message: "Foo", Code: 401},
				},
			},
			expectAllowed: false,
			expectResult:  &metav1.Status{Status: "Failure", Message: "Foo", Code: 401},
		},
		{
			name: "v1 result",
			uid:  "123",
			review: &admissionv1.AdmissionReview{
				TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
				Response: &admissionv1.AdmissionResponse{
					UID:     "123",
					Allowed: false,
					Result:  &metav1.Status{Status: "Failure", Message: "Foo", Code: 401},
				},
			},
			expectAllowed: false,
			expectResult:  &metav1.Status{Status: "Failure", Message: "Foo", Code: 401},
		},

		// Missing response
		{
			name:      "v1beta1 no response",
			uid:       "123",
			review:    &admissionv1beta1.AdmissionReview{},
			expectErr: "response was absent",
		},
		{
			name: "v1 no response",
			uid:  "123",
			review: &admissionv1.AdmissionReview{
				TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
			},
			expectErr: "response was absent",
		},

		// v1 invalid responses
		{
			name: "v1 wrong group",
			uid:  "123",
			review: &admissionv1.AdmissionReview{
				TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io2/v1", Kind: "AdmissionReview"},
				Response: &admissionv1.AdmissionResponse{
					UID:     "123",
					Allowed: true,
				},
			},
			expectErr: "expected webhook response of admission.k8s.io/v1, Kind=AdmissionReview",
		},
		{
			name: "v1 wrong version",
			uid:  "123",
			review: &admissionv1.AdmissionReview{
				TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v2", Kind: "AdmissionReview"},
				Response: &admissionv1.AdmissionResponse{
					UID:     "123",
					Allowed: true,
				},
			},
			expectErr: "expected webhook response of admission.k8s.io/v1, Kind=AdmissionReview",
		},
		{
			name: "v1 wrong kind",
			uid:  "123",
			review: &admissionv1.AdmissionReview{
				TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview2"},
				Response: &admissionv1.AdmissionResponse{
					UID:     "123",
					Allowed: true,
				},
			},
			expectErr: "expected webhook response of admission.k8s.io/v1, Kind=AdmissionReview",
		},
		{
			name: "v1 wrong uid",
			uid:  "123",
			review: &admissionv1.AdmissionReview{
				TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
				Response: &admissionv1.AdmissionResponse{
					UID:     "1234",
					Allowed: true,
				},
			},
			expectErr: `expected response.uid="123"`,
		},
		{
			name:     "v1 patch without patch type",
			uid:      "123",
			mutating: true,
			review: &admissionv1.AdmissionReview{
				TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
				Response: &admissionv1.AdmissionResponse{
					UID:     "123",
					Allowed: true,
					Patch:   []byte(`[{"op":"add","path":"/foo","value":"bar"}]`),
				},
			},
			expectErr: `webhook returned response.patch but not response.patchType`,
		},
		{
			name:     "v1 patch type without patch",
			uid:      "123",
			mutating: true,
			review: &admissionv1.AdmissionReview{
				TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
				Response: &admissionv1.AdmissionResponse{
					UID:       "123",
					Allowed:   true,
					PatchType: &v1JSONPatch,
				},
			},
			expectErr: `webhook returned response.patchType but not response.patch`,
		},
		{
			name:     "v1 empty patch type",
			uid:      "123",
			mutating: true,
			review: &admissionv1.AdmissionReview{
				TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
				Response: &admissionv1.AdmissionResponse{
					UID:       "123",
					Allowed:   true,
					Patch:     []byte(`[{"op":"add","path":"/foo","value":"bar"}]`),
					PatchType: &emptyv1Patch,
				},
			},
			expectErr: `webhook returned invalid response.patchType of ""`,
		},
		{
			name:     "v1 invalid patch type",
			uid:      "123",
			mutating: true,
			review: &admissionv1.AdmissionReview{
				TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
				Response: &admissionv1.AdmissionResponse{
					UID:       "123",
					Allowed:   true,
					Patch:     []byte(`[{"op":"add","path":"/foo","value":"bar"}]`),
					PatchType: &invalidv1Patch,
				},
			},
			expectAllowed:   true,
			expectPatch:     []byte(`[{"op":"add","path":"/foo","value":"bar"}]`),
			expectPatchType: invalidv1Patch, // invalid patch types are caught when the mutating dispatcher evaluates the patch
		},
		{
			name:     "v1 patch for validating webhook",
			uid:      "123",
			mutating: false,
			review: &admissionv1.AdmissionReview{
				TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
				Response: &admissionv1.AdmissionResponse{
					UID:     "123",
					Allowed: true,
					Patch:   []byte(`[{"op":"add","path":"/foo","value":"bar"}]`),
				},
			},
			expectErr: `validating webhook may not return response.patch`,
		},
		{
			name:     "v1 patch type for validating webhook",
			uid:      "123",
			mutating: false,
			review: &admissionv1.AdmissionReview{
				TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
				Response: &admissionv1.AdmissionResponse{
					UID:       "123",
					Allowed:   true,
					PatchType: &invalidv1Patch,
				},
			},
			expectErr: `validating webhook may not return response.patchType`,
		},

		// v1beta1 invalid responses that we have to allow/fixup for compatibility
		{
			name: "v1beta1 wrong group/version/kind",
			uid:  "123",
			review: &admissionv1beta1.AdmissionReview{
				TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io2/v2", Kind: "AdmissionReview2"},
				Response: &admissionv1beta1.AdmissionResponse{
					Allowed: true,
				},
			},
			expectAllowed: true,
		},
		{
			name: "v1beta1 wrong uid",
			uid:  "123",
			review: &admissionv1beta1.AdmissionReview{
				Response: &admissionv1beta1.AdmissionResponse{
					UID:     "1234",
					Allowed: true,
				},
			},
			expectAllowed: true,
		},
		{
			name:     "v1beta1 validating returns patch/patchType",
			uid:      "123",
			mutating: false,
			review: &admissionv1beta1.AdmissionReview{
				Response: &admissionv1beta1.AdmissionResponse{
					UID:       "1234",
					Allowed:   true,
					Patch:     []byte(`[{"op":"add","path":"/foo","value":"bar"}]`),
					PatchType: &v1beta1JSONPatch,
				},
			},
			expectAllowed: true,
		},
		{
			name:     "v1beta1 empty patch type",
			uid:      "123",
			mutating: true,
			review: &admissionv1beta1.AdmissionReview{
				Response: &admissionv1beta1.AdmissionResponse{
					Allowed:   true,
					Patch:     []byte(`[{"op":"add","path":"/foo","value":"bar"}]`),
					PatchType: &emptyv1beta1Patch,
				},
			},
			expectAllowed:   true,
			expectPatch:     []byte(`[{"op":"add","path":"/foo","value":"bar"}]`),
			expectPatchType: admissionv1.PatchTypeJSONPatch,
		},
		{
			name:     "v1beta1 invalid patchType",
			uid:      "123",
			mutating: true,
			review: &admissionv1beta1.AdmissionReview{
				Response: &admissionv1beta1.AdmissionResponse{
					Allowed:   true,
					Patch:     []byte(`[{"op":"add","path":"/foo","value":"bar"}]`),
					PatchType: &invalidv1beta1Patch,
				},
			},
			expectAllowed:   true,
			expectPatch:     []byte(`[{"op":"add","path":"/foo","value":"bar"}]`),
			expectPatchType: admissionv1.PatchTypeJSONPatch,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := VerifyAdmissionResponse(tc.uid, tc.mutating, tc.review)
			if err != nil {
				if len(tc.expectErr) > 0 {
					if !strings.Contains(err.Error(), tc.expectErr) {
						t.Errorf("expected error '%s', got %v", tc.expectErr, err)
					}
				} else {
					t.Errorf("unexpected error %v", err)
				}
				return
			} else if len(tc.expectErr) > 0 {
				t.Errorf("expected error '%s', got none", tc.expectErr)
				return
			}

			if e, a := tc.expectAuditAnnotations, result.AuditAnnotations; !reflect.DeepEqual(e, a) {
				t.Errorf("unexpected: %v", cmp.Diff(e, a))
			}
			if e, a := tc.expectAllowed, result.Allowed; !reflect.DeepEqual(e, a) {
				t.Errorf("unexpected: %v", cmp.Diff(e, a))
			}
			if e, a := tc.expectPatch, result.Patch; !reflect.DeepEqual(e, a) {
				t.Errorf("unexpected: %v", cmp.Diff(e, a))
			}
			if e, a := tc.expectPatchType, result.PatchType; !reflect.DeepEqual(e, a) {
				t.Errorf("unexpected: %v", cmp.Diff(e, a))
			}
			if e, a := tc.expectResult, result.Result; !reflect.DeepEqual(e, a) {
				t.Errorf("unexpected: %v", cmp.Diff(e, a))
			}
		})
	}
}

func TestCreateAdmissionObjects(t *testing.T) {
	internalObj := &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "2", Name: "myname", Namespace: "myns"}}
	internalObjOld := &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1", Name: "myname", Namespace: "myns"}}
	versionedObj := &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "2", Name: "myname", Namespace: "myns"}}
	versionedObjOld := &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1", Name: "myname", Namespace: "myns"}}
	userInfo := &user.DefaultInfo{
		Name:   "myuser",
		Groups: []string{"mygroup"},
		UID:    "myuid",
		Extra:  map[string][]string{"extrakey": {"value1", "value2"}},
	}
	attrs := admission.NewAttributesRecord(
		internalObj.DeepCopyObject(),
		internalObjOld.DeepCopyObject(),
		schema.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"},
		"myns",
		"myname",
		schema.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"},
		"",
		admission.Update,
		&metav1.UpdateOptions{FieldManager: "foo"},
		false,
		userInfo,
	)

	testcases := []struct {
		name       string
		attrs      *generic.VersionedAttributes
		invocation *generic.WebhookInvocation

		expectRequest  func(uid types.UID) runtime.Object
		expectResponse runtime.Object
		expectErr      string
	}{
		{
			name: "no supported versions",
			invocation: &generic.WebhookInvocation{
				Webhook: webhook.NewMutatingWebhookAccessor("mywebhook", "mycfg", &admissionregistrationv1.MutatingWebhook{}),
			},
			expectErr: "webhook does not accept known AdmissionReview versions",
		},
		{
			name: "no known supported versions",
			invocation: &generic.WebhookInvocation{
				Webhook: webhook.NewMutatingWebhookAccessor("mywebhook", "mycfg", &admissionregistrationv1.MutatingWebhook{
					AdmissionReviewVersions: []string{"vX"},
				}),
			},
			expectErr: "webhook does not accept known AdmissionReview versions",
		},
		{
			name: "v1",
			attrs: &generic.VersionedAttributes{
				VersionedObject:    versionedObj.DeepCopyObject(),
				VersionedOldObject: versionedObjOld.DeepCopyObject(),
				Attributes:         attrs,
			},
			invocation: &generic.WebhookInvocation{
				Resource:    schema.GroupVersionResource{Group: "extensions", Version: "v1beta1", Resource: "deployments"},
				Subresource: "",
				Kind:        schema.GroupVersionKind{Group: "extensions", Version: "v1beta1", Kind: "Deployment"},
				Webhook: webhook.NewMutatingWebhookAccessor("mywebhook", "mycfg", &admissionregistrationv1.MutatingWebhook{
					AdmissionReviewVersions: []string{"v1", "v1beta1"},
				}),
			},
			expectRequest: func(uid types.UID) runtime.Object {
				return &admissionv1.AdmissionReview{
					Request: &admissionv1.AdmissionRequest{
						UID:                uid,
						Kind:               metav1.GroupVersionKind{Group: "extensions", Version: "v1beta1", Kind: "Deployment"},
						Resource:           metav1.GroupVersionResource{Group: "extensions", Version: "v1beta1", Resource: "deployments"},
						SubResource:        "",
						RequestKind:        &metav1.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"},
						RequestResource:    &metav1.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"},
						RequestSubResource: "",
						Name:               "myname",
						Namespace:          "myns",
						Operation:          "UPDATE",
						UserInfo: authenticationv1.UserInfo{
							Username: "myuser",
							UID:      "myuid",
							Groups:   []string{"mygroup"},
							Extra:    map[string]authenticationv1.ExtraValue{"extrakey": {"value1", "value2"}},
						},
						Object:    runtime.RawExtension{Object: versionedObj},
						OldObject: runtime.RawExtension{Object: versionedObjOld},
						DryRun:    utilpointer.BoolPtr(false),
						Options:   runtime.RawExtension{Object: &metav1.UpdateOptions{FieldManager: "foo"}},
					},
				}
			},
			expectResponse: &admissionv1.AdmissionReview{},
		},
		{
			name: "v1beta1",
			attrs: &generic.VersionedAttributes{
				VersionedObject:    versionedObj.DeepCopyObject(),
				VersionedOldObject: versionedObjOld.DeepCopyObject(),
				Attributes:         attrs,
			},
			invocation: &generic.WebhookInvocation{
				Resource:    schema.GroupVersionResource{Group: "extensions", Version: "v1beta1", Resource: "deployments"},
				Subresource: "",
				Kind:        schema.GroupVersionKind{Group: "extensions", Version: "v1beta1", Kind: "Deployment"},
				Webhook: webhook.NewMutatingWebhookAccessor("mywebhook", "mycfg", &admissionregistrationv1.MutatingWebhook{
					AdmissionReviewVersions: []string{"v1beta1", "v1"},
				}),
			},
			expectRequest: func(uid types.UID) runtime.Object {
				return &admissionv1beta1.AdmissionReview{
					Request: &admissionv1beta1.AdmissionRequest{
						UID:                uid,
						Kind:               metav1.GroupVersionKind{Group: "extensions", Version: "v1beta1", Kind: "Deployment"},
						Resource:           metav1.GroupVersionResource{Group: "extensions", Version: "v1beta1", Resource: "deployments"},
						SubResource:        "",
						RequestKind:        &metav1.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"},
						RequestResource:    &metav1.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"},
						RequestSubResource: "",
						Name:               "myname",
						Namespace:          "myns",
						Operation:          "UPDATE",
						UserInfo: authenticationv1.UserInfo{
							Username: "myuser",
							UID:      "myuid",
							Groups:   []string{"mygroup"},
							Extra:    map[string]authenticationv1.ExtraValue{"extrakey": {"value1", "value2"}},
						},
						Object:    runtime.RawExtension{Object: versionedObj},
						OldObject: runtime.RawExtension{Object: versionedObjOld},
						DryRun:    utilpointer.BoolPtr(false),
						Options:   runtime.RawExtension{Object: &metav1.UpdateOptions{FieldManager: "foo"}},
					},
				}
			},
			expectResponse: &admissionv1beta1.AdmissionReview{},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			uid, request, response, err := CreateAdmissionObjects(tc.attrs, tc.invocation)
			if err != nil {
				if len(tc.expectErr) > 0 {
					if !strings.Contains(err.Error(), tc.expectErr) {
						t.Errorf("expected error '%s', got %v", tc.expectErr, err)
					}
				} else {
					t.Errorf("unexpected error %v", err)
				}
				return
			} else if len(tc.expectErr) > 0 {
				t.Errorf("expected error '%s', got none", tc.expectErr)
				return
			}

			if len(uid) == 0 {
				t.Errorf("expected uid, got none")
			}
			if e, a := tc.expectRequest(uid), request; !reflect.DeepEqual(e, a) {
				t.Errorf("unexpected: %v", cmp.Diff(e, a))
			}
			if e, a := tc.expectResponse, response; !reflect.DeepEqual(e, a) {
				t.Errorf("unexpected: %v", cmp.Diff(e, a))
			}
		})
	}
}
