/*
Copyright 2018 The Kubernetes Authors.

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

package editor

import (
	"bytes"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/rest/fake"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
)

func TestHashOnLineBreak(t *testing.T) {
	tests := []struct {
		original string
		expected string
	}{
		{
			original: "",
			expected: "",
		},
		{
			original: "\n",
			expected: "\n",
		},
		{
			original: "a\na\na\n",
			expected: "a\n# a\n# a\n",
		},
		{
			original: "a\n\n\na\n\n",
			expected: "a\n# \n# \n# a\n# \n",
		},
	}
	for _, test := range tests {
		r := hashOnLineBreak(test.original)
		if r != test.expected {
			t.Errorf("expected: %s, saw: %s", test.expected, r)
		}
	}
}

func TestManagedFieldsExtractAndRestore(t *testing.T) {
	tests := map[string]struct {
		object        runtime.Object
		managedFields map[types.UID][]metav1.ManagedFieldsEntry
	}{
		"single object, empty managedFields": {
			object: &corev1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("12345")}},
			managedFields: map[types.UID][]metav1.ManagedFieldsEntry{
				types.UID("12345"): nil,
			},
		},
		"multiple objects, empty managedFields": {
			object: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"kind":       "List",
					"apiVersion": "v1",
					"metadata":   map[string]interface{}{},
				},
				Items: []unstructured.Unstructured{
					{
						Object: map[string]interface{}{
							"apiVersion": "v1",
							"kind":       "Pod",
							"metadata": map[string]interface{}{
								"uid": "12345",
							},
							"spec":   map[string]interface{}{},
							"status": map[string]interface{}{},
						},
					},
					{
						Object: map[string]interface{}{
							"apiVersion": "v1",
							"kind":       "Pod",
							"metadata": map[string]interface{}{
								"uid": "98765",
							},
							"spec":   map[string]interface{}{},
							"status": map[string]interface{}{},
						},
					},
				},
			},
			managedFields: map[types.UID][]metav1.ManagedFieldsEntry{
				types.UID("12345"): nil,
				types.UID("98765"): nil,
			},
		},
		"single object, all managedFields": {
			object: &corev1.Pod{ObjectMeta: metav1.ObjectMeta{
				UID: types.UID("12345"),
				ManagedFields: []metav1.ManagedFieldsEntry{
					{
						Manager:   "test",
						Operation: metav1.ManagedFieldsOperationApply,
					},
				},
			}},
			managedFields: map[types.UID][]metav1.ManagedFieldsEntry{
				types.UID("12345"): {
					{
						Manager:   "test",
						Operation: metav1.ManagedFieldsOperationApply,
					},
				},
			},
		},
		"multiple objects, all managedFields": {
			object: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"kind":       "List",
					"apiVersion": "v1",
					"metadata":   map[string]interface{}{},
				},
				Items: []unstructured.Unstructured{
					{
						Object: map[string]interface{}{
							"apiVersion": "v1",
							"kind":       "Pod",
							"metadata": map[string]interface{}{
								"uid": "12345",
								"managedFields": []interface{}{
									map[string]interface{}{
										"manager":   "test",
										"operation": "Apply",
									},
								},
							},
							"spec":   map[string]interface{}{},
							"status": map[string]interface{}{},
						},
					},
					{
						Object: map[string]interface{}{
							"apiVersion": "v1",
							"kind":       "Pod",
							"metadata": map[string]interface{}{
								"uid": "98765",
								"managedFields": []interface{}{
									map[string]interface{}{
										"manager":   "test",
										"operation": "Update",
									},
								},
							},
							"spec":   map[string]interface{}{},
							"status": map[string]interface{}{},
						},
					},
				},
			},
			managedFields: map[types.UID][]metav1.ManagedFieldsEntry{
				types.UID("12345"): {
					{
						Manager:   "test",
						Operation: metav1.ManagedFieldsOperationApply,
					},
				},
				types.UID("98765"): {
					{
						Manager:   "test",
						Operation: metav1.ManagedFieldsOperationUpdate,
					},
				},
			},
		},
		"multiple objects, some managedFields": {
			object: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"kind":       "List",
					"apiVersion": "v1",
					"metadata":   map[string]interface{}{},
				},
				Items: []unstructured.Unstructured{
					{
						Object: map[string]interface{}{
							"apiVersion": "v1",
							"kind":       "Pod",
							"metadata": map[string]interface{}{
								"uid": "12345",
								"managedFields": []interface{}{
									map[string]interface{}{
										"manager":   "test",
										"operation": "Apply",
									},
								},
							},
							"spec":   map[string]interface{}{},
							"status": map[string]interface{}{},
						},
					},
					{
						Object: map[string]interface{}{
							"apiVersion": "v1",
							"kind":       "Pod",
							"metadata": map[string]interface{}{
								"uid": "98765",
							},
							"spec":   map[string]interface{}{},
							"status": map[string]interface{}{},
						},
					},
				},
			},
			managedFields: map[types.UID][]metav1.ManagedFieldsEntry{
				types.UID("12345"): {
					{
						Manager:   "test",
						Operation: metav1.ManagedFieldsOperationApply,
					},
				},
				types.UID("98765"): nil,
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			// operate on a copy, so we can compare the original and the modified object
			objCopy := test.object.DeepCopyObject()
			var infos []*resource.Info
			o := NewEditOptions(NormalEditMode, genericiooptions.NewTestIOStreamsDiscard())
			err := o.extractManagedFields(objCopy)
			if err != nil {
				t.Errorf("unexpected extraction error %v", err)
			}
			if meta.IsListType(objCopy) {
				infos = []*resource.Info{}
				meta.EachListItem(objCopy, func(obj runtime.Object) error {
					metaObjs, _ := meta.Accessor(obj)
					if metaObjs.GetManagedFields() != nil {
						t.Errorf("unexpected managedFileds after extraction")
					}
					infos = append(infos, &resource.Info{Object: obj})
					return nil
				})
			} else {
				metaObjs, _ := meta.Accessor(objCopy)
				if metaObjs.GetManagedFields() != nil {
					t.Errorf("unexpected managedFileds after extraction")
				}
				infos = []*resource.Info{{Object: objCopy}}
			}

			err = o.restoreManagedFields(infos)
			if err != nil {
				t.Errorf("unexpected restore error %v", err)
			}
			if !reflect.DeepEqual(test.object, objCopy) {
				t.Errorf("mismatched object after extract and restore managedFields: %#v", objCopy)
			}
			if test.managedFields != nil && !reflect.DeepEqual(test.managedFields, o.managedFields) {
				t.Errorf("mismatched managedFields %#v vs %#v", test.managedFields, o.managedFields)
			}
		})
	}
}

type testVisitor struct {
	updatedInfos []*resource.Info
}

func (tv *testVisitor) Visit(f resource.VisitorFunc) error {
	var err error
	for _, ui := range tv.updatedInfos {
		err = f(ui, err)
	}
	return err
}

var unregMapping = &meta.RESTMapping{
	Resource: schema.GroupVersionResource{
		Group:    "a",
		Version:  "b",
		Resource: "c",
	},
	GroupVersionKind: schema.GroupVersionKind{
		Group:   "a",
		Version: "b",
		Kind:    "d",
	},
}

func TestEditOptions_visitToPatch(t *testing.T) {

	expectedErr := func(err error) bool {
		return err != nil && strings.Contains(err.Error(), "At least one of apiVersion, kind and name was changed")
	}

	type args struct {
		originalInfos []*resource.Info
		patchVisitor  resource.Visitor
		results       *editResults
	}
	tests := []struct {
		name     string
		args     args
		checkErr func(err error) bool
	}{
		{
			name: "name-diff",
			args: args{
				originalInfos: []*resource.Info{
					{
						Namespace: "ns",
						Name:      "before",
						Object: &unstructured.Unstructured{
							Object: map[string]interface{}{
								"apiVersion": "v1",
								"kind":       "Thingy",
								"metadata": map[string]interface{}{
									"uid":       "12345",
									"namespace": "ns",
									"name":      "before",
								},
								"spec": map[string]interface{}{},
							},
						},
						Mapping: unregMapping,
					},
				},
				patchVisitor: &testVisitor{
					updatedInfos: []*resource.Info{
						{
							Namespace: "ns",
							Name:      "after",
							Object: &unstructured.Unstructured{
								Object: map[string]interface{}{
									"apiVersion": "v1",
									"kind":       "Thingy",
									"metadata": map[string]interface{}{
										"uid":       "12345",
										"namespace": "ns",
										"name":      "after",
									},
									"spec": map[string]interface{}{},
								},
							},
							Mapping: unregMapping,
						},
					},
				},
				results: &editResults{},
			},
			checkErr: expectedErr,
		},
		{
			name: "kind-diff",
			args: args{
				originalInfos: []*resource.Info{
					{
						Namespace: "ns",
						Name:      "myname",
						Object: &unstructured.Unstructured{
							Object: map[string]interface{}{
								"apiVersion": "v1",
								"kind":       "Thingy",
								"metadata": map[string]interface{}{
									"uid":       "12345",
									"namespace": "ns",
									"name":      "myname",
								},
								"spec": map[string]interface{}{},
							},
						},
						Mapping: unregMapping,
					},
				},
				patchVisitor: &testVisitor{
					updatedInfos: []*resource.Info{
						{
							Namespace: "ns",
							Name:      "myname",
							Object: &unstructured.Unstructured{
								Object: map[string]interface{}{
									"apiVersion": "v1",
									"kind":       "OtherThingy",
									"metadata": map[string]interface{}{
										"uid":       "12345",
										"namespace": "ns",
										"name":      "myname",
									},
									"spec": map[string]interface{}{},
								},
							},
							Mapping: unregMapping,
						},
					},
				},
				results: &editResults{},
			},
			checkErr: expectedErr,
		},
		{
			name: "apiver-diff",
			args: args{
				originalInfos: []*resource.Info{
					{
						Namespace: "ns",
						Name:      "myname",
						Object: &unstructured.Unstructured{
							Object: map[string]interface{}{
								"apiVersion": "v1",
								"kind":       "Thingy",
								"metadata": map[string]interface{}{
									"uid":       "12345",
									"namespace": "ns",
									"name":      "myname",
								},
								"spec": map[string]interface{}{},
							},
						},
						Mapping: unregMapping,
					},
				},
				patchVisitor: &testVisitor{
					updatedInfos: []*resource.Info{
						{
							Namespace: "ns",
							Name:      "myname",
							Object: &unstructured.Unstructured{
								Object: map[string]interface{}{
									"apiVersion": "v1alpha1",
									"kind":       "Thingy",
									"metadata": map[string]interface{}{
										"uid":       "12345",
										"namespace": "ns",
										"name":      "myname",
									},
									"spec": map[string]interface{}{},
								},
							},
							Mapping: unregMapping,
						},
					},
				},
				results: &editResults{},
			},
			checkErr: expectedErr,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			o := &EditOptions{}
			if err := o.visitToPatch(tt.args.originalInfos, tt.args.patchVisitor, tt.args.results); !tt.checkErr(err) {
				t.Errorf("EditOptions.visitToPatch() %s error = %v", tt.name, err)
			}
		})
	}
}

func TestEditOptions_visitToPatchServerSideApply(t *testing.T) {
	deploymentMapping := &meta.RESTMapping{
		Resource: schema.GroupVersionResource{
			Group: "apps", Version: "v1", Resource: "deployments",
		},
		GroupVersionKind: schema.GroupVersionKind{
			Group: "apps", Version: "v1", Kind: "Deployment",
		},
		Scope: meta.RESTScopeNamespace,
	}

	originalObj := func() *unstructured.Unstructured {
		return &unstructured.Unstructured{
			Object: map[string]interface{}{
				"apiVersion": "apps/v1",
				"kind":       "Deployment",
				"metadata": map[string]interface{}{
					"uid":       "abc-123",
					"namespace": "ns",
					"name":      "nginx",
					"managedFields": []interface{}{
						map[string]interface{}{
							"manager":    "kubectl-create",
							"operation":  "Update",
							"apiVersion": "apps/v1",
							"fieldsType": "FieldsV1",
							"fieldsV1":   map[string]interface{}{},
						},
					},
				},
				"spec": map[string]interface{}{
					"replicas": int64(1),
					"selector": map[string]interface{}{
						"matchLabels": map[string]interface{}{
							"app": "nginx",
						},
					},
				},
			},
		}
	}

	editedObj := func() *unstructured.Unstructured {
		obj := originalObj()
		obj.Object["spec"].(map[string]interface{})["replicas"] = int64(3)
		return obj
	}

	tests := []struct {
		name                  string
		forceConflicts        bool
		serverStatus          int
		serverBody            string
		wantForceConflictHint bool
		checkRequest          func(t *testing.T, req *http.Request, body []byte)
	}{
		{
			name:         "full-object-without-managed-fields",
			serverStatus: http.StatusOK,
			serverBody:   `{"apiVersion":"apps/v1","kind":"Deployment","metadata":{"name":"nginx","namespace":"ns","uid":"abc-123"},"spec":{"replicas":3}}`,
			checkRequest: func(t *testing.T, req *http.Request, body []byte) {
				if got := req.Header.Get("Content-Type"); got != string(types.ApplyPatchType) {
					t.Errorf("Content-Type = %q, want %q", got, types.ApplyPatchType)
				}
				if req.Method != http.MethodPatch {
					t.Errorf("Method = %q, want PATCH", req.Method)
				}
				if got := req.URL.Query().Get("fieldManager"); got != "kubectl-edit" {
					t.Errorf("fieldManager = %q, want kubectl-edit", got)
				}
				if got := req.URL.Query().Get("force"); got == "true" {
					t.Errorf("force should not be true when ForceConflicts=false, got %q", got)
				}

				sent := &unstructured.Unstructured{}
				if err := sent.UnmarshalJSON(body); err != nil {
					t.Fatalf("body is not valid JSON: %v", err)
				}
				if mf, found, _ := unstructured.NestedSlice(sent.Object, "metadata", "managedFields"); found || mf != nil {
					t.Errorf("managedFields must be stripped from SSA request body")
				}
				replicas, _, _ := unstructured.NestedInt64(sent.Object, "spec", "replicas")
				if replicas != 3 {
					t.Errorf("replicas = %d, want 3 (edited value)", replicas)
				}
				app, found, err := unstructured.NestedString(sent.Object, "spec", "selector", "matchLabels", "app")
				if err != nil {
					t.Fatalf("failed to read unchanged selector: %v", err)
				}
				if !found {
					t.Fatal("unchanged selector was not included in SSA request")
				}
				if app != "nginx" {
					t.Errorf("selector app = %q, want %q", app, "nginx")
				}
			},
		},
		{
			name:                  "field-manager-conflict",
			serverStatus:          http.StatusConflict,
			serverBody:            `{"kind":"Status","apiVersion":"v1","status":"Failure","message":"Apply failed with 1 conflict: conflict with \"kubectl-create\" using apps/v1: .spec.replicas","reason":"Conflict","details":{"name":"nginx","group":"apps","kind":"deployments","causes":[{"reason":"FieldManagerConflict","message":"conflict with \"kubectl-create\"","field":".spec.replicas"}]},"code":409}`,
			wantForceConflictHint: true,
			checkRequest: func(t *testing.T, req *http.Request, body []byte) {
				if got := req.URL.Query().Get("force"); got == "true" {
					t.Errorf("force should not be true when ForceConflicts=false, got %q", got)
				}
			},
		},
		{
			name:         "non-field-manager-conflict",
			serverStatus: http.StatusConflict,
			serverBody:   `{"kind":"Status","apiVersion":"v1","status":"Failure","message":"the object has been modified","reason":"Conflict","code":409}`,
			checkRequest: func(t *testing.T, req *http.Request, body []byte) {
				if got := req.URL.Query().Get("force"); got == "true" {
					t.Errorf("force should not be true when ForceConflicts=false, got %q", got)
				}
			},
		},
		{
			name:           "force-conflicts",
			forceConflicts: true,
			serverStatus:   http.StatusOK,
			serverBody:     `{"apiVersion":"apps/v1","kind":"Deployment","metadata":{"name":"nginx","namespace":"ns","uid":"abc-123"},"spec":{"replicas":3}}`,
			checkRequest: func(t *testing.T, req *http.Request, body []byte) {
				if got := req.URL.Query().Get("force"); got != "true" {
					t.Errorf("force = %q, want true", got)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				capturedReq  *http.Request
				capturedBody []byte
			)

			fakeClient := &fake.RESTClient{
				NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
				GroupVersion:         schema.GroupVersion{Group: "apps", Version: "v1"},
				VersionedAPIPath:     "/apis/apps/v1",
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					capturedReq = req
					if req.Body != nil {
						capturedBody, _ = io.ReadAll(req.Body)
					}
					return &http.Response{
						StatusCode: tt.serverStatus,
						Header:     cmdtesting.DefaultHeader(),
						Body:       io.NopCloser(bytes.NewReader([]byte(tt.serverBody))),
					}, nil
				}),
			}

			originalInfo := &resource.Info{
				Client:    fakeClient,
				Mapping:   deploymentMapping,
				Namespace: "ns",
				Name:      "nginx",
				Object:    originalObj(),
			}
			updatedInfo := &resource.Info{
				Client:    fakeClient,
				Mapping:   deploymentMapping,
				Namespace: "ns",
				Name:      "nginx",
				Object:    editedObj(),
			}

			ioStreams, _, _, errOut := genericiooptions.NewTestIOStreams()
			o := &EditOptions{
				IOStreams:       ioStreams,
				ServerSideApply: true,
				ForceConflicts:  tt.forceConflicts,
				FieldManager:    "kubectl-edit",
				ToPrinter: func(string) (printers.ResourcePrinter, error) {
					return printers.NewDiscardingPrinter(), nil
				},
			}

			if err := o.visitToPatch([]*resource.Info{originalInfo}, &testVisitor{updatedInfos: []*resource.Info{updatedInfo}}, &editResults{}); err != nil {
				t.Fatalf("visitToPatch() error = %v", err)
			}

			gotForceConflictHint := strings.Contains(errOut.String(), "--force-conflicts")
			if gotForceConflictHint != tt.wantForceConflictHint {
				t.Errorf("stderr contains --force-conflicts = %t, want %t; stderr: %q", gotForceConflictHint, tt.wantForceConflictHint, errOut.String())
			}

			if capturedReq == nil {
				t.Fatalf("no HTTP request was captured")
			}
			if tt.checkRequest != nil {
				tt.checkRequest(t, capturedReq, capturedBody)
			}
		})
	}
}
