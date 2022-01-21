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

package audit

import (
	"net/http"
	"reflect"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
)

func TestLogAnnotation(t *testing.T) {
	ev := &auditinternal.Event{
		Level:   auditinternal.LevelMetadata,
		AuditID: "fake id",
	}
	LogAnnotation(ev, "foo", "bar")
	LogAnnotation(ev, "foo", "baz")
	assert.Equal(t, "bar", ev.Annotations["foo"], "audit annotation should not be overwritten.")

	LogAnnotation(ev, "qux", "")
	LogAnnotation(ev, "qux", "baz")
	assert.Equal(t, "", ev.Annotations["qux"], "audit annotation should not be overwritten.")
}

func TestMaybeTruncateUserAgent(t *testing.T) {
	req := &http.Request{}
	req.Header = http.Header{}

	ua := "short-agent"
	req.Header.Set("User-Agent", ua)
	assert.Equal(t, ua, maybeTruncateUserAgent(req))

	ua = ""
	for i := 0; i < maxUserAgentLength*2; i++ {
		ua = ua + "a"
	}
	req.Header.Set("User-Agent", ua)
	assert.NotEqual(t, ua, maybeTruncateUserAgent(req))
}

func TestCopyWithoutManagedFields(t *testing.T) {
	tests := []struct {
		name     string
		object   runtime.Object
		expected runtime.Object
		ok       bool
		err      error
	}{
		{
			name:   "object specified is not a meta.Accessor or a list or a table",
			object: &metav1.Status{},
		},
		{
			name: "object specified is a meta.Accessor and has managed fields",
			object: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo",
					Namespace: "bar",
					ManagedFields: []metav1.ManagedFieldsEntry{
						{Manager: "a", Operation: metav1.ManagedFieldsOperationUpdate, Time: &metav1.Time{Time: time.Now()}},
						{Manager: "b", Operation: metav1.ManagedFieldsOperationUpdate, Time: &metav1.Time{Time: time.Now()}},
					},
				},
			},
			expected: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo",
					Namespace: "bar",
				},
			},
			ok: true,
		},
		{
			name: "object specified is a list and its items have managed fields",
			object: &corev1.PodList{
				Items: []corev1.Pod{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "foo",
							Namespace: "ns1",
							ManagedFields: []metav1.ManagedFieldsEntry{
								{Manager: "a", Operation: metav1.ManagedFieldsOperationUpdate, Time: &metav1.Time{Time: time.Now()}},
								{Manager: "b", Operation: metav1.ManagedFieldsOperationUpdate, Time: &metav1.Time{Time: time.Now()}},
							},
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "bar",
							Namespace: "ns2",
							ManagedFields: []metav1.ManagedFieldsEntry{
								{Manager: "c", Operation: metav1.ManagedFieldsOperationUpdate, Time: &metav1.Time{Time: time.Now()}},
								{Manager: "d", Operation: metav1.ManagedFieldsOperationUpdate, Time: &metav1.Time{Time: time.Now()}},
							},
						},
					},
				},
			},
			expected: &corev1.PodList{
				Items: []corev1.Pod{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "foo",
							Namespace: "ns1",
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "bar",
							Namespace: "ns2",
						},
					},
				},
			},
			ok: true,
		},
		{
			name: "object specified is a Table and objects in its rows have managed fields",
			object: &metav1.Table{
				Rows: []metav1.TableRow{
					{
						Object: runtime.RawExtension{
							Object: &corev1.Pod{
								ObjectMeta: metav1.ObjectMeta{
									Name:      "foo",
									Namespace: "ns1",
									ManagedFields: []metav1.ManagedFieldsEntry{
										{Manager: "a", Operation: metav1.ManagedFieldsOperationUpdate, Time: &metav1.Time{Time: time.Now()}},
										{Manager: "b", Operation: metav1.ManagedFieldsOperationUpdate, Time: &metav1.Time{Time: time.Now()}},
									},
								},
							},
						},
					},
					{
						Object: runtime.RawExtension{
							Object: &corev1.Pod{
								ObjectMeta: metav1.ObjectMeta{
									Name:      "bar",
									Namespace: "ns2",
									ManagedFields: []metav1.ManagedFieldsEntry{
										{Manager: "c", Operation: metav1.ManagedFieldsOperationUpdate, Time: &metav1.Time{Time: time.Now()}},
										{Manager: "d", Operation: metav1.ManagedFieldsOperationUpdate, Time: &metav1.Time{Time: time.Now()}},
									},
								},
							},
						},
					},
					// add an empty row to make sure we don't panic
					{
						Object: runtime.RawExtension{},
					},
				},
			},
			expected: &metav1.Table{
				Rows: []metav1.TableRow{
					{
						Object: runtime.RawExtension{
							Object: &corev1.Pod{
								ObjectMeta: metav1.ObjectMeta{
									Name:      "foo",
									Namespace: "ns1",
								},
							},
						},
					},
					{
						Object: runtime.RawExtension{
							Object: &corev1.Pod{
								ObjectMeta: metav1.ObjectMeta{
									Name:      "bar",
									Namespace: "ns2",
								},
							},
						},
					},
					{
						Object: runtime.RawExtension{},
					},
				},
			},
			ok: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			original := test.object.DeepCopyObject()
			objectGot, ok, err := copyWithoutManagedFields(test.object)

			if test.err != err {
				t.Errorf("expected error: %v, but got: %v", test.err, err)
			}
			if test.ok != ok {
				t.Errorf("expected ok: %t, but got: %t", test.ok, ok)
			}

			switch {
			case test.expected == nil:
				if objectGot != nil {
					t.Errorf("expected the returned object to be nil, but got %#v", objectGot)
				}
			default:
				// verify that a deep copy of the specified object is made before mutating it.
				if expected, actual := reflect.ValueOf(test.object), reflect.ValueOf(objectGot); expected.Pointer() == actual.Pointer() {
					t.Error("expected the returned object to be a deep copy of the input object")
				}

				if !cmp.Equal(test.expected, objectGot) {
					t.Errorf("expected and actual do not match, diff: %s", cmp.Diff(test.expected, objectGot))
				}
			}

			// we always expect the original object to be unchanged.
			if !cmp.Equal(original, test.object) {
				t.Errorf("the original object has mutated, diff: %s", cmp.Diff(original, test.object))
			}
		})
	}
}
