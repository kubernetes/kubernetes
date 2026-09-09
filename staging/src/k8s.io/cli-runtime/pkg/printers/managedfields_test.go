/*
Copyright 2021 The Kubernetes Authors.

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

package printers

import (
	"io"
	"testing"

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

type testResourcePrinter func(object runtime.Object, writer io.Writer) error

func (p testResourcePrinter) PrintObj(o runtime.Object, w io.Writer) error {
	return p(o, w)
}

func TestOmitManagedFieldsPrinter(t *testing.T) {
	testCases := []struct {
		name     string
		object   runtime.Object
		expected runtime.Object
	}{
		{
			name: "pod without managedFields",
			object: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pod1"},
			},
			expected: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pod1"},
			},
		},
		{
			name: "pod with managedFields",
			object: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod1",
					ManagedFields: []metav1.ManagedFieldsEntry{
						{Manager: "kubectl", Operation: metav1.ManagedFieldsOperationApply},
					},
				},
			},
			expected: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pod1"},
			},
		},
		{
			name: "pod list",
			object: &v1.PodList{
				Items: []v1.Pod{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:          "pod1",
							ManagedFields: []metav1.ManagedFieldsEntry{},
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "pod2",
							ManagedFields: []metav1.ManagedFieldsEntry{
								{Manager: "kubectl", Operation: metav1.ManagedFieldsOperationApply},
							},
						},
					},
					{ObjectMeta: metav1.ObjectMeta{Name: "pod3"}},
				},
			},
			expected: &v1.PodList{
				Items: []v1.Pod{
					{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}},
					{ObjectMeta: metav1.ObjectMeta{Name: "pod2"}},
					{ObjectMeta: metav1.ObjectMeta{Name: "pod3"}},
				},
			},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			r := require.New(t)
			delegate := func(o runtime.Object, w io.Writer) error {
				r.Equal(tc.expected, o)
				return nil
			}
			p := OmitManagedFieldsPrinter{Delegate: testResourcePrinter(delegate)}
			r.NoError(p.PrintObj(tc.object, nil))
		})
	}
}
