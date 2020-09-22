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

package printers

import (
	"bytes"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

func TestNamePrinter_PrintObj(t *testing.T) {
	type fields struct {
		ShortOutput   bool
		Operation     string
		WithNamespace bool
	}
	type args struct {
		obj runtime.Object
	}

	tests := []struct {
		name    string
		fields  fields
		args    args
		wantW   string
		wantErr bool
	}{
		{
			name:   "print object without options",
			fields: fields{},
			args: args{
				obj: &corev1.Pod{
					TypeMeta: metav1.TypeMeta{
						Kind: "Pod",
					},
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-pod",
						Namespace: "test-namespace",
					},
				},
			},
			wantW:   "pod/test-pod\n",
			wantErr: false,
		},
		{
			name: "print object with operation",
			fields: fields{
				Operation: "created",
			},
			args: args{
				obj: &corev1.Pod{
					TypeMeta: metav1.TypeMeta{
						Kind: "Pod",
					},
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-pod",
						Namespace: "test-namespace",
					},
				},
			},
			wantW:   "pod/test-pod created\n",
			wantErr: false,
		},
		{
			name: "print object with namespace",
			fields: fields{
				WithNamespace: true,
			},
			args: args{
				obj: &corev1.Pod{
					TypeMeta: metav1.TypeMeta{
						Kind: "Pod",
					},
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-pod",
						Namespace: "test-namespace",
					},
				},
			},
			wantW:   "test-namespace/pod/test-pod\n",
			wantErr: false,
		},
	}

	for _, tt := range tests {
		tt := tt

		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			p := &NamePrinter{
				ShortOutput:   tt.fields.ShortOutput,
				Operation:     tt.fields.Operation,
				WithNamespace: tt.fields.WithNamespace,
			}

			w := &bytes.Buffer{}

			if err := p.PrintObj(tt.args.obj, w); (err != nil) != tt.wantErr {
				t.Errorf("NamePrinter.PrintObj() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if gotW := w.String(); gotW != tt.wantW {
				t.Errorf("NamePrinter.PrintObj() = %v, want %v", gotW, tt.wantW)
			}
		})
	}
}
