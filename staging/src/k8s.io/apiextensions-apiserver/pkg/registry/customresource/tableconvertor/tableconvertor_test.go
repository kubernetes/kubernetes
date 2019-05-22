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

package tableconvertor

import (
	"context"
	"fmt"
	"reflect"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1beta1 "k8s.io/apimachinery/pkg/apis/meta/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/client-go/util/jsonpath"
)

func Test_cellForJSONValue(t *testing.T) {
	tests := []struct {
		headerType string
		value      interface{}
		want       interface{}
	}{
		{"integer", int64(42), int64(42)},
		{"integer", float64(3.14), int64(3)},
		{"integer", true, nil},
		{"integer", "foo", nil},

		{"number", int64(42), float64(42)},
		{"number", float64(3.14), float64(3.14)},
		{"number", true, nil},
		{"number", "foo", nil},

		{"boolean", int64(42), nil},
		{"boolean", float64(3.14), nil},
		{"boolean", true, true},
		{"boolean", "foo", nil},

		{"string", int64(42), nil},
		{"string", float64(3.14), nil},
		{"string", true, nil},
		{"string", "foo", "foo"},

		{"date", int64(42), nil},
		{"date", float64(3.14), nil},
		{"date", true, nil},
		{"date", time.Now().Add(-time.Hour*12 - 30*time.Minute).UTC().Format(time.RFC3339), "12h"},
		{"date", time.Now().Add(+time.Hour*12 + 30*time.Minute).UTC().Format(time.RFC3339), "<invalid>"},
		{"date", "", "<unknown>"},

		{"unknown", "foo", nil},
	}
	for _, tt := range tests {
		t.Run(fmt.Sprintf("%#v of type %s", tt.value, tt.headerType), func(t *testing.T) {
			if got := cellForJSONValue(tt.headerType, tt.value); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("cellForJSONValue() = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func Test_convertor_ConvertToTable(t *testing.T) {
	type fields struct {
		headers           []metav1beta1.TableColumnDefinition
		additionalColumns []*jsonpath.JSONPath
	}
	type args struct {
		ctx          context.Context
		obj          runtime.Object
		tableOptions runtime.Object
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		want    *metav1beta1.Table
		wantErr bool
	}{
		{
			name: "Return table for object",
			fields: fields{
				headers: []metav1beta1.TableColumnDefinition{{Name: "name", Type: "string"}},
			},
			args: args{
				obj: &metav1beta1.PartialObjectMetadata{
					ObjectMeta: metav1.ObjectMeta{Name: "blah", CreationTimestamp: metav1.NewTime(time.Unix(1, 0))},
				},
				tableOptions: nil,
			},
			want: &metav1beta1.Table{
				ColumnDefinitions: []metav1beta1.TableColumnDefinition{{Name: "name", Type: "string"}},
				Rows: []metav1beta1.TableRow{
					{
						Cells: []interface{}{"blah"},
						Object: runtime.RawExtension{
							Object: &metav1beta1.PartialObjectMetadata{
								ObjectMeta: metav1.ObjectMeta{Name: "blah", CreationTimestamp: metav1.NewTime(time.Unix(1, 0))},
							},
						},
					},
				},
			},
		},
		{
			name: "Return table for list",
			fields: fields{
				headers: []metav1beta1.TableColumnDefinition{{Name: "name", Type: "string"}},
			},
			args: args{
				obj: &metav1beta1.PartialObjectMetadataList{
					Items: []metav1beta1.PartialObjectMetadata{
						{ObjectMeta: metav1.ObjectMeta{Name: "blah", CreationTimestamp: metav1.NewTime(time.Unix(1, 0))}},
						{ObjectMeta: metav1.ObjectMeta{Name: "blah-2", CreationTimestamp: metav1.NewTime(time.Unix(2, 0))}},
					},
				},
				tableOptions: nil,
			},
			want: &metav1beta1.Table{
				ColumnDefinitions: []metav1beta1.TableColumnDefinition{{Name: "name", Type: "string"}},
				Rows: []metav1beta1.TableRow{
					{
						Cells: []interface{}{"blah"},
						Object: runtime.RawExtension{
							Object: &metav1beta1.PartialObjectMetadata{
								ObjectMeta: metav1.ObjectMeta{Name: "blah", CreationTimestamp: metav1.NewTime(time.Unix(1, 0))},
							},
						},
					},
					{
						Cells: []interface{}{"blah-2"},
						Object: runtime.RawExtension{
							Object: &metav1beta1.PartialObjectMetadata{
								ObjectMeta: metav1.ObjectMeta{Name: "blah-2", CreationTimestamp: metav1.NewTime(time.Unix(2, 0))},
							},
						},
					},
				},
			},
		},
		{
			name: "Accept TableOptions",
			fields: fields{
				headers: []metav1beta1.TableColumnDefinition{{Name: "name", Type: "string"}},
			},
			args: args{
				obj: &metav1beta1.PartialObjectMetadata{
					ObjectMeta: metav1.ObjectMeta{Name: "blah", CreationTimestamp: metav1.NewTime(time.Unix(1, 0))},
				},
				tableOptions: &metav1beta1.TableOptions{},
			},
			want: &metav1beta1.Table{
				ColumnDefinitions: []metav1beta1.TableColumnDefinition{{Name: "name", Type: "string"}},
				Rows: []metav1beta1.TableRow{
					{
						Cells: []interface{}{"blah"},
						Object: runtime.RawExtension{
							Object: &metav1beta1.PartialObjectMetadata{
								ObjectMeta: metav1.ObjectMeta{Name: "blah", CreationTimestamp: metav1.NewTime(time.Unix(1, 0))},
							},
						},
					},
				},
			},
		},
		{
			name: "Omit headers from TableOptions",
			fields: fields{
				headers: []metav1beta1.TableColumnDefinition{{Name: "name", Type: "string"}},
			},
			args: args{
				obj: &metav1beta1.PartialObjectMetadata{
					ObjectMeta: metav1.ObjectMeta{Name: "blah", CreationTimestamp: metav1.NewTime(time.Unix(1, 0))},
				},
				tableOptions: &metav1beta1.TableOptions{NoHeaders: true},
			},
			want: &metav1beta1.Table{
				Rows: []metav1beta1.TableRow{
					{
						Cells: []interface{}{"blah"},
						Object: runtime.RawExtension{
							Object: &metav1beta1.PartialObjectMetadata{
								ObjectMeta: metav1.ObjectMeta{Name: "blah", CreationTimestamp: metav1.NewTime(time.Unix(1, 0))},
							},
						},
					},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &convertor{
				headers:           tt.fields.headers,
				additionalColumns: tt.fields.additionalColumns,
			}
			got, err := c.ConvertToTable(tt.args.ctx, tt.args.obj, tt.args.tableOptions)
			if (err != nil) != tt.wantErr {
				t.Errorf("convertor.ConvertToTable() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("convertor.ConvertToTable() = %s", diff.ObjectReflectDiff(tt.want, got))
			}
		})
	}
}
