/*
Copyright 2024 The Kubernetes Authors.

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

package cbor

import (
	"fmt"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"

	"github.com/google/go-cmp/cmp"
)

func TestTranscodeRawTypes(t *testing.T) {
	for _, tc := range []struct {
		In  interface{}
		Out interface{}
	}{
		{
			In:  nil,
			Out: nil,
		},
		{
			In:  &runtime.RawExtension{Raw: []byte{0xd9, 0xd9, 0xf7, 0x07}},
			Out: &runtime.RawExtension{Raw: []byte(`7`)},
		},
		{
			In:  &runtime.RawExtension{},
			Out: &runtime.RawExtension{},
		},
		{
			In:  runtime.RawExtension{Raw: []byte{0xd9, 0xd9, 0xf7, 0x07}},
			Out: runtime.RawExtension{Raw: []byte{0xd9, 0xd9, 0xf7, 0x07}}, // not addressable
		},
		{
			In:  &[...]runtime.RawExtension{{Raw: []byte{0xd9, 0xd9, 0xf7, 0x07}}, {Raw: []byte{0xd9, 0xd9, 0xf7, 0x08}}, {Raw: []byte{0xd9, 0xd9, 0xf7, 0x09}}},
			Out: &[...]runtime.RawExtension{{Raw: []byte(`7`)}, {Raw: []byte(`8`)}, {Raw: []byte(`9`)}},
		},
		{
			In:  &[0]runtime.RawExtension{},
			Out: &[0]runtime.RawExtension{},
		},
		{
			In:  &[]runtime.RawExtension{{Raw: []byte{0xd9, 0xd9, 0xf7, 0x07}}, {Raw: []byte{0xd9, 0xd9, 0xf7, 0x08}}, {Raw: []byte{0xd9, 0xd9, 0xf7, 0x09}}},
			Out: &[]runtime.RawExtension{{Raw: []byte(`7`)}, {Raw: []byte(`8`)}, {Raw: []byte(`9`)}},
		},
		{
			In:  &[]runtime.RawExtension{},
			Out: &[]runtime.RawExtension{},
		},
		{
			In:  &[]string{"foo"},
			Out: &[]string{"foo"},
		},
		{
			In:  (*runtime.RawExtension)(nil),
			Out: (*runtime.RawExtension)(nil),
		},
		{
			In:  &struct{ I fmt.Stringer }{I: &runtime.RawExtension{Raw: []byte{0xd9, 0xd9, 0xf7, 0x07}}},
			Out: &struct{ I fmt.Stringer }{I: &runtime.RawExtension{Raw: []byte(`7`)}},
		},
		{
			In:  &struct{ I fmt.Stringer }{I: nil},
			Out: &struct{ I fmt.Stringer }{I: nil},
		},
		{
			In:  &struct{ I int64 }{I: 7},
			Out: &struct{ I int64 }{I: 7},
		},
		{
			In: &struct {
				E runtime.RawExtension
				I int64
			}{E: runtime.RawExtension{Raw: []byte{0xd9, 0xd9, 0xf7, 0x07}}, I: 7},
			Out: &struct {
				E runtime.RawExtension
				I int64
			}{E: runtime.RawExtension{Raw: []byte(`7`)}, I: 7},
		},
		{
			In: &struct {
				runtime.RawExtension
			}{RawExtension: runtime.RawExtension{Raw: []byte{0xd9, 0xd9, 0xf7, 0x07}}},
			Out: &struct {
				runtime.RawExtension
			}{RawExtension: runtime.RawExtension{Raw: []byte(`7`)}},
		},
		{
			In:  &map[string]string{"hello": "world"},
			Out: &map[string]string{"hello": "world"},
		},
		{
			In:  &map[string]runtime.RawExtension{"hello": {Raw: []byte{0xd9, 0xd9, 0xf7, 0x07}}},
			Out: &map[string]runtime.RawExtension{"hello": {Raw: []byte{0xd9, 0xd9, 0xf7, 0x07}}}, // not addressable
		},
		{
			In:  &map[string][]runtime.RawExtension{"hello": {{Raw: []byte{0xd9, 0xd9, 0xf7, 0x07}}}},
			Out: &map[string][]runtime.RawExtension{"hello": {{Raw: []byte(`7`)}}},
		},
		{
			In:  &metav1.FieldsV1{Raw: []byte{0xa0}},
			Out: &metav1.FieldsV1{Raw: []byte(`{}`)},
		},
		{
			In:  &metav1.FieldsV1{},
			Out: &metav1.FieldsV1{},
		},
		{
			In:  metav1.FieldsV1{Raw: []byte{0xa0}},
			Out: metav1.FieldsV1{Raw: []byte{0xa0}}, // not addressable
		},
		{
			In:  &[...]metav1.FieldsV1{{Raw: []byte{0xa0}}, {Raw: []byte{0xf6}}},
			Out: &[...]metav1.FieldsV1{{Raw: []byte(`{}`)}, {Raw: []byte(`null`)}},
		},
		{
			In:  &[0]metav1.FieldsV1{},
			Out: &[0]metav1.FieldsV1{},
		},
		{
			In:  &[]metav1.FieldsV1{{Raw: []byte{0xa0}}, {Raw: []byte{0xf6}}},
			Out: &[]metav1.FieldsV1{{Raw: []byte(`{}`)}, {Raw: []byte(`null`)}},
		},
		{
			In:  &[]metav1.FieldsV1{},
			Out: &[]metav1.FieldsV1{},
		},
		{
			In:  (*metav1.FieldsV1)(nil),
			Out: (*metav1.FieldsV1)(nil),
		},
		{
			In:  &struct{ I fmt.Stringer }{I: &metav1.FieldsV1{Raw: []byte{0xa0}}},
			Out: &struct{ I fmt.Stringer }{I: &metav1.FieldsV1{Raw: []byte(`{}`)}},
		},
		{
			In: &struct {
				E metav1.FieldsV1
				I int64
			}{E: metav1.FieldsV1{Raw: []byte{0xa0}}, I: 7},
			Out: &struct {
				E metav1.FieldsV1
				I int64
			}{E: metav1.FieldsV1{Raw: []byte(`{}`)}, I: 7},
		},
		{
			In: &struct {
				metav1.FieldsV1
			}{FieldsV1: metav1.FieldsV1{Raw: []byte{0xa0}}},
			Out: &struct {
				metav1.FieldsV1
			}{FieldsV1: metav1.FieldsV1{Raw: []byte(`{}`)}},
		},
		{
			In:  &map[string]metav1.FieldsV1{"hello": {Raw: []byte{0xa0}}},
			Out: &map[string]metav1.FieldsV1{"hello": {Raw: []byte{0xa0}}}, // not addressable
		},
		{
			In:  &map[string][]metav1.FieldsV1{"hello": {{Raw: []byte{0xa0}}}},
			Out: &map[string][]metav1.FieldsV1{"hello": {{Raw: []byte(`{}`)}}},
		},
	} {
		t.Run(fmt.Sprintf("%#v", tc.In), func(t *testing.T) {
			if err := transcodeRawTypes(tc.In); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if diff := cmp.Diff(tc.Out, tc.In); diff != "" {
				t.Errorf("unexpected diff:\n%s", diff)
			}
		})
	}
}
