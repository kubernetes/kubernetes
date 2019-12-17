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

package internal

import (
	"bytes"
	"compress/gzip"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"sigs.k8s.io/structured-merge-diff/fieldpath"
)

// EmptyFields represents a set with no paths
// It looks like metav1.Fields{Raw: []byte("{}")}
var EmptyFields metav1.FieldsV1 = func() metav1.FieldsV1 {
	f, err := SetToFields(*fieldpath.NewSet())
	if err != nil {
		panic("should never happen")
	}
	return f
}()

// EmptyFieldsV2 represents a set with no paths
// It looks like metav1.Fields{Raw: []byte("{}")}
var EmptyFieldsV2 = func() []byte {
	f, err := SetToFieldsV2(*fieldpath.NewSet())
	if err != nil {
		panic("should never happen")
	}
	return f
}()

// FieldsToSet creates a set paths from an input trie of fields
func FieldsToSet(f metav1.FieldsV1) (s fieldpath.Set, err error) {
	err = s.FromJSON(bytes.NewReader(f.Raw))
	return s, err
}

// SetToFields creates a trie of fields from an input set of paths
func SetToFields(s fieldpath.Set) (f metav1.FieldsV1, err error) {
	f.Raw, err = s.ToJSON()
	return f, err
}

// FieldsToSetV2 TODO
func FieldsToSetV2(b []byte) (s fieldpath.Set, err error) {
	r, err := gzip.NewReader(bytes.NewReader(b))
	if err != nil {
		return s, err
	}
	defer r.Close()
	err = s.FromJSON(r)
	return s, err
}

// SetToFieldsV2 TODO
func SetToFieldsV2(s fieldpath.Set) ([]byte, error) {
	buf := bytes.Buffer{}
	w := gzip.NewWriter(&buf)
	defer w.Close()
	err := s.ToJSONStream_V2Experimental(w)
	if err != nil {
		return nil, err
	}
	err = w.Flush()
	if err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
