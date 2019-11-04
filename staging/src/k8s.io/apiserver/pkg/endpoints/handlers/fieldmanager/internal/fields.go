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
	"encoding/base64"
	"fmt"

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

// FieldsToSet creates a set paths from an input trie of fields
func FieldsToSet(f metav1.FieldsV1) (s fieldpath.Set, err error) {
	if len(f.Raw) == 0 {
		return s, nil
	}
	// Remove surrounding d-quotes
	br := base64.NewDecoder(base64.RawStdEncoding, bytes.NewReader(f.Raw[1:len(f.Raw)-1]))
	zr, err := gzip.NewReader(br)
	if err != nil {
		return fieldpath.Set{}, fmt.Errorf("failed to create gzip reader: %v", err)
	}
	if err := s.FromJSON(zr); err != nil {
		return fieldpath.Set{}, fmt.Errorf("failed to decode json: %v", err)
	}
	if err := zr.Close(); err != nil {
		return fieldpath.Set{}, fmt.Errorf("failed to close gzip reader: %v", err)
	}
	return s, nil
}

// SetToFields creates a trie of fields from an input set of paths
func SetToFields(s fieldpath.Set) (f metav1.FieldsV1, err error) {
	var buf bytes.Buffer
	buf.WriteRune('"')
	bw := base64.NewEncoder(base64.RawStdEncoding, &buf)
	zw := gzip.NewWriter(bw)
	b, err := s.ToJSON()
	if err != nil {
		return metav1.FieldsV1{}, fmt.Errorf("failed to encode json: %v", err)
	}
	if _, err := zw.Write(b); err != nil {
		return metav1.FieldsV1{}, fmt.Errorf("failed to gzip json: %v", err)
	}
	if err := zw.Close(); err != nil {
		return metav1.FieldsV1{}, fmt.Errorf("failed to close gzip writer: %v", err)
	}
	buf.WriteRune('"')
	f.Raw = buf.Bytes()
	return f, nil
}
