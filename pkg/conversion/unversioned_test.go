/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package conversion_test

import (
	"encoding/json"
	"reflect"
	"testing"

	// TODO: Ideally we should create the necessary package structure in e.g.,
	// pkg/conversion/test/... instead of importing pkg/api here.
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
)

var status = &unversioned.Status{
	Status:  unversioned.StatusFailure,
	Code:    200,
	Reason:  unversioned.StatusReasonUnknown,
	Message: "",
}

func TestV1EncodeDecodeStatus(t *testing.T) {

	v1Codec := testapi.Default.Codec()

	encoded, err := v1Codec.Encode(status)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	typeMeta := unversioned.TypeMeta{}
	if err := json.Unmarshal(encoded, &typeMeta); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if typeMeta.Kind != "Status" {
		t.Errorf("Kind is not set to \"Status\". Got %v", string(encoded))
	}
	if typeMeta.APIVersion != "v1" {
		t.Errorf("APIVersion is not set to \"v1\". Got %v", string(encoded))
	}
	decoded, err := v1Codec.Decode(encoded)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(status, decoded) {
		t.Errorf("expected: %v, got: %v", status, decoded)
	}
}

func TestExperimentalEncodeDecodeStatus(t *testing.T) {
	// TODO: caesarxuchao: use the testapi.Experimental.Codec() once the PR that
	// moves experimental from v1 to v1alpha1 got merged.
	// expCodec := testapi.Experimental.Codec()
	expCodec := runtime.CodecFor(api.Scheme, "experimental/v1alpha1")
	encoded, err := expCodec.Encode(status)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	typeMeta := unversioned.TypeMeta{}
	if err := json.Unmarshal(encoded, &typeMeta); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if typeMeta.Kind != "Status" {
		t.Errorf("Kind is not set to \"Status\". Got %s", encoded)
	}
	if typeMeta.APIVersion != "" {
		t.Errorf("APIVersion is not set to \"\". Got %s", encoded)
	}
	decoded, err := expCodec.Decode(encoded)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(status, decoded) {
		t.Errorf("expected: %v, got: %v", status, decoded)
	}
}
