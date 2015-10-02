/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package apiserver

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
)

type testPatchType struct {
	unversioned.TypeMeta `json:",inline"`

	testPatchSubType `json:",inline"`
}

type testPatchSubType struct {
	StringField string `json:"theField"`
}

func (*testPatchType) IsAnAPIObject() {}

func TestPatchAnonymousField(t *testing.T) {
	originalJS := `{"kind":"testPatchType","theField":"my-value"}`
	patch := `{"theField": "changed!"}`
	expectedJS := `{"kind":"testPatchType","theField":"changed!"}`

	actualBytes, err := getPatchedJS(string(api.StrategicMergePatchType), []byte(originalJS), []byte(patch), &testPatchType{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if string(actualBytes) != expectedJS {
		t.Errorf("expected %v, got %v", expectedJS, string(actualBytes))
	}

}
