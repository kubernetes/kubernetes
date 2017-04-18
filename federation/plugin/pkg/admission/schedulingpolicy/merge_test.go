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

package schedulingpolicy

import (
	"encoding/json"
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestMergeAnnotations(t *testing.T) {

	tests := []struct {
		input       *api.Pod
		annotations string
		expected    string
	}{
		{&api.Pod{}, `{"foo": "bar"}`, `{"foo": "bar"}`},
		{&api.Pod{ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{}}}, `{"foo": "bar"}`, `{"foo": "bar"}`},
		{&api.Pod{ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{"foo": "baz"}}}, `{"foo": "bar"}`, `{"foo": "bar"}`},
		{&api.Pod{ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{"baz": "qux"}}}, `{"foo": "bar"}`, `{"baz": "qux", "foo": "bar"}`},
	}

	for _, tc := range tests {

		annotations := map[string]string{}

		if err := json.Unmarshal([]byte(tc.annotations), &annotations); err != nil {
			panic(err)
		}

		expected := map[string]string{}

		if err := json.Unmarshal([]byte(tc.expected), &expected); err != nil {
			panic(err)
		}

		mergeAnnotations(tc.input, annotations)

		if !reflect.DeepEqual(tc.input.ObjectMeta.Annotations, expected) {
			t.Errorf("Expected annotations to equal %v but got: %v", expected, tc.input.ObjectMeta.Annotations)
		}
	}

}
