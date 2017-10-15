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

package service

import (
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestGetPatchBytes(t *testing.T) {
	svc := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Finalizers: []string{"foo"},
		},
	}
	updated := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Finalizers: []string{"foo", "bar"},
		},
	}

	b, err := getPatchBytes(svc, updated)
	if err != nil {
		t.Fatal(err)
	}
	expected := `{"metadata":{"$setElementOrder/finalizers":["foo","bar"],"finalizers":["bar"]}}`
	if string(b) != expected {
		t.Errorf("getPatchBytes(%+v, %+v) = %s ; want %s", svc, updated, string(b), expected)
	}
}
