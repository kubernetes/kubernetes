/*
Copyright 2014 Google Inc. All rights reserved.

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

package v1beta2_test

import (
	"testing"

	newer "github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta2"
)

func TestServiceEmptySelector(t *testing.T) {
	// Nil map should be preserved
	svc := &v1beta2.Service{Selector: nil}
	data, err := newer.Scheme.EncodeToVersion(svc, "v1beta2")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	obj, err := newer.Scheme.Decode(data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	selector := obj.(*newer.Service).Spec.Selector
	if selector != nil {
		t.Errorf("unexpected selector: %#v", obj)
	}

	// Empty map should be preserved
	svc2 := &v1beta2.Service{Selector: map[string]string{}}
	data, err = newer.Scheme.EncodeToVersion(svc2, "v1beta2")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	obj, err = newer.Scheme.Decode(data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	selector = obj.(*newer.Service).Spec.Selector
	if selector == nil || len(selector) != 0 {
		t.Errorf("unexpected selector: %#v", obj)
	}
}
