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

package util_test

import (
	"errors"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

func TestCacheCRDFinder(t *testing.T) {
	called := 0
	getter := func() ([]schema.GroupKind, error) {
		called += 1
		return nil, nil
	}
	finder := util.NewCRDFinder(getter)
	if called != 0 {
		t.Fatalf("Creating the finder shouldn't call the getter, has called = %v", called)
	}
	_, err := finder.HasCRD(schema.GroupKind{Group: "", Kind: "Pod"})
	if err != nil {
		t.Fatalf("Failed to call HasCRD: %v", err)
	}
	if called != 1 {
		t.Fatalf("First call should call the getter, has called = %v", called)
	}

	_, err = finder.HasCRD(schema.GroupKind{Group: "", Kind: "Pod"})
	if err != nil {
		t.Fatalf("Failed to call HasCRD: %v", err)
	}
	if called != 1 {
		t.Fatalf("Second call should NOT call the getter, has called = %v", called)
	}
}

func TestCRDFinderErrors(t *testing.T) {
	getter := func() ([]schema.GroupKind, error) {
		return nil, errors.New("not working")
	}
	finder := util.NewCRDFinder(getter)
	found, err := finder.HasCRD(schema.GroupKind{Group: "", Kind: "Pod"})
	if found == true {
		t.Fatalf("Found the CRD with non-working getter function")
	}
	if err == nil {
		t.Fatalf("Error in getter should be reported")
	}
}

func TestCRDFinder(t *testing.T) {
	getter := func() ([]schema.GroupKind, error) {
		return []schema.GroupKind{
			{
				Group: "crd.com",
				Kind:  "MyCRD",
			},
			{
				Group: "crd.com",
				Kind:  "MyNewCRD",
			},
		}, nil
	}
	finder := util.NewCRDFinder(getter)

	if found, _ := finder.HasCRD(schema.GroupKind{Group: "crd.com", Kind: "MyCRD"}); !found {
		t.Fatalf("Failed to find CRD MyCRD")
	}
	if found, _ := finder.HasCRD(schema.GroupKind{Group: "crd.com", Kind: "Random"}); found {
		t.Fatalf("Found crd Random that doesn't exist")
	}
}
