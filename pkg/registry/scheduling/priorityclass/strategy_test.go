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

package priorityclass

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/scheduling"
)

func TestPriorityClassStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if Strategy.NamespaceScoped() {
		t.Errorf("PriorityClass must not be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("PriorityClass should not allow create on update")
	}

	priorityClass := &scheduling.PriorityClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-class",
		},
		Value: 10,
	}

	Strategy.PrepareForCreate(ctx, priorityClass)

	errs := Strategy.Validate(ctx, priorityClass)
	if len(errs) != 0 {
		t.Errorf("unexpected error validating %v", errs)
	}

	newPriorityClass := &scheduling.PriorityClass{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "valid-class-2",
			ResourceVersion: "4",
		},
		Value: 20,
	}

	Strategy.PrepareForUpdate(ctx, newPriorityClass, priorityClass)

	errs = Strategy.ValidateUpdate(ctx, newPriorityClass, priorityClass)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}
}
