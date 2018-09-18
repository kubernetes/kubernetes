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

package podgroup

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/scheduling"
)

func TestPodGroupStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if !Strategy.NamespaceScoped() {
		t.Errorf("PodGroup must be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("PodGroup should not allow create on update")
	}

	podGroup := &scheduling.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "valid-group",
			Namespace: "default",
		},
		Spec: scheduling.PodGroupSpec{
			NumMember: 1,
		},
	}

	Strategy.PrepareForCreate(ctx, podGroup)

	errs := Strategy.Validate(ctx, podGroup)
	if len(errs) != 0 {
		t.Errorf("unexpected error validating %v", errs)
	}

	newPodGroup := &scheduling.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "valid-group-2",
			ResourceVersion: "4",
		},
		Spec: scheduling.PodGroupSpec{
			NumMember: 1,
		},
	}

	Strategy.PrepareForUpdate(ctx, newPodGroup, podGroup)

	errs = Strategy.ValidateUpdate(ctx, newPodGroup, podGroup)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}
}
