/*
Copyright 2022 The Kubernetes Authors.

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

package podscheduling

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/core"
)

func TestPodSchedulingStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if !Strategy.NamespaceScoped() {
		t.Errorf("PodScheduling must be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("PodScheduling should not allow create on update")
	}

	podScheduling := &core.PodScheduling{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "valid-pod",
			Namespace: "default",
		},
		Spec: core.PodSchedulingSpec{
			SelectedNode: "worker",
		},
	}

	Strategy.PrepareForCreate(ctx, podScheduling)

	errs := Strategy.Validate(ctx, podScheduling)
	if len(errs) != 0 {
		t.Errorf("unexpected error validating for create %v", errs)
	}

	// Update with no changes is okay.
	newPodScheduling := podScheduling.DeepCopy()
	newPodScheduling.ResourceVersion = "4"

	Strategy.PrepareForUpdate(ctx, newPodScheduling, podScheduling)

	errs = Strategy.ValidateUpdate(ctx, newPodScheduling, podScheduling)
	if len(errs) != 0 {
		t.Errorf("unexpected validation errors: %v", errs)
	}

	// Changing name not allowed.
	newPodScheduling = podScheduling.DeepCopy()
	newPodScheduling.Name = "valid-claim-2"
	newPodScheduling.ResourceVersion = "4"

	Strategy.PrepareForUpdate(ctx, newPodScheduling, podScheduling)

	errs = Strategy.ValidateUpdate(ctx, newPodScheduling, podScheduling)
	if len(errs) == 0 {
		t.Errorf("expected a validation error")
	}
}
