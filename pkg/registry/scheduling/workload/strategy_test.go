/*
Copyright 2025 The Kubernetes Authors.

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

package workload

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/scheduling"
)

var workload = &scheduling.Workload{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "foo",
		Namespace: metav1.NamespaceDefault,
	},
	Spec: scheduling.WorkloadSpec{
		PodGroups: []scheduling.PodGroup{
			{
				Name: "bar",
				Policy: scheduling.PodGroupPolicy{
					Gang: &scheduling.GangSchedulingPolicy{
						MinCount: 5,
					},
				},
			},
		},
	},
}

func TestWorkloadStrategy(t *testing.T) {
	if !Strategy.NamespaceScoped() {
		t.Errorf("Workload must be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("Workload should not allow create on update")
	}
}

func TestPodSchedulingStrategyCreate(t *testing.T) {
	t.Run("simple", func(t *testing.T) {
		ctx := genericapirequest.NewDefaultContext()
		workload := workload.DeepCopy()

		Strategy.PrepareForCreate(ctx, workload)
		errs := Strategy.Validate(ctx, workload)
		if len(errs) != 0 {
			t.Errorf("Unexpected validation error: %v", errs)
		}
	})

	t.Run("failed validation", func(t *testing.T) {
		ctx := genericapirequest.NewDefaultContext()
		workload := workload.DeepCopy()
		workload.Spec.PodGroups[0].Policy.Gang.MinCount = -1

		Strategy.PrepareForCreate(ctx, workload)
		errs := Strategy.Validate(ctx, workload)
		if len(errs) == 0 {
			t.Errorf("Expected validation error")
		}
	})
}

func TestPodSchedulingStrategyUpdate(t *testing.T) {
	t.Run("no changes", func(t *testing.T) {
		ctx := genericapirequest.NewDefaultContext()
		workload := workload.DeepCopy()
		newWorkload := workload.DeepCopy()
		newWorkload.ResourceVersion = "4"

		Strategy.PrepareForUpdate(ctx, newWorkload, workload)
		errs := Strategy.ValidateUpdate(ctx, newWorkload, workload)
		if len(errs) != 0 {
			t.Errorf("Unexpected validation error: %v", errs)
		}
	})

	t.Run("name update", func(t *testing.T) {
		ctx := genericapirequest.NewDefaultContext()
		workload := workload.DeepCopy()
		newWorkload := workload.DeepCopy()
		newWorkload.Name += "bar"
		newWorkload.ResourceVersion = "4"

		Strategy.PrepareForUpdate(ctx, newWorkload, workload)
		errs := Strategy.ValidateUpdate(ctx, newWorkload, workload)
		if len(errs) == 0 {
			t.Errorf("Expected validation error")
		}
	})

	t.Run("spec update", func(t *testing.T) {
		ctx := genericapirequest.NewDefaultContext()
		workload := workload.DeepCopy()
		newWorkload := workload.DeepCopy()
		newWorkload.Spec.ControllerRef = &scheduling.TypedLocalObjectReference{
			Kind: "foo",
			Name: "baz",
		}
		newWorkload.ResourceVersion = "4"

		Strategy.PrepareForUpdate(ctx, newWorkload, workload)
		errs := Strategy.ValidateUpdate(ctx, newWorkload, workload)
		if len(errs) != 0 {
			t.Errorf("Unexpected validation error: %v", errs)
		}
	})

	t.Run("invalid spec update", func(t *testing.T) {
		ctx := genericapirequest.NewDefaultContext()
		workload := workload.DeepCopy()
		newWorkload := workload.DeepCopy()
		newWorkload.Spec.PodGroups[0].Policy.Gang.MinCount = 4
		newWorkload.ResourceVersion = "4"

		Strategy.PrepareForUpdate(ctx, newWorkload, workload)
		errs := Strategy.ValidateUpdate(ctx, newWorkload, workload)
		if len(errs) == 0 {
			t.Errorf("Expected validation error")
		}
	})
}
