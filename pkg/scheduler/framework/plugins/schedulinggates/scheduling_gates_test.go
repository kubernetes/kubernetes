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

package schedulinggates

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestPreEnqueue(t *testing.T) {
	tests := []struct {
		name string
		pod  *v1.Pod
		want *fwk.Status
	}{
		{
			name: "pod does not carry scheduling gates",
			pod:  st.MakePod().Name("p").Obj(),
			want: nil,
		},
		{
			name: "pod carries scheduling gates",
			pod:  st.MakePod().Name("p").SchedulingGates([]string{"foo", "bar"}).Obj(),
			want: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "waiting for scheduling gates: [foo bar]"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			p, err := New(ctx, nil, nil, feature.Features{})
			if err != nil {
				t.Fatalf("Creating plugin: %v", err)
			}

			got := p.(fwk.PreEnqueuePlugin).PreEnqueue(ctx, tt.pod)
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("unexpected status (-want, +got):\n%s", diff)
			}
		})
	}
}

func TestIsSchedulableAfterUpdateTargetPodSchedulingGatesEliminated(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	p, err := New(ctx, nil, nil, feature.Features{})
	if err != nil {
		t.Fatalf("Creating plugin: %v", err)
	}

	pod := st.MakePod().Name("p").SchedulingGates([]string{"foo"}).Obj()
	oldObj := st.MakePod().Name("p").SchedulingGates([]string{"foo"}).Obj()
	newObj := st.MakePod().Name("p").Obj()

	actualHint, err := p.(*SchedulingGates).isSchedulableAfterUpdateTargetPodSchedulingGatesEliminated(logger, pod, oldObj, newObj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if diff := cmp.Diff(fwk.Queue, actualHint); diff != "" {
		t.Errorf("unexpected hint (-want, +got):\n%s", diff)
	}
}
