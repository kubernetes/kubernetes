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

package runtime

import (
	"context"
	"testing"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2/ktesting"
)

const (
	testPodUID = types.UID("pod-1")
)

func TestPodsInBindingMap(t *testing.T) {
	tests := []struct {
		name     string
		scenario func(t *testing.T, ctx context.Context, cancel context.CancelCauseFunc, m *podsInPreBindMap)
	}{
		{
			name: "add and get",
			scenario: func(t *testing.T, ctx context.Context, cancel context.CancelCauseFunc, m *podsInPreBindMap) {
				m.add(testPodUID, cancel)
				pod := m.get(testPodUID)
				if pod == nil {
					t.Fatalf("expected pod to be in map")
				}
			},
		},
		{
			name: "add, remove and get",
			scenario: func(t *testing.T, ctx context.Context, cancel context.CancelCauseFunc, m *podsInPreBindMap) {
				m.add(testPodUID, cancel)
				m.remove(testPodUID)
				if m.get(testPodUID) != nil {
					t.Errorf("expected pod to be removed from map")
				}
			},
		},
		{
			name: "Verify CancelPod logic",
			scenario: func(t *testing.T, ctx context.Context, cancel context.CancelCauseFunc, m *podsInPreBindMap) {
				m.add(testPodUID, cancel)
				pod := m.get(testPodUID)

				// First cancel should succeed
				if !pod.CancelPod("test cancel") {
					t.Errorf("First CancelPod should return true")
				}
				if ctx.Err() == nil {
					t.Errorf("Context should be cancelled")
				}

				// Second cancel should also return true
				if !pod.CancelPod("test cancel") {
					t.Errorf("Second CancelPod should return true")
				}
			},
		},
		{
			name: "Verify MarkBound logic",
			scenario: func(t *testing.T, ctx context.Context, cancel context.CancelCauseFunc, m *podsInPreBindMap) {
				m.add(testPodUID, cancel)
				pod := m.get(testPodUID)

				if !pod.MarkPrebound() {
					t.Errorf("MarkBound should return true for fresh pod")
				}
				if !pod.finished {
					t.Errorf("finished should be true")
				}

				// Try to cancel after binding
				if pod.CancelPod("test cancel") {
					t.Errorf("CancelPod should return false after MarkBound")
				}
				if ctx.Err() != nil {
					t.Errorf("Context should NOT be cancelled if MarkBound succeeded first")
				}
			},
		},
		{
			name: "Verify MarkBound fails on cancelled pod",
			scenario: func(t *testing.T, ctx context.Context, cancel context.CancelCauseFunc, m *podsInPreBindMap) {
				m.add(testPodUID, cancel)
				pod := m.get(testPodUID)

				pod.CancelPod("test cancel")
				if pod.MarkPrebound() {
					t.Errorf("MarkBound should return false on cancelled pod")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := NewPodsInPreBindMap()
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancelCause(ctx)
			defer cancel(nil)
			tt.scenario(t, ctx, cancel, m)
		})
	}
}
