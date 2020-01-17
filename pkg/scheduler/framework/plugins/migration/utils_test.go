/*
Copyright 2019 The Kubernetes Authors.

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

package migration

import (
	"errors"
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
)

func TestPredicateResultToFrameworkStatus(t *testing.T) {
	tests := []struct {
		name       string
		err        error
		reasons    []predicates.PredicateFailureReason
		wantStatus *framework.Status
	}{
		{
			name: "Success",
		},
		{
			name:       "Error",
			err:        errors.New("Failed with error"),
			wantStatus: framework.NewStatus(framework.Error, "Failed with error"),
		},
		{
			name:       "Error with reason",
			err:        errors.New("Failed with error"),
			reasons:    []predicates.PredicateFailureReason{predicates.ErrDiskConflict},
			wantStatus: framework.NewStatus(framework.Error, "Failed with error"),
		},
		{
			name:       "Unschedulable",
			reasons:    []predicates.PredicateFailureReason{predicates.ErrExistingPodsAntiAffinityRulesNotMatch},
			wantStatus: framework.NewStatus(framework.Unschedulable, "node(s) didn't satisfy existing pods anti-affinity rules"),
		},
		{
			name:       "Unschedulable and Unresolvable",
			reasons:    []predicates.PredicateFailureReason{predicates.ErrDiskConflict, predicates.ErrNodeSelectorNotMatch},
			wantStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable, "node(s) didn't match node selector"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotStatus := PredicateResultToFrameworkStatus(tt.reasons, tt.err)
			if !reflect.DeepEqual(tt.wantStatus, gotStatus) {
				t.Errorf("Got status %v, want %v", gotStatus, tt.wantStatus)
			}
		})
	}
}
