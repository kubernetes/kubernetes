/*
Copyright 2021 The Kubernetes Authors.

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

package rest

import (
	"context"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilpointer "k8s.io/utils/pointer"
	"k8s.io/utils/ptr"
)

type mockStrategy struct {
	RESTDeleteStrategy
	RESTGracefulDeleteStrategy
}

func (m mockStrategy) ObjectKinds(obj runtime.Object) ([]schema.GroupVersionKind, bool, error) {
	gvk := obj.GetObjectKind().GroupVersionKind()
	if len(gvk.Kind) == 0 {
		return nil, false, runtime.NewMissingKindErr("object has no kind field ")
	}
	if len(gvk.Version) == 0 {
		return nil, false, runtime.NewMissingVersionErr("object has no apiVersion field")
	}
	return []schema.GroupVersionKind{obj.GetObjectKind().GroupVersionKind()}, false, nil
}

func TestBeforeDelete(t *testing.T) {
	type args struct {
		strategy RESTDeleteStrategy
		ctx      context.Context
		pod      *v1.Pod
		options  *metav1.DeleteOptions
	}

	makePod := func(deletionGracePeriodSeconds int64) *v1.Pod {
		return &v1.Pod{
			TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "Pod"},
			ObjectMeta: metav1.ObjectMeta{
				DeletionTimestamp:          &metav1.Time{},
				DeletionGracePeriodSeconds: &deletionGracePeriodSeconds,
			},
		}
	}
	makeOption := func(gracePeriodSeconds int64) *metav1.DeleteOptions {
		return &metav1.DeleteOptions{
			GracePeriodSeconds: &gracePeriodSeconds,
		}
	}

	tests := []struct {
		name                           string
		args                           args
		wantGraceful                   bool
		wantGracefulPending            bool
		wantGracePeriodSeconds         *int64
		wantDeletionGracePeriodSeconds *int64
		wantErr                        bool
	}{
		{
			name: "when DeletionGracePeriodSeconds=-1, GracePeriodSeconds=-1",
			args: args{
				pod:     makePod(-1),
				options: makeOption(-1),
			},
			// want 1
			wantDeletionGracePeriodSeconds: utilpointer.Int64(1),
			wantGracePeriodSeconds:         utilpointer.Int64(1),
			wantGraceful:                   false,
			wantGracefulPending:            true,
		},
		{
			name: "when DeletionGracePeriodSeconds=-1, GracePeriodSeconds=0",
			args: args{
				pod:     makePod(-1),
				options: makeOption(0),
			},
			// want 0
			wantDeletionGracePeriodSeconds: utilpointer.Int64(0),
			wantGracePeriodSeconds:         utilpointer.Int64(0),
			wantGraceful:                   true,
			wantGracefulPending:            false,
		},
		{
			name: "when DeletionGracePeriodSeconds=-1, GracePeriodSeconds=1",
			args: args{
				pod:     makePod(-1),
				options: makeOption(1),
			},
			// want 1
			wantDeletionGracePeriodSeconds: utilpointer.Int64(1),
			wantGracePeriodSeconds:         utilpointer.Int64(1),
			wantGraceful:                   false,
			wantGracefulPending:            true,
		},
		{
			name: "when DeletionGracePeriodSeconds=-1, GracePeriodSeconds=2",
			args: args{
				pod:     makePod(-1),
				options: makeOption(2),
			},
			// want 1
			wantDeletionGracePeriodSeconds: utilpointer.Int64(1),
			wantGracePeriodSeconds:         utilpointer.Int64(2),
			wantGraceful:                   false,
			wantGracefulPending:            true,
		},

		{
			name: "when DeletionGracePeriodSeconds=0, GracePeriodSeconds=-1",
			args: args{
				pod:     makePod(0),
				options: makeOption(-1),
			},
			// want 0
			wantDeletionGracePeriodSeconds: utilpointer.Int64(0),
			wantGracePeriodSeconds:         utilpointer.Int64(1),
			wantGraceful:                   false,
			wantGracefulPending:            false,
		},
		{
			name: "when DeletionGracePeriodSeconds=0, GracePeriodSeconds=0",
			args: args{
				pod:     makePod(0),
				options: makeOption(0),
			},
			// want 0
			wantDeletionGracePeriodSeconds: utilpointer.Int64(0),
			wantGracePeriodSeconds:         utilpointer.Int64(0),
			wantGraceful:                   false,
			wantGracefulPending:            false,
		},
		{
			name: "when DeletionGracePeriodSeconds=0, GracePeriodSeconds=1",
			args: args{
				pod:     makePod(0),
				options: makeOption(1),
			},
			// want 0
			wantDeletionGracePeriodSeconds: utilpointer.Int64(0),
			wantGracePeriodSeconds:         utilpointer.Int64(1),
			wantGraceful:                   false,
			wantGracefulPending:            false,
		},
		{
			name: "when DeletionGracePeriodSeconds=0, GracePeriodSeconds=2",
			args: args{
				pod:     makePod(0),
				options: makeOption(2),
			},
			// want 0
			wantDeletionGracePeriodSeconds: utilpointer.Int64(0),
			wantGracePeriodSeconds:         utilpointer.Int64(2),
			wantGraceful:                   false,
			wantGracefulPending:            false,
		},

		{
			name: "when DeletionGracePeriodSeconds=1, GracePeriodSeconds=-1",
			args: args{
				pod:     makePod(1),
				options: makeOption(-1),
			},
			// want 1
			wantDeletionGracePeriodSeconds: utilpointer.Int64(1),
			wantGracePeriodSeconds:         utilpointer.Int64(1),
			wantGraceful:                   false,
			wantGracefulPending:            true,
		},
		{
			name: "when DeletionGracePeriodSeconds=1, GracePeriodSeconds=0",
			args: args{
				pod:     makePod(1),
				options: makeOption(0),
			},
			// want 0
			wantDeletionGracePeriodSeconds: utilpointer.Int64(0),
			wantGracePeriodSeconds:         utilpointer.Int64(0),
			wantGraceful:                   true,
			wantGracefulPending:            false,
		},
		{
			name: "when DeletionGracePeriodSeconds=1, GracePeriodSeconds=1",
			args: args{
				pod:     makePod(1),
				options: makeOption(1),
			},
			// want 1
			wantDeletionGracePeriodSeconds: utilpointer.Int64(1),
			wantGracePeriodSeconds:         utilpointer.Int64(1),
			wantGraceful:                   false,
			wantGracefulPending:            true,
		},
		{
			name: "when DeletionGracePeriodSeconds=1, GracePeriodSeconds=2",
			args: args{
				pod:     makePod(1),
				options: makeOption(2),
			},
			// want 1
			wantDeletionGracePeriodSeconds: utilpointer.Int64(1),
			wantGracePeriodSeconds:         utilpointer.Int64(2),
			wantGraceful:                   false,
			wantGracefulPending:            true,
		},

		{
			name: "when DeletionGracePeriodSeconds=2, GracePeriodSeconds=-1",
			args: args{
				pod:     makePod(2),
				options: makeOption(-1),
			},
			// want 1
			wantDeletionGracePeriodSeconds: utilpointer.Int64(1),
			wantGracePeriodSeconds:         utilpointer.Int64(1),
			wantGraceful:                   true,
			wantGracefulPending:            false,
		},
		{
			name: "when DeletionGracePeriodSeconds=2, GracePeriodSeconds=0",
			args: args{
				pod:     makePod(2),
				options: makeOption(0),
			},
			// want 0
			wantDeletionGracePeriodSeconds: utilpointer.Int64(0),
			wantGracePeriodSeconds:         utilpointer.Int64(0),
			wantGraceful:                   true,
			wantGracefulPending:            false,
		},
		{
			name: "when DeletionGracePeriodSeconds=2, GracePeriodSeconds=1",
			args: args{
				pod:     makePod(2),
				options: makeOption(1),
			},
			// want 1
			wantDeletionGracePeriodSeconds: utilpointer.Int64(1),
			wantGracePeriodSeconds:         utilpointer.Int64(1),
			wantGraceful:                   true,
			wantGracefulPending:            false,
		},
		{
			name: "when DeletionGracePeriodSeconds=2, GracePeriodSeconds=2",
			args: args{
				pod:     makePod(2),
				options: makeOption(2),
			},
			// want 2
			wantDeletionGracePeriodSeconds: utilpointer.Int64(2),
			wantGracePeriodSeconds:         utilpointer.Int64(2),
			wantGraceful:                   false,
			wantGracefulPending:            true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.args.strategy == nil {
				tt.args.strategy = mockStrategy{}
			}
			if tt.args.ctx == nil {
				tt.args.ctx = context.Background()
			}

			gotGraceful, gotGracefulPending, err := BeforeDelete(tt.args.strategy, tt.args.ctx, tt.args.pod, tt.args.options)
			if (err != nil) != tt.wantErr {
				t.Errorf("BeforeDelete() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if gotGraceful != tt.wantGraceful {
				t.Errorf("BeforeDelete() gotGraceful = %v, want %v", gotGraceful, tt.wantGraceful)
			}
			if gotGracefulPending != tt.wantGracefulPending {
				t.Errorf("BeforeDelete() gotGracefulPending = %v, want %v", gotGracefulPending, tt.wantGracefulPending)
			}
			if gotGracefulPending != tt.wantGracefulPending {
				t.Errorf("BeforeDelete() gotGracefulPending = %v, want %v", gotGracefulPending, tt.wantGracefulPending)
			}
			if !utilpointer.Int64Equal(tt.args.pod.DeletionGracePeriodSeconds, tt.wantDeletionGracePeriodSeconds) {
				t.Errorf("metadata.DeletionGracePeriodSeconds = %v, want %v", ptr.Deref(tt.args.pod.DeletionGracePeriodSeconds, 0), ptr.Deref(tt.wantDeletionGracePeriodSeconds, 0))
			}
			if !utilpointer.Int64Equal(tt.args.options.GracePeriodSeconds, tt.wantGracePeriodSeconds) {
				t.Errorf("options.GracePeriodSeconds = %v, want %v", ptr.Deref(tt.args.options.GracePeriodSeconds, 0), ptr.Deref(tt.wantGracePeriodSeconds, 0))
			}
		})
	}
}
