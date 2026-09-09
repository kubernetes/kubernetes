/*
Copyright 2023 The Kubernetes Authors.

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

package conditions

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/watch"
)

func TestPodRunning(t *testing.T) {
	tests := []struct {
		name    string
		event   watch.Event
		want    bool
		wantErr bool
	}{
		{
			name: "Watch type is deleted",
			event: watch.Event{
				Type: watch.Deleted,
			},
			want:    false,
			wantErr: true,
		},
		{
			name: "Pod Status type is PodRunning",
			event: watch.Event{
				Type: watch.Added,
				Object: &corev1.Pod{
					Status: corev1.PodStatus{
						Phase: corev1.PodRunning,
					},
				},
			},
			want:    true,
			wantErr: false,
		},
		{
			name: "Pod Status is PodFailed",
			event: watch.Event{
				Type: watch.Added,
				Object: &corev1.Pod{
					Status: corev1.PodStatus{
						Phase: corev1.PodFailed,
					},
				},
			},
			want:    false,
			wantErr: true,
		},
		{
			name: "Object type is not pod",
			event: watch.Event{
				Type:   watch.Added,
				Object: &corev1.Node{},
			},
			want:    false,
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := PodRunning(tt.event)
			if (err != nil) != tt.wantErr {
				t.Errorf("PodRunning() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("PodRunning() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestPodCompleted(t *testing.T) {
	tests := []struct {
		name    string
		event   watch.Event
		want    bool
		wantErr bool
	}{
		{
			name: "Watch type is deleted",
			event: watch.Event{
				Type: watch.Deleted,
			},
			want:    false,
			wantErr: true,
		},
		{
			name: "Pod Status is PodSucceeded",
			event: watch.Event{
				Type: watch.Added,
				Object: &corev1.Pod{
					Status: corev1.PodStatus{
						Phase: corev1.PodSucceeded,
					},
				},
			},
			want:    true,
			wantErr: false,
		},
		{
			name: "Pod Status is PodRunning",
			event: watch.Event{
				Type: watch.Added,
				Object: &corev1.Pod{
					Status: corev1.PodStatus{
						Phase: corev1.PodRunning,
					},
				},
			},
			want:    false,
			wantErr: false,
		},
		{
			name: "Object type is not pod",
			event: watch.Event{
				Type:   watch.Added,
				Object: &corev1.Node{},
			},
			want:    false,
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := PodCompleted(tt.event)
			if (err != nil) != tt.wantErr {
				t.Errorf("PodCompleted() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("PodCompleted() = %v, want %v", got, tt.want)
			}
		})
	}
}
