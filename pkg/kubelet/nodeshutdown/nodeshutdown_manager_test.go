/*
Copyright 2024 The Kubernetes Authors.

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

package nodeshutdown

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
)

func makePod(name string, priority int32, terminationGracePeriod *int64) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			UID:  types.UID(name),
		},
		Spec: v1.PodSpec{
			Priority:                      &priority,
			TerminationGracePeriodSeconds: terminationGracePeriod,
		},
	}
}

func Test_migrateConfig(t *testing.T) {
	type shutdownConfig struct {
		shutdownGracePeriodRequested    time.Duration
		shutdownGracePeriodCriticalPods time.Duration
	}
	tests := []struct {
		name string
		args shutdownConfig
		want []kubeletconfig.ShutdownGracePeriodByPodPriority
	}{
		{
			name: "both shutdownGracePeriodRequested and shutdownGracePeriodCriticalPods",
			args: shutdownConfig{
				shutdownGracePeriodRequested:    300 * time.Second,
				shutdownGracePeriodCriticalPods: 120 * time.Second,
			},
			want: []kubeletconfig.ShutdownGracePeriodByPodPriority{
				{
					Priority:                   scheduling.DefaultPriorityWhenNoDefaultClassExists,
					ShutdownGracePeriodSeconds: 180,
				},
				{
					Priority:                   scheduling.SystemCriticalPriority,
					ShutdownGracePeriodSeconds: 120,
				},
			},
		},
		{
			name: "only shutdownGracePeriodRequested",
			args: shutdownConfig{
				shutdownGracePeriodRequested:    100 * time.Second,
				shutdownGracePeriodCriticalPods: 0 * time.Second,
			},
			want: []kubeletconfig.ShutdownGracePeriodByPodPriority{
				{
					Priority:                   scheduling.DefaultPriorityWhenNoDefaultClassExists,
					ShutdownGracePeriodSeconds: 100,
				},
				{
					Priority:                   scheduling.SystemCriticalPriority,
					ShutdownGracePeriodSeconds: 0,
				},
			},
		},
		{
			name: "empty configuration",
			args: shutdownConfig{
				shutdownGracePeriodRequested:    0 * time.Second,
				shutdownGracePeriodCriticalPods: 0 * time.Second,
			},
			want: nil,
		},
		{
			name: "wrong configuration",
			args: shutdownConfig{
				shutdownGracePeriodRequested:    1 * time.Second,
				shutdownGracePeriodCriticalPods: 100 * time.Second,
			},
			want: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := migrateConfig(tt.args.shutdownGracePeriodRequested, tt.args.shutdownGracePeriodCriticalPods); !assert.Equal(t, tt.want, got) {
				t.Errorf("migrateConfig() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_groupByPriority(t *testing.T) {
	type args struct {
		shutdownGracePeriodByPodPriority []kubeletconfig.ShutdownGracePeriodByPodPriority
		pods                             []*v1.Pod
	}
	tests := []struct {
		name string
		args args
		want []podShutdownGroup
	}{
		{
			name: "migrate config",
			args: args{
				shutdownGracePeriodByPodPriority: migrateConfig(300*time.Second /* shutdownGracePeriodRequested */, 120*time.Second /* shutdownGracePeriodCriticalPods */),
				pods: []*v1.Pod{
					makePod("normal-pod", scheduling.DefaultPriorityWhenNoDefaultClassExists, nil),
					makePod("highest-user-definable-pod", scheduling.HighestUserDefinablePriority, nil),
					makePod("critical-pod", scheduling.SystemCriticalPriority, nil),
				},
			},
			want: []podShutdownGroup{
				{
					ShutdownGracePeriodByPodPriority: kubeletconfig.ShutdownGracePeriodByPodPriority{
						Priority:                   scheduling.DefaultPriorityWhenNoDefaultClassExists,
						ShutdownGracePeriodSeconds: 180,
					},
					Pods: []*v1.Pod{
						makePod("normal-pod", scheduling.DefaultPriorityWhenNoDefaultClassExists, nil),
						makePod("highest-user-definable-pod", scheduling.HighestUserDefinablePriority, nil),
					},
				},
				{
					ShutdownGracePeriodByPodPriority: kubeletconfig.ShutdownGracePeriodByPodPriority{
						Priority:                   scheduling.SystemCriticalPriority,
						ShutdownGracePeriodSeconds: 120,
					},
					Pods: []*v1.Pod{
						makePod("critical-pod", scheduling.SystemCriticalPriority, nil),
					},
				},
			},
		},
		{
			name: "pod priority",
			args: args{
				shutdownGracePeriodByPodPriority: []kubeletconfig.ShutdownGracePeriodByPodPriority{
					{
						Priority:                   1,
						ShutdownGracePeriodSeconds: 10,
					},
					{
						Priority:                   2,
						ShutdownGracePeriodSeconds: 20,
					},
					{
						Priority:                   3,
						ShutdownGracePeriodSeconds: 30,
					},
					{
						Priority:                   4,
						ShutdownGracePeriodSeconds: 40,
					},
				},
				pods: []*v1.Pod{
					makePod("pod-0", 0, nil),
					makePod("pod-1", 1, nil),
					makePod("pod-2", 2, nil),
					makePod("pod-3", 3, nil),
					makePod("pod-4", 4, nil),
					makePod("pod-5", 5, nil),
				},
			},
			want: []podShutdownGroup{
				{
					ShutdownGracePeriodByPodPriority: kubeletconfig.ShutdownGracePeriodByPodPriority{
						Priority:                   1,
						ShutdownGracePeriodSeconds: 10,
					},
					Pods: []*v1.Pod{
						makePod("pod-0", 0, nil),
						makePod("pod-1", 1, nil),
					},
				},
				{
					ShutdownGracePeriodByPodPriority: kubeletconfig.ShutdownGracePeriodByPodPriority{
						Priority:                   2,
						ShutdownGracePeriodSeconds: 20,
					},
					Pods: []*v1.Pod{
						makePod("pod-2", 2, nil),
					},
				},
				{
					ShutdownGracePeriodByPodPriority: kubeletconfig.ShutdownGracePeriodByPodPriority{
						Priority:                   3,
						ShutdownGracePeriodSeconds: 30,
					},
					Pods: []*v1.Pod{
						makePod("pod-3", 3, nil),
					},
				},
				{
					ShutdownGracePeriodByPodPriority: kubeletconfig.ShutdownGracePeriodByPodPriority{
						Priority:                   4,
						ShutdownGracePeriodSeconds: 40,
					},
					Pods: []*v1.Pod{
						makePod("pod-4", 4, nil),
						makePod("pod-5", 5, nil),
					},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := groupByPriority(tt.args.shutdownGracePeriodByPodPriority, tt.args.pods); !assert.Equal(t, tt.want, got) {
				t.Errorf("groupByPriority() = %v, want %v", got, tt.want)
			}
		})
	}
}
