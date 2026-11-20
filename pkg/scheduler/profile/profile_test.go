/*
Copyright 2020 The Kubernetes Authors.

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

package profile

import (
	"context"
	"fmt"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/events"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
)

var fakeRegistry = frameworkruntime.Registry{
	"QueueSort": newFakePlugin("QueueSort"),
	"Bind1":     newFakePlugin("Bind1"),
	"Bind2":     newFakePlugin("Bind2"),
	"Another":   newFakePlugin("Another"),
}

func TestNewMap(t *testing.T) {
	cases := []struct {
		name    string
		cfgs    []config.KubeSchedulerProfile
		wantErr string
	}{
		{
			name: "valid",
			cfgs: []config.KubeSchedulerProfile{
				{
					SchedulerName: "profile-1",
					Plugins: &config.Plugins{
						QueueSort: config.PluginSet{
							Enabled: []config.Plugin{
								{Name: "QueueSort"},
							},
						},
						Bind: config.PluginSet{
							Enabled: []config.Plugin{
								{Name: "Bind1"},
							},
						},
					},
				},
				{
					SchedulerName: "profile-2",
					Plugins: &config.Plugins{
						QueueSort: config.PluginSet{
							Enabled: []config.Plugin{
								{Name: "QueueSort"},
							},
						},
						Bind: config.PluginSet{
							Enabled: []config.Plugin{
								{Name: "Bind2"},
							},
						},
					},
					PluginConfig: []config.PluginConfig{
						{
							Name: "Bind2",
							Args: &runtime.Unknown{Raw: []byte("{}")},
						},
					},
				},
			},
		},
		{
			name: "different queue sort",
			cfgs: []config.KubeSchedulerProfile{
				{
					SchedulerName: "profile-1",
					Plugins: &config.Plugins{
						QueueSort: config.PluginSet{
							Enabled: []config.Plugin{
								{Name: "QueueSort"},
							},
						},
						Bind: config.PluginSet{
							Enabled: []config.Plugin{
								{Name: "Bind1"},
							},
						},
					},
				},
				{
					SchedulerName: "profile-2",
					Plugins: &config.Plugins{
						QueueSort: config.PluginSet{
							Enabled: []config.Plugin{
								{Name: "Another"},
							},
						},
						Bind: config.PluginSet{
							Enabled: []config.Plugin{
								{Name: "Bind2"},
							},
						},
					},
				},
			},
			wantErr: "different queue sort plugins",
		},
		{
			name: "different queue sort args",
			cfgs: []config.KubeSchedulerProfile{
				{
					SchedulerName: "profile-1",
					Plugins: &config.Plugins{
						QueueSort: config.PluginSet{
							Enabled: []config.Plugin{
								{Name: "QueueSort"},
							},
						},
						Bind: config.PluginSet{
							Enabled: []config.Plugin{
								{Name: "Bind1"},
							},
						},
					},
					PluginConfig: []config.PluginConfig{
						{
							Name: "QueueSort",
							Args: &runtime.Unknown{Raw: []byte("{}")},
						},
					},
				},
				{
					SchedulerName: "profile-2",
					Plugins: &config.Plugins{
						QueueSort: config.PluginSet{
							Enabled: []config.Plugin{
								{Name: "QueueSort"},
							},
						},
						Bind: config.PluginSet{
							Enabled: []config.Plugin{
								{Name: "Bind2"},
							},
						},
					},
				},
			},
			wantErr: "different queue sort plugin args",
		},
		{
			name: "duplicate scheduler name",
			cfgs: []config.KubeSchedulerProfile{
				{
					SchedulerName: "profile-1",
					Plugins: &config.Plugins{
						QueueSort: config.PluginSet{
							Enabled: []config.Plugin{
								{Name: "QueueSort"},
							},
						},
						Bind: config.PluginSet{
							Enabled: []config.Plugin{
								{Name: "Bind1"},
							},
						},
					},
				},
				{
					SchedulerName: "profile-1",
					Plugins: &config.Plugins{
						QueueSort: config.PluginSet{
							Enabled: []config.Plugin{
								{Name: "QueueSort"},
							},
						},
						Bind: config.PluginSet{
							Enabled: []config.Plugin{
								{Name: "Bind2"},
							},
						},
					},
				},
			},
			wantErr: "duplicate profile",
		},
		{
			name: "scheduler name is needed",
			cfgs: []config.KubeSchedulerProfile{
				{
					Plugins: &config.Plugins{
						QueueSort: config.PluginSet{
							Enabled: []config.Plugin{
								{Name: "QueueSort"},
							},
						},
						Bind: config.PluginSet{
							Enabled: []config.Plugin{
								{Name: "Bind1"},
							},
						},
					},
				},
			},
			wantErr: "scheduler name is needed",
		},
		{
			name: "plugins required for profile",
			cfgs: []config.KubeSchedulerProfile{
				{
					SchedulerName: "profile-1",
				},
			},
			wantErr: "plugins required for profile",
		},
		{
			name: "invalid framework configuration",
			cfgs: []config.KubeSchedulerProfile{
				{
					SchedulerName: "invalid-profile",
					Plugins: &config.Plugins{
						QueueSort: config.PluginSet{
							Enabled: []config.Plugin{
								{Name: "QueueSort"},
							},
						},
					},
				},
			},
			wantErr: "at least one bind plugin is needed",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			m, err := NewMap(ctx, tc.cfgs, fakeRegistry, nilRecorderFactory)
			defer func() {
				if m != nil {
					// to close all frameworks registered in this map.
					err := m.Close()
					if err != nil {
						t.Errorf("error closing map: %v", err)
					}
				}
			}()

			if err := checkErr(err, tc.wantErr); err != nil {
				t.Fatal(err)
			}
			if len(tc.wantErr) != 0 {
				return
			}
			if len(m) != len(tc.cfgs) {
				t.Errorf("got %d profiles, want %d", len(m), len(tc.cfgs))
			}
		})
	}
}

func TestFrameworkForPod(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	cfgs := []config.KubeSchedulerProfile{
		{
			SchedulerName: v1.DefaultSchedulerName,
			Plugins: &config.Plugins{
				QueueSort: config.PluginSet{Enabled: []config.Plugin{{Name: "QueueSort"}}},
				Bind:      config.PluginSet{Enabled: []config.Plugin{{Name: "Bind1"}}},
			},
		},
		{
			SchedulerName: "custom-scheduler",
			Plugins: &config.Plugins{
				QueueSort: config.PluginSet{Enabled: []config.Plugin{{Name: "QueueSort"}}},
				Bind:      config.PluginSet{Enabled: []config.Plugin{{Name: "Bind2"}}},
			},
		},
	}

	m, err := NewMap(ctx, cfgs, fakeRegistry, nilRecorderFactory)
	if err != nil {
		t.Fatalf("failed to create profile map: %v", err)
	}
	defer func() {
		err := m.Close()
		if err != nil {
			t.Errorf("error closing map: %v", err)
		}
	}()

	tests := []struct {
		name          string
		schedulerName string
		wantErr       string
	}{
		{
			name:          "explicit scheduler name matching profile",
			schedulerName: "custom-scheduler",
		},
		{
			name:          "empty scheduler name defaults to default-scheduler",
			schedulerName: "",
		},
		{
			name:          "unknown scheduler name returns error",
			schedulerName: "non-existent",
			wantErr:       `profile not found for scheduler name "non-existent"`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pod := &v1.Pod{
				Spec: v1.PodSpec{
					SchedulerName: tt.schedulerName,
				},
			}
			fwk, err := m.FrameworkForPod(pod)
			if tt.wantErr != "" {
				if err.Error() != tt.wantErr {
					t.Errorf("FrameworkForPod() error = %v, wantErr %v", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
			if fwk == nil {
				t.Errorf("expected non-nil framework for scheduler name %q", tt.schedulerName)
			}
		})
	}
}

func TestFrameworkForPodGroup(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	cfgs := []config.KubeSchedulerProfile{
		{
			SchedulerName: v1.DefaultSchedulerName,
			Plugins: &config.Plugins{
				QueueSort: config.PluginSet{
					Enabled: []config.Plugin{{Name: "QueueSort"}},
				},
				Bind: config.PluginSet{
					Enabled: []config.Plugin{{Name: "Bind1"}},
				},
			},
		},
		{
			SchedulerName: "custom-scheduler",
			Plugins: &config.Plugins{
				QueueSort: config.PluginSet{
					Enabled: []config.Plugin{{Name: "QueueSort"}},
				},
				Bind: config.PluginSet{
					Enabled: []config.Plugin{{Name: "Bind2"}},
				},
			},
		},
	}

	m, err := NewMap(ctx, cfgs, fakeRegistry, nilRecorderFactory)
	if err != nil {
		t.Fatalf("NewMap() error = %v", err)
	}

	successTests := []struct {
		name            string
		podGroupInfo    fwk.PodGroupInfo
		wantProfileName string
	}{
		{
			name: "explicit scheduler name matching profile",
			podGroupInfo: &framework.PodGroupInfo{
				GenericPodGroup: fwk.NewGenericPodGroup(&schedulingv1beta1.PodGroup{}),
				UnscheduledPods: []*v1.Pod{
					{Spec: v1.PodSpec{SchedulerName: "custom-scheduler"}},
				},
			},
			wantProfileName: "custom-scheduler",
		},
		{
			name: "empty scheduler name defaults to default-scheduler",
			podGroupInfo: &framework.PodGroupInfo{
				GenericPodGroup: fwk.NewGenericPodGroup(&schedulingv1beta1.PodGroup{}),
				UnscheduledPods: []*v1.Pod{
					{Spec: v1.PodSpec{SchedulerName: ""}},
				},
			},
			wantProfileName: v1.DefaultSchedulerName,
		},
		{
			name: "composite pod group resolves profile from child leaf pod group",
			podGroupInfo: framework.NewPodGroupInfo(
				fwk.NewGenericCompositePodGroup(&schedulingv1beta1.CompositePodGroup{}),
				&framework.PodGroupInfo{
					GenericPodGroup: fwk.NewGenericPodGroup(&schedulingv1beta1.PodGroup{}),
					UnscheduledPods: []*v1.Pod{
						{Spec: v1.PodSpec{SchedulerName: "custom-scheduler"}},
					},
				},
			),
			wantProfileName: "custom-scheduler",
		},
		{
			name: "composite pod group skips empty first leaf child and resolves from sibling",
			podGroupInfo: framework.NewPodGroupInfo(
				fwk.NewGenericCompositePodGroup(&schedulingv1beta1.CompositePodGroup{}),
				&framework.PodGroupInfo{
					GenericPodGroup: fwk.NewGenericPodGroup(&schedulingv1beta1.PodGroup{}),
				},
				&framework.PodGroupInfo{
					GenericPodGroup: fwk.NewGenericPodGroup(&schedulingv1beta1.PodGroup{}),
					UnscheduledPods: []*v1.Pod{
						{Spec: v1.PodSpec{SchedulerName: "custom-scheduler"}},
					},
				},
			),
			wantProfileName: "custom-scheduler",
		},
	}

	for _, tt := range successTests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := m.FrameworkForPodGroup(tt.podGroupInfo)
			if err != nil {
				t.Fatalf("FrameworkForPodGroup() unexpected error = %v", err)
			}
			if got == nil {
				t.Fatal("FrameworkForPodGroup() returned nil framework")
			}
			if got.ProfileName() != tt.wantProfileName {
				t.Errorf("FrameworkForPodGroup().ProfileName() = %q, want %q", got.ProfileName(), tt.wantProfileName)
			}
		})
	}

	errorTests := []struct {
		name         string
		podGroupInfo fwk.PodGroupInfo
	}{
		{
			name:         "nil pod group returns error",
			podGroupInfo: nil,
		},
		{
			name: "empty pod group returns error",
			podGroupInfo: &framework.PodGroupInfo{
				GenericPodGroup: fwk.NewGenericPodGroup(&schedulingv1beta1.PodGroup{}),
			},
		},
		{
			name: "unknown scheduler name returns error",
			podGroupInfo: &framework.PodGroupInfo{
				GenericPodGroup: fwk.NewGenericPodGroup(&schedulingv1beta1.PodGroup{}),
				UnscheduledPods: []*v1.Pod{
					{Spec: v1.PodSpec{SchedulerName: "non-existent"}},
				},
			},
		},
	}

	for _, tt := range errorTests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := m.FrameworkForPodGroup(tt.podGroupInfo)
			if err == nil {
				t.Fatalf("FrameworkForPodGroup() = %v, want error", got)
			}
			if got != nil {
				t.Errorf("FrameworkForPodGroup() returned non-nil framework %v on error", got)
			}
		})
	}
}

type fakePlugin struct {
	name string
}

func (p *fakePlugin) Name() string {
	return p.name
}

func (p *fakePlugin) Less(fwk.QueuedEntityInfo, fwk.QueuedEntityInfo) bool {
	return false
}

func (p *fakePlugin) Bind(context.Context, fwk.CycleState, *v1.Pod, string) *fwk.Status {
	return nil
}

func newFakePlugin(name string) func(ctx context.Context, object runtime.Object, handle fwk.Handle) (fwk.Plugin, error) {
	return func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
		return &fakePlugin{name: name}, nil
	}
}

func nilRecorderFactory(_ string) events.EventRecorderLogger {
	return nil
}

func checkErr(err error, wantErr string) error {
	if len(wantErr) == 0 {
		return err
	}
	if err == nil || !strings.Contains(err.Error(), wantErr) {
		return fmt.Errorf("got error %q, want %q", err, wantErr)
	}
	return nil
}
