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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/events"
	"k8s.io/klog/v2/ktesting"
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

type fakePlugin struct {
	name string
}

func (p *fakePlugin) Name() string {
	return p.name
}

func (p *fakePlugin) Less(*framework.QueuedPodInfo, *framework.QueuedPodInfo) bool {
	return false
}

func (p *fakePlugin) Bind(context.Context, *framework.CycleState, *v1.Pod, string) *framework.Status {
	return nil
}

func newFakePlugin(name string) func(ctx context.Context, object runtime.Object, handle framework.Handle) (framework.Plugin, error) {
	return func(_ context.Context, _ runtime.Object, _ framework.Handle) (framework.Plugin, error) {
		return &fakePlugin{name: name}, nil
	}
}

func nilRecorderFactory(_ string) events.EventRecorder {
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
