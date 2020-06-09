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
	"k8s.io/api/events/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/events"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
)

var fakeRegistry = framework.Registry{
	"QueueSort": newFakePlugin,
	"Bind1":     newFakePlugin,
	"Bind2":     newFakePlugin,
	"Another":   newFakePlugin,
}

func TestNewProfile(t *testing.T) {
	cases := []struct {
		name    string
		cfg     config.KubeSchedulerProfile
		wantErr string
	}{
		{
			name: "valid",
			cfg: config.KubeSchedulerProfile{
				SchedulerName: "valid-profile",
				Plugins: &config.Plugins{
					QueueSort: &config.PluginSet{
						Enabled: []config.Plugin{
							{Name: "QueueSort"},
						},
					},
					Bind: &config.PluginSet{
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
		},
		{
			name: "invalid framework configuration",
			cfg: config.KubeSchedulerProfile{
				SchedulerName: "invalid-profile",
				Plugins: &config.Plugins{
					QueueSort: &config.PluginSet{
						Enabled: []config.Plugin{
							{Name: "QueueSort"},
						},
					},
				},
			},
			wantErr: "at least one bind plugin is needed",
		},
		{
			name: "one queue sort plugin required for profile",
			cfg: config.KubeSchedulerProfile{
				SchedulerName: "profile-1",
				Plugins: &config.Plugins{
					Bind: &config.PluginSet{
						Enabled: []config.Plugin{
							{Name: "Bind1"},
						},
					},
				},
			},
			wantErr: "no queue sort plugin is enabled",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := fake.NewSimpleClientset()
			b := events.NewBroadcaster(&events.EventSinkImpl{Interface: c.EventsV1beta1().Events("")})
			p, err := NewProfile(tc.cfg, fakeFrameworkFactory, NewRecorderFactory(b))
			if err := checkErr(err, tc.wantErr); err != nil {
				t.Fatal(err)
			}
			if len(tc.wantErr) != 0 {
				return
			}

			called := make(chan struct{})
			var ctrl string
			stopFn := b.StartEventWatcher(func(obj runtime.Object) {
				e, _ := obj.(*v1beta1.Event)
				ctrl = e.ReportingController
				close(called)
			})
			p.Recorder.Eventf(&v1.Pod{}, nil, v1.EventTypeNormal, "", "", "")
			<-called
			stopFn()
			if ctrl != tc.cfg.SchedulerName {
				t.Errorf("got controller name %q in event, want %q", ctrl, tc.cfg.SchedulerName)
			}
		})
	}
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
						QueueSort: &config.PluginSet{
							Enabled: []config.Plugin{
								{Name: "QueueSort"},
							},
						},
						Bind: &config.PluginSet{
							Enabled: []config.Plugin{
								{Name: "Bind1"},
							},
						},
					},
				},
				{
					SchedulerName: "profile-2",
					Plugins: &config.Plugins{
						QueueSort: &config.PluginSet{
							Enabled: []config.Plugin{
								{Name: "QueueSort"},
							},
						},
						Bind: &config.PluginSet{
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
						QueueSort: &config.PluginSet{
							Enabled: []config.Plugin{
								{Name: "QueueSort"},
							},
						},
						Bind: &config.PluginSet{
							Enabled: []config.Plugin{
								{Name: "Bind1"},
							},
						},
					},
				},
				{
					SchedulerName: "profile-2",
					Plugins: &config.Plugins{
						QueueSort: &config.PluginSet{
							Enabled: []config.Plugin{
								{Name: "Another"},
							},
						},
						Bind: &config.PluginSet{
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
						QueueSort: &config.PluginSet{
							Enabled: []config.Plugin{
								{Name: "QueueSort"},
							},
						},
						Bind: &config.PluginSet{
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
						QueueSort: &config.PluginSet{
							Enabled: []config.Plugin{
								{Name: "QueueSort"},
							},
						},
						Bind: &config.PluginSet{
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
						QueueSort: &config.PluginSet{
							Enabled: []config.Plugin{
								{Name: "QueueSort"},
							},
						},
						Bind: &config.PluginSet{
							Enabled: []config.Plugin{
								{Name: "Bind1"},
							},
						},
					},
				},
				{
					SchedulerName: "profile-1",
					Plugins: &config.Plugins{
						QueueSort: &config.PluginSet{
							Enabled: []config.Plugin{
								{Name: "QueueSort"},
							},
						},
						Bind: &config.PluginSet{
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
						QueueSort: &config.PluginSet{
							Enabled: []config.Plugin{
								{Name: "QueueSort"},
							},
						},
						Bind: &config.PluginSet{
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
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m, err := NewMap(tc.cfgs, fakeFrameworkFactory, nilRecorderFactory)
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

type fakePlugin struct{}

func (p *fakePlugin) Name() string {
	return ""
}

func (p *fakePlugin) Less(*framework.QueuedPodInfo, *framework.QueuedPodInfo) bool {
	return false
}

func (p *fakePlugin) Bind(context.Context, *framework.CycleState, *v1.Pod, string) *framework.Status {
	return nil
}

func newFakePlugin(_ runtime.Object, _ framework.FrameworkHandle) (framework.Plugin, error) {
	return &fakePlugin{}, nil
}

func fakeFrameworkFactory(cfg config.KubeSchedulerProfile, opts ...framework.Option) (framework.Framework, error) {
	return framework.NewFramework(fakeRegistry, cfg.Plugins, cfg.PluginConfig, opts...)
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
