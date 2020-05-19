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

package podtopologyspread

import (
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	"k8s.io/kubernetes/pkg/scheduler/internal/cache"
)

func TestNew(t *testing.T) {
	cases := []struct {
		name    string
		args    config.PodTopologySpreadArgs
		wantErr string
	}{
		{name: "empty args"},
		{
			name: "valid constraints",
			args: config.PodTopologySpreadArgs{
				DefaultConstraints: []v1.TopologySpreadConstraint{
					{
						MaxSkew:           1,
						TopologyKey:       "node",
						WhenUnsatisfiable: v1.ScheduleAnyway,
					},
					{
						MaxSkew:           5,
						TopologyKey:       "zone",
						WhenUnsatisfiable: v1.DoNotSchedule,
					},
				},
			},
		},
		{
			name: "repeated constraints",
			args: config.PodTopologySpreadArgs{
				DefaultConstraints: []v1.TopologySpreadConstraint{
					{
						MaxSkew:           1,
						TopologyKey:       "node",
						WhenUnsatisfiable: v1.ScheduleAnyway,
					},
					{
						MaxSkew:           5,
						TopologyKey:       "node",
						WhenUnsatisfiable: v1.ScheduleAnyway,
					},
				},
			},
			wantErr: "Duplicate value",
		},
		{
			name: "unknown whenUnsatisfiable",
			args: config.PodTopologySpreadArgs{
				DefaultConstraints: []v1.TopologySpreadConstraint{
					{
						MaxSkew:           1,
						TopologyKey:       "node",
						WhenUnsatisfiable: "Unknown",
					},
				},
			},
			wantErr: "Unsupported value",
		},
		{
			name: "negative maxSkew",
			args: config.PodTopologySpreadArgs{
				DefaultConstraints: []v1.TopologySpreadConstraint{
					{
						MaxSkew:           -1,
						TopologyKey:       "node",
						WhenUnsatisfiable: v1.ScheduleAnyway,
					},
				},
			},
			wantErr: "must be greater than zero",
		},
		{
			name: "empty topologyKey",
			args: config.PodTopologySpreadArgs{
				DefaultConstraints: []v1.TopologySpreadConstraint{
					{
						MaxSkew:           1,
						WhenUnsatisfiable: v1.ScheduleAnyway,
					},
				},
			},
			wantErr: "can not be empty",
		},
		{
			name: "with label selector",
			args: config.PodTopologySpreadArgs{
				DefaultConstraints: []v1.TopologySpreadConstraint{
					{
						MaxSkew:           1,
						TopologyKey:       "rack",
						WhenUnsatisfiable: v1.ScheduleAnyway,
						LabelSelector: &metav1.LabelSelector{
							MatchLabels: map[string]string{
								"foo": "bar",
							},
						},
					},
				},
			},
			wantErr: "constraint must not define a selector",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			informerFactory := informers.NewSharedInformerFactory(fake.NewSimpleClientset(), 0)
			f, err := framework.NewFramework(nil, nil, nil,
				framework.WithSnapshotSharedLister(cache.NewSnapshot(nil, nil)),
				framework.WithInformerFactory(informerFactory),
			)
			if err != nil {
				t.Fatal(err)
			}
			_, err = New(&tc.args, f)
			if len(tc.wantErr) != 0 {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Errorf("must fail, got error %q, want %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
		})
	}
}
