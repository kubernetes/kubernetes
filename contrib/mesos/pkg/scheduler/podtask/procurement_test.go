/*
Copyright 2015 The Kubernetes Authors.

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

package podtask

import (
	"testing"

	"github.com/mesos/mesos-go/mesosproto"
	"github.com/mesos/mesos-go/mesosutil"

	mesos "github.com/mesos/mesos-go/mesosproto"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podtask/hostport"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/resources"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"reflect"
)

func TestNewPodResourcesProcurement(t *testing.T) {
	executor := mesosutil.NewExecutorInfo(
		mesosutil.NewExecutorID("executor-id"),
		mesosutil.NewCommandInfo("executor-cmd"),
	)
	executor.Data = []byte{0, 1, 2}
	executor.Resources = []*mesosproto.Resource{
		scalar("cpus", 0.1, "*"),
		scalar("mem", 64.0, "*"),
	}
	executor.Command = &mesosproto.CommandInfo{
		Arguments: []string{},
	}

	offer := &mesosproto.Offer{
		Resources: []*mesosproto.Resource{
			scalar("cpus", 4.0, "*"),
			scalar("mem", 512.0, "*"),
		},
	}

	task, _ := New(
		api.NewDefaultContext(),
		Config{
			Prototype:        executor,
			FrameworkRoles:   []string{"*"},
			DefaultPodRoles:  []string{"*"},
			HostPortStrategy: hostport.StrategyWildcard,
		},
		&api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:      "test",
				Namespace: api.NamespaceDefault,
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Resources: api.ResourceRequirements{
							Limits: api.ResourceList{
								api.ResourceCPU: *resource.NewQuantity(
									3,
									resource.DecimalSI,
								),
								api.ResourceMemory: *resource.NewQuantity(
									128*1024*1024,
									resource.BinarySI,
								),
							},
						},
					},
				},
			},
		},
	)

	procurement := NewPodResourcesProcurement()

	ps := NewProcureState(offer)
	if err := procurement.Procure(task, &api.Node{}, ps); err != nil {
		t.Error(err)
	}

	if len(ps.spec.Resources) == 0 {
		t.Errorf("expected procured resources but got none")
	}
}

func TestProcureRoleResources(t *testing.T) {
	for i, tt := range []struct {
		offered []*mesos.Resource

		name  string // cpu or mem
		want  float64
		roles []string

		consumed []*mesos.Resource
		left     []*mesos.Resource
	}{
		{
			offered: []*mesos.Resource{
				scalar("mem", 128.0, "*"),
				scalar("mem", 32.0, "slave_public"),
			},

			name:  "mem",
			want:  128.0,
			roles: []string{"slave_public", "*"},

			consumed: []*mesos.Resource{
				scalar("mem", 32.0, "slave_public"),
				scalar("mem", 96.0, "*"),
			},
			left: []*mesos.Resource{
				scalar("mem", 32.0, "*"),
			},
		},
		{
			offered: []*mesos.Resource{
				scalar("mem", 128.0, "*"),
				scalar("mem", 32.0, "slave_public"),
			},

			name:  "mem",
			want:  128.0,
			roles: []string{"slave_public"},

			consumed: nil,
			left: []*mesos.Resource{
				scalar("mem", 128.0, "*"),
				scalar("mem", 32.0, "slave_public"),
			},
		},
		{
			offered: []*mesos.Resource{
				scalar("cpus", 1.5, "slave_public"),
				scalar("cpus", 1, "slave_public"),
				scalar("mem", 128.0, "slave_public"),
				scalar("mem", 64.0, "slave_public"),
				scalar("mem", 128.0, "*"),
			},

			name:  "mem",
			want:  200.0,
			roles: []string{"slave_public", "*"},

			consumed: []*mesos.Resource{
				scalar("mem", 128.0, "slave_public"),
				scalar("mem", 64.0, "slave_public"),
				scalar("mem", 8.0, "*"),
			},
			left: []*mesos.Resource{
				scalar("cpus", 1.5, "slave_public"),
				scalar("cpus", 1, "slave_public"),
				scalar("mem", 120, "*"),
			},
		},
		{
			offered: []*mesos.Resource{
				scalar("mem", 128.0, "*"),
			},

			name:  "mem",
			want:  128.0,
			roles: []string{"slave_public", "*"},

			consumed: []*mesos.Resource{
				scalar("mem", 128, "*"),
			},
			left: []*mesos.Resource{},
		},
		{
			offered: []*mesos.Resource{
				scalar("cpu", 32.0, "slave_public"),
			},

			name:  "mem",
			want:  128.0,
			roles: []string{"slave_public", "*"},

			consumed: nil,
			left: []*mesos.Resource{
				scalar("cpu", 32.0, "slave_public"),
			},
		},
		{
			offered: nil,

			name:  "mem",
			want:  160.0,
			roles: []string{"slave_public", "*"},

			consumed: nil, left: nil,
		},
	} {
		consumed, remaining := procureScalarResources(tt.name, tt.want, tt.roles, tt.offered)

		if !reflect.DeepEqual(consumed, tt.consumed) {
			t.Errorf("test #%d (consumed):\ngot  %v\nwant %v", i, consumed, tt.consumed)
		}

		if !reflect.DeepEqual(remaining, tt.left) {
			t.Errorf("test #%d (remaining):\ngot  %v\nwant %v", i, remaining, tt.left)
		}
	}
}

func scalar(name string, value float64, role string) *mesos.Resource {
	res := mesosutil.NewScalarResource(name, value)
	res.Role = resources.StringPtrTo(role)
	return res
}
