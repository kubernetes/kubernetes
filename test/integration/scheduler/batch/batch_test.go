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

package batch

import (
	"context"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	kubeschedulerconfigv1 "k8s.io/kube-scheduler/config/v1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	kubeschedulerscheme "k8s.io/kubernetes/pkg/scheduler/apis/config/scheme"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	testutil "k8s.io/kubernetes/test/integration/util"
)

type podDef struct {
	name                string
	nodeSelector        map[string]string
	nodeAffinity        []string
	expectedNode        string
	expectBatched       bool
	scheduler           string
	expectUnschedulable bool
	priority            int32
}

type nodeDef struct {
	name    string
	labels  map[string]string
	maxPods int
}

type scenario struct {
	name  string
	pods  []podDef
	nodes []nodeDef
}

func TestBatchScenarios(t *testing.T) {
	table := []*scenario{
		{
			name: "one pod one node",
			pods: []podDef{
				{
					name:         "1ppn-batchp1",
					expectedNode: "1ppn-batchn1",
				},
			},
			nodes: []nodeDef{
				{
					name:    "1ppn-batchn1",
					maxPods: 1,
				},
			},
		},
		{
			name: "distinct pods on distinct nodes",
			pods: []podDef{
				{
					name:         "dpdn-batchp1",
					nodeSelector: map[string]string{"forpod": "1"},
					expectedNode: "dpdn-batchn1",
				},
				{
					name:         "dpdn-batchp2",
					nodeSelector: map[string]string{"forpod": "2"},
					expectedNode: "dpdn-batchn2",
				},
				{
					name:         "dpdn-batchp3",
					nodeSelector: map[string]string{"forpod": "3"},
					expectedNode: "dpdn-batchn3",
				},
			},
			nodes: []nodeDef{
				{
					name:    "dpdn-batchn3",
					maxPods: 10,
					labels:  map[string]string{"forpod": "3"},
				},
				{
					name:    "dpdn-batchn2",
					maxPods: 10,
					labels:  map[string]string{"forpod": "2"},
				},
				{
					name:    "dpdn-batchn1",
					maxPods: 10,
					labels:  map[string]string{"forpod": "1"},
				},
			},
		},

		{
			name: "three pod batch",
			pods: []podDef{
				{
					name:         "tpb-batchp1",
					expectedNode: "tpb-batchn1",
					nodeAffinity: []string{"tpb-batchn1", "tpb-batchn2", "tpb-batchn3"},
				},
				{
					name:          "tpb-batchp2",
					expectedNode:  "tpb-batchn2",
					nodeAffinity:  []string{"tpb-batchn1", "tpb-batchn2", "tpb-batchn3"},
					expectBatched: true,
				},
				{
					name:          "tpb-batchp3",
					expectedNode:  "tpb-batchn3",
					nodeAffinity:  []string{"tpb-batchn1", "tpb-batchn2", "tpb-batchn3"},
					expectBatched: true,
				},
			},
			nodes: []nodeDef{
				{
					name:    "tpb-batchn3",
					maxPods: 1,
				},
				{
					name:    "tpb-batchn2",
					maxPods: 1,
				},
				{
					name:    "tpb-batchn1",
					maxPods: 1,
				},
			},
		},
		{
			name: "two consecutive batches",
			pods: []podDef{
				{
					name:         "tcb-batchp1",
					expectedNode: "tcb-batchn1",
					nodeAffinity: []string{"tcb-batchn1", "tcb-batchn2"},
				},
				{
					name:          "tcb-batchp2",
					expectedNode:  "tcb-batchn2",
					nodeAffinity:  []string{"tcb-batchn1", "tcb-batchn2"},
					expectBatched: true,
				},
				{
					name:         "tcb-batchp3",
					expectedNode: "tcb-batchn4",
					nodeAffinity: []string{"tcb-batchn4", "tcb-batchn3"},
				},
				{
					name:          "tcb-batchp4",
					expectedNode:  "tcb-batchn3",
					nodeAffinity:  []string{"tcb-batchn4", "tcb-batchn3"},
					expectBatched: true,
				},
			},
			nodes: []nodeDef{
				{
					name:    "tcb-batchn4",
					maxPods: 1,
				},
				{
					name:    "tcb-batchn3",
					maxPods: 1,
				},
				{
					name:    "tcb-batchn2",
					maxPods: 1,
				},
				{
					name:    "tcb-batchn1",
					maxPods: 1,
				},
			},
		},
		{
			name: "rescoring enables batching for multiple pods per node",
			pods: []podDef{
				{
					name:         "mppn-batchp1",
					expectedNode: "mppn-batchn1",
					nodeAffinity: []string{"mppn-batchn1", "mppn-batchn2"},
				},
				{
					name:          "mppn-batchp2",
					expectedNode:  "mppn-batchn1",
					nodeAffinity:  []string{"mppn-batchn1", "mppn-batchn2"},
					expectBatched: true,
				},
				{
					name:         "mppn-batchp3",
					expectedNode: "mppn-batchn4",
					nodeAffinity: []string{"mppn-batchn4", "mppn-batchn3"},
				},
				{
					name:          "mppn-batchp4",
					expectedNode:  "mppn-batchn4",
					nodeAffinity:  []string{"mppn-batchn4", "mppn-batchn3"},
					expectBatched: true,
				},
			},
			nodes: []nodeDef{
				{
					name:    "mppn-batchn4",
					maxPods: 2,
				},
				{
					name:    "mppn-batchn3",
					maxPods: 2,
				},
				{
					name:    "mppn-batchn2",
					maxPods: 2,
				},
				{
					name:    "mppn-batchn1",
					maxPods: 2,
				},
			},
		},
		{
			name: "rescoring falls back to next cached node when last chosen node is not feasible",
			pods: []podDef{
				{
					name:         "nfmb-batchp1",
					expectedNode: "nfmb-batchn1",
					nodeAffinity: []string{"nfmb-batchn1", "nfmb-batchn2"},
				},
				{
					name:          "nfmb-batchp2",
					expectedNode:  "nfmb-batchn1",
					nodeAffinity:  []string{"nfmb-batchn1", "nfmb-batchn2"},
					expectBatched: true,
				},
				{
					name:          "nfmb-batchp3",
					expectedNode:  "nfmb-batchn2",
					nodeAffinity:  []string{"nfmb-batchn1", "nfmb-batchn2"},
					expectBatched: true,
				},
			},
			nodes: []nodeDef{
				{
					name:    "nfmb-batchn1",
					maxPods: 2,
				},
				{
					name:    "nfmb-batchn2",
					maxPods: 2,
				},
			},
		},
		{
			name: "no batching between schedulers",
			pods: []podDef{
				{
					name:         "bts--batchp1",
					expectedNode: "bts--batchn1",
					nodeAffinity: []string{"bts--batchn1", "bts--batchn2", "bts--batchn3"},
				},
				{
					name:         "bts--batchp2",
					expectedNode: "bts--batchn2",
					nodeAffinity: []string{"bts--batchn1", "bts--batchn2", "bts--batchn3"},
					scheduler:    "mysched",
				},
			},
			nodes: []nodeDef{
				{
					name:    "bts--batchn3",
					maxPods: 1,
				},
				{
					name:    "bts--batchn2",
					maxPods: 1,
				},
				{
					name:    "bts--batchn1",
					maxPods: 1,
				},
			},
		},
		{
			name: "no batching missing sign",
			pods: []podDef{
				{
					name:         "nsg-batchp1",
					expectedNode: "nsg-batchn1",
					nodeAffinity: []string{"nsg-batchn1", "nsg-batchn2", "nsg-batchn3"},
					scheduler:    "nosign",
				},
				{
					name:         "nsg-batchp2",
					expectedNode: "nsg-batchn2",
					nodeAffinity: []string{"nsg-batchn1", "nsg-batchn2", "nsg-batchn3"},
					scheduler:    "nosign",
				},
			},
			nodes: []nodeDef{
				{
					name:    "nsg-batchn3",
					maxPods: 1,
				},
				{
					name:    "nsg-batchn2",
					maxPods: 1,
				},
				{
					name:    "nsg-batchn1",
					maxPods: 1,
				},
			},
		},
		{
			name: "no batching empty sign",
			pods: []podDef{
				{
					name:         "esg-batchp1",
					expectedNode: "esg-batchn1",
					nodeAffinity: []string{"esg-batchn1", "esg-batchn2", "esg-batchn3"},
					scheduler:    "emptysign",
				},
				{
					name:         "esg-batchp2",
					expectedNode: "esg-batchn2",
					nodeAffinity: []string{"esg-batchn1", "esg-batchn2", "esg-batchn3"},
					scheduler:    "emptysign",
				},
			},
			nodes: []nodeDef{
				{
					name:    "esg-batchn3",
					maxPods: 1,
				},
				{
					name:    "esg-batchn2",
					maxPods: 1,
				},
				{
					name:    "esg-batchn1",
					maxPods: 1,
				},
			},
		},
		{
			name: "unschedulable pod",
			pods: []podDef{
				{
					name:         "usp-batchp1",
					expectedNode: "usp-batchn1",
				},
				{
					name:                "usp-batchp2",
					expectBatched:       false,
					expectUnschedulable: true,
				},
			},
			nodes: []nodeDef{
				{
					name:    "usp-batchn1",
					maxPods: 1,
				},
			},
		},
	}

	for _, tt := range table {
		t.Run(tt.name, func(t *testing.T) {
			profiles, registry := initSchedulerProfiles(t)
			testCtx := testutil.InitTestSchedulerWithNS(t, "batch",
				scheduler.WithProfiles(profiles...),
				scheduler.WithFrameworkOutOfTreeRegistry(registry),
				scheduler.WithMaxBatchAge(time.Minute),
			)

			getter, ok := testCtx.Scheduler.Profiles["default-scheduler"].(interface {
				TotalBatchedPods() int64
			})
			if !ok {
				t.Fatal("Profile default-scheduler does not implement batchGetter")
			}
			cs := testCtx.ClientSet

			for _, n := range tt.nodes {
				_, err := testutil.CreateNode(cs, newNode(&n))
				if err != nil {
					t.Fatal("Failed adding node", "node", n, err)
				}
			}

			finalPods := []*v1.Pod{}
			batched := []bool{}
			for _, pd := range tt.pods {
				prevBatched := getter.TotalBatchedPods()

				p := newPod(&pd, testCtx.NS.Name)
				createdPod, err := cs.CoreV1().Pods(p.Namespace).Create(testCtx.Ctx, p, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("Failed to create pod %s/%s, error: %v",
						p.Namespace, p.Name, err)
				}

				if err := testutil.WaitForPodToScheduleWithTimeout(testCtx.Ctx, cs, createdPod, 5*time.Second); err != nil {
					if !pd.expectUnschedulable {
						t.Errorf("Failed to schedule pod %s/%s on the node, err: %v",
							p.Namespace, p.Name, err)
					} else {
						break
					}
				}

				if pd.expectUnschedulable {
					t.Fatalf("Expected pod to be unschedulable but it was scheduled")
				}

				finalPod, err := cs.CoreV1().Pods(p.Namespace).Get(testCtx.Ctx, p.Name, metav1.GetOptions{})
				if err != nil {
					t.Fatalf("Failed to get pod %v", err)
				}
				finalPods = append(finalPods, finalPod)

				currBatched := getter.TotalBatchedPods()

				batched = append(batched, currBatched > prevBatched)
			}

			for i, p := range finalPods {
				if p.Spec.NodeName != tt.pods[i].expectedNode {
					t.Fatalf("Invalid node %q for pod %q. Expected %q", p.Spec.NodeName, p.Name, tt.pods[i].expectedNode)
				}
				if batched[i] != tt.pods[i].expectBatched {
					t.Fatalf("Expected pod %q batched %t, actually %t", p.Name, tt.pods[i].expectBatched, batched[i])
				}
			}
		})
	}
}

func newPod(d *podDef, ns string) *v1.Pod {
	aff := &v1.NodeAffinity{}
	if len(d.nodeAffinity) > 0 {
		for i, node := range d.nodeAffinity {
			a := v1.PreferredSchedulingTerm{
				Weight: int32(len(d.nodeAffinity) - i),
				Preference: v1.NodeSelectorTerm{
					MatchFields: []v1.NodeSelectorRequirement{
						{
							Key:      "metadata.name",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{node},
						},
					},
				},
			}
			aff.PreferredDuringSchedulingIgnoredDuringExecution = append(aff.PreferredDuringSchedulingIgnoredDuringExecution, a)
		}
	}

	ret := testutil.InitPausePod(&testutil.PausePodConfig{
		Name:      d.name,
		Affinity:  &v1.Affinity{NodeAffinity: aff},
		Namespace: ns,
		Resources: &v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceCPU:    *(resource.NewQuantity(10, resource.DecimalSI)),
				v1.ResourceMemory: *(resource.NewQuantity(4*1024*1024, resource.DecimalSI)),
			},
		},
		NodeSelector:  d.nodeSelector,
		SchedulerName: d.scheduler,
	})

	if d.priority != 0 {
		ret.Spec.Priority = &d.priority
	}

	return ret
}

func resources(maxPods int) v1.ResourceList {
	return v1.ResourceList{
		v1.ResourceCPU:    *(resource.NewQuantity(100, resource.DecimalSI)),
		v1.ResourceMemory: *(resource.NewQuantity(4*1024*1024*1024, resource.DecimalSI)),
		v1.ResourcePods:   *resource.NewQuantity(int64(maxPods), resource.DecimalSI),
	}
}

func newNode(d *nodeDef) *v1.Node {
	n := st.MakeNode()
	n.Name(d.name)
	n.Labels = d.labels
	n.Status.Capacity = resources(d.maxPods)
	n.Status.Allocatable = resources(d.maxPods)
	return n.Obj()
}

func newDefaultComponentConfig() (*config.KubeSchedulerConfiguration, error) {
	gvk := kubeschedulerconfigv1.SchemeGroupVersion.WithKind("KubeSchedulerConfiguration")
	cfg := config.KubeSchedulerConfiguration{}
	_, _, err := kubeschedulerscheme.Codecs.UniversalDecoder().Decode(nil, &gvk, &cfg)
	if err != nil {
		return nil, err
	}

	// Clear pod topo spread defaults.
	profile := cfg.Profiles[0]
	for _, cfg := range profile.PluginConfig {
		if cfg.Name == names.PodTopologySpread {
			tps := cfg.Args.(*config.PodTopologySpreadArgs)
			tps.DefaultConstraints = []v1.TopologySpreadConstraint{}
			tps.DefaultingType = config.ListDefaulting
		}
	}

	return &cfg, nil
}

type testPluginNoSign struct{}

var _ fwk.FilterPlugin = &testPluginNoSign{}

func (pl *testPluginNoSign) Name() string {
	return "nosign"
}

func (pl *testPluginNoSign) Filter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	return nil
}

func newNoSignPlugin(_ context.Context, injArgs runtime.Object, f fwk.Handle) (fwk.Plugin, error) {
	return &testPluginNoSign{}, nil
}

type testPluginEmptySign struct{}

var _ fwk.FilterPlugin = &testPluginEmptySign{}
var _ fwk.SignPlugin = &testPluginEmptySign{}

func (pl *testPluginEmptySign) Name() string {
	return "emptysign"
}

func (pl *testPluginEmptySign) Filter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	return nil
}

func (pl *testPluginEmptySign) SignPod(ctx context.Context, pod *v1.Pod) ([]fwk.SignFragment, *fwk.Status) {
	return nil, fwk.NewStatus(fwk.Unschedulable)
}

func newEmptySignPlugin(_ context.Context, injArgs runtime.Object, f fwk.Handle) (fwk.Plugin, error) {
	return &testPluginEmptySign{}, nil
}

func initSchedulerProfiles(t *testing.T) ([]config.KubeSchedulerProfile, frameworkruntime.Registry) {
	cfg, err := newDefaultComponentConfig()
	if err != nil {
		t.Fatalf("Error creating default component config: %v", err)
	}

	newProfile := cfg.Profiles[0].DeepCopy()
	newProfile.SchedulerName = "mysched"
	cfg.Profiles = append(cfg.Profiles, *newProfile)

	newProfile = cfg.Profiles[0].DeepCopy()
	newProfile.SchedulerName = "nosign"
	newProfile.Plugins.Filter.Enabled = append(newProfile.Plugins.Filter.Enabled, config.Plugin{Name: "nosign"})
	cfg.Profiles = append(cfg.Profiles, *newProfile)

	newProfile = cfg.Profiles[0].DeepCopy()
	newProfile.SchedulerName = "emptysign"
	newProfile.Plugins.Filter.Enabled = append(newProfile.Plugins.Filter.Enabled, config.Plugin{Name: "emptysign"})
	cfg.Profiles = append(cfg.Profiles, *newProfile)

	registry := frameworkruntime.Registry{
		"nosign":    newNoSignPlugin,
		"emptysign": newEmptySignPlugin,
	}

	return cfg.Profiles, registry
}
