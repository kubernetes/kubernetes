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
	"strings"
	"testing"
	"time"

	"go.etcd.io/etcd/pkg/v3/featuregate"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/informers"
	restclient "k8s.io/client-go/rest"
	kubeschedulerconfigv1 "k8s.io/kube-scheduler/config/v1"
	fwk "k8s.io/kube-scheduler/framework"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	kubeschedulerscheme "k8s.io/kubernetes/pkg/scheduler/apis/config/scheme"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/integration/util"

	"k8s.io/kubernetes/pkg/scheduler"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	testutil "k8s.io/kubernetes/test/integration/util"
	"k8s.io/kubernetes/test/utils/ktesting"
)

type podDef struct {
	name          string
	nodeSelector  map[string]string
	nodeAffinity  []string
	expectedNode  string
	expectBatched bool
	nnn           string
	scheduler     string
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
			name: "multiple pods per node means no batching",
			pods: []podDef{
				{
					name:         "mppn-batchp1",
					expectedNode: "mppn-batchn1",
					nodeAffinity: []string{"mppn-batchn1", "mppn-batchn2"},
				},
				{
					name:         "mppn-batchp2",
					expectedNode: "mppn-batchn1",
					nodeAffinity: []string{"mppn-batchn1", "mppn-batchn2"},
				},
				{
					name:         "mppn-batchp3",
					expectedNode: "mppn-batchn4",
					nodeAffinity: []string{"mppn-batchn4", "mppn-batchn3"},
				},
				{
					name:         "mppn-batchp4",
					expectedNode: "mppn-batchn4",
					nodeAffinity: []string{"mppn-batchn4", "mppn-batchn3"},
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
		/*
			{
				name: "nnn trumps hint",
				pods: []podDef{
					{
						name:         "nnn-batchp1",
						expectedNode: "nnn-batchn1",
						nodeAffinity: []string{"nnn-batchn1", "nnn-batchn2", "nnn-batchn3"},
					},
					{
						name:          "nnn-batchp2",
						expectedNode:  "nnn-batchn3",
						nodeAffinity:  []string{"nnn-batchn1", "nnn-batchn2", "nnn-batchn3"},
						nnn:           "nnn-batchn3",
						expectBatched: false,
					},
				},
				nodes: []nodeDef{
					{
						name:    "nnn-batchn3",
						maxPods: 1,
					},
					{
						name:    "nnn-batchn2",
						maxPods: 1,
					},
					{
						name:    "nnn-batchn1",
						maxPods: 1,
					},
				},
			},
		*/
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
	}

	for _, tt := range table {
		t.Run(tt.name, func(t *testing.T) {
			finalPods, batched := runScenario(t, tt, true)
			for i, p := range finalPods {
				if p.Spec.NodeName != tt.pods[i].expectedNode {
					t.Fatalf("Invalid node '%s' for pod '%s'. Expected '%s'", p.Spec.NodeName, p.Name, tt.pods[i].expectedNode)
				}
				if batched[i] != tt.pods[i].expectBatched {
					t.Fatalf("Expected pod %s batched %t, actually %t", p.Name, tt.pods[i].expectBatched, batched[i])
				}
			}
		})
	}
}

func newPod(d *podDef) *v1.Pod {
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

	return testutil.InitPausePod(&testutil.PausePodConfig{
		Name:      d.name,
		Affinity:  &v1.Affinity{NodeAffinity: aff},
		Namespace: "default",
		Resources: &v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceCPU:    *(resource.NewQuantity(10, resource.DecimalSI)),
				v1.ResourceMemory: *(resource.NewQuantity(4*1024*1024, resource.DecimalSI)),
			},
		},
		NodeSelector:  d.nodeSelector,
		SchedulerName: d.scheduler,
	})
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
	return "nosign"
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

// To access the test-only field in the framework
type batchGetter interface {
	TotalBatchedPods() int64
}

func runScenario(t *testing.T, tt *scenario, batch bool) ([]*v1.Pod, []bool) {
	_, tCtx := ktesting.NewTestContext(t)

	cfg, err := newDefaultComponentConfig()
	if err != nil {
		tCtx.Fatalf("Error creating default component config: %v", err)
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

	enabledFeatures := map[featuregate.Feature]bool{}
	if batch {
		enabledFeatures[featuregate.Feature(features.OpportunisticBatching)] = true
	}

	scheduler, _, testCtx := mustSetupCluster(tCtx, cfg, nil, frameworkruntime.Registry{
		"nosign":    newNoSignPlugin,
		"emptysign": newEmptySignPlugin,
	})

	getter := scheduler.Profiles["default-scheduler"].(batchGetter)

	cs := testCtx.Client()

	// Add nodes.
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

		p := newPod(&pd)
		createdPod, err := cs.CoreV1().Pods(p.Namespace).Create(testCtx, p, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create pod %s/%s, error: %v",
				p.Namespace, p.Name, err)
		}

		if err := testutil.WaitForPodToSchedule(testCtx, cs, createdPod); err != nil {
			t.Errorf("Failed to schedule pod %s/%s on the node, err: %v",
				p.Namespace, p.Name, err)
		}

		finalPod, err := cs.CoreV1().Pods(p.Namespace).Get(testCtx, p.Name, metav1.GetOptions{})
		finalPods = append(finalPods, finalPod)

		currBatched := getter.TotalBatchedPods()

		batched = append(batched, currBatched > prevBatched)
	}

	return finalPods, batched
}

// mustSetupCluster starts the following components:
// - k8s api server
// - scheduler
// - some of the kube-controller-manager controllers
//
// It returns regular and dynamic clients, and destroyFunc which should be used to
// remove resources after finished.
// Notes on rate limiter:
//   - client rate limit is set to 5000.
func mustSetupCluster(tCtx ktesting.TContext, config *config.KubeSchedulerConfiguration, enabledFeatures map[featuregate.Feature]bool, outOfTreePluginRegistry frameworkruntime.Registry) (*scheduler.Scheduler, informers.SharedInformerFactory, ktesting.TContext) {
	var runtimeConfig []string
	customFlags := []string{
		// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
		"--disable-admission-plugins=ServiceAccount,TaintNodesByCondition,Priority",
		"--runtime-config=" + strings.Join(runtimeConfig, ","),
	}
	serverOpts := apiservertesting.NewDefaultTestServerOptions()
	// Timeout sufficiently long to handle deleting pods of the largest test cases.
	serverOpts.RequestTimeout = 10 * time.Minute
	server, err := apiservertesting.StartTestServer(tCtx, serverOpts, customFlags, framework.SharedEtcd())
	if err != nil {
		tCtx.Fatalf("start apiserver: %v", err)
	}
	// Cleanup will be in reverse order: first the clients by canceling the
	// child context (happens automatically), then the server.
	tCtx.Cleanup(server.TearDownFn)
	tCtx = ktesting.WithCancel(tCtx)

	// TODO: client connection configuration, such as QPS or Burst is configurable in theory, this could be derived from the `config`, need to
	// support this when there is any testcase that depends on such configuration.
	cfg := restclient.CopyConfig(server.ClientConfig)
	cfg.QPS = 5000.0
	cfg.Burst = 5000

	// use default component config if config here is nil
	if config == nil {
		var err error
		config, err = newDefaultComponentConfig()
		if err != nil {
			tCtx.Fatalf("Error creating default component config: %v", err)
		}
	}

	tCtx = ktesting.WithRESTConfig(tCtx, cfg)

	// Not all config options will be effective but only those mostly related with scheduler performance will
	// be applied to start a scheduler, most of them are defined in `scheduler.schedulerOptions`.
	scheduler, informerFactory := util.StartScheduler(tCtx, config, outOfTreePluginRegistry)
	util.StartFakePVController(tCtx, tCtx.Client(), informerFactory)
	runGC := util.CreateGCController(tCtx, tCtx, *cfg, informerFactory)
	runNS := util.CreateNamespaceController(tCtx, tCtx, *cfg, informerFactory)

	runResourceClaimController := func() {}

	informerFactory.Start(tCtx.Done())
	informerFactory.WaitForCacheSync(tCtx.Done())
	go runGC()
	go runNS()
	go runResourceClaimController()

	return scheduler, informerFactory, tCtx

}
