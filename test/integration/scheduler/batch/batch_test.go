package batch

import (
	"strings"
	"testing"
	"time"

	"go.etcd.io/etcd/pkg/v3/featuregate"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	restclient "k8s.io/client-go/rest"
	kubeschedulerconfigv1 "k8s.io/kube-scheduler/config/v1"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	kubeschedulerscheme "k8s.io/kubernetes/pkg/scheduler/apis/config/scheme"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/integration/util"

	"k8s.io/kubernetes/pkg/scheduler"
	testutil "k8s.io/kubernetes/test/integration/util"
	imageutils "k8s.io/kubernetes/test/utils/image"
	"k8s.io/kubernetes/test/utils/ktesting"
)

type Scenario struct {
	Name  string
	Pods  []PodDef
	Nodes []NodeDef
}

type PodDef struct {
	Name          string
	NodeSelector  map[string]string
	Affinity      []string
	ExpectedNode  string
	ExpectBatched bool
}

type NodeDef struct {
	Name    string
	Labels  map[string]string
	MaxPods int
}

var defaultResources = v1.ResourceList{}

func TestBatchScenarios(t *testing.T) {
	table := []*Scenario{
		{
			Name: "one pod one node",
			Pods: []PodDef{
				{
					Name:         "1ppn-batchp1",
					ExpectedNode: "1ppn-batchn1",
				},
			},
			Nodes: []NodeDef{
				{
					Name:    "1ppn-batchn1",
					MaxPods: 1,
				},
			},
		},
		{
			Name: "distinct pods on distinct nodes",
			Pods: []PodDef{
				{
					Name:         "dpdn-batchp1",
					NodeSelector: map[string]string{"forpod": "1"},
					ExpectedNode: "dpdn-batchn1",
				},
				{
					Name:         "dpdn-batchp2",
					NodeSelector: map[string]string{"forpod": "2"},
					ExpectedNode: "dpdn-batchn2",
				},
				{
					Name:         "dpdn-batchp3",
					NodeSelector: map[string]string{"forpod": "3"},
					ExpectedNode: "dpdn-batchn3",
				},
			},
			Nodes: []NodeDef{
				{
					Name:    "dpdn-batchn3",
					MaxPods: 10,
					Labels:  map[string]string{"forpod": "3"},
				},
				{
					Name:    "dpdn-batchn2",
					MaxPods: 10,
					Labels:  map[string]string{"forpod": "2"},
				},
				{
					Name:    "dpdn-batchn1",
					MaxPods: 10,
					Labels:  map[string]string{"forpod": "1"},
				},
			},
		},

		{
			Name: "three pod batch",
			Pods: []PodDef{
				{
					Name:         "tpb-batchp1",
					ExpectedNode: "tpb-batchn1",
					Affinity:     []string{"tpb-batchn1", "tpb-batchn2", "tpb-batchn3"},
				},
				{
					Name:          "tpb-batchp2",
					ExpectedNode:  "tpb-batchn2",
					Affinity:      []string{"tpb-batchn1", "tpb-batchn2", "tpb-batchn3"},
					ExpectBatched: true,
				},
				{
					Name:          "tpb-batchp3",
					ExpectedNode:  "tpb-batchn3",
					Affinity:      []string{"tpb-batchn1", "tpb-batchn2", "tpb-batchn3"},
					ExpectBatched: true,
				},
			},
			Nodes: []NodeDef{
				{
					Name:    "tpb-batchn3",
					MaxPods: 1,
				},
				{
					Name:    "tpb-batchn2",
					MaxPods: 1,
				},
				{
					Name:    "tpb-batchn1",
					MaxPods: 1,
				},
			},
		},
		{
			Name: "two consecutive batches",
			Pods: []PodDef{
				{
					Name:         "tcb-batchp1",
					ExpectedNode: "tcb-batchn1",
					Affinity:     []string{"tcb-batchn1", "tcb-batchn2"},
				},
				{
					Name:          "tcb-batchp2",
					ExpectedNode:  "tcb-batchn2",
					Affinity:      []string{"tcb-batchn1", "tcb-batchn2"},
					ExpectBatched: true,
				},
				{
					Name:         "tcb-batchp3",
					ExpectedNode: "tcb-batchn4",
					Affinity:     []string{"tcb-batchn4", "tcb-batchn3"},
				},
				{
					Name:          "tcb-batchp4",
					ExpectedNode:  "tcb-batchn3",
					Affinity:      []string{"tcb-batchn4", "tcb-batchn3"},
					ExpectBatched: true,
				},
			},
			Nodes: []NodeDef{
				{
					Name:    "tcb-batchn4",
					MaxPods: 1,
				},
				{
					Name:    "tcb-batchn3",
					MaxPods: 1,
				},
				{
					Name:    "tcb-batchn2",
					MaxPods: 1,
				},
				{
					Name:    "tcb-batchn1",
					MaxPods: 1,
				},
			},
		},
		{
			Name: "multiple pods per node means no batching",
			Pods: []PodDef{
				{
					Name:         "mppn-batchp1",
					ExpectedNode: "mppn-batchn1",
					Affinity:     []string{"mppn-batchn1", "mppn-batchn2"},
				},
				{
					Name:         "mppn-batchp2",
					ExpectedNode: "mppn-batchn1",
					Affinity:     []string{"mppn-batchn1", "mppn-batchn2"},
				},
				{
					Name:         "mppn-batchp3",
					ExpectedNode: "mppn-batchn4",
					Affinity:     []string{"mppn-batchn4", "mppn-batchn3"},
				},
				{
					Name:         "mppn-batchp4",
					ExpectedNode: "mppn-batchn4",
					Affinity:     []string{"mppn-batchn4", "mppn-batchn3"},
				},
			},
			Nodes: []NodeDef{
				{
					Name:    "mppn-batchn4",
					MaxPods: 2,
				},
				{
					Name:    "mppn-batchn3",
					MaxPods: 2,
				},
				{
					Name:    "mppn-batchn2",
					MaxPods: 2,
				},
				{
					Name:    "mppn-batchn1",
					MaxPods: 2,
				},
			},
		},
	}

	for _, tt := range table {
		t.Run(tt.Name, func(t *testing.T) {
			finalPods, batched := runScenario(t, tt, true)
			for i, p := range finalPods {
				if p.Spec.NodeName != tt.Pods[i].ExpectedNode {
					t.Fatalf("Invalid node '%s' for pod '%s'. Expected '%s'", p.Spec.NodeName, p.Name, tt.Pods[i].ExpectedNode)
				}
				if batched[i] != tt.Pods[i].ExpectBatched {
					t.Fatalf("Expected pod %s batched %t, actually %t", p.Name, tt.Pods[i].ExpectBatched, batched[i])
				}
			}
		})
	}
}

func newPod(d *PodDef) *v1.Pod {
	aff := &v1.NodeAffinity{}
	if len(d.Affinity) > 0 {
		for i, node := range d.Affinity {
			a := v1.PreferredSchedulingTerm{
				Weight: int32(len(d.Affinity) - i),
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

	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      d.Name,
			UID:       types.UID(d.Name),
			Namespace: "default",
		},
		Spec: v1.PodSpec{
			NodeSelector: d.NodeSelector,
			Affinity: &v1.Affinity{
				NodeAffinity: aff,
			},
			Containers: []v1.Container{
				{
					Name:  "c",
					Image: imageutils.GetPauseImageName(),
					Ports: []v1.ContainerPort{
						{ContainerPort: 1000},
					},
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    *(resource.NewQuantity(10, resource.DecimalSI)),
							v1.ResourceMemory: *(resource.NewQuantity(4*1024*1024, resource.DecimalSI)),
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU:    *(resource.NewQuantity(10, resource.DecimalSI)),
							v1.ResourceMemory: *(resource.NewQuantity(4*1024*1024, resource.DecimalSI)),
						},
					},
				},
			},
		},
	}
}

func newNode(d *NodeDef) *v1.Node {
	return &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:      d.Name,
			UID:       types.UID(d.Name),
			Namespace: "default",
			Labels:    d.Labels,
		},
		Status: v1.NodeStatus{
			Capacity: v1.ResourceList{
				v1.ResourceCPU:    *(resource.NewQuantity(100, resource.DecimalSI)),
				v1.ResourceMemory: *(resource.NewQuantity(4*1024*1024*1024, resource.DecimalSI)),
				v1.ResourcePods:   *resource.NewQuantity(int64(d.MaxPods), resource.DecimalSI),
			},
			Allocatable: v1.ResourceList{
				v1.ResourceCPU:    *(resource.NewQuantity(100, resource.DecimalSI)),
				v1.ResourceMemory: *(resource.NewQuantity(4*1024*1024*1024, resource.DecimalSI)),
				v1.ResourcePods:   *resource.NewQuantity(int64(d.MaxPods), resource.DecimalSI),
			},
		},
	}
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

// To access the test-only field in the framework
type batchGetter interface {
	TotalBatchedPods() int64
}

func runScenario(t *testing.T, tt *Scenario, batch bool) ([]*v1.Pod, []bool) {
	_, tCtx := ktesting.NewTestContext(t)

	cfg, err := newDefaultComponentConfig()
	if err != nil {
		tCtx.Fatalf("Error creating default component config: %v", err)
	}

	enabledFeatures := map[featuregate.Feature]bool{}
	if batch {
		enabledFeatures[featuregate.Feature(features.OpportunisticBatching)] = true
	}

	scheduler, _, testCtx := mustSetupCluster(tCtx, cfg, nil, nil)

	getter := scheduler.Profiles["default-scheduler"].(batchGetter)

	cs := testCtx.Client()

	// Add nodes.
	for _, n := range tt.Nodes {
		_, err := testutil.CreateNode(cs, newNode(&n))
		if err != nil {
			t.Fatal("Failed adding node", "node", n, err)
		}
	}

	finalPods := []*v1.Pod{}
	batched := []bool{}
	for _, pd := range tt.Pods {
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
