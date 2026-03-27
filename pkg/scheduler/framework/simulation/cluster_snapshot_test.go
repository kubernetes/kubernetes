package simulation

import (
	"fmt"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	schedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func init() {
	metrics.Register()
}

var (
	defaultConfig = &schedulerconfig.KubeSchedulerConfiguration{
		Profiles: []schedulerconfig.KubeSchedulerProfile{
			{
				SchedulerName: v1.DefaultSchedulerName,
				Plugins: &schedulerconfig.Plugins{
					QueueSort: schedulerconfig.PluginSet{
						Enabled: []schedulerconfig.Plugin{
							{Name: "PrioritySort"},
						},
					},
					PreFilter: schedulerconfig.PluginSet{
						Enabled: []schedulerconfig.Plugin{
							{Name: "NodeResourcesFit"},
						},
					},
					Filter: schedulerconfig.PluginSet{
						Enabled: []schedulerconfig.Plugin{
							{Name: "NodeResourcesFit"},
						},
					},
					Bind: schedulerconfig.PluginSet{
						Enabled: []schedulerconfig.Plugin{
							{Name: "DefaultBinder"},
						},
					},
				},
				PluginConfig: []schedulerconfig.PluginConfig{
					{
						Name: "NodeResourcesFit",
						Args: &schedulerconfig.NodeResourcesFitArgs{
							ScoringStrategy: &schedulerconfig.ScoringStrategy{
								Type: schedulerconfig.LeastAllocated,
							},
						},
					},
				},
			},
		},
	}
	smallRes = map[v1.ResourceName]string{
		v1.ResourceCPU:    "100m",
		v1.ResourceMemory: "100",
	}
	mediumRes = map[v1.ResourceName]string{
		v1.ResourceCPU:    "200m",
		v1.ResourceMemory: "200",
	}
	largeRes = map[v1.ResourceName]string{
		v1.ResourceCPU:    "300m",
		v1.ResourceMemory: "300",
	}
	veryLargeRes = map[v1.ResourceName]string{
		v1.ResourceCPU:    "500m",
		v1.ResourceMemory: "500",
	}
)

func TestCreation(t *testing.T) {
	ctx := t.Context()

	NewSchedulingSimulator(ctx, &ClusterData{}, defaultConfig, Delta)
}

func TestSchedulePods(t *testing.T) {
	ctx := t.Context()
	cs := clientsetfake.NewClientset()

	informerFactory := func() informers.SharedInformerFactory {
		return informers.NewSharedInformerFactory(cs, 0)
	}

	amount := 10000
	nodes := make([]*v1.Node, amount)
	candidateNodes := make([]string, amount)
	pods := make([]*v1.Pod, amount)
	for i := 0; i < amount; i++ {
		candidateNodes[i] = fmt.Sprintf("node%d", i)
		nodes[i] = st.MakeNode().Name(candidateNodes[i]).Capacity(largeRes).Obj()
		pods[i] = st.MakePod().Name(fmt.Sprintf("pod%d", i)).Namespace("ns1").UID(fmt.Sprintf("pod%dns1", i)).Req(largeRes).Obj()
	}

	schedulablePods := make([]SchedulablePod, amount)
	for i, p := range pods {
		schedulablePods[i] = SchedulablePod{
			pod:            p,
			candidateNodes: candidateNodes,
		}
	}
	unschedulablePod := SchedulablePod{
		pod:            st.MakePod().Name("unschedulable").Namespace("ns1").UID("unschedulable").Req(largeRes).Obj(),
		candidateNodes: candidateNodes,
	}

	testSimulator := func(sim SchedulingSimulator) {
		sim.Fork()
		res, err := sim.SchedulePods(schedulablePods, Options{})
		if err != nil {
			t.Fatalf("failed to schedule pods: %v", err)
		}
		if len(res) != amount {
			t.Errorf("expected %d scheduled pods, got %d", amount, len(res))
		}
		sim.Commit()

		sim.Fork()
		res, err = sim.SchedulePods([]SchedulablePod{unschedulablePod}, Options{})
		if err != nil {
			t.Fatalf("failed to schedule pods: %v", err)
		}
		if len(res) != 0 {
			t.Errorf("expected %d scheduled pods, got %d", 0, len(res))
		}
		sim.Commit()
	}

	for _, storageType := range []StorageType{Delta, UndoLog} {
		sim, err := NewSchedulingSimulator(ctx, &ClusterData{informerFactory: informerFactory(), nodes: nodes}, defaultConfig, storageType)
		if err != nil {
			t.Fatalf("failed to create simulator: %v", err)
		}
		testSimulator(sim)
	}

}

func BenchmarkSimulator(b *testing.B) {
	singlePodReq := 100
	resources := func(amount int) map[v1.ResourceName]string {
		return map[v1.ResourceName]string{
			v1.ResourceCPU:    fmt.Sprintf("%dm", amount),
			v1.ResourceMemory: fmt.Sprintf("%d", amount),
		}
	}
	nodes := func(count, podsPerNode int) []*v1.Node {
		res := make([]*v1.Node, count)
		cap := resources(podsPerNode * singlePodReq)
		for i := 0; i < count; i++ {
			res[i] = st.MakeNode().Name(fmt.Sprintf("node%d", i)).Capacity(cap).Obj()
		}
		return res
	}

	pods := make([]*v1.Pod, 100000)
	podReq := resources(singlePodReq)
	for i := 0; i < 100000; i++ {
		pods[i] = st.MakePod().Name(fmt.Sprintf("pod%d", i)).Namespace("ns1").UID(fmt.Sprintf("pod%dns1", i)).Req(podReq).Obj()
	}

	schedulablePods := func(podCount, nodeCount int) []SchedulablePod {
		candidatesNodes := make([]string, nodeCount)
		for i := 0; i < nodeCount; i++ {
			candidatesNodes[i] = fmt.Sprintf("node%d", i)
		}

		schedulablePods := make([]SchedulablePod, podCount)
		for i := 0; i < podCount; i++ {
			schedulablePods[i] = SchedulablePod{
				pod:            pods[i],
				candidateNodes: candidatesNodes,
			}
		}
		return schedulablePods
	}
	scheduledPods := func(amount int) []ScheduledPod {
		res := make([]ScheduledPod, amount)
		for i := 0; i < amount; i++ {
			res[i] = ScheduledPod{
				pod:          pods[i],
				selectedNode: fmt.Sprintf("node%d", i),
			}
		}
		return res
	}

	informerFactory := func() informers.SharedInformerFactory {
		cs := clientsetfake.NewClientset()
		return informers.NewSharedInformerFactory(cs, 0)
	}

	type operation func(SchedulingSimulator, []SchedulablePod, Options, []ScheduledPod) error
	var (
		schedulePods operation = func(sim SchedulingSimulator, schedulablePods []SchedulablePod, opts Options, _ []ScheduledPod) error {
			_, err := sim.SchedulePods(schedulablePods, opts)
			return err
		}
		unschedulablePods operation = func(sim SchedulingSimulator, _ []SchedulablePod, _ Options, scheduledPods []ScheduledPod) error {
			return sim.UnschedulePods(scheduledPods)
		}
		fork operation = func(sim SchedulingSimulator, _ []SchedulablePod, _ Options, _ []ScheduledPod) error {
			sim.Fork()
			return nil
		}
		commit operation = func(sim SchedulingSimulator, _ []SchedulablePod, _ Options, _ []ScheduledPod) error {
			return sim.Commit()
		}
		revert operation = func(sim SchedulingSimulator, _ []SchedulablePod, _ Options, _ []ScheduledPod) error {
			return sim.Revert()
		}
	)

	tests := []struct {
		name            string
		initNodes       []*v1.Node
		initPods        []*v1.Pod
		schedulablePods []SchedulablePod
		scheduledPods   []ScheduledPod
		ops             []operation
	}{
		{
			name:            "Fork -> Schedule 1000 pods -> Commit",
			initNodes:       nodes(1000, 1),
			schedulablePods: schedulablePods(1000, 1000),
			ops: []operation{
				fork,
				schedulePods,
				commit,
			},
		},
		{
			name:            "Fork -> Schedule 1000 pods -> Unschedule 1000 pods -> Commit",
			initNodes:       nodes(1000, 1),
			schedulablePods: schedulablePods(1000, 1000),
			scheduledPods:   scheduledPods(1000),
			ops: []operation{
				fork,
				schedulePods,
				unschedulablePods,
				commit,
			},
		},
		{
			name:            "Fork -> Schedule 1000 pods -> Revert",
			initNodes:       nodes(1000, 1),
			schedulablePods: schedulablePods(1000, 1000),
			ops: []operation{
				fork,
				schedulePods,
				revert,
			},
		},
		{
			name:            "Fork -> Schedule 30k pods among 1k nodes -> Commit",
			initNodes:       nodes(1000, 30),
			schedulablePods: schedulablePods(30000, 1000),
			ops: []operation{
				fork,
				schedulePods,
				commit,
			},
		},
		{
			name:            "Fork -> Schedule 30k pods among 1k nodes -> Revert",
			initNodes:       nodes(1000, 30),
			schedulablePods: schedulablePods(30000, 1000),
			ops: []operation{
				fork,
				schedulePods,
				revert,
			},
		},
	}
	for _, tt := range tests {
		for _, storageType := range []StorageType{Delta, UndoLog} {

			benchName := fmt.Sprintf("%s/%s", tt.name, storageType)

			b.Run(benchName, func(b *testing.B) {
				ctx := b.Context()

				sim, err := NewSchedulingSimulator(ctx, &ClusterData{
					informerFactory: informerFactory(),
					nodes:           tt.initNodes,
				}, defaultConfig, storageType)

				if err != nil {
					b.Fatalf("failed to create simulator: %v", err)
				}

				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					for _, op := range tt.ops {
						if err := op(sim, tt.schedulablePods, Options{}, nil); err != nil {
							b.Fatal(err)
						}
					}
				}
			})
		}
	}

}
