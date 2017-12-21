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

package benchmark

import (
	"fmt"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/kubernetes/pkg/kubelet/apis"
	"k8s.io/kubernetes/test/integration/framework"
	testutils "k8s.io/kubernetes/test/utils"

	"github.com/golang/glog"
)

// BenchmarkScheduling benchmarks the scheduling rate when the cluster has
// various quantities of nodes and scheduled pods.
func BenchmarkScheduling(b *testing.B) {
	tests := []struct{ nodes, existingPods, minPods int }{
		{nodes: 100, existingPods: 0, minPods: 100},
		{nodes: 100, existingPods: 1000, minPods: 100},
		{nodes: 1000, existingPods: 0, minPods: 100},
		{nodes: 1000, existingPods: 1000, minPods: 100},
	}
	setupStrategy := testutils.NewSimpleWithControllerCreatePodStrategy("rc1")
	testStrategy := testutils.NewSimpleWithControllerCreatePodStrategy("rc2")
	for _, test := range tests {
		name := fmt.Sprintf("%vNodes/%vPods", test.nodes, test.existingPods)
		b.Run(name, func(b *testing.B) {
			benchmarkScheduling(test.nodes, test.existingPods, test.minPods, setupStrategy, testStrategy, b)
		})
	}
}

// BenchmarkSchedulingAntiAffinity benchmarks the scheduling rate of pods with
// PodAntiAffinity rules when the cluster has various quantities of nodes and
// scheduled pods.
func BenchmarkSchedulingAntiAffinity(b *testing.B) {
	tests := []struct{ nodes, existingPods, minPods int }{
		{nodes: 500, existingPods: 250, minPods: 250},
		{nodes: 500, existingPods: 5000, minPods: 250},
	}
	// The setup strategy creates pods with no affinity rules.
	setupStrategy := testutils.NewSimpleWithControllerCreatePodStrategy("setup")
	// The test strategy creates pods with anti-affinity for each other.
	testBasePod := makeBasePodWithAntiAffinity(
		map[string]string{"name": "test", "color": "green"},
		map[string]string{"color": "green"})
	testStrategy := testutils.NewCustomCreatePodStrategy(testBasePod)
	for _, test := range tests {
		name := fmt.Sprintf("%vNodes/%vPods", test.nodes, test.existingPods)
		b.Run(name, func(b *testing.B) {
			benchmarkScheduling(test.nodes, test.existingPods, test.minPods, setupStrategy, testStrategy, b)
		})
	}

}

// makeBasePodWithAntiAffinity creates a Pod object to be used as a template.
// The Pod has a PodAntiAffinity requirement against pods with the given labels.
func makeBasePodWithAntiAffinity(podLabels, affinityLabels map[string]string) *v1.Pod {
	basePod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "affinity-pod-",
			Labels:       podLabels,
		},
		Spec: testutils.MakePodSpec(),
	}
	basePod.Spec.Affinity = &v1.Affinity{
		PodAntiAffinity: &v1.PodAntiAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
				{
					LabelSelector: &metav1.LabelSelector{
						MatchLabels: affinityLabels,
					},
					TopologyKey: apis.LabelHostname,
				},
			},
		},
	}
	return basePod
}

// benchmarkScheduling benchmarks scheduling rate with specific number of nodes
// and specific number of pods already scheduled.
// This will schedule numExistingPods pods before the benchmark starts, and at
// least minPods pods during the benchmark.
func benchmarkScheduling(numNodes, numExistingPods, minPods int,
	setupPodStrategy, testPodStrategy testutils.TestPodCreateStrategy,
	b *testing.B) {
	if b.N < minPods {
		b.N = minPods
	}
	schedulerConfigFactory, finalFunc := mustSetupScheduler()
	defer finalFunc()
	c := schedulerConfigFactory.GetClient()

	nodePreparer := framework.NewIntegrationTestNodePreparer(
		c,
		[]testutils.CountToStrategy{{Count: numNodes, Strategy: &testutils.TrivialNodePrepareStrategy{}}},
		"scheduler-perf-",
	)
	if err := nodePreparer.PrepareNodes(); err != nil {
		glog.Fatalf("%v", err)
	}
	defer nodePreparer.CleanupNodes()

	config := testutils.NewTestPodCreatorConfig()
	config.AddStrategy("sched-test", numExistingPods, setupPodStrategy)
	podCreator := testutils.NewTestPodCreator(c, config)
	podCreator.CreatePods()

	for {
		scheduled, err := schedulerConfigFactory.GetScheduledPodLister().List(labels.Everything())
		if err != nil {
			glog.Fatalf("%v", err)
		}
		if len(scheduled) >= numExistingPods {
			break
		}
		time.Sleep(1 * time.Second)
	}
	// start benchmark
	b.ResetTimer()
	config = testutils.NewTestPodCreatorConfig()
	config.AddStrategy("sched-test", b.N, testPodStrategy)
	podCreator = testutils.NewTestPodCreator(c, config)
	podCreator.CreatePods()
	for {
		// This can potentially affect performance of scheduler, since List() is done under mutex.
		// TODO: Setup watch on apiserver and wait until all pods scheduled.
		scheduled, err := schedulerConfigFactory.GetScheduledPodLister().List(labels.Everything())
		if err != nil {
			glog.Fatalf("%v", err)
		}
		if len(scheduled) >= numExistingPods+b.N {
			break
		}
		// Note: This might introduce slight deviation in accuracy of benchmark results.
		// Since the total amount of time is relatively large, it might not be a concern.
		time.Sleep(100 * time.Millisecond)
	}
}
