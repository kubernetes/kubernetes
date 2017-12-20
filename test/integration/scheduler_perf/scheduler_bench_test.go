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

	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/kubernetes/test/integration/framework"
	testutils "k8s.io/kubernetes/test/utils"

	"github.com/golang/glog"
)

// BenchmarkScheduling benchmarks the scheduling rate when the cluster has
// various quantities of nodes and scheduled pods.
func BenchmarkScheduling(b *testing.B) {
	tests := []struct{ nodes, pods int }{
		{nodes: 100, pods: 0},
		{nodes: 100, pods: 1000},
		{nodes: 1000, pods: 0},
		{nodes: 1000, pods: 1000},
	}
	for _, test := range tests {
		name := fmt.Sprintf("%vNodes/%vPods", test.nodes, test.pods)
		b.Run(name, func(b *testing.B) { benchmarkScheduling(test.nodes, test.pods, b) })
	}
}

// benchmarkScheduling benchmarks scheduling rate with specific number of nodes
// and specific number of pods already scheduled. Since an operation takes relatively
// long time, b.N should be small: 10 - 100.
func benchmarkScheduling(numNodes, numScheduledPods int, b *testing.B) {
	schedulerConfigFactory, finalFunc := mustSetupScheduler(metricServer)
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
	config.AddStrategy("sched-test", numScheduledPods, testutils.NewSimpleWithControllerCreatePodStrategy("rc1"))
	podCreator := testutils.NewTestPodCreator(c, config)
	podCreator.CreatePods()

	for {
		scheduled, err := schedulerConfigFactory.GetScheduledPodLister().List(labels.Everything())
		if err != nil {
			glog.Fatalf("%v", err)
		}
		if len(scheduled) >= numScheduledPods {
			break
		}
		time.Sleep(1 * time.Second)
	}
	// start benchmark
	b.ResetTimer()
	config = testutils.NewTestPodCreatorConfig()
	config.AddStrategy("sched-test", b.N, testutils.NewSimpleWithControllerCreatePodStrategy("rc2"))
	podCreator = testutils.NewTestPodCreator(c, config)
	podCreator.CreatePods()
	for {
		// This can potentially affect performance of scheduler, since List() is done under mutex.
		// TODO: Setup watch on apiserver and wait until all pods scheduled.
		scheduled, err := schedulerConfigFactory.GetScheduledPodLister().List(labels.Everything())
		if err != nil {
			glog.Fatalf("%v", err)
		}
		if len(scheduled) >= numScheduledPods+b.N {
			break
		}
		// Note: This might introduce slight deviation in accuracy of benchmark results.
		// Since the total amount of time is relatively large, it might not be a concern.
		time.Sleep(100 * time.Millisecond)
	}
}
