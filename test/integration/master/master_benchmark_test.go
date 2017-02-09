// +build benchmark,!no-etcd,!integration

/*
Copyright 2014 The Kubernetes Authors.

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

package master

import (
	"flag"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/golang/glog"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/kubernetes/pkg/api"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/test/integration/framework"
)

// Command line flag globals, parsed in init and used by the benchmarks:
//	* pods && !tasks: Start -pods, scale number of parallel tasks with b.N
//  * !pods && tasks: Start -tasks, scale pods with b.N
//  * pods && tasks: Ignore b.N, benchmark behaves like a test constrained by -benchtime.
//  * !pods && !tasks: scale pods and workers with b.N.
// -workers specifies the number of workers to shard tasks across.
// Eg: go test bench . -bench-pods 3000 -bench-tasks 100 -bench-tasks 10:
// Create 100 tasks each listing 3000 pods, and run them 10 at a time.
var (
	Workers int
	Pods    int
	Tasks   int
)

const Glog_fatalf = 3

func init() {
	q := flag.Int("bench-quiet", 3, "Quietness, don't glog severities <= given level during the benchmark.")
	pods := flag.Int("bench-pods", -1, "Number of pods for the benchmark. If unspecified, uses b.N.")
	workers := flag.Int("bench-workers", -1, "Number workers for the benchmark. If unspecified, uses tasks.")
	tasks := flag.Int("bench-tasks", -1, "Number of tasks for the benchmark. These tasks are sharded across the workers. If unspecified, uses b.N.")
	flag.Parse()

	// Unfortunately this v level goes in the opposite direction as stderrthreshold.
	flag.Set("v", fmt.Sprintf("%d", *q))

	// We need quiet logs to parse benchmark results, which includes Errorf.
	flag.Set("logtostderr", "false")
	flag.Set("stderrthreshold", fmt.Sprintf("%d", Glog_fatalf-*q))
	Pods = *pods
	Workers = *workers
	Tasks = *tasks
}

// getPods returns the cmd line -pods or b.N if -pods wasn't specified.
// Benchmarks can call getPods to get the number of pods they need to
// create for a given benchmark.
func getPods(bN int) int {
	if Pods < 0 {
		return bN
	}
	return Pods
}

// getTasks returns the cmd line -workers or b.N if -workers wasn't specified.
// Benchmarks would call getTasks to get the number of workers required to
// perform the benchmark in parallel.
func getTasks(bN int) int {
	if Tasks < 0 {
		return bN
	}
	return Tasks
}

// getIterations returns the number of iterations required by each benchmark for
// go to produce reliable timing results.
func getIterations(bN int) int {
	// Anything with constant pods is only linear if we iterate b.N times.
	if Pods > 0 {
		return bN
	}
	return 1
}

// startPodsOnNodes creates numPods sharded across numNodes
func startPodsOnNodes(ns string, numPods, numNodes int, restClient clientset.Interface) {
	podsPerNode := numPods / numNodes
	if podsPerNode < 1 {
		podsPerNode = 1
	}
	framework.RunParallel(func(id int) error {
		return framework.StartPods(podsPerNode, fmt.Sprintf("host.%d", id), restClient)
	}, numNodes, -1)
}

// Benchmark pod listing by waiting on `Tasks` listers to list `Pods` pods via `Workers`.
func BenchmarkPodList(b *testing.B) {
	b.StopTimer()
	m := framework.NewMasterComponents(&framework.Config{nil, true, false, 250.0, 500})
	defer m.Stop(true, true)

	ns := framework.CreateTestingNamespace("benchmark-pod-list", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	numPods, numTasks, iter := getPods(b.N), getTasks(b.N), getIterations(b.N)
	podsPerNode := numPods / numTasks
	if podsPerNode < 1 {
		podsPerNode = 1
	}
	glog.Infof("Starting benchmark: b.N %d, pods %d, workers %d, podsPerNode %d",
		b.N, numPods, numTasks, podsPerNode)

	startPodsOnNodes(ns.Name, numPods, numTasks, m.RestClient)
	// Stop the rc manager so it doesn't steal resources
	m.Stop(false, true)

	b.StartTimer()
	for i := 0; i < iter; i++ {
		framework.RunParallel(func(id int) error {
			host := fmt.Sprintf("host.%d", id)
			now := time.Now()
			defer func() {
				glog.V(3).Infof("Worker %d: Node %v listing pods took %v", id, host, time.Since(now))
			}()
			if pods, err := m.ClientSet.Core().Pods(ns.Name).List(metav1.ListOptions{
				LabelSelector: labels.Everything(),
				FieldSelector: fields.OneTermEqualSelector(api.PodHostField, host),
			}); err != nil {
				return err
			} else if len(pods.Items) < podsPerNode {
				glog.Fatalf("List retrieved %d pods, which is less than %d", len(pods.Items), podsPerNode)
			}
			return nil
		}, numTasks, Workers)
	}
	b.StopTimer()
}

// Benchmark pod listing by waiting on `Tasks` listers to list `Pods` pods via `Workers`.
func BenchmarkPodListEtcd(b *testing.B) {
	b.StopTimer()
	m := framework.NewMasterComponents(&framework.Config{nil, true, false, 250.0, 500})
	defer m.Stop(true, true)

	ns := framework.CreateTestingNamespace("benchmark-pod-list-etcd", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	numPods, numTasks, iter := getPods(b.N), getTasks(b.N), getIterations(b.N)
	podsPerNode := numPods / numTasks
	if podsPerNode < 1 {
		podsPerNode = 1
	}

	startPodsOnNodes(ns.Name, numPods, numTasks, m.RestClient)
	// Stop the rc manager so it doesn't steal resources
	m.Stop(false, true)

	glog.Infof("Starting benchmark: b.N %d, pods %d, workers %d, podsPerNode %d",
		b.N, numPods, numTasks, podsPerNode)

	b.StartTimer()
	for i := 0; i < iter; i++ {
		framework.RunParallel(func(id int) error {
			now := time.Now()
			defer func() {
				glog.V(3).Infof("Worker %d: listing pods took %v", id, time.Since(now))
			}()
			pods, err := m.ClientSet.Core().Pods(ns.Name).List(metav1.ListOptions{
				LabelSelector: labels.Everything(),
				FieldSelector: fields.Everything(),
			})
			if err != nil {
				return err
			}
			if len(pods.Items) < numPods {
				glog.Fatalf("List retrieved %d pods, which is less than %d", len(pods.Items), numPods)
			}
			return nil
		}, numTasks, Workers)
	}
	b.StopTimer()
}

func TestMain(m *testing.M) {
	os.Exit(m.Run())
}
