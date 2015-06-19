// +build benchmark,!no-etcd,!integration

/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package integration

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/test/integration/framework"
)

// Simple kubelet benchmarks.
// Example usage: go test -bench BenchmarkKubeletCreate -tags 'benchmark' -benchtime 0s -bench-pods 30 -bench-quiet 1
//
// TODO:
// 1. Expose FakeDocker as a cmd line flag.
// 2. Figure out a way to cleanup containers created by real docker.
// 3. Support clean kublet restarts so we can run multiple iterations of the benchmark.

func BenchmarkKubeletCreates(b *testing.B) {
	b.StopTimer()

	// FakeDocker = false uses docker on host machine.
	k := framework.CreateKubeletOrDie(&framework.TestKubeletConfig{RestClient: nil, FakeDocker: true})
	pods := framework.NewPodList(GetPods(b.N), "localhost", api.PodPending)

	b.StartTimer()
	framework.StartPodWorkers(k, pods, true)
	b.StopTimer()

	k.KillUnwantedPods(map[types.UID]kubelet.Empty{}, framework.GetPods(k, false))
}

func BenchmarkKubeletDeletes(b *testing.B) {
	b.StopTimer()

	// FakeDocker = false uses docker on host machine.
	k := framework.CreateKubeletOrDie(&framework.TestKubeletConfig{RestClient: nil, FakeDocker: true})
	pods := framework.NewPodList(GetPods(b.N), "localhost", api.PodPending)
	framework.StartPodWorkers(k, pods, true)
	runningPods := framework.GetPods(k, false)

	b.StartTimer()
	k.KillUnwantedPods(map[types.UID]kubelet.Empty{}, runningPods)
}
