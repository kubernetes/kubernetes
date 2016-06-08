/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package docker_performance

import (
	"sync"

	"k8s.io/kubernetes/test/docker_validation/performance/helpers"

	docker "github.com/fsouza/go-dockerclient"
	. "github.com/onsi/ginkgo"
)

// Docker configuration
var (
	endpoint = "unix:///var/run/docker.sock"
)

const (
	concurrencyCount = 2
	containerCount   = 10
)

type Task func(id int) error

//TODO merge this with comparable utilities in integration/framework/master_utils.go
// RunParallel spawns a goroutine per task in the given queue
func RunParallel(task Task, count int) {
	var wg sync.WaitGroup
	semCh := make(chan struct{}, count)
	wg.Add(count)
	for id := 0; id < count; id++ {
		go func(id int) {
			semCh <- struct{}{}
			task(id)
			<-semCh
			wg.Done()
		}(id)
	}
	wg.Wait()
	close(semCh)
}

var _ = Describe("Docker performance Test", func() {
	var client *docker.Client

	BeforeEach(func() {
		client, _ = docker.NewClient(endpoint)
		client.PullImage(docker.PullImageOptions{Repository: "busybox", Tag: "latest"}, docker.AuthConfiguration{})
	})

	Context("when benchmark on container operations [Performance]", func() {
		var ids []string
		BeforeEach(func() {
			ids = helpers.CreateContainers(client, containerCount)
		})

		Measure("start containers", func(b Benchmarker) {
			b.Time("start", func() {
				for id := 0; id < containerCount; id++ {
					client.StartContainer(ids[id], &docker.HostConfig{})
				}
			})
			helpers.StopContainers(client, ids)
		}, concurrencyCount)

		Measure("start containers in parallel", func(b Benchmarker) {
			b.Time("start in parallel", func() {
				RunParallel(func(id int) error {
					client.StartContainer(ids[id], &docker.HostConfig{})
					return nil
				}, containerCount)
			})
			helpers.StopContainers(client, ids)
		}, concurrencyCount)

		Measure("list containers", func(b Benchmarker) {
			listAll := true
			b.Time("list", func() {
				for id := 0; id < containerCount; id++ {
					client.ListContainers(docker.ListContainersOptions{All: listAll})
				}
			})
		}, concurrencyCount)

		Measure("list containers in parallel", func(b Benchmarker) {
			listAll := true
			b.Time("list in parallel", func() {
				RunParallel(func(id int) error {
					client.ListContainers(docker.ListContainersOptions{All: listAll})
					return nil
				}, containerCount)
			})
		}, concurrencyCount)

		AfterEach(func() {
			helpers.RemoveContainers(client, ids)
		})
	})
})
