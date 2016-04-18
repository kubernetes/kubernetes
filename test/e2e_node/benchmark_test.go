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

package e2e_node

import (
	"errors"
	"fmt"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/restclient"
	client "k8s.io/kubernetes/pkg/client/unversioned"

	. "github.com/onsi/ginkgo"
)

const (
	concurrencyCount = 10
	iterCount        = 10
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

var _ = Describe("Container Conformance Test", func() {
	var cl *client.Client

	BeforeEach(func() {
		// Setup the apiserver client
		cl = client.NewOrDie(&restclient.Config{Host: *apiServerAddress})
	})

	Describe("container benchmark blackbox test", func() {
		Context("when list testing images", func() {
			var conformImages []ConformanceImage
			conformImageTags := []string{
				"gcr.io/google_containers/busybox",
				"gcr.io/google_containers/nginx",
			}
			Measure("it should pull successfully [Performance]", func(b Benchmarker) {
				b.Time("pull images", func() {
					for _, imageTag := range conformImageTags {
						image, _ := NewConformanceImage("docker", imageTag)
						conformImages = append(conformImages, image)

						for start := time.Now(); time.Since(start) < imageRetryTimeout; time.Sleep(imagePullInterval) {
							if err := image.Pull(); err == nil {
								break
							}
						}
					}
				})
			}, 1)
			Measure("it should list images successfully [Performance]", func(b Benchmarker) {
				b.Time("list images", func() {
					RunParallel(func(id int) error {
						image, _ := NewConformanceImage("docker", "")
						_, err := image.List()
						return err
					}, concurrencyCount)
				})
			}, iterCount)

			Measure("it should remove successfully [Performance]", func(b Benchmarker) {
				b.Time("remove images", func() {
					for _, image := range conformImages {
						image.Remove()
					}
				})
			}, 1)
		})

		Context("when running containers concurrency", func() {
			Measure("it should start containers successfully [Performance]", func(b Benchmarker) {
				b.Time("start containers", func() {
					RunParallel(func(id int) error {
						containerName := fmt.Sprintf("busybox-%d", id)
						container := ConformanceContainer{
							Container: api.Container{
								Image:           "gcr.io/google_containers/busybox",
								Name:            containerName,
								Command:         []string{"sh", "-c", "env"},
								ImagePullPolicy: api.PullIfNotPresent,
							},
							Client:   cl,
							NodeName: *nodeName,
						}
						if err := container.Create(); err != nil {
							return err
						}

						for start := time.Now(); time.Since(start) < retryTimeout; time.Sleep(pollInterval) {
							if pod, err := container.Get(); err != nil {
								return err
							} else if pod.Phase != api.PodPending {
								return nil
							}
						}

						return errors.New("Pending to start container")
					}, concurrencyCount)
				})
				b.Time("get container", func() {
					RunParallel(func(id int) error {
						containerName := fmt.Sprintf("busybox-%d", id)
						container := ConformanceContainer{
							Container: api.Container{
								Name: containerName,
							},
							Client: cl}
						_, err := container.Get()
						return err
					}, concurrencyCount)
				})
				b.Time("list containers", func() {
					RunParallel(func(id int) error {
						container := ConformanceContainer{Client: cl}
						_, err := container.List()
						return err
					}, concurrencyCount)
				})
				b.Time("stop container", func() {
					RunParallel(func(id int) error {
						containerName := fmt.Sprintf("busybox-%d", id)
						container := ConformanceContainer{
							Container: api.Container{
								Name: containerName,
							},
							Client: cl}
						return container.Delete()
					}, concurrencyCount)
				})
			}, iterCount)
		})
	})
})
