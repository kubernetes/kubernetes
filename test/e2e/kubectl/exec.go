/*
Copyright 2024 The Kubernetes Authors.

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

// OWNER = sig/cli

package kubectl

import (
	"context"
	"time"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
)

func worker(f *framework.Framework, pod *v1.Pod, id int, jobs <-chan int, results chan<- error) {
	for j := range jobs {
		framework.Logf("Worker: %d Job: %d", id, j)
		func() {
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			stdout, stderr, err := e2epod.ExecWithOptionsContext(ctx, f, e2epod.ExecOptions{
				Command:            []string{"date"},
				Namespace:          f.Namespace.Name,
				PodName:            pod.Name,
				ContainerName:      pod.Spec.Containers[0].Name,
				Stdin:              nil,
				CaptureStdout:      true,
				CaptureStderr:      true,
				PreserveWhitespace: false,
			})
			if err != nil {
				framework.Logf("Try: %d Error: %v stdout: %s stderr: %s", j, err, stdout, stderr)
			}
			results <- err
		}()
	}
}

var _ = SIGDescribe("Kubectl exec", func() {
	f := framework.NewDefaultFramework("exec")

	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline
	f.It("should be able to execute 1000 times in a container", func(ctx context.Context) {
		const size = 1000
		ns := f.Namespace.Name
		podName := "test-exec-pod"
		jobs := make(chan int, size)
		results := make(chan error, size)

		pod := e2epod.NewAgnhostPod(ns, podName, nil, nil, nil)
		pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)

		// 10 workers for 1000 executions
		ginkgo.By("Starting workers to exec on pod")
		for w := 0; w < 10; w++ {
			framework.Logf("Starting worker %d", w)
			go worker(f, pod, w, jobs, results)
		}
		for i := 0; i < size; i++ {
			framework.Logf("Sending job %d", i)
			jobs <- i
		}
		ginkgo.By("All jobs processed")
		close(jobs)

		errors := []error{}
		for c := 0; c < size; c++ {
			framework.Logf("Getting results %d", c)
			err := <-results
			if err != nil {
				errors = append(errors, err)
			}
		}
		// Accept a 99% success rate to be able to handle infrastructure errors
		if len(errors) > (size*1)/100 {
			framework.Failf("Exec failed %d times with following errors : %v", len(errors), errors)
		}
	})
})
