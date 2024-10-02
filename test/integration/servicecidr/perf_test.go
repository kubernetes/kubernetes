/*
Copyright 2023 The Kubernetes Authors.

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

package servicecidr

import (
	"context"
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
)

// TestServiceAllocPerformance measure the latency to create N services with a parallelism of K
// using the old and the new ClusterIP allocators.
// The test is skipped to run on CI and is left to execute manually to check for possible regressions.
// The current results with 100 works and 15k services on a (n2-standard-48) vCPU: 48 RAM: 192 GB are:
// legacy perf_test.go:139: [RESULT] Duration 1m9.646167533s: [quantile:0.5  value:0.462886801 quantile:0.9  value:0.496662838 quantile:0.99  value:0.725845905]
// new perf_test.go:139: [RESULT] Duration 2m12.900694343s: [quantile:0.5  value:0.481814448 quantile:0.9  value:1.3867615469999999 quantile:0.99  value:1.888190671]
func TestServiceAllocPerformance(t *testing.T) {
	t.Skip("KEP-1880 performance comparison")
	serviceCreation := metrics.NewHistogram(&metrics.HistogramOpts{
		Name:    "service_duration_seconds",
		Help:    "A summary of the Service creation durations in seconds.",
		Buckets: metrics.DefBuckets,
	})
	legacyregistry.MustRegister(serviceCreation)

	svc := func(i, j int) *v1.Service {
		return &v1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name: fmt.Sprintf("svc-%v-%v", i, j),
			},
			Spec: v1.ServiceSpec{
				Type: v1.ServiceTypeClusterIP,
				Ports: []v1.ServicePort{
					{Port: 80},
				},
			},
		}
	}

	worker := func(client clientset.Interface, id int, jobs <-chan int, results chan<- error) {
		for j := range jobs {
			t.Logf("Worker: %d Job: %d", id, j)
			func() {
				now := time.Now()
				defer func() {
					t.Logf("worker %d job %d took %v", id, j, time.Since(now))
					serviceCreation.Observe(time.Since(now).Seconds())
				}()
				ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
				defer cancel()
				_, err := client.CoreV1().Services(metav1.NamespaceDefault).Create(ctx, svc(id, j), metav1.CreateOptions{})
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				results <- err
			}()
		}
	}

	for _, gate := range []bool{false, true} {
		t.Run(fmt.Sprintf("feature-gate=%v", gate), func(t *testing.T) {
			etcdOptions := framework.SharedEtcd()
			apiServerOptions := kubeapiservertesting.NewDefaultTestServerOptions()
			s1 := kubeapiservertesting.StartTestServerOrDie(t,
				apiServerOptions,
				[]string{
					"--runtime-config=networking.k8s.io/v1beta1=true",
					"--service-cluster-ip-range=" + "10.0.0.0/12",
					"--advertise-address=10.0.0.1",
					"--disable-admission-plugins=ServiceAccount",
					fmt.Sprintf("--feature-gates=%s=true,%s=true", features.MultiCIDRServiceAllocator, features.DisableAllocatorDualWrite),
				},
				etcdOptions)

			defer s1.TearDownFn()

			client, err := clientset.NewForConfig(s1.ClientConfig)
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			legacyregistry.Reset()

			// 100 workers for 15k services
			nworkers := 100
			nservices := 150
			jobs := make(chan int, nservices)
			results := make(chan error, nservices)
			t.Log("Starting workers to create ClusterIP Service")
			now := time.Now()
			for w := 0; w < nworkers; w++ {
				t.Logf("Starting worker %d", w)
				go worker(client, w, jobs, results)
			}
			for i := 0; i < nservices; i++ {
				t.Logf("Sending job %d", i)
				jobs <- i
			}
			t.Log("All jobs processed")
			close(jobs)

			for c := 0; c < nservices; c++ {
				t.Logf("Getting results %d", c)
				err := <-results
				if err != nil {
					t.Errorf("error creating service: %v", err)
				}
			}

			vec, err := testutil.GetHistogramVecFromGatherer(legacyregistry.DefaultGatherer, serviceCreation.Name, map[string]string{})
			if err != nil {
				t.Error(err)
			}

			t.Logf("[RESULT] feature-gate=%v Duration: %v Avg: %.4f p95: %.4f p99: %.4f", gate, time.Since(now), vec.Average(), vec.Quantile(0.95), vec.Quantile(0.99))
		})
	}
}
