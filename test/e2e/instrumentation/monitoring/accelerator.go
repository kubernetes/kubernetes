/*
Copyright 2017 The Kubernetes Authors.

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

package monitoring

import (
	"context"
	"os"
	"time"

	"github.com/onsi/ginkgo"
	"golang.org/x/oauth2/google"
	gcm "google.golang.org/api/monitoring/v3"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/gpu"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	instrumentation "k8s.io/kubernetes/test/e2e/instrumentation/common"
	"k8s.io/kubernetes/test/e2e/scheduling"
	"k8s.io/kubernetes/test/utils/image"
)

// Stackdriver container accelerator metrics, as described here:
// https://cloud.google.com/monitoring/api/metrics_gcp#gcp-container
var acceleratorMetrics = []string{
	"accelerator/duty_cycle",
	"accelerator/memory_total",
	"accelerator/memory_used",
}

var _ = instrumentation.SIGDescribe("Stackdriver Monitoring", func() {
	ginkgo.BeforeEach(func() {
		framework.SkipUnlessProviderIs("gce", "gke")
	})

	f := framework.NewDefaultFramework("stackdriver-monitoring")

	ginkgo.It("should have accelerator metrics [Feature:StackdriverAcceleratorMonitoring]", func() {
		testStackdriverAcceleratorMonitoring(f)
	})

})

func testStackdriverAcceleratorMonitoring(f *framework.Framework) {
	projectID := framework.TestContext.CloudConfig.ProjectID

	ctx := context.Background()
	client, err := google.DefaultClient(ctx, gcm.CloudPlatformScope)

	gcmService, err := gcm.New(client)

	framework.ExpectNoError(err)

	// set this env var if accessing Stackdriver test endpoint (default is prod):
	// $ export STACKDRIVER_API_ENDPOINT_OVERRIDE=https://test-monitoring.sandbox.googleapis.com/
	basePathOverride := os.Getenv("STACKDRIVER_API_ENDPOINT_OVERRIDE")
	if basePathOverride != "" {
		gcmService.BasePath = basePathOverride
	}

	scheduling.SetupNVIDIAGPUNode(f, false)

	f.PodClient().Create(&v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: rcName,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Name:    rcName,
					Image:   image.GetE2EImage(image.CudaVectorAdd),
					Command: []string{"/bin/sh", "-c"},
					Args:    []string{"nvidia-smi && sleep infinity"},
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							gpu.NVIDIAGPUResourceName: *resource.NewQuantity(1, resource.DecimalSI),
						},
					},
				},
			},
		},
	})

	metricsMap := map[string]bool{}
	pollingFunction := checkForAcceleratorMetrics(projectID, gcmService, time.Now(), metricsMap)
	err = wait.Poll(pollFrequency, pollTimeout, pollingFunction)
	if err != nil {
		e2elog.Logf("Missing metrics: %+v", metricsMap)
	}
	framework.ExpectNoError(err)
}

func checkForAcceleratorMetrics(projectID string, gcmService *gcm.Service, start time.Time, metricsMap map[string]bool) func() (bool, error) {
	return func() (bool, error) {
		counter := 0
		for _, metric := range acceleratorMetrics {
			metricsMap[metric] = false
		}
		for _, metric := range acceleratorMetrics {
			// TODO: check only for metrics from this cluster
			ts, err := fetchTimeSeries(projectID, gcmService, metric, start, time.Now())
			framework.ExpectNoError(err)
			if len(ts) > 0 {
				counter = counter + 1
				metricsMap[metric] = true
				e2elog.Logf("Received %v timeseries for metric %v", len(ts), metric)
			} else {
				e2elog.Logf("No timeseries for metric %v", metric)
			}
		}
		if counter < 3 {
			return false, nil
		}
		return true, nil
	}
}
