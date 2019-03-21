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
	"fmt"
	"math"
	"os"
	"time"

	"golang.org/x/oauth2/google"

	"github.com/onsi/ginkgo"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/common"
	"k8s.io/kubernetes/test/e2e/framework"
	instrumentation "k8s.io/kubernetes/test/e2e/instrumentation/common"

	gcm "google.golang.org/api/monitoring/v3"
)

var (
	// Stackdriver container metrics, as described here:
	// https://cloud.google.com/monitoring/api/metrics#gcp-container
	stackdriverMetrics = []string{
		"uptime",
		"memory/bytes_total",
		"memory/bytes_used",
		"cpu/reserved_cores",
		"cpu/usage_time",
		"memory/page_fault_count",
		"disk/bytes_used",
		"disk/bytes_total",
		"cpu/utilization",
	}

	pollFrequency = time.Second * 5
	pollTimeout   = time.Minute * 7

	rcName            = "resource-consumer"
	memoryUsed        = 64
	memoryLimit int64 = 200
	tolerance         = 0.25
)

var _ = instrumentation.SIGDescribe("Stackdriver Monitoring", func() {
	ginkgo.BeforeEach(func() {
		framework.SkipUnlessProviderIs("gce", "gke")
	})

	f := framework.NewDefaultFramework("stackdriver-monitoring")

	ginkgo.It("should have cluster metrics [Feature:StackdriverMonitoring]", func() {
		testStackdriverMonitoring(f, 1, 100, 200)
	})

})

func testStackdriverMonitoring(f *framework.Framework, pods, allPodsCPU int, perPodCPU int64) {
	projectID := framework.TestContext.CloudConfig.ProjectID

	ctx := context.Background()
	client, err := google.DefaultClient(ctx, gcm.CloudPlatformScope)

	// Hack for running tests locally
	// If this is your use case, create application default credentials:
	// $ gcloud auth application-default login
	// and uncomment following lines (comment out the two lines above): (DON'T set the env var below)
	/*
		ts, err := google.DefaultTokenSource(oauth2.NoContext)
		framework.Logf("Couldn't get application default credentials, %v", err)
		if err != nil {
			framework.Failf("Error accessing application default credentials, %v", err)
		}
		client := oauth2.NewClient(oauth2.NoContext, ts)
	*/

	gcmService, err := gcm.New(client)

	// set this env var if accessing Stackdriver test endpoint (default is prod):
	// $ export STACKDRIVER_API_ENDPOINT_OVERRIDE=https://test-monitoring.sandbox.googleapis.com/
	basePathOverride := os.Getenv("STACKDRIVER_API_ENDPOINT_OVERRIDE")
	if basePathOverride != "" {
		gcmService.BasePath = basePathOverride
	}

	framework.ExpectNoError(err)

	rc := common.NewDynamicResourceConsumer(rcName, f.Namespace.Name, common.KindDeployment, pods, allPodsCPU, memoryUsed, 0, perPodCPU, memoryLimit, f.ClientSet, f.InternalClientset, f.ScalesGetter)
	defer rc.CleanUp()

	rc.WaitForReplicas(pods, 15*time.Minute)

	metricsMap := map[string]bool{}
	pollingFunction := checkForMetrics(projectID, gcmService, time.Now(), metricsMap, allPodsCPU, perPodCPU)
	err = wait.Poll(pollFrequency, pollTimeout, pollingFunction)
	if err != nil {
		framework.Logf("Missing metrics: %+v\n", metricsMap)
	}
	framework.ExpectNoError(err)
}

func checkForMetrics(projectID string, gcmService *gcm.Service, start time.Time, metricsMap map[string]bool, cpuUsed int, cpuLimit int64) func() (bool, error) {
	return func() (bool, error) {
		counter := 0
		correctUtilization := false
		for _, metric := range stackdriverMetrics {
			metricsMap[metric] = false
		}
		for _, metric := range stackdriverMetrics {
			// TODO: check only for metrics from this cluster
			ts, err := fetchTimeSeries(projectID, gcmService, metric, start, time.Now())
			framework.ExpectNoError(err)
			if len(ts) > 0 {
				counter = counter + 1
				metricsMap[metric] = true
				framework.Logf("Received %v timeseries for metric %v\n", len(ts), metric)
			} else {
				framework.Logf("No timeseries for metric %v\n", metric)
			}

			var sum float64
			switch metric {
			case "cpu/utilization":
				for _, t := range ts {
					max := t.Points[0]
					maxEnd, _ := time.Parse(time.RFC3339, max.Interval.EndTime)
					for _, p := range t.Points {
						pEnd, _ := time.Parse(time.RFC3339, p.Interval.EndTime)
						if pEnd.After(maxEnd) {
							max = p
							maxEnd, _ = time.Parse(time.RFC3339, max.Interval.EndTime)
						}
					}
					sum = sum + *max.Value.DoubleValue
					framework.Logf("Received %v points for metric %v\n",
						len(t.Points), metric)
				}
				framework.Logf("Most recent cpu/utilization sum*cpu/limit: %v\n", sum*float64(cpuLimit))
				if math.Abs(sum*float64(cpuLimit)-float64(cpuUsed)) > tolerance*float64(cpuUsed) {
					return false, nil
				}
				correctUtilization = true
			}
		}
		if counter < 9 || !correctUtilization {
			return false, nil
		}
		return true, nil
	}
}

func createMetricFilter(metric string, containerName string) string {
	return fmt.Sprintf(`metric.type="container.googleapis.com/container/%s" AND
				resource.label.container_name="%s"`, metric, containerName)
}

func fetchTimeSeries(projectID string, gcmService *gcm.Service, metric string, start time.Time, end time.Time) ([]*gcm.TimeSeries, error) {
	response, err := gcmService.Projects.TimeSeries.
		List(fullProjectName(projectID)).
		Filter(createMetricFilter(metric, rcName)).
		IntervalStartTime(start.Format(time.RFC3339)).
		IntervalEndTime(end.Format(time.RFC3339)).
		Do()
	if err != nil {
		return nil, err
	}
	return response.TimeSeries, nil
}

func fullProjectName(name string) string {
	return fmt.Sprintf("projects/%s", name)
}
