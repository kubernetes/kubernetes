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

package e2e

import (
	"fmt"
	"time"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"

	. "github.com/onsi/ginkgo"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"

	gcm "google.golang.org/api/monitoring/v3"
)

var _ = framework.KubeDescribe("Stackdriver metrics", func() {
	BeforeEach(func() {
		framework.SkipUnlessProviderIs("gke")
	})

	It("should publish metrics to Stackdriver [Feature:Stackdriver][Flaky]", func() {
		framework.Logf("Start creating Stackdriver monitoring test")
		testStackdriverMonitoring()
	})
})

var (
	stackdriverMetrics = []string{
		"uptime",
		"cpu/reserved_cores",
		"cpu/usage_time",
		"cpu/utilization",
		"memory/bytes_total",
		"memory/bytes_used",
		"memory/page_fault_count",
		"disk/bytes_used",
		"disk/bytes_total",
	}
)

func testStackdriverMonitoring() {
	projectId := framework.TestContext.CloudConfig.ProjectID
	pollingFunction := checkForMetrics(projectId, time.Now())
	framework.ExpectNoError(wait.Poll(time.Second*5, time.Minute*7, pollingFunction))
}

func checkForMetrics(projectId string, start time.Time) func() (bool, error) {
	return func() (bool, error) {
		for _, metric := range stackdriverMetrics {
			// TODO: check only for metrics from this cluster
			ts, err := fetchTimeSeries(projectId, metric, start)
			if err != nil {
				framework.Failf("Error fetching %v, %v", metric, err)
			}
			if len(ts) < 1 {
				return false, nil
			}
		}
		return true, nil
	}
}

func createMetricFilter(metric string) string {
	return fmt.Sprintf("metric.type=\"container.googleapis.com/container/%s\"", metric)
}

func fetchTimeSeries(projectId string, metric string, start time.Time) ([]*gcm.TimeSeries, error) {
	ts := google.ComputeTokenSource("")

	// Hack for running tests locally
	// If this is your use case, create application default credentials:
	//
	// $ gcloud auth application-default login
	//
	// and uncomment following lines:
	//
	// ts, err := google.DefaultTokenSource(oauth2.NoContext)
	// framework.Logf("Couldn't get application default credentials, %v", err)
	// if err != nil {
	// 	framework.Failf("Error accessing application default credentials, %v", err)
	// }

	client := oauth2.NewClient(oauth2.NoContext, ts)
	gcmService, err := gcm.New(client)
	if err != nil {
		return nil, err
	}
	response, err := gcmService.Projects.TimeSeries.
		List(fullProjectName(projectId)).
		Filter(createMetricFilter(metric)).
		IntervalStartTime(start.Format(time.RFC3339)).
		IntervalEndTime(time.Now().Format(time.RFC3339)).
		Do()
	if err != nil {
		return nil, err
	}

	return response.TimeSeries, nil
}

func fullProjectName(name string) string {
	return fmt.Sprintf("projects/%s", name)
}
