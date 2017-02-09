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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"

	gcm "google.golang.org/api/monitoring/v3"
)

var _ = framework.KubeDescribe("Stackdriver metrics", func() {
	f := framework.NewDefaultFramework("monitoring")

	BeforeEach(func() {
		framework.SkipUnlessProviderIs("gke")
	})

	It("should publish metrics to Stackdriver [Feature:Stackdriver][Flaky]", func() {
		framework.Logf("Start creating Stackdriver monitoring test")
		testStackdriverMonitoring(f.ClientSet)
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

func testStackdriverMonitoring(c clientset.Interface) {
	_, err := expectedPods(c)
	time.Sleep(time.Minute * 5)
	projectId := framework.TestContext.CloudConfig.ProjectID
	for _, metric := range stackdriverMetrics {
		_, err = fetchTimeSeries(projectId, metric)
		if err != nil {
			framework.Failf("Error fetching %v, %v", metric, err)
		}
	}
}

func createMetricFilter(metric string) string {
	return fmt.Sprintf("metric.type=\"container.googleapis.com/container/%s\"", metric)
}

func fetchTimeSeries(projectId string, metric string) (*gcm.ListTimeSeriesResponse, error) {
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
	now := time.Now()
	start := now.Add(time.Duration(-300) * time.Second)
	response, err := gcmService.Projects.TimeSeries.
		List(fullProjectName(projectId)).
		Filter(createMetricFilter(metric)).
		IntervalStartTime(start.Format(time.RFC3339)).
		IntervalEndTime(now.Format(time.RFC3339)).
		Do()
	if err != nil {
		return nil, err
	}
	if len(response.TimeSeries) > 0 {
		return response, nil
	}
	return nil, fmt.Errorf("No uptime timeseries found")
}

func fullProjectName(name string) string {
	return fmt.Sprintf("projects/%s", name)
}

func expectedPods(c clientset.Interface) ([]string, error) {
	expectedPods := []string{}
	rcLabels := []string{"heapster"}

	for _, rcLabel := range rcLabels {
		selector := labels.Set{"k8s-app": rcLabel}.AsSelector()
		options := metav1.ListOptions{LabelSelector: selector.String()}
		deploymentList, err := c.Extensions().Deployments(metav1.NamespaceSystem).List(options)
		if err != nil {
			return nil, err
		}
		rcList, err := c.Core().ReplicationControllers(metav1.NamespaceSystem).List(options)
		if err != nil {
			return nil, err
		}
		psList, err := c.Apps().StatefulSets(metav1.NamespaceSystem).List(options)
		if err != nil {
			return nil, err
		}
		if (len(rcList.Items) + len(deploymentList.Items) + len(psList.Items)) != 1 {
			return nil, fmt.Errorf("expected to find one replica for RC or deployment with label %s but got %d",
				rcLabel, len(rcList.Items))
		}

		// Check all the replication controllers.
		for _, rc := range rcList.Items {
			selector := labels.Set(rc.Spec.Selector).AsSelector()
			options := metav1.ListOptions{LabelSelector: selector.String()}
			podList, err := c.Core().Pods(metav1.NamespaceSystem).List(options)
			if err != nil {
				return nil, err
			}
			for _, pod := range podList.Items {
				if pod.DeletionTimestamp != nil {
					continue
				}
				expectedPods = append(expectedPods, string(pod.UID))
			}
		}
		// Do the same for all deployments.
		for _, rc := range deploymentList.Items {
			selector := labels.Set(rc.Spec.Selector.MatchLabels).AsSelector()
			options := metav1.ListOptions{LabelSelector: selector.String()}
			podList, err := c.Core().Pods(metav1.NamespaceSystem).List(options)
			if err != nil {
				return nil, err
			}
			for _, pod := range podList.Items {
				if pod.DeletionTimestamp != nil {
					continue
				}
				expectedPods = append(expectedPods, string(pod.UID))
			}
		}
		// And for pet sets.
		for _, ps := range psList.Items {
			selector := labels.Set(ps.Spec.Selector.MatchLabels).AsSelector()
			options := metav1.ListOptions{LabelSelector: selector.String()}
			podList, err := c.Core().Pods(metav1.NamespaceSystem).List(options)
			if err != nil {
				return nil, err
			}
			for _, pod := range podList.Items {
				if pod.DeletionTimestamp != nil {
					continue
				}
				expectedPods = append(expectedPods, string(pod.UID))
			}
		}

	}
	return expectedPods, nil
}
