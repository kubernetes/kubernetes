/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"strings"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/metrics"
	"k8s.io/kubernetes/pkg/util/sets"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// Missing = Assumed minus Observed, Invalid = Observed minus Assumed
func validateLabelSet(labelSet map[string][]string, data metrics.Metrics, invalidLabels map[string]sets.String, missingLabels map[string]sets.String) {
	for metric, labels := range labelSet {
		vector, found := data[metric]
		Expect(found).To(Equal(true))
		if found && len(vector) > 0 {
			for _, observation := range vector {
				for label := range observation.Metric {
					// We need to check if it's a known label for this metric.
					// Omit Prometheus internal metrics.
					if strings.HasPrefix(string(label), "__") {
						continue
					}
					invalidLabel := true
					for _, knownLabel := range labels {
						if string(label) == knownLabel {
							invalidLabel = false
						}
					}
					if invalidLabel && invalidLabels != nil {
						if _, ok := invalidLabels[metric]; !ok {
							invalidLabels[metric] = sets.NewString()
						}
						invalidLabels[metric].Insert(string(label))
					}
				}
			}
		}
	}
}

func checkNecessaryMetrics(response metrics.Metrics, necessaryMetrics map[string][]string) {
	missingLabels := make(map[string]sets.String)
	validateLabelSet(metrics.CommonMetrics, response, nil, missingLabels)
	validateLabelSet(necessaryMetrics, response, nil, missingLabels)

	Expect(missingLabels).To(BeEmpty())
}

func checkMetrics(response metrics.Metrics, assumedMetrics map[string][]string) {
	invalidLabels := make(map[string]sets.String)
	missingLabels := make(map[string]sets.String)
	validateLabelSet(metrics.CommonMetrics, response, invalidLabels, missingLabels)
	validateLabelSet(assumedMetrics, response, invalidLabels, missingLabels)

	Expect(missingLabels).To(BeEmpty())
	Expect(invalidLabels).To(BeEmpty())
}

var _ = Describe("MetricsGrabber", func() {
	framework := NewDefaultFramework("metrics-grabber")
	var c *client.Client
	var grabber *metrics.MetricsGrabber
	BeforeEach(func() {
		var err error
		c = framework.Client
		expectNoError(err)
		grabber, err = metrics.NewMetricsGrabber(c, true, true, true, true)
		expectNoError(err)
	})

	It("should grab all metrics from API server.", func() {
		// From @gmarek 9/19/2016 - this test can safely be ignored for upgrade testing
		// TODO(gmarek): Add details about why this can be safely ignored
		// See issue https://github.com/kubernetes/kubernetes/issues/32704
		SkipUnlessServerVersionLT(serverVersion13, c)
		By("Connecting to /metrics endpoint")
		unknownMetrics := sets.NewString()
		response, err := grabber.GrabFromApiServer(unknownMetrics)
		expectNoError(err)
		Expect(unknownMetrics).To(BeEmpty())

		checkMetrics(metrics.Metrics(response), metrics.KnownApiServerMetrics)
	})

	It("should grab all metrics from a Kubelet.", func() {
		By("Proxying to Node through the API server")
		nodes := ListSchedulableNodesOrDie(c)
		Expect(nodes.Items).NotTo(BeEmpty())
		response, err := grabber.GrabFromKubelet(nodes.Items[0].Name)
		expectNoError(err)
		checkNecessaryMetrics(metrics.Metrics(response), metrics.NecessaryKubeletMetrics)
	})

	It("should grab all metrics from a Scheduler.", func() {
		By("Proxying to Pod through the API server")
		// Check if master Node is registered
		nodes, err := c.Nodes().List(api.ListOptions{})
		expectNoError(err)

		var masterRegistered = false
		for _, node := range nodes.Items {
			if strings.HasSuffix(node.Name, "master") {
				masterRegistered = true
			}
		}
		if !masterRegistered {
			Logf("Master is node registered. Skipping testing Scheduler metrics.")
			return
		}
		unknownMetrics := sets.NewString()
		response, err := grabber.GrabFromScheduler(unknownMetrics)
		expectNoError(err)
		Expect(unknownMetrics).To(BeEmpty())

		checkMetrics(metrics.Metrics(response), metrics.KnownSchedulerMetrics)
	})

	It("should grab all metrics from a ControllerManager.", func() {
		By("Proxying to Pod through the API server")
		// Check if master Node is registered
		nodes, err := c.Nodes().List(api.ListOptions{})
		expectNoError(err)

		var masterRegistered = false
		for _, node := range nodes.Items {
			if strings.HasSuffix(node.Name, "master") {
				masterRegistered = true
			}
		}
		if !masterRegistered {
			Logf("Master is node registered. Skipping testing ControllerManager metrics.")
			return
		}
		unknownMetrics := sets.NewString()
		response, err := grabber.GrabFromControllerManager(unknownMetrics)
		expectNoError(err)
		Expect(unknownMetrics).To(BeEmpty())

		checkMetrics(metrics.Metrics(response), metrics.KnownControllerManagerMetrics)
	})
})
