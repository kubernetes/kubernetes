/*
Copyright 2015 The Kubernetes Authors.

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

package benchmark

import (
	"fmt"
	"sort"
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/kubernetes/test/integration/util"
)

// mustSetupScheduler starts the following components:
// - k8s api server (a.k.a. master)
// - scheduler
// It returns clientset and destroyFunc which should be used to
// remove resources after finished.
// Notes on rate limiter:
//   - client rate limit is set to 5000.
func mustSetupScheduler() (util.ShutdownFunc, coreinformers.PodInformer, clientset.Interface) {
	apiURL, apiShutdown := util.StartApiserver()
	clientSet := clientset.NewForConfigOrDie(&restclient.Config{
		Host:          apiURL,
		ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}},
		QPS:           5000.0,
		Burst:         5000,
	})
	_, podInformer, schedulerShutdown := util.StartScheduler(clientSet)

	shutdownFunc := func() {
		schedulerShutdown()
		apiShutdown()
	}

	return shutdownFunc, podInformer, clientSet
}

func getScheduledPods(podInformer coreinformers.PodInformer) ([]*v1.Pod, error) {
	pods, err := podInformer.Lister().List(labels.Everything())
	if err != nil {
		return nil, err
	}

	scheduled := make([]*v1.Pod, 0, len(pods))
	for i := range pods {
		pod := pods[i]
		if len(pod.Spec.NodeName) > 0 {
			scheduled = append(scheduled, pod)
		}
	}
	return scheduled, nil
}

type bucket struct {
	cumulativeCount uint64
	upperBound      float64
}

type histogram struct {
	sampleCount uint64
	sampleSum   float64
	buckets     []bucket
}

func (h *histogram) string() string {
	var lines []string
	var last uint64
	for _, b := range h.buckets {
		lines = append(lines, fmt.Sprintf("%v %v", b.upperBound, b.cumulativeCount-last))
		last = b.cumulativeCount
	}
	return strings.Join(lines, "\n")
}

func (h *histogram) subtract(y *histogram) {
	h.sampleCount -= y.sampleCount
	h.sampleSum -= y.sampleSum
	for i := range h.buckets {
		h.buckets[i].cumulativeCount -= y.buckets[i].cumulativeCount
	}
}

func (h *histogram) deepCopy() *histogram {
	var buckets []bucket
	for _, b := range h.buckets {
		buckets = append(buckets, b)
	}
	return &histogram{
		sampleCount: h.sampleCount,
		sampleSum:   h.sampleSum,
		buckets:     buckets,
	}
}

type histogramSet map[string]map[string]*histogram

func (hs histogramSet) string(metricName string) string {
	sets := make(map[string][]uint64)
	names := []string{}
	labels := []string{}
	hLen := 0
	for name, h := range hs {
		sets[name] = []uint64{}
		names = append(names, name)
		var last uint64
		for _, b := range h[metricName].buckets {
			sets[name] = append(sets[name], b.cumulativeCount-last)
			last = b.cumulativeCount
			if hLen == 0 {
				labels = append(labels, fmt.Sprintf("%v", b.upperBound))
			}
		}

		hLen = len(h[metricName].buckets)
	}

	sort.Strings(names)

	lines := []string{"# " + strings.Join(names, " ")}
	for i := 0; i < hLen; i++ {
		counts := []string{}
		for _, name := range names {
			counts = append(counts, fmt.Sprintf("%v", sets[name][i]))
		}
		lines = append(lines, fmt.Sprintf("%v %v", labels[i], strings.Join(counts, " ")))
	}

	return strings.Join(lines, "\n")
}

var prevHistogram = make(map[string]*histogram)

func collectRelativeMetrics(metrics []string) map[string]*histogram {
	rh := make(map[string]*histogram)
	m, _ := legacyregistry.DefaultGatherer.Gather()
	for _, mFamily := range m {
		if mFamily.Name == nil {
			continue
		}
		if !strings.HasPrefix(*mFamily.Name, "scheduler") {
			continue
		}

		metricFound := false
		for _, metricsName := range metrics {
			if *mFamily.Name == metricsName {
				metricFound = true
				break
			}
		}

		if !metricFound {
			continue
		}

		hist := mFamily.GetMetric()[0].GetHistogram()
		h := &histogram{
			sampleCount: *hist.SampleCount,
			sampleSum:   *hist.SampleSum,
		}

		for _, bckt := range hist.Bucket {
			b := bucket{
				cumulativeCount: *bckt.CumulativeCount,
				upperBound:      *bckt.UpperBound,
			}
			h.buckets = append(h.buckets, b)
		}

		rh[*mFamily.Name] = h.deepCopy()
		if prevHistogram[*mFamily.Name] != nil {
			rh[*mFamily.Name].subtract(prevHistogram[*mFamily.Name])
		}
		prevHistogram[*mFamily.Name] = h

	}

	return rh
}
