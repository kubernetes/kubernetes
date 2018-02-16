/*
Copyright 2018 The Kubernetes Authors.

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

// testing helpers copied from k8s.io/kube-state-metrics/collectors/deployment_test.go
// TODO: share in a public package?

package collectors

import (
	"bytes"
	"fmt"
	"reflect"
	"sort"
	"strings"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"github.com/prometheus/common/expfmt"
)

// gatherAndCompare retrieves all metrics exposed by a collector and compares it
// to an expected output in the Prometheus text exposition format.
// metricNames allows only comparing the given metrics. All are compared if it's nil.
func gatherAndCompare(c prometheus.Collector, expected string, metricNames []string) error {
	expected = removeUnusedWhitespace(expected)

	reg := prometheus.NewPedanticRegistry()
	if err := reg.Register(c); err != nil {
		return fmt.Errorf("registering collector failed: %s", err)
	}
	metrics, err := reg.Gather()
	if err != nil {
		return fmt.Errorf("gathering metrics failed: %s", err)
	}
	if metricNames != nil {
		metrics = filterMetrics(metrics, metricNames)
	}
	var tp expfmt.TextParser
	expectedMetrics, err := tp.TextToMetricFamilies(bytes.NewReader([]byte(expected)))
	if err != nil {
		return fmt.Errorf("parsing expected metrics failed: %s", err)
	}

	if !reflect.DeepEqual(metrics, normalizeMetricFamilies(expectedMetrics)) {
		// Encode the gathered output to the readbale text format for comparison.
		var buf1 bytes.Buffer
		enc := expfmt.NewEncoder(&buf1, expfmt.FmtText)
		for _, mf := range metrics {
			if err := enc.Encode(mf); err != nil {
				return fmt.Errorf("encoding result failed: %s", err)
			}
		}
		// Encode normalized expected metrics again to generate them in the same ordering
		// the registry does to spot differences more easily.
		var buf2 bytes.Buffer
		enc = expfmt.NewEncoder(&buf2, expfmt.FmtText)
		for _, mf := range normalizeMetricFamilies(expectedMetrics) {
			if err := enc.Encode(mf); err != nil {
				return fmt.Errorf("encoding result failed: %s", err)
			}
		}

		return fmt.Errorf(`metric output does not match expectation; want:
%s

got:

%s`, buf2.String(), buf1.String())
	}
	return nil
}

func filterMetrics(metrics []*dto.MetricFamily, names []string) []*dto.MetricFamily {
	var filtered []*dto.MetricFamily
	for _, m := range metrics {
		drop := true
		for _, name := range names {
			if m.GetName() == name {
				drop = false
				break
			}
		}
		if !drop {
			filtered = append(filtered, m)
		}
	}
	return filtered
}

func removeUnusedWhitespace(s string) string {
	var (
		trimmedLine  string
		trimmedLines []string
		lines        = strings.Split(s, "\n")
	)

	for _, l := range lines {
		trimmedLine = strings.TrimSpace(l)

		if len(trimmedLine) > 0 {
			trimmedLines = append(trimmedLines, trimmedLine)
		}
	}

	// The Prometheus metrics representation parser expects an empty line at the
	// end otherwise fails with an unexpected EOF error.
	return strings.Join(trimmedLines, "\n") + "\n"
}

// The below sorting code is copied form the Prometheus client library modulo the added
// label pair sorting.
// https://github.com/prometheus/client_golang/blob/ea6e1db4cb8127eeb0b6954f7320363e5451820f/prometheus/registry.go#L642-L684

// metricSorter is a sortable slice of *dto.Metric.
type metricSorter []*dto.Metric

func (s metricSorter) Len() int {
	return len(s)
}

func (s metricSorter) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (s metricSorter) Less(i, j int) bool {
	sort.Sort(prometheus.LabelPairSorter(s[i].Label))
	sort.Sort(prometheus.LabelPairSorter(s[j].Label))

	if len(s[i].Label) != len(s[j].Label) {
		return len(s[i].Label) < len(s[j].Label)
	}

	for n, lp := range s[i].Label {
		vi := lp.GetValue()
		vj := s[j].Label[n].GetValue()
		if vi != vj {
			return vi < vj
		}
	}

	if s[i].TimestampMs == nil {
		return false
	}
	if s[j].TimestampMs == nil {
		return true
	}
	return s[i].GetTimestampMs() < s[j].GetTimestampMs()
}

// normalizeMetricFamilies returns a MetricFamily slice with empty
// MetricFamilies pruned and the remaining MetricFamilies sorted by name within
// the slice, with the contained Metrics sorted within each MetricFamily.
func normalizeMetricFamilies(metricFamiliesByName map[string]*dto.MetricFamily) []*dto.MetricFamily {
	for _, mf := range metricFamiliesByName {
		sort.Sort(metricSorter(mf.Metric))
	}
	names := make([]string, 0, len(metricFamiliesByName))
	for name, mf := range metricFamiliesByName {
		if len(mf.Metric) > 0 {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	result := make([]*dto.MetricFamily, 0, len(names))
	for _, name := range names {
		result = append(result, metricFamiliesByName[name])
	}
	return result
}
