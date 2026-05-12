// Copyright 2020 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package validations

import (
	"errors"
	"strings"

	dto "github.com/prometheus/client_model/go"
)

// LintHistogramSummaryReserved detects when other types of metrics use names or labels
// reserved for use by histograms and/or summaries.
func LintHistogramSummaryReserved(mf *dto.MetricFamily) []error {
	// These rules do not apply to untyped metrics.
	t := mf.GetType()
	if t == dto.MetricType_UNTYPED {
		return nil
	}

	var problems []error

	isHistogram := t == dto.MetricType_HISTOGRAM
	isSummary := t == dto.MetricType_SUMMARY

	n := mf.GetName()

	if !isHistogram && strings.HasSuffix(n, "_bucket") {
		problems = append(problems, errors.New(`non-histogram metrics should not have "_bucket" suffix`))
	}
	if !isHistogram && !isSummary && strings.HasSuffix(n, "_count") {
		problems = append(problems, errors.New(`non-histogram and non-summary metrics should not have "_count" suffix`))
	}
	if !isHistogram && !isSummary && strings.HasSuffix(n, "_sum") {
		problems = append(problems, errors.New(`non-histogram and non-summary metrics should not have "_sum" suffix`))
	}

	for _, m := range mf.GetMetric() {
		for _, l := range m.GetLabel() {
			ln := l.GetName()

			if !isHistogram && ln == "le" {
				problems = append(problems, errors.New(`non-histogram metrics should not have "le" label`))
			}
			if !isSummary && ln == "quantile" {
				problems = append(problems, errors.New(`non-summary metrics should not have "quantile" label`))
			}
		}
	}

	return problems
}
