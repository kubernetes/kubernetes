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

// Package promlint provides a linter for Prometheus metrics.
package promlint

import (
	"fmt"
	"io"
	"regexp"
	"sort"
	"strings"

	"github.com/prometheus/common/expfmt"

	dto "github.com/prometheus/client_model/go"
)

// A Linter is a Prometheus metrics linter.  It identifies issues with metric
// names, types, and metadata, and reports them to the caller.
type Linter struct {
	// The linter will read metrics in the Prometheus text format from r and
	// then lint it, _and_ it will lint the metrics provided directly as
	// MetricFamily proto messages in mfs. Note, however, that the current
	// constructor functions New and NewWithMetricFamilies only ever set one
	// of them.
	r   io.Reader
	mfs []*dto.MetricFamily
}

// A Problem is an issue detected by a Linter.
type Problem struct {
	// The name of the metric indicated by this Problem.
	Metric string

	// A description of the issue for this Problem.
	Text string
}

// newProblem is helper function to create a Problem.
func newProblem(mf *dto.MetricFamily, text string) Problem {
	return Problem{
		Metric: mf.GetName(),
		Text:   text,
	}
}

// New creates a new Linter that reads an input stream of Prometheus metrics in
// the Prometheus text exposition format.
func New(r io.Reader) *Linter {
	return &Linter{
		r: r,
	}
}

// NewWithMetricFamilies creates a new Linter that reads from a slice of
// MetricFamily protobuf messages.
func NewWithMetricFamilies(mfs []*dto.MetricFamily) *Linter {
	return &Linter{
		mfs: mfs,
	}
}

// Lint performs a linting pass, returning a slice of Problems indicating any
// issues found in the metrics stream. The slice is sorted by metric name
// and issue description.
func (l *Linter) Lint() ([]Problem, error) {
	var problems []Problem

	if l.r != nil {
		d := expfmt.NewDecoder(l.r, expfmt.FmtText)

		mf := &dto.MetricFamily{}
		for {
			if err := d.Decode(mf); err != nil {
				if err == io.EOF {
					break
				}

				return nil, err
			}

			problems = append(problems, lint(mf)...)
		}
	}
	for _, mf := range l.mfs {
		problems = append(problems, lint(mf)...)
	}

	// Ensure deterministic output.
	sort.SliceStable(problems, func(i, j int) bool {
		if problems[i].Metric == problems[j].Metric {
			return problems[i].Text < problems[j].Text
		}
		return problems[i].Metric < problems[j].Metric
	})

	return problems, nil
}

// lint is the entry point for linting a single metric.
func lint(mf *dto.MetricFamily) []Problem {
	fns := []func(mf *dto.MetricFamily) []Problem{
		lintHelp,
		lintMetricUnits,
		lintCounter,
		lintHistogramSummaryReserved,
		lintMetricTypeInName,
		lintReservedChars,
		lintCamelCase,
		lintUnitAbbreviations,
	}

	var problems []Problem
	for _, fn := range fns {
		problems = append(problems, fn(mf)...)
	}

	// TODO(mdlayher): lint rules for specific metrics types.
	return problems
}

// lintHelp detects issues related to the help text for a metric.
func lintHelp(mf *dto.MetricFamily) []Problem {
	var problems []Problem

	// Expect all metrics to have help text available.
	if mf.Help == nil {
		problems = append(problems, newProblem(mf, "no help text"))
	}

	return problems
}

// lintMetricUnits detects issues with metric unit names.
func lintMetricUnits(mf *dto.MetricFamily) []Problem {
	var problems []Problem

	unit, base, ok := metricUnits(*mf.Name)
	if !ok {
		// No known units detected.
		return nil
	}

	// Unit is already a base unit.
	if unit == base {
		return nil
	}

	problems = append(problems, newProblem(mf, fmt.Sprintf("use base unit %q instead of %q", base, unit)))

	return problems
}

// lintCounter detects issues specific to counters, as well as patterns that should
// only be used with counters.
func lintCounter(mf *dto.MetricFamily) []Problem {
	var problems []Problem

	isCounter := mf.GetType() == dto.MetricType_COUNTER
	isUntyped := mf.GetType() == dto.MetricType_UNTYPED
	hasTotalSuffix := strings.HasSuffix(mf.GetName(), "_total")

	switch {
	case isCounter && !hasTotalSuffix:
		problems = append(problems, newProblem(mf, `counter metrics should have "_total" suffix`))
	case !isUntyped && !isCounter && hasTotalSuffix:
		problems = append(problems, newProblem(mf, `non-counter metrics should not have "_total" suffix`))
	}

	return problems
}

// lintHistogramSummaryReserved detects when other types of metrics use names or labels
// reserved for use by histograms and/or summaries.
func lintHistogramSummaryReserved(mf *dto.MetricFamily) []Problem {
	// These rules do not apply to untyped metrics.
	t := mf.GetType()
	if t == dto.MetricType_UNTYPED {
		return nil
	}

	var problems []Problem

	isHistogram := t == dto.MetricType_HISTOGRAM
	isSummary := t == dto.MetricType_SUMMARY

	n := mf.GetName()

	if !isHistogram && strings.HasSuffix(n, "_bucket") {
		problems = append(problems, newProblem(mf, `non-histogram metrics should not have "_bucket" suffix`))
	}
	if !isHistogram && !isSummary && strings.HasSuffix(n, "_count") {
		problems = append(problems, newProblem(mf, `non-histogram and non-summary metrics should not have "_count" suffix`))
	}
	if !isHistogram && !isSummary && strings.HasSuffix(n, "_sum") {
		problems = append(problems, newProblem(mf, `non-histogram and non-summary metrics should not have "_sum" suffix`))
	}

	for _, m := range mf.GetMetric() {
		for _, l := range m.GetLabel() {
			ln := l.GetName()

			if !isHistogram && ln == "le" {
				problems = append(problems, newProblem(mf, `non-histogram metrics should not have "le" label`))
			}
			if !isSummary && ln == "quantile" {
				problems = append(problems, newProblem(mf, `non-summary metrics should not have "quantile" label`))
			}
		}
	}

	return problems
}

// lintMetricTypeInName detects when metric types are included in the metric name.
func lintMetricTypeInName(mf *dto.MetricFamily) []Problem {
	var problems []Problem
	n := strings.ToLower(mf.GetName())

	for i, t := range dto.MetricType_name {
		if i == int32(dto.MetricType_UNTYPED) {
			continue
		}

		typename := strings.ToLower(t)
		if strings.Contains(n, "_"+typename+"_") || strings.HasSuffix(n, "_"+typename) {
			problems = append(problems, newProblem(mf, fmt.Sprintf(`metric name should not include type '%s'`, typename)))
		}
	}
	return problems
}

// lintReservedChars detects colons in metric names.
func lintReservedChars(mf *dto.MetricFamily) []Problem {
	var problems []Problem
	if strings.Contains(mf.GetName(), ":") {
		problems = append(problems, newProblem(mf, "metric names should not contain ':'"))
	}
	return problems
}

var camelCase = regexp.MustCompile(`[a-z][A-Z]`)

// lintCamelCase detects metric names and label names written in camelCase.
func lintCamelCase(mf *dto.MetricFamily) []Problem {
	var problems []Problem
	if camelCase.FindString(mf.GetName()) != "" {
		problems = append(problems, newProblem(mf, "metric names should be written in 'snake_case' not 'camelCase'"))
	}

	for _, m := range mf.GetMetric() {
		for _, l := range m.GetLabel() {
			if camelCase.FindString(l.GetName()) != "" {
				problems = append(problems, newProblem(mf, "label names should be written in 'snake_case' not 'camelCase'"))
			}
		}
	}
	return problems
}

// lintUnitAbbreviations detects abbreviated units in the metric name.
func lintUnitAbbreviations(mf *dto.MetricFamily) []Problem {
	var problems []Problem
	n := strings.ToLower(mf.GetName())
	for _, s := range unitAbbreviations {
		if strings.Contains(n, "_"+s+"_") || strings.HasSuffix(n, "_"+s) {
			problems = append(problems, newProblem(mf, "metric names should not contain abbreviated units"))
		}
	}
	return problems
}

// metricUnits attempts to detect known unit types used as part of a metric name,
// e.g. "foo_bytes_total" or "bar_baz_milligrams".
func metricUnits(m string) (unit string, base string, ok bool) {
	ss := strings.Split(m, "_")

	for unit, base := range units {
		// Also check for "no prefix".
		for _, p := range append(unitPrefixes, "") {
			for _, s := range ss {
				// Attempt to explicitly match a known unit with a known prefix,
				// as some words may look like "units" when matching suffix.
				//
				// As an example, "thermometers" should not match "meters", but
				// "kilometers" should.
				if s == p+unit {
					return p + unit, base, true
				}
			}
		}
	}

	return "", "", false
}

// Units and their possible prefixes recognized by this library.  More can be
// added over time as needed.
var (
	// map a unit to the appropriate base unit.
	units = map[string]string{
		// Base units.
		"amperes": "amperes",
		"bytes":   "bytes",
		"celsius": "celsius", // Also allow Celsius because it is common in typical Prometheus use cases.
		"grams":   "grams",
		"joules":  "joules",
		"kelvin":  "kelvin", // SI base unit, used in special cases (e.g. color temperature, scientific measurements).
		"meters":  "meters", // Both American and international spelling permitted.
		"metres":  "metres",
		"seconds": "seconds",
		"volts":   "volts",

		// Non base units.
		// Time.
		"minutes": "seconds",
		"hours":   "seconds",
		"days":    "seconds",
		"weeks":   "seconds",
		// Temperature.
		"kelvins":    "kelvin",
		"fahrenheit": "celsius",
		"rankine":    "celsius",
		// Length.
		"inches": "meters",
		"yards":  "meters",
		"miles":  "meters",
		// Bytes.
		"bits": "bytes",
		// Energy.
		"calories": "joules",
		// Mass.
		"pounds": "grams",
		"ounces": "grams",
	}

	unitPrefixes = []string{
		"pico",
		"nano",
		"micro",
		"milli",
		"centi",
		"deci",
		"deca",
		"hecto",
		"kilo",
		"kibi",
		"mega",
		"mibi",
		"giga",
		"gibi",
		"tera",
		"tebi",
		"peta",
		"pebi",
	}

	// Common abbreviations that we'd like to discourage.
	unitAbbreviations = []string{
		"s",
		"ms",
		"us",
		"ns",
		"sec",
		"b",
		"kb",
		"mb",
		"gb",
		"tb",
		"pb",
		"m",
		"h",
		"d",
	}
)
