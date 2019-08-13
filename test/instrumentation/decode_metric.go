/*
Copyright 2019 The Kubernetes Authors.

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

package main

import (
	"fmt"
	"go/ast"
	"go/token"
	"sort"
	"strconv"
	"strings"

	"k8s.io/component-base/metrics"
)

func decodeMetricCalls(fs []*ast.CallExpr, metricsImportName string) ([]metric, []error) {
	finder := metricDecoder{
		metricsImportName: metricsImportName,
	}
	ms := make([]metric, 0, len(fs))
	errors := []error{}
	for _, f := range fs {
		m, err := finder.decodeNewMetricCall(f)
		if err != nil {
			errors = append(errors, err)
			continue
		}
		ms = append(ms, m)
	}
	return ms, errors
}

type metricDecoder struct {
	metricsImportName string
}

func (c *metricDecoder) decodeNewMetricCall(fc *ast.CallExpr) (metric, error) {
	var m metric
	var err error
	se, ok := fc.Fun.(*ast.SelectorExpr)
	if !ok {
		return m, newDecodeErrorf(fc, errNotDirectCall)
	}
	functionName := se.Sel.String()
	functionImport, ok := se.X.(*ast.Ident)
	if !ok {
		return m, newDecodeErrorf(fc, errNotDirectCall)
	}
	if functionImport.String() != c.metricsImportName {
		return m, newDecodeErrorf(fc, errNotDirectCall)
	}
	switch functionName {
	case "NewCounter", "NewGauge", "NewHistogram":
		m, err = c.decodeMetric(fc)
	case "NewCounterVec", "NewGaugeVec", "NewHistogramVec":
		m, err = c.decodeMetricVec(fc)
	case "NewSummary", "NewSummaryVec":
		return m, newDecodeErrorf(fc, errStableSummary)
	default:
		return m, newDecodeErrorf(fc, errNotDirectCall)
	}
	if err != nil {
		return m, err
	}
	m.Type = getMetricType(functionName)
	return m, nil
}

func getMetricType(functionName string) string {
	switch functionName {
	case "NewCounter", "NewCounterVec":
		return counterMetricType
	case "NewGauge", "NewGaugeVec":
		return gaugeMetricType
	case "NewHistogram", "NewHistogramVec":
		return histogramMetricType
	default:
		panic("getMetricType expects correct function name")
	}
}

func (c *metricDecoder) decodeMetric(call *ast.CallExpr) (metric, error) {
	if len(call.Args) != 1 {
		return metric{}, newDecodeErrorf(call, errInvalidNewMetricCall)
	}
	return c.decodeOpts(call.Args[0])
}

func (c *metricDecoder) decodeMetricVec(call *ast.CallExpr) (metric, error) {
	if len(call.Args) != 2 {
		return metric{}, newDecodeErrorf(call, errInvalidNewMetricCall)
	}
	m, err := c.decodeOpts(call.Args[0])
	if err != nil {
		return m, err
	}
	labels, err := decodeLabels(call.Args[1])
	if err != nil {
		return m, err
	}
	sort.Strings(labels)
	m.Labels = labels
	return m, nil
}

func decodeLabels(expr ast.Expr) ([]string, error) {
	cl, ok := expr.(*ast.CompositeLit)
	if !ok {
		return nil, newDecodeErrorf(expr, errInvalidNewMetricCall)
	}
	labels := make([]string, len(cl.Elts))
	for i, el := range cl.Elts {
		bl, ok := el.(*ast.BasicLit)
		if !ok {
			return nil, newDecodeErrorf(bl, errLabels)
		}
		if bl.Kind != token.STRING {
			return nil, newDecodeErrorf(bl, errLabels)
		}
		labels[i] = strings.Trim(bl.Value, `"`)
	}
	return labels, nil
}

func (c *metricDecoder) decodeOpts(expr ast.Expr) (metric, error) {
	m := metric{
		Labels: []string{},
	}
	ue, ok := expr.(*ast.UnaryExpr)
	if !ok {
		return m, newDecodeErrorf(expr, errInvalidNewMetricCall)
	}
	cl, ok := ue.X.(*ast.CompositeLit)
	if !ok {
		return m, newDecodeErrorf(expr, errInvalidNewMetricCall)
	}

	for _, expr := range cl.Elts {
		kv, ok := expr.(*ast.KeyValueExpr)
		if !ok {
			return m, newDecodeErrorf(expr, errPositionalArguments)
		}
		key := fmt.Sprintf("%v", kv.Key)

		switch key {
		case "Namespace", "Subsystem", "Name", "Help":
			k, ok := kv.Value.(*ast.BasicLit)
			if !ok {
				return m, newDecodeErrorf(expr, errNonStringAttribute)
			}
			if k.Kind != token.STRING {
				return m, newDecodeErrorf(expr, errNonStringAttribute)
			}
			value := strings.Trim(k.Value, `"`)
			switch key {
			case "Namespace":
				m.Namespace = value
			case "Subsystem":
				m.Subsystem = value
			case "Name":
				m.Name = value
			case "Help":
				m.Help = value
			}
		case "Buckets":
			buckets, err := decodeBuckets(kv)
			if err != nil {
				return m, err
			}
			sort.Float64s(buckets)
			m.Buckets = buckets
		case "StabilityLevel":
			level, err := decodeStabilityLevel(kv.Value, c.metricsImportName)
			if err != nil {
				return m, err
			}
			m.StabilityLevel = string(*level)
		default:
			return m, newDecodeErrorf(expr, errFieldNotSupported, key)
		}
	}
	return m, nil
}

func decodeBuckets(kv *ast.KeyValueExpr) ([]float64, error) {
	cl, ok := kv.Value.(*ast.CompositeLit)
	if !ok {
		return nil, newDecodeErrorf(kv, errBuckets)
	}
	buckets := make([]float64, len(cl.Elts))
	for i, elt := range cl.Elts {
		bl, ok := elt.(*ast.BasicLit)
		if !ok {
			return nil, newDecodeErrorf(bl, errBuckets)
		}
		if bl.Kind != token.FLOAT && bl.Kind != token.INT {
			return nil, newDecodeErrorf(bl, errBuckets)
		}
		value, err := strconv.ParseFloat(bl.Value, 64)
		if err != nil {
			return nil, err
		}
		buckets[i] = value
	}
	return buckets, nil
}

func decodeStabilityLevel(expr ast.Expr, metricsFrameworkImportName string) (*metrics.StabilityLevel, error) {
	se, ok := expr.(*ast.SelectorExpr)
	if !ok {
		return nil, newDecodeErrorf(expr, errStabilityLevel)
	}
	s, ok := se.X.(*ast.Ident)
	if !ok {
		return nil, newDecodeErrorf(expr, errStabilityLevel)
	}
	if s.String() != metricsFrameworkImportName {
		return nil, newDecodeErrorf(expr, errStabilityLevel)
	}
	if se.Sel.Name != "ALPHA" && se.Sel.Name != "STABLE" {
		return nil, newDecodeErrorf(expr, errStabilityLevel)
	}
	stability := metrics.StabilityLevel(se.Sel.Name)
	return &stability, nil
}
