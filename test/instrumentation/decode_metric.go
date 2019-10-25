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

func decodeMetricCalls(fs []*ast.CallExpr, metricsImportName string, variables map[string]ast.Expr) ([]metric, []error) {
	finder := metricDecoder{
		kubeMetricsImportName: metricsImportName,
		variables:             variables,
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
	kubeMetricsImportName string
	variables             map[string]ast.Expr
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
	if functionImport.String() != c.kubeMetricsImportName {
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
		value, err := stringValue(bl)
		if err != nil {
			return nil, err
		}
		labels[i] = value
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
		case "Namespace", "Subsystem", "Name", "Help", "DeprecatedVersion":
			var value string
			var err error
			switch v := kv.Value.(type) {
			case *ast.BasicLit:
				value, err = stringValue(v)
				if err != nil {
					return m, err
				}
			case *ast.Ident:
				variableExpr, found := c.variables[v.Name]
				if !found {
					return m, newDecodeErrorf(expr, errBadVariableAttribute)
				}
				bl, ok := variableExpr.(*ast.BasicLit)
				if !ok {
					return m, newDecodeErrorf(expr, errNonStringAttribute)
				}
				value, err = stringValue(bl)
				if err != nil {
					return m, err
				}
			default:
				return m, newDecodeErrorf(expr, errNonStringAttribute)
			}
			switch key {
			case "Namespace":
				m.Namespace = value
			case "Subsystem":
				m.Subsystem = value
			case "Name":
				m.Name = value
			case "DeprecatedVersion":
				m.DeprecatedVersion = value
			case "Help":
				m.Help = value
			}
		case "Buckets":
			buckets, err := c.decodeBuckets(kv.Value)
			if err != nil {
				return m, err
			}
			sort.Float64s(buckets)
			m.Buckets = buckets
		case "StabilityLevel":
			level, err := decodeStabilityLevel(kv.Value, c.kubeMetricsImportName)
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

func stringValue(bl *ast.BasicLit) (string, error) {
	if bl.Kind != token.STRING {
		return "", newDecodeErrorf(bl, errNonStringAttribute)
	}
	return strings.Trim(bl.Value, `"`), nil
}

func (c *metricDecoder) decodeBuckets(expr ast.Expr) ([]float64, error) {
	switch v := expr.(type) {
	case *ast.CompositeLit:
		return decodeListOfFloats(v.Elts)
	case *ast.SelectorExpr:
		variableName := v.Sel.String()
		importName, ok := v.X.(*ast.Ident)
		if ok && importName.String() == c.kubeMetricsImportName && variableName == "DefBuckets" {
			return metrics.DefBuckets, nil
		}
	case *ast.CallExpr:
		se, ok := v.Fun.(*ast.SelectorExpr)
		if !ok {
			return nil, newDecodeErrorf(v, errBuckets)
		}
		functionName := se.Sel.String()
		functionImport, ok := se.X.(*ast.Ident)
		if !ok {
			return nil, newDecodeErrorf(v, errBuckets)
		}
		if functionImport.String() != c.kubeMetricsImportName {
			return nil, newDecodeErrorf(v, errBuckets)
		}
		firstArg, secondArg, thirdArg, err := decodeBucketArguments(v)
		if err != nil {
			return nil, err
		}
		switch functionName {
		case "LinearBuckets":
			return metrics.LinearBuckets(firstArg, secondArg, thirdArg), nil
		case "ExponentialBuckets":
			return metrics.ExponentialBuckets(firstArg, secondArg, thirdArg), nil
		}
	}
	return nil, newDecodeErrorf(expr, errBuckets)
}

func decodeListOfFloats(exprs []ast.Expr) ([]float64, error) {
	buckets := make([]float64, len(exprs))
	for i, elt := range exprs {
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

func decodeBucketArguments(fc *ast.CallExpr) (float64, float64, int, error) {
	if len(fc.Args) != 3 {
		return 0, 0, 0, newDecodeErrorf(fc, errBuckets)
	}
	strArgs := make([]string, len(fc.Args))
	for i, elt := range fc.Args {
		bl, ok := elt.(*ast.BasicLit)
		if !ok {
			return 0, 0, 0, newDecodeErrorf(bl, errBuckets)
		}
		if bl.Kind != token.FLOAT && bl.Kind != token.INT {
			return 0, 0, 0, newDecodeErrorf(bl, errBuckets)
		}
		strArgs[i] = bl.Value
	}
	firstArg, err := strconv.ParseFloat(strArgs[0], 64)
	if err != nil {
		return 0, 0, 0, newDecodeErrorf(fc.Args[0], errBuckets)
	}
	secondArg, err := strconv.ParseFloat(strArgs[1], 64)
	if err != nil {
		return 0, 0, 0, newDecodeErrorf(fc.Args[1], errBuckets)
	}
	thirdArg, err := strconv.ParseInt(strArgs[2], 10, 64)
	if err != nil {
		return 0, 0, 0, newDecodeErrorf(fc.Args[2], errBuckets)
	}

	return firstArg, secondArg, int(thirdArg), nil
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
