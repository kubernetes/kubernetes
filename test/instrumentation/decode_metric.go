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
	"time"

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
		if m != nil {
			ms = append(ms, *m)
		}
	}
	return ms, errors
}

type metricDecoder struct {
	kubeMetricsImportName string
	variables             map[string]ast.Expr
}

func (c *metricDecoder) decodeNewMetricCall(fc *ast.CallExpr) (*metric, error) {
	var m metric
	var err error
	se, ok := fc.Fun.(*ast.SelectorExpr)
	if !ok {
		// account for timing ratio histogram functions
		switch v := fc.Fun.(type) {
		case *ast.Ident:
			if v.Name == "NewTimingRatioHistogramVec" {
				m, err = c.decodeMetricVecForTimingRatioHistogram(fc)
				m.Type = timingRatioHistogram
				return &m, err
			}
		}
		return nil, newDecodeErrorf(fc, errNotDirectCall)
	}
	functionName := se.Sel.String()
	functionImport, ok := se.X.(*ast.Ident)
	if !ok {
		return nil, newDecodeErrorf(fc, errNotDirectCall)
	}
	if functionImport.String() != c.kubeMetricsImportName {
		return nil, nil
	}
	switch functionName {
	case "NewCounter", "NewGauge", "NewHistogram", "NewSummary", "NewTimingHistogram", "NewGaugeFunc":
		m, err = c.decodeMetric(fc)
	case "NewCounterVec", "NewGaugeVec", "NewHistogramVec", "NewSummaryVec", "NewTimingHistogramVec":
		m, err = c.decodeMetricVec(fc)
	case "Labels", "HandlerOpts", "HandlerFor", "HandlerWithReset":
		return nil, nil
	case "NewDesc":
		m, err = c.decodeDesc(fc)
	default:
		return &m, newDecodeErrorf(fc, errNotDirectCall)
	}
	if err != nil {
		return &m, err
	}
	m.Type = getMetricType(functionName)
	return &m, nil
}

func getMetricType(functionName string) string {
	switch functionName {
	case "NewDesc":
		return customType
	case "NewCounter", "NewCounterVec":
		return counterMetricType
	case "NewGauge", "NewGaugeVec", "NewGaugeFunc":
		return gaugeMetricType
	case "NewHistogram", "NewHistogramVec":
		return histogramMetricType
	case "NewSummary", "NewSummaryVec":
		return summaryMetricType
	case "NewTimingHistogram", "NewTimingHistogramVec", "NewTimingRatioHistogramVec":
		return timingRatioHistogram
	default:
		panic("getMetricType expects correct function name")
	}
}

func (c *metricDecoder) decodeMetric(call *ast.CallExpr) (metric, error) {
	if len(call.Args) > 2 {
		return metric{}, newDecodeErrorf(call, errInvalidNewMetricCall)
	}
	return c.decodeOpts(call.Args[0])
}

func (c *metricDecoder) decodeDesc(ce *ast.CallExpr) (metric, error) {
	m := &metric{}
	name, err := c.decodeString(ce.Args[0])
	if err != nil {
		return *m, newDecodeErrorf(ce, errorDecodingString)
	}
	m.Name = *name
	help, err := c.decodeString(ce.Args[1])
	if err != nil {
		return *m, newDecodeErrorf(ce, errorDecodingString)
	}
	m.Help = *help
	labels, err := c.decodeLabels(ce.Args[2])
	if err != nil {
		return *m, newDecodeErrorf(ce, errorDecodingLabels)
	}
	m.Labels = labels
	cLabels, err := c.decodeConstLabels(ce.Args[3])
	if err != nil {
		return *m, newDecodeErrorf(ce, "can't decode const labels")
	}
	m.ConstLabels = cLabels
	sl, err := decodeStabilityLevel(ce.Args[4], c.kubeMetricsImportName)
	if err != nil {
		return *m, newDecodeErrorf(ce, "can't decode stability level")
	}
	if sl != nil {
		m.StabilityLevel = string(*sl)
	}
	deprecatedVersion, err := c.decodeString(ce.Args[5])
	if err != nil {
		return *m, newDecodeErrorf(ce, errorDecodingString)
	}
	if deprecatedVersion != nil {
		m.DeprecatedVersion = *deprecatedVersion
	}
	return *m, nil
}

func (c *metricDecoder) decodeString(expr ast.Expr) (*string, error) {
	switch e := expr.(type) {
	case *ast.BasicLit:
		value, err := stringValue(e)
		if err != nil {
			return nil, err
		}
		return &value, nil
	case *ast.CallExpr:
		firstArg, secondArg, thirdArg, err := c.decodeBuildFQNameArguments(e)
		if err != nil {
			return nil, newDecodeErrorf(expr, errNonStringAttribute)
		}
		se, ok := e.Fun.(*ast.SelectorExpr)
		if ok {
			functionName := se.Sel.Name
			switch functionName {
			case "BuildFQName":
				n := metrics.BuildFQName(firstArg, secondArg, thirdArg)
				return &n, nil
			}
		}
	case *ast.Ident:
		variableExpr, found := c.variables[e.Name]
		if !found {
			return nil, newDecodeErrorf(expr, errBadVariableAttribute)
		}
		bl, ok := variableExpr.(*ast.BasicLit)
		if !ok {
			return nil, newDecodeErrorf(expr, errNonStringAttribute)
		}
		value, err := stringValue(bl)
		if err != nil {
			return nil, err
		}
		return &value, nil
	case *ast.SelectorExpr:
		s, ok := e.X.(*ast.Ident)
		if !ok {
			return nil, newDecodeErrorf(expr, errExprNotIdent, e.X)
		}
		variableExpr, found := c.variables[strings.Join([]string{s.Name, e.Sel.Name}, ".")]
		if !found {
			return nil, newDecodeErrorf(expr, errBadImportedVariableAttribute)
		}
		bl, ok := variableExpr.(*ast.BasicLit)
		if !ok {
			return nil, newDecodeErrorf(expr, errNonStringAttribute)
		}
		value, err := stringValue(bl)
		if err != nil {
			return nil, err
		}
		return &value, nil
	case *ast.BinaryExpr:
		var binaryExpr *ast.BinaryExpr
		binaryExpr = e
		var okay bool
		var value string
		okay = true

		for okay {
			yV, okay := binaryExpr.Y.(*ast.BasicLit)
			if !okay {
				return nil, newDecodeErrorf(expr, errNonStringAttribute)
			}
			yVal, err := stringValue(yV)
			if err != nil {
				return nil, newDecodeErrorf(expr, errNonStringAttribute)
			}
			value = fmt.Sprintf("%s%s", yVal, value)
			x, okay := binaryExpr.X.(*ast.BinaryExpr)
			if !okay {
				// should be basicLit
				xV, okay := binaryExpr.X.(*ast.BasicLit)
				if !okay {
					return nil, newDecodeErrorf(expr, errNonStringAttribute)
				}
				xVal, err := stringValue(xV)
				if err != nil {
					return nil, newDecodeErrorf(expr, errNonStringAttribute)
				}
				value = fmt.Sprintf("%s%s", xVal, value)
				break
			}
			binaryExpr = x
		}
		return &value, nil
	}
	return nil, newDecodeErrorf(expr, errorDecodingString)
}

func (c *metricDecoder) decodeMetricVec(call *ast.CallExpr) (metric, error) {
	if len(call.Args) != 2 {
		return metric{}, newDecodeErrorf(call, errInvalidNewMetricCall)
	}
	m, err := c.decodeOpts(call.Args[0])
	if err != nil {
		return m, err
	}
	labels, err := c.decodeLabels(call.Args[1])
	if err != nil {
		return m, err
	}
	sort.Strings(labels)
	m.Labels = labels
	return m, nil
}

func (c *metricDecoder) decodeMetricVecForTimingRatioHistogram(call *ast.CallExpr) (metric, error) {
	m, err := c.decodeOpts(call.Args[0])
	if err != nil {
		return m, err
	}
	labels, err := c.decodeLabelsFromArray(call.Args[1:])
	if err != nil {
		return m, err
	}
	sort.Strings(labels)
	m.Labels = labels
	return m, nil
}

func (c *metricDecoder) decodeLabelsFromArray(exprs []ast.Expr) ([]string, error) {
	retval := []string{}
	for _, e := range exprs {
		v, err := c.decodeString(e)
		if err != nil || v == nil {
			return nil, newDecodeErrorf(e, errNonStringAttribute)
		}
		retval = append(retval, *v)
	}

	return retval, nil
}

func (c *metricDecoder) decodeLabels(expr ast.Expr) ([]string, error) {
	cl, ok := expr.(*ast.CompositeLit)
	if !ok {
		switch e := expr.(type) {
		case *ast.Ident:
			if e.Name == "nil" {
				return []string{}, nil
			}
			variableExpr, found := c.variables[e.Name]
			if !found {
				return nil, newDecodeErrorf(expr, errorFindingVariableForLabels)
			}
			cl2, ok := variableExpr.(*ast.CompositeLit)
			if !ok {
				return nil, newDecodeErrorf(expr, errorFindingVariableForLabels)
			}
			cl = cl2
		}
	}
	return c.decodeLabelsFromArray(cl.Elts)
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
			s, err := c.decodeString(kv.Value)
			if err != nil {
				return m, newDecodeErrorf(expr, err.Error())
			}
			value = *s
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
		case "ConstLabels":
			labels, err := c.decodeConstLabels(kv.Value)
			if err != nil {
				return m, err
			}
			m.ConstLabels = labels
		case "AgeBuckets", "BufCap":
			uintVal, err := c.decodeUint32(kv.Value)
			if err != nil {
				print(key)
				return m, err
			}
			if key == "AgeBuckets" {
				m.AgeBuckets = uintVal
			}
			if key == "BufCap" {
				m.BufCap = uintVal
			}

		case "Objectives":
			obj, err := c.decodeObjectives(kv.Value)
			if err != nil {
				print(key)
				return m, err
			}
			m.Objectives = obj
		case "MaxAge":
			int64Val, err := c.decodeInt64(kv.Value)
			if err != nil {
				return m, err
			}
			m.MaxAge = int64Val
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
	case *ast.Ident:
		variableExpr, found := c.variables[v.Name]
		if !found {
			return nil, newDecodeErrorf(v, "couldn't find variable for bucket")
		}
		switch v2 := variableExpr.(type) {
		case *ast.CompositeLit:
			return decodeListOfFloats(v2, v2.Elts)
		case *ast.CallExpr:
			float64s, err2, done := c.decodeBucketFunctionCall(v2)
			if done {
				return float64s, err2
			}
		default:
			return nil, newDecodeErrorf(v, errorFindingVariableForBuckets)
		}

	case *ast.CompositeLit:
		return decodeListOfFloats(v, v.Elts)
	case *ast.SelectorExpr:
		variableName := v.Sel.String()
		importName, ok := v.X.(*ast.Ident)
		if ok && importName.String() == c.kubeMetricsImportName && variableName == "DefBuckets" {
			return metrics.DefBuckets, nil
		}
	case *ast.CallExpr:
		float64s, err2, done := c.decodeBucketFunctionCall(v)
		if done {
			return float64s, err2
		}
	}
	return nil, newDecodeErrorf(expr, errBuckets)
}

func (c *metricDecoder) decodeBucketFunctionCall(v *ast.CallExpr) ([]float64, error, bool) {
	se, ok := v.Fun.(*ast.SelectorExpr)
	if !ok {
		// support merged
		if ai, ok := v.Fun.(*ast.Ident); ok && ai.Name == "merge" {
			merged := []float64{}
			for _, arg := range v.Args {
				v2, ok := arg.(*ast.CallExpr)
				if !ok {
					return nil, newDecodeErrorf(v2, errBuckets), true
				}
				se, ok = v2.Fun.(*ast.SelectorExpr)
				if ok {
					functionName := se.Sel.String()
					functionImport, ok := se.X.(*ast.Ident)
					if !ok {
						return nil, newDecodeErrorf(v, errBuckets), true
					}
					if functionImport.String() != c.kubeMetricsImportName {
						return nil, newDecodeErrorf(v, errBuckets), true
					}
					firstArg, secondArg, thirdArg, err := decodeBucketArguments(v2)
					if err != nil {
						return nil, newDecodeErrorf(v, errBuckets), true
					}
					switch functionName {
					case "LinearBuckets":
						merged = append(merged, metrics.LinearBuckets(firstArg, secondArg, thirdArg)...)
					case "ExponentialBuckets":
						merged = append(merged, metrics.ExponentialBuckets(firstArg, secondArg, thirdArg)...)
					case "ExponentialBucketsRange":
						merged = append(merged, metrics.ExponentialBucketsRange(firstArg, secondArg, thirdArg)...)
					}
				}
			}
			return merged, nil, true
		}
		return nil, newDecodeErrorf(v, errBuckets), true
	}
	functionName := se.Sel.String()
	functionImport, ok := se.X.(*ast.Ident)
	if !ok {
		return nil, newDecodeErrorf(v, errBuckets), true
	}
	if functionImport.String() != c.kubeMetricsImportName {
		return nil, newDecodeErrorf(v, errBuckets), true
	}
	switch functionName {
	case "LinearBuckets":
		firstArg, secondArg, thirdArg, err := decodeBucketArguments(v)
		if err != nil {
			return nil, err, true
		}
		return metrics.LinearBuckets(firstArg, secondArg, thirdArg), nil, true
	case "ExponentialBuckets":
		firstArg, secondArg, thirdArg, err := decodeBucketArguments(v)
		if err != nil {
			return nil, err, true
		}
		return metrics.ExponentialBuckets(firstArg, secondArg, thirdArg), nil, true
	case "ExponentialBucketsRange":
		firstArg, secondArg, thirdArg, err := decodeBucketArguments(v)
		if err != nil {
			return nil, err, true
		}
		return metrics.ExponentialBucketsRange(firstArg, secondArg, thirdArg), nil, true
	case "MergeBuckets":
		merged := []float64{}
		for _, arg := range v.Args {
			switch argExpr := arg.(type) {
			case *ast.CompositeLit:
				fs, err := decodeListOfFloats(argExpr, argExpr.Elts)
				if err != nil {
					return nil, newDecodeErrorf(v, errBuckets), true
				}
				merged = append(merged, fs...)
			case *ast.CallExpr:
				se, ok = argExpr.Fun.(*ast.SelectorExpr)
				if ok {
					functionName := se.Sel.String()
					functionImport, ok := se.X.(*ast.Ident)
					if !ok {
						return nil, newDecodeErrorf(v, errBuckets), true
					}
					if functionImport.String() != c.kubeMetricsImportName {
						return nil, newDecodeErrorf(v, errBuckets), true
					}
					firstArg, secondArg, thirdArg, err := decodeBucketArguments(argExpr)
					if err != nil {
						return nil, newDecodeErrorf(v, errBuckets), true
					}
					switch functionName {
					case "LinearBuckets":
						merged = append(merged, metrics.LinearBuckets(firstArg, secondArg, thirdArg)...)
					case "ExponentialBuckets":
						merged = append(merged, metrics.LinearBuckets(firstArg, secondArg, thirdArg)...)
					}
				}
			}
		}
		return merged, nil, true
	}
	return nil, nil, false
}

func (c *metricDecoder) decodeObjectives(expr ast.Expr) (map[float64]float64, error) {
	switch v := expr.(type) {
	case *ast.CompositeLit:
		return decodeFloatMap(v.Elts)
	case *ast.Ident:
		variableExpr, found := c.variables[v.Name]
		if !found {
			return nil, newDecodeErrorf(expr, errBadVariableAttribute)
		}
		return decodeFloatMap(variableExpr.(*ast.CompositeLit).Elts)
	}
	return nil, newDecodeErrorf(expr, errObjectives)
}

func (c *metricDecoder) decodeUint32(expr ast.Expr) (uint32, error) {
	switch v := expr.(type) {
	case *ast.BasicLit:
		if v.Kind != token.FLOAT && v.Kind != token.INT {
			print(v.Kind)
		}
		value, err := strconv.ParseUint(v.Value, 10, 32)
		if err != nil {
			return 0, err
		}
		return uint32(value), nil
	case *ast.SelectorExpr:
		variableName := v.Sel.String()
		importName, ok := v.X.(*ast.Ident)
		if ok && importName.String() == c.kubeMetricsImportName {
			if variableName == "DefAgeBuckets" {
				// hardcode this for now
				return metrics.DefAgeBuckets, nil
			}
			if variableName == "DefBufCap" {
				// hardcode this for now
				return metrics.DefBufCap, nil
			}
		}
	case *ast.CallExpr:
		_, ok := v.Fun.(*ast.SelectorExpr)
		if !ok {
			return 0, newDecodeErrorf(v, errDecodeUint32)
		}
		return 0, nil
	}
	return 0, newDecodeErrorf(expr, errDecodeUint32)
}

func (c *metricDecoder) decodeInt64(expr ast.Expr) (int64, error) {
	switch v := expr.(type) {
	case *ast.BasicLit:
		if v.Kind != token.FLOAT && v.Kind != token.INT {
			print(v.Kind)
		}

		value, err := strconv.ParseInt(v.Value, 10, 64)
		if err != nil {
			return 0, err
		}
		return value, nil
	case *ast.SelectorExpr:
		variableName := v.Sel.String()
		importName, ok := v.X.(*ast.Ident)
		if ok && importName.String() == c.kubeMetricsImportName {
			if variableName == "DefMaxAge" {
				// hardcode this for now. This is a duration, but we'll output it as
				// an int64 representing nanoseconds.
				return int64(metrics.DefMaxAge), nil
			}
		}
	case *ast.Ident:
		variableExpr, found := c.variables[v.Name]
		if found {
			be, ok := variableExpr.(*ast.BinaryExpr)
			if ok {
				i, err2, done := c.extractTimeExpression(be)
				if done {
					return i, err2
				}
			}
		}
	case *ast.CallExpr:
		_, ok := v.Fun.(*ast.SelectorExpr)
		if !ok {
			return 0, newDecodeErrorf(v, errDecodeInt64)
		}
		return 0, nil
	case *ast.BinaryExpr:
		i, err2, done := c.extractTimeExpression(v)
		if done {
			return i, err2
		}
	}
	return 0, newDecodeErrorf(expr, errDecodeInt64)
}

func (c *metricDecoder) extractTimeExpression(v *ast.BinaryExpr) (int64, error, bool) {
	x := v.X.(*ast.BasicLit)
	if x.Kind != token.FLOAT && x.Kind != token.INT {
		print(x.Kind)
	}

	xValue, err := strconv.ParseInt(x.Value, 10, 64)
	if err != nil {
		return 0, err, true
	}

	switch y := v.Y.(type) {
	case *ast.SelectorExpr:
		variableName := y.Sel.String()
		importName, ok := y.X.(*ast.Ident)
		if ok && importName.String() == "time" {
			if variableName == "Hour" {
				return xValue * int64(time.Hour), nil, true
			}
			if variableName == "Minute" {
				return xValue * int64(time.Minute), nil, true
			}
			if variableName == "Second" {
				return xValue * int64(time.Second), nil, true
			}
		}
	}
	return 0, nil, false
}

func decodeFloatMap(exprs []ast.Expr) (map[float64]float64, error) {
	buckets := map[float64]float64{}
	for _, elt := range exprs {
		bl, ok := elt.(*ast.KeyValueExpr)
		if !ok {
			return nil, newDecodeErrorf(bl, errObjectives)
		}
		keyExpr, ok := bl.Key.(*ast.BasicLit)
		if !ok {
			return nil, newDecodeErrorf(bl, errObjectives)
		}
		valueExpr, ok := bl.Value.(*ast.BasicLit)
		if !ok {
			return nil, newDecodeErrorf(bl, errObjectives)
		}
		valueForKey, err := strconv.ParseFloat(keyExpr.Value, 64)
		if err != nil {
			return nil, newDecodeErrorf(bl, errObjectives)
		}
		valueForValue, err := strconv.ParseFloat(valueExpr.Value, 64)
		if err != nil {
			return nil, newDecodeErrorf(bl, errObjectives)
		}
		buckets[valueForKey] = valueForValue
	}
	return buckets, nil
}

func decodeListOfFloats(expr ast.Expr, exprs []ast.Expr) ([]float64, error) {
	buckets := make([]float64, len(exprs))
	for i, elt := range exprs {
		bl, ok := elt.(*ast.BasicLit)
		if !ok {
			return nil, newDecodeErrorf(expr, errBuckets)
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
func (c *metricDecoder) decodeBuildFQNameArguments(fc *ast.CallExpr) (string, string, string, error) {
	if len(fc.Args) != 3 {
		return "", "", "", newDecodeErrorf(fc, "can't decode fq name args")
	}
	strArgs := make([]string, len(fc.Args))
	for i, elt := range fc.Args {
		s, err := c.decodeString(elt)
		if err != nil || s == nil {
			return "", "", "", newDecodeErrorf(fc, err.Error())
		}
		strArgs[i] = *s
	}
	return strArgs[0], strArgs[1], strArgs[2], nil
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

	stability := metrics.StabilityLevel(se.Sel.Name)
	return &stability, nil
}

func (c *metricDecoder) decodeConstLabels(expr ast.Expr) (map[string]string, error) {
	retval := map[string]string{}
	switch v := expr.(type) {
	case *ast.CompositeLit:
		for _, e2 := range v.Elts {
			kv := e2.(*ast.KeyValueExpr)
			key := ""
			switch k := kv.Key.(type) {

			case *ast.Ident:
				variableExpr, found := c.variables[k.Name]
				if !found {
					return nil, newDecodeErrorf(expr, errBadVariableAttribute)
				}
				bl, ok := variableExpr.(*ast.BasicLit)
				if !ok {
					return nil, newDecodeErrorf(expr, errNonStringAttribute)
				}
				k2, err := stringValue(bl)
				if err != nil {
					return nil, err
				}
				key = k2
			case *ast.BasicLit:
				k2, err := stringValue(k)
				if err != nil {
					return nil, err
				}
				key = k2
			}
			val, err := stringValue(kv.Value.(*ast.BasicLit))
			if err != nil {
				return nil, err
			}
			retval[key] = val
		}
	}
	return retval, nil
}
