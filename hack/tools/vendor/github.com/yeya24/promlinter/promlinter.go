package promlinter

import (
	"fmt"
	"go/ast"
	"go/token"
	"sort"
	"strconv"
	"strings"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil/promlint"
	dto "github.com/prometheus/client_model/go"
)

var (
	metricsType       map[string]dto.MetricType
	constMetricArgNum map[string]int
	validOptsFields   map[string]bool
	lintFuncText      map[string][]string
	LintFuncNames     []string
)

func init() {
	metricsType = map[string]dto.MetricType{
		"Counter":         dto.MetricType_COUNTER,
		"NewCounter":      dto.MetricType_COUNTER,
		"NewCounterVec":   dto.MetricType_COUNTER,
		"Gauge":           dto.MetricType_GAUGE,
		"NewGauge":        dto.MetricType_GAUGE,
		"NewGaugeVec":     dto.MetricType_GAUGE,
		"NewHistogram":    dto.MetricType_HISTOGRAM,
		"NewHistogramVec": dto.MetricType_HISTOGRAM,
		"NewSummary":      dto.MetricType_SUMMARY,
		"NewSummaryVec":   dto.MetricType_SUMMARY,
	}

	constMetricArgNum = map[string]int{
		"MustNewConstMetric": 3,
		"MustNewHistogram":   4,
		"MustNewSummary":     4,
		"NewLazyConstMetric": 3,
	}

	// Doesn't contain ConstLabels since we don't need this field here.
	validOptsFields = map[string]bool{
		"Name":      true,
		"Namespace": true,
		"Subsystem": true,
		"Help":      true,
	}

	lintFuncText = map[string][]string{
		"Help":                     {"no help text"},
		"MetricUnits":              {"use base unit"},
		"Counter":                  {"counter metrics should"},
		"HistogramSummaryReserved": {"non-histogram", "non-summary"},
		"MetricTypeInName":         {"metric name should not include type"},
		"ReservedChars":            {"metric names should not contain ':'"},
		"CamelCase":                {"'snake_case' not 'camelCase'"},
		"lintUnitAbbreviations":    {"metric names should not contain abbreviated units"},
	}

	LintFuncNames = []string{"Help", "MetricUnits", "Counter", "HistogramSummaryReserved",
		"MetricTypeInName", "ReservedChars", "CamelCase", "lintUnitAbbreviations"}
}

type Setting struct {
	Strict            bool
	DisabledLintFuncs []string
}

// Issue contains metric name, error text and metric position.
type Issue struct {
	Text   string
	Metric string
	Pos    token.Position
}

type MetricFamilyWithPos struct {
	MetricFamily *dto.MetricFamily
	Pos          token.Position
}

type visitor struct {
	fs      *token.FileSet
	metrics []MetricFamilyWithPos
	issues  []Issue
	strict  bool
}

type opt struct {
	namespace string
	subsystem string
	name      string
}

func RunList(fs *token.FileSet, files []*ast.File, strict bool) []MetricFamilyWithPos {
	v := &visitor{
		fs:      fs,
		metrics: make([]MetricFamilyWithPos, 0),
		issues:  make([]Issue, 0),
		strict:  strict,
	}

	for _, file := range files {
		ast.Walk(v, file)
	}

	sort.Slice(v.metrics, func(i, j int) bool {
		return v.metrics[i].Pos.String() < v.metrics[j].Pos.String()
	})
	return v.metrics
}

func RunLint(fs *token.FileSet, files []*ast.File, s Setting) []Issue {
	v := &visitor{
		fs:      fs,
		metrics: make([]MetricFamilyWithPos, 0),
		issues:  make([]Issue, 0),
		strict:  s.Strict,
	}

	for _, file := range files {
		ast.Walk(v, file)
	}

	// lint metrics
	for _, mfp := range v.metrics {
		problems, err := promlint.NewWithMetricFamilies([]*dto.MetricFamily{mfp.MetricFamily}).Lint()
		if err != nil {
			panic(err)
		}

		for _, p := range problems {
			for _, disabledFunc := range s.DisabledLintFuncs {
				for _, pattern := range lintFuncText[disabledFunc] {
					if strings.Contains(p.Text, pattern) {
						goto END
					}
				}
			}

			v.issues = append(v.issues, Issue{
				Pos:    mfp.Pos,
				Metric: p.Metric,
				Text:   p.Text,
			})

		END:
		}
	}

	sort.Slice(v.issues, func(i, j int) bool {
		return v.issues[i].Pos.String() < v.issues[j].Pos.String()
	})
	return v.issues
}

func (v *visitor) Visit(n ast.Node) ast.Visitor {
	if n == nil {
		return v
	}

	switch t := n.(type) {
	case *ast.CallExpr:
		return v.parseCallerExpr(t)

	case *ast.SendStmt:
		return v.parseSendMetricChanExpr(t)
	}

	return v
}

func (v *visitor) parseCallerExpr(call *ast.CallExpr) ast.Visitor {
	var (
		metricType dto.MetricType
		methodName string
		ok         bool
	)

	switch stmt := call.Fun.(type) {

	/*
		That's the case of setting alias . to client_golang/prometheus or promauto package.

			import . "github.com/prometheus/client_golang/prometheus"
			metric := NewCounter(CounterOpts{})
	*/
	case *ast.Ident:
		if stmt.Name == "NewCounterFunc" {
			return v.parseOpts(call.Args[0], dto.MetricType_COUNTER)
		}

		if stmt.Name == "NewGaugeFunc" {
			return v.parseOpts(call.Args[0], dto.MetricType_GAUGE)
		}

		if metricType, ok = metricsType[stmt.Name]; !ok {
			return v
		}
		methodName = stmt.Name

	/*
		This case covers the most of cases to initialize metrics.

			prometheus.NewCounter(CounterOpts{})

			promauto.With(nil).NewCounter(CounterOpts{})

			factory := promauto.With(nil)
			factory.NewCounter(CounterOpts{})

			prometheus.NewCounterFunc()
	*/
	case *ast.SelectorExpr:
		if stmt.Sel.Name == "NewCounterFunc" {
			return v.parseOpts(call.Args[0], dto.MetricType_COUNTER)
		}

		if stmt.Sel.Name == "NewGaugeFunc" {
			return v.parseOpts(call.Args[0], dto.MetricType_GAUGE)
		}

		if stmt.Sel.Name == "NewFamilyGenerator" && len(call.Args) == 5 {
			return v.parseKSMMetrics(call.Args[0], call.Args[1], call.Args[2])
		}

		if metricType, ok = metricsType[stmt.Sel.Name]; !ok {
			return v
		}
		methodName = stmt.Sel.Name

	default:
		return v
	}

	argNum := 1
	if strings.HasSuffix(methodName, "Vec") {
		argNum = 2
	}
	// The methods used to initialize metrics should have at least one arg.
	if len(call.Args) < argNum && v.strict {
		v.issues = append(v.issues, Issue{
			Pos:    v.fs.Position(call.Pos()),
			Metric: "",
			Text:   fmt.Sprintf("%s should have at least %d arguments", methodName, argNum),
		})
		return v
	}

	if len(call.Args) == 0 {
		return v
	}

	return v.parseOpts(call.Args[0], metricType)
}

func (v *visitor) parseOpts(optArg ast.Node, metricType dto.MetricType) ast.Visitor {
	// position for the first arg of the CallExpr
	optsPosition := v.fs.Position(optArg.Pos())
	opts, help := v.parseOptsExpr(optArg)
	if opts == nil {
		return v
	}
	currentMetric := dto.MetricFamily{
		Type: &metricType,
		Help: help,
	}

	metricName := prometheus.BuildFQName(opts.namespace, opts.subsystem, opts.name)
	// We skip the invalid metric if the name is an empty string.
	// This kind of metric declaration might be used as a stud metric
	// https://github.com/thanos-io/thanos/blob/main/cmd/thanos/tools_bucket.go#L538.
	if metricName == "" {
		return v
	}
	currentMetric.Name = &metricName

	v.metrics = append(v.metrics, MetricFamilyWithPos{MetricFamily: &currentMetric, Pos: optsPosition})
	return v
}

// Parser for kube-state-metrics generators.
func (v *visitor) parseKSMMetrics(nameArg ast.Node, helpArg ast.Node, metricTypeArg ast.Node) ast.Visitor {
	optsPosition := v.fs.Position(nameArg.Pos())
	currentMetric := dto.MetricFamily{}
	name, ok := v.parseValue("name", nameArg)
	if !ok {
		return v
	}
	currentMetric.Name = &name

	help, ok := v.parseValue("help", helpArg)
	if !ok {
		return v
	}
	currentMetric.Help = &help

	switch stmt := metricTypeArg.(type) {
	case *ast.SelectorExpr:
		if metricType, ok := metricsType[stmt.Sel.Name]; !ok {
			return v
		} else {
			currentMetric.Type = &metricType
		}
	}

	v.metrics = append(v.metrics, MetricFamilyWithPos{MetricFamily: &currentMetric, Pos: optsPosition})
	return v
}

func (v *visitor) parseSendMetricChanExpr(chExpr *ast.SendStmt) ast.Visitor {
	var (
		ok             bool
		requiredArgNum int
		methodName     string
		metricType     dto.MetricType
	)

	call, ok := chExpr.Value.(*ast.CallExpr)
	if !ok {
		return v
	}

	switch stmt := call.Fun.(type) {
	case *ast.Ident:
		if requiredArgNum, ok = constMetricArgNum[stmt.Name]; !ok {
			return v
		}
		methodName = stmt.Name

	case *ast.SelectorExpr:
		if requiredArgNum, ok = constMetricArgNum[stmt.Sel.Name]; !ok {
			return v
		}
		methodName = stmt.Sel.Name
	}

	if len(call.Args) < requiredArgNum && v.strict {
		v.issues = append(v.issues, Issue{
			Metric: "",
			Pos:    v.fs.Position(call.Pos()),
			Text:   fmt.Sprintf("%s should have at least %d arguments", methodName, requiredArgNum),
		})
		return v
	}

	name, help := v.parseConstMetricOptsExpr(call.Args[0])
	if name == nil {
		return v
	}

	metric := &dto.MetricFamily{
		Name: name,
		Help: help,
	}
	switch methodName {
	case "MustNewConstMetric", "NewLazyConstMetric":
		switch t := call.Args[1].(type) {
		case *ast.Ident:
			metric.Type = getConstMetricType(t.Name)
		case *ast.SelectorExpr:
			metric.Type = getConstMetricType(t.Sel.Name)
		}

	case "MustNewHistogram":
		metricType = dto.MetricType_HISTOGRAM
		metric.Type = &metricType
	case "MustNewSummary":
		metricType = dto.MetricType_SUMMARY
		metric.Type = &metricType
	}

	v.metrics = append(v.metrics, MetricFamilyWithPos{MetricFamily: metric, Pos: v.fs.Position(call.Pos())})
	return v
}

func (v *visitor) parseOptsExpr(n ast.Node) (*opt, *string) {
	switch stmt := n.(type) {
	case *ast.CompositeLit:
		return v.parseCompositeOpts(stmt)

	case *ast.Ident:
		if stmt.Obj != nil {
			if decl, ok := stmt.Obj.Decl.(*ast.AssignStmt); ok && len(decl.Rhs) > 0 {
				if t, ok := decl.Rhs[0].(*ast.CompositeLit); ok {
					return v.parseCompositeOpts(t)
				}
			}
		}

	case *ast.UnaryExpr:
		return v.parseOptsExpr(stmt.X)
	}

	return nil, nil
}

func (v *visitor) parseCompositeOpts(stmt *ast.CompositeLit) (*opt, *string) {
	metricOption := &opt{}
	var help *string
	for _, elt := range stmt.Elts {
		kvExpr, ok := elt.(*ast.KeyValueExpr)
		if !ok {
			continue
		}
		object, ok := kvExpr.Key.(*ast.Ident)
		if !ok {
			continue
		}

		if _, ok := validOptsFields[object.Name]; !ok {
			continue
		}

		// If failed to parse field value, stop parsing.
		stringLiteral, ok := v.parseValue(object.Name, kvExpr.Value)
		if !ok {
			return nil, nil
		}

		switch object.Name {
		case "Namespace":
			metricOption.namespace = stringLiteral
		case "Subsystem":
			metricOption.subsystem = stringLiteral
		case "Name":
			metricOption.name = stringLiteral
		case "Help":
			help = &stringLiteral
		}
	}

	return metricOption, help
}

func (v *visitor) parseValue(object string, n ast.Node) (string, bool) {
	switch t := n.(type) {

	// make sure it is string literal value
	case *ast.BasicLit:
		if t.Kind == token.STRING {
			return mustUnquote(t.Value), true
		}

		return "", false

	case *ast.Ident:
		if t.Obj == nil {
			return "", false
		}

		if vs, ok := t.Obj.Decl.(*ast.ValueSpec); ok {
			return v.parseValue(object, vs)
		}

	case *ast.ValueSpec:
		if len(t.Values) == 0 {
			return "", false
		}
		return v.parseValue(object, t.Values[0])

	// For binary expr, we only support adding two strings like `foo` + `bar`.
	case *ast.BinaryExpr:
		if t.Op == token.ADD {
			x, ok := v.parseValue(object, t.X)
			if !ok {
				return "", false
			}

			y, ok := v.parseValue(object, t.Y)
			if !ok {
				return "", false
			}

			return x + y, true
		}

	// We can only cover some basic cases here
	case *ast.CallExpr:
		return v.parseValueCallExpr(object, t)

	default:
		if v.strict {
			v.issues = append(v.issues, Issue{
				Pos:    v.fs.Position(n.Pos()),
				Metric: "",
				Text:   fmt.Sprintf("parsing %s with type %T is not supported", object, t),
			})
		}
	}

	return "", false
}

func (v *visitor) parseValueCallExpr(object string, call *ast.CallExpr) (string, bool) {
	var (
		methodName string
		namespace  string
		subsystem  string
		name       string
		ok         bool
	)
	switch expr := call.Fun.(type) {
	case *ast.SelectorExpr:
		methodName = expr.Sel.Name
	case *ast.Ident:
		methodName = expr.Name
	default:
		return "", false
	}

	if methodName == "BuildFQName" && len(call.Args) == 3 {
		namespace, ok = v.parseValue("namespace", call.Args[0])
		if !ok {
			return "", false
		}
		subsystem, ok = v.parseValue("subsystem", call.Args[1])
		if !ok {
			return "", false
		}
		name, ok = v.parseValue("name", call.Args[2])
		if !ok {
			return "", false
		}
		return prometheus.BuildFQName(namespace, subsystem, name), true
	}

	if v.strict {
		v.issues = append(v.issues, Issue{
			Metric: "",
			Pos:    v.fs.Position(call.Pos()),
			Text:   fmt.Sprintf("parsing %s with function %s is not supported", object, methodName),
		})
	}

	return "", false
}

func (v *visitor) parseConstMetricOptsExpr(n ast.Node) (*string, *string) {
	switch stmt := n.(type) {
	case *ast.CallExpr:
		return v.parseNewDescCallExpr(stmt)

	case *ast.Ident:
		if stmt.Obj != nil {
			switch t := stmt.Obj.Decl.(type) {
			case *ast.AssignStmt:
				if len(t.Rhs) > 0 {
					if call, ok := t.Rhs[0].(*ast.CallExpr); ok {
						return v.parseNewDescCallExpr(call)
					}
				}
			case *ast.ValueSpec:
				if len(t.Values) > 0 {
					if call, ok := t.Values[0].(*ast.CallExpr); ok {
						return v.parseNewDescCallExpr(call)
					}
				}
			}

			if v.strict {
				v.issues = append(v.issues, Issue{
					Pos:    v.fs.Position(stmt.Pos()),
					Metric: "",
					Text:   fmt.Sprintf("parsing desc of type %T is not supported", stmt.Obj.Decl),
				})
			}
		}

	default:
		if v.strict {
			v.issues = append(v.issues, Issue{
				Pos:    v.fs.Position(stmt.Pos()),
				Metric: "",
				Text:   fmt.Sprintf("parsing desc of type %T is not supported", stmt),
			})
		}
	}

	return nil, nil
}

func (v *visitor) parseNewDescCallExpr(call *ast.CallExpr) (*string, *string) {
	var (
		help string
		name string
		ok   bool
	)

	switch expr := call.Fun.(type) {
	case *ast.Ident:
		if expr.Name != "NewDesc" {
			if v.strict {
				v.issues = append(v.issues, Issue{
					Pos:    v.fs.Position(expr.Pos()),
					Metric: "",
					Text:   fmt.Sprintf("parsing desc with function %s is not supported", expr.Name),
				})
			}
			return nil, nil
		}
	case *ast.SelectorExpr:
		if expr.Sel.Name != "NewDesc" {
			if v.strict {
				v.issues = append(v.issues, Issue{
					Pos:    v.fs.Position(expr.Sel.Pos()),
					Metric: "",
					Text:   fmt.Sprintf("parsing desc with function %s is not supported", expr.Sel.Name),
				})
			}
			return nil, nil
		}
	default:
		if v.strict {
			v.issues = append(v.issues, Issue{
				Pos:    v.fs.Position(expr.Pos()),
				Metric: "",
				Text:   fmt.Sprintf("parsing desc of %T is not supported", expr),
			})
		}
		return nil, nil
	}

	// k8s.io/component-base/metrics.NewDesc has 6 args
	// while prometheus.NewDesc has 4 args
	if len(call.Args) < 4 && v.strict {
		v.issues = append(v.issues, Issue{
			Metric: "",
			Pos:    v.fs.Position(call.Pos()),
			Text:   "NewDesc should have at least 4 args",
		})
		return nil, nil
	}

	name, ok = v.parseValue("fqName", call.Args[0])
	if !ok {
		return nil, nil
	}
	help, ok = v.parseValue("help", call.Args[1])
	if !ok {
		return nil, nil
	}

	return &name, &help
}

func mustUnquote(str string) string {
	stringLiteral, err := strconv.Unquote(str)
	if err != nil {
		panic(err)
	}

	return stringLiteral
}

func getConstMetricType(name string) *dto.MetricType {
	metricType := dto.MetricType_UNTYPED
	if name == "CounterValue" {
		metricType = dto.MetricType_COUNTER
	} else if name == "GaugeValue" {
		metricType = dto.MetricType_GAUGE
	}

	return &metricType
}
