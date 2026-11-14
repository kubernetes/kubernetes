// Copyright The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package expfmt

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	goabnf "github.com/pandatix/go-abnf"
	dto "github.com/prometheus/client_model/go"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/timestamppb"

	"github.com/prometheus/common/model"
)

const openMetricsExemplarMaxRunes = 128

// openMetrics10ABNF is the OpenMetrics 1.0 ABNF grammar (RFC 5234) from
// https://github.com/prometheus/OpenMetrics/blob/v1.0.0/specification/OpenMetrics.md#abnf,
// extended with UTF-8 quoted metric and label names. When legacy validation is
// active on the decoder, quoted names that fail legacy validation are rejected
// during semantic validation.
//
// NOTE: Every OpenMetrics 1.0 metric-descriptor, sample, and eof construct is
// strictly single-line (normal-char excludes LF %x0A). Adding the `line` rule
// allows streaming line-by-line from an io.Reader via goabnf.ParseForest per
// line while keeping the full exposition grammar intact.
const openMetrics10ABNF = `exposition = metricset eof-line

metricset = *(metric-descriptor / sample)

line = metric-descriptor / sample / eof-line
eof-line = HASH SP eof [ LF ]

metric-descriptor = HASH SP type SP metric-descriptor-name SP metric-type LF
metric-descriptor =/ HASH SP help SP metric-descriptor-name SP help-value LF
metric-descriptor =/ HASH SP unit SP metric-descriptor-name SP unit-value LF

metric-descriptor-name = metricname / metricname-utf8
help-value = escaped-string
unit-value = *metricname-char

metric-type = counter / gauge / histogram / gaugehistogram / stateset
metric-type =/ info / summary / unknown

sample = metricname-and-labels SP sample-value [SP sample-timestamp] [exemplar] LF

sample-value = number
sample-timestamp = timestamp

exemplar = SP HASH SP labels SP exemplar-value [SP exemplar-timestamp]
exemplar-value = number
exemplar-timestamp = timestamp

metricname-and-labels = (metricname / metricname-utf8) [labels] / labels-with-metricname
labels = "{" [label *(COMMA label)] "}"
labels-with-metricname = "{" [label *(COMMA label) COMMA] metricname-utf8 *(COMMA label) "}"

label = label-key EQ DQUOTE label-value DQUOTE
label-key = label-name / label-name-utf8
label-name-utf8 = DQUOTE escaped-string-non-empty DQUOTE
label-value = escaped-string

number = realnumber
; Case insensitive
number =/ [SIGN] ("inf" / "infinity")
number =/ "nan"

timestamp = realnumber

; Leading 0s explicitly okay
realnumber = [SIGN] 1*DIGIT
realnumber =/ [SIGN] 1*DIGIT ["." *DIGIT] [ "e" [SIGN] 1*DIGIT ]
realnumber =/ [SIGN] *DIGIT "." 1*DIGIT [ "e" [SIGN] 1*DIGIT ]

; RFC 5234 is case insensitive.
; Uppercase
eof = %d69.79.70
type = %d84.89.80.69
help = %d72.69.76.80
unit = %d85.78.73.84
; Lowercase
counter = %d99.111.117.110.116.101.114
gauge = %d103.97.117.103.101
histogram = %d104.105.115.116.111.103.114.97.109
gaugehistogram = gauge histogram
stateset = %d115.116.97.116.101.115.101.116
info = %d105.110.102.111
summary = %d115.117.109.109.97.114.121
unknown = %d117.110.107.110.111.119.110

BS = "\"
EQ = "="
COMMA = ","
HASH = "#"
SIGN = "-" / "+"

metricname = metricname-initial-char 0*metricname-char

metricname-char = metricname-initial-char / DIGIT
metricname-initial-char = ALPHA / "_" / ":"
metricname-utf8 = DQUOTE escaped-string-non-empty DQUOTE

label-name = label-name-initial-char *label-name-char

label-name-char = label-name-initial-char / DIGIT
label-name-initial-char = ALPHA / "_"

escaped-string = *escaped-char
escaped-string-non-empty = 1*escaped-char

escaped-char = normal-char
escaped-char =/ BS (DQUOTE / BS / normal-char)

; Any unicode character, except newline, double quote, and backslash
normal-char = %x00-09 / %x0B-21 / %x23-5B / %x5D-D7FF / %xE000-10FFFF
`

var (
	om10GrammarOnce sync.Once
	om10Grammar     *goabnf.Grammar
	om10GrammarErr  error

	omEscapedReplacer = strings.NewReplacer(
		`\"`, `"`,
		`\\`, `\`,
		`\n`, "\n",
	)
)

func getOpenMetrics10Grammar() (*goabnf.Grammar, error) {
	om10GrammarOnce.Do(func() {
		raw := strings.ReplaceAll(openMetrics10ABNF, "\r\n", "\n")
		raw = strings.ReplaceAll(raw, "\n", "\r\n")
		om10Grammar, om10GrammarErr = goabnf.ParseABNF([]byte(raw))
	})
	return om10Grammar, om10GrammarErr
}

type openMetricsFamilyState struct {
	mf         *dto.MetricFamily
	hasType    bool
	hasHelp    bool
	hasUnit    bool
	hasSamples bool
	emitted    bool

	counters   map[uint64]*dto.Metric
	summaries  map[uint64]*dto.Metric
	histograms map[uint64]*dto.Metric
}

// openMetricsParser streams OpenMetrics 1.0 text exposition format line-by-line
// using the OpenMetrics 1.0 ABNF grammar.
type openMetricsParser struct {
	br          *bufio.Reader
	g           *goabnf.Grammar
	initErr     error
	scheme      model.ValidationScheme
	families    map[string]*openMetricsFamilyState
	curFamily   *openMetricsFamilyState
	scratchLbl  map[string]string
	lineNum     int
	pendingLine []byte
	pendingNode *goabnf.ParseTree
	done        bool
}

func newOpenMetricsParser(r io.Reader, scheme model.ValidationScheme) *openMetricsParser {
	br, ok := r.(*bufio.Reader)
	if !ok {
		br = bufio.NewReader(r)
	}
	g, err := getOpenMetrics10Grammar()
	if err != nil {
		err = fmt.Errorf("failed to initialize OpenMetrics 1.0 ABNF grammar: %w", err)
	}
	return &openMetricsParser{
		br:         br,
		g:          g,
		initErr:    err,
		scheme:     scheme,
		families:   make(map[string]*openMetricsFamilyState),
		scratchLbl: make(map[string]string),
	}
}

func (p *openMetricsParser) validationScheme() model.ValidationScheme {
	if p.scheme == model.UnsetValidation {
		return model.LegacyValidation
	}
	return p.scheme
}

func emitOpenMetricsFamily(st *openMetricsFamilyState, v *dto.MetricFamily, lineNum int) error {
	if st.mf.Type == nil {
		st.mf.Type = dto.MetricType_UNTYPED.Enum()
	}
	name := st.mf.Name
	if st.mf.GetType() == dto.MetricType_COUNTER {
		for _, m := range st.mf.Metric {
			if m.GetCounter().Value == nil {
				return ParseError{
					Line: lineNum,
					Msg:  fmt.Sprintf("counter %q missing _total value", st.mf.GetName()),
				}
			}
		}
		name = proto.String(st.mf.GetName() + "_total")
	}
	for _, m := range st.histograms {
		normalizeHistogram(m.GetHistogram())
	}
	v.Name = name
	v.Help = st.mf.Help
	v.Type = st.mf.Type
	v.Metric = st.mf.Metric
	v.Unit = st.mf.Unit
	st.emitted = true
	st.mf.Metric = nil
	st.counters = nil
	st.summaries = nil
	st.histograms = nil
	return nil
}

// next reads from the input stream until the next MetricFamily is complete and
// populates v, or returns io.EOF when # EOF is reached.
func (p *openMetricsParser) next(v *dto.MetricFamily) error {
	if p.initErr != nil {
		return p.initErr
	}
	for {
		if p.done {
			return io.EOF
		}

		var (
			line []byte
			node *goabnf.ParseTree
		)
		if p.pendingNode != nil {
			line = p.pendingLine
			node = p.pendingNode
			p.pendingLine = nil
			p.pendingNode = nil
		} else {
			rawLine, readErr := p.br.ReadBytes('\n')
			if len(rawLine) == 0 {
				if errors.Is(readErr, io.EOF) {
					return ParseError{
						Line: max(1, p.lineNum+1),
						Msg:  "data does not end with # EOF",
					}
				}
				return readErr
			}
			if readErr != nil && !errors.Is(readErr, io.EOF) {
				return readErr
			}
			p.lineNum++
			line = rawLine

			forest, err := goabnf.ParseForest(line, p.g, "line")
			if err != nil {
				return err
			}
			if !forest.Valid() {
				return formatABNFLineParseError(p.lineNum, forest.ParseError())
			}
			tree := forest.Tree()
			if tree == nil || len(tree.Children) == 0 {
				return ParseError{Line: p.lineNum, Msg: "failed to extract parse tree"}
			}
			node = tree.Children[0]
		}

		switch node.Rule {
		case "eof-line":
			if _, peekErr := p.br.Peek(1); !errors.Is(peekErr, io.EOF) {
				if peekErr != nil {
					return peekErr
				}
				return ParseError{
					Line: p.lineNum + 1,
					Msg:  "unexpected data after # EOF",
				}
			}
			p.done = true
			if p.curFamily != nil && p.curFamily.hasSamples && !p.curFamily.emitted {
				st := p.curFamily
				p.curFamily = nil
				return emitOpenMetricsFamily(st, v, p.lineNum)
			}
			return io.EOF

		case "metric-descriptor":
			if err := p.handleDescriptor(line, node); err != nil {
				return err
			}

		case "sample":
			if p.curFamily != nil && p.curFamily.hasSamples && !p.curFamily.emitted {
				sampleName := extractSampleMetricName(line, node)
				if _, ok := matchFamilySample(p.curFamily, sampleName); !ok && sampleName != p.curFamily.mf.GetName() {
					p.pendingLine = line
					p.pendingNode = node
					st := p.curFamily
					p.curFamily = nil
					return emitOpenMetricsFamily(st, v, p.lineNum)
				}
			}
			if err := p.handleSample(line, node); err != nil {
				return err
			}
		}
	}
}

func (p *openMetricsParser) getOrCreateDescriptorFamily(name string) (*openMetricsFamilyState, error) {
	if !p.validationScheme().IsValidMetricName(name) {
		return nil, ParseError{Line: p.lineNum, Msg: fmt.Sprintf("invalid metric name %q", name)}
	}
	if p.curFamily != nil && !p.curFamily.emitted && p.curFamily.mf.GetName() == name {
		if p.curFamily.hasSamples {
			return nil, ParseError{Line: p.lineNum, Msg: fmt.Sprintf("metadata for metric %q after samples", name)}
		}
		return p.curFamily, nil
	}
	if prev, exists := p.families[name]; exists {
		if prev.hasSamples {
			return nil, ParseError{Line: p.lineNum, Msg: fmt.Sprintf("metadata for metric %q after samples", name)}
		}
		return prev, nil
	}
	st := &openMetricsFamilyState{
		mf: &dto.MetricFamily{
			Name: proto.String(name),
		},
	}
	p.families[name] = st
	if p.curFamily == nil || !p.curFamily.hasSamples {
		p.curFamily = st
	}
	return st, nil
}

func (p *openMetricsParser) handleDescriptor(input []byte, node *goabnf.ParseTree) error {
	var (
		descKind string
		mfName   string
		valNode  *goabnf.ParseTree
	)
	for _, c := range node.Children {
		switch c.Rule {
		case "type", "help", "unit":
			descKind = c.Rule
		case "metric-descriptor-name":
			mfName = extractMetricNameNode(input, c)
		case "metric-type", "help-value", "unit-value":
			valNode = c
		}
	}

	st, err := p.getOrCreateDescriptorFamily(mfName)
	if err != nil {
		return err
	}

	switch descKind {
	case "type":
		if st.hasType {
			return ParseError{Line: p.lineNum, Msg: fmt.Sprintf("second TYPE line for metric name %q", mfName)}
		}
		st.hasType = true
		rawType := string(input[valNode.Start:valNode.End])
		switch rawType {
		case "counter":
			st.mf.Type = dto.MetricType_COUNTER.Enum()
		case "gauge":
			st.mf.Type = dto.MetricType_GAUGE.Enum()
		case "histogram":
			st.mf.Type = dto.MetricType_HISTOGRAM.Enum()
		case "gaugehistogram":
			st.mf.Type = dto.MetricType_GAUGE_HISTOGRAM.Enum()
		case "summary":
			st.mf.Type = dto.MetricType_SUMMARY.Enum()
		case "unknown":
			st.mf.Type = dto.MetricType_UNTYPED.Enum()
		case "info", "stateset":
			// TODO(bwplotka): Expand client_model (dto.MetricType) to support INFO and STATESET metric types.
			return ParseError{Line: p.lineNum, Msg: fmt.Sprintf("unsupported metric type %q", rawType)}
		default:
			return ParseError{Line: p.lineNum, Msg: fmt.Sprintf("unknown metric type %q", rawType)}
		}
	case "help":
		if st.hasHelp {
			return ParseError{Line: p.lineNum, Msg: fmt.Sprintf("second HELP line for metric name %q", mfName)}
		}
		st.hasHelp = true
		rawHelp := string(input[valNode.Start:valNode.End])
		if strings.HasSuffix(rawHelp, "\r") {
			return ParseError{Line: p.lineNum, Msg: fmt.Sprintf("unexpected carriage return in HELP for metric %q", mfName)}
		}
		st.mf.Help = proto.String(unescapeOpenMetricsString(rawHelp))
	case "unit":
		if st.hasUnit {
			return ParseError{Line: p.lineNum, Msg: fmt.Sprintf("second UNIT line for metric name %q", mfName)}
		}
		st.hasUnit = true
		unit := string(input[valNode.Start:valNode.End])
		if unit != "" {
			if !strings.HasSuffix(mfName, "_"+unit) {
				return ParseError{Line: p.lineNum, Msg: fmt.Sprintf("unit %q not a suffix of metric %q", unit, mfName)}
			}
			st.mf.Unit = proto.String(unit)
		}
	}
	return nil
}

type sampleRole int

const (
	roleValue sampleRole = iota
	roleCreated
	roleCount
	roleSum
	roleBucket
	roleQuantile
)

func (p *openMetricsParser) resolveSampleFamily(sampleName string) (*openMetricsFamilyState, sampleRole, error) {
	if !p.validationScheme().IsValidMetricName(sampleName) {
		return nil, roleValue, ParseError{Line: p.lineNum, Msg: fmt.Sprintf("invalid metric name %q", sampleName)}
	}

	if p.curFamily != nil && !p.curFamily.emitted {
		if role, ok := matchFamilySample(p.curFamily, sampleName); ok {
			if p.curFamily.mf.Type == nil {
				p.curFamily.mf.Type = dto.MetricType_UNTYPED.Enum()
			}
			p.curFamily.hasSamples = true
			return p.curFamily, role, nil
		}
		// If curFamily was explicitly typed as counter/histogram/gaugehistogram,
		// a bare sample matching the base family name is missing its required suffix.
		if p.curFamily.hasType && sampleName == p.curFamily.mf.GetName() {
			return nil, roleValue, ParseError{Line: p.lineNum, Msg: fmt.Sprintf("sample %q missing required suffix for type %s", sampleName, p.curFamily.mf.GetType())}
		}
	}

	if st, exists := p.families[sampleName]; exists {
		if role, ok := matchFamilySample(st, sampleName); ok {
			if st.emitted {
				return nil, roleValue, ParseError{Line: p.lineNum, Msg: fmt.Sprintf("interleaved metric family %q", st.mf.GetName())}
			}
			if st.mf.Type == nil {
				st.mf.Type = dto.MetricType_UNTYPED.Enum()
			}
			st.hasSamples = true
			p.curFamily = st
			return st, role, nil
		}
		if st.hasType {
			return nil, roleValue, ParseError{Line: p.lineNum, Msg: fmt.Sprintf("sample %q missing required suffix for type %s", sampleName, st.mf.GetType())}
		}
	}

	for _, st := range p.families {
		if role, ok := matchFamilySample(st, sampleName); ok {
			if st.emitted {
				return nil, roleValue, ParseError{Line: p.lineNum, Msg: fmt.Sprintf("interleaved metric family %q", st.mf.GetName())}
			}
			if st.mf.Type == nil {
				st.mf.Type = dto.MetricType_UNTYPED.Enum()
			}
			st.hasSamples = true
			p.curFamily = st
			return st, role, nil
		}
	}

	st := &openMetricsFamilyState{
		mf: &dto.MetricFamily{
			Name: proto.String(sampleName),
			Type: dto.MetricType_UNTYPED.Enum(),
		},
		hasSamples: true,
	}
	p.families[sampleName] = st
	p.curFamily = st
	return st, roleValue, nil
}

func matchFamilySample(st *openMetricsFamilyState, sampleName string) (sampleRole, bool) {
	if st == nil || st.mf == nil {
		return roleValue, false
	}
	mfName := st.mf.GetName()
	if st.mf.Type == nil {
		if sampleName == mfName {
			return roleValue, true
		}
		return roleValue, false
	}

	switch st.mf.GetType() {
	case dto.MetricType_COUNTER:
		switch sampleName {
		case mfName + "_total":
			return roleValue, true
		case mfName + "_created":
			return roleCreated, true
		}
	case dto.MetricType_GAUGE, dto.MetricType_UNTYPED:
		if sampleName == mfName {
			return roleValue, true
		}
	case dto.MetricType_SUMMARY:
		switch sampleName {
		case mfName:
			return roleQuantile, true
		case mfName + "_count":
			return roleCount, true
		case mfName + "_sum":
			return roleSum, true
		case mfName + "_created":
			return roleCreated, true
		}
	case dto.MetricType_HISTOGRAM:
		switch sampleName {
		case mfName + "_bucket":
			return roleBucket, true
		case mfName + "_count":
			return roleCount, true
		case mfName + "_sum":
			return roleSum, true
		case mfName + "_created":
			return roleCreated, true
		}
	case dto.MetricType_GAUGE_HISTOGRAM:
		switch sampleName {
		case mfName + "_bucket":
			return roleBucket, true
		case mfName + "_gcount", mfName + "_count":
			return roleCount, true
		case mfName + "_gsum", mfName + "_sum":
			return roleSum, true
		case mfName + "_created":
			return roleCreated, true
		}
	}
	return roleValue, false
}

func (p *openMetricsParser) handleSample(input []byte, node *goabnf.ParseTree) error {
	var (
		nameAndLabelsNode *goabnf.ParseTree
		valStr            string
		hasTS             bool
		tsStr             string
		exemplarNode      *goabnf.ParseTree
	)
	for _, c := range node.Children {
		switch c.Rule {
		case "metricname-and-labels":
			nameAndLabelsNode = c
		case "sample-value":
			valStr = string(input[c.Start:c.End])
		case "sample-timestamp":
			hasTS = true
			tsStr = string(input[c.Start:c.End])
		case "exemplar":
			exemplarNode = c
		}
	}

	sampleName, rawLabels := extractNameAndLabels(input, nameAndLabelsNode)
	st, role, err := p.resolveSampleFamily(sampleName)
	if err != nil {
		return err
	}

	val, err := parseOpenMetricsFloat(valStr)
	if err != nil {
		return ParseError{Line: p.lineNum, Msg: fmt.Sprintf("invalid sample value %q: %v", valStr, err)}
	}

	var tsMs *int64
	if hasTS {
		parsedTS, err := parseSampleTimestampMs(tsStr)
		if err != nil {
			return ParseError{Line: p.lineNum, Msg: fmt.Sprintf("invalid timestamp %q: %v", tsStr, err)}
		}
		tsMs = proto.Int64(parsedTS)
	}

	var ex *dto.Exemplar
	if exemplarNode != nil {
		ex, err = p.parseExemplar(input, exemplarNode, true)
		if err != nil {
			return err
		}
	}

	// Validate labels and extract special 'quantile' or 'le' label if applicable.
	clear(p.scratchLbl)
	p.scratchLbl[string(model.MetricNameLabel)] = st.mf.GetName()
	var (
		labelPairs []*dto.LabelPair
		quantile   = math.NaN()
		bucketLE   = math.NaN()
	)
	seenLabels := make(map[string]struct{}, len(rawLabels))
	for _, rl := range rawLabels {
		if rl.name == string(model.MetricNameLabel) {
			return ParseError{Line: p.lineNum, Msg: fmt.Sprintf("label name %q is reserved", model.MetricNameLabel)}
		}
		if !p.validationScheme().IsValidLabelName(rl.name) {
			return ParseError{Line: p.lineNum, Msg: fmt.Sprintf("invalid label name %q", rl.name)}
		}
		if !model.LabelValue(rl.value).IsValid() {
			return ParseError{Line: p.lineNum, Msg: fmt.Sprintf("invalid label value %q", rl.value)}
		}
		if _, dup := seenLabels[rl.name]; dup {
			return ParseError{Line: p.lineNum, Msg: fmt.Sprintf("duplicate label %q for metric %q", rl.name, st.mf.GetName())}
		}
		seenLabels[rl.name] = struct{}{}

		if role == roleQuantile && rl.name == model.QuantileLabel {
			q, err := parseOpenMetricsFloat(rl.value)
			if err != nil || math.IsNaN(q) || q < 0 || q > 1 {
				return ParseError{Line: p.lineNum, Msg: fmt.Sprintf("invalid quantile %q for summary %q", rl.value, st.mf.GetName())}
			}
			quantile = q
			continue
		}
		if role == roleBucket && rl.name == model.BucketLabel {
			le, err := parseOpenMetricsFloat(rl.value)
			if err != nil || math.IsNaN(le) {
				return ParseError{Line: p.lineNum, Msg: fmt.Sprintf("invalid bucket bound %q for histogram %q", rl.value, st.mf.GetName())}
			}
			bucketLE = le
			continue
		}
		p.scratchLbl[rl.name] = rl.value
		labelPairs = append(labelPairs, &dto.LabelPair{
			Name:  proto.String(rl.name),
			Value: proto.String(rl.value),
		})
	}

	switch st.mf.GetType() {
	case dto.MetricType_GAUGE:
		if ex != nil {
			return ParseError{Line: p.lineNum, Msg: fmt.Sprintf("exemplar not allowed on gauge metric %q", st.mf.GetName())}
		}
		st.mf.Metric = append(st.mf.Metric, &dto.Metric{
			Label:       labelPairs,
			Gauge:       &dto.Gauge{Value: proto.Float64(val)},
			TimestampMs: tsMs,
		})

	case dto.MetricType_UNTYPED:
		if ex != nil {
			return ParseError{Line: p.lineNum, Msg: fmt.Sprintf("exemplar not allowed on untyped metric %q", st.mf.GetName())}
		}
		st.mf.Metric = append(st.mf.Metric, &dto.Metric{
			Label:       labelPairs,
			Untyped:     &dto.Untyped{Value: proto.Float64(val)},
			TimestampMs: tsMs,
		})

	case dto.MetricType_COUNTER:
		if st.counters == nil {
			st.counters = make(map[uint64]*dto.Metric)
		}
		sig := model.LabelsToSignature(p.scratchLbl)
		switch role {
		case roleValue:
			if math.IsNaN(val) || val < 0 {
				return ParseError{Line: p.lineNum, Msg: fmt.Sprintf("counter value must not be negative or NaN for %q", st.mf.GetName())}
			}
			m := st.counters[sig]
			if m == nil || m.Counter.Value != nil {
				m = &dto.Metric{
					Label:   labelPairs,
					Counter: &dto.Counter{},
				}
				st.counters[sig] = m
				st.mf.Metric = append(st.mf.Metric, m)
			}
			m.Counter.Value = proto.Float64(val)
			m.Counter.Exemplar = ex
			m.TimestampMs = tsMs
		case roleCreated:
			if ex != nil {
				return ParseError{Line: p.lineNum, Msg: fmt.Sprintf("exemplar not allowed on _created line for %q", st.mf.GetName())}
			}
			createdTS, err := parseTimestampToProto(valStr)
			if err != nil {
				return ParseError{Line: p.lineNum, Msg: fmt.Sprintf("invalid created timestamp %q: %v", valStr, err)}
			}
			m := st.counters[sig]
			if m == nil || m.Counter.CreatedTimestamp != nil {
				m = &dto.Metric{
					Label:   labelPairs,
					Counter: &dto.Counter{},
				}
				st.counters[sig] = m
				st.mf.Metric = append(st.mf.Metric, m)
			}
			m.Counter.CreatedTimestamp = createdTS
		}

	case dto.MetricType_SUMMARY:
		if ex != nil {
			return ParseError{Line: p.lineNum, Msg: fmt.Sprintf("exemplar not allowed on summary metric %q", st.mf.GetName())}
		}
		if st.summaries == nil {
			st.summaries = make(map[uint64]*dto.Metric)
		}
		sig := model.LabelsToSignature(p.scratchLbl)
		m := st.summaries[sig]
		if m == nil {
			m = &dto.Metric{
				Label:   labelPairs,
				Summary: &dto.Summary{},
			}
			st.summaries[sig] = m
			st.mf.Metric = append(st.mf.Metric, m)
		}
		if role != roleCreated && tsMs != nil {
			m.TimestampMs = tsMs
		}
		switch role {
		case roleQuantile:
			if math.IsNaN(quantile) {
				return ParseError{Line: p.lineNum, Msg: fmt.Sprintf("missing quantile label for summary %q", st.mf.GetName())}
			}
			if !math.IsNaN(val) && val < 0 {
				return ParseError{Line: p.lineNum, Msg: fmt.Sprintf("negative quantile value for summary %q", st.mf.GetName())}
			}
			m.Summary.Quantile = append(m.Summary.Quantile, &dto.Quantile{
				Quantile: proto.Float64(quantile),
				Value:    proto.Float64(val),
			})
		case roleCount:
			if math.IsNaN(val) || val < 0 || float64(uint64(val)) != val {
				return ParseError{Line: p.lineNum, Msg: fmt.Sprintf("invalid count %q for summary %q", valStr, st.mf.GetName())}
			}
			m.Summary.SampleCount = proto.Uint64(uint64(val))
		case roleSum:
			if math.IsNaN(val) || val < 0 {
				return ParseError{Line: p.lineNum, Msg: fmt.Sprintf("invalid sum %q for summary %q", valStr, st.mf.GetName())}
			}
			m.Summary.SampleSum = proto.Float64(val)
		case roleCreated:
			createdTS, err := parseTimestampToProto(valStr)
			if err != nil {
				return ParseError{Line: p.lineNum, Msg: fmt.Sprintf("invalid created timestamp %q: %v", valStr, err)}
			}
			m.Summary.CreatedTimestamp = createdTS
		}

	case dto.MetricType_HISTOGRAM, dto.MetricType_GAUGE_HISTOGRAM:
		if st.histograms == nil {
			st.histograms = make(map[uint64]*dto.Metric)
		}
		sig := model.LabelsToSignature(p.scratchLbl)
		m := st.histograms[sig]
		if m == nil {
			m = &dto.Metric{
				Label:     labelPairs,
				Histogram: &dto.Histogram{},
			}
			st.histograms[sig] = m
			st.mf.Metric = append(st.mf.Metric, m)
		}
		if role != roleCreated && tsMs != nil {
			m.TimestampMs = tsMs
		}
		switch role {
		case roleBucket:
			if math.IsNaN(bucketLE) {
				return ParseError{Line: p.lineNum, Msg: fmt.Sprintf("missing le label for histogram %q", st.mf.GetName())}
			}
			if math.IsNaN(val) || val < 0 {
				return ParseError{Line: p.lineNum, Msg: fmt.Sprintf("negative or NaN bucket count for histogram %q", st.mf.GetName())}
			}
			b := &dto.Bucket{
				UpperBound: proto.Float64(bucketLE),
				Exemplar:   ex,
			}
			if uintVal := uint64(val); float64(uintVal) == val {
				b.CumulativeCount = proto.Uint64(uintVal)
			} else {
				b.CumulativeCountFloat = proto.Float64(val)
			}
			m.Histogram.Bucket = append(m.Histogram.Bucket, b)
		case roleCount:
			if ex != nil {
				return ParseError{Line: p.lineNum, Msg: fmt.Sprintf("exemplar not allowed on count sample for histogram %q", st.mf.GetName())}
			}
			if math.IsNaN(val) || val < 0 {
				return ParseError{Line: p.lineNum, Msg: fmt.Sprintf("negative or NaN count for histogram %q", st.mf.GetName())}
			}
			if uintVal := uint64(val); float64(uintVal) == val {
				m.Histogram.SampleCount = proto.Uint64(uintVal)
			} else {
				m.Histogram.SampleCountFloat = proto.Float64(val)
			}
		case roleSum:
			if ex != nil {
				return ParseError{Line: p.lineNum, Msg: fmt.Sprintf("exemplar not allowed on sum sample for histogram %q", st.mf.GetName())}
			}
			if math.IsNaN(val) || (st.mf.GetType() == dto.MetricType_HISTOGRAM && val < 0) {
				return ParseError{Line: p.lineNum, Msg: fmt.Sprintf("invalid sum %q for histogram %q", valStr, st.mf.GetName())}
			}
			m.Histogram.SampleSum = proto.Float64(val)
		case roleCreated:
			if ex != nil {
				return ParseError{Line: p.lineNum, Msg: fmt.Sprintf("exemplar not allowed on _created sample for histogram %q", st.mf.GetName())}
			}
			createdTS, err := parseTimestampToProto(valStr)
			if err != nil {
				return ParseError{Line: p.lineNum, Msg: fmt.Sprintf("invalid created timestamp %q: %v", valStr, err)}
			}
			m.Histogram.CreatedTimestamp = createdTS
		}
	}

	return nil
}

func (p *openMetricsParser) parseExemplar(input []byte, node *goabnf.ParseTree, enforceRuneLimit bool) (*dto.Exemplar, error) {
	var (
		rawLabels []parsedLabel
		valStr    string
		hasTS     bool
		tsStr     string
	)
	for _, c := range node.Children {
		switch c.Rule {
		case "labels", "labels-in-braces":
			rawLabels = extractLabelsFromList(input, c)
		case "exemplar-value", "number":
			valStr = string(input[c.Start:c.End])
		case "exemplar-timestamp", "timestamp":
			hasTS = true
			tsStr = string(input[c.Start:c.End])
		}
	}

	val, err := parseOpenMetricsFloat(valStr)
	if err != nil || math.IsNaN(val) || math.IsInf(val, 0) {
		return nil, ParseError{Line: p.lineNum, Msg: fmt.Sprintf("invalid exemplar value %q", valStr)}
	}

	var (
		pairs     = make([]*dto.LabelPair, 0, len(rawLabels))
		seenNames = make(map[string]struct{}, len(rawLabels))
		totalRune int
	)
	for _, rl := range rawLabels {
		if !p.validationScheme().IsValidLabelName(rl.name) {
			return nil, ParseError{Line: p.lineNum, Msg: fmt.Sprintf("invalid exemplar label name %q", rl.name)}
		}
		if !model.LabelValue(rl.value).IsValid() {
			return nil, ParseError{Line: p.lineNum, Msg: fmt.Sprintf("invalid exemplar label value %q", rl.value)}
		}
		if _, dup := seenNames[rl.name]; dup {
			return nil, ParseError{Line: p.lineNum, Msg: fmt.Sprintf("duplicate exemplar label name %q", rl.name)}
		}
		seenNames[rl.name] = struct{}{}
		totalRune += utf8.RuneCountInString(rl.name) + utf8.RuneCountInString(rl.value)
		pairs = append(pairs, &dto.LabelPair{
			Name:  proto.String(rl.name),
			Value: proto.String(rl.value),
		})
	}
	if enforceRuneLimit && totalRune > openMetricsExemplarMaxRunes {
		return nil, ParseError{Line: p.lineNum, Msg: fmt.Sprintf("exemplar label set exceeds %d runes (%d)", openMetricsExemplarMaxRunes, totalRune)}
	}

	ex := &dto.Exemplar{
		Label: pairs,
		Value: proto.Float64(val),
	}
	if hasTS {
		tsProto, err := parseTimestampToProto(tsStr)
		if err != nil {
			return nil, ParseError{Line: p.lineNum, Msg: fmt.Sprintf("invalid exemplar timestamp %q: %v", tsStr, err)}
		}
		ex.Timestamp = tsProto
	}
	return ex, nil
}

type parsedLabel struct {
	name  string
	value string
}

func extractSampleMetricName(input []byte, sampleNode *goabnf.ParseTree) string {
	for _, c := range sampleNode.Children {
		if c.Rule == "metricname-and-labels" {
			name, _ := extractNameAndLabels(input, c)
			return name
		}
	}
	return ""
}

func extractMetricNameNode(input []byte, node *goabnf.ParseTree) string {
	for _, c := range node.Children {
		switch c.Rule {
		case "metricname":
			return string(input[c.Start:c.End])
		case "metricname-utf8":
			return unescapeOpenMetricsString(string(input[c.Start+1 : c.End-1]))
		}
	}
	return ""
}

func extractNameAndLabels(input []byte, node *goabnf.ParseTree) (string, []parsedLabel) {
	var (
		metricName string
		labels     []parsedLabel
	)
	var walk func(n *goabnf.ParseTree)
	walk = func(n *goabnf.ParseTree) {
		for _, c := range n.Children {
			switch c.Rule {
			case "metricname":
				metricName = string(input[c.Start:c.End])
			case "metricname-utf8":
				metricName = unescapeOpenMetricsString(string(input[c.Start+1 : c.End-1]))
			case "label":
				labels = append(labels, extractSingleLabel(input, c))
			case "labels", "labels-in-braces", "labels-with-metricname", "name-and-labels-in-braces":
				walk(c)
			}
		}
	}
	walk(node)
	return metricName, labels
}

func extractLabelsFromList(input []byte, node *goabnf.ParseTree) []parsedLabel {
	var labels []parsedLabel
	for _, c := range node.Children {
		if c.Rule == "label" {
			labels = append(labels, extractSingleLabel(input, c))
		}
	}
	return labels
}

func extractSingleLabel(input []byte, node *goabnf.ParseTree) parsedLabel {
	var out parsedLabel
	for _, c := range node.Children {
		switch c.Rule {
		case "label-key":
			for _, kc := range c.Children {
				switch kc.Rule {
				case "label-name":
					out.name = string(input[kc.Start:kc.End])
				case "label-name-utf8":
					out.name = unescapeOpenMetricsString(string(input[kc.Start+1 : kc.End-1]))
				default:
					if kc.Rule == "" && kc.End > kc.Start+2 && input[kc.Start] == '"' && input[kc.End-1] == '"' {
						out.name = unescapeOpenMetricsString(string(input[kc.Start+1 : kc.End-1]))
					}
				}
			}
		case "label-value":
			out.value = unescapeOpenMetricsString(string(input[c.Start:c.End]))
		}
	}
	return out
}

func unescapeOpenMetricsString(s string) string {
	if strings.IndexByte(s, '\\') < 0 {
		return s
	}
	return omEscapedReplacer.Replace(s)
}

func parseOpenMetricsFloat(s string) (float64, error) {
	if strings.ContainsAny(s, "pP_") {
		return 0, fmt.Errorf("unsupported character in float %q", s)
	}
	return strconv.ParseFloat(s, 64)
}

func parseSampleTimestampMs(s string) (int64, error) {
	f, err := parseOpenMetricsFloat(s)
	if err != nil {
		return 0, err
	}
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return 0, fmt.Errorf("invalid timestamp %v", f)
	}
	return int64(math.Round(f * 1000)), nil
}

func parseTimestampToProto(s string) (*timestamppb.Timestamp, error) {
	if ts, ok := parseExactDecimalTimestamp(s); ok {
		return ts, nil
	}
	f, err := parseOpenMetricsFloat(s)
	if err != nil {
		return nil, err
	}
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return nil, fmt.Errorf("invalid timestamp %v", f)
	}
	sec, frac := math.Modf(f)
	nanos := int64(math.Round(frac * 1e9))
	return timestamppb.New(time.Unix(int64(sec), nanos)), nil
}

func parseExactDecimalTimestamp(s string) (*timestamppb.Timestamp, bool) {
	neg := false
	switch {
	case strings.HasPrefix(s, "-"):
		neg = true
		s = s[1:]
	case strings.HasPrefix(s, "+"):
		s = s[1:]
	}
	var exp int
	if idx := strings.IndexAny(s, "eE"); idx >= 0 {
		eVal, err := strconv.Atoi(s[idx+1:])
		if err != nil || eVal < -18 || eVal > 18 {
			return nil, false
		}
		exp = eVal
		s = s[:idx]
	}
	secPart, fracPart, _ := strings.Cut(s, ".")
	digits := secPart + fracPart
	if digits == "" {
		return nil, false
	}
	decPos := len(secPart) + exp
	switch {
	case decPos <= 0:
		secPart = "0"
		fracPart = strings.Repeat("0", -decPos) + digits
	case decPos >= len(digits):
		secPart = digits + strings.Repeat("0", decPos-len(digits))
		fracPart = ""
	default:
		secPart = digits[:decPos]
		fracPart = digits[decPos:]
	}
	var sec int64
	if secPart != "" {
		v, err := strconv.ParseInt(secPart, 10, 64)
		if err != nil {
			return nil, false
		}
		sec = v
	}
	var nanos int64
	if fracPart != "" {
		if len(fracPart) > 9 {
			fracPart = fracPart[:9]
		} else {
			fracPart += strings.Repeat("0", 9-len(fracPart))
		}
		v, err := strconv.ParseInt(fracPart, 10, 64)
		if err != nil {
			return nil, false
		}
		nanos = v
	}
	if neg {
		sec = -sec
		nanos = -nanos
	}
	return timestamppb.New(time.Unix(sec, nanos)), true
}

func formatABNFLineParseError(lineNum int, perr *goabnf.ParseError) error {
	if perr == nil {
		return ParseError{Line: lineNum, Msg: "syntax error"}
	}
	return ParseError{
		Line: lineNum,
		Msg:  perr.Error(),
	}
}
