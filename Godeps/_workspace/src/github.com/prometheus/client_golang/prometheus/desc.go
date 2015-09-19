package prometheus

import (
	"bytes"
	"errors"
	"fmt"
	"hash/fnv"
	"regexp"
	"sort"
	"strings"

	"github.com/golang/protobuf/proto"

	dto "github.com/prometheus/client_model/go"
)

var (
	metricNameRE = regexp.MustCompile(`^[a-zA-Z_][a-zA-Z0-9_:]*$`)
	labelNameRE  = regexp.MustCompile("^[a-zA-Z_][a-zA-Z0-9_]*$")
)

// reservedLabelPrefix is a prefix which is not legal in user-supplied
// label names.
const reservedLabelPrefix = "__"

// Labels represents a collection of label name -> value mappings. This type is
// commonly used with the With(Labels) and GetMetricWith(Labels) methods of
// metric vector Collectors, e.g.:
//     myVec.With(Labels{"code": "404", "method": "GET"}).Add(42)
//
// The other use-case is the specification of constant label pairs in Opts or to
// create a Desc.
type Labels map[string]string

// Desc is the descriptor used by every Prometheus Metric. It is essentially
// the immutable meta-data of a Metric. The normal Metric implementations
// included in this package manage their Desc under the hood. Users only have to
// deal with Desc if they use advanced features like the ExpvarCollector or
// custom Collectors and Metrics.
//
// Descriptors registered with the same registry have to fulfill certain
// consistency and uniqueness criteria if they share the same fully-qualified
// name: They must have the same help string and the same label names (aka label
// dimensions) in each, constLabels and variableLabels, but they must differ in
// the values of the constLabels.
//
// Descriptors that share the same fully-qualified names and the same label
// values of their constLabels are considered equal.
//
// Use NewDesc to create new Desc instances.
type Desc struct {
	// fqName has been built from Namespace, Subsystem, and Name.
	fqName string
	// help provides some helpful information about this metric.
	help string
	// constLabelPairs contains precalculated DTO label pairs based on
	// the constant labels.
	constLabelPairs []*dto.LabelPair
	// VariableLabels contains names of labels for which the metric
	// maintains variable values.
	variableLabels []string
	// id is a hash of the values of the ConstLabels and fqName. This
	// must be unique among all registered descriptors and can therefore be
	// used as an identifier of the descriptor.
	id uint64
	// dimHash is a hash of the label names (preset and variable) and the
	// Help string. Each Desc with the same fqName must have the same
	// dimHash.
	dimHash uint64
	// err is an error that occured during construction. It is reported on
	// registration time.
	err error
}

// NewDesc allocates and initializes a new Desc. Errors are recorded in the Desc
// and will be reported on registration time. variableLabels and constLabels can
// be nil if no such labels should be set. fqName and help must not be empty.
//
// variableLabels only contain the label names. Their label values are variable
// and therefore not part of the Desc. (They are managed within the Metric.)
//
// For constLabels, the label values are constant. Therefore, they are fully
// specified in the Desc. See the Opts documentation for the implications of
// constant labels.
func NewDesc(fqName, help string, variableLabels []string, constLabels Labels) *Desc {
	d := &Desc{
		fqName:         fqName,
		help:           help,
		variableLabels: variableLabels,
	}
	if help == "" {
		d.err = errors.New("empty help string")
		return d
	}
	if !metricNameRE.MatchString(fqName) {
		d.err = fmt.Errorf("%q is not a valid metric name", fqName)
		return d
	}
	// labelValues contains the label values of const labels (in order of
	// their sorted label names) plus the fqName (at position 0).
	labelValues := make([]string, 1, len(constLabels)+1)
	labelValues[0] = fqName
	labelNames := make([]string, 0, len(constLabels)+len(variableLabels))
	labelNameSet := map[string]struct{}{}
	// First add only the const label names and sort them...
	for labelName := range constLabels {
		if !checkLabelName(labelName) {
			d.err = fmt.Errorf("%q is not a valid label name", labelName)
			return d
		}
		labelNames = append(labelNames, labelName)
		labelNameSet[labelName] = struct{}{}
	}
	sort.Strings(labelNames)
	// ... so that we can now add const label values in the order of their names.
	for _, labelName := range labelNames {
		labelValues = append(labelValues, constLabels[labelName])
	}
	// Now add the variable label names, but prefix them with something that
	// cannot be in a regular label name. That prevents matching the label
	// dimension with a different mix between preset and variable labels.
	for _, labelName := range variableLabels {
		if !checkLabelName(labelName) {
			d.err = fmt.Errorf("%q is not a valid label name", labelName)
			return d
		}
		labelNames = append(labelNames, "$"+labelName)
		labelNameSet[labelName] = struct{}{}
	}
	if len(labelNames) != len(labelNameSet) {
		d.err = errors.New("duplicate label names")
		return d
	}
	h := fnv.New64a()
	var b bytes.Buffer // To copy string contents into, avoiding []byte allocations.
	for _, val := range labelValues {
		b.Reset()
		b.WriteString(val)
		b.WriteByte(separatorByte)
		h.Write(b.Bytes())
	}
	d.id = h.Sum64()
	// Sort labelNames so that order doesn't matter for the hash.
	sort.Strings(labelNames)
	// Now hash together (in this order) the help string and the sorted
	// label names.
	h.Reset()
	b.Reset()
	b.WriteString(help)
	b.WriteByte(separatorByte)
	h.Write(b.Bytes())
	for _, labelName := range labelNames {
		b.Reset()
		b.WriteString(labelName)
		b.WriteByte(separatorByte)
		h.Write(b.Bytes())
	}
	d.dimHash = h.Sum64()

	d.constLabelPairs = make([]*dto.LabelPair, 0, len(constLabels))
	for n, v := range constLabels {
		d.constLabelPairs = append(d.constLabelPairs, &dto.LabelPair{
			Name:  proto.String(n),
			Value: proto.String(v),
		})
	}
	sort.Sort(LabelPairSorter(d.constLabelPairs))
	return d
}

// NewInvalidDesc returns an invalid descriptor, i.e. a descriptor with the
// provided error set. If a collector returning such a descriptor is registered,
// registration will fail with the provided error. NewInvalidDesc can be used by
// a Collector to signal inability to describe itself.
func NewInvalidDesc(err error) *Desc {
	return &Desc{
		err: err,
	}
}

func (d *Desc) String() string {
	lpStrings := make([]string, 0, len(d.constLabelPairs))
	for _, lp := range d.constLabelPairs {
		lpStrings = append(
			lpStrings,
			fmt.Sprintf("%s=%q", lp.GetName(), lp.GetValue()),
		)
	}
	return fmt.Sprintf(
		"Desc{fqName: %q, help: %q, constLabels: {%s}, variableLabels: %v}",
		d.fqName,
		d.help,
		strings.Join(lpStrings, ","),
		d.variableLabels,
	)
}

func checkLabelName(l string) bool {
	return labelNameRE.MatchString(l) &&
		!strings.HasPrefix(l, reservedLabelPrefix)
}
