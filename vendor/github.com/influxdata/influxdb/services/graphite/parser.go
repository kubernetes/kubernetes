package graphite

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/influxdata/influxdb/models"
)

// Minimum and maximum supported dates for timestamps.
var (
	MinDate = time.Date(1901, 12, 13, 0, 0, 0, 0, time.UTC)
	MaxDate = time.Date(2038, 1, 19, 0, 0, 0, 0, time.UTC)
)

var defaultTemplate *template

func init() {
	var err error
	defaultTemplate, err = NewTemplate("measurement*", nil, DefaultSeparator)
	if err != nil {
		panic(err)
	}
}

// Parser encapsulates a Graphite Parser.
type Parser struct {
	matcher *matcher
	tags    models.Tags
}

// Options are configurable values that can be provided to a Parser
type Options struct {
	Separator   string
	Templates   []string
	DefaultTags models.Tags
}

// NewParserWithOptions returns a graphite parser using the given options
func NewParserWithOptions(options Options) (*Parser, error) {

	matcher := newMatcher()
	matcher.AddDefaultTemplate(defaultTemplate)

	for _, pattern := range options.Templates {

		template := pattern
		filter := ""
		// Format is [filter] <template> [tag1=value1,tag2=value2]
		parts := strings.Fields(pattern)
		if len(parts) < 1 {
			continue
		} else if len(parts) >= 2 {
			if strings.Contains(parts[1], "=") {
				template = parts[0]
			} else {
				filter = parts[0]
				template = parts[1]
			}
		}

		// Parse out the default tags specific to this template
		var tags models.Tags
		if strings.Contains(parts[len(parts)-1], "=") {
			tagStrs := strings.Split(parts[len(parts)-1], ",")
			for _, kv := range tagStrs {
				parts := strings.Split(kv, "=")
				tags.SetString(parts[0], parts[1])
			}
		}

		tmpl, err := NewTemplate(template, tags, options.Separator)
		if err != nil {
			return nil, err
		}
		matcher.Add(filter, tmpl)
	}
	return &Parser{matcher: matcher, tags: options.DefaultTags}, nil
}

// NewParser returns a GraphiteParser instance.
func NewParser(templates []string, defaultTags models.Tags) (*Parser, error) {
	return NewParserWithOptions(
		Options{
			Templates:   templates,
			DefaultTags: defaultTags,
			Separator:   DefaultSeparator,
		})
}

// Parse performs Graphite parsing of a single line.
func (p *Parser) Parse(line string) (models.Point, error) {
	// Break into 3 fields (name, value, timestamp).
	fields := strings.Fields(line)
	if len(fields) != 2 && len(fields) != 3 {
		return nil, fmt.Errorf("received %q which doesn't have required fields", line)
	}

	// decode the name and tags
	template := p.matcher.Match(fields[0])
	measurement, tags, field, err := template.Apply(fields[0])
	if err != nil {
		return nil, err
	}

	// Could not extract measurement, use the raw value
	if measurement == "" {
		measurement = fields[0]
	}

	// Parse value.
	v, err := strconv.ParseFloat(fields[1], 64)
	if err != nil {
		return nil, fmt.Errorf(`field "%s" value: %s`, fields[0], err)
	}

	if math.IsNaN(v) || math.IsInf(v, 0) {
		return nil, &UnsupportedValueError{Field: fields[0], Value: v}
	}

	fieldValues := map[string]interface{}{}
	if field != "" {
		fieldValues[field] = v
	} else {
		fieldValues["value"] = v
	}

	// If no 3rd field, use now as timestamp
	timestamp := time.Now().UTC()

	if len(fields) == 3 {
		// Parse timestamp.
		unixTime, err := strconv.ParseFloat(fields[2], 64)
		if err != nil {
			return nil, fmt.Errorf(`field "%s" time: %s`, fields[0], err)
		}

		// -1 is a special value that gets converted to current UTC time
		// See https://github.com/graphite-project/carbon/issues/54
		if unixTime != float64(-1) {
			// Check if we have fractional seconds
			timestamp = time.Unix(int64(unixTime), int64((unixTime-math.Floor(unixTime))*float64(time.Second)))
			if timestamp.Before(MinDate) || timestamp.After(MaxDate) {
				return nil, fmt.Errorf("timestamp out of range")
			}
		}
	}

	// Set the default tags on the point if they are not already set
	for _, t := range p.tags {
		if _, ok := tags[string(t.Key)]; !ok {
			tags[string(t.Key)] = string(t.Value)
		}
	}
	return models.NewPoint(measurement, models.NewTags(tags), fieldValues, timestamp)
}

// ApplyTemplate extracts the template fields from the given line and
// returns the measurement name and tags.
func (p *Parser) ApplyTemplate(line string) (string, map[string]string, string, error) {
	// Break line into fields (name, value, timestamp), only name is used
	fields := strings.Fields(line)
	if len(fields) == 0 {
		return "", make(map[string]string), "", nil
	}
	// decode the name and tags
	template := p.matcher.Match(fields[0])
	name, tags, field, err := template.Apply(fields[0])
	// Set the default tags on the point if they are not already set
	for _, t := range p.tags {
		if _, ok := tags[string(t.Key)]; !ok {
			tags[string(t.Key)] = string(t.Value)
		}
	}
	return name, tags, field, err
}

// template represents a pattern and tags to map a graphite metric string to a influxdb Point
type template struct {
	tags              []string
	defaultTags       models.Tags
	greedyMeasurement bool
	separator         string
}

// NewTemplate returns a new template ensuring it has a measurement
// specified.
func NewTemplate(pattern string, defaultTags models.Tags, separator string) (*template, error) {
	tags := strings.Split(pattern, ".")
	hasMeasurement := false
	template := &template{tags: tags, defaultTags: defaultTags, separator: separator}

	for _, tag := range tags {
		if strings.HasPrefix(tag, "measurement") {
			hasMeasurement = true
		}
		if tag == "measurement*" {
			template.greedyMeasurement = true
		}
	}

	if !hasMeasurement {
		return nil, fmt.Errorf("no measurement specified for template. %q", pattern)
	}

	return template, nil
}

// Apply extracts the template fields from the given line and returns the measurement
// name and tags
func (t *template) Apply(line string) (string, map[string]string, string, error) {
	fields := strings.Split(line, ".")
	var (
		measurement            []string
		tags                   = make(map[string][]string)
		field                  string
		hasFieldWildcard       = false
		hasMeasurementWildcard = false
	)

	// Set any default tags
	for _, t := range t.defaultTags {
		tags[string(t.Key)] = append(tags[string(t.Key)], string(t.Value))
	}

	// See if an invalid combination has been specified in the template:
	for _, tag := range t.tags {
		if tag == "measurement*" {
			hasMeasurementWildcard = true
		} else if tag == "field*" {
			hasFieldWildcard = true
		}
	}
	if hasFieldWildcard && hasMeasurementWildcard {
		return "", nil, "", fmt.Errorf("either 'field*' or 'measurement*' can be used in each template (but not both together): %q", strings.Join(t.tags, t.separator))
	}

	for i, tag := range t.tags {
		if i >= len(fields) {
			continue
		}

		if tag == "measurement" {
			measurement = append(measurement, fields[i])
		} else if tag == "field" {
			if len(field) != 0 {
				return "", nil, "", fmt.Errorf("'field' can only be used once in each template: %q", line)
			}
			field = fields[i]
		} else if tag == "field*" {
			field = strings.Join(fields[i:], t.separator)
			break
		} else if tag == "measurement*" {
			measurement = append(measurement, fields[i:]...)
			break
		} else if tag != "" {
			tags[tag] = append(tags[tag], fields[i])
		}
	}

	// Convert to map of strings.
	out_tags := make(map[string]string)
	for k, values := range tags {
		out_tags[k] = strings.Join(values, t.separator)
	}

	return strings.Join(measurement, t.separator), out_tags, field, nil
}

// matcher determines which template should be applied to a given metric
// based on a filter tree.
type matcher struct {
	root            *node
	defaultTemplate *template
}

func newMatcher() *matcher {
	return &matcher{
		root: &node{},
	}
}

// Add inserts the template in the filter tree based the given filter
func (m *matcher) Add(filter string, template *template) {
	if filter == "" {
		m.AddDefaultTemplate(template)
		return
	}
	m.root.Insert(filter, template)
}

func (m *matcher) AddDefaultTemplate(template *template) {
	m.defaultTemplate = template
}

// Match returns the template that matches the given graphite line
func (m *matcher) Match(line string) *template {
	tmpl := m.root.Search(line)
	if tmpl != nil {
		return tmpl
	}

	return m.defaultTemplate
}

// node is an item in a sorted k-ary tree.  Each child is sorted by its value.
// The special value of "*", is always last.
type node struct {
	value    string
	children nodes
	template *template
}

func (n *node) insert(values []string, template *template) {
	// Add the end, set the template
	if len(values) == 0 {
		n.template = template
		return
	}

	// See if the the current element already exists in the tree. If so, insert the
	// into that sub-tree
	for _, v := range n.children {
		if v.value == values[0] {
			v.insert(values[1:], template)
			return
		}
	}

	// New element, add it to the tree and sort the children
	newNode := &node{value: values[0]}
	n.children = append(n.children, newNode)
	sort.Sort(&n.children)

	// Inherit template if value is wildcard
	if values[0] == "*" {
		newNode.template = n.template
	}

	// Now insert the rest of the tree into the new element
	newNode.insert(values[1:], template)
}

// Insert inserts the given string template into the tree.  The filter string is separated
// on "." and each part is used as the path in the tree.
func (n *node) Insert(filter string, template *template) {
	n.insert(strings.Split(filter, "."), template)
}

func (n *node) search(lineParts []string) *template {
	// Nothing to search
	if len(lineParts) == 0 || len(n.children) == 0 {
		return n.template
	}

	// If last element is a wildcard, don't include in this search since it's sorted
	// to the end but lexicographically it would not always be and sort.Search assumes
	// the slice is sorted.
	length := len(n.children)
	if n.children[length-1].value == "*" {
		length--
	}

	// Find the index of child with an exact match
	i := sort.Search(length, func(i int) bool {
		return n.children[i].value >= lineParts[0]
	})

	// Found an exact match, so search that child sub-tree
	if i < len(n.children) && n.children[i].value == lineParts[0] {
		return n.children[i].search(lineParts[1:])
	}
	// Not an exact match, see if we have a wildcard child to search
	if n.children[len(n.children)-1].value == "*" {
		return n.children[len(n.children)-1].search(lineParts[1:])
	}
	return n.template
}

func (n *node) Search(line string) *template {
	return n.search(strings.Split(line, "."))
}

type nodes []*node

// Less returns a boolean indicating whether the filter at position j
// is less than the filter at position k.  Filters are order by string
// comparison of each component parts.  A wildcard value "*" is never
// less than a non-wildcard value.
//
// For example, the filters:
//             "*.*"
//             "servers.*"
//             "servers.localhost"
//             "*.localhost"
//
// Would be sorted as:
//             "servers.localhost"
//             "servers.*"
//             "*.localhost"
//             "*.*"
func (n *nodes) Less(j, k int) bool {
	if (*n)[j].value == "*" && (*n)[k].value != "*" {
		return false
	}

	if (*n)[j].value != "*" && (*n)[k].value == "*" {
		return true
	}

	return (*n)[j].value < (*n)[k].value
}

func (n *nodes) Swap(i, j int) { (*n)[i], (*n)[j] = (*n)[j], (*n)[i] }
func (n *nodes) Len() int      { return len(*n) }
