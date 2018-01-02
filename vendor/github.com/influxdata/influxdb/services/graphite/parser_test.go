package graphite_test

import (
	"reflect"
	"strconv"
	"testing"
	"time"

	"github.com/influxdata/influxdb/models"
	"github.com/influxdata/influxdb/services/graphite"
)

func BenchmarkParse(b *testing.B) {
	p, err := graphite.NewParser([]string{
		"*.* .wrong.measurement*",
		"servers.* .host.measurement*",
		"servers.localhost .host.measurement*",
		"*.localhost .host.measurement*",
		"*.*.cpu .host.measurement*",
		"a.b.c .host.measurement*",
		"influxd.*.foo .host.measurement*",
		"prod.*.mem .host.measurement*",
	}, nil)

	if err != nil {
		b.Fatalf("unexpected error creating parser, got %v", err)
	}

	for i := 0; i < b.N; i++ {
		p.Parse("servers.localhost.cpu.load 11 1435077219")
	}
}

func TestTemplateApply(t *testing.T) {
	var tests = []struct {
		test        string
		input       string
		template    string
		measurement string
		tags        map[string]string
		err         string
	}{
		{
			test:        "metric only",
			input:       "cpu",
			template:    "measurement",
			measurement: "cpu",
		},
		{
			test:        "metric with single series",
			input:       "cpu.server01",
			template:    "measurement.hostname",
			measurement: "cpu",
			tags:        map[string]string{"hostname": "server01"},
		},
		{
			test:        "metric with multiple series",
			input:       "cpu.us-west.server01",
			template:    "measurement.region.hostname",
			measurement: "cpu",
			tags:        map[string]string{"hostname": "server01", "region": "us-west"},
		},
		{
			test:        "metric with multiple tags",
			input:       "server01.example.org.cpu.us-west",
			template:    "hostname.hostname.hostname.measurement.region",
			measurement: "cpu",
			tags:        map[string]string{"hostname": "server01.example.org", "region": "us-west"},
		},
		{
			test: "no metric",
			tags: make(map[string]string),
			err:  `no measurement specified for template. ""`,
		},
		{
			test:        "ignore unnamed",
			input:       "foo.cpu",
			template:    "measurement",
			measurement: "foo",
			tags:        make(map[string]string),
		},
		{
			test:        "name shorter than template",
			input:       "foo",
			template:    "measurement.A.B.C",
			measurement: "foo",
			tags:        make(map[string]string),
		},
		{
			test:        "wildcard measurement at end",
			input:       "prod.us-west.server01.cpu.load",
			template:    "env.zone.host.measurement*",
			measurement: "cpu.load",
			tags:        map[string]string{"env": "prod", "zone": "us-west", "host": "server01"},
		},
		{
			test:        "skip fields",
			input:       "ignore.us-west.ignore-this-too.cpu.load",
			template:    ".zone..measurement*",
			measurement: "cpu.load",
			tags:        map[string]string{"zone": "us-west"},
		},
		{
			test:        "conjoined fields",
			input:       "prod.us-west.server01.cpu.util.idle.percent",
			template:    "env.zone.host.measurement.measurement.field*",
			measurement: "cpu.util",
			tags:        map[string]string{"env": "prod", "zone": "us-west", "host": "server01"},
		},
	}

	for _, test := range tests {
		tmpl, err := graphite.NewTemplate(test.template, nil, graphite.DefaultSeparator)
		if errstr(err) != test.err {
			t.Fatalf("err does not match.  expected %v, got %v", test.err, err)
		}
		if err != nil {
			// If we erred out,it was intended and the following tests won't work
			continue
		}

		measurement, tags, _, _ := tmpl.Apply(test.input)
		if measurement != test.measurement {
			t.Fatalf("name parse failer.  expected %v, got %v", test.measurement, measurement)
		}
		if len(tags) != len(test.tags) {
			t.Fatalf("unexpected number of tags.  expected %v, got %v", test.tags, tags)
		}
		for k, v := range test.tags {
			if tags[k] != v {
				t.Fatalf("unexpected tag value for tags[%s].  expected %q, got %q", k, v, tags[k])
			}
		}
	}
}

func TestParseMissingMeasurement(t *testing.T) {
	_, err := graphite.NewParser([]string{"a.b.c"}, nil)
	if err == nil {
		t.Fatalf("expected error creating parser, got nil")
	}
}

func TestParse(t *testing.T) {
	testTime := time.Now().Round(time.Second)
	epochTime := testTime.Unix()
	strTime := strconv.FormatInt(epochTime, 10)

	var tests = []struct {
		test        string
		input       string
		measurement string
		tags        map[string]string
		value       float64
		time        time.Time
		template    string
		err         string
	}{
		{
			test:        "normal case",
			input:       `cpu.foo.bar 50 ` + strTime,
			template:    "measurement.foo.bar",
			measurement: "cpu",
			tags: map[string]string{
				"foo": "foo",
				"bar": "bar",
			},
			value: 50,
			time:  testTime,
		},
		{
			test:        "metric only with float value",
			input:       `cpu 50.554 ` + strTime,
			measurement: "cpu",
			template:    "measurement",
			value:       50.554,
			time:        testTime,
		},
		{
			test:     "missing metric",
			input:    `1419972457825`,
			template: "measurement",
			err:      `received "1419972457825" which doesn't have required fields`,
		},
		{
			test:     "should error parsing invalid float",
			input:    `cpu 50.554z 1419972457825`,
			template: "measurement",
			err:      `field "cpu" value: strconv.ParseFloat: parsing "50.554z": invalid syntax`,
		},
		{
			test:     "should error parsing invalid int",
			input:    `cpu 50z 1419972457825`,
			template: "measurement",
			err:      `field "cpu" value: strconv.ParseFloat: parsing "50z": invalid syntax`,
		},
		{
			test:     "should error parsing invalid time",
			input:    `cpu 50.554 14199724z57825`,
			template: "measurement",
			err:      `field "cpu" time: strconv.ParseFloat: parsing "14199724z57825": invalid syntax`,
		},
		{
			test:     "measurement* and field* (invalid)",
			input:    `prod.us-west.server01.cpu.util.idle.percent 99.99 1419972457825`,
			template: "env.zone.host.measurement*.field*",
			err:      `either 'field*' or 'measurement*' can be used in each template (but not both together): "env.zone.host.measurement*.field*"`,
		},
	}

	for _, test := range tests {
		p, err := graphite.NewParser([]string{test.template}, nil)
		if err != nil {
			t.Fatalf("unexpected error creating graphite parser: %v", err)
		}

		point, err := p.Parse(test.input)
		if errstr(err) != test.err {
			t.Fatalf("err does not match.  expected %v, got %v", test.err, err)
		}
		if err != nil {
			// If we erred out,it was intended and the following tests won't work
			continue
		}
		if point.Name() != test.measurement {
			t.Fatalf("name parse failer.  expected %v, got %v", test.measurement, point.Name())
		}
		if len(point.Tags()) != len(test.tags) {
			t.Fatalf("tags len mismatch.  expected %d, got %d", len(test.tags), len(point.Tags()))
		}
		f := point.Fields()["value"].(float64)
		if point.Fields()["value"] != f {
			t.Fatalf("floatValue value mismatch.  expected %v, got %v", test.value, f)
		}
		if point.Time().UnixNano()/1000000 != test.time.UnixNano()/1000000 {
			t.Fatalf("time value mismatch.  expected %v, got %v", test.time.UnixNano(), point.Time().UnixNano())
		}
	}
}

func TestParseNaN(t *testing.T) {
	p, err := graphite.NewParser([]string{"measurement*"}, nil)
	if err != nil {
		t.Fatalf("unexpected error creating parser, got %v", err)
	}

	_, err = p.Parse("servers.localhost.cpu_load NaN 1435077219")
	if err == nil {
		t.Fatalf("expected error. got nil")
	}

	if _, ok := err.(*graphite.UnsupportedValueError); !ok {
		t.Fatalf("expected *graphite.ErrUnsupportedValue, got %v", reflect.TypeOf(err))
	}
}

func TestFilterMatchDefault(t *testing.T) {
	p, err := graphite.NewParser([]string{"servers.localhost .host.measurement*"}, nil)
	if err != nil {
		t.Fatalf("unexpected error creating parser, got %v", err)
	}

	exp := models.MustNewPoint("miss.servers.localhost.cpu_load",
		models.NewTags(map[string]string{}),
		models.Fields{"value": float64(11)},
		time.Unix(1435077219, 0))

	pt, err := p.Parse("miss.servers.localhost.cpu_load 11 1435077219")
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	if exp.String() != pt.String() {
		t.Errorf("parse mismatch: got %v, exp %v", pt.String(), exp.String())
	}
}

func TestFilterMatchMultipleMeasurement(t *testing.T) {
	p, err := graphite.NewParser([]string{"servers.localhost .host.measurement.measurement*"}, nil)
	if err != nil {
		t.Fatalf("unexpected error creating parser, got %v", err)
	}

	exp := models.MustNewPoint("cpu.cpu_load.10",
		models.NewTags(map[string]string{"host": "localhost"}),
		models.Fields{"value": float64(11)},
		time.Unix(1435077219, 0))

	pt, err := p.Parse("servers.localhost.cpu.cpu_load.10 11 1435077219")
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	if exp.String() != pt.String() {
		t.Errorf("parse mismatch: got %v, exp %v", pt.String(), exp.String())
	}
}

func TestFilterMatchMultipleMeasurementSeparator(t *testing.T) {
	p, err := graphite.NewParserWithOptions(graphite.Options{
		Templates: []string{"servers.localhost .host.measurement.measurement*"},
		Separator: "_",
	})
	if err != nil {
		t.Fatalf("unexpected error creating parser, got %v", err)
	}

	exp := models.MustNewPoint("cpu_cpu_load_10",
		models.NewTags(map[string]string{"host": "localhost"}),
		models.Fields{"value": float64(11)},
		time.Unix(1435077219, 0))

	pt, err := p.Parse("servers.localhost.cpu.cpu_load.10 11 1435077219")
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	if exp.String() != pt.String() {
		t.Errorf("parse mismatch: got %v, exp %v", pt.String(), exp.String())
	}
}

func TestFilterMatchSingle(t *testing.T) {
	p, err := graphite.NewParser([]string{"servers.localhost .host.measurement*"}, nil)
	if err != nil {
		t.Fatalf("unexpected error creating parser, got %v", err)
	}

	exp := models.MustNewPoint("cpu_load",
		models.NewTags(map[string]string{"host": "localhost"}),
		models.Fields{"value": float64(11)},
		time.Unix(1435077219, 0))

	pt, err := p.Parse("servers.localhost.cpu_load 11 1435077219")
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	if exp.String() != pt.String() {
		t.Errorf("parse mismatch: got %v, exp %v", pt.String(), exp.String())
	}
}

func TestParseNoMatch(t *testing.T) {
	p, err := graphite.NewParser([]string{"servers.*.cpu .host.measurement.cpu.measurement"}, nil)
	if err != nil {
		t.Fatalf("unexpected error creating parser, got %v", err)
	}

	exp := models.MustNewPoint("servers.localhost.memory.VmallocChunk",
		models.NewTags(map[string]string{}),
		models.Fields{"value": float64(11)},
		time.Unix(1435077219, 0))

	pt, err := p.Parse("servers.localhost.memory.VmallocChunk 11 1435077219")
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	if exp.String() != pt.String() {
		t.Errorf("parse mismatch: got %v, exp %v", pt.String(), exp.String())
	}
}

func TestFilterMatchWildcard(t *testing.T) {
	p, err := graphite.NewParser([]string{"servers.* .host.measurement*"}, nil)
	if err != nil {
		t.Fatalf("unexpected error creating parser, got %v", err)
	}

	exp := models.MustNewPoint("cpu_load",
		models.NewTags(map[string]string{"host": "localhost"}),
		models.Fields{"value": float64(11)},
		time.Unix(1435077219, 0))

	pt, err := p.Parse("servers.localhost.cpu_load 11 1435077219")
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	if exp.String() != pt.String() {
		t.Errorf("parse mismatch: got %v, exp %v", pt.String(), exp.String())
	}
}

func TestFilterMatchExactBeforeWildcard(t *testing.T) {
	p, err := graphite.NewParser([]string{
		"servers.* .wrong.measurement*",
		"servers.localhost .host.measurement*"}, nil)
	if err != nil {
		t.Fatalf("unexpected error creating parser, got %v", err)
	}

	exp := models.MustNewPoint("cpu_load",
		models.NewTags(map[string]string{"host": "localhost"}),
		models.Fields{"value": float64(11)},
		time.Unix(1435077219, 0))

	pt, err := p.Parse("servers.localhost.cpu_load 11 1435077219")
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	if exp.String() != pt.String() {
		t.Errorf("parse mismatch: got %v, exp %v", pt.String(), exp.String())
	}
}

func TestFilterMatchMostLongestFilter(t *testing.T) {
	p, err := graphite.NewParser([]string{
		"*.* .wrong.measurement*",
		"servers.* .wrong.measurement*",
		"servers.localhost .wrong.measurement*",
		"servers.localhost.cpu .host.resource.measurement*", // should match this
		"*.localhost .wrong.measurement*",
	}, nil)

	if err != nil {
		t.Fatalf("unexpected error creating parser, got %v", err)
	}

	exp := models.MustNewPoint("cpu_load",
		models.NewTags(map[string]string{"host": "localhost", "resource": "cpu"}),
		models.Fields{"value": float64(11)},
		time.Unix(1435077219, 0))

	pt, err := p.Parse("servers.localhost.cpu.cpu_load 11 1435077219")
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	if exp.String() != pt.String() {
		t.Errorf("parse mismatch: got %v, exp %v", pt.String(), exp.String())
	}
}

func TestFilterMatchMultipleWildcards(t *testing.T) {
	p, err := graphite.NewParser([]string{
		"*.* .wrong.measurement*",
		"servers.* .host.measurement*", // should match this
		"servers.localhost .wrong.measurement*",
		"*.localhost .wrong.measurement*",
	}, nil)

	if err != nil {
		t.Fatalf("unexpected error creating parser, got %v", err)
	}

	exp := models.MustNewPoint("cpu_load",
		models.NewTags(map[string]string{"host": "server01"}),
		models.Fields{"value": float64(11)},
		time.Unix(1435077219, 0))

	pt, err := p.Parse("servers.server01.cpu_load 11 1435077219")
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	if exp.String() != pt.String() {
		t.Errorf("parse mismatch: got %v, exp %v", pt.String(), exp.String())
	}
}

func TestParseDefaultTags(t *testing.T) {
	p, err := graphite.NewParser([]string{"servers.localhost .host.measurement*"}, models.NewTags(map[string]string{
		"region": "us-east",
		"zone":   "1c",
		"host":   "should not set",
	}))
	if err != nil {
		t.Fatalf("unexpected error creating parser, got %v", err)
	}

	exp := models.MustNewPoint("cpu_load",
		models.NewTags(map[string]string{"host": "localhost", "region": "us-east", "zone": "1c"}),
		models.Fields{"value": float64(11)},
		time.Unix(1435077219, 0))

	pt, err := p.Parse("servers.localhost.cpu_load 11 1435077219")
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	if exp.String() != pt.String() {
		t.Errorf("parse mismatch: got %v, exp %v", pt.String(), exp.String())
	}
}

func TestParseDefaultTemplateTags(t *testing.T) {
	p, err := graphite.NewParser([]string{"servers.localhost .host.measurement* zone=1c"}, models.NewTags(map[string]string{
		"region": "us-east",
		"host":   "should not set",
	}))
	if err != nil {
		t.Fatalf("unexpected error creating parser, got %v", err)
	}

	exp := models.MustNewPoint("cpu_load",
		models.NewTags(map[string]string{"host": "localhost", "region": "us-east", "zone": "1c"}),
		models.Fields{"value": float64(11)},
		time.Unix(1435077219, 0))

	pt, err := p.Parse("servers.localhost.cpu_load 11 1435077219")
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	if exp.String() != pt.String() {
		t.Errorf("parse mismatch: got %v, exp %v", pt.String(), exp.String())
	}
}

func TestParseDefaultTemplateTagsOverridGlobal(t *testing.T) {
	p, err := graphite.NewParser([]string{"servers.localhost .host.measurement* zone=1c,region=us-east"}, models.NewTags(map[string]string{
		"region": "shot not be set",
		"host":   "should not set",
	}))
	if err != nil {
		t.Fatalf("unexpected error creating parser, got %v", err)
	}

	exp := models.MustNewPoint("cpu_load",
		models.NewTags(map[string]string{"host": "localhost", "region": "us-east", "zone": "1c"}),
		models.Fields{"value": float64(11)},
		time.Unix(1435077219, 0))

	pt, err := p.Parse("servers.localhost.cpu_load 11 1435077219")
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	if exp.String() != pt.String() {
		t.Errorf("parse mismatch: got %v, exp %v", pt.String(), exp.String())
	}
}

func TestParseTemplateWhitespace(t *testing.T) {
	p, err := graphite.NewParser([]string{"servers.localhost        .host.measurement*           zone=1c"}, models.NewTags(map[string]string{
		"region": "us-east",
		"host":   "should not set",
	}))
	if err != nil {
		t.Fatalf("unexpected error creating parser, got %v", err)
	}

	exp := models.MustNewPoint("cpu_load",
		models.NewTags(map[string]string{"host": "localhost", "region": "us-east", "zone": "1c"}),
		models.Fields{"value": float64(11)},
		time.Unix(1435077219, 0))

	pt, err := p.Parse("servers.localhost.cpu_load 11 1435077219")
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	if exp.String() != pt.String() {
		t.Errorf("parse mismatch: got %v, exp %v", pt.String(), exp.String())
	}
}

// Test basic functionality of ApplyTemplate
func TestApplyTemplate(t *testing.T) {
	o := graphite.Options{
		Separator: "_",
		Templates: []string{"current.* measurement.measurement"},
	}
	p, err := graphite.NewParserWithOptions(o)
	if err != nil {
		t.Fatalf("unexpected error creating parser, got %v", err)
	}

	measurement, _, _, _ := p.ApplyTemplate("current.users")
	if measurement != "current_users" {
		t.Errorf("Parser.ApplyTemplate unexpected result. got %s, exp %s",
			measurement, "current_users")
	}
}

// Test basic functionality of ApplyTemplate
func TestApplyTemplateNoMatch(t *testing.T) {
	o := graphite.Options{
		Separator: "_",
		Templates: []string{"foo.bar measurement.measurement"},
	}
	p, err := graphite.NewParserWithOptions(o)
	if err != nil {
		t.Fatalf("unexpected error creating parser, got %v", err)
	}

	measurement, _, _, _ := p.ApplyTemplate("current.users")
	if measurement != "current.users" {
		t.Errorf("Parser.ApplyTemplate unexpected result. got %s, exp %s",
			measurement, "current.users")
	}
}

// Test that most specific template is chosen
func TestApplyTemplateSpecific(t *testing.T) {
	o := graphite.Options{
		Separator: "_",
		Templates: []string{
			"current.* measurement.measurement",
			"current.*.* measurement.measurement.service",
		},
	}
	p, err := graphite.NewParserWithOptions(o)
	if err != nil {
		t.Fatalf("unexpected error creating parser, got %v", err)
	}

	measurement, tags, _, _ := p.ApplyTemplate("current.users.facebook")
	if measurement != "current_users" {
		t.Errorf("Parser.ApplyTemplate unexpected result. got %s, exp %s",
			measurement, "current_users")
	}
	service, ok := tags["service"]
	if !ok {
		t.Error("Expected for template to apply a 'service' tag, but not found")
	}
	if service != "facebook" {
		t.Errorf("Expected service='facebook' tag, got service='%s'", service)
	}
}

// Test that most specific template is N/A
func TestApplyTemplateSpecificIsNA(t *testing.T) {
	o := graphite.Options{
		Separator: "_",
		Templates: []string{
			"current.* measurement.service",
			"current.*.*.test measurement.measurement.service",
		},
	}
	p, err := graphite.NewParserWithOptions(o)
	if err != nil {
		t.Fatalf("unexpected error creating parser, got %v", err)
	}

	measurement, _, _, _ := p.ApplyTemplate("current.users.facebook")
	if measurement != "current" {
		t.Errorf("Parser.ApplyTemplate unexpected result. got %s, exp %s",
			measurement, "current")
	}
}

func TestApplyTemplateTags(t *testing.T) {
	o := graphite.Options{
		Separator: "_",
		Templates: []string{"current.* measurement.measurement region=us-west"},
	}
	p, err := graphite.NewParserWithOptions(o)
	if err != nil {
		t.Fatalf("unexpected error creating parser, got %v", err)
	}

	measurement, tags, _, _ := p.ApplyTemplate("current.users")
	if measurement != "current_users" {
		t.Errorf("Parser.ApplyTemplate unexpected result. got %s, exp %s",
			measurement, "current_users")
	}

	region, ok := tags["region"]
	if !ok {
		t.Error("Expected for template to apply a 'region' tag, but not found")
	}
	if region != "us-west" {
		t.Errorf("Expected region='us-west' tag, got region='%s'", region)
	}
}

func TestApplyTemplateField(t *testing.T) {
	o := graphite.Options{
		Separator: "_",
		Templates: []string{"current.* measurement.measurement.field"},
	}
	p, err := graphite.NewParserWithOptions(o)
	if err != nil {
		t.Fatalf("unexpected error creating parser, got %v", err)
	}

	measurement, _, field, err := p.ApplyTemplate("current.users.logged_in")

	if measurement != "current_users" {
		t.Errorf("Parser.ApplyTemplate unexpected result. got %s, exp %s",
			measurement, "current_users")
	}

	if field != "logged_in" {
		t.Errorf("Parser.ApplyTemplate unexpected result. got %s, exp %s",
			field, "logged_in")
	}
}

func TestApplyTemplateFieldError(t *testing.T) {
	o := graphite.Options{
		Separator: "_",
		Templates: []string{"current.* measurement.field.field"},
	}
	p, err := graphite.NewParserWithOptions(o)
	if err != nil {
		t.Fatalf("unexpected error creating parser, got %v", err)
	}

	_, _, _, err = p.ApplyTemplate("current.users.logged_in")
	if err == nil {
		t.Errorf("Parser.ApplyTemplate unexpected result. got %s, exp %s", err,
			"'field' can only be used once in each template: current.users.logged_in")
	}
}

// Test Helpers
func errstr(err error) string {
	if err != nil {
		return err.Error()
	}
	return ""
}
