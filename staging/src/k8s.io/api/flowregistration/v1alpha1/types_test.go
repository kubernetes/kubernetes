package v1alpha1

import (
	"bytes"
	"fmt"
	"html/template"
	"net/http"
	"testing"

	"gopkg.in/yaml.v2"
)

const fsvalueyaml = `kind: FlowSchema
meta:
  name: test-template
spec:
  requestPriority:
    name: test-template
  flowDistinguisher:
    source: user
  match:
  - and:
      {{.MatchType}}:
        field: {{.Field}}
        value: {{.Value}}
`

const fspatternyaml = `kind: FlowSchema
meta:
  name: test-template
spec:
  requestPriority:
    name: test-template
  flowDistinguisher:
    source: user
  match:
  - and:
      {{.MatchType}}:
        field: {{.Field}}
        pattern: {{.Pattern}}
`

const fssetyaml = `kind: FlowSchema
meta:
  name: test-template
spec:
  requestPriority:
    name: test-template
  flowDistinguisher:
    source: user
  match:
  - and:
      {{.MatchType}}:
        field: {{.Field}}
        set: [{{range .Set}} "{{.}}", {{end}}]
`

type FlowSchemaRunner struct {
	FlowSchema  FlowSchema
	FieldGetter FieldGetter
}

func (f *FlowSchemaRunner) ExecuteFullMatch(req *http.Request) (bool, error) {
	// iterate over matches
	matched := false
	for _, match := range f.FlowSchema.Spec.Match {
		fmatched, err := match.And.Execute(req, f.FieldGetter)
		if err != nil {
			return false, err
		}
		matched = matched || fmatched
	}
	return matched, nil
}

type TestFieldGetter struct {
	User      []string
	Groups    []string
	Namespace []string
	Resource  []string
}

func (fg *TestFieldGetter) GetField(req *http.Request, field string) (bool, []string, error) {
	switch field {
	case "user":
		return true, fg.User, nil
	case "groups":
		return true, fg.Groups, nil
	case "namespace":
		return true, fg.Namespace, nil
	case "resource":
		return true, fg.Resource, nil
	default:
		panic(field)
	}
}

type TestMatchTypeSingle struct {
	Name            string
	MatchType       string
	Field           string
	Value           string
	Pattern         string
	Set             []string
	TestFieldGetter *TestFieldGetter
	ShouldMatch     bool
}

func generateTestYaml(test TestMatchTypeSingle, yamltmpl string) string {
	tmpl, err := template.New("testequals").Parse(yamltmpl)
	if err != nil {
		panic(err)
	}

	var tpl bytes.Buffer
	err = tmpl.Execute(&tpl, test)
	if err != nil {
		panic(err)
	}
	fmt.Println(tpl.String())
	return tpl.String()
}

func yamltofsr(fsyaml string, tfg FieldGetter) FlowSchemaRunner {
	data := []byte(fsyaml)
	var fs FlowSchema
	if err := yaml.Unmarshal(data, &fs); err != nil {
		panic(err)
	}
	return FlowSchemaRunner{
		// TODO(aaron-prindle) change this to correct
		FlowSchema:  fs,
		FieldGetter: tfg,
	}

}

func TestMatchTypesSingleMatched(t *testing.T) {
	var tests = []TestMatchTypeSingle{
		// {"match all - matching namespace",
		// 	"equals", "namespace", "n1", "", []string{},
		// 	&TestFieldGetter{
		// 		Namespace: []string{"n1"},
		// 	}, true},

		{"equals/namespace - matching namespace",
			"equals", "namespace", "n1", "", []string{},
			&TestFieldGetter{
				Namespace: []string{"n1"},
			}, true},
		{"equals/namespace - not matching namespace",
			"equals", "namespace", "n1", "", []string{},
			&TestFieldGetter{
				Namespace: []string{"n2"},
			}, false},
		{"notEquals/namespace - matching namespace",
			"notEquals", "namespace", "n1", "", []string{},
			&TestFieldGetter{
				Namespace: []string{"n1"},
			}, false},

		{"inSet/Resource - matching superset of groups",
			"inSet", "resource", "", "", []string{"r1", "r2"},
			&TestFieldGetter{
				Resource: []string{"r1"},
			}, true},
		{"inSet/resource - missing group from superset",
			"inSet", "resource", "", "", []string{"r1", "r2"},
			&TestFieldGetter{
				Resource: []string{"r3"},
			}, false},
		{"notInSet/resource - !matching superset of groups",
			"notInSet", "resource", "", "", []string{"r1", "r2"},
			&TestFieldGetter{
				Resource: []string{"r1"},
			}, false},

		{"superSet/groups - matching superset of groups",
			"superSet", "groups", "", "", []string{"g1", "g2"},
			&TestFieldGetter{
				Groups: []string{"g1", "g2", "g3"},
			}, true},
		{"superSet/groups - missing group from superset",
			"superSet", "groups", "", "", []string{"g1", "g2"},
			&TestFieldGetter{
				Groups: []string{"g1", "g3", "g4"},
			}, false},
		{"notSuperSet/groups - !matching superset of groups",
			"notSuperSet", "groups", "", "", []string{"g1", "g2"},
			&TestFieldGetter{
				Groups: []string{"g1", "g2", "g4"},
			}, false},

		{"patternMatch/user - matching superset of groups",
			"patternMatch", "user", "", "system:controller:.*", []string{},
			&TestFieldGetter{
				User: []string{"system:controller:matchingpart"},
			}, true},
		{"patternMatch/user - missing group from superset",
			"patternMatch", "user", "", "system:controller:.*", []string{},
			&TestFieldGetter{
				User: []string{"notmatchingprefix"},
			}, false},
		{"notPatternMatch/user - !matching superset of groups",
			"notPatternMatch", "user", "", "system:controller:.*",
			[]string{}, &TestFieldGetter{
				User: []string{"system:controller:matchingpart"},
			}, false},
	}

	for _, test := range tests {
		var fsyaml string
		if test.Value != "" {
			fsyaml = generateTestYaml(test, fsvalueyaml)
		} else if test.Pattern != "" {
			fsyaml = generateTestYaml(test, fspatternyaml)
		} else {
			fsyaml = generateTestYaml(test, fssetyaml)
		}
		fsr := yamltofsr(fsyaml, test.TestFieldGetter)
		matched, err := fsr.ExecuteFullMatch(&http.Request{})
		if err != nil {
			panic(fmt.Sprintf("fullmatch returned error %v", err))
		}
		if matched != test.ShouldMatch {
			t.Fatalf("Test %s expected to have matched:%v, actually:%v",
				test.Name, test.ShouldMatch, matched)
		}
	}
}

const fsmultipleyaml = `kind: FlowSchema
meta:
  name: test-template
spec:
  requestPriority:
    name: test-template
  flowDistinguisher:
    source: user
  match:
  - and:
      equals:
        field: {{.EqualsField}}
        value: {{.Value}}
      inSet:
        field: {{.InSetField}}
        set: [{{range .Set}} "{{.}}", {{end}}]
`

type TestMatchTypeMultiple struct {
	Name            string
	EqualsField     string
	Value           string
	InSetField      string
	Set             []string
	TestFieldGetter *TestFieldGetter
	ShouldMatch     bool
}

func generateTestYamlMultiple(test TestMatchTypeMultiple, yamltmpl string) string {
	tmpl, err := template.New("testequals").Parse(yamltmpl)
	if err != nil {
		panic(err)
	}

	var tpl bytes.Buffer
	err = tmpl.Execute(&tpl, test)
	if err != nil {
		panic(err)
	}
	return tpl.String()
}

// tests that matches within an and are AND'd
func TestMatchTypesMultipleMatched(t *testing.T) {
	var tests = []TestMatchTypeMultiple{
		{"equals & inSet - matches both",
			"namespace", "n1", "resource", []string{"r1", "r2", "r3"},
			&TestFieldGetter{
				Namespace: []string{"n1"}, //matching
				Resource:  []string{"r1"}, // matching
			}, true},
		{"equals & inSet - matches only one",
			"namespace", "n1", "resource", []string{"r1", "r2", "r3"},
			&TestFieldGetter{
				Namespace: []string{"n1"}, //matching
				Resource:  []string{"r4"}, // not matching
			}, false},
	}

	for _, test := range tests {
		var fsyaml string
		fsyaml = generateTestYamlMultiple(test, fsmultipleyaml)
		fsr := yamltofsr(fsyaml, test.TestFieldGetter)
		matched, err := fsr.ExecuteFullMatch(&http.Request{})
		if err != nil {
			panic("fullmatch returned error")
		}

		if matched != test.ShouldMatch {
			t.Fatalf("Test %s expected to have matched:%v, actually:%v",
				test.Name, test.ShouldMatch, matched)
		}
	}
}

const fsmultipleandyaml = `kind: FlowSchema
meta:
  name: test-template
spec:
  requestPriority:
    name: test-template
  flowDistinguisher:
    source: user
  match:
  - and:
      equals:
        field: {{.EqualsField1}}
        value: {{.Value1}}
      inSet:
        field: {{.InSetField2}}
        set: [{{range .Set2}} "{{.}}", {{end}}]
  - and:
      equals:
        field: {{.EqualsField2}}
        value: {{.Value2}}
      inSet:
        field: {{.InSetField2}}
        set: [{{range .Set2}} "{{.}}", {{end}}]
`

type TestMatchTypeMultipleAnd struct {
	Name            string
	EqualsField1    string
	Value1          string
	InSetField1     string
	Set1            []string
	EqualsField2    string
	Value2          string
	InSetField2     string
	Set2            []string
	TestFieldGetter *TestFieldGetter
	ShouldMatch     bool
}

func generateTestYamlMultipleAnd(test TestMatchTypeMultipleAnd, yamltmpl string) string {
	tmpl, err := template.New("testequals").Parse(yamltmpl)
	if err != nil {
		panic(err)
	}

	var tpl bytes.Buffer
	err = tmpl.Execute(&tpl, test)
	if err != nil {
		panic(err)
	}
	return tpl.String()
}

// tests that multiple and clauses are OR'd
func TestMatchTypesMultipleAnd(t *testing.T) {
	var tests = []TestMatchTypeMultipleAnd{
		{"equals & inSet - matches both",
			"namespace", "n1", "resource", []string{"r1", "r2", "r3"},
			"namespace", "n1", "resource", []string{"r1", "r2", "r3"},
			&TestFieldGetter{
				Namespace: []string{"n1"}, //matching, matching
				Resource:  []string{"r1"}, // matching, matching
			}, true},
		{"equals & inSet - matches only one",
			"namespace", "n1", "resource", []string{"r1", "r2", "r3"},
			"namespace", "n1", "resource", []string{"r1", "r2", "r4"},
			&TestFieldGetter{
				Namespace: []string{"n1"}, //matching, matching
				Resource:  []string{"r1"}, // matching, not matching
			}, true},
	}

	for _, test := range tests {
		var fsyaml string
		fsyaml = generateTestYamlMultipleAnd(test, fsmultipleandyaml)
		fsr := yamltofsr(fsyaml, test.TestFieldGetter)
		matched, err := fsr.ExecuteFullMatch(&http.Request{})
		if err != nil {
			panic("fullmatch returned error")
		}
		if matched != test.ShouldMatch {
			t.Fatalf("Test %s expected to have matched:%v, actually:%v",
				test.Name, test.ShouldMatch, matched)
		}
	}
}
