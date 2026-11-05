/*
Copyright The Kubernetes Authors.

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

// Package bridge compares the experimental Lean transition with managedfields.
// It is a test adapter, not part of the API server request path.
package bridge

import (
	"encoding/json"
	"fmt"
	"sort"

	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"
	"sigs.k8s.io/structured-merge-diff/v6/value"
	"sigs.k8s.io/yaml"
)

type Corpus struct {
	Version int    `json:"version"`
	Cases   []Case `json:"cases"`
}

type Case struct {
	Name      string              `json:"name"`
	Schema    string              `json:"schema,omitempty"`
	RootType  string              `json:"rootType,omitempty"`
	Semantics string              `json:"semantics,omitempty"`
	Bootstrap bool                `json:"bootstrap,omitempty"`
	Required  map[string][]string `json:"required,omitempty"`
	Steps     []Step              `json:"steps"`
}

type Step struct {
	Operation      string         `json:"operation,omitempty"`
	Manager        string         `json:"manager"`
	Force          bool           `json:"force,omitempty"`
	Apply          map[string]any `json:"apply"`
	Expect         *Expected      `json:"expect,omitempty"`
	GoExpect       *Expected      `json:"goExpect,omitempty"`
	Difference     string         `json:"difference,omitempty"`
	DifferenceKind string         `json:"differenceKind,omitempty"`
}

// Expected is authored independently of either engine, using ordinary object YAML.
type Expected struct {
	Error  string              `json:"error,omitempty"`
	Object map[string]any      `json:"object"`
	Owners map[string][]string `json:"owners"`
}

type Snapshot struct {
	Error  string              `json:"error"`
	Cells  map[string]string   `json:"cells"`
	Owners map[string][]string `json:"owners"`
}

type segment struct {
	Element *struct {
		Value string `json:"value"`
	} `json:"element,omitempty"`
	Field *struct {
		Name string `json:"name"`
	} `json:"field,omitempty"`
	Item *struct {
		Keys [][2]string `json:"keys"`
	} `json:"item,omitempty"`
}

type path []segment

func (p path) field(name string) path {
	return append(append(path{}, p...), segment{Field: &struct {
		Name string `json:"name"`
	}{name}})
}

func (p path) keys(keys [][2]string) path {
	return append(append(path{}, p...), segment{Item: &struct {
		Keys [][2]string `json:"keys"`
	}{keys}})
}
func (p path) item(name string) path {
	text, _ := json.Marshal(name)
	return p.keys([][2]string{{"name", string(text)}})
}
func (p path) element(text string) path {
	return append(append(path{}, p...), segment{Element: &struct {
		Value string `json:"value"`
	}{text}})
}
func jsonValue(text string) any {
	var v any
	if err := json.Unmarshal([]byte(text), &v); err != nil {
		panic(err)
	}
	return v
}

func (p path) smd() fieldpath.Path {
	out := fieldpath.Path{}
	for _, step := range p {
		if step.Field != nil {
			out = append(out, fieldpath.FieldNameElement(step.Field.Name))
		} else if step.Element != nil {
			out = append(out, fieldpath.ValueElement(value.NewValueInterface(jsonValue(step.Element.Value))))
		} else {
			fields := []value.Field{}
			for _, pair := range step.Item.Keys {
				fields = append(fields, value.Field{Name: pair[0], Value: value.NewValueInterface(jsonValue(pair[1]))})
			}
			out = append(out, fieldpath.KeyElement(fields...))
		}
	}
	return out
}

type content struct {
	Kind string
	Text string
}

func (c content) MarshalJSON() ([]byte, error) {
	if c.Kind == "value" {
		return json.Marshal(map[string]any{"value": map[string]string{"text": c.Text}})
	}
	return json.Marshal(c.Kind)
}

func (c *content) UnmarshalJSON(data []byte) error {
	if err := json.Unmarshal(data, &c.Kind); err == nil {
		if c.Kind != "object" && c.Kind != "collection" && c.Kind != "item" {
			return fmt.Errorf("unknown model content %q", c.Kind)
		}
		return nil
	}
	var v struct {
		Value *struct {
			Text string `json:"text"`
		} `json:"value"`
	}
	if err := json.Unmarshal(data, &v); err != nil {
		return err
	}
	if v.Value == nil {
		return fmt.Errorf("invalid model content: %s", data)
	}
	c.Kind, c.Text = "value", v.Value.Text
	return nil
}

type cell struct {
	Path    path    `json:"path"`
	Content content `json:"content"`
}

type footprint struct {
	Path path   `json:"path"`
	Kind string `json:"kind"`
}

type claim struct {
	Manager string    `json:"manager"`
	Target  footprint `json:"target"`
}

type state struct {
	Cells  []cell  `json:"cells"`
	Claims []claim `json:"claims"`
}

type request struct {
	Named     []path      `json:"named"`
	Operation string      `json:"operation"`
	Manager   string      `json:"manager"`
	Desired   []footprint `json:"desired"`
	Writes    []cell      `json:"writes"`
	Force     bool        `json:"force"`
	Context   []cell      `json:"context"`
}

type itemSpec struct {
	Path     path `json:"path"`
	Required any  `json:"required"`
}

type fixture struct {
	Objects   []itemSpec `json:"objects"`
	Bootstrap bool       `json:"bootstrap"`
	Name      string     `json:"name"`
	Semantics string     `json:"semantics"`
	Schema    []itemSpec `json:"schema"`
	Initial   state      `json:"initial"`
	Steps     []struct {
		Request  request        `json:"request"`
		Expected map[string]any `json:"expected"`
	} `json:"steps"`
}

func ReadCorpus(data []byte) (Corpus, error) {
	return readCorpus(data, false)
}

// ReadReproducer permits generated request prefixes without golden outcomes.
// Documented exceptions still require both complete, independent expectations.
func ReadReproducer(data []byte) (Corpus, error) {
	return readCorpus(data, true)
}

func readCorpus(data []byte, allowMissingOracles bool) (Corpus, error) {
	var corpus Corpus
	if err := yaml.UnmarshalStrict(data, &corpus); err != nil {
		return corpus, err
	}
	if corpus.Version != 1 || len(corpus.Cases) == 0 {
		return corpus, fmt.Errorf("corpus must have version 1 and at least one case")
	}
	names := map[string]bool{}
	for _, c := range corpus.Cases {
		if c.Name == "" || names[c.Name] || len(c.Steps) == 0 {
			return corpus, fmt.Errorf("empty/duplicate case name or empty steps: %q", c.Name)
		}
		names[c.Name] = true
		for list, fields := range c.Required {
			allowed := map[string]bool{"host": true, "port": true, "note": true}
			if list == "items[].children" {
				allowed = map[string]bool{"url": true, "note": true}
			} else if list != "items" {
				return corpus, fmt.Errorf("%s: unsupported required-list schema path %q", c.Name, list)
			}
			seen := map[string]bool{}
			for _, field := range fields {
				if !allowed[field] || seen[field] {
					return corpus, fmt.Errorf("%s: invalid/duplicate required scalar %q in %s", c.Name, field, list)
				}
				seen[field] = true
			}
		}
		for i, step := range c.Steps {
			if step.Manager == "" || step.Apply == nil {
				return corpus, fmt.Errorf("%s step %d needs manager and apply", c.Name, i+1)
			}
			if step.Expect == nil {
				if !allowMissingOracles || step.GoExpect != nil || step.Difference != "" {
					return corpus, fmt.Errorf("%s step %d needs an independent expected result", c.Name, i+1)
				}
			} else if step.Expect.Object == nil || step.Expect.Owners == nil {
				return corpus, fmt.Errorf("%s step %d needs manager, apply, and complete expected object/owners", c.Name, i+1)
			}
			if (step.GoExpect != nil) != (step.Difference != "") {
				return corpus, fmt.Errorf("%s step %d: goExpect and difference must occur together", c.Name, i+1)
			}
			if step.Difference != "" && step.DifferenceKind != "proposal" {
				return corpus, fmt.Errorf("%s step %d: classify differenceKind as proposal", c.Name, i+1)
			}
			if step.Difference == "" && step.DifferenceKind != "" {
				return corpus, fmt.Errorf("%s step %d: differenceKind needs a documented difference", c.Name, i+1)
			}
			if step.GoExpect != nil && (step.GoExpect.Object == nil || step.GoExpect.Owners == nil) {
				return corpus, fmt.Errorf("%s step %d: goExpect needs complete object/owners", c.Name, i+1)
			}
		}
	}
	return corpus, nil
}

// flatten is the default-schema convenience used by adapter tests.
func flatten(object map[string]any, required map[string][]string) ([]cell, []itemSpec, error) {
	s, err := schemaFor(Case{})
	if err != nil {
		return nil, nil, err
	}
	out, err := s.flatten(object, required)
	return out.cells, out.specs, err
}

func cellMap(cells []cell) (map[string]string, error) {
	out := map[string]string{}
	for _, cell := range cells {
		key := cell.Path.smd().String()
		if _, exists := out[key]; exists {
			return nil, fmt.Errorf("duplicate model cell %s", key)
		}
		out[key] = cell.Content.Kind
		if cell.Content.Kind == "value" {
			out[key] += ":" + cell.Content.Text
		}
	}
	return out, nil
}

func expectedSnapshot(e Expected) (Snapshot, error) { return expectedSnapshotFor(e, Case{}) }
func expectedSnapshotFor(e Expected, c Case) (Snapshot, error) {
	schema, err := schemaFor(c)
	if err != nil {
		return Snapshot{}, err
	}
	flat, err := schema.flatten(e.Object, nil)
	cells := flat.cells
	if err != nil {
		return Snapshot{}, err
	}
	mapped, err := cellMap(cells)
	out := Snapshot{e.Error, mapped, map[string][]string{}}
	for manager, fields := range e.Owners {
		if len(fields) == 0 {
			return out, fmt.Errorf("empty owner %q must be omitted", manager)
		}
		out.Owners[manager] = append([]string{}, fields...)
		sort.Strings(out.Owners[manager])
		for i := 1; i < len(fields); i++ {
			if out.Owners[manager][i] == out.Owners[manager][i-1] {
				return out, fmt.Errorf("duplicate expected claim for %s", manager)
			}
		}
	}
	return out, err
}

func modelSnapshot(s state, errorName string) (Snapshot, error) {
	cells, err := cellMap(s.Cells)
	out := Snapshot{errorName, cells, map[string][]string{}}
	seen := map[string]bool{}
	for _, claim := range s.Claims {
		field := claim.Target.Path.smd().String()
		key := claim.Manager + ":" + field
		if seen[key] {
			return out, fmt.Errorf("duplicate model claim %s", key)
		}
		seen[key] = true
		out.Owners[claim.Manager] = append(out.Owners[claim.Manager], field)
	}
	for _, fields := range out.Owners {
		sort.Strings(fields)
	}
	return out, err
}

func normalize(c Case) (fixture, error) {
	mode := c.Semantics
	if mode == "" {
		if len(c.Required) > 0 {
			mode = "full"
		} else {
			mode = "current"
		}
	}
	if mode != "full" && mode != "current" && mode != "predrag" {
		return fixture{}, fmt.Errorf("unknown semantics %s", mode)
	}
	if c.Bootstrap {
		if mode != "current" {
			return fixture{}, fmt.Errorf("bootstrap comparison requires current Kubernetes semantics")
		}
		for _, step := range c.Steps {
			if step.Operation != "" && step.Operation != "apply" {
				return fixture{}, fmt.Errorf("bootstrap comparison supports Apply histories; probabilistic Update tracking is outside this model")
			}
		}
	}
	schema, err := schemaFor(c)
	if err != nil {
		return fixture{}, err
	}
	out := fixture{Name: c.Name, Semantics: mode, Bootstrap: c.Bootstrap, Schema: []itemSpec{}, Objects: []itemSpec{}, Initial: state{[]cell{}, []claim{}}}
	known := map[string]bool{}
	objectSeen := map[string]bool{}
	named := []path{}
	namedSeen := map[string]bool{}
	for _, step := range c.Steps {
		flat, err := schema.flatten(step.Apply, c.Required)
		cells, specs := flat.cells, flat.specs
		for _, spec := range flat.objects {
			key := spec.Path.smd().String()
			if !objectSeen[key] {
				out.Objects = append(out.Objects, spec)
				objectSeen[key] = true
			}
		}
		for _, p := range flat.named {
			key := p.smd().String()
			if !namedSeen[key] {
				named = append(named, p)
				namedSeen[key] = true
			}
		}
		if err != nil {
			return out, err
		}
		for _, spec := range specs {
			key := spec.Path.smd().String()
			if !known[key] {
				out.Schema = append(out.Schema, spec)
				known[key] = true
			}
		}
		operation := step.Operation
		if operation == "" {
			operation = "apply"
		}
		manager := step.Manager
		if operation == "update" {
			manager += "@update"
		}
		if operation != "apply" && operation != "update" {
			return out, fmt.Errorf("unknown operation %s", operation)
		}
		r := request{Operation: operation, Manager: manager, Desired: []footprint{}, Writes: []cell{}, Context: []cell{}, Force: step.Force}
		for _, cell := range cells {
			kind := cell.Content.Kind
			implicit := !flat.owned[cell.Path.smd().String()]
			if implicit {
				r.Context = append(r.Context, cell)
				continue
			}
			r.Writes = append(r.Writes, cell)
			if kind == "object" || kind == "collection" {
				kind = "presence"
			}
			r.Desired = append(r.Desired, footprint{cell.Path, kind})
		}
		out.Steps = append(out.Steps, struct {
			Request  request        `json:"request"`
			Expected map[string]any `json:"expected"`
		}{r, map[string]any{
			"error": nil, "present": []any{}, "absent": []any{},
			"fullManagers": []any{}, "partialManagers": []any{},
		}})
	}
	for i := range out.Steps {
		out.Steps[i].Request.Named = named
	}
	return out, nil
}
