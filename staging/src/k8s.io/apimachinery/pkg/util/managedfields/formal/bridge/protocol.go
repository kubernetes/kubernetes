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
	"strings"

	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"
	"sigs.k8s.io/structured-merge-diff/v6/value"
	"sigs.k8s.io/yaml"
)

type Corpus struct {
	Version int    `json:"version"`
	Cases   []Case `json:"cases"`
}

type Case struct {
	Name     string              `json:"name"`
	Required map[string][]string `json:"required,omitempty"`
	Steps    []Step              `json:"steps"`
}

type Step struct {
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

func (p path) item(name string) path {
	return append(append(path{}, p...), segment{Item: &struct {
		Keys [][2]string `json:"keys"`
	}{[][2]string{{"name", name}}}})
}

func (p path) smd() fieldpath.Path {
	out := fieldpath.Path{}
	for _, step := range p {
		if step.Field != nil {
			out = append(out, fieldpath.FieldNameElement(step.Field.Name))
		} else {
			fields := []value.Field{}
			for _, pair := range step.Item.Keys {
				fields = append(fields, value.Field{Name: pair[0], Value: value.NewValueInterface(pair[1])})
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
	Manager string      `json:"manager"`
	Desired []footprint `json:"desired"`
	Writes  []cell      `json:"writes"`
	Force   bool        `json:"force"`
}

type itemSpec struct {
	Path     path `json:"path"`
	Required any  `json:"required"`
}

type fixture struct {
	Name    string     `json:"name"`
	Schema  []itemSpec `json:"schema"`
	Initial state      `json:"initial"`
	Steps   []struct {
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
			if step.Difference != "" && step.DifferenceKind != "proposal" && step.DifferenceKind != "model-gap" {
				return corpus, fmt.Errorf("%s step %d: classify differenceKind as proposal or model-gap", c.Name, i+1)
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

// Flatten retains all value/container/item cells. Only ownership of container
// presence is projected out: SMD does not persist all of those Lean assertions.
func flatten(object map[string]any, required map[string][]string) ([]cell, []itemSpec, error) {
	cells, specs := []cell{}, []itemSpec{}
	var walk func(any, path, string) error
	walk = func(node any, p path, schemaPath string) error {
		if schemaPath == "atomic" {
			text, err := json.Marshal(node)
			if err != nil {
				return err
			}
			cells = append(cells, cell{p, content{"value", string(text)}})
			return nil
		}
		switch node := node.(type) {
		case map[string]any:
			isItem := len(p) > 0 && p[len(p)-1].Item != nil
			if len(p) > 0 && !isItem {
				cells = append(cells, cell{p, content{Kind: "object"}})
			}
			keys := make([]string, 0, len(node))
			for key := range node {
				keys = append(keys, key)
			}
			sort.Strings(keys)
			for _, key := range keys {
				if isItem && key == "name" {
					continue
				}
				childSchema := key
				if schemaPath != "" {
					childSchema = schemaPath + "." + key
				}
				if err := walk(node[key], p.field(key), childSchema); err != nil {
					return err
				}
			}
		case []any:
			cells = append(cells, cell{p, content{Kind: "collection"}})
			seen := map[string]bool{}
			for _, element := range node {
				object, ok := element.(map[string]any)
				if !ok {
					return fmt.Errorf("%s: only associative lists of objects are supported", p.smd())
				}
				name, ok := object["name"].(string)
				if !ok || seen[name] {
					return fmt.Errorf("%s: missing/non-string/duplicate list-map key name", p.smd())
				}
				seen[name] = true
				itemPath := p.item(name)
				core := any("none")
				for _, field := range required[schemaPath] {
					if field == "" || strings.Contains(field, ".") {
						return fmt.Errorf("%s: adapter requires direct scalar required fields", schemaPath)
					}
					leaf := map[string]any{"value": map[string]any{"path": itemPath.field(field)}}
					if core == "none" {
						core = leaf
					} else {
						core = map[string]any{"both": map[string]any{"left": core, "right": leaf}}
					}
				}
				specs = append(specs, itemSpec{itemPath, core})
				cells = append(cells, cell{itemPath, content{Kind: "item"}})
				if err := walk(object, itemPath, schemaPath+"[]"); err != nil {
					return err
				}
			}
		case string:
			text, _ := json.Marshal(node)
			cells = append(cells, cell{p, content{"value", string(text)}})
		default:
			return fmt.Errorf("%s: adapter supports strings, granular objects, associative lists, and atomic maps; got %T", p.smd(), node)
		}
		return nil
	}
	err := walk(object, nil, "")
	return cells, specs, err
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

func expectedSnapshot(e Expected) (Snapshot, error) {
	cells, _, err := flatten(e.Object, nil)
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
		if claim.Target.Kind == "presence" {
			continue
		}
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
	out := fixture{Name: c.Name, Schema: []itemSpec{}, Initial: state{[]cell{}, []claim{}}}
	known := map[string]bool{}
	for _, step := range c.Steps {
		cells, specs, err := flatten(step.Apply, c.Required)
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
		r := request{Manager: step.Manager, Desired: []footprint{}, Writes: cells, Force: step.Force}
		for _, cell := range cells {
			kind := cell.Content.Kind
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
	return out, nil
}
