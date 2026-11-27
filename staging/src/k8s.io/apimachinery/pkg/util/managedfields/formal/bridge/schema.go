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

package bridge

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
)

// The adapter walks an independently decoded schema; it never asks ToFieldSet
// which ownership the model should expect.
type schemaNode struct {
	NamedType           string      `json:"namedType,omitempty"`
	Scalar              string      `json:"scalar,omitempty"`
	Map                 *schemaMap  `json:"map,omitempty"`
	List                *schemaList `json:"list,omitempty"`
	ElementRelationship string      `json:"elementRelationship,omitempty"`
}
type schemaMap struct {
	Fields              []schemaField `json:"fields,omitempty"`
	ElementType         *schemaNode   `json:"elementType,omitempty"`
	ElementRelationship string        `json:"elementRelationship,omitempty"`
}
type schemaField struct {
	Name     string     `json:"name"`
	Type     schemaNode `json:"type"`
	Required bool       `json:"required,omitempty"`
	Default  any        `json:"default,omitempty"`
}
type schemaList struct {
	ElementType         schemaNode `json:"elementType"`
	ElementRelationship string     `json:"elementRelationship"`
	Keys                []string   `json:"keys,omitempty"`
}
type schemaType struct {
	Name string `json:"name"`
	schemaNode
}
type schemaDocument struct {
	Types []schemaType `json:"types"`
}
type structuralSchema struct {
	document schemaDocument
	types    map[string]schemaNode
	root     string
	legacy   string
}

func schemaFor(c Case) (*structuralSchema, error) {
	text, root := c.Schema, c.RootType
	if text == "" {
		text = schemaYAML
	}
	if root == "" {
		root = "Document"
	}
	var document schemaDocument
	if err := decodeStrictYAML([]byte(text), &document); err != nil {
		return nil, err
	}
	s := &structuralSchema{document: document, types: map[string]schemaNode{}, root: root}
	for index := range document.Types {
		t := &document.Types[index]
		if err := normalizeSchemaNumbers(&t.schemaNode); err != nil {
			return nil, fmt.Errorf("schema type %s: %w", t.Name, err)
		}
		if _, exists := s.types[t.Name]; exists {
			return nil, fmt.Errorf("duplicate type %s", t.Name)
		}
		s.types[t.Name] = t.schemaNode
	}
	if _, ok := s.types[root]; !ok {
		return nil, fmt.Errorf("missing root type %s", root)
	}
	rootNode, err := s.resolve(schemaNode{NamedType: root})
	if err != nil {
		return nil, err
	}
	if rootNode.Map == nil {
		return nil, fmt.Errorf("root type %s must be an object", root)
	}
	// Kubernetes keeps its API envelope even when all managed data disappears.
	for _, name := range []string{"apiVersion", "kind"} {
		found := false
		for _, field := range rootNode.Map.Fields {
			if field.Name == name {
				found = true
				break
			}
		}
		if !found {
			rootNode.Map.Fields = append(rootNode.Map.Fields, schemaField{Name: name, Type: schemaNode{Scalar: "string"}})
		}
	}
	// Decode a separate copy so required metadata remains available to the model.
	var legacy schemaDocument
	data, err := json.Marshal(document)
	if err != nil {
		return nil, err
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	if err := decoder.Decode(&legacy); err != nil {
		return nil, err
	}
	var strip func(*schemaNode)
	strip = func(n *schemaNode) {
		if n.Map != nil {
			for i := range n.Map.Fields {
				n.Map.Fields[i].Required = false
				strip(&n.Map.Fields[i].Type)
			}
			if n.Map.ElementType != nil {
				strip(n.Map.ElementType)
			}
		}
		if n.List != nil {
			strip(&n.List.ElementType)
		}
	}
	for i := range legacy.Types {
		strip(&legacy.Types[i].schemaNode)
	}
	data, err = json.Marshal(legacy)
	s.legacy = string(data)
	return s, err
}

func normalizeSchemaNumbers(node *schemaNode) error {
	if node.Map != nil {
		for index := range node.Map.Fields {
			field := &node.Map.Fields[index]
			converted, err := nativeNumbers(field.Default)
			if err != nil {
				return err
			}
			field.Default = converted
			if err := normalizeSchemaNumbers(&field.Type); err != nil {
				return err
			}
		}
		if node.Map.ElementType != nil {
			if err := normalizeSchemaNumbers(node.Map.ElementType); err != nil {
				return err
			}
		}
	}
	if node.List != nil {
		return normalizeSchemaNumbers(&node.List.ElementType)
	}
	return nil
}

func (s *structuralSchema) resolve(n schemaNode) (schemaNode, error) {
	seen := map[string]bool{}
	for n.NamedType != "" {
		name, relationship := n.NamedType, n.ElementRelationship
		if seen[name] {
			return n, fmt.Errorf("cyclic type aliases at %s", name)
		}
		seen[name] = true
		resolved, ok := s.types[name]
		if !ok {
			return n, fmt.Errorf("unresolved type %s", name)
		}
		n = resolved
		if relationship != "" {
			if n.Map != nil {
				copy := *n.Map
				copy.ElementRelationship = relationship
				n.Map = &copy
			}
			if n.List != nil {
				copy := *n.List
				copy.ElementRelationship = relationship
				n.List = &copy
			}
		}
	}
	return n, nil
}

func containsString(xs []string, x string) bool {
	for _, candidate := range xs {
		if candidate == x {
			return true
		}
	}
	return false
}
func corePair(left, right any) any {
	if left == "none" {
		return right
	}
	return map[string]any{"both": map[string]any{"left": left, "right": right}}
}

func (s *structuralSchema) requiredCore(n schemaNode, p path, keys []string, depth int) (any, error) {
	if depth > 128 {
		return nil, fmt.Errorf("required schema has an unbounded mandatory object chain at %s", p.smd())
	}
	n, err := s.resolve(n)
	if err != nil {
		return nil, err
	}
	core := any("none")
	if n.Map == nil {
		return core, nil
	}
	for _, f := range n.Map.Fields {
		if !f.Required || containsString(keys, f.Name) {
			continue
		}
		child, err := s.resolve(f.Type)
		if err != nil {
			return nil, err
		}
		fp := p.field(f.Name)
		kind := "value"
		body := map[string]any{"path": fp}
		if child.Map != nil && child.Map.ElementRelationship != "atomic" {
			kind = "object"
			children, err := s.requiredCore(child, fp, nil, depth+1)
			if err != nil {
				return nil, err
			}
			body["children"] = children
		} else if child.List != nil && child.List.ElementRelationship != "atomic" {
			kind = "collection"
		}
		core = corePair(core, map[string]any{kind: body})
	}
	return core, nil
}

// flattened keeps data and actual sparse ownership separate, including empty
// maps, unknown map entries, scalar sets, and arbitrary composite list-map keys.
type flattened struct {
	cells   []cell
	specs   []itemSpec
	owned   map[string]bool
	named   []path
	objects []itemSpec
}

func (s *structuralSchema) flatten(object map[string]any, required map[string][]string) (flattened, error) {
	out := flattened{cells: []cell{}, specs: []itemSpec{}, owned: map[string]bool{}, named: []path{}, objects: []itemSpec{}}
	add := func(p path, kind, text string, owned bool) {
		out.cells = append(out.cells, cell{p, content{kind, text}})
		if owned {
			out.owned[p.smd().String()] = true
		}
	}
	var walk func(any, schemaNode, path, string, []string, int) error
	walk = func(node any, typ schemaNode, p path, schemaPath string, keyFields []string, depth int) error {
		if depth > 256 {
			return fmt.Errorf("object depth exceeds adapter limit at %s", p.smd())
		}
		typ, err := s.resolve(typ)
		if err != nil {
			return err
		}
		atomic := typ.Map != nil && typ.Map.ElementRelationship == "atomic" || typ.List != nil && typ.List.ElementRelationship == "atomic"
		if node == nil || atomic || typ.Scalar != "" {
			text, err := json.Marshal(node)
			if err != nil {
				return err
			}
			add(p, "value", string(text), true)
			return nil
		}
		if typ.Map != nil {
			m, ok := node.(map[string]any)
			if !ok {
				return fmt.Errorf("%s: expected object, got %T", p.smd(), node)
			}
			isItem := len(p) > 0 && p[len(p)-1].Item != nil
			if !isItem {
				core, err := s.requiredCore(typ, p, nil, 0)
				if err != nil {
					return err
				}
				out.objects = append(out.objects, itemSpec{p, core})
			}
			if len(p) > 0 && !isItem {
				add(p, "object", "", len(m) == 0)
			}
			keys := make([]string, 0, len(m))
			for key := range m {
				keys = append(keys, key)
			}
			sort.Strings(keys)
			for _, key := range keys {
				if containsString(keyFields, key) {
					continue
				}
				childType := typ.Map.ElementType
				known := false
				for _, f := range typ.Map.Fields {
					if f.Name == key {
						copy := f.Type
						childType = &copy
						known = true
						break
					}
				}
				if childType == nil {
					return fmt.Errorf("%s: field %s is absent from schema", p.smd(), key)
				}
				childSchema := key
				if schemaPath != "" {
					childSchema = schemaPath + "." + key
				}
				fp := p.field(key)
				if err := walk(m[key], *childType, fp, childSchema, nil, depth+1); err != nil {
					return err
				}
				if !known {
					out.owned[fp.smd().String()] = true
				} else {
					out.named = append(out.named, fp)
				}
			}
			return nil
		}
		if typ.List != nil {
			list, ok := node.([]any)
			if !ok {
				return fmt.Errorf("%s: expected list, got %T", p.smd(), node)
			}
			add(p, "collection", "", false)
			seen := map[string]bool{}
			for _, element := range list {
				var ip path
				if len(typ.List.Keys) > 0 {
					m, ok := element.(map[string]any)
					if !ok {
						return fmt.Errorf("%s: list-map element must be an object", p.smd())
					}
					pairs := [][2]string{}
					names := append([]string{}, typ.List.Keys...)
					sort.Strings(names)
					for _, name := range names {
						v, ok := m[name]
						if !ok {
							return fmt.Errorf("%s: missing list-map key %s", p.smd(), name)
						}
						if v == nil {
							return fmt.Errorf("%s: null list-map key %s", p.smd(), name)
						}
						switch v.(type) {
						case string, bool, float64, int64, int, json.Number:
						default:
							return fmt.Errorf("%s: non-scalar key %s", p.smd(), name)
						}
						text, err := json.Marshal(v)
						if err != nil {
							return err
						}
						pairs = append(pairs, [2]string{name, string(text)})
					}
					ip = p.keys(pairs)
					core, err := s.requiredCore(typ.List.ElementType, ip, names, 0)
					if err != nil {
						return err
					}
					for _, field := range required[schemaPath] {
						core = corePair(core, map[string]any{"value": map[string]any{"path": ip.field(field)}})
					}
					out.specs = append(out.specs, itemSpec{ip, core})
					add(ip, "item", "", true)
					if err := walk(element, typ.List.ElementType, ip, schemaPath+"[]", names, depth+1); err != nil {
						return err
					}
				} else {
					switch element.(type) {
					case nil, string, bool, float64, int64, int, json.Number:
					default:
						return fmt.Errorf("%s: set elements must be scalars", p.smd())
					}
					text, err := json.Marshal(element)
					if err != nil {
						return err
					}
					ip = p.element(string(text))
					add(ip, "value", string(text), true)
				}
				key := ip.smd().String()
				if seen[key] {
					return fmt.Errorf("%s: duplicate associative key or set element", key)
				}
				seen[key] = true
			}
			return nil
		}
		return fmt.Errorf("%s: empty type reference", p.smd())
	}
	err := walk(object, schemaNode{NamedType: s.root}, path{}, "", nil, 0)
	return out, err
}
