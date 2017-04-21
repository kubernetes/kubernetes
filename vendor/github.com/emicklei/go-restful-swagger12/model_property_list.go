package swagger

// Copyright 2015 Ernest Micklei. All rights reserved.
// Use of this source code is governed by a license
// that can be found in the LICENSE file.

import (
	"bytes"
	"encoding/json"
)

// NamedModelProperty associates a name to a ModelProperty
type NamedModelProperty struct {
	Name     string
	Property ModelProperty
}

// ModelPropertyList encapsulates a list of NamedModelProperty (association)
type ModelPropertyList struct {
	List []NamedModelProperty
}

// At returns the ModelPropety by its name unless absent, then ok is false
func (l *ModelPropertyList) At(name string) (p ModelProperty, ok bool) {
	for _, each := range l.List {
		if each.Name == name {
			return each.Property, true
		}
	}
	return p, false
}

// Put adds or replaces a ModelProperty with this name
func (l *ModelPropertyList) Put(name string, prop ModelProperty) {
	// maybe replace existing
	for i, each := range l.List {
		if each.Name == name {
			// replace
			l.List[i] = NamedModelProperty{Name: name, Property: prop}
			return
		}
	}
	// add
	l.List = append(l.List, NamedModelProperty{Name: name, Property: prop})
}

// Do enumerates all the properties, each with its assigned name
func (l *ModelPropertyList) Do(block func(name string, value ModelProperty)) {
	for _, each := range l.List {
		block(each.Name, each.Property)
	}
}

// MarshalJSON writes the ModelPropertyList as if it was a map[string]ModelProperty
func (l ModelPropertyList) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer
	encoder := json.NewEncoder(&buf)
	buf.WriteString("{\n")
	for i, each := range l.List {
		buf.WriteString("\"")
		buf.WriteString(each.Name)
		buf.WriteString("\": ")
		encoder.Encode(each.Property)
		if i < len(l.List)-1 {
			buf.WriteString(",\n")
		}
	}
	buf.WriteString("}")
	return buf.Bytes(), nil
}

// UnmarshalJSON reads back a ModelPropertyList. This is an expensive operation.
func (l *ModelPropertyList) UnmarshalJSON(data []byte) error {
	raw := map[string]interface{}{}
	json.NewDecoder(bytes.NewReader(data)).Decode(&raw)
	for k, v := range raw {
		// produces JSON bytes for each value
		data, err := json.Marshal(v)
		if err != nil {
			return err
		}
		var m ModelProperty
		json.NewDecoder(bytes.NewReader(data)).Decode(&m)
		l.Put(k, m)
	}
	return nil
}
