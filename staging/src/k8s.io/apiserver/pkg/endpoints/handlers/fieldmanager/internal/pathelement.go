/*
Copyright 2018 The Kubernetes Authors.

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

package internal

import (
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"sigs.k8s.io/structured-merge-diff/fieldpath"
	"sigs.k8s.io/structured-merge-diff/value"
)

const (
	// Field indicates that the content of this path element is a field's name
	Field = "f"

	// Value indicates that the content of this path element is a field's value
	Value = "v"

	// Index indicates that the content of this path element is an index in an array
	Index = "i"

	// Key indicates that the content of this path element is a key value map
	Key = "k"

	// Separator separates the type of a path element from the contents
	Separator = ":"

	// LevelSeparator separates the level from the path element
	LevelSeparator = "."
)

func NewLeveledElement(s string, element *fieldpath.LeveledElement) error {
	split := strings.SplitN(s, LevelSeparator, 2)
	if level, err := strconv.Atoi(split[0]); err != nil {
		return err
	} else {
		element.Level = level
	}
	if element.Level != -1 {
		if len(split) < 2 {
			return fmt.Errorf("missing .: %v", s)
		}
		newPathElement(split[1], &element.Element)
	}
	return nil
}

func LeveledElementString(element *fieldpath.LeveledElement) (string, error) {
	pe := ""
	if element.Level != -1 {
		var err error
		pe, err = pathElementString(element.Element)
		if err != nil {
			return "", err
		}
	}
	return fmt.Sprintf("%d%s%s", element.Level, LevelSeparator, pe), nil
}

// NewPathElement parses a serialized path element
func newPathElement(s string, element *fieldpath.PathElement) error {
	split := strings.SplitN(s, Separator, 2)
	if len(split) < 2 {
		return fmt.Errorf("missing colon: %v", s)
	}
	switch split[0] {
	case Field:
		element.FieldName = &split[1]
		return nil
	case Value:
		val, err := value.FromJSON([]byte(split[1]))
		if err != nil {
			return err
		}
		element.Value = &val
		return nil
	case Index:
		i, err := strconv.Atoi(split[1])
		if err != nil {
			return err
		}
		element.Index = &i
		return nil
	case Key:
		kv := map[string]json.RawMessage{}
		err := json.Unmarshal([]byte(split[1]), &kv)
		if err != nil {
			return err
		}
		fields := []value.Field{}
		for k, v := range kv {
			b, err := json.Marshal(v)
			if err != nil {
				return err
			}
			val, err := value.FromJSON(b)
			if err != nil {
				return err
			}

			fields = append(fields, value.Field{
				Name:  k,
				Value: val,
			})
		}
		element.Key = fields
		return nil
	default:
		// Ignore unknown key types
		return nil
	}
}

// PathElementString serializes a path element
func pathElementString(pe fieldpath.PathElement) (string, error) {
	switch {
	case pe.FieldName != nil:
		return Field + Separator + *pe.FieldName, nil
	case len(pe.Key) > 0:
		kv := map[string]json.RawMessage{}
		for _, k := range pe.Key {
			b, err := k.Value.ToJSON()
			if err != nil {
				return "", err
			}
			m := json.RawMessage{}
			err = json.Unmarshal(b, &m)
			if err != nil {
				return "", err
			}
			kv[k.Name] = m
		}
		b, err := json.Marshal(kv)
		if err != nil {
			return "", err
		}
		return Key + ":" + string(b), nil
	case pe.Value != nil:
		b, err := pe.Value.ToJSON()
		if err != nil {
			return "", err
		}
		return Value + ":" + string(b), nil
	case pe.Index != nil:
		return Index + ":" + strconv.Itoa(*pe.Index), nil
	default:
		return "", errors.New("Invalid type of path element")
	}
}
