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

package apply

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
	Field = "f"
	Value = "v"
	Index = "i"
	Key   = "k"

	Separator = ":"
)

func NewPathElement(s string) (fieldpath.PathElement, error) {
	split := strings.SplitN(s, Separator, 2)
	if len(split) < 2 {
		return fieldpath.PathElement{}, fmt.Errorf("missing colon: %v", s)
	}
	switch split[0] {
	case Field:
		return fieldpath.PathElement{
			FieldName: &split[1],
		}, nil
	case Value:
		val, err := value.FromJSON([]byte(split[1]))
		if err != nil {
			return fieldpath.PathElement{}, err
		}
		return fieldpath.PathElement{
			Value: &val,
		}, nil
	case Index:
		i, err := strconv.Atoi(split[1])
		if err != nil {
			return fieldpath.PathElement{}, err
		}
		return fieldpath.PathElement{
			Index: &i,
		}, nil
	case Key:
		kv := map[string]string{}
		err := json.Unmarshal([]byte(split[1]), &kv)
		if err != nil {
			return fieldpath.PathElement{}, err
		}
		fields := []value.Field{}
		for k, v := range kv {
			fmt.Println(v)
			val, err := value.FromJSON([]byte(v))
			if err != nil {
				return fieldpath.PathElement{}, err
			}

			fields = append(fields, value.Field{
				Name:  k,
				Value: val,
			})
		}
		return fieldpath.PathElement{
			Key: fields,
		}, nil
	default:
		// Ignore unknown key types
		return fieldpath.PathElement{}, nil
	}
}

func PathElementString(pe fieldpath.PathElement) (string, error) {
	switch {
	case pe.FieldName != nil:
		return Field + Separator + *pe.FieldName, nil
	case len(pe.Key) > 0:
		kv := map[string]string{}
		for _, k := range pe.Key {
			b, err := k.Value.ToJSON()
			if err != nil {
				return "", err
			}
			kv[k.Name] = string(b)
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
