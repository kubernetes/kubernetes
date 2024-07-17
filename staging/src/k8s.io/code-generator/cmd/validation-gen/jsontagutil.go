/*
Copyright 2021 The Kubernetes Authors.

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

package main

import (
	"reflect"
	"strings"

	"k8s.io/gengo/v2/types"
)

// TODO: This implements the same functionality as https://github.com/kubernetes/kubernetes/blob/master/staging/src/k8s.io/apimachinery/pkg/runtime/converter.go#L236
// but is based on the highly efficient approach from https://golang.org/src/encoding/json/encode.go

// JSONTags represents a go json field tag.
type JSONTags struct {
	name      string
	omit      bool
	inline    bool
	omitempty bool
}

func (t JSONTags) String() string {
	var tag string
	if !t.inline {
		tag += t.name
	}
	if t.omitempty {
		tag += ",omitempty"
	}
	if t.inline {
		tag += ",inline"
	}
	return tag
}

func lookupJSONTags(m types.Member) (JSONTags, bool) {
	tag := reflect.StructTag(m.Tags).Get("json")
	if tag == "" || tag == "-" {
		return JSONTags{}, false
	}
	name, opts := parseTag(tag)
	if name == "" {
		name = m.Name
	}
	return JSONTags{
		name:      name,
		omit:      false,
		inline:    opts.Contains("inline"),
		omitempty: opts.Contains("omitempty"),
	}, true
}

type tagOptions string

// parseTag splits a struct field's json tag into its name and
// comma-separated options.
func parseTag(tag string) (string, tagOptions) {
	if idx := strings.Index(tag, ","); idx != -1 {
		return tag[:idx], tagOptions(tag[idx+1:])
	}
	return tag, ""
}

// Contains reports whether a comma-separated listAlias of options
// contains a particular substr flag. substr must be surrounded by a
// string boundary or commas.
func (o tagOptions) Contains(optionName string) bool {
	if len(o) == 0 {
		return false
	}
	s := string(o)
	for s != "" {
		var next string
		i := strings.Index(s, ",")
		if i >= 0 {
			s, next = s[:i], s[i+1:]
		}
		if s == optionName {
			return true
		}
		s = next
	}
	return false
}
