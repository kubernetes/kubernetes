/*
Copyright 2015 The Kubernetes Authors.

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

package gengo

import (
	"bytes"
	"fmt"
	"slices"
	"strings"

	"k8s.io/gengo/v2/codetags"
)

// ExtractCommentTags parses comments for lines of the form:
//
//	'marker' + "key=value".
//
// Values are optional; "" is the default.  A tag can be specified more than
// one time and all values are returned.  If the resulting map has an entry for
// a key, the value (a slice) is guaranteed to have at least 1 element.
//
// Example: if you pass "+" for 'marker', and the following lines are in
// the comments:
//
//	+foo=value1
//	+bar
//	+foo=value2
//	+baz="qux"
//
// Then this function will return:
//
//	map[string][]string{"foo":{"value1, "value2"}, "bar": {""}, "baz": {`"qux"`}}
//
// Deprecated: Prefer codetags.Extract and codetags.Parse.
func ExtractCommentTags(marker string, lines []string) map[string][]string {
	out := map[string][]string{}
	for _, line := range lines {
		line = strings.Trim(line, " ")
		if len(line) == 0 {
			continue
		}
		if !strings.HasPrefix(line, marker) {
			continue
		}
		kv := strings.SplitN(line[len(marker):], "=", 2)
		if len(kv) == 2 {
			out[kv[0]] = append(out[kv[0]], kv[1])
		} else if len(kv) == 1 {
			out[kv[0]] = append(out[kv[0]], "")
		}
	}
	return out
}

// ExtractSingleBoolCommentTag parses comments for lines of the form:
//
//	'marker' + "key=value1"
//
// If the tag is not found, the default value is returned.  Values are asserted
// to be boolean ("true" or "false"), and any other value will cause an error
// to be returned.  If the key has multiple values, the first one will be used.
//
// This function is a wrapper around codetags.Extract and codetags.Parse, but only supports tags with
// a single position arg of type string, and a value of type bool.
func ExtractSingleBoolCommentTag(marker string, key string, defaultVal bool, lines []string) (bool, error) {
	tags, err := ExtractFunctionStyleCommentTags(marker, []string{key}, lines, ParseValues(true))
	if err != nil {
		return false, err
	}
	values := tags[key]
	if values == nil {
		return defaultVal, nil
	}
	if values[0].Value == "true" {
		return true, nil
	}
	if values[0].Value == "false" {
		return false, nil
	}
	return false, fmt.Errorf("tag value for %q is not boolean: %q", key, values[0])
}

// ExtractFunctionStyleCommentTags parses comments for special metadata tags.
//
// This function is a wrapper around codetags.Extract and codetags.Parse, but only supports tags with
// a single position arg of type string.
func ExtractFunctionStyleCommentTags(marker string, tagNames []string, lines []string, options ...TagOption) (map[string][]Tag, error) {
	opts := tagOpts{}
	for _, o := range options {
		o(&opts)
	}

	out := map[string][]Tag{}

	tags := codetags.Extract(marker, lines)
	for tagName, tagLines := range tags {
		if len(tagNames) > 0 && !slices.Contains(tagNames, tagName) {
			continue
		}
		for _, line := range tagLines {
			typedTag, err := codetags.Parse(line, codetags.RawValues(!opts.parseValues))
			if err != nil {
				return nil, err
			}
			tag, err := toStringArgs(typedTag)
			if err != nil {
				return nil, err
			}
			out[tagName] = append(out[tagName], tag)
		}
	}

	return out, nil
}

// TagOption provides an option for extracting tags.
type TagOption func(opts *tagOpts)

// ParseValues enables parsing of tag values. When enabled, tag values must
// be valid quoted strings, ints, booleans, identifiers, or tags. Otherwise, a
// parse error will be returned. Also, when enabled, trailing comments are
// ignored.
// Default: disabled
func ParseValues(enabled bool) TagOption {
	return func(opts *tagOpts) {
		opts.parseValues = enabled
	}
}

type tagOpts struct {
	parseValues bool
}

func toStringArgs(tag codetags.Tag) (Tag, error) {
	var stringArgs []string
	if len(tag.Args) > 1 {
		return Tag{}, fmt.Errorf("expected one argument, got: %v", tag.Args)
	}
	for _, arg := range tag.Args {
		if len(arg.Name) > 0 {
			return Tag{}, fmt.Errorf("unexpected named argument: %q", arg.Name)
		}
		if arg.Type != codetags.ArgTypeString {
			return Tag{}, fmt.Errorf("unexpected argument type: %s", arg.Type)
		} else {
			stringArgs = append(stringArgs, arg.Value)
		}
	}
	return Tag{
		Name:  tag.Name,
		Args:  stringArgs,
		Value: tag.Value,
	}, nil
}

// Tag represents a single comment tag.
type Tag struct {
	// Name is the name of the tag with no arguments.
	Name string
	// Args is a list of optional arguments to the tag.
	Args []string
	// Value is the value of the tag.
	Value string
}

func (t Tag) String() string {
	buf := bytes.Buffer{}
	buf.WriteString(t.Name)
	if len(t.Args) > 0 {
		buf.WriteString("(")
		for i, a := range t.Args {
			if i > 0 {
				buf.WriteString(", ")
			}
			buf.WriteString(a)
		}
		buf.WriteString(")")
	}
	return buf.String()
}
