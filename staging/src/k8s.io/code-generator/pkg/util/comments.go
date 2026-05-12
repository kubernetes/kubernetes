/*
Copyright 2025 The Kubernetes Authors.

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

package util

import (
	"fmt"

	"k8s.io/gengo/v2"
)

// ExtractCommentTagsWithoutArguments parses comments for special metadata tags. The
// marker argument should be unique enough to identify the tags needed, and
// should not be a marker for tags you don't want, or else the caller takes
// responsibility for making that distinction.
//
// The tagNames argument is a list of specific tags being extracted. If this is
// nil or empty, all lines which match the marker are considered.  If this is
// specified, only lines with begin with marker + one of the tags will be
// considered.  This is useful when a common marker is used which may match
// lines which fail this syntax (e.g. which predate this definition).
//
// This function looks for input lines of the following forms:
//   - 'marker' + "key=value"
//   - 'marker' + "key()=value"
//   - 'marker' + "key(arg)=value"
//
// The arg is forbidden.  This function only consider tags with no arguments specified
// (either as "key=value" or as // "key()=value").  Finding tags with an argument will
// result in an error.
//
// The value is optional.  If not specified, the resulting Tag will have "" as
// the value.
//
// Tag comment-lines may have a trailing end-of-line comment.
//
// The map returned here is keyed by the Tag's name without args.
//
// A tag can be specified more than one time and all values are returned.  If
// the resulting map has an entry for a key, the value (a slice) is guaranteed
// to have at least 1 element.
//
// Example: if you pass "+" for 'marker', and the following lines are in
// the comments:
//
//	+foo=val1   // foo
//	+bar
//	+foo=val2   // also foo
//	+foo()=val3 // still foo
//	+baz="qux"
//
// Then this function will return:
//
//	map[string][]string{"foo":{"val1", "val2", "val3"}, "bar": {""}, "baz": {`"qux"`}}
func ExtractCommentTagsWithoutArguments(marker string, tagNames []string, lines []string) (map[string][]string, error) {
	functionStyleTags, err := gengo.ExtractFunctionStyleCommentTags(marker, tagNames, lines)
	if err != nil {
		return nil, err
	}

	out := make(map[string][]string)
	for tagName, tags := range functionStyleTags {
		values := make([]string, 0)

		for _, tag := range tags {
			if tag.Args == nil {
				values = append(values, tag.Value)
			} else {
				return nil, fmt.Errorf(`failed to parse tag %s: expected no arguments, found "%s"`, tag, tag.Args[0])
			}
		}

		if len(values) > 0 {
			out[tagName] = values
		}
	}

	return out, nil
}
