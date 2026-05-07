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

package apidefinitions

import (
	"errors"

	"k8s.io/gengo/v2/codetags"
)

const tagGroupName = "groupName"

// extractGeneratorTag returns the values of "+<tagName>" in comments.
// A sole "false" value sets optedOut and clears values; any other value
// (including "false" mixed with other values) is returned as-is.
func extractGeneratorTag(comments []string, tagName string) (values []string, optedOut bool, err error) {
	if tagName == "" {
		return nil, false, nil
	}
	values, err = tagValues(comments, tagName)
	if err != nil {
		return nil, false, err
	}
	if len(values) == 1 && values[0] == "false" {
		return nil, true, nil
	}
	return values, false, nil
}

func tagValues(lines []string, name string) ([]string, error) {
	tags, err := extractTags(lines, name)
	if err != nil {
		return nil, err
	}
	if len(tags[name]) == 0 {
		return nil, nil
	}
	out := make([]string, 0, len(tags[name]))
	for _, t := range tags[name] {
		out = append(out, t.Value)
	}
	return out, nil
}

func extractTags(lines []string, names ...string) (map[string][]codetags.Tag, error) {
	raw := codetags.Extract("+", lines)
	out := map[string][]codetags.Tag{}
	for _, name := range names {
		for _, line := range raw[name] {
			t, err := codetags.Parse(line, codetags.RawValues(true))
			if err != nil {
				return nil, err
			}
			out[name] = append(out[name], t)
		}
	}
	return out, nil
}

// ErrGroupUndeclared indicates that no +groupName= tag declares a group
// for the package.
var ErrGroupUndeclared = errors.New("no group declared for package")

// GroupNameForPackage returns the +groupName= value from comments, or
// ErrGroupUndeclared if no such tag is present.
func GroupNameForPackage(comments []string) (string, error) {
	values, err := tagValues(comments, tagGroupName)
	if err != nil {
		return "", err
	}
	if len(values) > 0 {
		return values[0], nil
	}
	return "", ErrGroupUndeclared
}
