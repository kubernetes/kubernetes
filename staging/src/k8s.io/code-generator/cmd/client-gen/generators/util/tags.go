/*
Copyright 2016 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"strings"

	"k8s.io/gengo/types"
)

var supportedTags = []string{
	"genclient",
	"genclient:nonNamespaced",
	"genclient:noVerbs",
	"genclient:onlyVerbs",
	"genclient:skipVerbs",
	"genclient:noStatus",
	"genclient:readonly",
}

// SupportedVerbs is a list of supported verbs for +onlyVerbs and +skipVerbs.
var SupportedVerbs = []string{
	"create",
	"update",
	"updateStatus",
	"delete",
	"deleteCollection",
	"get",
	"list",
	"watch",
	"patch",
}

// ReadonlyVerbs represents a list of read-only verbs.
var ReadonlyVerbs = []string{
	"get",
	"list",
	"watch",
}

// Tags represents a genclient configuration for a single type.
type Tags struct {
	// +genclient
	GenerateClient bool
	// +genclient:nonNamespaced
	NonNamespaced bool
	// +genclient:noStatus
	NoStatus bool
	// +genclient:noVerbs
	NoVerbs bool
	// +genclient:skipVerbs=get,update
	// +genclient:onlyVerbs=create,delete
	SkipVerbs []string
}

// HasVerb returns true if we should include the given verb in final client interface and
// generate the function for it.
func (t Tags) HasVerb(verb string) bool {
	if len(t.SkipVerbs) == 0 {
		return true
	}
	for _, s := range t.SkipVerbs {
		if verb == s {
			return false
		}
	}
	return true
}

// MustParseClientGenTags calls ParseClientGenTags but instead of returning error it panics.
func MustParseClientGenTags(lines []string) Tags {
	tags, err := ParseClientGenTags(lines)
	if err != nil {
		panic(err.Error())
	}
	return tags
}

// ParseClientGenTags parse the provided genclient tags and validates that no unknown
// tags are provided.
func ParseClientGenTags(lines []string) (Tags, error) {
	ret := Tags{}
	values := types.ExtractCommentTags("+", lines)
	value := []string{}
	value, ret.GenerateClient = values["genclient"]
	// Check the old format and error when used to avoid generating client when //+genclient=false
	if len(value) > 0 && len(value[0]) > 0 {
		return ret, fmt.Errorf("+genclient=%s is invalid, use //+genclient if you want to generate client or omit it when you want to disable generation", value)
	}
	_, ret.NonNamespaced = values["genclient:nonNamespaced"]
	// Check the old format and error when used
	if value := values["nonNamespaced"]; len(value) > 0 && len(value[0]) > 0 {
		return ret, fmt.Errorf("+nonNamespaced=%s is invalid, use //+genclient:nonNamespaced instead", value[0])
	}
	_, ret.NoVerbs = values["genclient:noVerbs"]
	_, ret.NoStatus = values["genclient:noStatus"]
	onlyVerbs := []string{}
	if _, isReadonly := values["genclient:readonly"]; isReadonly {
		onlyVerbs = ReadonlyVerbs
	}
	// Check the old format and error when used
	if value := values["readonly"]; len(value) > 0 && len(value[0]) > 0 {
		return ret, fmt.Errorf("+readonly=%s is invalid, use //+genclient:readonly instead", value[0])
	}
	if v, exists := values["genclient:skipVerbs"]; exists {
		ret.SkipVerbs = strings.Split(v[0], ",")
	}
	if v, exists := values["genclient:onlyVerbs"]; exists || len(onlyVerbs) > 0 {
		if len(v) > 0 {
			onlyVerbs = append(onlyVerbs, strings.Split(v[0], ",")...)
		}
		skipVerbs := []string{}
		for _, m := range SupportedVerbs {
			skip := true
			for _, o := range onlyVerbs {
				if o == m {
					skip = false
					break
				}
			}
			// Check for conflicts
			for _, v := range skipVerbs {
				if v == m {
					return ret, fmt.Errorf("verb %q used both in genclient:skipVerbs and genclient:onlyVerbs", v)
				}
			}
			if skip {
				skipVerbs = append(skipVerbs, m)
			}
		}
		ret.SkipVerbs = skipVerbs
	}
	return ret, validateClientGenTags(values)
}

// validateTags validates that only supported genclient tags were provided.
func validateClientGenTags(values map[string][]string) error {
	for _, k := range supportedTags {
		delete(values, k)
	}
	for key := range values {
		if strings.HasPrefix(key, "genclient") {
			return errors.New("unknown tag detected: " + key)
		}
	}
	return nil
}
