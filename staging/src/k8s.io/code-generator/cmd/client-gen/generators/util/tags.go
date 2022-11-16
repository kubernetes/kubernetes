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
	"genclient:method",
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
	"apply",
	"applyStatus",
}

// ReadonlyVerbs represents a list of read-only verbs.
var ReadonlyVerbs = []string{
	"get",
	"list",
	"watch",
}

// genClientPrefix is the default prefix for all genclient tags.
const genClientPrefix = "genclient:"

// unsupportedExtensionVerbs is a list of verbs we don't support generating
// extension client functions for.
var unsupportedExtensionVerbs = []string{
	"updateStatus",
	"deleteCollection",
	"watch",
	"delete",
}

// inputTypeSupportedVerbs is a list of verb types that supports overriding the
// input argument type.
var inputTypeSupportedVerbs = []string{
	"create",
	"update",
	"apply",
}

// resultTypeSupportedVerbs is a list of verb types that supports overriding the
// resulting type.
var resultTypeSupportedVerbs = []string{
	"create",
	"update",
	"get",
	"list",
	"patch",
	"apply",
}

// Extensions allows to extend the default set of client verbs
// (CRUD+watch+patch+list+deleteCollection) for a given type with custom defined
// verbs. Custom verbs can have custom input and result types and also allow to
// use a sub-resource in a request instead of top-level resource type.
//
// Example:
//
// +genclient:method=UpdateScale,verb=update,subresource=scale,input=Scale,result=Scale
//
// type ReplicaSet struct { ... }
//
// The 'method=UpdateScale' is the name of the client function.
// The 'verb=update' here means the client function will use 'PUT' action.
// The 'subresource=scale' means we will use SubResource template to generate this client function.
// The 'input' is the input type used for creation (function argument).
// The 'result' (not needed in this case) is the result type returned from the
// client function.
type extension struct {
	// VerbName is the name of the custom verb (Scale, Instantiate, etc..)
	VerbName string
	// VerbType is the type of the verb (only verbs from SupportedVerbs are
	// supported)
	VerbType string
	// SubResourcePath defines a path to a sub-resource to use in the request.
	// (optional)
	SubResourcePath string
	// InputTypeOverride overrides the input parameter type for the verb. By
	// default the original type is used. Overriding the input type only works for
	// "create" and "update" verb types. The given type must exists in the same
	// package as the original type.
	// (optional)
	InputTypeOverride string
	// ResultTypeOverride overrides the resulting object type for the verb. By
	// default the original type is used. Overriding the result type works.
	// (optional)
	ResultTypeOverride string
}

// IsSubresource indicates if this extension should generate the sub-resource.
func (e *extension) IsSubresource() bool {
	return len(e.SubResourcePath) > 0
}

// HasVerb checks if the extension matches the given verb.
func (e *extension) HasVerb(verb string) bool {
	return e.VerbType == verb
}

// Input returns the input override package path and the type.
func (e *extension) Input() (string, string) {
	parts := strings.Split(e.InputTypeOverride, ".")
	return parts[len(parts)-1], strings.Join(parts[0:len(parts)-1], ".")
}

// Result returns the result override package path and the type.
func (e *extension) Result() (string, string) {
	parts := strings.Split(e.ResultTypeOverride, ".")
	return parts[len(parts)-1], strings.Join(parts[0:len(parts)-1], ".")
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
	// +genclient:method=UpdateScale,verb=update,subresource=scale,input=Scale,result=Scale
	Extensions []extension
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
	var value []string
	value, ret.GenerateClient = values["genclient"]
	// Check the old format and error when used to avoid generating client when //+genclient=false
	if len(value) > 0 && len(value[0]) > 0 {
		return ret, fmt.Errorf("+genclient=%s is invalid, use //+genclient if you want to generate client or omit it when you want to disable generation", value)
	}
	_, ret.NonNamespaced = values[genClientPrefix+"nonNamespaced"]
	// Check the old format and error when used
	if value := values["nonNamespaced"]; len(value) > 0 && len(value[0]) > 0 {
		return ret, fmt.Errorf("+nonNamespaced=%s is invalid, use //+genclient:nonNamespaced instead", value[0])
	}
	_, ret.NoVerbs = values[genClientPrefix+"noVerbs"]
	_, ret.NoStatus = values[genClientPrefix+"noStatus"]
	onlyVerbs := []string{}
	if _, isReadonly := values[genClientPrefix+"readonly"]; isReadonly {
		onlyVerbs = ReadonlyVerbs
	}
	// Check the old format and error when used
	if value := values["readonly"]; len(value) > 0 && len(value[0]) > 0 {
		return ret, fmt.Errorf("+readonly=%s is invalid, use //+genclient:readonly instead", value[0])
	}
	if v, exists := values[genClientPrefix+"skipVerbs"]; exists {
		ret.SkipVerbs = strings.Split(v[0], ",")
	}
	if v, exists := values[genClientPrefix+"onlyVerbs"]; exists || len(onlyVerbs) > 0 {
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
	var err error
	if ret.Extensions, err = parseClientExtensions(values); err != nil {
		return ret, err
	}
	return ret, validateClientGenTags(values)
}

func parseClientExtensions(tags map[string][]string) ([]extension, error) {
	var ret []extension
	for name, values := range tags {
		if !strings.HasPrefix(name, genClientPrefix+"method") {
			continue
		}
		for _, value := range values {
			// the value comes in this form: "Foo,verb=create"
			ext := extension{}
			parts := strings.Split(value, ",")
			if len(parts) == 0 {
				return nil, fmt.Errorf("invalid of empty extension verb name: %q", value)
			}
			// The first part represents the name of the extension
			ext.VerbName = parts[0]
			if len(ext.VerbName) == 0 {
				return nil, fmt.Errorf("must specify a verb name (// +genclient:method=Foo,verb=create)")
			}
			// Parse rest of the arguments
			params := parts[1:]
			for _, p := range params {
				parts := strings.Split(p, "=")
				if len(parts) != 2 {
					return nil, fmt.Errorf("invalid extension tag specification %q", p)
				}
				key, val := strings.TrimSpace(parts[0]), strings.TrimSpace(parts[1])
				if len(val) == 0 {
					return nil, fmt.Errorf("empty value of %q for %q extension", key, ext.VerbName)
				}
				switch key {
				case "verb":
					ext.VerbType = val
				case "subresource":
					ext.SubResourcePath = val
				case "input":
					ext.InputTypeOverride = val
				case "result":
					ext.ResultTypeOverride = val
				default:
					return nil, fmt.Errorf("unknown extension configuration key %q", key)
				}
			}
			// Validate resulting extension configuration
			if len(ext.VerbType) == 0 {
				return nil, fmt.Errorf("verb type must be specified (use '// +genclient:method=%s,verb=create')", ext.VerbName)
			}
			if len(ext.ResultTypeOverride) > 0 {
				supported := false
				for _, v := range resultTypeSupportedVerbs {
					if ext.VerbType == v {
						supported = true
						break
					}
				}
				if !supported {
					return nil, fmt.Errorf("%s: result type is not supported for %q verbs (supported verbs: %#v)", ext.VerbName, ext.VerbType, resultTypeSupportedVerbs)
				}
			}
			if len(ext.InputTypeOverride) > 0 {
				supported := false
				for _, v := range inputTypeSupportedVerbs {
					if ext.VerbType == v {
						supported = true
						break
					}
				}
				if !supported {
					return nil, fmt.Errorf("%s: input type is not supported for %q verbs (supported verbs: %#v)", ext.VerbName, ext.VerbType, inputTypeSupportedVerbs)
				}
			}
			for _, t := range unsupportedExtensionVerbs {
				if ext.VerbType == t {
					return nil, fmt.Errorf("verb %q is not supported by extension generator", ext.VerbType)
				}
			}
			ret = append(ret, ext)
		}
	}
	return ret, nil
}

// validateTags validates that only supported genclient tags were provided.
func validateClientGenTags(values map[string][]string) error {
	for _, k := range supportedTags {
		delete(values, k)
	}
	for key := range values {
		if strings.HasPrefix(key, strings.TrimSuffix(genClientPrefix, ":")) {
			return errors.New("unknown tag detected: " + key)
		}
	}
	return nil
}
