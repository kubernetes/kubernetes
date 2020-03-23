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

package openapi

import (
	"bytes"
	"fmt"
	"reflect"
	"sort"
	"strings"
	"unicode"

	restful "github.com/emicklei/go-restful"
	"github.com/go-openapi/spec"

	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kube-openapi/pkg/util"
)

var verbs = util.NewTrie([]string{"get", "log", "read", "replace", "patch", "delete", "deletecollection", "watch", "connect", "proxy", "list", "create", "patch"})

const (
	extensionGVK = "x-kubernetes-group-version-kind"
)

// ToValidOperationID makes an string a valid op ID (e.g. removing punctuations and whitespaces and make it camel case)
func ToValidOperationID(s string, capitalizeFirstLetter bool) string {
	var buffer bytes.Buffer
	capitalize := capitalizeFirstLetter
	for i, r := range s {
		if unicode.IsLetter(r) || r == '_' || (i != 0 && unicode.IsDigit(r)) {
			if capitalize {
				buffer.WriteRune(unicode.ToUpper(r))
				capitalize = false
			} else {
				buffer.WriteRune(r)
			}
		} else {
			capitalize = true
		}
	}
	return buffer.String()
}

// GetOperationIDAndTags returns a customize operation ID and a list of tags for kubernetes API server's OpenAPI spec to prevent duplicate IDs.
func GetOperationIDAndTags(r *restful.Route) (string, []string, error) {
	op := r.Operation
	path := r.Path
	var tags []string
	prefix, exists := verbs.GetPrefix(op)
	if !exists {
		return op, tags, fmt.Errorf("operation names should start with a verb. Cannot determine operation verb from %v", op)
	}
	op = op[len(prefix):]
	parts := strings.Split(strings.Trim(path, "/"), "/")
	// Assume /api is /apis/core, remove this when we actually server /api/... on /apis/core/...
	if len(parts) >= 1 && parts[0] == "api" {
		parts = append([]string{"apis", "core"}, parts[1:]...)
	}
	if len(parts) >= 2 && parts[0] == "apis" {
		trimmed := strings.TrimSuffix(parts[1], ".k8s.io")
		prefix = prefix + ToValidOperationID(trimmed, prefix != "")
		tag := ToValidOperationID(trimmed, false)
		if len(parts) > 2 {
			prefix = prefix + ToValidOperationID(parts[2], prefix != "")
			tag = tag + "_" + ToValidOperationID(parts[2], false)
		}
		tags = append(tags, tag)
	} else if len(parts) >= 1 {
		tags = append(tags, ToValidOperationID(parts[0], false))
	}
	return prefix + ToValidOperationID(op, prefix != ""), tags, nil
}

type groupVersionKinds []v1.GroupVersionKind

func (s groupVersionKinds) Len() int {
	return len(s)
}

func (s groupVersionKinds) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (s groupVersionKinds) Less(i, j int) bool {
	if s[i].Group == s[j].Group {
		if s[i].Version == s[j].Version {
			return s[i].Kind < s[j].Kind
		}
		return s[i].Version < s[j].Version
	}
	return s[i].Group < s[j].Group
}

func (s groupVersionKinds) JSON() []interface{} {
	j := []interface{}{}
	for _, gvk := range s {
		j = append(j, map[string]interface{}{
			"group":   gvk.Group,
			"version": gvk.Version,
			"kind":    gvk.Kind,
		})
	}
	return j
}

// DefinitionNamer is the type to customize OpenAPI definition name.
type DefinitionNamer struct {
	typeGroupVersionKinds map[string]groupVersionKinds
}

func gvkConvert(gvk schema.GroupVersionKind) v1.GroupVersionKind {
	return v1.GroupVersionKind{
		Group:   gvk.Group,
		Version: gvk.Version,
		Kind:    gvk.Kind,
	}
}

func friendlyName(name string) string {
	nameParts := strings.Split(name, "/")
	// Reverse first part. e.g., io.k8s... instead of k8s.io...
	if len(nameParts) > 0 && strings.Contains(nameParts[0], ".") {
		parts := strings.Split(nameParts[0], ".")
		for i, j := 0, len(parts)-1; i < j; i, j = i+1, j-1 {
			parts[i], parts[j] = parts[j], parts[i]
		}
		nameParts[0] = strings.Join(parts, ".")
	}
	return strings.Join(nameParts, ".")
}

func typeName(t reflect.Type) string {
	path := t.PkgPath()
	if strings.Contains(path, "/vendor/") {
		path = path[strings.Index(path, "/vendor/")+len("/vendor/"):]
	}
	return fmt.Sprintf("%s.%s", path, t.Name())
}

// NewDefinitionNamer constructs a new DefinitionNamer to be used to customize OpenAPI spec.
func NewDefinitionNamer(schemes ...*runtime.Scheme) *DefinitionNamer {
	ret := &DefinitionNamer{
		typeGroupVersionKinds: map[string]groupVersionKinds{},
	}
	for _, s := range schemes {
		for gvk, rtype := range s.AllKnownTypes() {
			newGVK := gvkConvert(gvk)
			exists := false
			for _, existingGVK := range ret.typeGroupVersionKinds[typeName(rtype)] {
				if newGVK == existingGVK {
					exists = true
					break
				}
			}
			if !exists {
				ret.typeGroupVersionKinds[typeName(rtype)] = append(ret.typeGroupVersionKinds[typeName(rtype)], newGVK)
			}
		}
	}
	for _, gvk := range ret.typeGroupVersionKinds {
		sort.Sort(gvk)
	}
	return ret
}

// GetDefinitionName returns the name and tags for a given definition
func (d *DefinitionNamer) GetDefinitionName(name string) (string, spec.Extensions) {
	if groupVersionKinds, ok := d.typeGroupVersionKinds[name]; ok {
		return friendlyName(name), spec.Extensions{
			extensionGVK: groupVersionKinds.JSON(),
		}
	}
	return friendlyName(name), nil
}
