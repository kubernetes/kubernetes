/*
Copyright 2019 The Kubernetes Authors.

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

package naming

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
)

var (
	marshalerType   = reflect.TypeOf((*json.Marshaler)(nil)).Elem()
	unmarshalerType = reflect.TypeOf((*json.Unmarshaler)(nil)).Elem()
)

// VerifyGroupNames ensures that all groups in the scheme ends with the k8s.io suffix.
// Exceptions can be tolerated using the legacyUnsuffixedGroups parameter
func VerifyGroupNames(scheme *runtime.Scheme, legacyUnsuffixedGroups sets.String) error {
	errs := []error{}
	for _, gv := range scheme.PrioritizedVersionsAllGroups() {
		if !strings.HasSuffix(gv.Group, ".k8s.io") && !legacyUnsuffixedGroups.Has(gv.Group) {
			errs = append(errs, fmt.Errorf("Group %s does not have the standard kubernetes API group suffix of .k8s.io", gv.Group))
		}
	}
	return errors.NewAggregate(errs)
}

// VerifyTagNaming ensures that all types in the scheme have JSON tags set on external types, and JSON tags not set on internal types.
// Exceptions can be tolerated using the typesAllowedTags and allowedNonstandardJSONNames parameters
func VerifyTagNaming(scheme *runtime.Scheme, typesAllowedTags map[reflect.Type]bool, allowedNonstandardJSONNames map[reflect.Type]string) error {
	errs := []error{}
	for gvk, knownType := range scheme.AllKnownTypes() {
		var err error
		if gvk.Version == runtime.APIVersionInternal {
			err = errors.NewAggregate(ensureNoTags(gvk, knownType, nil, typesAllowedTags))
		} else {
			err = errors.NewAggregate(ensureTags(gvk, knownType, nil, allowedNonstandardJSONNames))
		}
		if err != nil {
			errs = append(errs, err)
		}
	}
	return errors.NewAggregate(errs)
}

func ensureNoTags(gvk schema.GroupVersionKind, tp reflect.Type, parents []reflect.Type, typesAllowedTags map[reflect.Type]bool) []error {
	errs := []error{}
	if _, ok := typesAllowedTags[tp]; ok {
		return errs
	}

	// Don't look at the same type multiple times
	if containsType(parents, tp) {
		return nil
	}
	parents = append(parents, tp)

	switch tp.Kind() {
	case reflect.Map, reflect.Slice, reflect.Ptr:
		errs = append(errs, ensureNoTags(gvk, tp.Elem(), parents, typesAllowedTags)...)

	case reflect.String, reflect.Bool, reflect.Float32, reflect.Float64, reflect.Int32, reflect.Int64, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr, reflect.Interface:
		// no-op

	case reflect.Struct:
		for i := 0; i < tp.NumField(); i++ {
			f := tp.Field(i)
			if f.PkgPath != "" {
				continue // Ignore unexported fields
			}
			jsonTag := f.Tag.Get("json")
			protoTag := f.Tag.Get("protobuf")
			if len(jsonTag) > 0 || len(protoTag) > 0 {
				errs = append(errs, fmt.Errorf("Internal types should not have json or protobuf tags. %#v has tag on field %v: %v.\n%s", gvk, f.Name, f.Tag, fmtParentString(parents)))
			}

			errs = append(errs, ensureNoTags(gvk, f.Type, parents, typesAllowedTags)...)
		}

	default:
		errs = append(errs, fmt.Errorf("Unexpected type %v in %#v.\n%s", tp.Kind(), gvk, fmtParentString(parents)))
	}
	return errs
}

func ensureTags(gvk schema.GroupVersionKind, tp reflect.Type, parents []reflect.Type, allowedNonstandardJSONNames map[reflect.Type]string) []error {
	errs := []error{}
	// This type handles its own encoding/decoding and doesn't need json tags
	if tp.Implements(marshalerType) && (tp.Implements(unmarshalerType) || reflect.PtrTo(tp).Implements(unmarshalerType)) {
		return errs
	}

	// Don't look at the same type multiple times
	if containsType(parents, tp) {
		return nil
	}
	parents = append(parents, tp)

	switch tp.Kind() {
	case reflect.Map, reflect.Slice, reflect.Ptr:
		errs = append(errs, ensureTags(gvk, tp.Elem(), parents, allowedNonstandardJSONNames)...)

	case reflect.String, reflect.Bool, reflect.Float32, reflect.Float64, reflect.Int32, reflect.Int64, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr, reflect.Interface:
		// no-op

	case reflect.Struct:
		for i := 0; i < tp.NumField(); i++ {
			f := tp.Field(i)
			jsonTag := f.Tag.Get("json")
			if len(jsonTag) == 0 {
				errs = append(errs, fmt.Errorf("external types should have json tags. %#v tags on field %v are: %s.\n%s", gvk, f.Name, f.Tag, fmtParentString(parents)))
			}

			jsonTagName := strings.Split(jsonTag, ",")[0]
			if len(jsonTagName) > 0 && (jsonTagName[0] < 'a' || jsonTagName[0] > 'z') && jsonTagName != "-" && allowedNonstandardJSONNames[tp] != jsonTagName {
				errs = append(errs, fmt.Errorf("external types should have json names starting with lowercase letter. %#v has json tag on field %v with name %s.\n%s", gvk, f.Name, jsonTagName, fmtParentString(parents)))
			}

			errs = append(errs, ensureTags(gvk, f.Type, parents, allowedNonstandardJSONNames)...)
		}

	default:
		errs = append(errs, fmt.Errorf("Unexpected type %v in %#v.\n%s", tp.Kind(), gvk, fmtParentString(parents)))
	}
	return errs
}

func fmtParentString(parents []reflect.Type) string {
	str := "Type parents:\n"
	for i, tp := range parents {
		str += fmt.Sprintf("%s%v\n", strings.Repeat(" ", i), tp)
	}
	return str
}

// containsType returns true if s contains t, false otherwise
func containsType(s []reflect.Type, t reflect.Type) bool {
	for _, u := range s {
		if t == u {
			return true
		}
	}
	return false
}
