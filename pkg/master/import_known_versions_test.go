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

package master

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/intstr"
)

func TestGroupVersions(t *testing.T) {
	// legacyUnsuffixedGroups contains the groups released prior to deciding that kubernetes API groups should be dns-suffixed
	// new groups should be suffixed with ".k8s.io" (https://github.com/kubernetes/kubernetes/pull/31887#issuecomment-244462396)
	legacyUnsuffixedGroups := sets.NewString(
		"",
		"apps",
		"autoscaling",
		"batch",
		"componentconfig",
		"extensions",
		"federation",
		"policy",
	)

	// No new groups should be added to the legacyUnsuffixedGroups exclusion list
	if len(legacyUnsuffixedGroups) != 8 {
		t.Errorf("No additional unnamespaced groups should be created")
	}

	for _, gv := range api.Registry.RegisteredGroupVersions() {
		if !strings.HasSuffix(gv.Group, ".k8s.io") && !legacyUnsuffixedGroups.Has(gv.Group) {
			t.Errorf("Group %s does not have the standard kubernetes API group suffix of .k8s.io", gv.Group)
		}
	}
}

func TestTypeTags(t *testing.T) {
	for gvk, knownType := range api.Scheme.AllKnownTypes() {
		if gvk.Version == runtime.APIVersionInternal {
			ensureNoTags(t, gvk, knownType, nil)
		} else {
			ensureTags(t, gvk, knownType, nil)
		}
	}
}

// These types are registered in external versions, and therefore include json tags,
// but are also registered in internal versions (or referenced from internal types),
// so we explicitly allow tags for them
var typesAllowedTags = map[reflect.Type]bool{
	reflect.TypeOf(intstr.IntOrString{}):    true,
	reflect.TypeOf(metav1.Time{}):           true,
	reflect.TypeOf(metav1.Duration{}):       true,
	reflect.TypeOf(metav1.TypeMeta{}):       true,
	reflect.TypeOf(metav1.ListMeta{}):       true,
	reflect.TypeOf(metav1.OwnerReference{}): true,
	reflect.TypeOf(metav1.LabelSelector{}):  true,
	reflect.TypeOf(metav1.GetOptions{}):     true,
	reflect.TypeOf(metav1.ExportOptions{}):  true,
}

func ensureNoTags(t *testing.T, gvk schema.GroupVersionKind, tp reflect.Type, parents []reflect.Type) {
	if _, ok := typesAllowedTags[tp]; ok {
		return
	}

	parents = append(parents, tp)

	switch tp.Kind() {
	case reflect.Map, reflect.Slice, reflect.Ptr:
		ensureNoTags(t, gvk, tp.Elem(), parents)

	case reflect.String, reflect.Bool, reflect.Float32, reflect.Int, reflect.Int32, reflect.Int64, reflect.Uint8, reflect.Uintptr, reflect.Uint32, reflect.Uint64, reflect.Interface:
		// no-op

	case reflect.Struct:
		for i := 0; i < tp.NumField(); i++ {
			f := tp.Field(i)
			jsonTag := f.Tag.Get("json")
			protoTag := f.Tag.Get("protobuf")
			if len(jsonTag) > 0 || len(protoTag) > 0 {
				t.Errorf("Internal types should not have json or protobuf tags. %#v has tag on field %v: %v", gvk, f.Name, f.Tag)
				for i, tp := range parents {
					t.Logf("%s%v:", strings.Repeat("  ", i), tp)
				}
			}

			ensureNoTags(t, gvk, f.Type, parents)
		}

	default:
		t.Errorf("Unexpected type %v in %#v", tp.Kind(), gvk)
		for i, tp := range parents {
			t.Logf("%s%v:", strings.Repeat("  ", i), tp)
		}
	}
}

var (
	marshalerType   = reflect.TypeOf((*json.Marshaler)(nil)).Elem()
	unmarshalerType = reflect.TypeOf((*json.Unmarshaler)(nil)).Elem()
)

// These fields are limited exceptions to the standard JSON naming structure.
// Additions should only be made if a non-standard field name was released and cannot be changed for compatibility reasons.
var allowedNonstandardJSONNames = map[reflect.Type]string{
	reflect.TypeOf(v1.DaemonEndpoint{}): "Port",
}

func ensureTags(t *testing.T, gvk schema.GroupVersionKind, tp reflect.Type, parents []reflect.Type) {
	// This type handles its own encoding/decoding and doesn't need json tags
	if tp.Implements(marshalerType) && (tp.Implements(unmarshalerType) || reflect.PtrTo(tp).Implements(unmarshalerType)) {
		return
	}

	parents = append(parents, tp)

	switch tp.Kind() {
	case reflect.Map, reflect.Slice, reflect.Ptr:
		ensureTags(t, gvk, tp.Elem(), parents)

	case reflect.String, reflect.Bool, reflect.Float32, reflect.Int, reflect.Int32, reflect.Int64, reflect.Uint8, reflect.Uintptr, reflect.Uint32, reflect.Uint64, reflect.Interface:
		// no-op

	case reflect.Struct:
		for i := 0; i < tp.NumField(); i++ {
			f := tp.Field(i)
			jsonTag := f.Tag.Get("json")
			if len(jsonTag) == 0 {
				t.Errorf("External types should have json tags. %#v tags on field %v are: %s", gvk, f.Name, f.Tag)
				for i, tp := range parents {
					t.Logf("%s%v", strings.Repeat("  ", i), tp)
				}
			}

			jsonTagName := strings.Split(jsonTag, ",")[0]
			if len(jsonTagName) > 0 && (jsonTagName[0] < 'a' || jsonTagName[0] > 'z') && jsonTagName != "-" && allowedNonstandardJSONNames[tp] != jsonTagName {
				t.Errorf("External types should have json names starting with lowercase letter. %#v has json tag on field %v with name %s", gvk, f.Name, jsonTagName)
				t.Log(tp)
				t.Log(allowedNonstandardJSONNames[tp])
				for i, tp := range parents {
					t.Logf("%s%v", strings.Repeat("  ", i), tp)
				}
			}

			ensureTags(t, gvk, f.Type, parents)
		}

	default:
		t.Errorf("Unexpected type %v in %#v", tp.Kind(), gvk)
		for i, tp := range parents {
			t.Logf("%s%v:", strings.Repeat("  ", i), tp)
		}
	}
}
