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

package api

import (
	"reflect"
	"sort"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
)

// expectedProtobufWireType returns the wire type word that a protobuf struct
// tag must carry for a field of the given Go type, or "" if the type is not
// something the protobuf generator maps.
func expectedProtobufWireType(t reflect.Type) string {
	switch t.Kind() {
	case reflect.Pointer:
		return expectedProtobufWireType(t.Elem())
	case reflect.Slice:
		if t.Elem().Kind() == reflect.Uint8 {
			return "bytes"
		}
		return expectedProtobufWireType(t.Elem())
	case reflect.Map, reflect.Struct, reflect.Interface, reflect.String:
		return "bytes"
	case reflect.Bool, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return "varint"
	case reflect.Float64:
		return "fixed64"
	case reflect.Float32:
		return "fixed32"
	}
	return ""
}

// TestProtobufTagWireTypes checks that the wire type in every `protobuf:"..."`
// struct tag matches the Go type of the field. The generated marshallers do
// not read these tags, so a wrong wire type does not change the wire format,
// but libraries that derive protobuf descriptors from struct tags at runtime
// (for example google.golang.org/protobuf's legacy message support) trust the
// tag and fail when it disagrees with the Go type.
func TestProtobufTagWireTypes(t *testing.T) {
	scheme := runtime.NewScheme()
	for _, builder := range groups {
		if err := builder.AddToScheme(scheme); err != nil {
			t.Fatalf("unexpected error adding to scheme: %v", err)
		}
	}

	visited := map[reflect.Type]bool{}
	var mismatches []string
	var visit func(t reflect.Type)
	visit = func(t reflect.Type) {
		for t.Kind() == reflect.Pointer || t.Kind() == reflect.Slice || t.Kind() == reflect.Map {
			t = t.Elem()
		}
		if t.Kind() != reflect.Struct || visited[t] {
			return
		}
		visited[t] = true
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			visit(f.Type)
			tag, ok := f.Tag.Lookup("protobuf")
			if !ok || tag == "-" {
				continue
			}
			got, _, _ := strings.Cut(tag, ",")
			want := expectedProtobufWireType(f.Type)
			if want != "" && got != want {
				mismatches = append(mismatches, t.PkgPath()+"."+t.Name()+"."+f.Name+": Go type "+f.Type.String()+" needs protobuf wire type "+want+", tag says "+got)
			}
		}
	}
	for _, typ := range scheme.AllKnownTypes() {
		visit(typ)
	}
	sort.Strings(mismatches)
	for _, m := range mismatches {
		t.Error(m)
	}
}
