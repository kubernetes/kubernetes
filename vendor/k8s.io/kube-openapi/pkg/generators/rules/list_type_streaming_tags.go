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

package rules

import (
	"reflect"

	"k8s.io/gengo/v2/types"
)

// StreamingListTypeFieldOrder implements APIRule interface.
// Fields must be ordered TypeMeta, ListMeta, Items
type StreamingListTypeFieldOrder struct{}

func (l *StreamingListTypeFieldOrder) Name() string {
	return "streaming_list_type_field_order"
}
func (l *StreamingListTypeFieldOrder) Validate(t *types.Type) ([]string, error) {
	if !isListType(t) {
		return nil, nil
	}
	var fields []string
	if t.Members[0].Name != "TypeMeta" {
		fields = append(fields, "TypeMeta")
	}
	if t.Members[1].Name != "ListMeta" {
		fields = append(fields, "ListMeta")
	}
	if t.Members[2].Name != "Items" {
		fields = append(fields, "Items")
	}
	return fields, nil
}

// StreamingListTypeJSONTags implements APIRule interface.
// Fields must be JSON-tagged
type StreamingListTypeJSONTags struct{}

func (l *StreamingListTypeJSONTags) Name() string {
	return "streaming_list_type_json_tags"
}

func (l *StreamingListTypeJSONTags) Validate(t *types.Type) ([]string, error) {
	if !isListType(t) {
		return nil, nil
	}
	var fields []string
	for _, m := range t.Members {
		switch m.Name {
		case "TypeMeta":
			if reflect.StructTag(m.Tags).Get("json") != ",inline" {
				fields = append(fields, "TypeMeta")
			}
		case "ListMeta":
			if reflect.StructTag(m.Tags).Get("json") != "metadata,omitempty" {
				fields = append(fields, "ListMeta")
			}
		case "Items":
			if reflect.StructTag(m.Tags).Get("json") != "items" {
				fields = append(fields, "Items")
			}
		}
	}
	return fields, nil
}

// StreamingListTypeProtoTags implements APIRule interface.
// Fields must be Proto-tagged with specific tags for streaming to work.
type StreamingListTypeProtoTags struct{}

func (l *StreamingListTypeProtoTags) Name() string {
	return "streaming_list_type_proto_tags"
}
func (l *StreamingListTypeProtoTags) Validate(t *types.Type) ([]string, error) {
	if !isListType(t) {
		return nil, nil
	}
	var fields []string
	for _, m := range t.Members {
		switch m.Name {
		case "TypeMeta":
			if v := reflect.StructTag(m.Tags).Get("protobuf"); v != "" {
				fields = append(fields, "TypeMeta")
			}
		case "ListMeta":
			if v := reflect.StructTag(m.Tags).Get("protobuf"); v != "" && v != "bytes,1,opt,name=metadata" {
				fields = append(fields, "ListMeta")
			}
		case "Items":
			if v := reflect.StructTag(m.Tags).Get("protobuf"); v != "" && v != "bytes,2,rep,name=items" {
				fields = append(fields, "Items")
			}
		}
	}
	return fields, nil
}

func isListType(t *types.Type) bool {
	return len(t.Members) == 3 &&
		hasNamedMember(t, "TypeMeta") &&
		hasNamedMember(t, "ListMeta") &&
		hasNamedMember(t, "Items")
}
