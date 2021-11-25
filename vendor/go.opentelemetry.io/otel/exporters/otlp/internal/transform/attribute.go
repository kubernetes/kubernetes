// Copyright The OpenTelemetry Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package transform

import (
	"reflect"

	"go.opentelemetry.io/otel/attribute"
	commonpb "go.opentelemetry.io/proto/otlp/common/v1"

	"go.opentelemetry.io/otel/sdk/resource"
)

// Attributes transforms a slice of KeyValues into a slice of OTLP attribute key-values.
func Attributes(attrs []attribute.KeyValue) []*commonpb.KeyValue {
	if len(attrs) == 0 {
		return nil
	}

	out := make([]*commonpb.KeyValue, 0, len(attrs))
	for _, kv := range attrs {
		out = append(out, toAttribute(kv))
	}
	return out
}

// ResourceAttributes transforms a Resource into a slice of OTLP attribute key-values.
func ResourceAttributes(resource *resource.Resource) []*commonpb.KeyValue {
	if resource.Len() == 0 {
		return nil
	}

	out := make([]*commonpb.KeyValue, 0, resource.Len())
	for iter := resource.Iter(); iter.Next(); {
		out = append(out, toAttribute(iter.Attribute()))
	}

	return out
}

func toAttribute(v attribute.KeyValue) *commonpb.KeyValue {
	result := &commonpb.KeyValue{
		Key:   string(v.Key),
		Value: new(commonpb.AnyValue),
	}
	switch v.Value.Type() {
	case attribute.BOOL:
		result.Value.Value = &commonpb.AnyValue_BoolValue{
			BoolValue: v.Value.AsBool(),
		}
	case attribute.INT64:
		result.Value.Value = &commonpb.AnyValue_IntValue{
			IntValue: v.Value.AsInt64(),
		}
	case attribute.FLOAT64:
		result.Value.Value = &commonpb.AnyValue_DoubleValue{
			DoubleValue: v.Value.AsFloat64(),
		}
	case attribute.STRING:
		result.Value.Value = &commonpb.AnyValue_StringValue{
			StringValue: v.Value.AsString(),
		}
	case attribute.ARRAY:
		result.Value.Value = &commonpb.AnyValue_ArrayValue{
			ArrayValue: &commonpb.ArrayValue{
				Values: arrayValues(v),
			},
		}
	default:
		result.Value.Value = &commonpb.AnyValue_StringValue{
			StringValue: "INVALID",
		}
	}
	return result
}

func arrayValues(kv attribute.KeyValue) []*commonpb.AnyValue {
	a := kv.Value.AsArray()
	aType := reflect.TypeOf(a)
	var valueFunc func(reflect.Value) *commonpb.AnyValue
	switch aType.Elem().Kind() {
	case reflect.Bool:
		valueFunc = func(v reflect.Value) *commonpb.AnyValue {
			return &commonpb.AnyValue{
				Value: &commonpb.AnyValue_BoolValue{
					BoolValue: v.Bool(),
				},
			}
		}
	case reflect.Int, reflect.Int64:
		valueFunc = func(v reflect.Value) *commonpb.AnyValue {
			return &commonpb.AnyValue{
				Value: &commonpb.AnyValue_IntValue{
					IntValue: v.Int(),
				},
			}
		}
	case reflect.Uintptr:
		valueFunc = func(v reflect.Value) *commonpb.AnyValue {
			return &commonpb.AnyValue{
				Value: &commonpb.AnyValue_IntValue{
					IntValue: int64(v.Uint()),
				},
			}
		}
	case reflect.Float64:
		valueFunc = func(v reflect.Value) *commonpb.AnyValue {
			return &commonpb.AnyValue{
				Value: &commonpb.AnyValue_DoubleValue{
					DoubleValue: v.Float(),
				},
			}
		}
	case reflect.String:
		valueFunc = func(v reflect.Value) *commonpb.AnyValue {
			return &commonpb.AnyValue{
				Value: &commonpb.AnyValue_StringValue{
					StringValue: v.String(),
				},
			}
		}
	}

	results := make([]*commonpb.AnyValue, aType.Len())
	for i, aValue := 0, reflect.ValueOf(a); i < aValue.Len(); i++ {
		results[i] = valueFunc(aValue.Index(i))
	}
	return results
}
