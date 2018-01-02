package abe

import (
    "time"
)

type ExamplepbABitOfEverything struct {
    BoolValue  bool  `json:"bool_value,omitempty"`
    DoubleValue  float64  `json:"double_value,omitempty"`
    EnumValue  ExamplepbNumericEnum  `json:"enum_value,omitempty"`
    Fixed32Value  int64  `json:"fixed32_value,omitempty"`
    Fixed64Value  string  `json:"fixed64_value,omitempty"`
    FloatValue  float32  `json:"float_value,omitempty"`
    Int32Value  int32  `json:"int32_value,omitempty"`
    Int64Value  string  `json:"int64_value,omitempty"`
    MapValue  map[string]ExamplepbNumericEnum  `json:"map_value,omitempty"`
    MappedNestedValue  map[string]ABitOfEverythingNested  `json:"mapped_nested_value,omitempty"`
    MappedStringValue  map[string]string  `json:"mapped_string_value,omitempty"`
    Nested  []ABitOfEverythingNested  `json:"nested,omitempty"`
    NonConventionalNameValue  string  `json:"nonConventionalNameValue,omitempty"`
    OneofEmpty  ProtobufEmpty  `json:"oneof_empty,omitempty"`
    OneofString  string  `json:"oneof_string,omitempty"`
    RepeatedStringValue  []string  `json:"repeated_string_value,omitempty"`
    Sfixed32Value  int32  `json:"sfixed32_value,omitempty"`
    Sfixed64Value  string  `json:"sfixed64_value,omitempty"`
    SingleNested  ABitOfEverythingNested  `json:"single_nested,omitempty"`
    Sint32Value  int32  `json:"sint32_value,omitempty"`
    Sint64Value  string  `json:"sint64_value,omitempty"`
    StringValue  string  `json:"string_value,omitempty"`
    TimestampValue  time.Time  `json:"timestamp_value,omitempty"`
    Uint32Value  int64  `json:"uint32_value,omitempty"`
    Uint64Value  string  `json:"uint64_value,omitempty"`
    Uuid  string  `json:"uuid,omitempty"`
    
}
