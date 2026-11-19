// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package ext

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"time"

	"cel.dev/cel-go/cel"
	"cel.dev/cel-go/common/cost"
	"cel.dev/cel-go/common/types"
	"cel.dev/cel-go/common/types/ref"
	"cel.dev/cel-go/common/types/traits"
	"go.yaml.in/yaml/v3"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/structpb"
)

const (
	maxJSONSize = 10 * 1024 * 1024 // 10MB maximum allowed JSON string size
	maxYAMLSize = 10 * 1024 * 1024 // 10MB maximum allowed YAML string size
)

// Encoders returns a cel.EnvOption to configure extended functions for string, byte, and object
// encodings.
//
// Note: all functions use the 'base64', 'json', and 'yaml' namespaces.
//
// Version 2 of this library depends on the CEL optional type. Please ensure that the
// cel.OptionalTypes() is enabled when using encoder extensions at version 2 or greater.
//
// # Base64.Decode
//
// Decodes base64-encoded string to bytes.
//
// This function will return an error if the string input is not base64-encoded.
//
//	base64.decode(<string>) -> <bytes>
//
// Examples:
//
//	base64.decode('aGVsbG8=')  // return b'hello'
//	base64.decode('aGVsbG8')   // return b'hello'
//
// # Base64.DecodeUrl
//
// Introduced at version: 2
//
// Decodes a base64url-encoded string to bytes.
//
// This function will return an error if the string input is not base64url-encoded.
//
//	base64.decodeUrl(<string>) -> <bytes>
//
// Examples:
//
//	base64.decodeUrl('aGVsbG8=')  // return b'hello'
//	base64.decodeUrl('aGVsbG8')   // return b'hello'
//	base64.decodeUrl('____')      // return b'\xff\xff\xff'
//
// # Base64.Encode
//
// Encodes bytes to a base64-encoded string.
//
//	base64.encode(<bytes>)  -> <string>
//
// Examples:
//
//	base64.encode(b'hello') // return b'aGVsbG8='
//
// # Base64.EncodeUrl
//
// Introduced at version: 2
//
// Encodes bytes to a base64url-encoded string.
//
//	base64.encodeUrl(<bytes>)  -> <string>
//
// Examples:
//
//	base64.encodeUrl(b'hello')        // return 'aGVsbG8='
//	base64.encodeUrl(b'\xff\xff\xff') // return '____'
//
// # JSON.Encode
//
// Introduced at version: 1
//
// Encodes a CEL value to a JSON string.
//
//	json.encode(<dyn>) -> <string>
//
// Examples:
//
//	json.encode({'hello': 'world'}) // return '{"hello":"world"}'
//
// # JSON.Parse
//
// Introduced at version: 2
//
// Parses a JSON string to a CEL value or a specific type.
//
//	json.parse(<string>) -> <optional_type(dyn)>
//	json.parse(<string>, <type(T)>) -> <optional_type(T)>
//
// Examples:
//
//	json.parse('{"hello":"world"}') // return optional.of({'hello': 'world'})
//	json.parse('123', int) // return optional.of(123)
//
// # YAML.Encode
//
// Introduced at version: 2
//
// Encodes a CEL value to a YAML string.
//
//	yaml.encode(<dyn>) -> <string>
//
// Examples:
//
//	yaml.encode({'hello': 'world'}) // return 'hello: world\n'
//
// # YAML.Parse
//
// Introduced at version: 2
//
// Parses a YAML string to a CEL value or a specific type.
//
//	yaml.parse(<string>) -> <optional_type(dyn)>
//	yaml.parse(<string>, <type(T)>) -> <optional_type(T)>
//
// Examples:
//
//	yaml.parse('hello: world') // return optional.of({'hello': 'world'})
//	yaml.parse('123', int) // return optional.of(123)
func Encoders(options ...EncodersOption) cel.EnvOption {
	l := &encoderLib{version: math.MaxUint32}
	for _, o := range options {
		l = o(l)
	}
	return cel.Lib(l)
}

// EncodersOption declares a functional operator for configuring encoder extensions.
type EncodersOption func(*encoderLib) *encoderLib

// EncodersVersion sets the library version for encoder extensions.
func EncodersVersion(version uint32) EncodersOption {
	return func(lib *encoderLib) *encoderLib {
		lib.version = version
		return lib
	}
}

type encoderLib struct {
	version uint32
}

func (*encoderLib) LibraryName() string {
	return "cel.lib.ext.encoders"
}

func (lib *encoderLib) CompileOptions() []cel.EnvOption {
	opts := []cel.EnvOption{
		cel.Function("base64.decode",
			cel.Overload("base64_decode_string", []*cel.Type{cel.StringType}, cel.BytesType,
				cel.UnaryBinding(func(str ref.Val) ref.Val {
					s := str.(types.String)
					return bytesOrError(base64DecodeString(string(s)))
				}))),
		cel.Function("base64.encode",
			cel.Overload("base64_encode_bytes", []*cel.Type{cel.BytesType}, cel.StringType,
				cel.UnaryBinding(func(bytes ref.Val) ref.Val {
					b := bytes.(types.Bytes)
					return stringOrError(base64EncodeBytes([]byte(b)))
				}))),
	}
	if lib.version >= 1 {
		opts = append(opts,
			cel.CostModel(
				base64DecodeCostModel("base64_decode_string"),
				base64EncodeCostModel("base64_encode_bytes"),
				unboundedCostModel("json_encode_dyn"),
			),
			cel.Function("json.encode",
				cel.Overload("json_encode_dyn", []*cel.Type{cel.DynType}, cel.StringType,
					cel.UnaryBinding(func(val ref.Val) ref.Val {
						return stringOrError(jsonEncodeValue(val))
					}))),
		)
	}
	if lib.version >= 2 {
		var adapt types.Adapter = types.DefaultTypeAdapter
		var prov types.Provider
		opts = append(opts,
			cel.CostModel(
				base64DecodeCostModel("base64_decode_url_string"),
				base64EncodeCostModel("base64_encode_url_bytes"),
				unboundedCostModel("json_parse_string"),
				unboundedCostModel("json_parse_string_type"),
				unboundedCostModel("yaml_encode_dyn"),
				unboundedCostModel("yaml_parse_string"),
				unboundedCostModel("yaml_parse_string_type"),
			),
			func(e *cel.Env) (*cel.Env, error) {
				adapt = e.CELTypeAdapter()
				prov = e.CELTypeProvider()
				return e, nil
			},
			cel.Function("base64.decodeUrl",
				cel.Overload("base64_decode_url_string", []*cel.Type{cel.StringType}, cel.BytesType,
					cel.UnaryBinding(func(str ref.Val) ref.Val {
						s := str.(types.String)
						return bytesOrError(base64DecodeURLString(string(s)))
					}))),
			cel.Function("base64.encodeUrl",
				cel.Overload("base64_encode_url_bytes", []*cel.Type{cel.BytesType}, cel.StringType,
					cel.UnaryBinding(func(bytes ref.Val) ref.Val {
						b := bytes.(types.Bytes)
						return stringOrError(base64EncodeURLBytes([]byte(b)))
					}))),
			cel.Function("json.parse",
				cel.Overload("json_parse_string",
					[]*cel.Type{cel.StringType},
					cel.OptionalType(cel.DynType),
					cel.UnaryBinding(func(val ref.Val) ref.Val {
						str := val.(types.String)
						return jsonParseString(adapt, string(str))
					}),
				),
				cel.Overload("json_parse_string_type",
					[]*cel.Type{cel.StringType, types.NewTypeTypeWithParam(cel.TypeParamType("T"))},
					cel.OptionalType(cel.TypeParamType("T")),
					cel.BinaryBinding(func(strVal, typeVal ref.Val) ref.Val {
						str := strVal.(types.String)
						targetType := typeVal.(ref.Type)
						return jsonParseWithType(adapt, prov, string(str), targetType)
					}),
				),
			),
			cel.Function("yaml.encode",
				cel.Overload("yaml_encode_dyn", []*cel.Type{cel.DynType}, cel.StringType,
					cel.UnaryBinding(func(val ref.Val) ref.Val {
						return stringOrError(yamlEncodeValue(val))
					}))),
			cel.Function("yaml.parse",
				cel.Overload("yaml_parse_string",
					[]*cel.Type{cel.StringType},
					cel.OptionalType(cel.DynType),
					cel.UnaryBinding(func(val ref.Val) ref.Val {
						str := val.(types.String)
						return yamlParseString(adapt, string(str))
					}),
				),
				cel.Overload("yaml_parse_string_type",
					[]*cel.Type{cel.StringType, types.NewTypeTypeWithParam(cel.TypeParamType("T"))},
					cel.OptionalType(cel.TypeParamType("T")),
					cel.BinaryBinding(func(strVal, typeVal ref.Val) ref.Val {
						str := strVal.(types.String)
						targetType := typeVal.(ref.Type)
						return yamlParseWithType(adapt, prov, string(str), targetType)
					}),
				),
			),
		)
	}
	return opts
}

func (lib *encoderLib) ProgramOptions() []cel.ProgramOption {
	return nil
}

// YAMLEncode encodes a CEL ref.Val to its YAML string representation.
func YAMLEncode(val ref.Val) (string, error) {
	return yamlEncodeValue(val)
}

// YAMLParse parses a YAML string into a dynamic CEL ref.Val using the given type adapter.
// If adapter is nil, types.DefaultTypeAdapter is used.
func YAMLParse(adapter types.Adapter, str string) (ref.Val, error) {
	if adapter == nil {
		adapter = types.DefaultTypeAdapter
	}
	if len(str) > maxYAMLSize {
		return nil, fmt.Errorf("yaml parse error: string size exceeds maximum allowed limit of %d bytes", maxYAMLSize)
	}
	jsonStr, ok := yamlToJSON(str)
	if !ok {
		return nil, fmt.Errorf("yaml parse error: invalid YAML input")
	}
	return JSONParse(adapter, jsonStr)
}

// YAMLParseWithType parses a YAML string into a typed CEL ref.Val conforming to targetType.
// If adapter is nil, types.DefaultTypeAdapter is used. If provider implements types.Adapter, it will be used as the adapter.
func YAMLParseWithType(adapter types.Adapter, provider types.Provider, str string, targetType ref.Type) (ref.Val, error) {
	if adapter == nil {
		adapter = types.DefaultTypeAdapter
	}
	if provAdapter, ok := provider.(types.Adapter); ok && provAdapter != nil {
		adapter = provAdapter
	}
	if len(str) > maxYAMLSize {
		return nil, fmt.Errorf("yaml parse error: string size exceeds maximum allowed limit of %d bytes", maxYAMLSize)
	}
	jsonStr, ok := yamlToJSON(str)
	if !ok {
		return nil, fmt.Errorf("yaml parse error: failed to parse YAML to type %v", targetType)
	}
	return JSONParseWithType(adapter, provider, jsonStr, targetType)
}

// JSONEncode encodes a CEL ref.Val to its JSON string representation.
func JSONEncode(val ref.Val) (string, error) {
	return jsonEncodeValue(val)
}

// JSONParse parses a JSON string into a dynamic CEL ref.Val using the given type adapter.
// If adapter is nil, types.DefaultTypeAdapter is used.
func JSONParse(adapter types.Adapter, str string) (ref.Val, error) {
	if adapter == nil {
		adapter = types.DefaultTypeAdapter
	}
	if len(str) > maxJSONSize {
		return nil, fmt.Errorf("json parse error: string size exceeds maximum allowed limit of %d bytes", maxJSONSize)
	}
	obj, ok := jsonParseDynamic(str)
	if !ok {
		return nil, fmt.Errorf("json parse error: invalid JSON input")
	}
	return adapter.NativeToValue(obj), nil
}

// JSONParseWithType parses a JSON string into a typed CEL ref.Val conforming to targetType.
// If adapter is nil, types.DefaultTypeAdapter is used. If provider implements types.Adapter, it will be used as the adapter.
func JSONParseWithType(adapter types.Adapter, provider types.Provider, str string, targetType ref.Type) (ref.Val, error) {
	if adapter == nil {
		adapter = types.DefaultTypeAdapter
	}
	if provAdapter, ok := provider.(types.Adapter); ok && provAdapter != nil {
		adapter = provAdapter
	}
	if len(str) > maxJSONSize {
		return nil, fmt.Errorf("json parse error: string size exceeds maximum allowed limit of %d bytes", maxJSONSize)
	}
	val, ok := jsonParseValue(adapter, provider, str, targetType)
	if !ok {
		return nil, fmt.Errorf("json parse error: failed to parse JSON to type %v", targetType)
	}
	return adapter.NativeToValue(val), nil
}

// Base64Encode encodes a byte slice into a standard base64 string.
func Base64Encode(bytes []byte) string {
	return base64.StdEncoding.EncodeToString(bytes)
}

// Base64Decode decodes a base64 encoded string into a byte slice.
func Base64Decode(str string) ([]byte, error) {
	return base64DecodeString(str)
}

func base64DecodeString(str string) ([]byte, error) {
	b, err := base64.StdEncoding.DecodeString(str)
	if err == nil {
		return b, nil
	}
	return base64.RawStdEncoding.DecodeString(str)
}

func base64DecodeURLString(str string) ([]byte, error) {
	b, err := base64.URLEncoding.DecodeString(str)
	if err == nil {
		return b, nil
	}
	if _, tryAltEncoding := err.(base64.CorruptInputError); tryAltEncoding {
		return base64.RawURLEncoding.DecodeString(str)
	}
	return nil, err
}

func base64EncodeBytes(bytes []byte) (string, error) {
	return base64.StdEncoding.EncodeToString(bytes), nil
}

func base64EncodeURLBytes(bytes []byte) (string, error) {
	return base64.URLEncoding.EncodeToString(bytes), nil
}

func estimateEncode(estimator cost.Estimator, target *cost.AstNode, args []cost.AstNode) *cost.CallEstimate {
	if len(args) != 1 {
		return nil
	}
	sz := cost.EstimateSize(estimator, args[0])
	estimate := sz.MultiplyByCostFactor(cost.StringCostFactor).Add(cost.CallCostEstimate)
	resSize := estimateEncodeSize(sz)
	return &cost.CallEstimate{CostEstimate: estimate, ResultSize: &resSize}
}

func estimateDecode(estimator cost.Estimator, target *cost.AstNode, args []cost.AstNode) *cost.CallEstimate {
	if len(args) != 1 {
		return nil
	}
	sz := cost.EstimateSize(estimator, args[0])
	estimate := sz.MultiplyByCostFactor(cost.StringCostFactor).Add(cost.CallCostEstimate)
	resSize := estimateDecodeSize(sz)
	return &cost.CallEstimate{CostEstimate: estimate, ResultSize: &resSize}
}

func estimateEncodeSize(sz cost.SizeEstimate) cost.SizeEstimate {
	minVal := (sz.Min*4 + 2) / 3
	maxVal := (sz.Max*4 + 2) / 3
	if sz.Max > math.MaxUint64/4 {
		maxVal = math.MaxUint64
	}
	return cost.SizeEstimate{Min: minVal, Max: maxVal}
}

func estimateDecodeSize(sz cost.SizeEstimate) cost.SizeEstimate {
	minVal := sz.Min * 3 / 4
	maxVal := sz.Max * 3 / 4
	return cost.SizeEstimate{Min: minVal, Max: maxVal}
}

func jsonEncodeValue(val ref.Val) (string, error) {
	switch v := val.(type) {
	case types.Bool:
		if v {
			return "true", nil
		}
		return "false", nil
	case types.Null:
		return "null", nil
	case types.Int:
		if v >= -9007199254740991 && v <= 9007199254740991 {
			return strconv.FormatInt(int64(v), 10), nil
		}
		return strconv.Quote(strconv.FormatInt(int64(v), 10)), nil
	case types.Uint:
		if v <= 9007199254740991 {
			return strconv.FormatUint(uint64(v), 10), nil
		}
		return strconv.Quote(strconv.FormatUint(uint64(v), 10)), nil
	case types.Double:
		f := float64(v)
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return "", fmt.Errorf("json encode error: NaN and Inf are not supported in JSON")
		}
		b, _ := json.Marshal(f)
		return string(b), nil
	case types.String:
		str := string(v)
		if len(str) < 128 && !strings.ContainsAny(str, "\"\\\n\r\t\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f<>&") {
			return strconv.Quote(str), nil
		}
		b, _ := json.Marshal(str)
		return string(b), nil
	case types.Bytes:
		return strconv.Quote(base64.StdEncoding.EncodeToString([]byte(v))), nil
	}
	if msg, ok := getProtoMessage(val); ok {
		jsonBytes, err := protojson.Marshal(msg)
		if err != nil {
			return "", err
		}
		return string(jsonBytes), nil
	}
	if val.Type().HasTrait(traits.FieldTesterType) {
		if encoded, ok := jsonEncodeNativeStruct(val.Value()); ok {
			return encoded, nil
		}
	}
	if lister, ok := val.(traits.Lister); ok {
		return jsonEncodeList(lister)
	}
	if mapper, ok := val.(traits.Mapper); ok {
		return jsonEncodeMap(mapper)
	}
	native, err := val.ConvertToNative(types.JSONValueType)
	if err != nil {
		return "", err
	}
	jsonValue, ok := native.(*structpb.Value)
	if !ok {
		return "", fmt.Errorf("cannot convert %T to JSON value", native)
	}
	jsonBytes, err := protojson.Marshal(jsonValue)
	if err != nil {
		return "", err
	}
	return string(jsonBytes), nil
}

func getProtoMessage(val ref.Val) (proto.Message, bool) {
	if msg, ok := val.(proto.Message); ok && msg != nil {
		return msg, true
	}
	if msg, ok := val.Value().(proto.Message); ok && msg != nil {
		return msg, true
	}
	return nil, false
}

func jsonEncodeNativeStruct(nativeVal any) (string, bool) {
	if nativeVal == nil {
		return "", false
	}
	if _, isProto := nativeVal.(proto.Message); isProto {
		return "", false
	}
	rt := reflect.TypeOf(nativeVal)
	if rt.Kind() == reflect.Pointer {
		rt = rt.Elem()
	}
	if rt.Kind() == reflect.Struct {
		jsonBytes, err := json.Marshal(nativeVal)
		if err == nil {
			return string(jsonBytes), true
		}
	}
	return "", false
}

func jsonEncodeList(lister traits.Lister) (string, error) {
	szVal := lister.Size()
	szInt, ok := szVal.(types.Int)
	if !ok {
		return "", fmt.Errorf("invalid list size: %v", szVal)
	}
	sz := int(szInt)
	if sz == 0 {
		return "[]", nil
	}
	var sb strings.Builder
	sb.WriteByte('[')
	for i := 0; i < sz; i++ {
		if i > 0 {
			sb.WriteByte(',')
		}
		elem := lister.Get(types.Int(i))
		elemStr, err := jsonEncodeValue(elem)
		if err != nil {
			return "", err
		}
		sb.WriteString(elemStr)
	}
	sb.WriteByte(']')
	return sb.String(), nil
}

func jsonEncodeMap(mapper traits.Mapper) (string, error) {
	szVal := mapper.Size()
	szInt, ok := szVal.(types.Int)
	if !ok {
		return "", fmt.Errorf("invalid map size: %v", szVal)
	}
	sz := int(szInt)
	if sz == 0 {
		return "{}", nil
	}
	var sb strings.Builder
	sb.WriteByte('{')
	it := mapper.Iterator()
	first := true
	for it.HasNext() == types.True {
		k := it.Next()
		kStr, ok := k.(types.String)
		if !ok {
			kStrVal, err := jsonEncodeValue(k)
			if err != nil {
				return "", err
			}
			kStr = types.String(kStrVal)
		}
		kQuoted, _ := json.Marshal(string(kStr))
		v := mapper.Get(k)
		vStr, err := jsonEncodeValue(v)
		if err != nil {
			return "", err
		}
		if !first {
			sb.WriteByte(',')
		}
		first = false
		sb.Write(kQuoted)
		sb.WriteByte(':')
		sb.WriteString(vStr)
	}
	sb.WriteByte('}')
	return sb.String(), nil
}

func jsonUnmarshalExact(str string, out any) error {
	dec := json.NewDecoder(strings.NewReader(str))
	if err := dec.Decode(out); err != nil {
		return err
	}
	if _, err := dec.Token(); err != io.EOF {
		return fmt.Errorf("unexpected trailing data")
	}
	return nil
}

func jsonParsePrimitive[T any](str string, convert func(any) (T, bool)) (T, bool) {
	dec := json.NewDecoder(strings.NewReader(str))
	dec.UseNumber()
	tok, err := dec.Token()
	if err != nil {
		var zero T
		return zero, false
	}
	val, ok := convert(tok)
	if !ok {
		var zero T
		return zero, false
	}
	if _, err := dec.Token(); err != io.EOF {
		var zero T
		return zero, false
	}
	return val, true
}

func jsonParseDynamic(str string) (any, bool) {
	trimmed := strings.TrimSpace(str)
	if len(trimmed) == 0 {
		return nil, false
	}
	if len(trimmed) >= 2 && trimmed[0] == '"' && trimmed[len(trimmed)-1] == '"' && !strings.ContainsRune(trimmed, '\\') {
		return trimmed[1 : len(trimmed)-1], true
	}
	if trimmed == "true" {
		return true, true
	}
	if trimmed == "false" {
		return false, true
	}
	if trimmed == "null" {
		return types.NullValue, true
	}
	if len(trimmed) > 0 && (trimmed[0] == '-' || (trimmed[0] >= '0' && trimmed[0] <= '9')) {
		if !strings.ContainsAny(trimmed, " \t\r\n,[]{}:") {
			if _, err := strconv.ParseInt(trimmed, 10, 64); err == nil {
				return json.Number(trimmed), true
			}
			if _, err := strconv.ParseUint(trimmed, 10, 64); err == nil {
				return json.Number(trimmed), true
			}
			if f, err := strconv.ParseFloat(trimmed, 64); err == nil && !math.IsNaN(f) && !math.IsInf(f, 0) {
				return json.Number(trimmed), true
			}
		}
	}
	dec := json.NewDecoder(strings.NewReader(str))
	dec.UseNumber()
	var obj any
	if err := dec.Decode(&obj); err != nil {
		return nil, false
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, false
	}
	return obj, true
}

func jsonParseString(adapter types.Adapter, str string) ref.Val {
	if len(str) > maxJSONSize {
		return types.NewErr("json parse error: string size exceeds maximum allowed limit of %d bytes", maxJSONSize)
	}
	obj, ok := jsonParseDynamic(str)
	if !ok {
		return types.OptionalNone
	}
	return types.OptionalOf(adapter.NativeToValue(obj))
}

func jsonParseWithType(adapter types.Adapter, provider types.Provider, str string, targetType ref.Type) ref.Val {
	if adapter == nil {
		adapter = types.DefaultTypeAdapter
	}
	if provAdapter, ok := provider.(types.Adapter); ok && provAdapter != nil {
		adapter = provAdapter
	}
	if len(str) > maxJSONSize {
		return types.NewErr("json parse error: string size exceeds maximum allowed limit of %d bytes", maxJSONSize)
	}
	val, ok := jsonParseValue(adapter, provider, str, targetType)
	if !ok {
		return types.OptionalNone
	}
	return types.OptionalOf(adapter.NativeToValue(val))
}

func jsonParseValue(adapter types.Adapter, provider types.Provider, str string, targetType ref.Type) (any, bool) {
	switch t := targetType.(type) {
	case *types.Type:
		switch t.Kind() {
		case types.DynKind, types.AnyKind:
			return jsonParseDynamic(str)
		case types.StringKind:
			trimmed := strings.TrimSpace(str)
			if len(trimmed) >= 2 && trimmed[0] == '"' && trimmed[len(trimmed)-1] == '"' && !strings.ContainsRune(trimmed, '\\') {
				return trimmed[1 : len(trimmed)-1], true
			}
			return jsonParsePrimitive(str, func(t any) (any, bool) { s, ok := t.(string); return s, ok })
		case types.BoolKind:
			trimmed := strings.TrimSpace(str)
			if trimmed == "true" {
				return true, true
			}
			if trimmed == "false" {
				return false, true
			}
			return false, false
		case types.IntKind:
			trimmed := strings.TrimSpace(str)
			if i, err := strconv.ParseInt(trimmed, 10, 64); err == nil {
				return i, true
			}
			return false, false
		case types.UintKind:
			trimmed := strings.TrimSpace(str)
			if u, err := strconv.ParseUint(trimmed, 10, 64); err == nil {
				return u, true
			}
			return false, false
		case types.DoubleKind:
			trimmed := strings.TrimSpace(str)
			if len(trimmed) > 0 && (trimmed[0] == '-' || (trimmed[0] >= '0' && trimmed[0] <= '9')) {
				if f, err := strconv.ParseFloat(trimmed, 64); err == nil && !math.IsNaN(f) && !math.IsInf(f, 0) {
					return f, true
				}
			}
			return false, false
		case types.BytesKind:
			return jsonParsePrimitive(str, func(t any) (any, bool) {
				s, ok := t.(string)
				if !ok {
					return nil, false
				}
				b, err := base64DecodeString(s)
				return b, err == nil
			})
		case types.NullTypeKind:
			trimmed := strings.TrimSpace(str)
			if trimmed == "null" {
				return types.NullValue, true
			}
			return false, false
		case types.TimestampKind:
			return jsonParsePrimitive(str, func(t any) (any, bool) {
				s, ok := t.(string)
				if !ok {
					return nil, false
				}
				ts, err := types.ParseTimestamp(s)
				return ts, err == nil
			})
		case types.DurationKind:
			return jsonParsePrimitive(str, func(t any) (any, bool) {
				s, ok := t.(string)
				if !ok {
					return nil, false
				}
				dur, err := time.ParseDuration(s)
				return dur, err == nil
			})
		case types.ListKind:
			elemType := types.DynType
			if len(t.Parameters()) > 0 {
				elemType = t.Parameters()[0]
			}
			return jsonParseTypedList(adapter, provider, str, elemType)
		case types.MapKind:
			elemType := types.DynType
			if len(t.Parameters()) >= 2 {
				elemType = t.Parameters()[1]
			}
			return jsonParseTypedMap(adapter, provider, str, elemType)
		case types.StructKind:
			return jsonParseStructNative(provider, str, t.TypeName())
		}
	default:
		if st, ok := targetType.(types.StructTypeDescriptor); ok {
			if rt := st.ReflectType(); rt != nil {
				return jsonUnmarshalNative(str, rt)
			}
		}
		return jsonParseStructNative(provider, str, targetType.TypeName())
	}
	return nil, false
}

func jsonParseTypedList(adapter types.Adapter, provider types.Provider, str string, elemType *types.Type) (any, bool) {
	switch elemType.Kind() {
	case types.StringKind:
		var l []string
		if jsonUnmarshalExact(str, &l) == nil {
			return l, true
		}
		return nil, false
	case types.IntKind:
		var l []int64
		if jsonUnmarshalExact(str, &l) == nil {
			return l, true
		}
		return nil, false
	case types.UintKind:
		var l []uint64
		if jsonUnmarshalExact(str, &l) == nil {
			return l, true
		}
		return nil, false
	case types.DoubleKind:
		var l []float64
		if jsonUnmarshalExact(str, &l) == nil {
			return l, true
		}
		return nil, false
	case types.BoolKind:
		var l []bool
		if jsonUnmarshalExact(str, &l) == nil {
			return l, true
		}
		return nil, false
	case types.DynKind, types.AnyKind:
		var l []any
		if jsonUnmarshalExact(str, &l) == nil {
			return l, true
		}
		return nil, false
	}
	var rawList []json.RawMessage
	if jsonUnmarshalExact(str, &rawList) != nil {
		return nil, false
	}
	elements := make([]any, len(rawList))
	for i, rawElem := range rawList {
		elem, ok := jsonParseValue(adapter, provider, string(rawElem), elemType)
		if !ok {
			return nil, false
		}
		elements[i] = elem
	}
	return elements, true
}

func jsonParseTypedMap(adapter types.Adapter, provider types.Provider, str string, valType *types.Type) (any, bool) {
	switch valType.Kind() {
	case types.StringKind:
		var m map[string]string
		if jsonUnmarshalExact(str, &m) == nil {
			return m, true
		}
		return nil, false
	case types.IntKind:
		var m map[string]int64
		if jsonUnmarshalExact(str, &m) == nil {
			return m, true
		}
		return nil, false
	case types.UintKind:
		var m map[string]uint64
		if jsonUnmarshalExact(str, &m) == nil {
			return m, true
		}
		return nil, false
	case types.DoubleKind:
		var m map[string]float64
		if jsonUnmarshalExact(str, &m) == nil {
			return m, true
		}
		return nil, false
	case types.BoolKind:
		var m map[string]bool
		if jsonUnmarshalExact(str, &m) == nil {
			return m, true
		}
		return nil, false
	case types.DynKind, types.AnyKind:
		var m map[string]any
		if jsonUnmarshalExact(str, &m) == nil {
			return m, true
		}
		return nil, false
	}
	var rawMap map[string]json.RawMessage
	if jsonUnmarshalExact(str, &rawMap) != nil {
		return nil, false
	}
	entries := make(map[string]any, len(rawMap))
	for k, rawVal := range rawMap {
		val, ok := jsonParseValue(adapter, provider, string(rawVal), valType)
		if !ok {
			return nil, false
		}
		entries[k] = val
	}
	return entries, true
}

func jsonUnmarshalNative(str string, rt reflect.Type) (any, bool) {
	if rt.Kind() == reflect.Pointer {
		rt = rt.Elem()
	}
	if rt.Kind() == reflect.Struct {
		ptr := reflect.New(rt)
		if jsonUnmarshalExact(str, ptr.Interface()) != nil {
			return nil, false
		}
		return ptr.Interface(), true
	}
	return nil, false
}

func jsonUnmarshalProto(str string, template proto.Message) (proto.Message, bool) {
	newMsg := template.ProtoReflect().New().Interface()
	if err := protojson.Unmarshal([]byte(str), newMsg); err != nil {
		return nil, false
	}
	return newMsg, true
}

type structPrototype struct {
	protoMsg proto.Message
	refType  reflect.Type
}

var (
	structPrototypeCache sync.Map // map[string]structPrototype
)

func jsonParseStructNative(provider types.Provider, str string, typeName string) (any, bool) {
	switch typeName {
	case "google.protobuf.Value":
		return jsonUnmarshalProto(str, &structpb.Value{})
	case "google.protobuf.Struct":
		return jsonUnmarshalProto(str, &structpb.Struct{})
	case "google.protobuf.ListValue":
		return jsonUnmarshalProto(str, &structpb.ListValue{})
	}
	if protoVal, ok := structPrototypeCache.Load(typeName); ok {
		p := protoVal.(structPrototype)
		if p.protoMsg != nil {
			return jsonUnmarshalProto(str, p.protoMsg)
		}
		if p.refType != nil {
			return jsonUnmarshalNative(str, p.refType)
		}
	}
	if provider != nil {
		zeroVal := provider.NewValue(typeName, map[string]ref.Val{})
		if types.IsError(zeroVal) {
			return nil, false
		}
		nativeVal := zeroVal.Value()
		if nativeVal == nil {
			return nil, false
		}
		if msg, ok := nativeVal.(proto.Message); ok {
			structPrototypeCache.Store(typeName, structPrototype{protoMsg: msg})
			return jsonUnmarshalProto(str, msg)
		}
		rt := reflect.TypeOf(nativeVal)
		if rt.Kind() == reflect.Pointer {
			rt = rt.Elem()
		}
		structPrototypeCache.Store(typeName, structPrototype{refType: rt})
		return jsonUnmarshalNative(str, rt)
	}
	return nil, false
}

func yamlEncodeValue(val ref.Val) (string, error) {
	jsonStr, err := jsonEncodeValue(val)
	if err != nil {
		return "", err
	}
	dec := json.NewDecoder(strings.NewReader(jsonStr))
	dec.UseNumber()
	var obj any
	if err := dec.Decode(&obj); err != nil {
		return "", err
	}
	cleaned := normalizeJSONNumber(obj)
	yamlBytes, err := yaml.Marshal(cleaned)
	if err != nil {
		return "", err
	}
	return string(yamlBytes), nil
}

func yamlToJSON(str string) (string, bool) {
	trimmed := strings.TrimSpace(str)
	if len(trimmed) == 0 {
		return "", false
	}
	dec := yaml.NewDecoder(strings.NewReader(str))
	var node yaml.Node
	if err := dec.Decode(&node); err != nil {
		return "", false
	}
	if len(node.Content) == 0 {
		return "", false
	}
	var extra yaml.Node
	if err := dec.Decode(&extra); err != io.EOF {
		return "", false
	}
	var obj any
	if err := node.Decode(&obj); err != nil {
		return "", false
	}
	cleaned := cleanYAMLObj(obj)
	jsonBytes, err := json.Marshal(cleaned)
	if err != nil {
		return "", false
	}
	return string(jsonBytes), true
}

func yamlParseString(adapter types.Adapter, str string) ref.Val {
	if len(str) > maxYAMLSize {
		return types.NewErr("yaml parse error: string size exceeds maximum allowed limit of %d bytes", maxYAMLSize)
	}
	jsonStr, ok := yamlToJSON(str)
	if !ok {
		return types.OptionalNone
	}
	return jsonParseString(adapter, jsonStr)
}

func yamlParseWithType(adapter types.Adapter, provider types.Provider, str string, targetType ref.Type) ref.Val {
	if len(str) > maxYAMLSize {
		return types.NewErr("yaml parse error: string size exceeds maximum allowed limit of %d bytes", maxYAMLSize)
	}
	jsonStr, ok := yamlToJSON(str)
	if !ok {
		return types.OptionalNone
	}
	return jsonParseWithType(adapter, provider, jsonStr, targetType)
}

func cleanYAMLObj(val any) any {
	switch v := val.(type) {
	case map[string]any:
		m := make(map[string]any, len(v))
		for k, elem := range v {
			m[k] = cleanYAMLObj(elem)
		}
		return m
	case map[any]any:
		m := make(map[string]any, len(v))
		for k, elem := range v {
			m[fmt.Sprint(k)] = cleanYAMLObj(elem)
		}
		return m
	case []any:
		l := make([]any, len(v))
		for i, elem := range v {
			l[i] = cleanYAMLObj(elem)
		}
		return l
	case time.Time:
		return v.Format(time.RFC3339Nano)
	default:
		return val
	}
}

func normalizeJSONNumber(val any) any {
	switch v := val.(type) {
	case json.Number:
		if i, err := v.Int64(); err == nil {
			return i
		}
		if u, err := strconv.ParseUint(string(v), 10, 64); err == nil {
			return u
		}
		if f, err := v.Float64(); err == nil {
			return f
		}
		return string(v)
	case map[string]any:
		m := make(map[string]any, len(v))
		for k, elem := range v {
			m[k] = normalizeJSONNumber(elem)
		}
		return m
	case []any:
		l := make([]any, len(v))
		for i, elem := range v {
			l[i] = normalizeJSONNumber(elem)
		}
		return l
	default:
		return val
	}
}

func base64EncodeCostModel(id string) cost.OverloadModel {
	scaled := cost.Scale(cost.Length(cost.Arg(0)), 4.0/3)
	return cost.Overload(id,
		cost.EvalCost(cost.Sum(cost.Const(cost.CallCost), cost.StringScan(cost.Length(cost.Arg(0))))),
		// base64 of n bytes is 4 * ceil(n / 3) characters, between 4n/3 and 4n/3 + 3.
		cost.ResultSize(cost.Ranged(scaled, cost.Sum(scaled, cost.Const(3)))),
		cost.BeforeVersion(1, cost.ManualEstimator(estimateEncode)),
	)
}

func base64DecodeCostModel(id string) cost.OverloadModel {
	return cost.Overload(id,
		cost.EvalCost(cost.Sum(cost.Const(cost.CallCost), cost.StringScan(cost.Length(cost.Arg(0))))),
		// Decoding n characters yields at most 3n/4 bytes. It yields fewer for padding and for the
		// carriage returns and newlines the decoder skips, so there is no lower bound. A bytes value
		// has size at least 1 under the aggregate strategy.
		cost.ResultSize(cost.AtMost(cost.AtLeastOneQuantity(cost.Scale(cost.Length(cost.Arg(0)), 0.75)))),
		cost.BeforeVersion(1, cost.ManualEstimator(estimateDecode)),
	)
}

// unboundedCostModel charges the maximum cost and reports an unbounded result size for JSON and
// YAML functions: encoding traverses an arbitrary container tree without inspecting it through the
// cost tracker, and parsing/decoding (especially with YAML anchors and aliases) can construct an
// arbitrarily deeply nested or large value graph not bounded by the input length.
func unboundedCostModel(id string) cost.OverloadModel {
	unbounded := cost.AtMost(cost.Const(math.MaxUint64))
	return cost.Overload(id, cost.EvalCost(unbounded), cost.ResultSize(unbounded))
}
