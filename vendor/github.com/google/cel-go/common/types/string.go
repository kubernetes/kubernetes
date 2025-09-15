// Copyright 2018 Google LLC
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

package types

import (
	"fmt"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/google/cel-go/common/overloads"
	"github.com/google/cel-go/common/types/ref"

	anypb "google.golang.org/protobuf/types/known/anypb"
	structpb "google.golang.org/protobuf/types/known/structpb"
	wrapperspb "google.golang.org/protobuf/types/known/wrapperspb"
)

// String type implementation which supports addition, comparison, matching,
// and size functions.
type String string

var (
	stringOneArgOverloads = map[string]func(ref.Val, ref.Val) ref.Val{
		overloads.Contains:   StringContains,
		overloads.EndsWith:   StringEndsWith,
		overloads.StartsWith: StringStartsWith,
	}

	stringWrapperType = reflect.TypeOf(&wrapperspb.StringValue{})
)

// Add implements traits.Adder.Add.
func (s String) Add(other ref.Val) ref.Val {
	otherString, ok := other.(String)
	if !ok {
		return MaybeNoSuchOverloadErr(other)
	}
	return s + otherString
}

// Compare implements traits.Comparer.Compare.
func (s String) Compare(other ref.Val) ref.Val {
	otherString, ok := other.(String)
	if !ok {
		return MaybeNoSuchOverloadErr(other)
	}
	return Int(strings.Compare(s.Value().(string), otherString.Value().(string)))
}

// ConvertToNative implements ref.Val.ConvertToNative.
func (s String) ConvertToNative(typeDesc reflect.Type) (any, error) {
	switch typeDesc.Kind() {
	case reflect.String:
		return reflect.ValueOf(s).Convert(typeDesc).Interface(), nil
	case reflect.Ptr:
		switch typeDesc {
		case anyValueType:
			// Primitives must be wrapped before being set on an Any field.
			return anypb.New(wrapperspb.String(string(s)))
		case jsonValueType:
			// Convert to a protobuf representation of a JSON String.
			return structpb.NewStringValue(string(s)), nil
		case stringWrapperType:
			// Convert to a wrapperspb.StringValue.
			return wrapperspb.String(string(s)), nil
		}
		if typeDesc.Elem().Kind() == reflect.String {
			p := s.Value().(string)
			return &p, nil
		}
	case reflect.Interface:
		sv := s.Value()
		if reflect.TypeOf(sv).Implements(typeDesc) {
			return sv, nil
		}
		if reflect.TypeOf(s).Implements(typeDesc) {
			return s, nil
		}
	}
	return nil, fmt.Errorf(
		"unsupported native conversion from string to '%v'", typeDesc)
}

// ConvertToType implements ref.Val.ConvertToType.
func (s String) ConvertToType(typeVal ref.Type) ref.Val {
	switch typeVal {
	case IntType:
		if n, err := strconv.ParseInt(s.Value().(string), 10, 64); err == nil {
			return Int(n)
		}
	case UintType:
		if n, err := strconv.ParseUint(s.Value().(string), 10, 64); err == nil {
			return Uint(n)
		}
	case DoubleType:
		if n, err := strconv.ParseFloat(s.Value().(string), 64); err == nil {
			return Double(n)
		}
	case BoolType:
		if b, err := strconv.ParseBool(s.Value().(string)); err == nil {
			return Bool(b)
		}
	case BytesType:
		return Bytes(s)
	case DurationType:
		if d, err := time.ParseDuration(s.Value().(string)); err == nil {
			return durationOf(d)
		}
	case TimestampType:
		if t, err := time.Parse(time.RFC3339, s.Value().(string)); err == nil {
			if t.Unix() < minUnixTime || t.Unix() > maxUnixTime {
				return celErrTimestampOverflow
			}
			return timestampOf(t)
		}
	case StringType:
		return s
	case TypeType:
		return StringType
	}
	return NewErr("type conversion error from '%s' to '%s'", StringType, typeVal)
}

// Equal implements ref.Val.Equal.
func (s String) Equal(other ref.Val) ref.Val {
	otherString, ok := other.(String)
	return Bool(ok && s == otherString)
}

// IsZeroValue returns true if the string is empty.
func (s String) IsZeroValue() bool {
	return len(s) == 0
}

// Match implements traits.Matcher.Match.
func (s String) Match(pattern ref.Val) ref.Val {
	pat, ok := pattern.(String)
	if !ok {
		return MaybeNoSuchOverloadErr(pattern)
	}
	matched, err := regexp.MatchString(pat.Value().(string), s.Value().(string))
	if err != nil {
		return &Err{error: err}
	}
	return Bool(matched)
}

// Receive implements traits.Receiver.Receive.
func (s String) Receive(function string, overload string, args []ref.Val) ref.Val {
	switch len(args) {
	case 1:
		if f, found := stringOneArgOverloads[function]; found {
			return f(s, args[0])
		}
	}
	return NoSuchOverloadErr()
}

// Size implements traits.Sizer.Size.
func (s String) Size() ref.Val {
	return Int(len([]rune(s.Value().(string))))
}

// Type implements ref.Val.Type.
func (s String) Type() ref.Type {
	return StringType
}

// Value implements ref.Val.Value.
func (s String) Value() any {
	return string(s)
}

func (s String) format(sb *strings.Builder) {
	sb.WriteString(strconv.Quote(string(s)))
}

// StringContains returns whether the string contains a substring.
func StringContains(s, sub ref.Val) ref.Val {
	str, ok := s.(String)
	if !ok {
		return MaybeNoSuchOverloadErr(s)
	}
	subStr, ok := sub.(String)
	if !ok {
		return MaybeNoSuchOverloadErr(sub)
	}
	return Bool(strings.Contains(string(str), string(subStr)))
}

// StringEndsWith returns whether the target string contains the input suffix.
func StringEndsWith(s, suf ref.Val) ref.Val {
	str, ok := s.(String)
	if !ok {
		return MaybeNoSuchOverloadErr(s)
	}
	sufStr, ok := suf.(String)
	if !ok {
		return MaybeNoSuchOverloadErr(suf)
	}
	return Bool(strings.HasSuffix(string(str), string(sufStr)))
}

// StringStartsWith returns whether the target string contains the input prefix.
func StringStartsWith(s, pre ref.Val) ref.Val {
	str, ok := s.(String)
	if !ok {
		return MaybeNoSuchOverloadErr(s)
	}
	preStr, ok := pre.(String)
	if !ok {
		return MaybeNoSuchOverloadErr(pre)
	}
	return Bool(strings.HasPrefix(string(str), string(preStr)))
}
