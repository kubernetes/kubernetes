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
	"strconv"
	"strings"
	"time"

	"github.com/google/cel-go/common/overloads"
	"github.com/google/cel-go/common/types/ref"

	anypb "google.golang.org/protobuf/types/known/anypb"
	dpb "google.golang.org/protobuf/types/known/durationpb"
	structpb "google.golang.org/protobuf/types/known/structpb"
)

// Duration type that implements ref.Val and supports add, compare, negate,
// and subtract operators. This type is also a receiver which means it can
// participate in dispatch to receiver functions.
type Duration struct {
	time.Duration
}

func durationOf(d time.Duration) Duration {
	return Duration{Duration: d}
}

var (
	durationValueType = reflect.TypeOf(&dpb.Duration{})

	durationZeroArgOverloads = map[string]func(ref.Val) ref.Val{
		overloads.TimeGetHours:        DurationGetHours,
		overloads.TimeGetMinutes:      DurationGetMinutes,
		overloads.TimeGetSeconds:      DurationGetSeconds,
		overloads.TimeGetMilliseconds: DurationGetMilliseconds,
	}
)

// Add implements traits.Adder.Add.
func (d Duration) Add(other ref.Val) ref.Val {
	switch other.Type() {
	case DurationType:
		dur2 := other.(Duration)
		val, err := addDurationChecked(d.Duration, dur2.Duration)
		if err != nil {
			return WrapErr(err)
		}
		return durationOf(val)
	case TimestampType:
		ts := other.(Timestamp).Time
		val, err := addTimeDurationChecked(ts, d.Duration)
		if err != nil {
			return WrapErr(err)
		}
		return timestampOf(val)
	}
	return MaybeNoSuchOverloadErr(other)
}

// Compare implements traits.Comparer.Compare.
func (d Duration) Compare(other ref.Val) ref.Val {
	otherDur, ok := other.(Duration)
	if !ok {
		return MaybeNoSuchOverloadErr(other)
	}
	d1 := d.Duration
	d2 := otherDur.Duration
	switch {
	case d1 < d2:
		return IntNegOne
	case d1 > d2:
		return IntOne
	default:
		return IntZero
	}
}

// ConvertToNative implements ref.Val.ConvertToNative.
func (d Duration) ConvertToNative(typeDesc reflect.Type) (any, error) {
	// If the duration is already assignable to the desired type return it.
	if reflect.TypeOf(d.Duration).AssignableTo(typeDesc) {
		return d.Duration, nil
	}
	if reflect.TypeOf(d).AssignableTo(typeDesc) {
		return d, nil
	}
	switch typeDesc {
	case anyValueType:
		// Pack the duration as a dpb.Duration into an Any value.
		return anypb.New(dpb.New(d.Duration))
	case durationValueType:
		// Unwrap the CEL value to its underlying proto value.
		return dpb.New(d.Duration), nil
	case JSONValueType:
		// CEL follows the proto3 to JSON conversion.
		// Note, using jsonpb would wrap the result in extra double quotes.
		v := d.ConvertToType(StringType)
		if IsError(v) {
			return nil, v.(*Err)
		}
		return structpb.NewStringValue(string(v.(String))), nil
	}
	return nil, fmt.Errorf("type conversion error from 'Duration' to '%v'", typeDesc)
}

// ConvertToType implements ref.Val.ConvertToType.
func (d Duration) ConvertToType(typeVal ref.Type) ref.Val {
	switch typeVal {
	case StringType:
		return String(strconv.FormatFloat(d.Seconds(), 'f', -1, 64) + "s")
	case IntType:
		return Int(d.Duration)
	case DurationType:
		return d
	case TypeType:
		return DurationType
	}
	return NewErr("type conversion error from '%s' to '%s'", DurationType, typeVal)
}

// Equal implements ref.Val.Equal.
func (d Duration) Equal(other ref.Val) ref.Val {
	otherDur, ok := other.(Duration)
	return Bool(ok && d.Duration == otherDur.Duration)
}

// IsZeroValue returns true if the duration value is zero
func (d Duration) IsZeroValue() bool {
	return d.Duration == 0
}

// Negate implements traits.Negater.Negate.
func (d Duration) Negate() ref.Val {
	val, err := negateDurationChecked(d.Duration)
	if err != nil {
		return WrapErr(err)
	}
	return durationOf(val)
}

// Receive implements traits.Receiver.Receive.
func (d Duration) Receive(function string, overload string, args []ref.Val) ref.Val {
	if len(args) == 0 {
		if f, found := durationZeroArgOverloads[function]; found {
			return f(d)
		}
	}
	return NoSuchOverloadErr()
}

// Subtract implements traits.Subtractor.Subtract.
func (d Duration) Subtract(subtrahend ref.Val) ref.Val {
	subtraDur, ok := subtrahend.(Duration)
	if !ok {
		return MaybeNoSuchOverloadErr(subtrahend)
	}
	val, err := subtractDurationChecked(d.Duration, subtraDur.Duration)
	if err != nil {
		return WrapErr(err)
	}
	return durationOf(val)
}

// Type implements ref.Val.Type.
func (d Duration) Type() ref.Type {
	return DurationType
}

// Value implements ref.Val.Value.
func (d Duration) Value() any {
	return d.Duration
}

func (d Duration) format(sb *strings.Builder) {
	fmt.Fprintf(sb, `duration("%ss")`, strconv.FormatFloat(d.Seconds(), 'f', -1, 64))
}

// DurationGetHours returns the duration in hours.
func DurationGetHours(val ref.Val) ref.Val {
	dur, ok := val.(Duration)
	if !ok {
		return MaybeNoSuchOverloadErr(val)
	}
	return Int(dur.Hours())
}

// DurationGetMinutes returns duration in minutes.
func DurationGetMinutes(val ref.Val) ref.Val {
	dur, ok := val.(Duration)
	if !ok {
		return MaybeNoSuchOverloadErr(val)
	}
	return Int(dur.Minutes())
}

// DurationGetSeconds returns duration in seconds.
func DurationGetSeconds(val ref.Val) ref.Val {
	dur, ok := val.(Duration)
	if !ok {
		return MaybeNoSuchOverloadErr(val)
	}
	return Int(dur.Seconds())
}

// DurationGetMilliseconds returns duration in milliseconds.
func DurationGetMilliseconds(val ref.Val) ref.Val {
	dur, ok := val.(Duration)
	if !ok {
		return MaybeNoSuchOverloadErr(val)
	}
	return Int(dur.Milliseconds())
}
