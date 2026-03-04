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

package flag

import (
	"github.com/spf13/pflag"
)

// TrackerValue wraps a non-boolean value and stores true in the provided boolean when it is set.
type TrackerValue struct {
	value    pflag.Value
	provided *bool
}

// BoolTrackerValue wraps a boolean value and stores true in the provided boolean when it is set.
type BoolTrackerValue struct {
	boolValue
	provided *bool
}

type boolValue interface {
	pflag.Value
	IsBoolFlag() bool
}

var _ pflag.Value = &TrackerValue{}
var _ boolValue = &BoolTrackerValue{}

// NewTracker returns a Value wrapping the given value which stores true in the provided boolean when it is set.
func NewTracker(value pflag.Value, provided *bool) pflag.Value {
	if value == nil {
		panic("value must not be nil")
	}

	if provided == nil {
		panic("provided boolean must not be nil")
	}

	if boolValue, ok := value.(boolValue); ok {
		return &BoolTrackerValue{boolValue: boolValue, provided: provided}
	}
	return &TrackerValue{value: value, provided: provided}
}

func (f *TrackerValue) String() string {
	return f.value.String()
}

func (f *TrackerValue) Set(value string) error {
	err := f.value.Set(value)
	if err == nil {
		*f.provided = true
	}
	return err
}

func (f *TrackerValue) Type() string {
	return f.value.Type()
}

func (f *BoolTrackerValue) Set(value string) error {
	err := f.boolValue.Set(value)
	if err == nil {
		*f.provided = true
	}

	return err
}
