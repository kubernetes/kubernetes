/*
Copyright 2014 Google Inc. All rights reserved.

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

package util

import (
	"encoding/json"
	"fmt"
	"regexp"
	"runtime"
	"time"

	"github.com/golang/glog"
)

// For testing, bypass HandleCrash.
var ReallyCrash bool

// HandleCrash simply catches a crash and logs an error. Meant to be called via defer.
func HandleCrash() {
	if ReallyCrash {
		return
	}

	r := recover()
	if r != nil {
		callers := ""
		for i := 0; true; i++ {
			_, file, line, ok := runtime.Caller(i)
			if !ok {
				break
			}
			callers = callers + fmt.Sprintf("%v:%v\n", file, line)
		}
		glog.Infof("Recovered from panic: %#v (%v)\n%v", r, r, callers)
	}
}

// Forever loops forever running f every d.  Catches any panics, and keeps going.
func Forever(f func(), period time.Duration) {
	for {
		func() {
			defer HandleCrash()
			f()
		}()
		time.Sleep(period)
	}
}

// IntOrString is a type that can hold an int or a string.  When used in
// JSON or YAML marshalling and unmarshalling, it produces or consumes the
// inner type.  This allows you to have, for example, a JSON field that can
// accept a name or number.
type IntOrString struct {
	Kind   IntstrKind
	IntVal int
	StrVal string
}

// IntstrKind represents the stored type of IntOrString.
type IntstrKind int

const (
	IntstrInt    IntstrKind = iota // The IntOrString holds an int.
	IntstrString                   // The IntOrString holds a string.
)

// NewIntOrStringFromInt creates an IntOrString object with an int value.
func NewIntOrStringFromInt(val int) IntOrString {
	return IntOrString{Kind: IntstrInt, IntVal: val}
}

// NewIntOrStringFromString creates an IntOrString object with a string value.
func NewIntOrStringFromString(val string) IntOrString {
	return IntOrString{Kind: IntstrString, StrVal: val}
}

// UnmarshalYAML implements the yaml.Unmarshaler interface.
func (intstr *IntOrString) UnmarshalYAML(unmarshal func(interface{}) error) error {
	fmt.Printf("THREE UNMARSHAL INTORSTRING\n")
	var value interface{}
	if err := unmarshal(&value); err != nil {
		fmt.Printf("IOSERR1: %s", err)
		return err
	}
	switch v := value.(type) {
	case int:
		fmt.Printf("THREE UNMARSHAL INT FOUND: %v\n", v)
		intstr.Kind = IntstrInt
		intstr.IntVal = v
		return nil
	case string:
		fmt.Printf("THREE UNMARSHAL STRING FOUND: %v\n", v)
		intstr.Kind = IntstrString
		intstr.StrVal = v
		return nil
	}
	err := fmt.Errorf("invalid type %T for value %q", value, value)
	fmt.Printf("IOSERR2: %s", err)
	return err
}

// MarshalYAML implements the yaml.Marshaler interface.
func (intstr IntOrString) MarshalYAML() (interface{}, error) {
	switch intstr.Kind {
	case IntstrInt:
		return intstr.IntVal, nil
	case IntstrString:
		return intstr.StrVal, nil
	}
	return nil, fmt.Errorf("invalid IntOrString.Kind: %v", intstr.Kind)
}

// UnmarshalJSON implements the json.Unmarshaller interface.
func (intstr *IntOrString) UnmarshalJSON(value []byte) error {
	if value[0] == '"' {
		intstr.Kind = IntstrString
		return json.Unmarshal(value, &intstr.StrVal)
	}
	intstr.Kind = IntstrInt
	return json.Unmarshal(value, &intstr.IntVal)
}

// MarshalJSON implements the json.Marshaller interface.
func (intstr IntOrString) MarshalJSON() ([]byte, error) {
	switch intstr.Kind {
	case IntstrInt:
		return json.Marshal(intstr.IntVal)
	case IntstrString:
		return json.Marshal(intstr.StrVal)
	default:
		return []byte{}, fmt.Errorf("impossible IntOrString.Kind")
	}
}

// Takes a list of strings and compiles them into a list of regular expressions
func CompileRegexps(regexpStrings []string) ([]*regexp.Regexp, error) {
	regexps := []*regexp.Regexp{}
	for _, regexpStr := range regexpStrings {
		r, err := regexp.Compile(regexpStr)
		if err != nil {
			return []*regexp.Regexp{}, err
		}
		regexps = append(regexps, r)
	}
	return regexps, nil
}
