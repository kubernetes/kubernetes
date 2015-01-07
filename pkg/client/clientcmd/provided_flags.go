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

package clientcmd

import (
	"fmt"
	"strconv"

	"github.com/spf13/pflag"
)

// FlagProvider adds a check for whether .Set was called on this flag variable
type FlagProvider interface {
	// Provided returns true iff .Set was called on this flag
	Provided() bool
	pflag.Value
}

// StringFlag implements FlagProvider
type StringFlag struct {
	Default     string
	Value       string
	WasProvided bool
}

// SetDefault sets a default value for a flag while keeping Provided() false
func (flag *StringFlag) SetDefault(value string) {
	flag.Value = value
	flag.WasProvided = false
}

func (flag *StringFlag) Set(value string) error {
	flag.Value = value
	flag.WasProvided = true

	return nil
}

func (flag *StringFlag) Type() string {
	return "string"
}

func (flag *StringFlag) Provided() bool {
	return flag.WasProvided
}

func (flag *StringFlag) String() string {
	return flag.Value
}

// BoolFlag implements FlagProvider
type BoolFlag struct {
	Default     bool
	Value       bool
	WasProvided bool
}

// SetDefault sets a default value for a flag while keeping Provided() false
func (flag *BoolFlag) SetDefault(value bool) {
	flag.Value = value
	flag.WasProvided = false
}

func (flag *BoolFlag) Set(value string) error {
	boolValue, err := strconv.ParseBool(value)
	if err != nil {
		return err
	}

	flag.Value = boolValue
	flag.WasProvided = true

	return nil
}

func (flag *BoolFlag) Type() string {
	return "bool"
}

func (flag *BoolFlag) Provided() bool {
	return flag.WasProvided
}

func (flag *BoolFlag) String() string {
	return fmt.Sprintf("%t", flag.Value)
}
