// Copyright 2018 The etcd Authors
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

package flags

import (
	"flag"
	"fmt"
	"sort"
	"strings"
)

// UniqueStringsValue wraps a list of unique strings.
// The values are set in order.
type UniqueStringsValue struct {
	Values map[string]struct{}
}

// Set parses a command line set of strings, separated by comma.
// Implements "flag.Value" interface.
// The values are set in order.
func (us *UniqueStringsValue) Set(s string) error {
	values := strings.Split(s, ",")
	us.Values = make(map[string]struct{}, len(values))
	for _, v := range values {
		us.Values[v] = struct{}{}
	}
	return nil
}

// String implements "flag.Value" interface.
func (us *UniqueStringsValue) String() string {
	return strings.Join(us.stringSlice(), ",")
}

func (us *UniqueStringsValue) stringSlice() []string {
	ss := make([]string, 0, len(us.Values))
	for v := range us.Values {
		ss = append(ss, v)
	}
	sort.Strings(ss)
	return ss
}

// NewUniqueStringsValue implements string slice as "flag.Value" interface.
// Given value is to be separated by comma.
// The values are set in order.
func NewUniqueStringsValue(s string) (us *UniqueStringsValue) {
	us = &UniqueStringsValue{Values: make(map[string]struct{})}
	if s == "" {
		return us
	}
	if err := us.Set(s); err != nil {
		panic(fmt.Sprintf("new UniqueStringsValue should never fail: %v", err))
	}
	return us
}

// UniqueStringsFromFlag returns a string slice from the flag.
func UniqueStringsFromFlag(fs *flag.FlagSet, flagName string) []string {
	return (*fs.Lookup(flagName).Value.(*UniqueStringsValue)).stringSlice()
}

// UniqueStringsMapFromFlag returns a map of strings from the flag.
func UniqueStringsMapFromFlag(fs *flag.FlagSet, flagName string) map[string]struct{} {
	return (*fs.Lookup(flagName).Value.(*UniqueStringsValue)).Values
}
