// Copyright 2018 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package util

import (
	"io/ioutil"
	"strconv"
	"strings"
)

// ParseUint32s parses a slice of strings into a slice of uint32s.
func ParseUint32s(ss []string) ([]uint32, error) {
	us := make([]uint32, 0, len(ss))
	for _, s := range ss {
		u, err := strconv.ParseUint(s, 10, 32)
		if err != nil {
			return nil, err
		}

		us = append(us, uint32(u))
	}

	return us, nil
}

// ParseUint64s parses a slice of strings into a slice of uint64s.
func ParseUint64s(ss []string) ([]uint64, error) {
	us := make([]uint64, 0, len(ss))
	for _, s := range ss {
		u, err := strconv.ParseUint(s, 10, 64)
		if err != nil {
			return nil, err
		}

		us = append(us, u)
	}

	return us, nil
}

// ParsePInt64s parses a slice of strings into a slice of int64 pointers.
func ParsePInt64s(ss []string) ([]*int64, error) {
	us := make([]*int64, 0, len(ss))
	for _, s := range ss {
		u, err := strconv.ParseInt(s, 10, 64)
		if err != nil {
			return nil, err
		}

		us = append(us, &u)
	}

	return us, nil
}

// ReadUintFromFile reads a file and attempts to parse a uint64 from it.
func ReadUintFromFile(path string) (uint64, error) {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return 0, err
	}
	return strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
}

// ReadIntFromFile reads a file and attempts to parse a int64 from it.
func ReadIntFromFile(path string) (int64, error) {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return 0, err
	}
	return strconv.ParseInt(strings.TrimSpace(string(data)), 10, 64)
}

// ParseBool parses a string into a boolean pointer.
func ParseBool(b string) *bool {
	var truth bool
	switch b {
	case "enabled":
		truth = true
	case "disabled":
		truth = false
	default:
		return nil
	}
	return &truth
}
