// Copyright 2015 The rkt Authors
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

package main

import (
	"flag"
	"os"
	"path/filepath"
	"strings"

	"github.com/coreos/rkt/tools/common"
)

// toMap creates a map from passed strings. This function expects an
// even number of strings, otherwise it will bail out. Odd (first,
// third and so on) strings are keys, even (second, fourth and so on)
// strings are values.
func toMap(kv ...string) map[string]string {
	if len(kv)%2 != 0 {
		common.Die("Expected even number of strings in toMap")
	}
	r := make(map[string]string, len(kv))
	lastKey := ""
	for i, kv := range kv {
		if i%2 == 0 {
			lastKey = kv
		} else {
			r[lastKey] = kv
		}
	}
	return r
}

// appName returns application name, like depsgen
func appName() string {
	return filepath.Base(os.Args[0])
}

// replacePlaceholders replaces placeholders with values in kv in
// initial str. Placeholders are in form of !!!FOO!!!, but those
// passed here should be without exclamation marks.
func replacePlaceholders(str string, kv ...string) string {
	for ph, value := range toMap(kv...) {
		str = strings.Replace(str, "!!!"+ph+"!!!", value, -1)
	}
	return str
}

// standardFlags returns a new flag set with target flag already set up
func standardFlags(cmd string) (*flag.FlagSet, *string) {
	f := flag.NewFlagSet(appName()+" "+cmd, flag.ExitOnError)
	target := f.String("target", "", "Make target (example: $(FOO_BINARY))")
	return f, target
}
