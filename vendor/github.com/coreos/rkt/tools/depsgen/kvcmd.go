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
	"github.com/coreos/rkt/tools/common"
)

const (
	kvCmd = "kv"
)

func init() {
	cmds[kvCmd] = kvDeps
}

func kvDeps(args []string) string {
	target, keysValues := kvGetArgs(args)
	return GenerateKvDeps(target, keysValues)
}

func kvGetArgs(args []string) (string, map[string]string) {
	f, target := standardFlags(kvCmd)

	f.Parse(args)
	if *target == "" {
		common.Die("--target parameter must be specified and cannot be empty")
	}
	return *target, toMap(f.Args()...)
}
