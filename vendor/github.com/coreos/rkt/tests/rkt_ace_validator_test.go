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

// +build !fly,!kvm

package main

import (
	"fmt"
	"strings"
	"testing"

	"github.com/coreos/rkt/tests/testutils"
)

// Spin up the appc ACE validator pod and return the result.

// You'll find the actual application at vendor/github.com/appc/spec/ace/validator.go,
// the two ACIs are built from that

func TestAceValidator(t *testing.T) {
	newStringSet := func(strs ...string) map[string]struct{} {
		m := make(map[string]struct{}, len(strs))
		for _, s := range strs {
			m[s] = struct{}{}
		}
		return m
	}
	expected := []map[string]struct{}{
		newStringSet("prestart"),
		newStringSet("main", "sidekick"),
		// newStringSet("poststop"), // Disabled by caseyc for #2870
	}
	pattern := `ace-validator-(?:main|sidekick)\[\d+\]: ([[:alpha:]]+) OK`

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	if err := ctx.LaunchMDS(); err != nil {
		t.Fatalf("Cannot launch metadata service: %v", err)
	}

	aceMain := testutils.GetValueFromEnvOrPanic("RKT_ACE_MAIN_IMAGE")
	aceSidekick := testutils.GetValueFromEnvOrPanic("RKT_ACE_SIDEKICK_IMAGE")

	rktArgs := fmt.Sprintf("--debug --insecure-options=image run --mds-register --volume database,kind=empty %s %s",
		aceMain, aceSidekick)
	rktCmd := fmt.Sprintf("%s %s", ctx.Cmd(), rktArgs)

	child := spawnOrFail(t, rktCmd)
	defer waitOrFail(t, child, 0)

	out := ""
	for _, set := range expected {
		for len(set) > 0 {
			results, o, err := expectRegexWithOutput(child, pattern)
			out += o
			if err != nil {
				var keys []string
				for k := range set {
					keys = append(keys, fmt.Sprintf("%q", k))
				}
				ex := strings.Join(keys, " or ")
				t.Fatalf("Expected %s, but not found: %v\nOutput: %v", ex, err, out)
			}
			if len(results) != 2 {
				t.Fatalf("Unexpected regex results, expected a whole match and one submatch, got %#v", results)
			}
			aceStage := results[1]
			if _, ok := set[aceStage]; ok {
				t.Logf("Got expected ACE stage %q", aceStage)
				delete(set, aceStage)
			} else {
				t.Logf("Ignoring unknown ACE stage %q", aceStage)
			}
		}
	}
}
