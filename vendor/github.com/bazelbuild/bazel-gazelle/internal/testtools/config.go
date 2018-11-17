/* Copyright 2018 The Bazel Authors. All rights reserved.

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

package testtools

import (
	"flag"
	"testing"

	"github.com/bazelbuild/bazel-gazelle/internal/config"
	"github.com/bazelbuild/bazel-gazelle/internal/language"
)

// NewTestConfig returns a Config used for tests in any language extension.
// cexts is a list of configuration extensions to use. langs is a list of
// language extensions to use (languages are also configuration extensions,
// but it may be convenient to keep them separate). args is a list of
// command line arguments to apply. NewTestConfig calls t.Fatal if any
// error is encountered while processing flags.
func NewTestConfig(t *testing.T, cexts []config.Configurer, langs []language.Language, args []string) *config.Config {
	c := config.New()
	fs := flag.NewFlagSet("test", flag.ContinueOnError)

	for _, lang := range langs {
		cexts = append(cexts, lang)
	}
	for _, cext := range cexts {
		cext.RegisterFlags(fs, "update", c)
	}

	if err := fs.Parse(args); err != nil {
		t.Fatal(err)
	}
	for _, cext := range cexts {
		if err := cext.CheckFlags(fs, c); err != nil {
			t.Fatal(err)
		}
	}

	return c
}
