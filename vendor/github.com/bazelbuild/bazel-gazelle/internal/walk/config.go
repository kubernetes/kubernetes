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

package walk

import (
	"flag"
	"path"
	"strings"

	"github.com/bazelbuild/bazel-gazelle/internal/config"
	"github.com/bazelbuild/bazel-gazelle/internal/rule"
)

type walkConfig struct {
	excludes []string
	ignore   bool
}

const walkName = "_walk"

func getWalkConfig(c *config.Config) walkConfig {
	return c.Exts[walkName].(walkConfig)
}

func (wc *walkConfig) isExcluded(base string) bool {
	for _, x := range wc.excludes {
		if base == x {
			return true
		}
	}
	return false
}

type walkConfigurer struct{}

func (_ *walkConfigurer) RegisterFlags(fs *flag.FlagSet, cmd string, c *config.Config) {}

func (_ *walkConfigurer) CheckFlags(fs *flag.FlagSet, c *config.Config) error { return nil }

func (_ *walkConfigurer) KnownDirectives() []string {
	return []string{"exclude", "ignore"}
}

func (_ *walkConfigurer) Configure(c *config.Config, rel string, f *rule.File) {
	var wc walkConfig
	if raw, ok := c.Exts[walkName]; ok {
		wc = raw.(walkConfig)
		wc.ignore = false
		if rel != "" {
			prefix := path.Base(rel) + "/"
			excludes := make([]string, 0, len(wc.excludes))
			for _, x := range wc.excludes {
				if strings.HasPrefix(x, prefix) {
					excludes = append(excludes, x[len(prefix):])
				}
			}
			wc.excludes = excludes
		}
	}

	if f != nil {
		for _, d := range f.Directives {
			switch d.Key {
			case "exclude":
				wc.excludes = append(wc.excludes, d.Value)
			case "ignore":
				wc.ignore = true
			}
		}
	}

	c.Exts[walkName] = wc
}
