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

	"github.com/bazelbuild/bazel-gazelle/config"
	gzflag "github.com/bazelbuild/bazel-gazelle/flag"
	"github.com/bazelbuild/bazel-gazelle/rule"
)

// TODO(#472): store location information to validate each exclude. They
// may be set in one directory and used in another. Excludes work on
// declared generated files, so we can't just stat.

type walkConfig struct {
	excludes []string
	ignore   bool
	follow   []string
}

const walkName = "_walk"

func getWalkConfig(c *config.Config) *walkConfig {
	return c.Exts[walkName].(*walkConfig)
}

func (wc *walkConfig) isExcluded(rel, base string) bool {
	if base == ".git" {
		return true
	}
	f := path.Join(rel, base)
	for _, x := range wc.excludes {
		if f == x {
			return true
		}
	}
	return false
}

type Configurer struct{}

func (_ *Configurer) RegisterFlags(fs *flag.FlagSet, cmd string, c *config.Config) {
	wc := &walkConfig{}
	c.Exts[walkName] = wc
	fs.Var(&gzflag.MultiFlag{Values: &wc.excludes}, "exclude", "Path to file or directory that should be ignored (may be repeated)")
}

func (_ *Configurer) CheckFlags(fs *flag.FlagSet, c *config.Config) error { return nil }

func (_ *Configurer) KnownDirectives() []string {
	return []string{"exclude", "follow", "ignore"}
}

func (_ *Configurer) Configure(c *config.Config, rel string, f *rule.File) {
	wc := getWalkConfig(c)
	wcCopy := &walkConfig{}
	*wcCopy = *wc
	wcCopy.ignore = false

	if f != nil {
		for _, d := range f.Directives {
			switch d.Key {
			case "exclude":
				wcCopy.excludes = append(wcCopy.excludes, path.Join(rel, d.Value))
			case "follow":
				wcCopy.follow = append(wcCopy.follow, path.Join(rel, d.Value))
			case "ignore":
				wcCopy.ignore = true
			}
		}
	}

	c.Exts[walkName] = wcCopy
}
