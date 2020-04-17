/* Copyright 2019 The Bazel Authors. All rights reserved.

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

package golang

import (
	"encoding/json"
	"fmt"
	"io/ioutil"

	"github.com/bazelbuild/bazel-gazelle/label"
	"github.com/bazelbuild/bazel-gazelle/language"
	"github.com/bazelbuild/bazel-gazelle/rule"
	"golang.org/x/sync/errgroup"
)

type goDepLockFile struct {
	ImportPath   string
	GoVersion    string
	GodepVersion string
	Packages     []string
	Deps         []goDepProject
}

type goDepProject struct {
	ImportPath string
	Rev        string
}

func importReposFromGodep(args language.ImportReposArgs) language.ImportReposResult {
	data, err := ioutil.ReadFile(args.Path)
	if err != nil {
		return language.ImportReposResult{Error: err}
	}

	file := goDepLockFile{}
	if err := json.Unmarshal(data, &file); err != nil {
		return language.ImportReposResult{Error: err}
	}

	var eg errgroup.Group
	roots := make([]string, len(file.Deps))
	for i := range file.Deps {
		i := i
		eg.Go(func() error {
			p := file.Deps[i]
			repoRoot, _, err := args.Cache.Root(p.ImportPath)
			if err != nil {
				return err
			}
			roots[i] = repoRoot
			return nil
		})
	}
	if err := eg.Wait(); err != nil {
		return language.ImportReposResult{Error: err}
	}

	gen := make([]*rule.Rule, 0, len(file.Deps))
	repoToRev := make(map[string]string)
	for i, p := range file.Deps {
		repoRoot := roots[i]
		if rev, ok := repoToRev[repoRoot]; !ok {
			r := rule.NewRule("go_repository", label.ImportPathToBazelRepoName(repoRoot))
			r.SetAttr("importpath", repoRoot)
			r.SetAttr("commit", p.Rev)
			repoToRev[repoRoot] = p.Rev
			gen = append(gen, r)
		} else {
			if p.Rev != rev {
				return language.ImportReposResult{Error: fmt.Errorf("repo %s imported at multiple revisions: %s, %s", repoRoot, p.Rev, rev)}
			}
		}
	}
	return language.ImportReposResult{Gen: gen}
}
