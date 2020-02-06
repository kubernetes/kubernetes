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
	"path/filepath"
	"strings"

	"github.com/bazelbuild/bazel-gazelle/language"
	"github.com/bazelbuild/bazel-gazelle/rule"
	"golang.org/x/sync/errgroup"
)

// UpdateRepos generates go_repository rules corresponding to modules in
// args.Imports. Each module argument may specify a version with an '@' suffix
// (in the same format as 'go get'). If no version is specified, @latest
// is requested.
func (*goLang) UpdateRepos(args language.UpdateReposArgs) language.UpdateReposResult {
	gen := make([]*rule.Rule, len(args.Imports))
	var eg errgroup.Group
	for i := range args.Imports {
		i := i
		eg.Go(func() error {
			arg := args.Imports[i]
			modPath, query := arg, "latest"
			if i := strings.IndexByte(arg, '@'); i >= 0 {
				modPath, query = arg[:i], arg[i+1:]
			}
			name, version, sum, err := args.Cache.ModVersion(modPath, query)
			if err != nil {
				return err
			}
			gen[i] = rule.NewRule("go_repository", name)
			gen[i].SetAttr("importpath", modPath)
			gen[i].SetAttr("version", version)
			gen[i].SetAttr("sum", sum)
			setBuildAttrs(getGoConfig(args.Config), gen[i])
			return nil
		})
	}
	if err := eg.Wait(); err != nil {
		return language.UpdateReposResult{Error: err}
	}
	return language.UpdateReposResult{Gen: gen}
}

var repoImportFuncs = map[string]func(args language.ImportReposArgs) language.ImportReposResult{
	"Gopkg.lock":  importReposFromDep,
	"go.mod":      importReposFromModules,
	"Godeps.json": importReposFromGodep,
}

func (*goLang) CanImport(path string) bool {
	return repoImportFuncs[filepath.Base(path)] != nil
}

func (*goLang) ImportRepos(args language.ImportReposArgs) language.ImportReposResult {
	res := repoImportFuncs[filepath.Base(args.Path)](args)
	for _, r := range res.Gen {
		setBuildAttrs(getGoConfig(args.Config), r)
	}
	if args.Prune {
		genNamesSet := make(map[string]bool)
		for _, r := range res.Gen {
			genNamesSet[r.Name()] = true
		}
		for _, r := range args.Config.Repos {
			if name := r.Name(); r.Kind() == "go_repository" && !genNamesSet[name] {
				res.Empty = append(res.Empty, rule.NewRule("go_repository", name))
			}
		}
	}
	return res
}

func setBuildAttrs(gc *goConfig, r *rule.Rule) {
	if gc.buildExternalAttr != "" {
		r.SetAttr("build_external", gc.buildExternalAttr)
	}
	if gc.buildFileNamesAttr != "" {
		r.SetAttr("build_file_name", gc.buildFileNamesAttr)
	}
	if gc.buildFileGenerationAttr != "" {
		r.SetAttr("build_file_generation", gc.buildFileGenerationAttr)
	}
	if gc.buildTagsAttr != "" {
		r.SetAttr("build_tags", gc.buildTagsAttr)
	}
	if gc.buildFileProtoModeAttr != "" {
		r.SetAttr("build_file_proto_mode", gc.buildFileProtoModeAttr)
	}
	if gc.buildExtraArgsAttr != "" {
		extraArgs := strings.Split(gc.buildExtraArgsAttr, ",")
		r.SetAttr("build_extra_args", extraArgs)
	}
}
