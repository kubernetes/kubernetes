/*
Copyright 2017 The Kubernetes Authors.

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

package main

import (
	"fmt"
	"io/ioutil"
	"path/filepath"

	"github.com/bazelbuild/buildtools/build"
)

const (
	pkgSrcsTarget = "package-srcs"
	allSrcsTarget = "all-srcs"
)

// walkSource walks the source tree recursively from pkgPath, adding
// any BUILD files to v.newRules to be formatted.
//
// If AddSourcesRules is enabled in the kazel config, then we additionally add
// package-sources and recursive all-srcs filegroups rules to every BUILD file.
//
// Returns the list of children all-srcs targets that should be added to the
// all-srcs rule of the enclosing package.
func (v *Vendorer) walkSource(pkgPath string) ([]string, error) {
	// clean pkgPath since we access v.newRules directly
	pkgPath = filepath.Clean(pkgPath)
	for _, r := range v.skippedPaths {
		if r.MatchString(pkgPath) {
			return nil, nil
		}
	}
	files, err := ioutil.ReadDir(pkgPath)
	if err != nil {
		return nil, err
	}

	// Find any children packages we need to include in an all-srcs rule.
	var children []string
	for _, f := range files {
		if f.IsDir() {
			c, err := v.walkSource(filepath.Join(pkgPath, f.Name()))
			if err != nil {
				return nil, err
			}
			children = append(children, c...)
		}
	}

	// This path is a package either if we've added rules or if a BUILD file already exists.
	_, hasRules := v.newRules[pkgPath]
	isPkg := hasRules
	if !isPkg {
		isPkg, _ = findBuildFile(pkgPath)
	}

	if !isPkg {
		// This directory isn't a package (doesn't contain a BUILD file),
		// but there might be subdirectories that are packages,
		// so pass that up to our parent.
		return children, nil
	}

	// Enforce formatting the BUILD file, even if we're not adding srcs rules
	if !hasRules {
		v.addRules(pkgPath, nil)
	}

	if !v.cfg.AddSourcesRules {
		return nil, nil
	}

	pkgSrcsExpr := &build.LiteralExpr{Token: `glob(["**"])`}
	if pkgPath == "." {
		pkgSrcsExpr = &build.LiteralExpr{Token: `glob(["**"], exclude=["bazel-*/**", ".git/**"])`}
	}

	v.addRules(pkgPath, []*build.Rule{
		newRule("filegroup",
			pkgSrcsTarget,
			map[string]build.Expr{
				"srcs":       pkgSrcsExpr,
				"visibility": asExpr([]string{"//visibility:private"}),
			}),
		newRule("filegroup",
			allSrcsTarget,
			map[string]build.Expr{
				"srcs": asExpr(append(children, fmt.Sprintf(":%s", pkgSrcsTarget))),
				// TODO: should this be more restricted?
				"visibility": asExpr([]string{"//visibility:public"}),
			}),
	})
	return []string{fmt.Sprintf("//%s:%s", pkgPath, allSrcsTarget)}, nil
}
