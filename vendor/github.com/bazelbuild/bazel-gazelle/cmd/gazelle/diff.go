/* Copyright 2016 The Bazel Authors. All rights reserved.

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
	"io"
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/bazelbuild/bazel-gazelle/internal/config"
	"github.com/bazelbuild/bazel-gazelle/internal/rule"
	"github.com/pmezard/go-difflib/difflib"
)

func diffFile(c *config.Config, f *rule.File) error {
	rel, err := filepath.Rel(c.RepoRoot, f.Path)
	if err != nil {
		return fmt.Errorf("error getting old path for file %q: %v", f.Path, err)
	}
	rel = filepath.ToSlash(rel)

	date := "1970-01-01 00:00:00.000000000 +0000"
	diff := difflib.UnifiedDiff{
		Context:  3,
		FromDate: date,
		ToDate:   date,
	}

	if oldContent, err := ioutil.ReadFile(f.Path); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("error reading original file: %v", err)
	} else if err != nil {
		diff.FromFile = "/dev/null"
	} else if err == nil {
		diff.A = difflib.SplitLines(string(oldContent))
		if c.ReadBuildFilesDir == "" {
			path, err := filepath.Rel(c.RepoRoot, f.Path)
			if err != nil {
				return fmt.Errorf("error getting old path for file %q: %v", f.Path, err)
			}
			diff.FromFile = filepath.ToSlash(path)
		} else {
			diff.FromFile = f.Path
		}
	}

	newContent := f.Format()
	diff.B = difflib.SplitLines(string(newContent))
	outPath := findOutputPath(c, f)
	if c.WriteBuildFilesDir == "" {
		path, err := filepath.Rel(c.RepoRoot, f.Path)
		if err != nil {
			return fmt.Errorf("error getting new path for file %q: %v", f.Path, err)
		}
		diff.ToFile = filepath.ToSlash(path)
	} else {
		diff.ToFile = outPath
	}

	uc := getUpdateConfig(c)
	var out io.Writer = os.Stdout
	if uc.patchPath != "" {
		out = &uc.patchBuffer
	}
	if err := difflib.WriteUnifiedDiff(out, diff); err != nil {
		return fmt.Errorf("error diffing %s: %v", f.Path, err)
	}
	return nil
}
