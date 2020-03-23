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

package golang

import (
	"bytes"
	"encoding/json"
	"go/build"
	"io"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"

	"github.com/bazelbuild/bazel-gazelle/label"
	"github.com/bazelbuild/bazel-gazelle/language"
	"github.com/bazelbuild/bazel-gazelle/rule"
)

func importReposFromModules(args language.ImportReposArgs) language.ImportReposResult {
	// Copy go.mod to temporary directory. We may run commands that modify it,
	// and we want to leave the original alone.
	tempDir, err := copyGoModToTemp(args.Path)
	if err != nil {
		return language.ImportReposResult{Error: err}
	}
	defer os.RemoveAll(tempDir)

	// List all modules except for the main module, including implicit indirect
	// dependencies.
	type module struct {
		Path, Version, Sum string
		Main               bool
		Replace            *struct {
			Path, Version string
		}
	}
	// path@version can be used as a unique identifier for looking up sums
	pathToModule := map[string]*module{}
	data, err := goListModules(tempDir)
	if err != nil {
		return language.ImportReposResult{Error: err}
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	for dec.More() {
		mod := new(module)
		if err := dec.Decode(mod); err != nil {
			return language.ImportReposResult{Error: err}
		}
		if mod.Main {
			continue
		}
		if mod.Replace != nil {
			if filepath.IsAbs(mod.Replace.Path) || build.IsLocalImport(mod.Replace.Path) {
				log.Printf("go_repository does not support file path replacements for %s -> %s", mod.Path,
					mod.Replace.Path)
				continue
			}
			pathToModule[mod.Replace.Path+"@"+mod.Replace.Version] = mod
		} else {
			pathToModule[mod.Path+"@"+mod.Version] = mod
		}
	}
	// Load sums from go.sum. Ideally, they're all there.
	goSumPath := filepath.Join(filepath.Dir(args.Path), "go.sum")
	data, _ = ioutil.ReadFile(goSumPath)
	lines := bytes.Split(data, []byte("\n"))
	for _, line := range lines {
		line = bytes.TrimSpace(line)
		fields := bytes.Fields(line)
		if len(fields) != 3 {
			continue
		}
		path, version, sum := string(fields[0]), string(fields[1]), string(fields[2])
		if strings.HasSuffix(version, "/go.mod") {
			continue
		}
		if mod, ok := pathToModule[path+"@"+version]; ok {
			mod.Sum = sum
		}
	}
	// If sums are missing, run go mod download to get them.
	var missingSumArgs []string
	for pathVer, mod := range pathToModule {
		if mod.Sum == "" {
			missingSumArgs = append(missingSumArgs, pathVer)
		}
	}
	if len(missingSumArgs) > 0 {
		data, err := goModDownload(tempDir, missingSumArgs)
		if err != nil {
			return language.ImportReposResult{Error: err}
		}
		dec = json.NewDecoder(bytes.NewReader(data))
		for dec.More() {
			var dl module
			if err := dec.Decode(&dl); err != nil {
				return language.ImportReposResult{Error: err}
			}
			if mod, ok := pathToModule[dl.Path+"@"+dl.Version]; ok {
				mod.Sum = dl.Sum
			}
		}
	}

	// Translate to repository rules.
	gen := make([]*rule.Rule, 0, len(pathToModule))
	for pathVer, mod := range pathToModule {
		if mod.Sum == "" {
			log.Printf("could not determine sum for module %s", pathVer)
			continue
		}
		r := rule.NewRule("go_repository", label.ImportPathToBazelRepoName(mod.Path))
		r.SetAttr("importpath", mod.Path)
		r.SetAttr("sum", mod.Sum)
		if mod.Replace == nil {
			r.SetAttr("version", mod.Version)
		} else {
			r.SetAttr("replace", mod.Replace.Path)
			r.SetAttr("version", mod.Replace.Version)
		}
		gen = append(gen, r)
	}
	sort.Slice(gen, func(i, j int) bool {
		return gen[i].Name() < gen[j].Name()
	})
	return language.ImportReposResult{Gen: gen}
}

// goListModules invokes "go list" in a directory containing a go.mod file.
var goListModules = func(dir string) ([]byte, error) {
	goTool := findGoTool()
	cmd := exec.Command(goTool, "list", "-m", "-json", "all")
	cmd.Stderr = os.Stderr
	cmd.Dir = dir
	return cmd.Output()
}

// goModDownload invokes "go mod download" in a directory containing a
// go.mod file.
var goModDownload = func(dir string, args []string) ([]byte, error) {
	goTool := findGoTool()
	cmd := exec.Command(goTool, "mod", "download", "-json")
	cmd.Args = append(cmd.Args, args...)
	cmd.Stderr = os.Stderr
	cmd.Dir = dir
	return cmd.Output()
}

// copyGoModToTemp copies to given go.mod file to a temporary directory.
// go list tends to mutate go.mod files, but gazelle shouldn't do that.
func copyGoModToTemp(filename string) (tempDir string, err error) {
	goModOrig, err := os.Open(filename)
	if err != nil {
		return "", err
	}
	defer goModOrig.Close()

	tempDir, err = ioutil.TempDir("", "gazelle-temp-gomod")
	if err != nil {
		return "", err
	}

	goModCopy, err := os.Create(filepath.Join(tempDir, "go.mod"))
	if err != nil {
		os.Remove(tempDir)
		return "", err
	}
	defer func() {
		if cerr := goModCopy.Close(); err == nil && cerr != nil {
			err = cerr
		}
	}()

	_, err = io.Copy(goModCopy, goModOrig)
	if err != nil {
		os.RemoveAll(tempDir)
		return "", err
	}
	return tempDir, err
}

// findGoTool attempts to locate the go executable. If GOROOT is set, we'll
// prefer the one in there; otherwise, we'll rely on PATH. If the wrapper
// script generated by the gazelle rule is invoked by Bazel, it will set
// GOROOT to the configured SDK. We don't want to rely on the host SDK in
// that situation.
func findGoTool() string {
	path := "go" // rely on PATH by default
	if goroot, ok := os.LookupEnv("GOROOT"); ok {
		path = filepath.Join(goroot, "bin", "go")
	}
	if runtime.GOOS == "windows" {
		path += ".exe"
	}
	return path
}
