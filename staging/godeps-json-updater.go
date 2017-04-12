/*
Copyright 2016 The Kubernetes Authors.

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
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	flag "github.com/spf13/pflag"
)

var (
	godepsFile         = flag.String("godeps-file", "", "absolute path to Godeps.json")
	overrideImportPath = flag.String("override-import-path", "", "import path to be written into the Godeps.json, e.g., k8s.io/client-go")
	ignoredPrefixes    = flag.StringSlice("ignored-prefixes", []string{"k8s.io/"}, "any godep entry prefixed with the ignored-prefix will be deleted from Godeps.json")
	rewrittenPrefixes  = flag.StringSlice("rewritten-prefixes", []string{}, fmt.Sprintf("any godep entry prefixed with the rewritten-prefix will be filled will dummy rev %q; overridden by ignored-prefixes", dummyRev))
)

const dummyRev = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

type Dependency struct {
	ImportPath string
	Rev        string
}

type Godeps struct {
	ImportPath   string
	GoVersion    string
	GodepVersion string
	Packages     []string `json:",omitempty"` // Arguments to save, if any.
	Deps         []Dependency
}

// rewrites the Godeps.ImportPath, removes the Deps whose ImportPath contains "k8s.io/kubernetes" or "k8s.io/apimachinery".
// entries for k8s.io/apimahinery will be written by the publishing robot before publishing to the repository.
func main() {
	flag.Parse()
	var g Godeps
	if len(*godepsFile) == 0 {
		log.Fatalf("absolute path to Godeps.json is required")
	}
	f, err := os.OpenFile(*godepsFile, os.O_RDWR, 0666)
	if err != nil {
		log.Fatalf("cannot open file %q: %v", *godepsFile, err)
	}
	defer f.Close()
	err = json.NewDecoder(f).Decode(&g)
	if err != nil {
		log.Fatalf("Unable to parse %q: %v", *godepsFile, err)
	}
	if len(*overrideImportPath) != 0 {
		g.ImportPath = *overrideImportPath
	}
	// removes the Deps whose ImportPath contains "k8s.io/kubernetes"
	i := 0
	for _, dep := range g.Deps {
		ignored := false
		for _, ignoredPrefix := range *ignoredPrefixes {
			if strings.HasPrefix(dep.ImportPath, ignoredPrefix) {
				ignored = true
			}
		}
		if ignored {
			continue
		}
		rewritten := false
		for _, rewrittenPrefix := range *rewrittenPrefixes {
			if strings.HasPrefix(dep.ImportPath, rewrittenPrefix) {
				rewritten = true
			}
		}
		if rewritten {
			dep.Rev = dummyRev
		}
		g.Deps[i] = dep
		i++
	}
	g.Deps = g.Deps[:i]
	b, err := json.MarshalIndent(g, "", "\t")
	if err != nil {
		log.Fatal(err)
	}
	n, err := f.WriteAt(append(b, '\n'), 0)
	if err != nil {
		log.Fatal(err)
	}
	if err := f.Truncate(int64(n)); err != nil {
		log.Fatal(err)
	}
}
