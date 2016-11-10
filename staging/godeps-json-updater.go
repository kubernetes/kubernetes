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
	"flag"
	"log"
	"os"
	"strings"
)

var (
	godepsFile           = flag.String("godeps-file", "", "absolute path to Godeps.json")
	clientRepoImportPath = flag.String("client-go-import-path", "", "import path to a version of client-go, e.g., k8s.io/client-go/1.4")
)

type Dependency struct {
	ImportPath string
	Comment    string `json:",omitempty"`
	Rev        string
}

type Godeps struct {
	ImportPath   string
	GoVersion    string
	GodepVersion string
	Packages     []string `json:",omitempty"` // Arguments to save, if any.
	Deps         []Dependency
}

// rewrites the Godeps.ImportPath, removes the Deps whose ImportPath contains "k8s.io/kubernetes"
func main() {
	flag.Parse()
	var g Godeps
	if len(*godepsFile) == 0 {
		log.Fatalf("absolute ath to Godeps.json is required")
	}
	if len(*clientRepoImportPath) == 0 {
		log.Fatalf("import path to a version of client-go is required")
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
	// rewrites the Godeps.ImportPath
	g.ImportPath = *clientRepoImportPath
	// removes the Deps whose ImportPath contains "k8s.io/kubernetes"
	i := 0
	for _, dep := range g.Deps {
		if strings.Contains(dep.ImportPath, "k8s.io/kubernetes") {
			continue
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
