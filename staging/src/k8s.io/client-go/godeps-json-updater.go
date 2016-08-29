package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
)

var godepsFile string
var clientRepoImportPath string

func init() {
	flag.StringVar(&godepsFile, "godeps-file", "", "absolute path to Godeps.json")
	// e.g., k8s.io/client-go/1.4
	flag.StringVar(&clientRepoImportPath, "client-go-import-path", "", "import path to a version of client-go")
}

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
	if len(godepsFile) == 0 {
		panic("absolute ath to Godeps.json is required")
	}
	if len(clientRepoImportPath) == 0 {
		panic("import path to a version of client-go is required")
	}
	//f, err := os.Open(path)
	f, err := os.OpenFile(godepsFile, os.O_RDWR, 0666)
	if err != nil {
		panic(err.Error())
	}
	defer f.Close()
	err = json.NewDecoder(f).Decode(&g)
	if err != nil {
		panic(fmt.Sprintf("Unable to parse %s: %s", godepsFile, err.Error()))
	}
	// rewrites the Godeps.ImportPath
	g.ImportPath = clientRepoImportPath
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
		panic(err.Error())
	}
	n, err := f.WriteAt(append(b, '\n'), 0)
	if err != nil {
		panic(err.Error())
	}
	f.Truncate(int64(n))
}
