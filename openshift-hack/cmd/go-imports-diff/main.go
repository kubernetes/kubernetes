package main

import (
	"flag"
	"fmt"
	"go/parser"
	"go/token"
	"os"
	"sort"
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"
)

const testPackagePrefix = "k8s.io/kubernetes/test/e2e"

func main() {
	// Parse flags
	excludeList := flag.String("exclude", "", "Comma-separated list of imports to be ignored")
	flag.Parse()

	// Parse positional arguments
	args := flag.Args()
	if len(args) != 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s [flags] <baseFile> <compareFile>\n", os.Args[0])
		flag.PrintDefaults()
		os.Exit(2)
	}
	baseFile := args[0]
	compareFile := args[1]

	// Parse the base file
	baseNode, err := parser.ParseFile(token.NewFileSet(), baseFile, nil, parser.AllErrors)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to parse go file %s: %v\n", baseFile, err)
		os.Exit(1)
	}

	// Create a slice containing imports of base file
	baseImports := sets.New[string]()
	for _, imp := range baseNode.Imports {
		v := strings.Trim(imp.Path.Value, `"`)
		if !strings.Contains(v, testPackagePrefix) {
			continue
		}
		baseImports.Insert(v)
	}

	// Parse file that is compared with the base one
	compareNode, err := parser.ParseFile(token.NewFileSet(), compareFile, nil, parser.AllErrors)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to parse go file %s: %v\n", baseFile, err)
		os.Exit(1)
	}

	// Create a slice containing imports of compare file
	compareImports := sets.New[string]()
	for _, imp := range compareNode.Imports {
		v := strings.Trim(imp.Path.Value, `"`)
		if !strings.Contains(v, testPackagePrefix) {
			continue
		}
		compareImports.Insert(v)
	}

	// Compare imports of both files
	exclude := strings.Split(*excludeList, ",")
	diff := baseImports.Difference(compareImports).Delete(exclude...).UnsortedList()
	if len(diff) > 0 {
		sort.Strings(diff)
		fmt.Fprintf(os.Stderr, "Imports from %q not in %q:\n\n%s\n", baseFile, compareFile, strings.Join(diff, "\n"))
		os.Exit(1)
	}
}
