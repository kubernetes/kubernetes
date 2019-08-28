/*
Copyright 2019 The Kubernetes Authors.

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
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"gopkg.in/yaml.v2"
)

const (
	kubeMetricImportPath = `"k8s.io/component-base/metrics"`
	// Should equal to final directory name of kubeMetricImportPath
	kubeMetricsDefaultImportName = "metrics"
	prometheusImportPath         = `"github.com/prometheus/client_golang/prometheus"`
	// Should equal to final directory name of kubeMetricImportPath
	prometheusDefaultImportName = "prometheus"
)

func main() {
	flag.Parse()
	if len(flag.Args()) < 1 {
		fmt.Fprintf(os.Stderr, "USAGE: %s <DIR or FILE> [...]\n", os.Args[0])
		os.Exit(64)
	}

	stableMetrics := []metric{}
	errors := []error{}

	for _, arg := range flag.Args() {
		ms, es := searchPathForStableMetrics(arg)
		stableMetrics = append(stableMetrics, ms...)
		errors = append(errors, es...)
	}
	for _, err := range errors {
		fmt.Fprintf(os.Stderr, "%s\n", err)
	}
	if len(errors) != 0 {
		os.Exit(1)
	}
	sort.Sort(byFQName(stableMetrics))
	data, err := yaml.Marshal(stableMetrics)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s\n", err)
		os.Exit(1)
	}
	fmt.Print(string(data))
}

func searchPathForStableMetrics(path string) ([]metric, []error) {
	metrics := []metric{}
	errors := []error{}
	err := filepath.Walk(path, func(path string, info os.FileInfo, err error) error {
		if strings.HasPrefix(path, "vendor") {
			return filepath.SkipDir
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}
		ms, es := searchFileForStableMetrics(path, nil)
		errors = append(errors, es...)
		metrics = append(metrics, ms...)
		return nil
	})
	if err != nil {
		errors = append(errors, err)
	}
	return metrics, errors
}

// Pass either only filename of existing file or src including source code in any format and a filename that it comes from
func searchFileForStableMetrics(filename string, src interface{}) ([]metric, []error) {
	fileset := token.NewFileSet()
	tree, err := parser.ParseFile(fileset, filename, src, parser.AllErrors)
	if err != nil {
		return []metric{}, []error{err}
	}
	metricsImportName, err := getLocalNameOfImportedPackage(tree, kubeMetricImportPath, kubeMetricsDefaultImportName)
	if err != nil {
		return []metric{}, addFileInformationToErrors([]error{err}, fileset)
	}
	if metricsImportName == "" {
		return []metric{}, []error{}
	}
	prometheusImportName, err := getLocalNameOfImportedPackage(tree, prometheusImportPath, prometheusDefaultImportName)
	if err != nil {
		return []metric{}, addFileInformationToErrors([]error{err}, fileset)
	}

	stableMetricsFunctionCalls, errors := findStableMetricDeclaration(tree, metricsImportName)
	metrics, es := decodeMetricCalls(stableMetricsFunctionCalls, metricsImportName, prometheusImportName)
	errors = append(errors, es...)
	return metrics, addFileInformationToErrors(errors, fileset)
}

func getLocalNameOfImportedPackage(tree *ast.File, importPath, defaultImportName string) (string, error) {
	var importName string
	for _, im := range tree.Imports {
		if im.Path.Value == importPath {
			if im.Name == nil {
				importName = defaultImportName
			} else {
				if im.Name.Name == "." {
					return "", newDecodeErrorf(im, errImport)
				}
				importName = im.Name.Name
			}
		}
	}
	return importName, nil
}

func addFileInformationToErrors(es []error, fileset *token.FileSet) []error {
	for i := range es {
		if de, ok := es[i].(*decodeError); ok {
			es[i] = de.errorWithFileInformation(fileset)
		}
	}
	return es
}
