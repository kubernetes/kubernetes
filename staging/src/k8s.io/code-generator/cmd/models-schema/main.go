/*
Copyright The Kubernetes Authors.

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

// models-schema accepts a --package flag naming any Go package that exports
// GetOpenAPIDefinitions and writes the Swagger 2.0 schema for those types
package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"text/template"
)

var wrapperTmpl = template.Must(template.New("wrapper").Parse(`package main

import (
	"fmt"
	"os"

	openapi "{{.}}"
	"k8s.io/code-generator/pkg/util"
)

func main() {
	if err := util.OutputSchemas(openapi.GetOpenAPIDefinitions); err != nil {
		fmt.Fprintf(os.Stderr, "Failed: %v\n", err)
		os.Exit(1)
	}
}
`))

func run(pkg string) error {
	cwd, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("cannot determine working directory: %w", err)
	}

	tmp, err := os.CreateTemp(cwd, "models_schema_tmp_*.go")
	if err != nil {
		return fmt.Errorf("cannot create temp file: %w", err)
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath) // nolint:errcheck

	if err := wrapperTmpl.Execute(tmp, pkg); err != nil {
		return fmt.Errorf("cannot write wrapper: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("cannot close temp file: %w", err)
	}

	cmd := exec.Command("go", "run", tmpPath)
	// ensure we don't import anything outside current vendor/ directory
	cmd.Env = append(os.Environ(), "GOFLAGS=-mod=vendor")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func main() {
	pkg := flag.String("package", "", "Go import path of the package providing GetOpenAPIDefinitions")
	flag.Parse()
	if *pkg == "" {
		fmt.Fprintln(os.Stderr, "error: --package flag is required")
		os.Exit(1)
	}

	if err := run(*pkg); err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok { // nolint:errorlint
			os.Exit(exitErr.ExitCode())
		}
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}
