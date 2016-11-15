/*
Copyright 2014 The Kubernetes Authors.

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

package priorities

import (
	"fmt"
	"os/exec"
	"path/filepath"
	"testing"

	"k8s.io/gengo/parser"
	"k8s.io/gengo/types"
	"k8s.io/kubernetes/pkg/util/codeinspector"
)

func getPrioritySignatures() ([]*types.Signature, error) {
	filePath := "./../types.go"
	pkgName := filepath.Dir(filePath)
	builder := parser.New()
	if err := builder.AddDir(pkgName); err != nil {
		return nil, err
	}
	universe, err := builder.FindTypes()
	if err != nil {
		return nil, err
	}
	signatures := []string{"PriorityFunction", "PriorityMapFunction", "PriorityReduceFunction"}
	results := make([]*types.Signature, 0, len(signatures))
	for _, signature := range signatures {
		result, ok := universe[pkgName].Types[signature]
		if !ok {
			return nil, fmt.Errorf("%s type not defined", signature)
		}
		results = append(results, result.Signature)
	}
	return results, nil
}

func TestPrioritiesRegistered(t *testing.T) {
	var functions []*types.Type

	// Files and directories which priorities may be referenced
	targetFiles := []string{
		"./../../algorithmprovider/defaults/defaults.go", // Default algorithm
		"./../../factory/plugins.go",                     // Registered in init()
	}

	// List all golang source files under ./priorities/, excluding test files and sub-directories.
	files, err := codeinspector.GetSourceCodeFiles(".")

	if err != nil {
		t.Errorf("unexpected error: %v when listing files in current directory", err)
	}

	// Get all public priorities in files.
	for _, filePath := range files {
		fileFunctions, err := codeinspector.GetPublicFunctions("k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/priorities", filePath)
		if err == nil {
			functions = append(functions, fileFunctions...)
		} else {
			t.Errorf("unexpected error when parsing %s: %v", filePath, err)
		}
	}

	prioritySignatures, err := getPrioritySignatures()
	if err != nil {
		t.Fatalf("Couldn't get priorities signatures")
	}

	// Check if all public priorities are referenced in target files.
	for _, function := range functions {
		// Ignore functions that don't match priorities signatures.
		signature := function.Underlying.Signature
		match := false
		for _, prioritySignature := range prioritySignatures {
			if len(prioritySignature.Parameters) != len(signature.Parameters) {
				continue
			}
			if len(prioritySignature.Results) != len(signature.Results) {
				continue
			}
			// TODO: Check exact types of parameters and results.
			match = true
		}
		if !match {
			continue
		}

		args := []string{"-rl", function.Name.Name}
		args = append(args, targetFiles...)

		err := exec.Command("grep", args...).Run()
		if err != nil {
			switch err.Error() {
			case "exit status 2":
				t.Errorf("unexpected error when checking %s", function.Name)
			case "exit status 1":
				t.Errorf("priority %s is implemented as public but seems not registered or used in any other place",
					function.Name)
			}
		}
	}
}
