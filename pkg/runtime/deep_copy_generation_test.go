/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package runtime_test

import (
	"bufio"
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta3"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"

	"github.com/golang/glog"
)

func generateDeepCopies(t *testing.T, version string) bytes.Buffer {
	g := runtime.NewDeepCopyGenerator(api.Scheme.Raw())
	g.OverwritePackage(version, "")
	testedVersion := version
	if version == "api" {
		testedVersion = api.Scheme.Raw().InternalVersion
	}
	for _, knownType := range api.Scheme.KnownTypes(testedVersion) {
		if err := g.AddType(knownType); err != nil {
			glog.Errorf("error while generating deep-copy functions for %v: %v", knownType, err)
		}
	}

	var functions bytes.Buffer
	functionsWriter := bufio.NewWriter(&functions)
	if err := g.WriteImports(functionsWriter, version); err != nil {
		t.Fatalf("couldn't generate deep-copy function imports: %v", err)
	}
	if err := g.WriteDeepCopyFunctions(functionsWriter); err != nil {
		t.Fatalf("couldn't generate deep-copy functions: %v", err)
	}
	if err := g.RegisterDeepCopyFunctions(functionsWriter, version); err != nil {
		t.Fatalf("couldn't generate deep-copy function names: %v", err)
	}
	if err := functionsWriter.Flush(); err != nil {
		t.Fatalf("error while flushing writer")
	}

	return functions
}

func TestNoManualChangesToGenerateDeepCopies(t *testing.T) {
	versions := []string{"api", testapi.Version()}

	for _, version := range versions {
		fileName := ""
		if version == "api" {
			fileName = "../../pkg/api/deep_copy_generated.go"
		} else {
			fileName = fmt.Sprintf("../../pkg/api/%s/deep_copy_generated.go", version)
		}

		existingFunctions := bufferExistingGeneratedCode(t, fileName)
		generatedFunctions := generateDeepCopies(t, version)

		functionsTxt := fmt.Sprintf("%s.deep_copy.txt", version)
		ioutil.WriteFile(functionsTxt, generatedFunctions.Bytes(), os.FileMode(0644))

		if ok := compareBuffers(t, functionsTxt, existingFunctions, generatedFunctions); ok {
			os.Remove(functionsTxt)
		}
	}
}
