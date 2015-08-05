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
	"path"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	_ "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"

	"github.com/golang/glog"
)

func generateDeepCopies(t *testing.T, version string) bytes.Buffer {
	testedVersion := version
	registerTo := "api.Scheme"
	if testedVersion == "api" {
		testedVersion = api.Scheme.Raw().InternalVersion
		registerTo = "Scheme"
	}

	g := runtime.NewDeepCopyGenerator(api.Scheme.Raw(), path.Join("k8s.io/kubernetes/pkg/api", testedVersion), util.NewStringSet("k8s.io/kubernetes"))
	g.AddImport("k8s.io/kubernetes/pkg/api")
	g.OverwritePackage(version, "")

	for _, knownType := range api.Scheme.KnownTypes(testedVersion) {
		if err := g.AddType(knownType); err != nil {
			glog.Errorf("error while generating deep-copy functions for %v: %v", knownType, err)
		}
	}

	var functions bytes.Buffer
	functionsWriter := bufio.NewWriter(&functions)
	g.RepackImports()
	if err := g.WriteImports(functionsWriter); err != nil {
		t.Fatalf("couldn't generate deep-copy function imports: %v", err)
	}
	if err := g.WriteDeepCopyFunctions(functionsWriter); err != nil {
		t.Fatalf("couldn't generate deep-copy functions: %v", err)
	}
	if err := g.RegisterDeepCopyFunctions(functionsWriter, registerTo); err != nil {
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
