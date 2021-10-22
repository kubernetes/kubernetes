// Copyright 2017 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/googleapis/gnostic/lib"
)

func isURL(path string) bool {
	_, err := url.ParseRequestURI(path)
	if err != nil {
		return false
	}
	u, err := url.Parse(path)
	if err != nil || u.Scheme == "" || u.Host == "" {
		return false
	}
	return true
}

func testCompiler(t *testing.T, inputFile string, referenceFile string, expectErrors bool) {
	outputFormat := filepath.Ext(referenceFile)[1:]
	outputFile := strings.Replace(inputFile, filepath.Ext(inputFile), "."+outputFormat, 1)
	errorsFile := strings.Replace(inputFile, filepath.Ext(inputFile), ".errors", 1)
	if isURL(inputFile) {
		// put outputs in the current directory
		outputFile = filepath.Base(outputFile)
		errorsFile = filepath.Base(errorsFile)
	}
	// remove any preexisting output files
	os.Remove(outputFile)
	os.Remove(errorsFile)
	var err error
	// run the compiler
	args := []string{
		"gnostic",
		inputFile,
		"--" + outputFormat + "-out=.",
		"--errors-out=.",
		"--resolve-refs"}
	g := lib.NewGnostic(args)
	err = g.Main()
	// verify the output against a reference
	var testFile string
	if expectErrors {
		testFile = errorsFile
	} else {
		testFile = outputFile
	}
	err = exec.Command("diff", testFile, referenceFile).Run()
	if err != nil {
		t.Logf("Diff failed: %+v", err)
		t.FailNow()
	} else {
		// if the test succeeded, clean up
		os.Remove(outputFile)
		os.Remove(errorsFile)
	}
}

func testNormal(t *testing.T, inputFile string, referenceFile string) {
	testCompiler(t, inputFile, referenceFile, false)
}

func testErrors(t *testing.T, inputFile string, referenceFile string) {
	testCompiler(t, inputFile, referenceFile, true)
}

func TestPetstoreJSON(t *testing.T) {
	testNormal(t,
		"examples/v2.0/json/petstore.json",
		"testdata/v2.0/petstore.text")
}

func TestPetstoreYAML(t *testing.T) {
	testNormal(t,
		"examples/v2.0/yaml/petstore.yaml",
		"testdata/v2.0/petstore.text")
}

func TestSeparateYAML(t *testing.T) {
	testNormal(t,
		"examples/v2.0/yaml/petstore-separate/spec/swagger.yaml",
		"testdata/v2.0/yaml/petstore-separate/spec/swagger.text")
}

func TestSeparateJSON(t *testing.T) {
	testNormal(t,
		"examples/v2.0/json/petstore-separate/spec/swagger.json",
		"testdata/v2.0/yaml/petstore-separate/spec/swagger.text") // yaml and json results should be identical
}

func TestRemotePetstoreJSON(t *testing.T) {
	testNormal(t,
		"https://raw.githubusercontent.com/googleapis/openapi-compiler/master/examples/v2.0/json/petstore.json",
		"testdata/v2.0/petstore.text")
}

func TestRemotePetstoreYAML(t *testing.T) {
	testNormal(t,
		"https://raw.githubusercontent.com/googleapis/openapi-compiler/master/examples/v2.0/yaml/petstore.yaml",
		"testdata/v2.0/petstore.text")
}

func TestRemoteSeparateYAML(t *testing.T) {
	testNormal(t,
		"https://raw.githubusercontent.com/googleapis/openapi-compiler/master/examples/v2.0/yaml/petstore-separate/spec/swagger.yaml",
		"testdata/v2.0/yaml/petstore-separate/spec/swagger.text")
}

func TestRemoteSeparateJSON(t *testing.T) {
	testNormal(t,
		"https://raw.githubusercontent.com/googleapis/openapi-compiler/master/examples/v2.0/json/petstore-separate/spec/swagger.json",
		"testdata/v2.0/yaml/petstore-separate/spec/swagger.text")
}

func TestErrorBadProperties(t *testing.T) {
	testErrors(t,
		"examples/errors/petstore-badproperties.yaml",
		"testdata/errors/petstore-badproperties.errors")
}

func TestErrorUnresolvedRefs(t *testing.T) {
	testErrors(t,
		"examples/errors/petstore-unresolvedrefs.yaml",
		"testdata/errors/petstore-unresolvedrefs.errors")
}

func TestErrorMissingVersion(t *testing.T) {
	testErrors(t,
		"examples/errors/petstore-missingversion.yaml",
		"testdata/errors/petstore-missingversion.errors")
}

func TestJSONOutput(t *testing.T) {
	inputFile := "testdata/library-example-with-ext.json"

	textFile := "sample.text"
	jsonFile := "sample.json"
	textFile2 := "sample2.text"
	jsonFile2 := "sample2.json"

	os.Remove(textFile)
	os.Remove(jsonFile)
	os.Remove(textFile2)
	os.Remove(jsonFile2)

	var err error

	// Run the compiler once.
	args := []string{
		"gnostic",
		"--text-out=" + textFile,
		"--json-out=" + jsonFile,
		inputFile}
	g := lib.NewGnostic(args)
	err = g.Main()
	if err != nil {
		t.Logf("Compile failed for command %v: %+v", strings.Join(args, " "), err)
		t.FailNow()
	}

	// Run the compiler again, this time on the generated output.
	args = []string{
		"gnostic",
		"--text-out=" + textFile2,
		"--json-out=" + jsonFile2,
		jsonFile}
	g = lib.NewGnostic(args)
	err = g.Main()
	if err != nil {
		t.Logf("Compile failed for command %v: %+v", strings.Join(args, " "), err)
		t.FailNow()
	}

	// Verify that both models have the same internal representation.
	err = exec.Command("diff", textFile, textFile2).Run()
	if err != nil {
		t.Logf("Diff failed (%s vs %s): %+v", textFile, textFile2, err)
		t.FailNow()
	} else {
		// if the test succeeded, clean up
		os.Remove(textFile)
		os.Remove(jsonFile)
		os.Remove(textFile2)
		os.Remove(jsonFile2)
	}
}

func TestYAMLOutput(t *testing.T) {
	inputFile := "testdata/library-example-with-ext.json"

	textFile := "sample.text"
	yamlFile := "sample.yaml"
	textFile2 := "sample2.text"
	yamlFile2 := "sample2.yaml"

	os.Remove(textFile)
	os.Remove(yamlFile)
	os.Remove(textFile2)
	os.Remove(yamlFile2)

	var err error

	// Run the compiler once.
	args := []string{
		"gnostic",
		"--text-out=" + textFile,
		"--yaml-out=" + yamlFile,
		inputFile}
	g := lib.NewGnostic(args)
	err = g.Main()
	if err != nil {
		t.Logf("Compile failed for command %v: %+v", strings.Join(args, " "), err)
		t.FailNow()
	}

	// Run the compiler again, this time on the generated output.
	args = []string{
		"gnostic",
		"--text-out=" + textFile2,
		"--yaml-out=" + yamlFile2,
		yamlFile}
	g = lib.NewGnostic(args)
	err = g.Main()
	if err != nil {
		t.Logf("Compile failed for command %v: %+v", strings.Join(args, " "), err)
		t.FailNow()
	}

	// Verify that both models have the same internal representation.
	err = exec.Command("diff", textFile, textFile2).Run()
	if err != nil {
		t.Logf("Diff failed: %+v", err)
		t.FailNow()
	} else {
		// if the test succeeded, clean up
		os.Remove(textFile)
		os.Remove(yamlFile)
		os.Remove(textFile2)
		os.Remove(yamlFile2)
	}
}

// OpenAPI 3.0 tests

func TestPetstoreYAML_30(t *testing.T) {
	testNormal(t,
		"examples/v3.0/yaml/petstore.yaml",
		"testdata/v3.0/petstore.text")
}

func TestPetstoreJSON_30(t *testing.T) {
	testNormal(t,
		"examples/v3.0/json/petstore.json",
		"testdata/v3.0/petstore.text")
}

// Test that empty required fields are exported.

func TestEmptyRequiredFields_v2(t *testing.T) {
	testNormal(t,
		"examples/v2.0/yaml/empty-v2.yaml",
		"testdata/v2.0/json/empty-v2.json")
}

func TestEmptyRequiredFields_v3(t *testing.T) {
	testNormal(t,
		"examples/v3.0/yaml/empty-v3.yaml",
		"testdata/v3.0/json/empty-v3.json")
}

func TestDiscoveryJSON(t *testing.T) {
	testNormal(t,
		"examples/discovery/discovery-v1.json",
		"testdata/discovery/discovery-v1.text")
}
