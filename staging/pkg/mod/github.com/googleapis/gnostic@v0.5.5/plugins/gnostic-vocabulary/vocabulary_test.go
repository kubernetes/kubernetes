// Copyright 2020 Google LLC. All Rights Reserved.
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
	"io/ioutil"
	"os"
	"os/exec"
	"testing"
)

func testPlugin(t *testing.T, plugin string, inputFile string, outputFile string, referenceFile string) {
	// remove any preexisting output files
	os.Remove(outputFile)
	// run the compiler
	var err error
	output, err := exec.Command(
		"gnostic",
		"--"+plugin+"-out=-",
		inputFile).Output()
	if err != nil {
		t.Logf("Compile failed: %+v", err)
		t.FailNow()
	}
	_ = ioutil.WriteFile(outputFile, output, 0644)
	err = exec.Command("diff", outputFile, referenceFile).Run()
	if err != nil {
		t.Logf("Diff failed: %s vs %s %+v", outputFile, referenceFile, err)
		t.FailNow()
	} else {
		// if the test succeeded, clean up
		os.Remove(outputFile)
	}
}

func TestSamplePluginWithPetstoreV2(t *testing.T) {
	testPlugin(t,
		"vocabulary",
		"../../examples/v2.0/yaml/petstore.yaml",
		"vocabulary-petstore-v2.out",
		"../../testdata/v2.0/yaml/vocabulary-petstore.out")
}

func TestSamplePluginWithPetstoreV3(t *testing.T) {
	testPlugin(t,
		"vocabulary",
		"../../examples/v3.0/yaml/petstore.yaml",
		"vocabulary-petstore-v3.out",
		"../../testdata/v3.0/yaml/vocabulary-petstore.out")
}
