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

package gnostic_extension_v1

import (
	"os"
	"os/exec"
	"testing"
)

func TestExtensionHandlerWithLibraryExample(t *testing.T) {
	outputFile := "library-example-with-ext.text.out"
	inputFile := "../testdata/library-example-with-ext.json"
	referenceFile := "../testdata/library-example-with-ext.text.out"

	os.Remove(outputFile)
	// run the compiler
	var err error

	command := exec.Command(
		"gnostic",
		"--x-sampleone",
		"--x-sampletwo",
		"--text-out="+outputFile,
		"--resolve-refs",
		inputFile)

	_, err = command.Output()
	if err != nil {
		t.Logf("Compile failed for command %v: %+v", command, err)
		t.FailNow()
	}
	//_ = ioutil.WriteFile(outputFile, output, 0644)
	err = exec.Command("diff", outputFile, referenceFile).Run()
	if err != nil {
		t.Logf("Diff failed: %+v", err)
		t.FailNow()
	} else {
		// if the test succeeded, clean up
		os.Remove(outputFile)
	}
}
