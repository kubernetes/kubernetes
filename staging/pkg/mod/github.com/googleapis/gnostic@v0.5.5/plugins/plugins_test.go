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

package gnostic_plugin_v1

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

func TestSamplePluginWithPetstore(t *testing.T) {
	testPlugin(t,
		"summary",
		"../examples/v2.0/yaml/petstore.yaml",
		"sample-petstore.out",
		"../testdata/v2.0/yaml/sample-petstore.out")
}

func TestErrorInvalidPluginInvocations(t *testing.T) {
	var err error
	output, err := exec.Command(
		"gnostic",
		"../examples/v2.0/yaml/petstore.yaml",
		"--errors-out=-",
		"--plugin-out=foo=bar,:abc",
		"--plugin-out=,foo=bar:abc",
		"--plugin-out=foo=:abc",
		"--plugin-out==bar:abc",
		"--plugin-out=,,:abc",
		"--plugin-out=foo=bar=baz:abc",
	).Output()
	if err == nil {
		t.Logf("Invalid invocations were accepted")
		t.FailNow()
	}
	outputFile := "invalid-plugin-invocation.errors"
	_ = ioutil.WriteFile(outputFile, output, 0644)
	referenceFile := "../testdata/errors/invalid-plugin-invocation.errors"
	err = exec.Command("diff", outputFile, referenceFile).Run()
	if err != nil {
		t.Logf("Diff failed: %s vs %s %+v", outputFile, referenceFile, err)
		t.FailNow()
	} else {
		// if the test succeeded, clean up
		os.Remove(outputFile)
	}
}

func TestValidPluginInvocations(t *testing.T) {
	var err error
	output, err := exec.Command(
		"gnostic",
		"../examples/v2.0/yaml/petstore.yaml",
		"--errors-out=-",
		// verify an invocation with no parameters
		"--summary-out=!", // "!" indicates that no output should be generated
		// verify single pair of parameters
		"--summary-out=a=b:!",
		// verify multiple parameters
		"--summary-out=a=b,c=123,xyz=alphabetagammadelta:!",
		// verify that special characters / . - _ can be included in parameter keys and values
		"--summary-out=a/b/c=x/y/z:!",
		"--summary-out=a.b.c=x.y.z:!",
		"--summary-out=a-b-c=x-y-z:!",
		"--summary-out=a_b_c=x_y_z:!",
	).Output()
	if len(output) != 0 {
		t.Logf("Valid invocations generated invalid errors\n%s", string(output))
		t.FailNow()
	}
	if err != nil {
		t.Logf("Valid invocations were not accepted")
		t.Logf("%+v", err)
		t.FailNow()
	}
}
