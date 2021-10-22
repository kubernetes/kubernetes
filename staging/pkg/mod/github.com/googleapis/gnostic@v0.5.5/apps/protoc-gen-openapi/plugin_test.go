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
	"os"
	"os/exec"
	"testing"
)

func TestLibraryOpenAPI(t *testing.T) {
	var err error
	// Run protoc and the protoc-gen-openapi plugin to generate an OpenAPI spec.
	err = exec.Command("protoc",
		"-I", "../../",
		"-I", "../../third_party",
		"-I", "examples",
		"examples/google/example/library/v1/library.proto",
		"--openapi_out=.").Run()
	if err != nil {
		t.Logf("protoc failed: %+v", err)
		t.FailNow()
	}
	// Verify that the generated spec matches our expected version.
	err = exec.Command("diff", "openapi.yaml", "examples/google/example/library/v1/openapi.yaml").Run()
	if err != nil {
		t.Logf("Diff failed: %+v", err)
		t.FailNow()
	}
	// if the test succeeded, clean up
	os.Remove("openapi.yaml")
}

func TestBodyMappingOpenAPI(t *testing.T) {
	var err error
	// Run protoc and the protoc-gen-openapi plugin to generate an OpenAPI spec.
	err = exec.Command("protoc",
		"-I", "../../",
		"-I", "../../third_party",
		"-I", "examples",
		"examples/tests/bodymapping/message.proto",
		"--openapi_out=.").Run()
	if err != nil {
		t.Logf("protoc failed: %+v", err)
		t.FailNow()
	}
	// Verify that the generated spec matches our expected version.
	err = exec.Command("diff", "openapi.yaml", "examples/tests/bodymapping/openapi.yaml").Run()
	if err != nil {
		t.Logf("Diff failed: %+v", err)
		t.FailNow()
	}
	// if the test succeeded, clean up
	os.Remove("openapi.yaml")
}
